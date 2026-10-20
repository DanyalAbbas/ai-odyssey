"""
Kaggle training script for TechFest face-recognition challenge.

Design goals
------------
1. No identity leakage: train identities and evaluation identities are disjoint.
   Training data = LFW identities. Evaluation data = Pakistani Public Figures
   (evaluated locally, after ONNX export).  An in-notebook sanity check uses
   *held-out LFW identities* which are never optimised against.
2. Honest reporting: the pretrained backbone is measured with exactly the same
   protocol as the fine-tuned one, so the delta is meaningful.
3. Robust on Kaggle: no pip installs (torch/torchvision/onnxruntime are already
   in the image), explicit errors, portable paths, ONNX parity check.

Usage
-----
Kaggle (GPU T4):
    python kaggle_train_v2.py --epochs 8

Preflight only (CPU, no training):
    python kaggle_train_v2.py --preflight

Local smoke test (CPU, tiny):
    python kaggle_train_v2.py --smoke --data-root data/pak_fig/images
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import tarfile
import time
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")   # keep Kaggle logs clean

import numpy as np

# ---------------------------------------------------------------- environment
def _log(msg: str) -> None:
    print(msg, flush=True)


def describe_environment() -> dict:
    info: dict = {"python": sys.version.split()[0]}
    try:
        import torch
        import torchvision
        info["torch"] = torch.__version__
        info["torchvision"] = torchvision.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception as exc:  # pragma: no cover
        info["torch_error"] = repr(exc)
    try:
        import onnxruntime as ort
        info["onnxruntime"] = ort.__version__
        info["ort_providers"] = ort.get_available_providers()
    except Exception as exc:  # pragma: no cover
        info["onnxruntime_error"] = repr(exc)
    import cv2
    info["opencv"] = cv2.__version__
    info["cwd"] = os.getcwd()
    return info


# ---------------------------------------------------------------------- data
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_lfw_tgz() -> str | None:
    patterns = [
        "/kaggle/input/**/lfw-funneled.tgz",
        "/kaggle/input/**/lfw*.tgz",
    ]
    for pat in patterns:
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            return hits[0]
    return None


def extract_lfw(tgz_path: str, out_dir: str) -> str:
    """Extract LFW once and return the directory that holds the identity folders."""
    marker = os.path.join(out_dir, ".lfw_extracted")
    if not os.path.exists(marker):
        _log(f"Extracting {tgz_path} -> {out_dir} (one-off, ~40s)")
        os.makedirs(out_dir, exist_ok=True)
        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(out_dir)
        Path(marker).write_text("ok")
    # Locate the directory with the most sub-directories (identity folders)
    best, best_n = out_dir, -1
    for root, dirs, _files in os.walk(out_dir):
        if root.count(os.sep) - out_dir.count(os.sep) > 2:
            continue
        if len(dirs) > best_n:
            best, best_n = root, len(dirs)
    _log(f"LFW identity root: {best} ({best_n} sub-directories)")
    return best


def find_image_root() -> tuple[str, str]:
    """Return (root, source_name). Prefers LFW, falls back to an attached
    directory-structured face dataset (>=10 identity folders)."""
    tgz = find_lfw_tgz()
    if tgz:
        return extract_lfw(tgz, "/kaggle/working/lfw"), "LFW"
    for base in sorted(glob.glob("/kaggle/input/*")):
        if not os.path.isdir(base):
            continue
        for root, dirs, _files in os.walk(base):
            if root.count(os.sep) - base.count(os.sep) > 2:
                continue
            subdirs = [d for d in dirs if os.path.isdir(os.path.join(root, d))]
            if len(subdirs) >= 10:
                return root, os.path.basename(base)
    raise RuntimeError(
        "No dataset found. Attach 'atulanandjha/lfwpeople' (or any folder-per-identity "
        "dataset) to the notebook."
    )


def collect_identities(root: str, min_images: int = 2, max_images: int = 60):
    """Map identity -> sorted list of image paths, keeping identities with at
    least `min_images` usable images."""
    identities: dict[str, list[str]] = {}
    for person in sorted(os.listdir(root)):
        pdir = os.path.join(root, person)
        if not os.path.isdir(pdir):
            continue
        imgs = [os.path.join(pdir, f) for f in sorted(os.listdir(pdir))
                if os.path.splitext(f)[1].lower() in IMG_EXTS]
        if len(imgs) < min_images:
            continue
        # Cap very frequent identities so one person cannot dominate the loss.
        if len(imgs) > max_images:
            step = len(imgs) / max_images
            imgs = [imgs[int(i * step)] for i in range(max_images)]
        identities[person] = imgs
    return identities


class FaceDataset:
    """Minimal Dataset: root/<identity>/*.jpg -> (CHW float tensor, label)."""

    def __init__(self, identities: dict[str, list[str]], classes: list[str],
                 augment: bool, img_size: int = 112, seed: int = 0):
        self.img_size = img_size
        self.augment = augment
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.samples = [(p, self.class_to_idx[c]) for c in classes if c in identities
                        for p in identities[c]]
        if not self.samples:
            raise RuntimeError("Empty dataset: no samples for the requested classes")
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        import cv2
        import torch

        path, label = self.samples[idx]
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), np.uint8)
        else:
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.augment:
            img = self._augment(img, cv2)
        x = img.astype(np.float32) / 255.0
        x = (x - 0.5) / 0.5
        return torch.from_numpy(x).permute(2, 0, 1).contiguous(), label

    def _augment(self, img, cv2):
        rng = self.rng
        if rng.random() < 0.5:
            img = cv2.flip(img, 1)
        if rng.random() < 0.5:
            ang = rng.uniform(-10, 10)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        if rng.random() < 0.4:
            s = rng.uniform(0.3, 0.8)
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        if rng.random() < 0.3:
            k = rng.choice([3, 5])
            img = cv2.GaussianBlur(img, (k, k), 0)
        if rng.random() < 0.4:
            q = rng.randint(35, 90)
            ok, enc = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                                   [int(cv2.IMWRITE_JPEG_QUALITY), q])
            if ok:  # noqa: SIM108
                img = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        if rng.random() < 0.3:
            alpha = rng.uniform(0.85, 1.15)
            beta = rng.uniform(-10, 10)
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        return img


# --------------------------------------------------------------------- model
def build_backbone(emb_dim: int = 512, pretrained: bool = True):
    import torch.nn as nn
    from torchvision.models import resnet50

    if pretrained:
        try:
            from torchvision.models import ResNet50_Weights
            net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        except Exception as exc:  # older torchvision / offline
            _log(f"WARNING: weights=DEFAULT failed ({exc!r}); retrying legacy API")
            try:
                net = resnet50(pretrained=True)  # type: ignore[arg-type]
            except Exception as exc2:
                raise RuntimeError(
                    "Could not load ImageNet weights. Check internet access to "
                    f"download.pytorch.org. Original error: {exc2!r}"
                ) from exc2
    else:
        net = resnet50(weights=None)

    in_features = net.fc.in_features
    net.fc = nn.Linear(in_features, emb_dim)
    return net


def build_head(emb_dim: int, num_classes: int, m: float = 0.4, h: float = 0.333,
               s: float = 64.0):
    """AdaFace: quality-adaptive angular margin (Kim et al., CVPR 2022).

    Margin is relaxed for low-norm (low-quality) features, which is exactly the
    regime of the challenge (blur, CCTV, occlusions).
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class _AdaFace(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(num_classes, emb_dim))
            nn.init.xavier_uniform_(self.weight)
            self.m, self.h, self.s = m, h, s
            # running feature-norm statistics (EMA)
            self.register_buffer("running_mean", torch.tensor(20.0))
            self.register_buffer("running_std", torch.tensor(1.0))
            self.momentum = 0.9

        def forward(self, emb, norms, labels=None):
            cos = F.linear(F.normalize(emb), F.normalize(self.weight))
            if labels is None:
                return cos * self.s
            if self.training:
                with torch.no_grad():
                    b_mean = norms.mean()
                    b_std = norms.std(unbiased=False).clamp_min(1e-3)
                    self.running_mean.mul_(self.momentum).add_(b_mean * (1 - self.momentum))
                    self.running_std.mul_(self.momentum).add_(b_std * (1 - self.momentum))
                mean, std = b_mean, b_std
            else:
                mean, std = self.running_mean, self.running_std
            margin = self.m + self.h * (norms - mean) / std   # (B,)
            theta = torch.acos(cos.clamp(-1 + 1e-7, 1 - 1e-7))  # (B, C)
            target = torch.cos(theta + margin.unsqueeze(1))
            one_hot = F.one_hot(labels, num_classes=cos.shape[1]).to(cos.dtype)
            return (cos * (1 - one_hot) + target * one_hot) * self.s

    return _AdaFace()


# ------------------------------------------------------------------ protocol
def embed_all(model, paths: list[str], device, batch_size: int = 64, img_size: int = 112):
    """Deterministic (no augmentation) embeddings for evaluation."""
    import cv2
    import torch
    import torch.nn.functional as F

    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            chunk = paths[i:i + batch_size]
            batch = np.zeros((len(chunk), img_size, img_size, 3), np.float32)
            for j, p in enumerate(chunk):
                img = cv2.imread(p)
                if img is None:
                    continue
                img = cv2.cvtColor(cv2.resize(img, (img_size, img_size)), cv2.COLOR_BGR2RGB)
                batch[j] = img
            x = (batch / 255.0 - 0.5) / 0.5
            x = torch.from_numpy(x).permute(0, 3, 1, 2).to(device)
            out.append(F.normalize(model(x), dim=1).cpu())
    return torch.cat(out).numpy()


def rank1_and_verification(embs: np.ndarray, ids: list[str],
                           max_impostor_pairs: int = 50_000, seed: int = 0):
    """Identity-disjoint evaluation protocol.

    Gallery: first image of every identity.  Probes: all remaining images.
    Returns Rank-1, plus TAR@FAR=1e-3 and EER from genuine/impostor pairs.
    """
    # Defensive: cosine metrics assume unit-norm rows.
    norm = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
    embs = embs / norm
    ids_arr = np.array(ids)
    gallery_idx, probe_idx = [], []
    for uid in np.unique(ids_arr):
        members = np.where(ids_arr == uid)[0]
        gallery_idx.append(members[0])
        probe_idx.extend(members[1:].tolist())
    if not probe_idx:
        return {"rank1": float("nan"), "n_probes": 0}
    gallery = embs[gallery_idx]
    g_ids = ids_arr[gallery_idx]
    probes = embs[probe_idx]
    p_ids = ids_arr[probe_idx]

    sims = probes @ gallery.T                       # (P, G)
    order = np.argsort(-sims, axis=1)
    rank1 = float((g_ids[order[:, 0]] == p_ids).mean())

    # Verification pairs (genuine: same id across different images)
    rng = np.random.default_rng(seed)
    by_id: dict[str, list[int]] = {}
    for i, uid in enumerate(ids):
        by_id.setdefault(uid, []).append(i)
    genuine = []
    for uid, idxs in by_id.items():
        if len(idxs) < 2:
            continue
        for _ in range(min(20, len(idxs) * 2)):
            a, b = rng.choice(idxs, size=2, replace=False)
            genuine.append(float(embs[a] @ embs[b]))
    unique_ids = [u for u, idxs in by_id.items() if len(idxs) >= 2]
    n_imp = min(max_impostor_pairs, 50_000)
    imp = []
    if len(unique_ids) > 1:
        for _ in range(n_imp // 2 + 1):
            u, v = rng.choice(len(unique_ids), size=2, replace=False)
            a = rng.choice(by_id[unique_ids[u]])
            b = rng.choice(by_id[unique_ids[v]])
            imp.append(float(embs[a] @ embs[b]))
            if len(imp) >= n_imp:
                break
    genuine_arr = np.sort(np.array(genuine))
    imp_arr = np.sort(np.array(imp))

    # Vectorised ROC without materialising a (thresholds x pairs) matrix.
    thresholds = np.linspace(-1.0, 1.0, 1001)
    far = 1.0 - np.searchsorted(imp_arr, thresholds, side="left") / max(len(imp_arr), 1)
    tar = 1.0 - np.searchsorted(genuine_arr, thresholds, side="left") / max(len(genuine_arr), 1)
    idx = int(np.argmin(np.abs(far - 1e-3)))
    eer_idx = int(np.argmin(np.abs(far - (1 - tar))))
    return {
        "rank1": rank1,
        "n_probes": int(len(probe_idx)),
        "n_gallery": int(len(gallery_idx)),
        "n_genuine": int(len(genuine_arr)),
        "n_impostor": int(len(imp_arr)),
        "tar_at_far_1e-3": float(tar[idx]),
        "far_at_operating_point": float(far[idx]),
        "eer": float((far[eer_idx] + (1 - tar[eer_idx])) / 2),
        "mean_genuine_score": float(genuine_arr.mean()),
        "mean_impostor_score": float(imp_arr.mean()),
    }


# --------------------------------------------------------------------- train
def run(args) -> dict:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    t0 = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    data_root, source = find_image_root()
    _log(f"Data source: {source} -> {data_root}")

    identities = collect_identities(data_root, min_images=2, max_images=args.max_per_id)
    all_ids = sorted(identities)
    _log(f"Identities with >=2 images: {len(all_ids)} | images: "
         f"{sum(len(v) for v in identities.values())}")

    if len(all_ids) < 6:
        raise RuntimeError(f"Too few identities ({len(all_ids)}) to split")

    # --- identity-disjoint split (never split images of the same person) ---
    rng = random.Random(args.seed)
    rng.shuffle(all_ids)
    n_val = max(2, int(round(len(all_ids) * args.val_id_frac)))
    if args.smoke:
        n_val = max(2, min(4, len(all_ids) // 3))
    val_ids = sorted(all_ids[:n_val])
    train_ids = sorted(all_ids[n_val:])
    _log(f"Train identities: {len(train_ids)} | held-out identities: {len(val_ids)}")
    assert not set(train_ids) & set(val_ids), "identity leakage between splits!"

    ds = FaceDataset(identities, train_ids, augment=True, seed=args.seed)
    _log(f"Training images: {len(ds)} | classes: {len(train_ids)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_backbone(512, pretrained=not args.no_pretrained).to(device)
    head = build_head(512, len(train_ids), args.margin, args.h, args.scale).to(device)
    _log(f"Device: {device} | params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # ---------- baseline measurement BEFORE fine-tuning (same protocol) ----------
    val_paths, val_labels = [], []
    for uid in val_ids:
        val_paths.extend(identities[uid])
        val_labels.extend([uid] * len(identities[uid]))
    if args.max_eval_images and len(val_paths) > args.max_eval_images:
        step = len(val_paths) / args.max_eval_images
        keep = [int(i * step) for i in range(args.max_eval_images)]
        val_paths = [val_paths[i] for i in keep]
        val_labels = [val_labels[i] for i in keep]

    _log(f"Evaluating pretrained model on {len(val_paths)} held-out images "
         f"({len(val_ids)} unseen identities)...")
    base_embs = embed_all(model, val_paths, device, args.batch_size)
    baseline = rank1_and_verification(base_embs, val_labels, seed=args.seed)
    _log(f"BASELINE  rank1={baseline['rank1']:.4f} "
         f"TAR@FAR1e-3={baseline['tar_at_far_1e-3']:.4f} EER={baseline['eer']:.4f}")

    # ------------------------------- optimisation -------------------------------
    ds.rng = random.Random(args.seed + 1)  # per-epoch-ish augmentation stream
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=(device.type == "cuda"),
                        drop_last=len(ds) > args.batch_size)
    optimizer = torch.optim.AdamW([
        {"params": model.parameters(), "lr": args.lr_backbone},
        {"params": head.parameters(), "lr": args.lr_head},
    ], weight_decay=args.wd)
    steps_per_epoch = len(loader)
    total_steps = max(1, steps_per_epoch * args.epochs)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[args.lr_backbone * 3, args.lr_head], total_steps=total_steps,
        pct_start=0.15)
    use_amp = device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except TypeError:                      # torch < 2.3
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    criterion = torch.nn.CrossEntropyLoss()

    best = {"rank1": baseline["rank1"]}
    history = []
    for epoch in range(args.epochs):
        model.train()
        head.train()
        run_loss, seen, correct = 0.0, 0, 0
        for step, (imgs, labels) in enumerate(loader):
            if args.smoke and step >= args.smoke_steps:
                break
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                emb = model(imgs)
                logits = head(emb, emb.norm(dim=1).detach(), labels)
                loss = criterion(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            sched.step()
            run_loss += loss.item() * imgs.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            seen += imgs.size(0)
        train_acc = correct / max(seen, 1)
        _log(f"epoch {epoch+1}/{args.epochs} loss={run_loss/max(seen,1):.4f} "
             f"train_acc={train_acc:.4f} lr={sched.get_last_lr()[0]:.2e}")

        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            embs = embed_all(model, val_paths, device, args.batch_size)
            metrics = rank1_and_verification(embs, val_labels, seed=args.seed)
            metrics["epoch"] = epoch + 1
            metrics["train_acc"] = train_acc
            history.append(metrics)
            _log(f"  held-out rank1={metrics['rank1']:.4f} "
                 f"TAR@FAR1e-3={metrics['tar_at_far_1e-3']:.4f} EER={metrics['eer']:.4f}")
            if metrics["rank1"] > best.get("rank1", -1):
                best = dict(metrics)
                torch.save({"backbone": model.state_dict(),
                            "head": head.state_dict(),
                            "train_ids": train_ids, "val_ids": val_ids,
                            "epoch": epoch + 1, "img_size": args.img_size,
                            "channel_order": "RGB",
                            "normalization": "(x/255 - 0.5)/0.5"},
                           os.path.join(args.out_dir, "face_ft.pth"))
                _log("  saved new best checkpoint")

    results = {
        "source": source,
        "n_train_ids": len(train_ids),
        "n_val_ids": len(val_ids),
        "n_train_images": len(ds),
        "baseline": baseline,
        "best": best,
        "history": history,
        "args": vars(args),
        "elapsed_sec": round(time.time() - t0, 1),
    }

    # Always persist the final model, even if it never beat the zero-shot
    # baseline — otherwise a "no improvement" run produces no artifact.
    last_ckpt = os.path.join(args.out_dir, "face_ft_last.pth")
    torch.save({"backbone": model.state_dict(),
                "head": head.state_dict(),
                "train_ids": train_ids, "val_ids": val_ids,
                "epoch": args.epochs, "img_size": args.img_size,
                "channel_order": "RGB",
                "normalization": "(x/255 - 0.5)/0.5"}, last_ckpt)

    ckpt = os.path.join(args.out_dir, "face_ft.pth")
    export_src = "best" if os.path.exists(ckpt) else "last"
    export_path = ckpt if export_src == "best" else last_ckpt
    onnx_path = os.path.join(args.out_dir, "face_ft.onnx")
    _log(f"Exporting {export_src} checkpoint to ONNX...")
    results["onnx"] = export_onnx(export_path, onnx_path, args.img_size, device="cpu")
    results["onnx_source"] = export_src

    with open(os.path.join(args.out_dir, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    _log(f"Done in {results['elapsed_sec']}s")
    return results


def export_onnx(ckpt_path: str, onnx_path: str, img_size: int = 112, device: str = "cpu"):
    import torch
    import onnxruntime as ort

    model = build_backbone(512, pretrained=False)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["backbone"])
    model.eval().to(device)
    dummy = torch.randn(1, 3, img_size, img_size, device=device)

    kwargs = dict(input_names=["input"], output_names=["embedding"],
                  dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
                  opset_version=17)
    try:
        torch.onnx.export(model, dummy, onnx_path, dynamo=False, **kwargs)
    except TypeError:                      # torch < 2.5
        torch.onnx.export(model, dummy, onnx_path, **kwargs)

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    out = sess.run(None, {inp.name: dummy.cpu().numpy()})[0]
    with torch.no_grad():
        ref = model(dummy).cpu().numpy()
    cos = float(np.dot(ref.ravel(), out.ravel()) /
                (np.linalg.norm(ref) * np.linalg.norm(out) + 1e-12))
    _log(f"ONNX input={inp.name}{inp.shape} output_shape={out.shape} parity_cos={cos:.6f}")
    if cos < 0.999:
        raise RuntimeError(f"ONNX parity check failed (cos={cos:.6f})")
    return {"path": onnx_path, "input": inp.name, "shape": list(inp.shape),
            "parity_cos": cos, "size_mb": round(os.path.getsize(onnx_path) / 1e6, 1)}


def preflight() -> dict:
    info = describe_environment()
    _log(json.dumps(info, indent=2))
    # dataset
    try:
        root, source = find_image_root()
        ids = collect_identities(root, min_images=2, max_images=60)
        info.update({"data_source": source, "data_root": root,
                     "n_identities": len(ids),
                     "n_images": int(sum(len(v) for v in ids.values()))})
        _log(f"DATA OK: {source} | {len(ids)} identities | "
             f"{info['n_images']} images")
        sample = next(iter(ids.values()))[0]
        import cv2
        img = cv2.imread(sample)
        info["sample_image_shape"] = None if img is None else list(img.shape)
        _log(f"sample image: {sample} shape={info.get('sample_image_shape')}")
    except Exception as exc:
        info["data_error"] = repr(exc)
        _log(f"DATA FAIL: {exc!r}")

    # pretrained weights reachability
    try:
        import torchvision.models as tvm
        from torchvision.models import ResNet50_Weights
        m = tvm.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        info["pretrained_ok"] = True
        _log("PRETRAINED WEIGHTS OK (cached or downloaded)")
        del m
    except Exception as exc:
        info["pretrained_ok"] = False
        info["pretrained_error"] = repr(exc)
        _log(f"PRETRAINED FAIL: {exc!r}")

    _log("PREFLIGHT JSON " + json.dumps(info))
    return info


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=None, help="override auto-discovery")
    p.add_argument("--out-dir", default="/kaggle/working")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--img-size", type=int, default=112)
    p.add_argument("--lr-backbone", type=float, default=2e-5)
    p.add_argument("--lr-head", type=float, default=2e-3)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--margin", type=float, default=0.4)
    p.add_argument("--h", type=float, default=0.333)
    p.add_argument("--scale", type=float, default=64.0)
    p.add_argument("--val-id-frac", type=float, default=0.1)
    p.add_argument("--max-per-id", type=int, default=60)
    p.add_argument("--max-eval-images", type=int, default=1200)
    p.add_argument("--eval-every", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke-steps", type=int, default=6)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.data_root:
        # monkeypatch discovery for local testing
        globals()["find_image_root"] = lambda: (args.data_root, "local-override")
    if args.smoke:
        args.epochs = min(args.epochs, 1)
        args.batch_size = min(args.batch_size, 8)
        args.workers = 0
        args.max_per_id = min(args.max_per_id, 8)
        args.max_eval_images = 60
        args.out_dir = args.out_dir if args.out_dir != "/kaggle/working" else "/tmp/opencode/smoke"
    if args.preflight:
        preflight()
        return
    results = run(args)
    print("\nSUMMARY " + json.dumps({k: results[k] for k in
                                     ("baseline", "best", "n_train_ids", "n_val_ids")},
                                    indent=2))


if __name__ == "__main__":
    main()

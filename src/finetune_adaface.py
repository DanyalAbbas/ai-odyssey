#!/usr/bin/env python3
"""Fine-tune AdaFace IR-50 on the collected Pakistani corpus (Kaggle GPU).

This is the compliance run the brief asks for: train on data we collected, with
identities that never appear in the evaluation benchmark.

Data
----
`bench_train/<identity>/<n>.jpg` — 112x112 ArcFace-aligned crops produced by
`src/clean_train_faces.py`. The manifest carries gender and a hijab/dupatta flag;
the corpus is deliberately weighted towards veiled women.

Veiling
-------
Real niqab photos with identity labels do not exist, so veiling is trained
synthetically: each aligned crop has a probability of having a cloth occluder
painted over the lower face (niqab), the hairline and sides (hijab), or the
mouth and nose (surgical mask). Because the crop is aligned, the occluder lands
in the right place every time, and the label is preserved by construction.

Training
--------
Backbone: AdaFace IR-50 pretrained on MS1MV2, loaded from an attached Kaggle
dataset. BatchNorm statistics are frozen (small corpus) and the backbone runs at
a low learning rate so the pretrained representation is adapted, not destroyed.
ArcFace head is trained from scratch and discarded after training; the exported
ONNX is the 512-d embedding backbone.

Output
------
`fine_tuned_adaface_ir50.onnx` and `train_report.json` in /kaggle/working.

    python finetune_adaface.py --data-root /kaggle/input/<dataset>/bench_train
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

# --------------------------------------------------------------------------- #
# AdaFace IR-50
#
# Key layout is taken from the released checkpoint, not guessed: input_layer ->
# body.0..23 (no SE in the IR-50 variant) -> output_layer, plus a separate
# `head` (85,742-way ArcFace/AdaFace classifier) that is discarded here.
# --------------------------------------------------------------------------- #
class Flatten(nn.Module):
    def forward(self, x):
        return x.flatten(1)


class IRBlock(nn.Module):
    """BasicBlockIR from AdaFace ``net.py``.

    The stride is applied to the *second* 3x3 conv, and a block whose input and
    output widths match takes a parameter-free MaxPool shortcut. Both details
    differ from the usual arcface_torch block and matter for weight alignment.
    """

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        if inplanes == planes:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes))
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(inplanes),
            nn.Conv2d(inplanes, planes, 3, 1, 1, bias=False),
            nn.BatchNorm2d(planes),
            nn.PReLU(planes),
            nn.Conv2d(planes, planes, 3, stride, 1, bias=False),
            nn.BatchNorm2d(planes),
        )

    def forward(self, x):
        return self.res_layer(x) + self.shortcut_layer(x)


class IResNet(nn.Module):
    def __init__(self, layers, num_features=512, dropout=0.4):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
        )
        body, inplanes = [], 64
        for planes, n in zip([64, 128, 256, 512], layers):
            # every stage downsamples: 112 -> 56 -> 28 -> 14 -> 7
            body.append(IRBlock(inplanes, planes, 2))
            body.extend(IRBlock(planes, planes, 1) for _ in range(n - 1))
            inplanes = planes
        self.body = nn.Sequential(*body)
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(dropout),
            Flatten(),
            nn.Linear(512 * 7 * 7, num_features),
            nn.BatchNorm1d(num_features, affine=False),
        )

    def forward(self, x):
        x = self.output_layer(self.body(self.input_layer(x)))
        return F.normalize(x)          # training handles its own scale


def IR_50(num_features=512):
    return IResNet([3, 4, 14, 3], num_features)


# --------------------------------------------------------------------------- #
# Synthetic veiling# --------------------------------------------------------------------------- #
# Synthetic veiling
# --------------------------------------------------------------------------- #
PALETTE = [
    (18, 18, 22), (30, 34, 48), (58, 40, 30), (70, 70, 74),
    (90, 30, 40), (150, 140, 125), (220, 215, 205), (12, 40, 30),
]


def _soft_paint(img, mask, color):
    mask = cv2_blur(mask)
    return (img * (1 - mask[..., None]) + np.array(color, np.float32) * mask[..., None])


def cv2_blur(mask):
    import cv2
    return cv2.GaussianBlur(mask, (11, 11), 0)


def occlude(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """img: HxWx3 float32 in [0,255]. Returns occluded copy."""
    import cv2
    h, w = img.shape[:2]           # 112 x 112
    color = PALETTE[rng.integers(len(PALETTE))]
    kind = rng.random()

    if kind < 0.45:                # niqab: lower face plus the sides of the head
        mask = np.zeros((h, w), np.float32)
        cv2.ellipse(mask, (56, 84), (42, 38), 0, 0, 360, 1.0, -1)
        mask[52:, :14] = 1.0
        mask[52:, 98:] = 1.0
        mask[:34, :] = np.maximum(mask[:34, :], 0.85)
    elif kind < 0.72:              # hijab: hairline and sides, face visible
        mask = np.zeros((h, w), np.float32)
        mask[:26, :] = 1.0
        mask[:, :13] = 1.0
        mask[:, 99:] = 1.0
    else:                          # surgical mask: nose, mouth, chin
        mask = np.zeros((h, w), np.float32)
        cv2.rectangle(mask, (16, 60), (96, 106), 1.0, -1)

    return _soft_paint(img, mask, color)


class FaceDataset(Dataset):
    def __init__(self, root: Path, train=True, veil_p=0.6):
        self.base = ImageFolder(str(root))
        self.samples = self.base.samples
        self.targets = [t for _, t in self.base.samples]
        self.classes = self.base.classes
        self.train = train
        self.veil_p = veil_p

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        import cv2
        path, y = self.samples[i]
        img = cv2.imread(path)                      # BGR uint8
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

        if self.train:
            rng = np.random.default_rng()
            if rng.random() < 0.5:
                img = img[:, ::-1].copy()
            if rng.random() < self.veil_p:
                img = occlude(img, rng)
            # mild photometric jitter, no geometry (crop is already aligned)
            img = np.clip(img * rng.uniform(0.85, 1.15) + rng.uniform(-12, 12), 0, 255)

        x = (img / 255.0 - 0.5) / 0.5
        return torch.from_numpy(x.transpose(2, 0, 1)).float(), y


# --------------------------------------------------------------------------- #
# ArcFace head
# --------------------------------------------------------------------------- #
class ArcFace(nn.Module):
    def __init__(self, emb_dim, n_classes, s=64.0, m=0.35):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_classes, emb_dim))
        nn.init.xavier_uniform_(self.weight)
        self.s, self.m = s, m
        self.cos_m, self.sin_m = math.cos(m), math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, emb, label):
        cosine = F.linear(F.normalize(emb), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(1e-9, 1.0))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        onehot = torch.zeros_like(cosine)
        onehot.scatter_(1, label.view(-1, 1), 1.0)
        return self.s * (onehot * phi + (1 - onehot) * cosine)


# --------------------------------------------------------------------------- #
def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False


def _torch_load(path: Path):
    """Load either a plain state dict or a Lightning checkpoint.

    The public AdaFace checkpoint is a Lightning file, so unpickling it touches
    `pytorch_lightning.callbacks.ModelCheckpoint`. Kaggle ships Lightning; a bare
    environment may not, so register inert stubs and retry rather than adding a
    200 MB dependency for one class reference.
    """
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except ModuleNotFoundError:
        import sys as _sys
        import types
        names = ["pytorch_lightning", "pytorch_lightning.callbacks",
                 "pytorch_lightning.callbacks.model_checkpoint"]
        for name in names:
            _sys.modules.setdefault(name, types.ModuleType(name))

        class ModelCheckpoint:      # only needs to exist for unpickling
            pass

        _sys.modules[names[2]].ModelCheckpoint = ModelCheckpoint
        _sys.modules[names[0]].callbacks = _sys.modules[names[1]]
        _sys.modules[names[1]].model_checkpoint = _sys.modules[names[2]]
        return torch.load(str(path), map_location="cpu", weights_only=False)


def load_backbone(ckpt_path: Path, device):
    ckpt = _torch_load(ckpt_path)
    raw = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    # Lightning prefixes everything with "model."; the ArcFace/AdaFace head is
    # separate from the backbone and is discarded. Strip first, then filter --
    # filtering on the prefixed name would silently keep `model.head.*`.
    sd = {}
    for key, value in raw.items():
        key = key.replace("model.", "", 1)
        if key.split(".")[0] == "head":
            continue
        sd[key] = value
    net = IR_50()
    net.load_state_dict(sd, strict=True)   # layout drift must fail loudly
    print(f"backbone: {len(sd)} tensors loaded from {ckpt_path.name}", flush=True)
    return net.to(device)


def evaluate_rank1(model, ds, device, batch=64):
    """Rank-1 on held-out *images* of trained identities (sanity, not the real eval)."""
    model.eval()
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=2)
    embs, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=torch.cuda.is_available()):
                e = model(x)
            embs.append(F.normalize(e.float()).cpu())
            ys.append(y)
    E = torch.cat(embs)
    Y = torch.cat(ys).numpy()
    # reference: mean embedding per identity over ALL images (this is a fit check,
    # deliberately optimistic; the honest number is the benchmark run locally)
    refs, ids = [], []
    for c in np.unique(Y):
        refs.append(F.normalize(E[Y == c].mean(0), dim=0))
        ids.append(c)
    R = torch.stack(refs)
    sims = E @ R.T
    pred = torch.tensor(ids)[sims.argmax(1)].numpy()
    return float((pred == Y).mean())


def resolve_data_root(requested: str) -> Path:
    """Locate the corpus, tolerating how Kaggle names the mount.

    Kaggle mounts a dataset under /kaggle/input/<slug>/, and the mount is absent
    when the dataset version was still processing at push time. Falling back to
    any */bench_train and printing what *is* mounted turns a bare "missing
    dataset" into a diagnosable log line.
    """
    root = Path(requested)
    if root.is_dir():
        return root
    for cand in sorted(Path("/kaggle/input").glob("*/bench_train")):
        if cand.is_dir():
            print(f"data root {root} missing; using {cand}", flush=True)
            return cand
    print(f"missing dataset: {root}", flush=True)
    inputs = Path("/kaggle/input")
    if inputs.is_dir():
        print("mounted inputs:", flush=True)
        for p in sorted(inputs.iterdir()):
            print(f"  {p}", flush=True)
    raise SystemExit(2)


def resolve_ckpt(requested: str) -> Path:
    """Same idea for the AdaFace checkpoint dataset."""
    root = Path(requested)
    if root.is_file():
        return root
    for cand in sorted(Path("/kaggle/input").glob("*/adaface_ir50_ms1mv2.ckpt")):
        if cand.is_file():
            print(f"checkpoint {root} missing; using {cand}", flush=True)
            return cand
    raise SystemExit(f"missing checkpoint: {root}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/kaggle/input/techfest-pak-train-faces/bench_train")
    ap.add_argument("--ckpt", default="/kaggle/input/adaface-checkpoint/adaface_ir50_ms1mv2.ckpt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr-backbone", type=float, default=5e-5)
    ap.add_argument("--lr-head", type=float, default=3e-3)
    ap.add_argument("--veil-p", type=float, default=0.6)
    ap.add_argument("--out", default="/kaggle/working")
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    data_root = resolve_data_root(args.data_root)
    ckpt = resolve_ckpt(args.ckpt)

    ds = FaceDataset(data_root, train=True, veil_p=args.veil_p)
    print(f"train: {len(ds)} crops / {len(ds.classes)} identities", flush=True)
    if len(ds.classes) < 20:
        raise SystemExit("too few identities; refusing to fine-tune")

    # held-out images for a sanity check (2 per identity)
    rng = np.random.default_rng(0)
    idx = np.arange(len(ds))
    val_idx = []
    for c in np.unique(ds.targets):
        cand = idx[np.array(ds.targets) == c]
        val_idx += list(rng.choice(cand, size=min(2, len(cand)), replace=False))
    val_idx = set(int(i) for i in val_idx)
    train_ds = torch.utils.data.Subset(ds, [i for i in idx if i not in val_idx])
    val_ds = FaceDataset(data_root, train=False)
    val_ds.samples = [ds.samples[i] for i in sorted(val_idx)]
    val_ds.targets = [ds.targets[i] for i in sorted(val_idx)]
    print(f"  {len(train_ds)} train / {len(val_ds)} held-out images", flush=True)

    net = load_backbone(ckpt, device)
    head = ArcFace(512, len(ds.classes)).to(device)
    freeze_bn(net)

    opt = torch.optim.AdamW([
        {"params": [p for p in net.parameters() if p.requires_grad], "lr": args.lr_backbone},
        {"params": head.parameters(), "lr": args.lr_head},
    ], weight_decay=5e-4)
    steps = math.ceil(len(train_ds) / args.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps * args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                        num_workers=4, drop_last=len(train_ds) > args.batch,
                        persistent_workers=True)
    t0 = time.time()
    history = []
    for ep in range(1, args.epochs + 1):
        net.train()
        freeze_bn(net)                 # keep pretrained statistics
        running, seen, correct = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                emb = net(x)
                logits = head(emb, y)
                loss = F.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += float(loss) * len(y)
            seen += len(y)
            correct += int((logits.argmax(1) == y).sum())
        history.append({"epoch": ep, "loss": running / max(1, seen),
                        "train_acc": correct / max(1, seen)})
        print(f"epoch {ep:2d}/{args.epochs}  loss {running/max(1,seen):.4f}  "
              f"acc {correct/max(1,seen):.3f}  "
              f"({(time.time()-t0)/ep:.0f}s/ep)", flush=True)

    val_rank1 = evaluate_rank1(net, val_ds, device)
    print(f"held-out-image rank-1 (fit check): {val_rank1:.4f}", flush=True)

    # ---------------- export ----------------
    net.eval()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    onnx_path = out / "fine_tuned_adaface_ir50.onnx"
    dummy = torch.randn(1, 3, 112, 112, device=device)
    try:                       # torch >= 2.9 defaults to the dynamo exporter
        torch.onnx.export(net, dummy, str(onnx_path), opset_version=17,
                          input_names=["input"], output_names=["embedding"],
                          dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
                          dynamo=False)
    except TypeError:
        torch.onnx.export(net, dummy, str(onnx_path), opset_version=17,
                          input_names=["input"], output_names=["embedding"],
                          dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}})
    print(f"exported {onnx_path} ({onnx_path.stat().st_size/1e6:.0f} MB)", flush=True)

    # parity check
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        torch_out = net(dummy).cpu().numpy()[0]
    onnx_out = sess.run(None, {"input": dummy.cpu().numpy()})[0][0]
    cos = float(np.dot(torch_out, onnx_out) /
                (np.linalg.norm(torch_out) * np.linalg.norm(onnx_out)))
    print(f"onnx parity cosine: {cos:.6f}", flush=True)

    torch.save({k: v.cpu() for k, v in net.state_dict().items()},
               out / "fine_tuned_adaface_ir50_sd.pt")

    report = {"identities": len(ds.classes), "crops": len(ds),
              "train_images": len(train_ds), "heldout_images": len(val_ds),
              "epochs": args.epochs, "batch": args.batch,
              "lr_backbone": args.lr_backbone, "lr_head": args.lr_head,
              "veil_p": args.veil_p, "val_rank1_fitcheck": val_rank1,
              "onnx_parity_cosine": cos, "history": history,
              "classes": ds.classes, "device": str(device)}
    (out / "train_report.json").write_text(json.dumps(report, indent=2))
    print("done", flush=True)


if __name__ == "__main__":
    main()

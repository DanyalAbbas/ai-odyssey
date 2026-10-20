"""
Pakistani-context evaluation harness (baseline ArcFace vs fine-tuned ONNX).

Reports, for the 23-identity / 437-probe split in data/pak_fig:
  * end-to-end accuracy (FTE counted as failure) and Rank-1 among detections
  * TAR@FAR=1e-3 / 1e-4, EER, AUC
  * precision / recall / F1 at the operating threshold
  * threshold sweep
  * per-identity and per-quality-bucket breakdown (fairness proxy)
  * bootstrap 95% CIs

Embeddings are cached to disk (keyed by model + settings) because a full pass
over 1.3k images on CPU takes minutes.

Usage:
    python evaluate_pak.py --model baseline
    python evaluate_pak.py --model onnx --onnx face_ft.onnx
    python evaluate_pak.py --model baseline --det-thresh 0.2 --tag lowdet
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from imgio import imread_any

from insightface.app import FaceAnalysis
from insightface.utils import face_align

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
DEFAULT_CFG = dict(detector="v2-escalating", det_size=640, det_thresh=0.5,
                   min_iod=64, max_yaw=30, min_blur=40, allow_fallback=True)


# ------------------------------------------------------------------ embedder
class Embedder:
    """Detect (SCRFD) -> quality gate -> 5-pt alignment -> embedding."""

    def __init__(self, backend: str = "baseline", onnx_path: str | None = None,
                 det_size: int = 640, det_thresh: float = 0.5,
                 min_iod: float = 64, max_yaw: float = 30, min_blur: float = 40,
                 allow_fallback: bool = True, with_genderage: bool = False,
                 pad_retry: float = 0.3, flip_tta: bool = False,
                 batch_size: int = 32, pad_model: str | None = None,
                 pad_threshold: float = 0.5):
        if backend not in {"baseline", "antelope", "onnx"}:
            raise ValueError(f"Invalid backend: {backend!r}. Must be one of: baseline, antelope, onnx")
        self.backend = backend
        self.batch_size = batch_size
        self.det_thresh = det_thresh
        self.min_iod, self.max_yaw, self.min_blur = min_iod, max_yaw, min_blur
        self.allow_fallback = allow_fallback
        self.pad_retry = pad_retry
        self.flip_tta = flip_tta

        mods = ["detection", "genderage"] if with_genderage else ["detection"]
        if backend in {"baseline", "antelope"}:
            mods.append("recognition")
        pack = "antelopev2" if backend == "antelope" else "buffalo_l"
        self.pack = pack
        self.det_size = det_size
        self._lo = None
        self.app = FaceAnalysis(name=pack, allowed_modules=mods)
        self.app.prepare(ctx_id=-1, det_size=(det_size, det_size))

        self.sess = None
        if backend == "onnx":
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = os.cpu_count() or 4
            self.sess = ort.InferenceSession(onnx_path, so,
                                             providers=["CPUExecutionProvider"])
            self.input_name = self.sess.get_inputs()[0].name

        self.spoof = None
        if pad_model:
            from pad import SpoofDetector
            self.spoof = SpoofDetector(pad_model, threshold=pad_threshold)

    # ---- quality gate (same thresholds as inference.py) -------------------
    def quality(self, face, img, pad: int = 0):
        kps = face.kps
        iod = float(np.linalg.norm(kps[0] - kps[1]))
        if iod < self.min_iod:
            return False, f"low_iod_{iod:.0f}"
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        # when the image was padded, ignore the synthetic border pixels
        x1, y1 = max(pad, x1), max(pad, y1)
        x2, y2 = min(img.shape[1] - pad, x2), min(img.shape[0] - pad, y2)
        if y2 > y1 and x2 > x1:
            gray = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
            blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if blur < self.min_blur:
                return False, f"blur_{blur:.0f}"
        eye_mid = (kps[0] + kps[1]) / 2
        yaw = abs(kps[2][0] - eye_mid[0]) / (iod + 1e-6) * 90
        if yaw > self.max_yaw:
            return False, f"yaw_{yaw:.0f}"
        return True, "ok"

    @staticmethod
    def _area(f) -> float:
        return float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    @staticmethod
    def _iou(a, b) -> float:
        ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
        ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
        iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
        inter = iw * ih
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / ua if ua > 0 else 0.0

    def _coarse(self, img):
        """Detection pass at 320: rescues faces that are too large for the
        640 pass (SCRFD has an upper face-size limit, and small source images
        are upscaled past it)."""
        if self._lo is None:
            self._lo = FaceAnalysis(name=self.pack, allowed_modules=["detection"])
            self._lo.prepare(ctx_id=-1, det_size=(320, 320))
        return [f for f in self._lo.get(img) if f.det_score >= self.det_thresh]

    def _detect(self, img):
        """SCRFD expects scene context; tight crops and oversize faces both
        break it. Two escalations:
          1. nothing found        -> retry on a reflected-border copy
          2. only tiny/low-res faces found -> retry at 320 and merge
        """
        faces = [f for f in self.app.get(img) if f.det_score >= self.det_thresh]
        pad = 0
        if not faces and self.pad_retry > 0:
            h, w = img.shape[:2]
            b = int(round(max(h, w) * self.pad_retry))
            padded = cv2.copyMakeBorder(img, b, b, b, b, cv2.BORDER_REFLECT_101)
            faces = [f for f in self.app.get(padded) if f.det_score >= self.det_thresh]
            if faces:
                img, pad = padded, b
        if faces and self.pad_retry > 0:
            big = max(faces, key=self._area)
            iod = float(np.linalg.norm(big.kps[0] - big.kps[1]))
            frac = self._area(big) / max(1, img.shape[0] * img.shape[1])
            if iod < self.min_iod or frac < 0.02:
                extra = [f for f in self._coarse(img)
                         if all(self._iou(f.bbox, g.bbox) < 0.5 for g in faces)]
                faces = faces + extra
        return img, faces, pad

    def _onnx_embed(self, aligned_bgr: np.ndarray) -> np.ndarray:
        """aligned_bgr is a 112x112 BGR crop from face_align.norm_crop.
        Training used RGB in [-1, 1], so convert accordingly."""
        rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = ((rgb - 0.5) / 0.5).transpose(2, 0, 1)[None]
        out = self.sess.run(None, {self.input_name: x})[0][0]
        return out / (np.linalg.norm(out) + 1e-12)

    # ---- batched embedding ------------------------------------------------
    def embed_crops(self, crops: list) -> np.ndarray:
        """Embed a list of 112x112 BGR crops in batches.

        Detection is the expensive stage (~156 ms/img); recognition with batch=1
        costs another ~390 ms/img. Batching the recognition model recovers most
        of that, which matters because we run many evaluation variants.
        """
        if not crops:
            return np.zeros((0, 512), np.float32)
        out = np.zeros((len(crops), 512), np.float32)
        for i in range(0, len(crops), self.batch_size):
            chunk = [c for c in crops[i:i + self.batch_size]]
            if self.backend in {"baseline", "antelope"}:
                rec = self.app.models["recognition"]
                feats = np.asarray(rec.get_feat(chunk), np.float32)
            else:
                rgb = np.stack([cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in chunk])
                x = ((rgb.astype(np.float32) / 255.0 - 0.5) / 0.5).transpose(0, 3, 1, 2)
                feats = np.asarray(
                    self.sess.run(None, {self.input_name: x})[0], np.float32)
            norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12
            out[i:i + len(chunk)] = feats / norms
        return out

    def crop(self, img):
        """Detect + quality-gate + align one image. Returns (crop or None,
        reason, meta)."""
        img, faces, pad = self._detect(img)
        if not faces:
            return None, "no_face", None

        ok_faces = [(f, *self.quality(f, img, pad)) for f in faces]
        good = [f for f, ok, _ in ok_faces if ok]
        chosen, reason = (max(good, key=lambda f: (f.bbox[2] - f.bbox[0]) *
                             (f.bbox[3] - f.bbox[1])), "ok") if good else (None, None)

        if chosen is None:
            if not self.allow_fallback:
                return None, next((r for _f, ok, r in ok_faces if not ok), "quality"), None
            chosen = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) *
                         (f.bbox[3] - f.bbox[1]))
            reason = next((r for f, ok, r in ok_faces if f is chosen and not ok),
                          "fallback_quality")
        if pad and reason == "ok":
            reason = "ok_padded"

        meta = {"det_score": float(chosen.det_score),
                "bbox": [float(v) - pad for v in chosen.bbox],
                "pad": int(pad),
                "sex": getattr(chosen, "sex", None),
                "age": getattr(chosen, "age", None)}
        if self.spoof is not None:
            # gate on the *padded* image, since chosen.bbox lives in that frame
            meta.update(self.spoof.predict(img, chosen.bbox))

        aligned = face_align.norm_crop(img, landmark=chosen.kps, image_size=112)
        return aligned, reason, meta

    def embed(self, path):
        """Single-image embedding (used for spot checks / subgroup tooling)."""
        img = imread_any(path)
        if img is None:
            return None, "read_error", None
        crop, reason, meta = self.crop(img)
        if crop is None:
            return None, reason, None
        crops = [crop]
        if self.flip_tta:
            crop2, _, _ = self.crop(cv2.flip(img, 1))
            if crop2 is not None:
                crops.append(crop2)
        embs = self.embed_crops(crops)
        v = embs.sum(0)
        v = v / (np.linalg.norm(v) + 1e-12)
        if len(crops) > 1 and reason == "ok":
            reason = "ok_flip"
        return v, reason, meta


# ------------------------------------------------------------------- caching
def cache_key(cfg: dict) -> str:
    blob = json.dumps({k: cfg[k] for k in sorted(cfg)}, default=str)
    return hashlib.md5(blob.encode()).hexdigest()[:12]


def _check_root(z, root: Path):
    """Guard against two datasets that share a folder name (e.g. `gallery`)
    silently reusing each other's cache. Older caches have no `root` field and
    are accepted as-is."""
    if "root" not in z.files:
        return
    cached = str(z["root"])
    if Path(cached).resolve() != root.resolve():
        raise SystemExit(
            f"cache collision: {cached} != {root.resolve()}\n"
            f"pass a different --cache directory for this benchmark")


def embed_folder(emb: Embedder, root: Path, cache_dir: Path, cfg: dict,
                 with_meta: bool = False) -> dict:
    # Detection + alignment depend on one set of settings, the embedding on
    # another. Cache them separately so swapping the recognition model (the
    # expensive experiment) never re-runs detection.
    det_cfg = {k: cfg[k] for k in ("detector", "det_size", "det_thresh", "min_iod",
                                   "min_blur", "max_yaw", "pad_retry", "flip_tta",
                                   "allow_fallback", "pad") if k in cfg}
    emb_cfg = {k: cfg[k] for k in ("model", "model_bytes") if k in cfg}
    cache_dir.mkdir(parents=True, exist_ok=True)
    emb_key = {**det_cfg, **emb_cfg}
    emb_file = cache_dir / f"{root.name}_{cache_key(emb_key)}.npz"
    if emb_file.exists():
        z = np.load(emb_file, allow_pickle=True)
        _check_root(z, root)
        return {"files": list(z["files"]), "embs": list(z["embs"]),
                "reasons": list(z["reasons"]), "meta": list(z["meta"])}

    crops_file = cache_dir / f"{root.name}_crops_{cache_key(det_cfg)}.npz"
    if crops_file.exists():
        z = np.load(crops_file, allow_pickle=True)
        _check_root(z, root)
        files, reasons = list(z["files"]), list(z["reasons"])
        metas = list(z["meta"])
        crops, flips = z["crops"], z["flips"]
        print(f"  {root.name}: reused cached crops ({len(files)} images)", flush=True)
    else:
        paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS)
        print(f"  {root.name}: cropping {len(paths)} images "
              f"(detection is the slow stage)", flush=True)
        files, reasons, metas = [], [], []
        crops = np.zeros((len(paths), 112, 112, 3), np.uint8)
        flips = np.zeros((len(paths), 112, 112, 3), np.uint8)
        t0 = time.time()
        for i, p in enumerate(paths, 1):
            img = imread_any(p)
            files.append(str(p.relative_to(root)))
            if img is None:
                reasons.append("read_error"); metas.append({})
                continue
            crop, reason, meta = emb.crop(img)
            reasons.append(reason)
            metas.append(meta or {})
            if crop is not None:
                crops[i - 1] = crop
            if emb.flip_tta and crop is not None:
                crop2, _, _ = emb.crop(cv2.flip(img, 1))
                if crop2 is not None:
                    flips[i - 1] = crop2
            if i % 200 == 0:
                print(f"    {i}/{len(paths)} cropped ({time.time()-t0:.0f}s)", flush=True)
        np.savez_compressed(crops_file, files=np.array(files),
                            reasons=np.array(reasons),
                            meta=np.array(metas, dtype=object),
                            crops=crops, flips=flips,
                            root=str(root.resolve()))
        print(f"  {root.name}: crops cached in {time.time()-t0:.0f}s", flush=True)

    # batched embedding over every cached view
    flat, owner = [], []
    for i in range(len(files)):
        if crops[i].any():
            flat.append(crops[i]); owner.append(i)
        if emb.flip_tta and flips[i].any():
            flat.append(flips[i]); owner.append(i)
    t1 = time.time()
    print(f"  {root.name}: embedding {len(flat)} crops in batches of "
          f"{emb.batch_size}", flush=True)
    feats = emb.embed_crops(flat)

    acc: dict[int, list] = {}
    for v, i in zip(feats, owner):
        acc.setdefault(i, []).append(v)
    embs = [np.zeros(512, np.float32) for _ in files]
    for i, vs in acc.items():
        v = np.sum(vs, 0)
        embs[i] = v / (np.linalg.norm(v) + 1e-12)
    print(f"  {root.name}: embedded in {time.time()-t1:.0f}s", flush=True)

    np.savez_compressed(emb_file, files=np.array(files), embs=np.array(embs),
                        reasons=np.array(reasons), meta=np.array(metas, dtype=object),
                        root=str(root.resolve()))
    return {"files": files, "embs": embs, "reasons": reasons, "meta": metas}


# ------------------------------------------------------------------- metrics
def roc_points(genuine: np.ndarray, impostor: np.ndarray):
    thrs = np.linspace(-1.0, 1.0, 1001)
    far = 1.0 - np.searchsorted(np.sort(impostor), thrs, side="left") / max(len(impostor), 1)
    tar = 1.0 - np.searchsorted(np.sort(genuine), thrs, side="left") / max(len(genuine), 1)
    return thrs, far, tar


def metrics_at(genuine, impostor, threshold):
    tp = float((genuine >= threshold).sum())
    fn = float((genuine < threshold).sum())
    fp = float((impostor >= threshold).sum())
    tn = float((impostor < threshold).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"threshold": float(threshold), "TP": int(tp), "FP": int(fp),
            "FN": int(fn), "TN": int(tn),
            "precision": prec, "recall": rec, "f1": f1,
            "far": fp / (fp + tn) if fp + tn else 0.0,
            "frr": fn / (fn + tp) if fn + tp else 0.0}


def bootstrap_ci(values, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return [float("nan")] * 3
    boots = rng.choice(values, size=(n, values.size), replace=True).mean(1)
    return [float(values.mean()), float(np.percentile(boots, 2.5)),
            float(np.percentile(boots, 97.5))]


def evaluate(gallery_embs, gallery_ids, probe_embs, probe_ids, probe_meta,
             all_probe_files, threshold=0.4, probe_attrs=None, probe_files=None):
    G = np.asarray(gallery_embs, np.float32)
    G /= np.linalg.norm(G, axis=1, keepdims=True) + 1e-12
    P = np.asarray(probe_embs, np.float32)
    P /= np.linalg.norm(P, axis=1, keepdims=True) + 1e-12
    g_ids = np.asarray(gallery_ids)
    p_ids = np.asarray(probe_ids)

    S = P @ G.T                                   # (n_probe, n_gallery)
    top1 = S.argmax(1)
    correct = (g_ids[top1] == p_ids)
    top1_scores = S[np.arange(len(P)), top1]

    # Extract genuine scores: use MAX similarity for each probe's matching identities
    # (standard practice when multiple gallery samples exist per identity)
    genuine = np.array([S[i, (g_ids == p_ids[i])].max()
                        if (g_ids == p_ids[i]).any() else -1.0 for i in range(len(P))])
    imp_mask = np.ones_like(S, bool)
    for i in range(len(P)):
        imp_mask[i, g_ids == p_ids[i]] = False
    impostor = S[imp_mask]

    # Filter -1.0 sentinel values (probes without matching gallery identity) for statistical analysis
    genuine_valid = genuine[genuine >= 0]
    thrs, far, tar = roc_points(genuine_valid, impostor)
    i1e3 = int(np.argmin(np.abs(far - 1e-3)))
    i1e4 = int(np.argmin(np.abs(far - 1e-4)))
    eer_idx = int(np.argmin(np.abs(far - (1 - tar))))
    auc = float(-np.trapezoid(tar, far))

    rank1_flags = correct.astype(float)
    out = {
        "n_probes_detected": int(len(P)),
        "n_probes_total": int(len(all_probe_files)),
        "fte_rate": float(1 - len(P) / max(len(all_probe_files), 1)),
        "rank1_detected": float(correct.mean()) if len(P) else float("nan"),
        "rank1_ci": bootstrap_ci(rank1_flags),
        "end_to_end_accuracy": float(correct.sum() / max(len(all_probe_files), 1)),
        "rank1_at_threshold": float(((correct) & (top1_scores >= threshold)).mean())
        if len(P) else float("nan"),
        "tar_at_far_1e-3": float(tar[i1e3]), "far_at_1e-3": float(far[i1e3]),
        "tar_at_far_1e-4": float(tar[i1e4]), "far_at_1e-4": float(far[i1e4]),
        "eer": float((far[eer_idx] + 1 - tar[eer_idx]) / 2),
        "auc": auc,
        "genuine_mean": float(genuine_valid.mean()) if len(genuine_valid) > 0 else float("nan"),
        "impostor_mean": float(impostor.mean()),
        "operating_point": metrics_at(genuine_valid, impostor, threshold),
        "threshold_sweep": [metrics_at(genuine_valid, impostor, t)
                            for t in (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6)],
        "score_gap": float(genuine_valid.mean() - impostor.mean()) if len(genuine_valid) > 0 else float("nan"),
    }

    # per-identity breakdown (fairness-style worst-group analysis)
    per_id = {}
    for pid in sorted(set(probe_ids)):
        m = p_ids == pid
        genuine_pid = genuine[m]
        genuine_pid_valid = genuine_pid[genuine_pid >= 0]
        per_id[pid] = {"n_probes": int(m.sum()),
                       "rank1": float(correct[m].mean()),
                       "mean_genuine": float(genuine_pid_valid.mean()) if len(genuine_pid_valid) > 0 else float("nan")}
    out["per_identity"] = per_id
    ids_sorted = sorted(per_id.items(), key=lambda kv: kv[1]["rank1"])
    out["worst_identities"] = [{"id": k, **v} for k, v in ids_sorted[:5]]
    out["identity_rank1_min"] = ids_sorted[0][1]["rank1"]
    out["identity_rank1_median"] = float(np.median([v["rank1"] for v in per_id.values()]))
    out["n_identities_with_perfect_rank1"] = int(sum(v["rank1"] == 1.0
                                                      for v in per_id.values()))

    # per-quality-bucket breakdown
    buckets = {}
    for i, r in enumerate(probe_meta):
        fam = r if r.startswith("ok") else r.split("_")[0]
        buckets.setdefault(fam, []).append(i)
    out["per_quality"] = {
        k: {"n": len(v), "rank1": float(correct[v].mean()) if v else None}
        for k, v in buckets.items()}

    # per-group breakdown on proxy attributes (sex / age bucket from genderage)
    if probe_attrs:
        out["per_group"] = {}
        for name, values in probe_attrs.items():
            d = {}
            for gv in sorted({str(v) for v in values}):
                idx = [i for i, v in enumerate(values) if str(v) == gv]
                if idx:
                    d[gv] = {"n": len(idx), "rank1": float(correct[idx].mean()),
                             "mean_genuine": float(genuine[idx].mean())}
            out["per_group"][name] = d

    # keep the raw arrays out of the JSON; main() writes them to .npz
    out["_arrays"] = {
        "S": S.astype(np.float32),
        "p_ids": np.array(probe_ids, dtype=object),
        "g_ids": g_ids.astype(object),
        "probe_files": np.array(probe_files if probe_files is not None else
                                [""] * len(probe_ids), dtype=object),
        "correct": correct.astype(bool),
        "top1_scores": top1_scores.astype(np.float32),
        "genuine": genuine.astype(np.float32),
        "impostor": impostor.astype(np.float32),
    }
    return out


# ---------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["baseline", "antelope", "onnx"],
                    default="baseline")
    ap.add_argument("--onnx", default=None)
    ap.add_argument("--gallery", default="data/pak_fig/gallery")
    ap.add_argument("--probe", default="data/pak_fig/probe")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--det-size", type=int, default=640)
    ap.add_argument("--det-thresh", type=float, default=0.5)
    ap.add_argument("--min-iod", type=float, default=64)
    ap.add_argument("--pad-retry", type=float, default=0.3,
                    help="reflected border fraction used when SCRFD finds no face")
    ap.add_argument("--flip-tta", action="store_true", help="average embedding with h-flip")
    ap.add_argument("--gallery-agg", choices=["mean", "first", "medoid"], default="mean")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--no-fallback", action="store_true")
    ap.add_argument("--genderage", action="store_true")
    ap.add_argument("--pad", nargs="?", const="models/minifasnet_v2.onnx",
                    default=None, help="enable PAD gate with this ONNX model")
    ap.add_argument("--pad-threshold", type=float, default=0.5)
    ap.add_argument("--output", default=None)
    args = ap.parse_args(argv)

    cfg = dict(DEFAULT_CFG)
    cfg.update(det_size=args.det_size, det_thresh=args.det_thresh,
               min_iod=args.min_iod, allow_fallback=not args.no_fallback,
               pad_retry=args.pad_retry, flip_tta=args.flip_tta,
               pad=os.path.basename(args.pad) if args.pad else None,
               pad_threshold=args.pad_threshold)
    if args.model == "onnx":
        if not args.onnx or not os.path.exists(args.onnx):
            sys.exit(f"--onnx not found: {args.onnx}")
        cfg["model"] = os.path.basename(args.onnx)
        cfg["model_bytes"] = os.path.getsize(args.onnx)
    elif args.model == "antelope":
        cfg["model"] = "antelopev2/glintr100"
    else:
        cfg["model"] = "buffalo_l/w600k_r50"

    emb = Embedder(backend=args.model, onnx_path=args.onnx, det_size=args.det_size,
                   det_thresh=args.det_thresh, min_iod=args.min_iod,
                   allow_fallback=cfg["allow_fallback"],
                   with_genderage=args.genderage, pad_retry=args.pad_retry,
                   flip_tta=args.flip_tta, batch_size=args.batch_size,
                   pad_model=args.pad, pad_threshold=args.pad_threshold)

    cache_dir = Path(args.cache)
    g = embed_folder(emb, Path(args.gallery), cache_dir, cfg)
    p = embed_folder(emb, Path(args.probe), cache_dir, cfg, with_meta=True)

    # gallery = mean embedding per identity (same rule as inference.py)
    gal_embs, gal_ids = [], []
    for i, f in enumerate(g["files"]):
        pid = Path(f).parent.name
        if g["reasons"][i] == "no_face" or np.allclose(g["embs"][i], 0):
            continue
        gal_embs.append(g["embs"][i])
        gal_ids.append(pid)
    # aggregate per identity
    agg_embs, agg_ids = [], []
    for pid in sorted(set(gal_ids)):
        idx = [i for i, x in enumerate(gal_ids) if x == pid]
        vecs = np.stack([gal_embs[i] for i in idx])
        if args.gallery_agg == "first":
            v = vecs[0]
        elif args.gallery_agg == "medoid" and len(vecs) > 1:
            sims = vecs @ vecs.T
            v = vecs[int(np.argmax(sims.mean(1)))]
        else:
            v = vecs.mean(0)
        agg_embs.append(v / (np.linalg.norm(v) + 1e-12))
        agg_ids.append(pid)

    keep = [i for i, e in enumerate(p["embs"]) if not np.allclose(e, 0)]
    probe_embs = [p["embs"][i] for i in keep]
    probe_ids = [Path(p["files"][i]).parent.name for i in keep]
    probe_meta = [p["reasons"][i] for i in keep]
    all_files = p["files"]

    print(f"gallery: {len(agg_ids)} ids | probes: {len(probe_embs)}/{len(all_files)} detected")
    probe_attrs = None
    if args.genderage:
        sex = [p["meta"][i].get("sex") for i in keep]
        age = [p["meta"][i].get("age") for i in keep]
        probe_attrs = {
            "sex": [str(s) if s is not None else "unknown" for s in sex],
            "age_bucket": [
                "u30" if (a is not None and a < 30) else
                "30_50" if (a is not None and a < 50) else
                "50plus" if a is not None else "unknown" for a in age],
        }
    res = evaluate(agg_embs, agg_ids, probe_embs, probe_ids, probe_meta,
                   all_files, threshold=args.threshold, probe_attrs=probe_attrs,
                   probe_files=[p["files"][i] for i in keep])
    res["config"] = cfg
    if args.pad:
        lp = [p["meta"][i].get("live_prob") for i in keep]
        lp = [x for x in lp if x is not None]
        rejected = [p["files"][i] for i in keep
                    if p["meta"][i].get("live_prob") is not None
                    and p["meta"][i]["live_prob"] < args.pad_threshold]
        res["pad"] = {
            "model": os.path.basename(args.pad),
            "threshold": args.pad_threshold,
            "n_scored": len(lp),
            "bpcer": float(np.mean(np.array(lp) < args.pad_threshold)) if lp else None,
            "live_prob_mean": float(np.mean(lp)) if lp else None,
            "live_prob_p05": float(np.percentile(lp, 5)) if lp else None,
            "rejected_files": rejected,
        }

    out = args.output or f"eval_{args.tag or args.model}.json"
    arrays = res.pop("_arrays", None)
    if arrays is not None:
        np.savez_compressed(str(Path(out).with_suffix(".scores.npz")), **arrays)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=2, default=str)

    print("\n" + "=" * 62)
    print(f"RESULT [{args.tag or args.model}]")
    for k in ("n_probes_detected", "n_probes_total", "fte_rate", "rank1_detected",
              "end_to_end_accuracy", "tar_at_far_1e-3", "tar_at_far_1e-4", "eer",
              "auc", "genuine_mean", "impostor_mean", "score_gap"):
        print(f"  {k:24s} {res[k]:.4f}")
    op = res["operating_point"]
    print(f"  precision/recall/F1 @{args.threshold}: "
          f"{op['precision']:.4f} / {op['recall']:.4f} / {op['f1']:.4f}")
    if args.pad:
        pd = res["pad"]
        print(f"  PAD BPCER @{args.pad_threshold}: {pd['bpcer']:.4f} "
              f"(n={pd['n_scored']}, mean live prob {pd['live_prob_mean']:.3f})")
    print(f"  worst identity: {res['worst_identities'][0]}")
    print(f"  per-quality: {res['per_quality']}")
    print(f"saved -> {out}")
    return res


if __name__ == "__main__":
    main()

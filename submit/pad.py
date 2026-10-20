"""
Presentation-attack detection (PAD / anti-spoofing) gate.

Models: MiniFASNet-V2 (2.7x) + MiniFASNet-V1SE (4.0x) from
minivision-ai/Silent-Face-Anti-Spoofing (Apache-2.0), exported to ONNX by
`convert_pad.py`. The two are averaged, as upstream does.

Input : 80x80 BGR crop (2.7x / 4.0x margin around the face), **raw 0..255**
Output: 3-class softmax, ordered [print-attack, live, replay-attack]
        live score = p[live]

Preprocessing caveat: upstream's reference code applies ToTensor()/255, but on
our data that saturates the network to a constant output for every input. Raw
0..255 reproduces clean, sensible behaviour (live faces -> p[live] ~ 0.99,
print attacks -> p[print] ~ 0.95). The class order was determined empirically
on labelled data, not taken from a model card.

CLI:
    python pad.py --data data/pak_fig/probe                 # BPCER on real faces
    python pad.py --data <real_dir> --fake <attack_dir>     # + APCER
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
DEFAULT_MODELS = ["models/minifasnet_v2.onnx", "models/minifasnet_v1se.onnx"]


def _new_box(src_w: int, src_h: int, bbox, scale: float):
    """Margin box around the face, clipped to the image (upstream logic)."""
    x1, y1, x2, y2 = [float(v) for v in bbox]
    w, h = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
    scale = min((src_h - 1) / h, (src_w - 1) / w, scale)
    nw, nh = w * scale, h * scale
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return (max(0.0, cx - nw / 2), max(0.0, cy - nh / 2),
            min(src_w - 1.0, cx + nw / 2), min(src_h - 1.0, cy + nh / 2))


class SpoofDetector:
    def __init__(self, model_paths=None, threshold: float = 0.5,
                 batch_size: int = 32):
        import onnxruntime as ort
        paths = model_paths or DEFAULT_MODELS
        if isinstance(paths, str):
            paths = [paths]
        self.paths = paths
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1          # tiny model; avoids thread thrash
        self.sessions = []
        for p in paths:
            s = ort.InferenceSession(p, so, providers=["CPUExecutionProvider"])
            self.sessions.append((s, s.get_inputs()[0].name))
        self.threshold = threshold
        self.batch_size = batch_size
        # each model sees a different crop margin, matching upstream
        self.scales = [2.7, 4.0][:len(paths)]

    def crop(self, img: np.ndarray, bbox, scale: float = 2.7) -> np.ndarray:
        h, w = img.shape[:2]
        x1, y1, x2, y2 = _new_box(w, h, bbox, scale)
        x1, y1 = int(x1), int(y1)
        x2, y2 = int(np.ceil(x2)), int(np.ceil(y2))
        patch = img[y1:y2, x1:x2]
        if patch.size == 0:
            patch = img
        return cv2.resize(patch, (80, 80))

    def predict_crops(self, crops: list[np.ndarray]) -> np.ndarray:
        """Returns (n, 3) probabilities, one row per crop."""
        if not crops:
            return np.zeros((0, 3), np.float32)
        out = np.zeros((len(crops), 3), np.float32)
        for i in range(0, len(crops), self.batch_size):
            chunk = crops[i:i + self.batch_size]
            # raw 0..255, BGR, NCHW -- see module docstring
            x = np.stack(chunk).astype(np.float32).transpose(0, 3, 1, 2)
            probs = []
            for sess, input_name in self.sessions:
                logits = np.asarray(sess.run(None, {input_name: x})[0], np.float32)
                e = np.exp(logits - logits.max(1, keepdims=True))
                probs.append(e / (e.sum(1, keepdims=True) + 1e-12))
            out[i:i + len(chunk)] = np.mean(probs, axis=0)
        return out

    def predict(self, img: np.ndarray, bbox) -> dict:
        crops = [self.crop(img, bbox, s) for s in self.scales]
        probs = [self.predict_crops([c])[0] for c in crops]
        p = np.mean(probs, axis=0)
        return {"live_prob": float(p[1]), "p_print": float(p[0]),
                "p_live": float(p[1]), "p_replay": float(p[2]),
                "is_live": bool(p[1] >= self.threshold)}


def _iter_images(root: Path):
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS)


def _score_dir(app, det, root: Path, limit: int | None = None,
               skip: int = 0):
    rows, crops = [], []
    paths = _iter_images(root)[skip:]
    if limit:
        paths = paths[:limit]
    for p in paths:
        row = {"file": str(p.relative_to(root)), "live_prob": None}
        rows.append(row)
        img = cv2.imread(str(p))
        if img is None:
            continue
        faces = app.get(img)
        off = 0
        if not faces:
            b = int(round(max(img.shape[:2]) * 0.3))
            img = cv2.copyMakeBorder(img, b, b, b, b, cv2.BORDER_REFLECT_101)
            faces = app.get(img)
            off = b
        if not faces:
            continue
        f = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        row["det_score"] = float(f.det_score)
        row["bbox"] = [float(v) for v in (f.bbox - off)]
        # average the per-scale crops for this image
        cs = [det.crop(img, f.bbox, s) for s in det.scales]
        crops.append(cs)
        row["_crop_idx"] = len(crops) - 1

    flat = [c for cs in crops for c in cs]
    probs = det.predict_crops(flat)
    n = len(det.scales)
    for r in rows:
        j = r.pop("_crop_idx", None)
        if j is None:
            continue
        p = probs[j * n:(j + 1) * n].mean(0)
        r["live_prob"] = float(p[1])
        r["p_print"] = float(p[0])
        r["p_live"] = float(p[1])
        r["p_replay"] = float(p[2])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pak_fig/probe", help="real images")
    ap.add_argument("--fake", default=None, help="attack images (for APCER)")
    ap.add_argument("--model", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--output", default="data/pad_bpcer.json")
    args = ap.parse_args()

    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    det = SpoofDetector(args.model, threshold=args.threshold)

    real = _score_dir(app, det, Path(args.data), args.limit, args.skip)
    res = {"models": [Path(m).name for m in args.model],
           "threshold": args.threshold,
           "n_real": len(real),
           "n_real_scored": sum(r["live_prob"] is not None for r in real)}
    lp = [r["live_prob"] for r in real if r["live_prob"] is not None]
    if lp:
        res.update(bpcer=float(np.mean(np.array(lp) < args.threshold)),
                   live_mean=float(np.mean(lp)),
                   live_p05=float(np.percentile(lp, 5)),
                   false_rejects=[r["file"] for r in real
                                  if r["live_prob"] is not None
                                  and r["live_prob"] < args.threshold])
    if args.fake:
        fake = _score_dir(app, det, Path(args.fake), args.limit, args.skip)
        fp = [r["live_prob"] for r in fake if r["live_prob"] is not None]
        res.update(n_attack=len(fake), n_attack_scored=len(fp),
                   apcer=float(np.mean(np.array(fp) >= args.threshold)) if fp else None,
                   attack_live_mean=float(np.mean(fp)) if fp else None)
        res["attack_rows"] = fake
    res["real_rows"] = real

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"real: {res['n_real_scored']}/{res['n_real']} scored | "
          f"BPCER @ {args.threshold:.2f} = {res.get('bpcer', float('nan')):.4f} "
          f"(mean live {res.get('live_mean', float('nan')):.3f})")
    if args.fake:
        print(f"attack: {res['n_attack_scored']}/{res['n_attack']} scored | "
              f"APCER @ {args.threshold:.2f} = {res['apcer']:.4f} "
              f"(mean live {res['attack_live_mean']:.3f})")
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()

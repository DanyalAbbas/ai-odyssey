"""Veiled-face robustness: will a niqab / hijab wearer even enroll?

Recognition accuracy is the wrong first question for heavily covered faces. The
failure we actually hit on the Pakistani benchmark was *failure to enroll*: if
SCRFD does not find a face, no downstream metric matters. This script measures
the enrollment funnel on external veiled-face data:

    images -> detected -> quality gate passed -> liveness passed -> template

It also scores detection against YOLO ground-truth boxes where the dataset
provides them, and it runs the funnel twice (with and without the tight-crop
reflected-border retry) to show how much of the gap the retry closes.

    python veiled_eval.py --images data/raw/niqab/niqabi_images_annotated/images \
        --labels data/raw/niqab/niqabi_images_annotated/labels --name niqab_kaggle

    python veiled_eval.py --images data/raw/niqab_ofd/images --labels ... \
        --name niqab_ofd --limit 400
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from imgio import imread_any

from evaluate_pak import Embedder

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def yolo_to_xyxy(line: str, w: int, h: int):
    parts = line.split()
    if len(parts) < 5:
        return None
    cx, cy, bw, bh = (float(v) for v in parts[-4:])
    return (max(0.0, (cx - bw / 2) * w), max(0.0, (cy - bh / 2) * h),
            min(float(w), (cx + bw / 2) * w), min(float(h), (cy + bh / 2) * h))


def voc_to_xyxy(path: Path):
    """Pascal VOC box; returns every <object> box in the file."""
    import xml.etree.ElementTree as ET
    try:
        root = ET.parse(path).getroot()
    except Exception:
        return []
    boxes = []
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        if bb is None:
            continue
        try:
            boxes.append((float(bb.findtext("xmin")), float(bb.findtext("ymin")),
                          float(bb.findtext("xmax")), float(bb.findtext("ymax"))))
        except (TypeError, ValueError):
            continue
    return boxes


def iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


def run(emb: Embedder, paths, label_dir, pad_retry, det_thresh, iou_thr=0.4,
        label_format="yolo"):
    emb.pad_retry = pad_retry
    n_img = n_gt = n_gt_hit = n_gt_center = 0
    size_bk = {"<32": [0, 0], "32-64": [0, 0], "64-128": [0, 0], ">=128": [0, 0]}
    n_gt_big = n_gt_big_hit = 0
    det = enrol = live = 0
    pads = []
    iods, scores = [], []
    for p in paths:
        img = imread_any(p)
        if img is None:
            continue
        n_img += 1
        h, w = img.shape[:2]
        frame, faces, pad = emb._detect(img)
        faces = [f for f in faces if f.det_score >= det_thresh]
        if faces:
            det += 1
            pads.append(pad)
        if label_dir is not None:
            if label_format == "voc":
                lp = Path(label_dir) / (Path(p).stem + ".xml")
                gts = voc_to_xyxy(lp) if lp.exists() else []
            else:
                lp = Path(label_dir) / (Path(p).stem + ".txt")
                gts = ([b for b in (yolo_to_xyxy(l, w, h)
                                    for l in lp.read_text().splitlines() if l.strip())
                        if b] if lp.exists() else [])
            if gts:
                n_gt += len(gts)
                for g in gts:
                    hit = bool(faces) and max(iou(g, tuple(f.bbox)) for f in faces) >= iou_thr
                    gx, gy = (g[0] + g[2]) / 2, (g[1] + g[3]) / 2
                    # Some datasets annotate the eye region rather than the whole
                    # face; a centre-in-box test is robust to that convention.
                    centre = bool(faces) and any(
                        f.bbox[0] <= gx <= f.bbox[2] and f.bbox[1] <= gy <= f.bbox[3]
                        for f in faces)
                    n_gt_hit += int(hit)
                    n_gt_center += int(centre)
                    side = max(g[2] - g[0], g[3] - g[1])
                    k = ("<32" if side < 32 else "32-64" if side < 64
                         else "64-128" if side < 128 else ">=128")
                    size_bk[k][0] += 1
                    size_bk[k][1] += int(centre)
                    # a face has to be big enough for a 112x112 template to mean
                    # anything; below ~100 px the crop is mostly interpolation
                    if (g[2] - g[0]) >= 100 and (g[3] - g[1]) >= 100:
                        n_gt_big += 1
                        n_gt_big_hit += int(hit)
        if faces:
            f = max(faces, key=lambda z: (z.bbox[2] - z.bbox[0]) * (z.bbox[3] - z.bbox[1]))
            ok, _why = emb.quality(f, frame, pad)
            enrol += int(ok)
            iods.append(float(np.linalg.norm(f.kps[0] - f.kps[1])))
            scores.append(float(f.det_score))
            if emb.spoof is not None:
                live += int(emb.spoof.predict(frame, f.bbox)["live_prob"] >=
                            emb.spoof.threshold)
    return {
        "n_images": n_img,
        "detected": det,
        "detection_rate": det / n_img if n_img else float("nan"),
        "enrolled": enrol,
        "enroll_rate": enrol / n_img if n_img else float("nan"),
        "live": live if emb.spoof is not None else None,
        "live_rate": (live / det) if (emb.spoof is not None and det) else None,
        "gt_boxes": n_gt,
        "gt_matched": n_gt_hit,
        "gt_recall": (n_gt_hit / n_gt) if n_gt else None,
        "gt_center_rate": (n_gt_center / n_gt) if n_gt else None,
        "gt_center_by_size": {k: {"n": v[0],
                                   "rate": (v[1] / v[0]) if v[0] else None}
                               for k, v in size_bk.items()},
        "gt_boxes_large": n_gt_big,
        "gt_matched_large": n_gt_big_hit,
        "gt_recall_large": (n_gt_big_hit / n_gt_big) if n_gt_big else None,
        "pad_used_rate": (sum(1 for x in pads if x) / len(pads)) if pads else 0.0,
        "iod_median": float(np.median(iods)) if iods else None,
        "iod_p10": float(np.percentile(iods, 10)) if iods else None,
        "iod_p90": float(np.percentile(iods, 90)) if iods else None,
        "frac_iod_ge_64": float(np.mean(np.array(iods) >= 64)) if iods else None,
        "det_score_median": float(np.median(scores)) if scores else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--label-format", choices=["yolo", "voc"], default="yolo")
    ap.add_argument("--name", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--det-thresh", type=float, default=0.5)
    ap.add_argument("--pad-model", default="models/minifasnet_v2.onnx")
    ap.add_argument("--pad-threshold", type=float, default=0.02)
    ap.add_argument("--conditions", choices=["both", "retry", "no_retry"],
                    default="both")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    paths = sorted(p for p in Path(args.images).rglob("*") if p.suffix.lower() in IMG_EXTS)
    if args.limit:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(paths), size=min(args.limit, len(paths)), replace=False)
        paths = [paths[i] for i in sorted(idx)]
    print(f"{args.name}: {len(paths)} images", flush=True)

    emb = Embedder(backend="baseline", pad_model=args.pad_model,
                   pad_threshold=args.pad_threshold, det_thresh=args.det_thresh)

    out = {"name": args.name, "conditions": {}}
    conds = [("no_retry", 0.0), ("reflected_retry", 0.3)]
    if args.conditions == "retry":
        conds = conds[1:]
    elif args.conditions == "no_retry":
        conds = conds[:1]
    for label, pr in conds:
        r = run(emb, paths, args.labels, pr, args.det_thresh,
                label_format=args.label_format)
        out["conditions"][label] = r
        print(f"  {label:16s} det={r['detection_rate']:.3f} "
              f"enrol={r['enroll_rate']:.3f} live={(r['live_rate'] or 0):.3f} "
              f"gt_recall={(r['gt_recall'] or 0):.3f} "
              f"gt_recall_large={(r['gt_recall_large'] or 0):.3f} "
              f"gt_center={(r['gt_center_rate'] or 0):.3f} "
              f"pad_used={r['pad_used_rate']:.3f}", flush=True)

    dst = args.output or f"veiled_{args.name}.json"
    with open(dst, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"saved -> {dst}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Clean and align the scraped corpus into a training set.

Web image search returns noise: wrong people, group shots, memes, logos. Training
an identity classifier on that noise is how a fine-tune ends up worse than the
pretrained model, so every image goes through four filters:

1. exactly one detected face at det_score >= 0.6 (group photos are dropped rather
   than guessed),
2. box at least 64x64 px,
3. cosine >= 0.30 to the identity's running mean, pruned twice (drops the wrong
   person that slipped through search),
4. at least 10 survivors, or the identity is dropped entirely.

Output is 112x112 ArcFace-aligned crops under `data/bench_train/<identity>/` plus
a manifest carrying the gender / veiled curation flags. Occluders are painted on
at training time, not stored here — that keeps one copy of each face and lets the
mask augmentation be randomised per epoch.

    python src/clean_train_faces.py --min-keep 10
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "submit"))

import cv2  # noqa: E402
from insightface.utils import face_align  # noqa: E402

from imgio import imread_any  # noqa: E402
from inference import FacePipeline  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "data" / "raw" / "train_faces"))
    ap.add_argument("--out", default=str(ROOT / "data" / "bench_train"))
    ap.add_argument("--min-keep", type=int, default=10)
    ap.add_argument("--min-score", type=float, default=0.6)
    ap.add_argument("--min-box", type=int, default=64)
    ap.add_argument("--cos", type=float, default=0.30)
    ap.add_argument("--limit", type=int, default=0,
                    help="use at most N images per identity (0 = all); the "
                         "corpus carries ~45 per identity, more than needed")
    ap.add_argument("--det-size", type=int, default=640,
                    help="detector input size; 320 is enough for portrait-sized "
                         "faces and ~4x faster on a small corpus")
    args = ap.parse_args()

    raw, out = Path(args.raw), Path(args.out)
    meta = {}
    src_csv = raw / "sources.csv"
    if src_csv.exists():
        with open(src_csv, newline="") as fh:
            for row in csv.DictReader(fh):
                meta[row["identity"]] = row

    pipe = FacePipeline(models_root=str(ROOT / "submit"), pad_models=None,
                        det_size=(args.det_size, args.det_size))
    out.mkdir(parents=True, exist_ok=True)

    report, kept_total = [], 0
    for id_dir in sorted(p for p in raw.iterdir() if p.is_dir()):
        pid = id_dir.name
        files = sorted(p for p in id_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
        if args.limit:
            files = files[:args.limit]
        rows = []
        for p in files:
            img = imread_any(p)
            if img is None:
                continue
            faces = [r for r in pipe.get_faces(img, quality_filter=False, gate_pad=False)
                     if r[0].det_score >= args.min_score]
            if len(faces) != 1:
                continue
            f = faces[0][0]
            bw, bh = f.bbox[2] - f.bbox[0], f.bbox[3] - f.bbox[1]
            if min(bw, bh) < args.min_box:
                continue
            # get_faces already embedded this face; recomputing would double
            # the recognition cost across the whole corpus
            emb = faces[0][1]
            if emb is None:
                emb = pipe._embed(img, f)
            rows.append((p, img, f, unit(emb)))

        # two rounds of outlier pruning against the identity mean
        for _ in range(2):
            if len(rows) < 3:
                break
            mean = unit(np.mean([r[3] for r in rows], axis=0))
            rows = [r for r in rows if float(np.dot(unit(r[3]), mean)) >= args.cos]

        if len(rows) < args.min_keep:
            report.append({"identity": pid, "raw": len(files), "kept": 0,
                           "status": f"dropped (<{args.min_keep})"})
            print(f"  {pid:<22s} raw {len(files):3d} kept {len(rows):3d}  DROPPED", flush=True)
            continue

        dest = out / pid
        dest.mkdir(parents=True, exist_ok=True)
        for i, (_p, img, f, _e) in enumerate(rows):
            crop = face_align.norm_crop(img, landmark=f.kps, image_size=112)
            cv2.imwrite(str(dest / f"{i:04d}.jpg"), crop,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
        kept_total += len(rows)
        m = meta.get(pid, {})
        report.append({"identity": pid, "raw": len(files), "kept": len(rows),
                       "status": "ok", "gender": m.get("gender", ""),
                       "veiled": m.get("veiled", ""), "query": m.get("query", "")})
        print(f"  {pid:<22s} raw {len(files):3d} kept {len(rows):3d}", flush=True)

    kept = [r for r in report if r["status"] == "ok"]
    with open(out / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["identity", "raw", "kept", "status",
                                           "gender", "veiled", "query"])
        w.writeheader()
        w.writerows(report)

    n_f = sum(1 for r in kept if r.get("gender") == "f")
    n_v = sum(1 for r in kept if r.get("veiled") == "1")
    print(f"\n{len(kept)} identities / {kept_total} aligned crops -> {out}")
    print(f"  women: {n_f}   flagged veiled (hijab/dupatta): {n_v}")
    print(f"  dropped: {len(report) - len(kept)}")


if __name__ == "__main__":
    main()

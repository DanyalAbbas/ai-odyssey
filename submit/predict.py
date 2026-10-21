#!/usr/bin/env python3
"""Submission entry point for the TechFest hidden benchmark.

The organisers run this; we never see their data. So the contract is deliberately
narrow and the failure modes are loud:

    # 1:N identification (gallery/<identity>/<image>, probe/<anything>)
    python predict.py --gallery gallery/ --probe probe/ --output predictions.csv

    # 1:1 verification (CSV with two image-path columns)
    python predict.py --pairs pairs.csv --output predictions.csv

    # gallery as a flat folder + index file
    python predict.py --gallery imgs/ --gallery-index index.csv --probe probe/

Everything runs offline from the bundled model directory. Nothing is downloaded
at runtime. Unreadable images still produce a row (predicted_id=unknown) so the
output always has one line per input.

Output columns: probe,predicted_id,score,quality,match
Pairs output:  pair,left,right,score,match
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

# Threads must be set before onnxruntime is imported.
for _i, _a in enumerate(sys.argv):
    if _a == "--threads" and _i + 1 < len(sys.argv):
        os.environ.setdefault("OMP_NUM_THREADS", sys.argv[_i + 1])
    elif _a.startswith("--threads="):
        os.environ.setdefault("OMP_NUM_THREADS", _a.split("=", 1)[1])

import numpy as np  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

HERE = Path(__file__).resolve().parent


def list_images(root: Path):
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS)


def read_pairs(path: Path):
    """Accept any CSV whose first two columns are image paths.

    Tolerates a header row and extra columns (label, fold, ...).
    """
    rows, out = [], []
    with open(path, newline="") as fh:
        for i, row in enumerate(csv.reader(fh)):
            if not row or len(row) < 2:
                continue
            if i == 0 and not Path(row[0]).suffix.lower() in IMG_EXTS:
                continue  # header
            out.append((row[0], row[1]))
    return out


def gallery_from_dirs(root: Path):
    """Return {image_path: identity} for the standard nested layout."""
    if not root.is_dir():
        sys.exit(f"Gallery root not found or not a directory: {root}")
    mapping = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        for img in list_images(d):
            mapping[img] = d.name
    return mapping


def gallery_from_index(index: Path, root: Path):
    """Flat gallery + index CSV: columns `file,id` in either order."""
    mapping = {}
    with open(index, newline="") as fh:
        for i, row in enumerate(csv.reader(fh)):
            if len(row) < 2:
                continue
            if i == 0 and not Path(row[0]).suffix.lower() in IMG_EXTS:
                continue
            a, b = row[0].strip(), row[1].strip()
            if Path(a).suffix.lower() in IMG_EXTS:
                img, pid = a, b
            else:
                pid, img = a, b
            p = root / img
            if p.exists():
                mapping[p] = pid
    return mapping


def key_for(path: Path, root: Path, flat: bool) -> str:
    """Return relative path as string with forward slashes for cross-platform compatibility."""
    if flat:
        return path.name
    # Normalize path separators to forward slashes for CSV portability across platforms
    return str(path.relative_to(root)).replace("\\", "/")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gallery", help="gallery root (nested <id>/<img>)")
    ap.add_argument("--gallery-index", help="CSV file,id for a flat gallery")
    ap.add_argument("--probe", help="probe image, directory, or nested directory")
    ap.add_argument("--pairs", help="CSV of image pairs for 1:1 verification")
    ap.add_argument("--output", default="predictions.csv")
    ap.add_argument("--model", default="buffalo_l",
                    choices=["buffalo_l", "antelopev2", "onnx"])
    ap.add_argument("--onnx", help="path to a 512-d ONNX embedder when --model onnx")
    ap.add_argument("--models-dir", default=str(HERE),
                    help="root that CONTAINS models/buffalo_l (offline)")
    ap.add_argument("--threshold", type=float, default=0.40)
    ap.add_argument("--pad", action="store_true",
                    help="enable the liveness gate (slower, rejects some genuine faces)")
    ap.add_argument("--pad-threshold", type=float, default=0.02)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="process at most N inputs")
    args = ap.parse_args()

    t0 = time.time()
    from inference import FacePipeline

    kw = {}
    if args.model in {"buffalo_l", "antelopev2"}:
        kw["models_root"] = args.models_dir
    pipe = FacePipeline(model_name=args.model, onnx_path=args.onnx,
                        pad_models="default" if args.pad else None,
                        pad_threshold=args.pad_threshold, **kw)

    if args.pairs:
        return run_pairs(pipe, args, t0)
    if not (args.gallery and args.probe):
        ap.error("provide either --pairs, or --gallery and --probe")
    return run_identification(pipe, args, t0)


def run_identification(pipe, args, t0):
    gal_root = Path(args.gallery)
    if args.gallery_index:
        mapping = gallery_from_index(Path(args.gallery_index), gal_root)
    else:
        mapping = gallery_from_dirs(gal_root)
        if not mapping:
            sys.exit(f"no identity subdirectories under {gal_root}; "
                     f"pass --gallery-index for a flat gallery")

    # enrol: liveness OFF — an ID scan or printed portrait is a valid source
    per_id = {}
    for img, pid in sorted(mapping.items()):
        emb, reason = pipe.embed_image(img, quality_filter=True, gate_pad=False)
        if emb is not None:
            per_id.setdefault(pid, []).append(emb)
    gallery_ids, gallery_embs = [], []
    for pid in sorted(per_id):
        v = np.mean(np.stack(per_id[pid]), 0)
        v = v / (np.linalg.norm(v) + 1e-12)
        gallery_ids.append(pid)
        gallery_embs.append(v)
    print(f"gallery: {len(gallery_ids)} identities, "
          f"{sum(len(v) for v in per_id.values())} enrolled images", flush=True)
    if not gallery_ids:
        sys.exit("gallery produced zero identities")

    probe_root = Path(args.probe)
    flat = probe_root.is_file()
    probes = [probe_root] if flat else list_images(probe_root)
    if args.limit:
        probes = probes[: args.limit]
    print(f"probes: {len(probes)}", flush=True)

    rows = []
    for i, p in enumerate(probes, 1):
        emb, reason = pipe.embed_image(p, quality_filter=True, gate_pad=args.pad)
        best_id, score, is_match = pipe.match(emb, gallery_embs, gallery_ids, args.threshold)
        rows.append({"probe": key_for(p, probe_root, flat),
                     "predicted_id": best_id if is_match else "unknown",
                     "score": round(float(score), 4),
                     "quality": reason,
                     "match": bool(is_match)})
        if i % 50 == 0:
            print(f"  {i}/{len(probes)}  ({(time.time()-t0)/i:.2f}s/img)", flush=True)

    with open(args.output, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["probe", "predicted_id", "score", "quality", "match"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {args.output}  "
          f"({time.time()-t0:.0f}s, {(time.time()-t0)/max(1,len(rows)):.2f}s/img)")


def run_pairs(pipe, args, t0):
    pairs = read_pairs(Path(args.pairs))
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"pairs: {len(pairs)}", flush=True)
    cache = {}

    def emb_of(path):
        if path not in cache:
            cache[path] = pipe.embed_image(path, quality_filter=True, gate_pad=args.pad)[0]
        return cache[path]

    rows = []
    for i, (a, b) in enumerate(pairs, 1):
        ea, eb = emb_of(a), emb_of(b)
        if ea is None or eb is None:
            score, match = 0.0, False
        else:
            score = float(np.dot(ea, eb))
            match = score >= args.threshold
        rows.append({"pair": i, "left": a, "right": b,
                     "score": round(score, 4), "match": bool(match)})
        if i % 50 == 0:
            print(f"  {i}/{len(pairs)}", flush=True)

    with open(args.output, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["pair", "left", "right", "score", "match"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {args.output}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()

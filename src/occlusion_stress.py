"""Controlled occlusion stress test: hijab / niqab proxies.

Real data cannot answer the niqab question: our probe set contains headscarves
with the face fully visible, and there is no public Pakistani identity-labelled
niqab dataset. So we hold the *identity* fixed and vary only the occlusion,
which isolates the effect of covering from every other confound.

Procedure (probe-side occlusion only, which is the deployment case: clean
enrolment photo, covered face at the checkpoint):

  1. take the aligned 112x112 crops already cached by evaluate_pak.py
  2. paint over a geometric region with a cloth-like fill
  3. re-embed probes, keep the gallery clean
  4. score exactly as the main benchmark does

Occlusions are defined on the ArcFace alignment template, where the eyes sit at
fixed rows, so the regions are anatomically meaningful rather than arbitrary.

    python occlusion_stress.py --model baseline
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import cv2
import numpy as np

from evaluate_pak import Embedder, evaluate


def latest(pattern: str) -> str:
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise SystemExit(f"no cache matching {pattern}")
    return hits[-1]


def paint(img: np.ndarray, x0, y0, x1, y1, color=(70, 70, 75)):
    """Fill a rectangle with a slightly noisy cloth colour (not pure black,
    so the embedder is not reacting to a degenerate constant patch)."""
    h, w = img.shape[:2]
    x0, x1 = max(0, int(x0 * w)), min(w, int(x1 * w))
    y0, y1 = max(0, int(y0 * h)), min(h, int(y1 * h))
    if x1 <= x0 or y1 <= y0:
        return img
    patch = img[y0:y1, x0:x1].astype(np.int16)
    noise = np.random.default_rng(0).integers(-6, 7, patch.shape, dtype=np.int16)
    img[y0:y1, x0:x1] = np.clip(patch * 0 + np.array(color) + noise, 0, 255).astype(np.uint8)
    return img


CONDITIONS = {
    # name: list of (x0, y0, x1, y1) in relative coordinates
    "clean": [],
    # headscarf framing: hairline + sides, face untouched
    "hijab": [(0.00, 0.00, 1.00, 0.16), (0.00, 0.00, 0.10, 0.80),
              (0.90, 0.00, 1.00, 0.80)],
    # niqab, light: nose tip + mouth + chin covered, eyes clear
    "niqab_light": [(0.10, 0.52, 0.90, 1.00)],
    # niqab, full: everything below the eyes plus cheek sides
    "niqab_full": [(0.06, 0.45, 0.94, 1.00), (0.00, 0.30, 0.08, 1.00),
                   (0.92, 0.30, 1.00, 1.00)],
    # extreme: lower 3/4, only the eyes remain
    "niqab_extreme": [(0.00, 0.40, 1.00, 1.00)],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="baseline", choices=["baseline", "antelope", "onnx"])
    ap.add_argument("--onnx", default=None)
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--gallery-agg", default="mean")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    gz = np.load(latest(f"{args.cache}/gallery_crops_*.npz"), allow_pickle=True)
    pz = np.load(latest(f"{args.cache}/probe_crops_*.npz"), allow_pickle=True)
    g_files = [str(f) for f in gz["files"]]
    p_files = [str(f) for f in pz["files"]]
    g_crops, p_crops = gz["crops"], pz["crops"]

    emb = Embedder(backend=args.model, onnx_path=args.onnx, pad_model=None,
                   batch_size=32)

    def embed(arr):
        idx = [i for i in range(len(arr)) if arr[i].any()]
        feats = emb.embed_crops([arr[i] for i in idx])
        out = np.zeros((len(arr), 512), np.float32)
        out[idx] = feats
        return out

    G = embed(g_crops)

    # gallery: mean per identity
    gal_ids_all = [Path(f).parent.name for f in g_files]
    agg_embs, agg_ids = [], []
    for pid in sorted(set(gal_ids_all)):
        vs = np.stack([G[i] for i, x in enumerate(gal_ids_all) if x == pid and G[i].any()])
        if not len(vs):
            continue
        v = vs.mean(0)
        agg_embs.append(v / (np.linalg.norm(v) + 1e-12))
        agg_ids.append(pid)

    results = {}
    for name, rects in CONDITIONS.items():
        arr = p_crops.copy()
        if rects:
            for i in range(len(arr)):
                if not arr[i].any():
                    continue
                for r in rects:
                    arr[i] = paint(arr[i], *r)
        P = embed(arr)
        keep = [i for i in range(len(P)) if P[i].any()]
        p_ids = [Path(p_files[i]).parent.name for i in keep]
        res = evaluate(agg_embs, agg_ids, [P[i] for i in keep], p_ids,
                       [str(pz["reasons"][i]) for i in keep], p_files,
                       probe_files=[p_files[i] for i in keep])
        results[name] = {k: res[k] for k in
                         ("n_probes_detected", "rank1_detected", "rank1_ci",
                          "end_to_end_accuracy", "tar_at_far_1e-3", "eer",
                          "auc", "genuine_mean", "impostor_mean")}
        r = results[name]
        print(f"{name:15s} rank1={r['rank1_detected']:.4f} "
              f"CI[{r['rank1_ci'][1]:.3f},{r['rank1_ci'][2]:.3f}] "
              f"TAR@1e-3={r['tar_at_far_1e-3']:.4f} EER={r['eer']:.4f} "
              f"genuine={r['genuine_mean']:.3f}")

    out = args.output or f"occlusion_{args.model}.json"
    with open(out, "w") as fh:
        json.dump({"model": args.model, "conditions": results}, fh, indent=2)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()

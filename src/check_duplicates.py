"""
Near-duplicate check between gallery and probe splits.

A random image-level split can put two copies of the same source photo (different
crops/JPEG re-encodes) on both sides, which inflates Rank-1. This flags every
probe whose cosine similarity to a *gallery image of the same identity* exceeds
a threshold, and reports accuracy with those probes excluded.

    python check_duplicates.py                      # newest cache pair
    python check_duplicates.py --thr 0.985
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def newest(pattern: str) -> Path:
    files = sorted(Path("data/cache").glob(pattern), key=lambda p: p.stat().st_mtime)
    if not files:
        raise SystemExit(f"no cache files matching {pattern}; run evaluate_pak.py first")
    return files[-1]


def load(p: Path):
    z = np.load(p, allow_pickle=True)
    return list(z["files"]), np.asarray(z["embs"], np.float32), list(z["reasons"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery", default=None)
    ap.add_argument("--probe", default=None)
    ap.add_argument("--thr", type=float, default=0.985)
    ap.add_argument("--output", default="duplicate_audit.json")
    args = ap.parse_args()

    g_path = Path(args.gallery) if args.gallery else newest("gallery_*.npz")
    p_path = Path(args.probe) if args.probe else newest("probe_*.npz")
    print(f"gallery cache: {g_path.name}\nprobe cache  : {p_path.name}")

    g_files, G, g_reasons = load(g_path)
    p_files, P, p_reasons = load(p_path)

    keep_g = [i for i, r in enumerate(g_reasons) if not np.allclose(G[i], 0)]
    keep_p = [i for i, r in enumerate(p_reasons) if not np.allclose(P[i], 0)]
    G = G[keep_g]; g_ids = [Path(g_files[i]).parent.name for i in keep_g]
    P = P[keep_p]; p_ids = [Path(p_files[i]).parent.name for i in keep_p]

    G /= np.linalg.norm(G, axis=1, keepdims=True) + 1e-12
    P /= np.linalg.norm(P, axis=1, keepdims=True) + 1e-12

    S = P @ G.T
    g_ids_arr = np.array(g_ids)

    flagged, worst = [], []
    for i in range(len(P)):
        same = np.where(g_ids_arr == p_ids[i])[0]
        if len(same) == 0:
            continue
        s = S[i, same].max()
        worst.append((p_files[keep_p[i]], float(s)))
        if s >= args.thr:
            j = same[int(np.argmax(S[i, same]))]
            flagged.append({"probe": p_files[keep_p[i]],
                            "gallery": g_files[keep_g[j]],
                            "identity": p_ids[i],
                            "cosine": float(s)})

    worst.sort(key=lambda t: -t[1])
    flagged_ids = {f["identity"] for f in flagged}

    out = {
        "threshold": args.thr,
        "n_gallery_images": len(G),
        "n_probe_images": len(P),
        "n_flagged_pairs": len(flagged),
        "n_identities_affected": len(flagged_ids),
        "top_similarities": [{"probe": w[0], "cosine": w[1]} for w in worst[:10]],
        "flagged": flagged,
        "verdict": ("near-duplicates present — treat Rank-1 as optimistic"
                    if flagged else
                    "no probe/gallery near-duplicates above threshold"),
    }
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"flagged {len(flagged)} probe(s) with same-identity cosine >= {args.thr}")
    print(f"top same-identity sims: {[round(w[1], 4) for w in worst[:5]]}")
    print(out["verdict"])
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()

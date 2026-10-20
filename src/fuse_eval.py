"""Score-level fusion of several embedding models on the same probe set.

Cosine scores from different models are on comparable scales, so a plain mean
is a reasonable (and hard-to-overfit) fusion rule. We also report each
component so the fusion gain is visible rather than assumed.

    python fuse_eval.py --scores eval_buffalo.scores.npz eval_antelope.scores.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_pak import bootstrap_ci, metrics_at, roc_points


def summarise(S, p_ids, g_ids, n_total, threshold=0.4):
    top1 = S.argmax(1)
    correct = g_ids[top1] == p_ids
    top1_scores = S[np.arange(len(S)), top1]
    genuine = np.array([S[i, np.where(g_ids == p_ids[i])[0][0]]
                        if (g_ids == p_ids[i]).any() else -1.0
                        for i in range(len(S))])
    imp = np.ones_like(S, bool)
    for i in range(len(S)):
        imp[i, g_ids == p_ids[i]] = False
    impostor = S[imp]
    thrs, far, tar = roc_points(genuine, impostor)
    i1e3 = int(np.argmin(np.abs(far - 1e-3)))
    i1e4 = int(np.argmin(np.abs(far - 1e-4)))
    eer = float(far[int(np.argmin(np.abs(far - (1 - tar))))])
    mean, lo, hi = bootstrap_ci(correct.astype(float))
    return {
        "rank1": float(correct.mean()), "rank1_ci": [mean, lo, hi],
        "end_to_end": float(correct.sum() / max(n_total, 1)),
        "tar_at_far_1e-3": float(tar[i1e3]),
        "tar_at_far_1e-4": float(tar[i1e4]),
        "eer": eer,
        "auc": float(-np.trapezoid(tar, far)),
        "genuine_mean": float(genuine.mean()),
        "impostor_mean": float(impostor.mean()),
        "op": metrics_at(genuine, impostor, threshold),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", nargs="+", required=True)
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--output", default="eval_fusion.json")
    args = ap.parse_args()

    mats, p_ids, g_ids, names, totals = [], None, None, [], []
    for s in args.scores:
        z = np.load(s, allow_pickle=True)
        mats.append(z["S"])
        if p_ids is None:
            p_ids, g_ids = z["p_ids"], z["g_ids"]
        else:
            assert (z["p_ids"] == p_ids).all() and (z["g_ids"] == g_ids).all(), \
                "probe/gallery order differs between score files"
        names.append(Path(s).name.replace(".scores.npz", "").replace("eval_", ""))
        js = Path(s).with_suffix("").with_suffix(".json")
        totals.append(json.load(open(js))["n_probes_total"] if js.exists() else len(p_ids))

    n_total = max(totals)
    out = {"components": {n: summarise(m, p_ids, g_ids, n_total, args.threshold)
                          for n, m in zip(names, mats)}}
    fused = np.mean(mats, axis=0)
    out["fused"] = {"models": names, "rule": "mean cosine",
                    **summarise(fused, p_ids, g_ids, n_total, args.threshold)}
    with open(args.output, "w") as fh:
        json.dump(out, fh, indent=2, default=str)

    print(f"{'model':22s} {'rank1':>7s} {'e2e':>7s} {'TAR@1e-3':>9s} {'EER':>7s}")
    for n in names + ["FUSED"]:
        r = out["components"][n] if n != "FUSED" else out["fused"]
        print(f"{n:22s} {r['rank1']:7.4f} {r['end_to_end']:7.4f} "
              f"{r['tar_at_far_1e-3']:9.4f} {r['eer']:7.4f}")
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()

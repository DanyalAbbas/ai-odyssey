"""
Figures for the TechFest report, built from evaluate_pak.py outputs.

    python figures.py --run eval_baseline_pad.json --run eval_ft_lfw.json:Fine-tuned

Each --run accepts  path.json  or  path.json:Label
Expects the sibling  path.scores.npz  produced by evaluate_pak.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False})


def load(spec: str):
    path, _, label = spec.partition(":")
    j = json.loads(Path(path).read_text())
    npz_path = Path(path).with_suffix(".scores.npz")
    arrays = np.load(npz_path, allow_pickle=True) if npz_path.exists() else None
    return j, arrays, (label or Path(path).stem)


def roc_arrays(j, a):
    gen = a["genuine"].astype(np.float64)
    imp = a["impostor"].astype(np.float64)
    thrs = np.linspace(-1, 1, 2001)
    far = 1 - np.searchsorted(np.sort(imp), thrs, side="left") / len(imp)
    tar = 1 - np.searchsorted(np.sort(gen), thrs, side="left") / len(gen)
    return far, tar


def plot_roc(runs, out):
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for j, a, label in runs:
        far, tar = roc_arrays(j, a)
        ax.plot(far, tar, lw=1.6,
                label=f"{label} (EER {j['eer']*100:.2f}%)")
        i = int(np.argmin(np.abs(far - (1 - tar))))
        ax.plot(far[i], tar[i], "o", ms=4)
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1)
    ax.set_xlabel("False accept rate (log)")
    ax.set_ylabel("True accept rate")
    ax.set_title("ROC — verification (1:1)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def plot_det(runs, out):
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    for j, a, label in runs:
        far, tar = roc_arrays(j, a)
        ax.plot(far, 1 - tar, lw=1.6, label=label)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1e-4, 1); ax.set_ylim(1e-4, 1)
    ax.set_xlabel("False accept rate (log)")
    ax.set_ylabel("False reject rate (log)")
    ax.set_title("DET curve")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def plot_cmc(runs, out):
    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    for j, a, label in runs:
        S = a["S"]; g = a["g_ids"]; p = a["p_ids"]
        hits = []
        order = np.argsort(-S, axis=1)
        for k in range(1, min(10, S.shape[1]) + 1):
            hits.append(float(np.mean([(g[order[i, :k]] == p[i]).any()
                                       for i in range(len(p))])))
        ax.plot(range(1, len(hits) + 1), hits, "o-", ms=3, label=label)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Cumulative match rate")
    ax.set_title("CMC — identification (1:N, 23 identities)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def plot_threshold(j, a, out):
    gen = a["genuine"]; imp = a["impostor"]
    thrs = np.linspace(0.0, 0.85, 200)
    prec, rec, f1, acc = [], [], [], []
    for t in thrs:
        tp = (gen >= t).sum(); fp = (imp >= t).sum()
        fn = (gen < t).sum()
        p = tp / (tp + fp) if tp + fp else 0
        r = tp / (tp + fn) if tp + fn else 0
        prec.append(p); rec.append(r)
        f1.append(2 * p * r / (p + r) if p + r else 0)
        correct = a["correct"]; top1 = a["top1_scores"]
        acc.append(float(((correct) & (top1 >= t)).mean()))
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.plot(thrs, prec, label="precision")
    ax.plot(thrs, rec, label="recall (TAR)")
    ax.plot(thrs, f1, label="F1")
    ax.plot(thrs, acc, "--", label="1:N accuracy (with rejection)")
    best = thrs[int(np.argmax(f1))]
    ax.axvline(best, color="k", ls=":", lw=1, label=f"best F1 @ {best:.2f}")
    ax.set_xlabel("Cosine threshold")
    ax.set_ylabel("Score")
    ax.set_title("Operating-point selection")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def plot_identity(j, out):
    per = j["per_identity"]
    ids = sorted(per, key=lambda k: per[k]["rank1"])
    vals = [per[k]["rank1"] * 100 for k in ids]
    fig, ax = plt.subplots(figsize=(5.6, 6.0))
    ax.barh(range(len(ids)), vals, color="#3b6ea5")
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels([i.replace("_", " ") for i in ids], fontsize=7)
    ax.set_xlabel("Rank-1 (%)")
    ax.set_xlim(0, 105)
    ax.set_title(f"Per-identity accuracy (worst {vals[0]:.0f}%, median "
                 f"{np.median(vals):.0f}%)")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def plot_quality(j, out):
    q = j["per_quality"]
    keys = sorted(q, key=lambda k: -q[k]["n"])
    vals = [q[k]["rank1"] * 100 if q[k]["rank1"] is not None else 0 for k in keys]
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.bar(keys, vals, color="#6a9a5b")
    for i, k in enumerate(keys):
        ax.text(i, vals[i] + 1, f"n={q[k]['n']}", ha="center", fontsize=8)
    ax.set_ylabel("Rank-1 (%)")
    ax.set_ylim(0, 108)
    ax.set_title("Accuracy by enrolment path")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()

    runs = [load(s) for s in args.run]
    runs = [r for r in runs if r[1] is not None]
    if not runs:
        raise SystemExit("no .scores.npz found next to the given JSON files")

    out = Path(args.out); out.mkdir(exist_ok=True)
    plot_roc(runs, out / "roc.png")
    plot_det(runs, out / "det.png")
    plot_cmc(runs, out / "cmc.png")
    plot_threshold(*runs[0][:2], out / "threshold.png")
    plot_identity(runs[0][0], out / "per_identity.png")
    plot_quality(runs[0][0], out / "quality.png")
    print("wrote:", ", ".join(sorted(p.name for p in out.glob("*.png"))))


if __name__ == "__main__":
    main()

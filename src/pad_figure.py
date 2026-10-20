"""PAD figure: live-score distributions + BPCER/APCER trade-off.

Input: data/pad_eval.json (written by `pad.py --fake ...`).
Output: reports/fig_pad.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

with open("data/pad_eval.json") as f:
    d = json.load(f)
real = np.array([r["live_prob"] for r in d["real_rows"] if r["live_prob"] is not None])
att = np.array([r["live_prob"] for r in d["attack_rows"] if r["live_prob"] is not None])

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))

ax[0].hist(real, bins=30, range=(0, 1), alpha=0.65, label=f"real (n={len(real)})",
           color="#2a7fb8")
ax[0].hist(att, bins=30, range=(0, 1), alpha=0.65, label=f"print attack (n={len(att)})",
           color="#c0392b")
ax[0].axvline(0.05, color="k", ls="--", lw=1.2, label="operating point 0.05")
ax[0].set_xlabel("PAD live score")
ax[0].set_ylabel("images")
ax[0].set_title("Liveness score: real vs print attack")
ax[0].legend(fontsize=8)

thrs = np.linspace(0, 1, 401)
bpcer = [(real < t).mean() for t in thrs]
apcer = [(att >= t).mean() for t in thrs]
ax[1].plot(bpcer, apcer, lw=1.8, color="#2a7fb8")
for t in (0.02, 0.05, 0.10):
    i = int(np.argmin(np.abs(thrs - t)))
    ax[1].plot(bpcer[i], apcer[i], "o", ms=5)
    ax[1].annotate(f"{t:.2f}", (bpcer[i], apcer[i]), textcoords="offset points",
                   xytext=(6, 6), fontsize=8)
ax[1].set_xlabel("BPCER (real rejected)")
ax[1].set_ylabel("APCER (attack accepted)")
ax[1].set_title("PAD trade-off (NUAA, held-out slice)")
ax[1].grid(alpha=0.3)

fig.tight_layout()
Path("reports").mkdir(exist_ok=True)
fig.savefig("reports/fig_pad.png", dpi=160)
print("saved reports/fig_pad.png")

"""Print one comparison table from several eval_*.json files.

    python summarise.py eval_buffalo_pad.json eval_antelope_pad.json ...
"""
import json
import sys
from pathlib import Path


def main(paths):
    rows = []
    for p in paths:
        if not Path(p).exists():
            print(f"(missing {p})")
            continue
        with open(p) as f:
            d = json.load(f)
        pad = d.get("pad") or {}
        rows.append({
            "model": d.get("config", {}).get("model", Path(p).stem),
            "n": d["n_probes_detected"],
            "fte": d["fte_rate"],
            "rank1": d["rank1_detected"],
            "ci": d.get("rank1_ci", [0, 0, 0]),
            "e2e": d["end_to_end_accuracy"],
            "tar3": d["tar_at_far_1e-3"],
            "tar4": d["tar_at_far_1e-4"],
            "eer": d["eer"],
            "auc": d["auc"],
            "gap": d["score_gap"],
            "bpcer": pad.get("bpcer"),
            "op_f1": d["operating_point"]["f1"],
        })
    if not rows:
        return
    print(f"{'model':34s} {'det':>5s} {'rank1':>7s} {'95% CI':>15s} {'e2e':>7s} "
          f"{'TAR@1e-3':>9s} {'EER':>7s} {'F1':>6s} {'BPCER':>6s}")
    for r in rows:
        ci = f"[{r['ci'][1]:.3f},{r['ci'][2]:.3f}]"
        bp = f"{r['bpcer']:.3f}" if r["bpcer"] is not None else "  -  "
        print(f"{r['model'][:34]:34s} {r['n']:5d} {r['rank1']:7.4f} {ci:>15s} "
              f"{r['e2e']:7.4f} {r['tar3']:9.4f} {r['eer']:7.4f} "
              f"{r['op_f1']:6.3f} {bp:>6s}")


if __name__ == "__main__":
    main(sys.argv[1:])

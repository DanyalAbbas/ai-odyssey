"""
Hair-covered vs hair-visible subgroup accuracy (descriptive).

Joins the manually tagged female probes with the raw scores dumped by
evaluate_pak.py. n is small (59 probes), so this is reported with a bootstrap CI
and treated as descriptive evidence, not a fairness verdict.

    python subgroup_hijab.py --json eval_baseline_pad.json \
        --tags contacts/female_probes_tags.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def boot(values, n=2000, seed=0):
    v = np.asarray(values, float)
    if v.size == 0:
        return [float("nan")] * 3
    rng = np.random.default_rng(seed)
    b = rng.choice(v, size=(n, v.size), replace=True).mean(1)
    return [float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="eval_baseline_pad.json")
    ap.add_argument("--tags", default="contacts/female_probes_tags.csv")
    ap.add_argument("--output", default="subgroup_hijab.json")
    args = ap.parse_args()

    z = np.load(str(Path(args.json).with_suffix(".scores.npz")), allow_pickle=True)
    files = [str(f).replace("\\", "/") for f in z["probe_files"]]
    correct = z["correct"].astype(float)
    p_ids = [str(x) for x in z["p_ids"]]

    tags = {}
    with open(args.tags) as fh:
        for row in csv.DictReader(fh):
            tags[Path(row["path"]).name] = int(row["head_covered"])
    # tags keys are basenames; map probe_files by basename
    out = {"n_tagged_total": len(tags), "groups": {}}
    for label, want in (("head_covered", 1), ("hair_visible", 0)):
        idx = [i for i, f in enumerate(files)
               if tags.get(Path(f).name) == want]
        vals = correct[idx]
        out["groups"][label] = {
            "n": len(idx),
            "rank1": boot(vals) if len(idx) else [float("nan")] * 3,
            "n_correct": int(vals.sum()),
        }

    # which female probes were never enrolled
    tagged_names = set(tags)
    detected = {Path(f).name for f in files}
    out["n_tagged_not_enrolled"] = len(tagged_names - detected)
    out["not_enrolled"] = sorted(tagged_names - detected)

    Path(args.output).write_text(json.dumps(out, indent=2))
    for g, v in out["groups"].items():
        m, lo, hi = v["rank1"]
        print(f"{g:14s} n={v['n']:2d}  rank1={m*100:.1f}% "
              f"[{lo*100:.1f}, {hi*100:.1f}]  ({v['n_correct']}/{v['n']})")
    print(f"tagged probes not enrolled: {out['n_tagged_not_enrolled']}")
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()

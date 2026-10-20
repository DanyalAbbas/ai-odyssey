"""Per-group accuracy from a completed eval run.

Attributes are predicted, not annotated: InsightFace's genderage model is run on
the *already aligned* 112x112 probe crop, so the crop the embedder scored is the
same crop the attribute model sees. That keeps the grouping consistent with the
recognition decision, but it is an automatic proxy and is labelled as such in the
report — no hand-labelling was done for gender or age.

Hair covering comes from the single-annotator tag file for the 59 female probes.

    python group_eval.py --eval eval_buffalo_v2.json \
        --crops data/cache/probe_crops_<hash>.npz --tags contacts/female_probes_tags.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np

from insightface.model_zoo.attribute import Attribute


class AttrFace(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def boot(v, n=2000, seed=0):
    v = np.asarray(v, float)
    if v.size == 0:
        return [float("nan")] * 3
    rng = np.random.default_rng(seed)
    b = rng.choice(v, size=(n, v.size), replace=True).mean(1)
    return [float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def bucket(age):
    if age is None:
        return "unknown"
    return "u30" if age < 30 else "30_49" if age < 50 else "50plus"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="eval_buffalo_v2.json")
    ap.add_argument("--crops", default=None)
    ap.add_argument("--tags", default="contacts/female_probes_tags.csv")
    ap.add_argument("--output", default="groups.json")
    args = ap.parse_args()

    z = np.load(str(Path(args.eval).with_suffix(".scores.npz")), allow_pickle=True)
    files = [str(f).replace("\\", "/") for f in z["probe_files"]]
    correct = z["correct"].astype(float)

    crops_file = args.crops or max(
        glob.glob("data/cache/probe_crops_*.npz"), key=lambda p: Path(p).stat().st_mtime)
    c = np.load(crops_file, allow_pickle=True)
    cmeta = {str(f).replace("\\", "/"): m for f, m in zip(c["files"], c["meta"])}
    ccrops = {str(f).replace("\\", "/"): cr for f, cr in zip(c["files"], c["crops"])}

    attr = Attribute(model_file=str(Path.home() /
                      ".insightface/models/buffalo_l/genderage.onnx"))
    attr.prepare(ctx_id=-1)

    sex, age = {}, {}
    for f in files:
        crop = ccrops.get(f)
        if crop is None or not np.any(crop):
            sex[f], age[f] = "unknown", None
            continue
        face = AttrFace(bbox=np.array([0.0, 0.0, 112.0, 112.0]))
        g, a = attr.get(crop, face)
        sex[f] = "male" if g == 1 else "female"
        age[f] = int(a)

    groups = {}
    for key, values in (("sex", {f: sex[f] for f in files}),
                        ("age", {f: bucket(age[f]) for f in files})):
        for g in sorted(set(values.values())):
            vals = [correct[i] for i, f in enumerate(files) if values[f] == g]
            m, lo, hi = boot(vals)
            groups[f"{key}:{g}"] = {"n": len(vals), "rank1": m, "ci": [lo, hi]}

    # hair covering, female probes only
    tags = {}
    tp = Path(args.tags)
    if tp.exists():
        with open(tp) as fh:
            for row in csv.DictReader(fh):
                key = row.get("file") or row.get("probe") or row.get("path")
                if not key:
                    continue
                if "head_covered" in row:
                    raw = str(row["head_covered"]).strip()
                    tags[Path(key).name] = ("hair_covered" if raw in {"1", "true", "True"}
                                            else "hair_visible")
                else:
                    tags[Path(key).name] = row.get("tag") or row.get("group")

    for g in sorted(set(tags.values())):
        vals = [correct[i] for i, f in enumerate(files)
                if tags.get(Path(f).name) == g]
        if vals:
            m, lo, hi = boot(vals)
            groups[f"hair:{g}"] = {"n": len(vals), "rank1": m, "ci": [lo, hi]}

    print(f"{'group':22s} {'n':>4s}  rank1     95% CI")
    for k, v in groups.items():
        print(f"{k:22s} {v['n']:4d}  {v['rank1']:.4f}  [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}]")

    with open(args.output, "w") as fh:
        json.dump({"eval": args.eval, "attribute_source": "insightface genderage on aligned crop",
                   "groups": groups,
                   "per_probe": {f: {"sex": sex[f], "age": age[f],
                                     "hair": tags.get(Path(f).name),
                                     "correct": bool(correct[i])}
                                 for i, f in enumerate(files)}}, fh, indent=2)
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()

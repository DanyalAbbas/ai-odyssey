"""Build a paired masked-vs-clean 1:N benchmark from Masked-LFW.

Design
------
Masked-LFW ships a masked copy of an LFW image under the *same* filename, plus
the original LFW tree. That lets us do a paired experiment where the only thing
that changes between the two conditions is the mask:

    gallery  = clean photos, never the photo used as a probe
    probe A  = the masked photo
    probe B  = the clean original of that same photo

So `rank1(masked) - rank1(clean)` is the cost of covering the lower face, with
identity, pose, lighting and compression held fixed. Surgical masks are not
niqabs, but they occlude the same region (nose, mouth, chin) and, unlike any
niqab set, they come with ground-truth identities for 400+ people.

    python build_lfw_masked.py --out data/bench_lfw --max-probes 400
"""
from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict
from pathlib import Path

MASKED = Path("data/raw/lfw_masks/Masked_LFW_Dataset/Masked_LFW_Dataset")
CLEAN = Path("data/raw/lfw_masks/LFW_without_Mask/LFW_without_Mask")
STEM = re.compile(r"^(?P<id>.+)_(?P<num>\d{4})$")


def ident(stem: str) -> str | None:
    m = STEM.match(stem)
    return m.group("id") if m else None


def link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        os.symlink(src.resolve(), dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/bench_lfw")
    ap.add_argument("--max-probes", type=int, default=400)
    ap.add_argument("--gallery-per-id", type=int, default=3)
    args = ap.parse_args()

    masked = defaultdict(list)
    for p in sorted(MASKED.glob("*.jpg")):
        i = ident(p.stem)
        if i:
            masked[i].append(p)

    clean = defaultdict(list)
    for p in sorted(CLEAN.rglob("*.jpg")):
        i = ident(p.stem)
        if i:
            clean[i].append(p)

    out = Path(args.out)
    n_ids = n_probe = n_gal = 0
    for pid in sorted(masked):
        if pid not in clean:
            continue
        probes = masked[pid][:2]
        probe_stems = {p.stem for p in probes}
        gal = [p for p in clean[pid] if p.stem not in probe_stems][: args.gallery_per_id]
        if len(gal) < 2:
            continue
        for g in gal:
            link(g, out / "gallery" / pid / g.name)
            n_gal += 1
        for pr in probes:
            link(pr, out / "probe_masked" / pid / pr.name)
            src = CLEAN / pid / (pr.stem + ".jpg")
            if src.exists():
                link(src, out / "probe_clean" / pid / src.name)
            n_probe += 1
        n_ids += 1
        if n_probe >= args.max_probes:
            break

    print(f"identities={n_ids} gallery={n_gal} probes={n_probe}")
    print(f"wrote {out}/gallery, {out}/probe_masked, {out}/probe_clean")


if __name__ == "__main__":
    main()

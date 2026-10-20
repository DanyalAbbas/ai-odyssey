#!/usr/bin/env python3
"""Assemble and push the fine-tuning dataset and kernel to Kaggle.

Two artefacts are produced from what is already in the repo:

1. A **dataset** containing the cleaned training corpus (`data/bench_train`),
   the manifest, and the source list. Small (~30 MB of 112x112 crops), so the
   upload is quick.
2. A **script kernel** containing `src/finetune_adaface.py` as `train.py`. The
   kernel attaches the public AdaFace IR-50 checkpoint dataset plus our corpus
   dataset, trains on a T4, and writes the ONNX model to /kaggle/working.

Usage:
    python src/push_finetune.py --data data/bench_train --push-dataset
    python src/push_finetune.py --push-kernel
    python src/push_finetune.py --status

Nothing here overwrites an existing dataset version: a new version is created
with a message describing the corpus size, which keeps the training run
auditable.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAGE = Path("/tmp/opencode/kaggle_ft")
DATASET_DIR = STAGE / "dataset"
KERNEL_DIR = STAGE / "kernel"

CKPT_SOURCE = "iambestfeeder/adaface-checkpoint"
KERNEL_SLUG = "techfest-pakistan-finetune"
DATASET_SLUG = "techfest-pak-train-faces"


def kaggle_username() -> str:
    """Read the owner from the CLI config (newer CLIs store an access token,
    not kaggle.json, so the username only exists in `config view`)."""
    proc = subprocess.run(["kaggle", "config", "view"], text=True,
                          capture_output=True)
    for line in proc.stdout.splitlines():
        if line.strip().startswith("- username:"):
            return line.split(":", 1)[1].strip()
    cfg = Path.home() / ".kaggle" / "kaggle.json"
    if cfg.exists():
        return json.loads(cfg.read_text())["username"]
    sys.exit("could not determine the Kaggle username; run `kaggle config view`")


def run(cmd: list[str], check=True) -> subprocess.CompletedProcess:
    print("$", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, text=True, capture_output=True)
    out = (proc.stdout + proc.stderr).strip()
    if out:
        print(out[-3000:], flush=True)
    if check and proc.returncode != 0:
        sys.exit(proc.returncode)
    return proc


# --------------------------------------------------------------------- dataset
def build_dataset(data_dir: Path) -> Path:
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    DATASET_DIR.mkdir(parents=True)

    if not data_dir.exists():
        sys.exit(f"missing {data_dir}; run src/clean_train_faces.py first")
    identities = sorted(p for p in data_dir.iterdir() if p.is_dir())
    if not identities:
        sys.exit(f"no identity folders in {data_dir}")
    total = sum(len(list(p.glob('*.jpg'))) for p in identities)

    # copy (not symlink) because the CLI zips the directory
    shutil.copytree(data_dir, DATASET_DIR / "bench_train",
                    ignore=shutil.ignore_patterns("*.log", ".DS_Store"))
    # manifest.csv is written inside bench_train; sources.csv lives with the
    # raw scrape and is the provenance record for the corpus
    for src in [data_dir / "manifest.csv",
                data_dir.parent / "raw" / "train_faces" / "sources.csv"]:
        if src.exists():
            shutil.copy2(src, DATASET_DIR / src.name)

    (DATASET_DIR / "README.md").write_text(
        f"""# Pakistani face training corpus

Collected from public web sources for the TechFest Round 2 brief, which asks
teams to gather their own training data. {len(identities)} identities,
{total} ArcFace-aligned 112x112 crops.

The identities are public Pakistani figures (politicians, cricketers, actors,
journalists, clerics) chosen to be disjoint from the 23-identity evaluation
benchmark. `manifest.csv` records the per-identity crop count, gender, and a
hijab/dupatta flag; `sources.csv` records the search queries used.

No identity in this corpus appears in `data/pak_fig`, and the evaluation code
asserts that before scoring.
""")

    meta = {
        "title": "TechFest Pakistan training faces",
        "id": f"{kaggle_username()}/{DATASET_SLUG}",
        "licenses": [{"name": "other"}],
    }
    (DATASET_DIR / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"staged {len(identities)} identities / {total} crops in {DATASET_DIR}")
    return DATASET_DIR


def push_dataset(message: str) -> None:
    if not DATASET_DIR.exists():
        sys.exit("stage the dataset first (--stage-dataset)")
    created = run(["kaggle", "datasets", "create", "-p", str(DATASET_DIR),
                   "--dir-mode", "zip"], check=False)
    if created.returncode == 0:
        print("dataset created")
        return
    if "already exists" in (created.stdout + created.stderr).lower():
        run(["kaggle", "datasets", "version", "-p", str(DATASET_DIR),
             "-m", message, "--dir-mode", "zip"])
        print("dataset version pushed")
    else:
        sys.exit(created.returncode)


# ---------------------------------------------------------------------- kernel
def build_kernel() -> Path:
    if KERNEL_DIR.exists():
        shutil.rmtree(KERNEL_DIR)
    KERNEL_DIR.mkdir(parents=True)
    shutil.copy2(ROOT / "src" / "finetune_adaface.py", KERNEL_DIR / "train.py")

    user = kaggle_username()
    meta = {
        "id": f"{user}/{KERNEL_SLUG}",
        "title": "techfest pakistan finetune",
        "code_file": "train.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        # the checkpoint and the corpus are attached, nothing is fetched at run time
        "enable_internet": False,
        "dataset_sources": [CKPT_SOURCE, f"{user}/{DATASET_SLUG}"],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (KERNEL_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"staged kernel in {KERNEL_DIR}")
    return KERNEL_DIR


def push_kernel() -> None:
    if not KERNEL_DIR.exists():
        build_kernel()
    run(["kaggle", "kernels", "push", "-p", str(KERNEL_DIR)])


def status() -> None:
    user = kaggle_username()
    run(["kaggle", "kernels", "status", f"{user}/{KERNEL_SLUG}"], check=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/bench_train")
    ap.add_argument("--stage-dataset", action="store_true")
    ap.add_argument("--push-dataset", action="store_true")
    ap.add_argument("--build-kernel", action="store_true")
    ap.add_argument("--push-kernel", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    if args.stage_dataset or args.push_dataset:
        build_dataset((ROOT / args.data).resolve())
    if args.push_dataset:
        n = sum(len(list(p.glob('*.jpg'))) for p in (DATASET_DIR / "bench_train").iterdir())
        push_dataset(f"corpus update {datetime.now(timezone.utc):%Y-%m-%d} "
                     f"({n} crops)")
    if args.build_kernel or args.push_kernel:
        build_kernel()
    if args.push_kernel:
        push_kernel()
    if args.status:
        status()
    if not any(vars(args).values()):
        ap.print_help()


if __name__ == "__main__":
    main()

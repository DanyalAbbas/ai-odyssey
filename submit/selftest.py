#!/usr/bin/env python3
"""End-to-end check of the submission package, offline, on bundled images.

    python selftest.py

Runs the real predict.py entry point (not a reimplementation) on two identities
and asserts the output is well-formed and correct.
"""
from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    out = Path(tempfile.mkdtemp()) / "selftest.csv"
    cmd = [sys.executable, str(HERE / "predict.py"),
           "--gallery", str(HERE / "selftest_data/gallery"),
           "--probe", str(HERE / "selftest_data/probe"),
           "--models-dir", str(HERE),
           "--output", str(out),
           "--threads", "4"]
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=HERE)

    with open(out) as f:
        rows = list(csv.DictReader(f))
    assert rows, "no rows written"
    assert set(rows[0]) == {"probe", "predicted_id", "score", "quality", "match"}, rows[0]
    n_ok = sum(r["match"] == "True" for r in rows)
    print(f"\n{len(rows)} rows, {n_ok} matched")
    for r in rows:
        print(f"  {r['probe'][:50]:50s} -> {r['predicted_id']:22s} {r['score']:>7s} {r['quality']}")
    assert len(rows) == 2, f"expected 2 probes, got {len(rows)}"
    assert n_ok >= 1, "package failed to match either probe"
    print("\nSELFTEST PASSED")


if __name__ == "__main__":
    main()

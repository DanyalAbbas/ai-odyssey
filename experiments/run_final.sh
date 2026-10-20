#!/usr/bin/env bash
# Remaining experiments: occlusion, ablations, FTE. Run from anywhere.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
export OMP_NUM_THREADS=6
P="--pad --pad-threshold 0.05 --batch-size 32"

echo "=== LFW masked probes (synthetic lower-face masks)"; date
$PY src/evaluate_pak.py --gallery data/bench_lfw/gallery \
  --probe data/bench_lfw/probe_masked --cache data/cache_lfw \
  --tag lfw_masked --output results/eval_lfw_masked.json

echo "=== LFW clean probes (paired control)"; date
$PY src/evaluate_pak.py --gallery data/bench_lfw/gallery \
  --probe data/bench_lfw/probe_clean --cache data/cache_lfw \
  --tag lfw_clean --output results/eval_lfw_clean.json

echo "=== controlled occlusion stress (identity fixed)"; date
$PY src/occlusion_stress.py --model baseline --output results/occlusion_stress.json

echo "=== gallery aggregation ablations (cache reuse)"; date
$PY src/evaluate_pak.py --model baseline $P --tag gal_first \
  --gallery-agg first --output results/eval_gal_first.json
$PY src/evaluate_pak.py --model baseline $P --tag gal_medoid \
  --gallery-agg medoid --output results/eval_gal_medoid.json

echo "=== no-retry ablation (FTE)"; date
$PY src/evaluate_pak.py --model baseline $P --tag nopad --pad-retry 0 \
  --output results/eval_nopad.json

echo "ALL DONE"; date

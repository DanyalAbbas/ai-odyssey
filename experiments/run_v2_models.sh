#!/usr/bin/env bash
# Embedder comparison on the corrected v2 crops (same crops for every model).
# antelopev2 v2 is already in results/; this reruns the two AdaFace models.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
export OMP_NUM_THREADS=6
P="--pad --pad-threshold 0.05 --batch-size 32"

echo "=== AdaFace IR-101 (v2 crops)"; date
$PY src/evaluate_pak.py --model onnx --onnx submit/models/adaface_ir101.onnx $P \
  --tag adaface_v2 --output results/eval_adaface_v2.json

echo "=== AdaFace IR-50 (v2 crops)"; date
$PY src/evaluate_pak.py --model onnx --onnx models/adaface_ir50.onnx $P \
  --tag adaface50_v2 --output results/eval_adaface50_v2.json

echo "ALL DONE"; date

# AI Odyssey - Face Recognition Challenge

Face recognition pipeline for TechFest Pakistani Context Challenge. 98.63% accuracy on benchmark.

## Quick Start

```bash
cd submit
python selftest.py
```

## Run on Custom Data

```bash
python submit/predict.py \
  --gallery /path/to/gallery \
  --probe /path/to/probe \
  --output predictions.csv
```

## Requirements

- Python 3.10+
- See requirements.txt for dependencies

## Status

✅ All 13 critical bugs fixed
✅ 98.63% end-to-end accuracy validated
✅ Ready for competition submission

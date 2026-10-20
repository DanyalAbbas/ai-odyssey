# TechFest Face Recognition Challenge — Pakistani Context

End-to-end face recognition: detection → alignment → quality gating → embedding →
matching, with a liveness gate and a veiled-face evaluation.

**Headline result: 98.63 % rank-1 end-to-end (431/437 probes, 23 identities, 0 %
failure-to-enrol), EER 0.69 %, AUC 0.9995.** The final report is
[`REPORT.md`](REPORT.md); the research trail is [`RESEARCH_LOG.md`](RESEARCH_LOG.md).

## Stack

- **Detector:** SCRFD-10GF at 640 px, with two escalations — a reflected-border
  retry when nothing is found, and a 320 px pass when only tiny faces are found
- **Alignment:** 5-pt ArcFace `norm_crop` → 112×112
- **Embedding:** buffalo_l R50/WebFace600K by default; antelopev2 R100/Glint360K and
  AdaFace IR-101 as alternatives (all 512-d, L2-normalised)
- **Quality gate:** Laplacian blur ≥ 40, IOD ≥ 64 px, yaw ≤ 30° — failures fall
  back to the largest face and are flagged, never dropped
- **Matching:** cosine, threshold 0.40 (FRR 0.69 %, FAR 0.48 %)
- **Anti-spoof:** MiniFASNet v2 + v1se ensemble, raw BGR 0–255 input, gate at 0.02

## What gets submitted

The organisers run the hidden benchmark, so the artefact is a **runnable offline
package**, not a predictions file. See [`SUBMISSION.md`](SUBMISSION.md).

```bash
cd submit && python selftest.py     # 2 rows, both matched, SELFTEST PASSED
```

`submit/predict.py` handles 1:N (nested or flat gallery), 1:1 pairs, and always
emits one output row per input. Weights are vendored; nothing is downloaded at
runtime.

## Files

| File | Purpose |
|---|---|
| `submit/` | **The submission**: `predict.py`, `selftest.py`, vendored ONNX weights, `Dockerfile` |
| `src/` | All pipeline, evaluation and experiment code (see below) |
| `results/` | One JSON + score matrix per run, figures in `results/figures/` |
| `experiments/` | Reproducible drivers (`run_final.sh`, `run_v2_models.sh`) and their logs |

`submit/` is deliberately self-contained: it carries its own copies of
`inference.py`, `pad.py` and `imgio.py` so the package has no dependency on this
repo's layout. When changing runtime code, edit `src/` and copy across.
| `src/inference.py` | Production pipeline (PAD → detect → align → quality → embed → match) |
| `src/evaluate_pak.py` | Benchmark harness: embedder backends, crop/embedding caches, ablations, bootstrap CIs |
| `src/pad.py`, `src/imgio.py` | Liveness ensemble, and EXIF-aware image loading |
| `src/veiled_eval.py` | Enrolment funnel on real niqab data (detected → quality → liveness) |
| `src/occlusion_stress.py` | Controlled hijab/niqab occlusion with identity held fixed |
| `src/build_lfw_masked.py` | Paired masked/clean LFW benchmark builder |
| `src/subgroup_hijab.py` | Hair-covered vs hair-visible subgroup accuracy |
| `src/fuse_eval.py` | Score-level fusion of several embedders |
| `src/summarise.py` | One-table comparison across eval JSONs |
| `src/figures.py` | ROC / CMC / threshold / per-identity figures |
| `src/check_duplicates.py` | Near-duplicate leakage audit across the gallery/probe split |
| `src/diagnose_fte.py` | Failure-to-enrol diagnosis |
| `experiments/run_final.sh`, `experiments/run_v2_models.sh` | Exact command sequences behind the reported numbers |
| `src/kaggle_train_v2.py` | Kaggle GPU fine-tune ablation (negative result; not shipped) |

## Reproducing

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt

python src/evaluate_pak.py --model baseline --pad --pad-threshold 0.05 \
  --gallery data/pak_fig/gallery --probe data/pak_fig/probe \
  --output results/eval_buffalo_v2.json     # ~15 min CPU
./experiments/run_final.sh                  # occlusion / ablations / FTE suite
```

The benchmark data is the public *Pakistani Public Figures* set (CC0); the raw
images are not committed (see `.gitignore`) but every source and licence is listed
in `REPORT.md` §3.1.

## What was tried and rejected

Honest negatives, all documented in the report:

- **Fine-tuning on LFW** (Kaggle T4, 12 epochs, AdaFace head): 41.7 % rank-1 on
  held-out LFW identities vs 98.6 % for the pretrained pipeline — not promoted
  under a pre-registered rule (§6.2).
- **Score fusion of four embedders:** 98.17 %, *below* the best single model,
  because the residual errors are correlated (§6.5).
- **Larger embedders (R100, IR-101):** statistically tied with R50 — one probe
  difference, exact McNemar p = 1.0 (§6.5).
- **Upscaling for tight crops:** rescued nothing; the reflected border did (§4.1).

## Dates

- Report: Oct 22
- Hidden evaluation: Oct 23

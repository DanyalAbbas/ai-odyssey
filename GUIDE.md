# Reviewer's guide

The fast path through this repository: what it does, what to run, where every
number comes from, and what is deliberately *not* in git. The written report is
[`REPORT.md`](REPORT.md); decisions and dead ends are in
[`RESEARCH_LOG.md`](RESEARCH_LOG.md).

## Status at a glance

| Item | State |
|---|---|
| Pipeline + submission package | **done** — `submit/selftest.py` passes offline |
| Public benchmark (23 IDs, 437 probes) | **done** — 98.63 % rank-1, EER 0.69 % |
| Report, ablations, fairness, PAD | **done** |
| Compliance fine-tune on the self-collected corpus | **run completed, rejected** — 98.17 % vs 98.63 % rank-1 (§6.6.4) |

Nothing in the submission depends on the queued run. If it completes, its result
is appended and the pre-registered promotion rule in §6.6.4 is applied; if it
does not, the baseline stands and the report says so.

## What the system does

```
image ─► SCRFD-10GF @640 ──(no face)──► reflected-border retry
                          ──(tiny face)──► 320 px pass
      ─► 5-pt ArcFace norm_crop 112×112
      ─► quality gate (blur ≥ 40, IOD ≥ 64 px, yaw ≤ 30°)  ── rejections flagged, never dropped
      ─► R50 / ArcFace 512-d, L2-normalised
      ─► cosine against per-identity gallery means ─► match / unknown
                 ▲
      MiniFASNet v2 + v1se liveness gate (optional, --pad)
```

The decisive finding in the report is that the jump from 73 % to 98.6 % came
from the two detection escalations, not from a bigger embedder: four different
models land within one probe of each other.

## Run it (≈1 minute)

```bash
uv sync                                  # runtime deps only
uv run python submit/selftest.py         # 2 identities, expects "SELFTEST PASSED"
```

The self-test runs the real `submit/predict.py` on two bundled identities and
asserts the output contract. It does not touch the network and needs no dataset.

Full benchmark run, using the local data directory that ships outside git:

```bash
uv run python submit/predict.py \
  --gallery data/pak_fig/gallery --probe data/pak_fig/probe \
  --output /tmp/preds.csv                # ≈8 min on 8 CPU threads
```

Presentation UI:

```bash
uv run --extra demo python app/demo.py   # http://127.0.0.1:7860
```

`--extra demo` matters: a plain `uv run` syncs the project without Gradio.

## Where every reported number comes from

All commands are run from the repo root. `results/*.json` is committed evidence;
`*.scores.npz` next to it is the score matrix needed to redo threshold sweeps.

| Report claim | Command | Artefact |
|---|---|---|
| 98.63 % rank-1, EER, FAR/FRR, CIs | `uv run python src/evaluate_pak.py --model baseline --pad --pad-threshold 0.05 --output results/eval_buffalo_v2.json` | `results/eval_buffalo_v2.json` |
| Four-embedder comparison (§6.5) | `evaluate_pak.py --model antelope` / `--model onnx --onnx submit/models/adaface_ir101.onnx` / `--onnx models/adaface_ir50.onnx` | `eval_antelope_v2`, `eval_adaface_v2`, `eval_adaface50_v2` |
| Score fusion | `uv run python src/fuse_eval.py --scores results/eval_buffalo_v2.scores.npz results/eval_antelope_v2.scores.npz …` | `eval_fusion_v2.json` |
| Gallery aggregation (§8.2) | same eval with `--gallery-agg first` / `medoid` | `eval_gal_first`, `eval_gal_medoid` |
| Detection escalation ablation | same eval with `--pad-retry 0` | `eval_nopad.json` |
| Masked vs clean LFW (§7.4) | `./experiments/run_final.sh` (first two stages) | `eval_lfw_masked`, `eval_lfw_clean` |
| Controlled hijab/niqab/mask stress | `uv run python src/occlusion_stress.py --model baseline` | `occlusion_stress.json` |
| Hijab subgroup (§7.4.4) | `uv run python src/subgroup_hijab.py --json results/eval_buffalo_v2.json` | `subgroup_hijab.json` |
| Real niqab enrolment funnel (§7.4.2) | `uv run python src/veiled_eval.py --images … --labels …` | `veiled_niqab_*.json` |
| Liveness APCER/BPCER | `uv run python src/pad.py --fake <attack_dir> --output data/pad_eval.json` then `src/pad_figure.py` | `pad_eval.json` |
| Near-duplicate audit | `uv run python src/check_duplicates.py` | `duplicate_audit.json` |
| Per-identity / gender tables | `uv run python src/group_eval.py --eval results/eval_buffalo_v2.json` | `groups_buffalo_v2.json` |
| Figures | `uv run python src/figures.py --run results/eval_buffalo_v2.json` | `results/figures/` |
| Shipped CLI, end to end | `uv run python submit/predict.py --gallery data/pak_fig/gallery --probe data/pak_fig/probe --output results/final_predictions.csv` | `results/final_check.json` |

The full command sequences used for the sweep are in
`experiments/run_final.sh` and `experiments/run_v2_models.sh`, with their logs
in `experiments/logs/` (scratch, gitignored).

## What is in git, and what is not

Tracked: all code, the report, the research log, `results/*.json` + score
matrices + figures, and the two-identity self-test images.

Not tracked, by design:

| Path | Why | How to restore |
|---|---|---|
| `data/` | 1,286 eval images + external datasets; licensing and size | public sources listed in REPORT §3.1; `data/pak_fig/README.md` documents the benchmark |
| `submit/models/` (435 MB) | weights; checksums pin them in REPORT §12.4 | InsightFace model zoo (`buffalo_l`, `antelopev2`), AdaFace release, MiniFASNet `.pth` converted by `src/convert_pad.py` |
| `models/` | scratch copies of the same weights | as above |
| `submit.zip` | rebuildable from `submit/` | `cd submit && zip -rX ../submit.zip . -x '__pycache__/*' '*.pyc'` |
| `.venv/` | machine-specific | `uv sync` |

The local checkout a reviewer receives has all of these in place, so every
command above runs as written. A fresh clone can reproduce the code path and the
self-test, but needs the weights and data dropped in first.

## The self-collected corpus and the fine-tune

The brief asks teams to collect data and fine-tune, so we did:

1. **Collect** — `src/collect_train_faces.py` scraped 99 Pakistani public
   figures (4,203 images, provenance in `data/raw/train_faces/sources.csv`),
   refusing any name that collides with the eval set.
2. **Clean** — `src/clean_train_faces.py` aligned the largest face into
   `data/bench_train/`: 98 identities, 2,230 crops (54 women, 24 veiled).
3. **Train** — `src/finetune_adaface.py` fine-tunes AdaFace IR-50 (MS1MV2,
   reconstructed and validated at 98.40 % first) with an ArcFace head, frozen
   BatchNorm, and synthetic hijab/niqab/mask occlusion on aligned crops.
4. **Push** — `src/push_finetune.py` uploads the corpus as a private Kaggle
   dataset and the training script as a kernel (40 epochs, one T4). Kaggle now
   mounts datasets under `/kaggle/input/datasets/<owner>/<slug>/`, so the script
   searches by name rather than assuming a path.
5. **Evaluate** — the run completed 40 epochs (~11 s/epoch) and exported a
   167 MB ONNX model; it was scored with
   `OMP_NUM_THREADS=8 .venv/bin/python src/evaluate_pak.py --model onnx --onnx <export>.onnx --pad --pad-threshold 0.05 --output results/eval_ft_v1.json`.

The promotion rule was written *before* the run: replace the baseline only if
rank-1 and TAR@1e-3 both do not drop; a tie goes to the fine-tuned model; any
loss keeps the baseline. It lost both (98.17 % vs 98.63 % rank-1, TAR 95.88 vs
97.71), so the baseline ships and the run is reported as a negative result
(§6.6.4). The earlier LFW fine-tune failed the same rule at 41.7 % (§6.2).

## How the pieces fit

| Path | What it is |
|---|---|
| `submit/` | the deliverable: `predict.py`, vendored weights, `Dockerfile`, `selftest.py` |
| `src/` | the same pipeline plus every experiment harness |
| `app/demo.py` | Gradio UI; calls `submit/inference.py`, so the demo and the benchmark cannot drift |
| `results/` | one JSON + score matrix per run, plus `figures/` |
| `experiments/` | reproducible driver scripts |
| `data/` | local datasets and caches (gitignored) |

`submit/inference.py` and `src/inference.py` are intentionally identical copies
(the package must not import from the repo layout). If you change runtime
behaviour, change both.

## Points a reviewer should push on

These are the weaknesses we know about, stated so they are not mistaken for
oversights:

- The benchmark is **closed-set** (every probe has a mate) and only 23
  identities. Threshold 0.40 costs five probes relative to rank-1; open-set
  performance is reported through FAR/FRR/EER, not through rank-1.
- The gallery/probe split is by image, not identity, because that is what a
  1:N closed-set benchmark means. Near-duplicate leakage is audited
  (`duplicate_audit.json`) rather than assumed away.
- **Gender fairness has little power**: 3 of 23 identities are women. Age and
  skin tone are declared unreportable instead of estimated.
- Real niqab data carries **no identity labels**, so it measures detection,
  quality and liveness only. Recognition under veiling is measured
  synthetically, holding identity fixed (`occlusion_stress.json`).
- Liveness is evaluated on **print attacks from NUAA**, not replay or 3-D
  masks, and is off by default in the submission (`--pad` enables it).
- The fine-tune is one 40-epoch run on a small corpus; it is a compliance
  artefact with a pre-registered decision rule, not a claim that more data
  would not help.

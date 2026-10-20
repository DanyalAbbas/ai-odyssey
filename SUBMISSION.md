# What we submit

The TechFest team runs the hidden benchmark themselves. So the submission is a
**runnable package**, not a predictions file: they execute our code on data we
never see.

## Artefacts

| # | Artefact | Contents | Due |
|---|---|---|---|
| 1 | the **repo** (or `submit/`, or `submit.zip`) | `submit/predict.py`, vendored ONNX weights, `Dockerfile`, `requirements.txt`, `selftest.py` | Oct 22 |
| 2 | `REPORT.pdf` | The written report | Oct 22 |
| 3 | `predictions.csv` (local) | Proof the package works; not the scored artefact | — |

## The contract

The repository itself is runnable — no unzipping, no copying:

```bash
uv sync                                     # or: uv venv && uv pip install -r requirements.txt
uv run python submit/predict.py --gallery <gallery_dir> --probe <probe_dir> --output predictions.csv
```

`submit.zip` exists only for a portal that wants a single upload; it contains the
same `submit/` folder plus its weights, so it runs standalone.

Also supported, in case their protocol differs:

```bash
# flat gallery + index CSV (file,id)
uv run python submit/predict.py --gallery imgs/ --gallery-index index.csv --probe probe/

# 1:1 verification
uv run python submit/predict.py --pairs pairs.csv --output predictions.csv
```

No network access at runtime. Model weights are inside the package.

## Output

```
probe,predicted_id,score,quality,match
subdir/img.jpg,ahsan_iqbal,0.8665,ok,True
subdir/bad.jpg,unknown,0.0,no_face,False
```

One row per input, always — unreadable images become `unknown`/`read_error`
rather than being dropped.

## Frozen configuration

| Stage | Setting |
|---|---|
| Detection | SCRFD-10GF @640, thresh 0.5, reflected-border retry, coarse 320 px pass |
| Alignment | 5-pt ArcFace template, 112×112 |
| Quality | IOD ≥ 64, blur ≥ 40, yaw ≤ 30 (rejections flagged, not dropped) |
| Embedding | buffalo_l R50 (`w600k_r50`) by default; antelopev2 R100 or AdaFace IR-101 via flags |
| Gallery | mean of L2-normalised embeddings per identity |
| Matching | cosine, threshold 0.40 (FRR 0.69 %, FAR 0.48 % on the public benchmark) |
| Liveness | off by default; `--pad` enables MiniFASNet v2+v1se @ 0.02 |

## Verify before sending

```bash
uv run python submit/selftest.py
```

Expected: 2 rows, both matched, `SELFTEST PASSED`.

## Known risks

- **Runtime.** ~0.3 s/image on 8 CPU threads (R50). A 10k-image benchmark is
  ~50 min. R100/AdaFace is ~3× slower.
- **Protocol mismatch.** If the hidden task is neither 1:N nor a pairs CSV, the
  wrapper changes but the pipeline does not.
- **Multi-face images.** Resolved to the largest quality-passing face.

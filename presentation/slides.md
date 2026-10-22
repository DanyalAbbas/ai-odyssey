# Face Recognition in a Pakistani Context

**Team Mamdani** · NEDUET CS&IT

TechFest Round 2 · 22 October 2026

<!-- title -->

---

# The Brief

- Build an end-to-end face recognition system for Pakistani deployment
- **Collect your own data and fine-tune** a model
- Report accuracy, precision, recall, F1, ROC, FAR/FRR, EER, ablations
- **Per-group fairness** — head coverings, gender, occlusion
- **Anti-spoofing** is required, not optional

Our reading: the hard part is not the embedder. It is detection, calibration,
and being honest about what the benchmark can and cannot support.

---

# Headline

<div class="kpis">
<div><span class="big">98.63%</span><br/>rank-1 · 431/437</div>
<div><span class="big">0.69%</span><br/>EER</div>
<div><span class="big">0.9995</span><br/>AUC</div>
<div><span class="big">0%</span><br/>failure-to-enrol</div>
</div>

- 23 identities · 849 gallery images · 437 probes · all 437 enrolled
- TAR@1e-3 **97.71%** · precision 0.904 · recall 0.993 · F1 0.947 @ 0.40
- The shipped package, run end to end at threshold 0.40: **430/437 = 98.40%**

---

# Data

| Dataset | Role | Size |
|---|---|---|
| Pakistani Public Figures | evaluation | 23 IDs · 1,286 images (849 gallery / 437 probes) |
| Self-collected corpus | fine-tune | 99 → **98 IDs · 2,230 crops** (54 women, 24 veiled) |
| LFW | fine-tune ablation | 5,749 IDs · 13,233 images |
| Masked-LFW | occlusion control | 200 IDs · 400 clean + 400 masked probes |
| Niqab sets (Kaggle, OFD) | enrolment only | 532 + 400 images, **no identity labels** |
| NUAA | liveness | 150 real + 150 print attacks |

**No identity overlap** between training and evaluation — checked by name before
download and again before training. Near-duplicate audit: 2/437 probes; removing
both gives 98.62%, so the headline is not leakage-driven.

---

# Pipeline

<div class="flow">
<span>SCRFD-10GF<br/><small>640 px</small></span><b>→</b>
<span>reflected-border<br/>retry<small> if 0 faces</small></span><b>→</b>
<span>320 px pass<small> if tiny faces</small></span><b>→</b>
<span>5-pt alignment<br/><small>112×112</small></span><b>→</b>
<span>quality<br/><small>IOD · blur · yaw</small></span><b>→</b>
<span>R50 / ArcFace<br/><small>512-d, L2</small></span><b>→</b>
<span>cosine<br/><small>identity mean</small></span>
</div>

- **Liveness gate** (MiniFASNet v2 + v1se) sits in front of the matcher when enabled
- Quality failures are **flagged and salvaged**, never silently dropped
- One row out per image in, including `unknown` / `read_error`

---

# The Decisive Finding: Detection, Not the Model

| Stage | Probes enrolled | Rank-1 | TAR@1e-3 |
|---|---|---|---|
| Native SCRFD only | 328 / 437 (75.1%) | 73.0% | 1.14 |
| + reflected-border retry | **437 / 437 (100%)** | 97.94% | 97.03 |
| + 320 px coarse pass | 437 / 437 | **98.63%** | **97.71** |

- 109/437 were tight headshots where SCRFD found **zero** faces; the retry rescued **109/109**
- The 320 px pass fixed **2 silent identity swaps**: a background face was returned instead of the subject
- Swapping R50 for a 4× larger model moved the score by **one probe**

---

# Four Embedders, One Result

| Model | Rank-1 | TAR@1e-3 | EER |
|---|---|---|---|
| buffalo_l R50 | 98.63% | 97.71 | 0.69% |
| antelopev2 R100 | 98.86% | 97.94 | 0.46% |
| AdaFace IR-101 | 98.86% | **98.17** | 0.46% |
| AdaFace IR-50 | 98.63% | 97.71 | 0.46% |
| Mean fusion of all four | 98.86% | 97.94 | 0.46% |

- All six errors are **nested**: no model succeeds where another fails
- Fusion cannot help when the mistakes are identical
- **Ship R50:** ties on rank-1, 184 MB of the 435 MB bundle, ~3× faster on CPU
- IR-101 is the better *verification* choice: best genuine–impostor separation

---

# The Operating Point

<div class="two wide-right">
<div>

- Threshold 0.40: **FRR 0.69% · FAR 0.48%**
- 0.35 lowers FRR to 0.46% but doubles FAR to 0.89%
- Shipped CLI gives **98.40%** at 0.40; the harness gives 98.17% — one probe
  apart for a documented quality-clip artefact, not a different operating point

</div>
<div>

![](results/figures/roc.png)

</div>
</div>

---

# Veiled Faces: Ranking Survives, Calibration Does Not

| Condition | Rank-1 | TAR@1e-3 | Genuine mean |
|---|---|---|---|
| Clean | 97.94% | 97.03 | 0.79 |
| Hijab (hair + sides) | **97.94%** | 96.57 | 0.76 |
| Niqab light (eyes clear) | 96.57% | 90.39 | 0.50 |
| Niqab full | 91.99% | 83.98 | 0.36 |
| Niqab extreme (eye slit) | 89.02% | 72.99 | 0.28 |

- **The hijab is free** — the five alignment landmarks are untouched
- Even extreme covering keeps the right identity first 89% of the time
- But the score distribution moves: at a fixed 0.40 threshold, niqab probes are
  returned as `unknown`. Correct ranking, wrong calibration.

---

# Real Niqab Data: The Bottleneck Is Size

| Dataset | Images | Detected | Strict quality | Liveness |
|---|---|---|---|---|
| Kaggle niqab photos | 532 | **82.1%** | 22.0% | **97.9%** |
| OFD occluded faces | 400 | 61.3% | 19.8% | 93.9% |

- Median inter-ocular distance on detected faces: **48–50 px**, below our 64 px gate
- A 112×112 template from a 50 px face is mostly interpolation
- **Anti-spoofing is not biased against veils** — 94–98% pass, comparable to unveiled
- Real niqab data has no identity labels, so we measure enrolment, never recognition accuracy

---

# Fairness: Report What the Data Can Support

- **Gender, per identity** (not per probe): 20 male IDs → 98.41% (378 probes);
  3 female IDs → 100% (59 probes)
- Three identities is **low power** — this is a check, not a fairness claim
- An attribute model labelled 13/23 identities female, including bearded men —
  which is why per-probe gender labels were rejected
- **Age and skin tone are declared unreportable** rather than estimated
- Hijab subgroup: 97.94% — no measurable penalty for covering hair

---

# Liveness: Included and Measured

- MiniFASNet **v2 + v1se** ensemble, raw BGR 0–255 input, 80×80
- NUAA print attacks, held-out preprocessing slice:

| Threshold | APCER | BPCER | BPCER on our 437 probes |
|---|---|---|---|
| **0.02** | **1.33%** | **1.33%** | **2.29%** |
| 0.05 | 0.67% | 4.00% | 4.35% |

- UI gate at 0.05: **412/437 = 94.28%** end to end
- Among accepted probes: **98.56%** — the loss is rejection, not misidentification
- Trade-off stated, not hidden: lower the gate and admit more spoofs

---

# The Compliance Fine-Tune: Run, Rejected

- Corpus: **98 identities · 2,230 aligned crops** (54 women, 24 veiled), synthetic
  hijab/niqab/mask painted on the aligned crop
- AdaFace **IR-50 MS1MV2** + ArcFace head, frozen BN, **40 epochs** on one T4
  (2,034 train / 196 held out, ~11 s/epoch)
- Held-out fit check: 100% — the overfitting warning, not a result

| Model | Rank-1 | TAR@1e-3 | EER |
|---|---|---|---|
| buffalo_l R50 (shipped) | **98.63%** | **97.71** | 0.69% |
| IR-50 + corpus fine-tune | 98.17% | 95.88 | 1.37% |

**Rejected on a pre-registered rule** (no loss on rank-1 *or* TAR@1e-3).
Both lost probes are low-quality captures (22/24 vs 24/24); every other quality
bucket is unchanged. 2,230 clean crops cannot teach what MS1MV2 already knows
about blur and small faces.

---

# What Failed, and What We'd Do Next

- **LFW fine-tune:** 41.7% vs 97.9% — rejected by the same rule
- **Score fusion:** 98.86%, identical to the best single model
- **Larger embedders:** statistically tied (one probe, McNemar p = 1.0)
- **Upscaling tight crops:** rescued nothing; the reflected border did
- **Masked faces:** ranking holds, threshold does not — recalibrate per condition

Next: per-condition score normalisation, more real veiled data with labels,
and a fix for the harness quality-clip artefact before the next benchmark.

---

# What Ships

<div class="two">
<div>

**The package**

- `submit/predict.py` — 1:N nested, flat + index, 1:1 pairs
- Vendored ONNX weights, offline, no downloads
- One output row per input, always
- `selftest.py` → **SELFTEST PASSED**
- 435 MB bundle; R50 default; ~0.9 s/image on 8 CPU threads

</div>
<div>

**The demo**

```bash
uv run --extra demo python app/demo.py
```

- Identify · Verify 1:1 · Enrol
- Shows the aligned crop, top-5 scores,
  liveness, quality and detection trace
- Same `inference.py` as the benchmark

</div>
</div>

---

# Takeaway

<div class="close">

**Detection got us to 98.6%. The next problem is calibration, not capacity.**

We report the negative results, the low-power groups, and the one-probe
differences — because a benchmark you cannot interrogate is not evidence.

</div>

Team Mamdani · NEDUET CS&IT · TechFest 2026

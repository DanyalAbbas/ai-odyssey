# Research Log — TechFest Face Recognition Challenge

**Date:** Oct 19, 2026
**Team:** NEDUET CS&IT
**Deadline:** Oct 22 (report) | Oct 23 (hidden benchmark)

---

## 1. Problem Summary

Build an end-to-end face recognition pipeline for Pakistani context:
- Face detection → alignment → quality filtering → embedding → matching
- Fine-tune on self-collected or public data
- Evaluate on disjoint benchmark (no identity overlap)
- Report: accuracy, precision, recall, F1, ROC, FAR, FRR, EER + ablations
- Presentation: model choice, data, consent, challenges, per-group results

---

## 2. Pipeline Architecture (Current)

```
SCRFD-10GF (det_10g.onnx) → 5-pt ArcFace alignment → 112×112
  → ArcFace R50 (w600k_r50.onnx) → L2-normalize → cosine similarity
  + Quality gate: Laplacian blur + IOD≥64 + pose yaw<30°
  + Threshold: 0.4 (to be tuned)
```

**Model pack:** InsightFace `buffalo_l` (detector + embedding)
**Input:** 112×112 aligned face crop
**Output:** 512-d normalized embedding
**Matching:** cosine similarity, threshold 0.4

---

## 3. Models & Approaches Considered

### Embedding Models

| Model | Year | Venue | IJB-C@1e-4 | TinyFace R1 | License | Verdict |
|---|---|---|---|---|---|---|
| **ArcFace R50** (current) | 2019 | CVPR | 97.25 (buffalo_l) | — | Non-commercial | Baseline, working |
| **AdaFace IR-101** | 2022 | CVPR | 97.66 (W12M) | 72.29 | MIT | **Primary upgrade** — best low-quality |
| **FaceMoE** | 2026 | ECCV | — | 76.18 | MIT code + Apache-2.0 | LR/Surveillance leader; needs 8 GPUs |
| **PETALface** | 2025 | WACV | — | 76.66–81.38 | MIT | Best for few-thousand-image fine-tune (LoRA) |
| **LVFace** | 2025 | ICCV | 97.70–98.06 | — | Research-only | Zero-shot bake-off candidate |
| **TopoFR** | 2024 | NeurIPS | 97.60 (R100) | — | Apache-2.0 | HQ SOTA, full training code |
| **EdgeFace** | 2023/24 | T-BIOM | — | — | Weights CC-BY-NC | Efficiency only |
| **TransFace** | 2023 | ICCV | — | — | Research | Superseded by LVFace |
| FaceNet | 2015 | CVPR | — | — | MIT | Legacy, weak |
| CosFace/CurricularFace | 2018/2020 | CVPR | — | — | Research | Superseded |

### Detectors

| Detector | Year | WIDER Hard | Speed | Verdict |
|---|---|---|---|---|
| **SCRFD-10GF** | 2021 | 82.8 | ~5 ms | **Current** — best accuracy/FLOP |
| RetinaFace R50 | 2020 | 92.2 | Slower | Best landmarks, fallback |
| YOLO11-face | 2024 | 85.2 | ~14 ms | Easy to fine-tune |
| YuNet | 2023 | 75.0 | 1.6 ms | CPU/edge fallback |
| MTCNN | 2016 | 60.7 | Slow | Legacy only |

### FIQA (Quality Assessment)

| Method | Year | Type | Verdict |
|---|---|---|---|
| CR-FIQA | 2023 | Learned | Reference standard |
| CLIB-FIQA | 2024 | CLIP-based | Current SOTA |
| OFIQ | ISO 29794-5:2025 | Reference impl | Explainable gates |
| SER-FIQ | 2020 | Dropout-based | No training needed |
| Laplacian/BRISQUE | — | No-reference | Cheap pre-gate only |

### Anti-Spoofing

| Method | Year | Verdict |
|---|---|---|
| MiniFASNet/Silent-Face | 2020 | 1.85 MB, 8 ms CPU — practical gate |
| MVP-FAS | 2025 | CLIP-based, cross-domain SOTA |
| DINOv2+Registers | 2024 | Simple 87M baseline |

---

## 4. Datasets

### Training

| Dataset | Size | License | Status |
|---|---|---|---|
| WebFace4M | 500k imgs / 50k+ IDs | Research agreement | Download pending |
| LFW | 13k imgs / 5.7k IDs | Free | **Downloading in Kaggle notebook** |
| Pakistani Public Figures | 1,286 imgs / 23 IDs | CC0 | ✅ Downloaded locally |

### Evaluation

| Dataset | Size | Use | Status |
|---|---|---|---|
| Pakistani Public Figures (probe) | 437 imgs | Primary eval | ✅ Done — 74.1% |
| RFW Indian | ~10k imgs / 3k IDs | South Asian proxy | Email requested |
| BUPT-Balancedface Indian | ~325k imgs / 7k IDs | Balanced eval | Email requested |
| FairFace | 108k imgs | Demographic labels | Downloaded |
| halfFace | 166 subjects | South Asian masked | Pending |

### Synthetic (for augmentation, not training)

| Dataset | Size | Use |
|---|---|---|
| DigiFace-1M | 1.22M / 110k IDs | Consent-free augmentation |
| IDiff-Face | 500K / 10K IDs | Diffusion augmentation |
| Vec2Face | up to 15M / 300K IDs | SOTA synthetic |

---

## 5. Results So Far

### Baseline Inference (Pakistani Public Figures)

| Metric | Value |
|---|---|
| Gallery identities | 23 |
| Probe images | 437 |
| Matched | 324 |
| Accuracy | **74.1%** |
| Threshold | 0.4 (untuned) |

### Mini Test (5 identities)

| Metric | Value |
|---|---|
| Matched | 5/10 |
| Notes | Pipeline verified working |

---

## 6. Experiments Planned / In Progress

| Experiment | Purpose | Status |
|---|---|---|
| Threshold tuning | Optimize 0.30–0.50 | ❌ Not started |
| Ablation: no quality gate | Measure quality filter contribution | ❌ Not started |
| Ablation: no alignment | Measure alignment contribution | ❌ Not started |
| Ablation: no detector | Measure detector contribution | ❌ Not started |
| Ablation: flip TTA | Measure test-time augmentation | ❌ Not started |
| Ablation: fine-tuned model | Measure fine-tuning gain | ⏳ Queued on Kaggle |
| Per-group metrics | Fairness reporting | ❌ Not started |
| Leakage audit | Train/eval identity overlap check | ❌ Not started |

---

## 7. Fine-Tuning Strategy

**Primary method:** AdaFace IR-101 (`adaface_ir101_webface12m.ckpt`)
- Loss: AdaFace margin (`m=0.4, h=0.333, s=64`)
- Optimizer: AdamW, backbone LR 1e-4, head LR 1e-3
- Schedule: Cosine annealing, 20 epochs, batch 128
- Augmentation: flip, ±10° rotation, downscale→upsample, blur, JPEG (no chroma subsampling), mild brightness/contrast
- Mix: 70% WebFace4M + 30% local data
- Freeze stem + early stages; train last stage + BN + head

**Fallback:** PETALface LoRA (rank 8) if AdaFace weights unavailable

**Current notebook:** Kaggle `face-recognition-adaface-training-techfest` (ResNet-50 stand-in, downloads LFW automatically)

---

## 8. Key Papers & References

| Paper | Year | Key Insight |
|---|---|---|
| ArcFace | 2019 | Additive angular margin; 112×112 template standard |
| AdaFace | 2022 | Quality-adaptive margin; SOTA low-quality |
| SCRFD | 2021 | Sample/computation redistribution; efficient detection |
| CR-FIQA | 2023 | Face image quality via relative classifiability |
| RFW | 2019 | Racial Faces in the Wild — bias benchmark |
| FARFACE | 2024 | Global South FR bias (includes Pakistan) |
| FaceMoE | 2026 | MoE for LR/surveillance FR |
| PETALface | 2025 | Parameter-efficient FR adaptation via LoRA |
| MFRC-21 | 2021 | Masked FR challenge winners (ArcFace + mask aug) |
| G2D | 2024 | Generative-to-discriminative masked FR |
| DeSCon | 2026 | Contrastive debiasing for FR fairness |

---

## 9. Key Findings from Research

1. **No public Pakistani FR dataset exists.** Closest: RFW Indian, BUPT-Balancedface Indian.
2. **South Asian faces have documented craniofacial differences** (long face, narrow nose, distinct palpebral fissure angle) — but this is population-level, not per-person.
3. **Face super-resolution hurts FR** (2025–26 consensus): GFPGAN/CodeFormer reduce identity similarity. Don't restore before embedding.
4. **Adaptive score normalization** (s-norm) helps across cameras/qualities — easy win.
5. **MFRC-21 winners used ArcFace + synthetic mask augmentation** — proven recipe for occlusion.
6. **Fitzpatrick17k is dermatology, not FR** — use Monk scale for skin-tone fairness.
7. **NIST: South Asian is high-FMR group** — fine-tuning on local data is essential.
8. **JPEG chroma subsampling hurts darker skin tones** — disable it in augmentation.
9. **MediaPipe BlazeFace is short-range (≤2m)** — not suitable for CCTV; use SCRFD instead.
10. **Idea: use MediaPipe for pose/occlusion/liveness gating** as a sidecar, not for recognition.

---

## 10. Kaggle Notebook

- **URL:** https://www.kaggle.com/code/danyalabbas1/face-recognition-adaface-training-techfest
- **Version:** 5
- **Machine:** T4 GPU
- **Data:** LFW (auto-downloaded) or WebFace4M (if uploaded)
- **Model:** ResNet-50 stand-in (to be replaced with AdaFace IR-50)
- **Output:** `adaface_ir50_adapted.pth` + `adaface_ir50_adapted.onnx`
- **Status:** Queued (as of Oct 19)

---

## 11. Local Environment

- **Python:** 3.13 (uv venv)
- **Packages:** insightface 2.1, opencv 5.0.0, numpy 2.5.4, sklearn 1.9.1, onnxruntime 1.31.0
- **Disk:** 53 GB free
- **Scripts:** `inference.py`, `eval.py`, `leakage_audit.py`, `kaggle_train.py`

---

## 12. Files in Repo

| File | Purpose |
|---|---|
| `inference.py` | End-to-end pipeline (CPU) |
| `eval.py` | TAR@FAR, EER, Rank-1, per-group metrics |
| `leakage_audit.py` | Train/eval identity overlap check |
| `kaggle_train.py` | Kaggle GPU training (paste into notebook) |
| `requirements.txt` | Dependencies (uv-compatible) |
| `README.md` | Quick start |

---

## 13. Known Issues & Fixes

| Issue | Fix |
|---|---|
| `quality_ok` received `img.shape` instead of `img` | Changed to pass full image array |
| `sorted(PosixPath)` TypeError in `main()` | Use `rglob('*.jpg')` instead of `rglob('*')` + filter |
| `!pip install` syntax error in Kaggle script | Use `uv pip install` or remove; Kaggle has packages |
| `FaceDataset` map object not subscriptable | Wrap `glob()` with `list()` |

---

## 14. Next Steps (Priority Order)

1. **Threshold tuning** — sweep 0.30–0.50, pick best
2. **Ablation table** — run variants, measure contribution
3. **Per-group metrics** — fairness reporting
4. **Wait for Kaggle GPU** — fine-tune, download ONNX
5. **Re-run inference** with fine-tuned model
6. **Report skeleton** — write structure, fill placeholders
7. **Slides** — 10-12 slides with placeholder numbers
8. **Leakage audit** — document no overlap
9. **Submit** — Oct 22

---

## 15. Report Outline (Draft)

1. Title + Team + Abstract
2. Problem & Operational Context (Pakistani deployment)
3. Task Definition (1:1 vs 1:N, open vs closed set)
4. Data — Sources, Licenses, Consent, Demographics, Splits
5. Pipeline — Detector, Alignment, Quality, Embedder, Matching, Normalization
6. Evaluation Setup — Protocol, Metrics, CIs, Hidden Test Assumptions
7. Main Results — TAR@FAR, EER, Rank-1, ROC/DET, CMC
8. Per-Group Performance — Gender, Age, Skin Tone, Occlusion, Quality
9. Ablations — Cumulative + Leave-one-out
10. Error Analysis — Failure taxonomy with examples
11. Robustness & Limitations
12. Ethics, Consent, Privacy
13. Reproducibility — Code, Seeds, Configs, ONNX
14. References

---

## 16. Presentation Outline (Draft)

1. Problem + Pakistani context
2. Data + consent + splits
3. Pipeline diagram
4. Protocol + metric definitions
5. Headline results + CI
6. ROC/DET + CMC
7. Per-group fairness
8. Ablations
9. Failure cases
10. Limitations + future work
11. Takeaways

---

## 17. Open Questions

- [ ] What threshold maximizes accuracy on Pakistani Public Figures?
- [ ] Does fine-tuning on LFW/WebFace4M improve Pakistani faces?
- [ ] Should we use AdaFace or ArcFace as the primary embedding model?
- [ ] How to get real Pakistani training data (consent issues)?
- [ ] What is the hidden benchmark protocol (1:1 or 1:N)?
- [ ] Can we use MediaPipe as a quality/pose sidecar?
- [ ] Should we add anti-spoofing (MiniFASNet)?

---

## 18. Log

| Date | Action |
|---|---|
| Oct 19 | Research phase: 10 parallel agents surveyed 2024–2026 FR landscape |
| Oct 19 | Pipeline built: inference.py, eval.py, leakage_audit.py |
| Oct 19 | Baseline inference: 74.1% on Pakistani Public Figures (23 IDs, 437 probes) |
| Oct 19 | Kaggle notebook created (v5, T4 GPU, auto-downloads LFW) |
| Oct 19 | Kaggle session queued (waiting for GPU slot) |
| Oct 19 | Disk check: 53 GB free, venv OK, packages installed |

---

## 19. Lessons Learned

1. **Verify every citation.** Several arXiv IDs from agents were hallucinated or misattributed.
2. **Kaggle script mode doesn't support `!pip install`.** Use `uv pip install` or rely on pre-installed packages.
3. **Kaggle GPU sessions queue.** Plan CPU work in parallel.
4. **No public Pakistani FR dataset.** Indian proxy (RFW) is the best available.
5. **Super-resolution hurts FR.** Don't restore faces before embedding.
6. **Identity-disjoint splits are non-negotiable.** Leakage = optimistic bias = bad ranking.
7. **Per-group reporting is where reports lose points.** Do it even if numbers are mediocre.
8. **Ablations are explicitly scored.** Don't skip them.
9. **JPEG chroma subsampling hurts darker skin.** Disable it in augmentation.
10. **Start the report early.** Biggest time sink on competition day.

---

*This document is a living log. Update after every experiment, decision, or result.*
"""
## 2026-10-20 — Failure-to-enroll root cause: tight crops, not blur

**Symptom.** Baseline 1:N on pak_fig: 319/328 correctly matched among *detected*
probes (97.3%), but only 319/437 end-to-end (73.0%). 109 probes had no face.

**Diagnosis** (`diagnose_fte.py`, all 437 probes):
- 109/437 produced **zero** detections (not low-score: literally no box).
- 2x image upscale rescued **0/109** — because `FaceAnalysis.prepare(det_size=640)`
  already resamples internally, so manual upscaling is a no-op.
- Failures spread across 20/23 identities; no correlation with blur or image size
  (median 256x303 px).
- Visual inspection: they are **tight face crops** (e.g. alamy headshots) where the
  face fills the frame and is cut at the edges. SCRFD is trained on scenes and
  needs context around the face.

**Fix.** If the first detection pass returns nothing, retry on a copy padded with a
30% reflected border (`cv2.BORDER_REFLECT_101`), then align from the padded image.
Measured rescue rate on the 109 failures:

| pad | rescued | median det score |
|---|---|---|
| 0.15 | 109/109 | 0.774 |
| 0.30 | 109/109 | 0.835 |
| 0.50 | 109/109 | 0.830 |

Chose 0.30. Implemented in `evaluate_pak.Embedder._detect` and `inference.get_faces`.
Quality blur is computed with the synthetic border clipped out, so the gate still
sees real pixels.

**Implication for the report.** FTE was a *detector framing* artefact, not an image
quality problem. This is a deployment lesson: enrolment pipelines must handle both
scene photos and pre-cropped ID-style inputs.

## 2026-10-20 — Anti-spoofing (PAD): picking a model

**Requirement.** The pipeline must answer "is this a live face or a print/screen
replay?" before it answers "who is this?". No PAD anywhere in the current stack.

**Candidate A — Hub ONNX export (`garciafido/minifasnet-v2-anti-spoofing-onnx`).**
Apache-2.0, 1.7 MB, 80x80 BGR /255 in, 3-class softmax out. Downloaded and
benchmarked: it returns a near-constant `[0.00, 0.006, 0.994]` for *every* input —
real faces, random noise, every crop scale. A model whose prediction is invariant
to its input carries no signal, so this export was rejected.

**Candidate B — rebuild from upstream weights.** Cloned `MiniFASNet.py` from
minivision-ai/Silent-Face-Anti-Spoofing, loaded
`2.7_80x80_MiniFASNetV2.pth` (SHA-256 verified against the published digest,
`a5eb02e1...`), exported to ONNX ourselves (`convert_pad.py`). Parity with
PyTorch: 1.1e-8. The rebuilt model still returns a strong constant-ish bias
(logits ≈ [-3.7, -0.7, 4.4]) on real faces, which means the HF model was **not**
broken — the class order is. Working hypothesis: index 2 is *live*, not replay.

**Decision.** Determine the label mapping empirically on a labelled spoof set
(NUAA print attacks, Apache-2.0) rather than trusting the model card. Report
BPCER on our real probes and APCER on NUAA, and state clearly that pak_fig
contains no attacks, so any APCER number is from a different domain.

## 2026-10-20 — Kaggle fine-tune (v10) result

ResNet-50 (ImageNet) + AdaFace head, LFW, 1,512 train / 168 held-out identities,
12 epochs, ~150 s on GPU.

| Held-out LFW (168 unseen IDs) | pretrained | fine-tuned |
|---|---|---|
| Rank-1 | 9.6% | 41.7% |
| TAR@FAR=1e-3 | 5.2% | 27.6% |
| EER | 26.1% | 13.3% |

ONNX export parity = 1.000000 cosine. The fine-tune is a large *relative* gain
and still ~55 points below `buffalo_l`. Conclusion: with two days and 1.5k
identities, fine-tuning cannot beat a 600k-identity pretrained embedder — the
pre-registered promotion rule correctly rejects it. This is the evidence for the
"data scale, not architecture, is the constraint" claim in the report.

## 2026-10-20 — Anti-spoofing, resolved

Root cause of the constant PAD output: **preprocessing scale**. With input
scaled to [0,1] (the path used by upstream's reference `ToTensor()` code) both
MiniFASNet checkpoints saturate to a constant class for every input. With raw
0–255 BGR the models behave sensibly and discriminate strongly:

- NUAA print attacks vs real, held-out slice (150 + 150):
  - class order is [print, live, replay]
  - mean live score: real 0.80, attack 0.001
  - threshold sweep: 0.02 -> BPCER 1.3% / APCER 1.3%; 0.05 -> 4.0% / 0.7%;
    0.10 -> 7.3% / 0.0%
- Operating point chosen: 0.05.

This also means the third-party model card's stated order ([live, print, replay])
is wrong, and its export was unusable. Everything was re-derived from the
official .pth files (SHA-256 verified) and validated on labelled data.

## 2026-10-20 — Full baseline with detection fix: FTE 0, end-to-end 97.94%

First full 1,286-image pass with the reflected-border retry *and* the PAD gate,
batched recognition, and crop caching.

| | before fix | after fix |
|---|---|---|
| probes enrolled | 328/437 (75.1%) | 437/437 (100%) |
| rank-1 (enrolled) | 97.3% | 97.94% |
| **end-to-end** | **73.0%** | **97.94%** [96.57, 99.08] |
| TAR@FAR=1e-3 | — | 97.03% |
| EER | — | 1.14% |
| AUC | — | 0.9947 |
| PAD BPCER @0.05 | — | 4.35% |

The PAD BPCER on our own probes (4.35%) matches the NUAA estimate (4.0%), which
is a good sign that the liveness gate transfers across domains rather than
fitting one dataset.

Worst identity: `hamza_shehbaz`, rank-1 0.583 (12 probes), mean genuine 0.642.
Nineteen of 23 identities are at 100%.

Operational note: detection with PAD costs ~620 ms/image on this CPU, so the
harness caches aligned crops separately from embeddings; model swaps now cost
only the embedding pass (~130 s for the whole benchmark with R50).

**Cost of enforcing liveness.** On our probes the gate rejects some real faces,
which can only cost accuracy on an attack-free benchmark:

| PAD threshold | probes answered | end-to-end (gated) | BPCER (our probes) |
|---|---|---|---|
| off | 437/437 | 97.94% | — |
| 0.02 | 427/437 | 95.65% | 2.29% |
| 0.05 | 418/437 | 93.59% | 4.35% |
| 0.10 | 400/437 | 89.47% | 8.47% |

Decision: report 97.94% as the recognition figure (benchmark has no attacks) and
95.65% as the deployment figure with liveness enabled at 0.02. Shipping 0.02 as
the default because it costs only 2.3% of genuine users and still blocks 98.7% of
print attacks.

## 2026-10-20 — Veiled-face datasets: what exists and what does not

Requirement: test niqab / hijab / face-cover robustness. Survey of public data.

| Source | Size | Labels | Verdict |
|---|---|---|---|
| `ethicalstar/niqabi-ds` (Kaggle, MIT) | 532 imgs | YOLO face boxes | **used** — real niqab, no identity |
| `mostafaebrahiem/women-faces-with-hijab` (Kaggle) | 112 imgs | none | too small, unlabelled |
| **`Niqab Dataset for Occluded Face Detection`** (Zenodo 17011207, CC-BY-4.0) | ~10k imgs / 12k faces, >50% covered | face boxes | **used** — the dedicated niqab detection benchmark |
| `akashguna/lfw-dataset-with-masks` (Kaggle) | 0.2 GB | LFW identities + synthetic masks | identity-labelled lower-face occlusion |
| `trainingdatapro/asian-kyc-photo-dataset` (Kaggle) | 0.23 GB | ID + selfie per identity | Asian KYC faces, identity-labelled |
| `nanimasaka/masked-face-recognition-dataset` | 3.4 GB | `pN/masked/*.jpg` identities | real surgical masks, identity-labelled |
| AR Face Database (Ohio State) | 126 subj, 2 sessions | identity + scarf/sunglasses | request-only, not downloadable |
| CAS-PEAL-R1 | 1040 subj | identity + disguise | request-only |
| RMFRD (Wuhan Univ.) | 525 ids | identity, real masks | GitHub, large |

**Key negative result:** there is *no* public identity-labelled Pakistani
niqab/hijab recognition benchmark. Every veiled dataset with identities uses
surgical masks, not Islamic face coverings; every niqab dataset is
detection-only. So recognition accuracy on niqab cannot be measured against
ground truth from public data — only (a) enrollment rate, (b) controlled
occlusion with identities known, (c) proxy datasets with masks.

That is why the evaluation is split in two:
1. **`veiled_eval.py`** — enrollment funnel on real niqab/hijab imagery:
   detected -> quality gate -> liveness -> template, with YOLO-box detection
   recall, stratified by face size so "small in a group photo" is not confused
   with "detector cannot see a covered face".
2. **`occlusion_stress.py`** — identity held fixed, occlusion varied. Gives a
   clean degradation curve for hijab / light niqab / full niqab / eyes-only.

## 2026-10-20 — Embedder A/B (buffalo_l vs antelopev2), identical pipeline

Same crops (cached), same PAD gate, same alignment, same aggregation, 437 probes.

| model | rank-1 | TAR@1e-3 | TAR@1e-4 | EER | AUC | genuine | impostor | gap |
|---|---|---|---|---|---|---|---|---|
| buffalo_l R50 / WebFace600K | 97.94% | 97.03% | 29.29% | 1.14% | 0.9947 | 0.792 | 0.059 | 0.733 |
| antelopev2 R100 / Glint360K | **98.17%** | 97.25% | 34.78% | 1.13% | 0.9953 | 0.810 | 0.054 | **0.756** |

Paired comparison: antelope wins exactly **1** probe and loses 0.
Delta +0.23 pts, paired-bootstrap 95% CI [+0.00, +0.69], McNemar exact p = 1.0.

Conclusion: **statistically indistinguishable** on 437 probes. R100 does show a
consistently better score geometry (bigger genuine-impostor gap, +5.5 pts at
FAR=1e-4), which is the kind of thing that matters when the operating point is
pushed, but the headline rank-1 difference is one image and must not be sold as
an improvement. If we ship R100 it is for margin, not for a higher score.

## 2026-10-20 — Detection bug: SCRFD misses the *large* face (not just tight crops)

Chasing the 8 probes that every model got wrong produced the most important
finding of the day. Two of them are not recognition failures at all — the
detector returned the wrong face.

Case study, `asif_ali_zardari_..._4d6d.jpg` (291x330): the photo is a clear
frontal portrait with a small framed portrait of Benazir Bhutto in the
background. At `det_size=640` SCRFD returned **only the 59x74 background
portrait** (score 0.71) and missed the 195x224 foreground face entirely.
Quality gate flagged `low_iod_28`, fallback embedded the portrait, and the
probe was scored against the wrong person's face.

Cause: the 291 px image is upscaled 2.2x to 640, so the foreground face becomes
~430 px — past SCRFD-10GF's comfortable face-size range. The same image at
`det_size=320` detects the real face at 0.838. A reflected-border pass at 640
also finds it (0.854), because padding changes the effective scale.

So the single "retry when nothing is found" rule was half the fix. The second
half is: **retry when everything found is tiny**. Implemented as a second
escalation in both `evaluate_pak.Embedder._detect` and `inference.py`:

    nothing detected            -> padded retry at 640          (existing)
    only tiny faces (IOD<64 or  -> coarse pass at 320 + NMS      (new)
      largest face <2% of frame)

Both failing probes are now detected correctly (IOD 89 and 83). This also
explains why `nawaz_sharif_..._f60fcab8`, previously written off as a blurry
label error, was really a missed detection: the real face is found at 0.80.

Generalisation risk this closes: any probe with a small face in the background
and a large face in the foreground — extremely common in scraped news photos,
and the exact shape a hidden benchmark might take.

## 2026-10-20 — Niqab enrolment on real data (532 photos)

`ethicalstar/niqabi-ds`, 532 real niqab photos (eye-slit niqab, full veil, group
shots, full-body street photos).

| metric | native | + caught retry |
|---|---|---|
| images with >=1 detected face | 78.4 % | **82.1 %** |
| face large enough to enrol (IOD>=64 + blur + yaw) | 21.4 % | **22.0 %** |
| detected faces judged live | 98.3 % | **97.9 %** |
| median IOD | 50 px | 50 px |

Two conclusions:

1. **The liveness gate does not punish veils.** 98 % of niqab faces pass PAD, which
   is the same rate as our unveiled probes (95.7 %). A gate that rejected niqab
   would be a fairness bug; it does not.
2. **The enrolment limit here is subject distance, not the veil.** Median IOD is
   50 px — most of these are full-body street/social photos, and our gate requires
   64 px between the eyes. Of the faces detected, ~27 % yield a template. The
   detector itself finds a face in 82 % of images, which is the number that
   matters for "can it see a veiled face".

Dataset caveat: the shipped YOLO labels are **person boxes, not face boxes**
(verified visually: box spans head to feet, our detection lands on the face).
So detection recall cannot be computed from them; `veiled_eval.py` reports it as
0 by construction. The Zenodo OFD set, which has true face boxes, is used for
recall instead.

## 2026-10-20 — Final v2 results (corrected detector)

| model | rank-1 | TAR@1e-3 | TAR@1e-4 | EER | AUC | impostor mean |
|---|---|---|---|---|---|---|
| buffalo_l R50 | 98.63 % (431/437) | 97.71 | 29.29 | 0.69 | 0.9995 | 0.0594 |
| antelopev2 R100 | **98.86 % (432/437)** | 97.94 | 35.24 | **0.46** | 0.9996 | 0.0542 |
| AdaFace IR-101 | (pending) | | | | | |

Paired comparison buffalo vs antelope on the same 437 probes: 431 both correct,
0 buffalo-only, 1 antelope-only. Exact McNemar p = 1.000, bootstrap delta
+0.23 pts [+0.00, +0.69]. A statistical tie whose only difference is one probe;
antelope costs ~4x the CPU time, so the shipping default stays R50 and R100 is
offered as a flag.

Threshold 0.40 is confirmed as the operating point: FRR 0.69 %, FAR 0.48 %.
Dropping to 0.35 lowers FRR to 0.46 % but doubles FAR to 0.89 %.

Near-duplicate audit on the v2 caches: 2/437 probes have same-identity cosine
>= 0.985 to a gallery image. Excluding both correct probes gives 98.62 % vs
98.63 % as reported, so the headline is not leakage-driven.

## 2026-10-20 — Self-collected corpus for the compliance fine-tune

The brief requires teams to collect their own data and fine-tune; the LFW
ablation (§6.2) does not satisfy that. Corpus collected tonight: **99 Pakistani
public figures, 4,203 web images**, 45 images per identity from logged-out Bing
search, disjoint from the 23 evaluation identities (checked before download and
again before training).

Identity mix: 20 female (including labelled hijab/dupatta identities), 79 male;
politicians, cricketers, actors, comedians, clerics, journalists. Provenance in
`data/raw/train_faces/sources.csv`.

Backbone decision: AdaFace **IR-50 MS1MV2** rather than buffalo_l. The public
checkpoint is a Lightning file whose key layout (`input_layer`/`body.N.res_layer`
/`output_layer`) differs from arcface_torch; the block applies stride on the
second conv and uses a parameter-free MaxPool shortcut when widths match.
Reconstructed from the released tensors, loaded with `strict=True` (467 tensors)
and validated on the benchmark: **98.40 %** vs 98.63 % for the official ONNX —
one probe apart, which is the check that the architecture and preprocessing are
right, not a result in itself.

**Pre-registered promotion rule (written before the fine-tune ran):** the
fine-tuned model replaces buffalo_l in the submission only if, on the 437-probe
benchmark, it (a) does not lose rank-1 and (b) does not lose TAR@1e-3. A tie goes
to the fine-tuned model, because it is the artefact the brief asks for and its
domain adaptation is the point of the exercise; a loss of even one probe on
either metric keeps the stock model and the fine-tune is reported as a negative
result. The 98.40 % re-implementation is therefore the floor, and the run is
only worth shipping if it clears 98.63 % or at least holds TAR.

## 2026-10-21 — Remote history incident and review-agent fixes

At 20:26 UTC on 20 Oct the remote `main` was replaced by a new root commit
(`8ab18f16`, three commits) rather than updated: the history ending at `c32c431`
disappeared from GitHub, and with it the report (59 KB → 709 B), the research
log, the guide, `results/` (55 files), `app/`, and `experiments/`. The
replacement also committed `__pycache__` files and reverted `pyproject.toml` to
unpinned dependencies without the `demo`/`dev` extras.

The code changes in that commit were reviewed line by line and kept, as one
commit on the restored history:

- `evaluate_pak.py`: genuine score is the max over matching gallery columns
  (was the first), sentinel `-1.0` rows are filtered before metrics and ROC,
  `ValueError` instead of `assert`.
- `inference.py`: `_asset()` raises with every searched path, gallery directory
  check, `EPSILON` constant.
- `imgio.py`: PIL failure logged before the cv2 fallback.
- `predict.py`: forward-slash output keys, gallery root check.
- `selftest` / `fuse_eval` / `summarise` / `pad_figure`: context-managed reads.

Measured effect: none. Every saved run has one gallery vector per identity and
no probe without a mate, so max-vs-first and sentinel filtering are no-ops.
Re-running the baseline eval after the patch reproduces rank-1 98.63 %
(431/437), TAR@1e-3 97.71, EER 0.69 %, AUC 0.9995 and F1 0.9466 to the last
digit.

Recovery: the replacement commits are preserved as branch `review-fixes`; `main`
is restored to the full history with the fixes applied on top.

## 2026-10-21 — Compliance fine-tune result (negative)

Kaggle now mounts datasets at `/kaggle/input/datasets/<owner>/<slug>/`, not
`/kaggle/input/<slug>/`; two runs died on the old assumption before the resolver
searched by name. The third run trained all 40 epochs on a T4 (2,034 train /
196 held-out crops, 10–11 s/epoch), reached 100 % on the held-out images, and
exported a 167 MB ONNX. The kernel reports ERROR only because its final parity
check imports `onnxruntime`, which the image does not ship; the export had
already completed and downloaded. The script now saves the state dict before
export and skips parity when the module is absent.

Benchmark result (437 probes, same crops as every other run):

| Model | Rank-1 | TAR@1e-3 | EER | Genuine mean |
|---|---|---|---|---|
| buffalo_l R50 | 98.63 % (431) | 97.71 | 0.69 % | 0.791 |
| AdaFace IR-50 + corpus fine-tune | 98.17 % (429) | 95.88 | 1.37 % | 0.765 |

The pre-registered rule required no loss on either rank-1 or TAR@1e-3, so the
fine-tune is **not promoted** and buffalo_l ships. Both lost probes are in the
`low` quality bucket (22/24 vs 24/24); every other quality bucket is unchanged.
That is the expected failure: the corpus is clean and aligned, while the
benchmark's residual errors are blurred/tiny captures that the MS1MV2 prior
already handles. F1 at 0.40 rose slightly (0.9508 vs 0.9466) because the
genuine–impostor gap narrowed; it is a threshold artefact, not a promotion case.

## 2026-10-21 — Shipped-package verification

`submit/predict.py` was run end to end on the same 437 probes:
**430/437 = 98.40 %** at the 0.40 threshold (`results/final_predictions.csv`),
two probes below threshold and five wrong-identity. The harness reports 429/437
at the same threshold. The one-probe gap is a harness artefact: for a face lying
partly outside a reflected-border retry, `evaluate_pak.quality` clips the box to
empty and skips the blur test, so it accepts a face that `inference.quality_ok`
rejects; the shipped path then salvages the larger face (score 0.75 vs 0.35).
The harness numbers are left unchanged across all ablations for comparability;
the report now states both, and the shipped figure is the deliverable's number.

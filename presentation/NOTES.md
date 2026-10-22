## Team Mamdani — speaking notes

**Deck:** `PRESENTATION.pdf` (16 slides, 16:9). **Source:** `presentation/slides.md`.
**Rebuild:** `.venv/bin/python presentation/build.py` (needs pandoc + Chrome).

Total run time if every slide gets its full time: ~14 minutes. If you are held to
10, cut slides 11 (fairness) and 14 (negatives) to one sentence each and keep 6,
9 and 13 — those carry the argument.

| # | Slide | Time | One-line purpose |
|---|---|---|---|
| 1 | Title | 0:20 | Who, what, when |
| 2 | The Brief | 0:40 | Show we answered the actual brief |
| 3 | Headline | 0:50 | The number, stated once, cleanly |
| 4 | Data | 0:50 | Evaluation vs training, no overlap |
| 5 | Pipeline | 0:50 | Detection → align → quality → embed → match |
| 6 | Detection | 1:30 | **The hero result** |
| 7 | Embedders | 1:00 | Four models, one result; why R50 ships |
| 8 | Operating point | 0:50 | FAR/FRR and the threshold |
| 9 | Veiled faces | 1:00 | Ranking survives, calibration doesn't |
| 10 | Real niqab | 0:45 | Bottleneck is face size, not the model |
| 11 | Fairness | 0:50 | Report what the data supports |
| 12 | Liveness | 0:50 | Included, measured, cost stated |
| 13 | Fine-tune | 1:20 | **The negative result, owned** |
| 14 | Failures | 0:40 | What didn't work and what's next |
| 15 | What ships | 0:40 | The package and the demo |
| 16 | Takeaway | 0:30 | Close on the thesis |

---

### 1 · Title (0:20)

> "We built an end-to-end face recognition pipeline for Pakistani faces. The
> short version: 98.6% rank-1 on 23 identities, anti-spoofing included, and a
> fine-tune we ran, measured, and rejected. I'll spend most of the time on the
> three decisions that actually moved the number."

Do not read the slide. Move on.

### 2 · The Brief (0:40)

> "The brief asked for five things: a working system, our own data plus a
> fine-tune, the full metric set, per-group fairness, and anti-spoofing. We did
> all five. The finding that organises the rest of this talk is that the
> embedder was never the bottleneck."

### 3 · Headline (0:50)

> "98.63% rank-1: 431 of 437 probes, all 437 enrolled. EER 0.69%, AUC 0.9995.
> At the shipped threshold of 0.40, the packaged CLI returns 430 of 437, so
> 98.40% end to end. The two numbers differ by one probe and I'll explain why
> later — it's a harness artefact, not a different system."

If asked "which number do you stand behind?" → **98.63% is threshold-free top-1
after detection; 98.40% is the end-to-end number with the 0.40 decision.** The
report uses both and labels them.

### 4 · Data (0:50)

> "Evaluation is 23 Pakistani public figures, 1,286 images, split 849 gallery and
> 437 probes. Training is a corpus we collected ourselves: 99 identities, 4,203
> images, cleaned down to 98 identities and 2,230 crops. No identity appears in
> both. We also audited near-duplicates: two probes, and removing both leaves
> 98.62%, so the headline isn't leakage."

Mention the other sets only if asked: LFW for the failed fine-tune, masked-LFW
for occlusion, NUAA for liveness, niqab sets for enrolment-only measurement.

### 5 · Pipeline (0:50)

> "Detection, then alignment, then quality, then embedding, then cosine matching
> against the identity mean. Two non-standard things: if SCRFD finds nothing we
> retry on a reflected-border canvas, and if the face is tiny we run a second
> 320-pixel pass and keep the subject rather than the background face. Liveness
> sits in front of the matcher when enabled."

Point at the two escalation boxes while saying it. That is slide 6's setup.

### 6 · Detection — the hero slide (1:30)

> "This is the result I'd defend hardest. Native SCRFD enrolled only 328 of 437
> probes — 109 were tight headshots where it found zero faces. Every one of those
> 109 was recovered by the border retry: 73.0 to 97.94. Then two probes were
> silently matching the wrong face because a background face was larger than the
> subject; the 320-pixel second pass fixed both, and that's the last 0.7 points.
> Swapping in a four-times-larger embedder moved the score by one probe."

Pause on "109 out of 109". If interrupted, the point is: **failures were
detection failures, not recognition failures.**

### 7 · Four embedders (1:00)

> "Buffalo R50, antelope R100, AdaFace IR-101, IR-50 — all within one probe.
> The six errors are nested: no model succeeds where another fails, so fusion
> can't help, and it doesn't — 98.86%, same as the best single model. We ship
> R50 because it ties on rank-1, is 184 MB of the 435 MB bundle, and is about
> three times faster on CPU. IR-101 is the better verification model if the
> metric were TAR at a fixed FAR."

### 8 · Operating point (0:50)

> "At 0.40, FRR is 0.69% and FAR is 0.48%. Moving to 0.35 halves false rejects
> but nearly doubles false accepts. The ROC is the honest picture: the curve is
> vertical until about 1e-3, then flattens — that flat part is the six nested
> errors."

### 9 · Veiled faces (1:00)

> "We tested this two ways. Controlled: we painted hijab, niqab and eye-slit
> occluders onto aligned crops. The hijab is free — 97.94%, identical to clean —
> because none of the five landmarks are covered. Light niqab drops to 96.6,
> full to 92, extreme to 89. Ranking survives; the score distribution doesn't.
> At a fixed 0.40 those probes come back `unknown`. Correct ranking, wrong
> calibration."

Masked-LFW number if needed: 98.25 → 97.25 rank-1, but accepted-at-0.40 falls
97% → 85%. That is the same lesson on a second dataset.

### 10 · Real niqab (0:45)

> "On 532 real niqab photos, 82% are detected and only 22% pass our strict
> quality gate. Median inter-ocular distance is 48 pixels against a 64-pixel
> gate. But 97.9% pass liveness. So the failure is face size, not the recogniser
> and not the anti-spoofing. These images have no identity labels, so we never
> claim recognition accuracy on them."

### 11 · Fairness (0:50)

> "Twenty male identities score 98.41%; three female identities score 100%. That
> is three identities — a check, not a fairness claim, and we say so. An
> attribute model labelled 13 of 23 identities female, including bearded men, so
> per-probe gender labels would have been noise. Age and skin tone we declare
> unreportable rather than estimate."

### 12 · Liveness (0:50)

> "MiniFASNet v2 plus v1se. On NUAA, 1.33% APCER and 1.33% BPCER at 0.02. The
> UI gate is 0.05, where 4.35% of our real probes are rejected: 94.28% end to
> end, but 98.56% among accepted. The cost is rejection, not misidentification."

### 13 · Fine-tune — the negative (1:20)

> "We collected 98 identities and 2,230 aligned crops, painted synthetic veils
> on them, and fine-tuned AdaFace IR-50 on a Kaggle T4 for 40 epochs. It fit the
> held-out split 100% — that's the overfitting warning, not a result. On the
> benchmark: 98.17 against 98.63, and TAR at 1e-3 fell from 97.71 to 95.88. Both
> lost probes are in the low-quality bucket, 22 of 24 against 24 of 24. We had
> pre-registered the rule: promote only if neither rank-1 nor TAR drops. It
> failed, so stock buffalo ships."

Then the sentence that earns credibility:

> "Two thousand clean crops can't teach a model what MS1MV2 already knows about
> blur and small faces."

### 14 · Failures (0:40)

Fast, no dwelling: LFW fine-tune 41.7 vs 97.9; fusion tied; bigger models tied;
upscaling rescued nothing; masked threshold needs recalibration. Next steps are
per-condition normalisation and more labelled veiled data.

### 15 · What ships (0:40)

> "The repo runs in place. `submit/predict.py` handles nested 1:N, flat gallery
> with an index, and 1:1 pairs; one row out per image in, including unknown and
> read errors. Weights are vendored, no network. Selftest passes. 435 MB, about
> 0.9 seconds per image on eight CPU threads. The demo uses the same inference
> code, so what you see is what was scored."

### 16 · Takeaway (0:30)

> "Detection got us to 98.6%. The next problem is calibration, not capacity. We
> report the negatives, the low-power groups and the one-probe differences,
> because a benchmark you can't interrogate isn't evidence. Thank you."

---

## Demo script (5 minutes, only if asked)

Start it before the talk so the gallery cache is warm:

```bash
uv run --extra demo python app/demo.py
# http://127.0.0.1:7860  — Identify / Verify 1:1 / Gallery / How it works
```

First launch enrols the gallery and takes ~3 minutes on CPU; the cache under
`app/.cache/` makes later launches fast. Do **not** run `uv sync` while it is up.

1. **Identify — correct match.** Upload
   `data/pak_fig/probe/imran_khan/imran_khan_icrawler_1778786376249_eb1a3502.jpg`
   → shows the aligned crop, the top-5, quality and the detection that was used.
2. **Identify — hijab.** Upload
   `data/pak_fig/probe/hina_rabbani_khar/hina_rabbani_khar_icrawler_1778782327939_0b46549e.jpg`
   → correct, with liveness passing. This is the "hijab is free" point live.
3. **Identify — known failure.** Upload
   `data/pak_fig/probe/hamza_shehbaz/hamza_shehbaz_icrawler_1778784006032_3390ea89.jpg`
   → predicts Shehbaz Sharif. Say it before the UI does: father/son resemblance
   is the documented failure mode, and it is in the report.
4. **Verify 1:1.** Two images of the same identity → score and accept/reject at
   0.40.
5. **Gallery.** Enrol one new image and show the count change.

If the network or Gradio misbehaves, fall back to `results/final_predictions.csv`
and the deck. Do not debug the UI on stage.

---

## Numbers you must not misquote

| Claim | Correct value |
|---|---|
| Rank-1, threshold-free | **98.63%** (431/437) |
| End-to-end at 0.40, shipped CLI | **98.40%** (430/437) |
| EER / AUC | 0.69% / 0.9995 |
| TAR@1e-3 | 97.71% |
| Detection build-up | 73.0 → 97.94 → 98.63 |
| Probes rescued by border retry | **109/109** |
| Best single model | 98.86% (antelope R100 and AdaFace IR-101, tie) |
| Fusion of four | 98.86% (no gain) |
| Hijab, controlled | 97.94% (same as clean) |
| Extreme niqab, controlled | 89.02% rank-1 |
| Real niqab: detected / strict quality / liveness | 82.1% / 22.0% / 97.9% |
| PAD at 0.02 | 1.33% APCER / 1.33% BPCER |
| Fine-tune | 98.17% vs 98.63%; TAR 95.88 vs 97.71 → **rejected** |
| Runtime | ~0.9 s/image, 8 CPU threads |

**The headline is 98.63%, not 98.86%.** 98.86% belongs to the two models we did
not ship. If someone quotes 99%, correct them.

---

## Q&A prep

**"Why not use the bigger model? It scored higher."**
One probe higher, and the errors are nested — no model fixes a case another
misses, McNemar p = 1.0. R50 is 3× faster and half the bytes. If the metric were
TAR at 1e-3, IR-101 is the better choice and we say so in the report.

**"98.6% on 23 identities — is that real?"**
It is real for this benchmark and we bound it: 437 probes, 95% CI [97.5, 99.5],
and per-identity CMC in the report. It is not a claim about Pakistan at large;
the identities are public figures, and the corpus is small.

**"Did you leak training data into evaluation?"**
No. Identity-disjoint by name before collection and re-checked before training.
Near-duplicate audit found 2/437; removing both gives 98.62%.

**"Why did the fine-tune fail?"**
The corpus is 2,230 clean, mostly high-quality crops. The benchmark's remaining
errors are low-quality captures. MS1MV2 already covers that regime; our data
added nothing and cost a little. The rule was pre-registered, so we kept stock.

**"Why is 98.40 not 98.63?"**
98.63 is threshold-free top-1 after detection. 98.40 adds the 0.40 accept/reject
decision. The harness reports 98.17 because its quality function clips one padded
box to empty and skips the blur test; that's documented in the report, not
patched away.

**"Isn't the niqab result a fairness problem?"**
It's a calibration problem and an image-size problem. Ranking stays high; the
score shifts down. A deployment must recalibrate per condition or normalise, and
must not reuse the 0.40 threshold. We show the funnel so the claim is bounded.

**"Is the anti-spoofing real or a stub?"**
MiniFASNet v2 + v1se, evaluated on NUAA print attacks, with both APCER and BPCER
and the end-to-end rejection cost. It is a presentation-attack detector, not a
deepfake detector — we don't claim that.

**"What about privacy and consent?"**
Public figures, public images, research use; self-collected corpus is consented
and identity-disjoint; real niqab sets are unlabelled and used only for enrolment
statistics. No face is published in the repo beyond the public-figure benchmark.

**"What if the hidden benchmark is pairs, not 1:N?"**
`predict.py --pairs` is implemented and selftested. The pipeline is unchanged;
only the wrapper differs.

**"What would you do with another week?"**
Per-condition score normalisation, labelled veiled data (the bottleneck is
identity labels, not images), and the harness quality-clip fix so the ablation
table is exactly like-for-like.

**"Why 0.40?"**
It is the point where FRR and FAR are closest on this benchmark (0.69 vs 0.48)
and where the shipped CLI was verified. It is a deployment choice, not a
property of the model, and the report gives the full sweep.

# Results index

Every number in `REPORT.md` comes from one of these files. Each `eval_*.json`
has a sibling `*.scores.npz` holding the full probe×gallery score matrix, so
figures and thresholds can be recomputed without re-running the models.

## Current (v2 crops — corrected detector)

| File | What it is | Report |
|---|---|---|
| `eval_buffalo_v2.json` | **Shipped system**: buffalo_l R50, 98.63 % rank-1 | §6.1, §6.3–6.5, §8.1 |
| `eval_antelope_v2.json` | antelopev2 R100, 98.86 % | §6.5 |
| `eval_adaface_v2.json` | AdaFace IR-101 | §6.5 |
| `eval_adaface50_v2.json` | AdaFace IR-50 | §6.5 |
| `eval_lfw_masked.json` | Masked-LFW probes (lower face covered) | §7.4 |
| `eval_lfw_clean.json` | Masked-LFW clean control | §7.4 |
| `occlusion_stress.json` | Controlled hijab/niqab occlusion, identity fixed | §7.4 |
| `eval_gal_first.json`, `eval_gal_medoid.json` | Gallery aggregation ablations | §8.2 |
| `eval_nopad.json` | No border-retry (failure-to-enrol ablation) | §8.1 |
| `groups_buffalo_v2.json` | Per-probe gender/age/quality attributes | §7.1, §7.5 |
| `duplicate_audit.json` | Near-duplicate leakage audit | §6.1 |
| `veiled_niqab_kaggle.json` | Enrolment funnel, 532 real niqab photos | §7.4.2 |
| `veiled_niqab_ofd.json`, `veiled_niqab_ofd_retry.json` | OFD occluded faces (label-convention caveat) | §7.4.2 |
| `subgroup_hijab.json` | Hair-covered vs hair-visible | §7.4.1 |
| `pad_eval.json` | PAD on NUAA (150 real + 150 print attacks) | §4.4 |
| `fte_diagnosis.json` | Why 109 probes failed to enrol | §4.1 |
| `superseded/` | v1-crop runs, kept for the ablation narrative | §8.1 |
| `figures/` | ROC, CMC, per-identity, quality, threshold plots | §6.3–6.4 |
| `diagnostics/` | Contact sheets used for visual label checks | — |

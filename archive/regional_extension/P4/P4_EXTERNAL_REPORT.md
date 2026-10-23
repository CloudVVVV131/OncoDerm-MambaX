# P4 Derm7pt Fixed-Model External Evaluation

Status: **COMPLETE**. New training runs: **0**. All 15 original fixed bests evaluated; no prior-stage rerun.
Completed: 2026-10-16T05:48:28+08:00. Work directory: `${R52_PATH_12DFAC0CCEF0}`.
Execution window started 2026-10-16T05:27:36+08:00, deadline 2026-10-16T09:27:36+08:00.
GPU evaluation elapsed: 322.8 s; paired bootstrap elapsed: 19.3 s.

## 1. Scope and Use History

The user explicitly confirmed that Derm7pt was not used in current Frontiers training, auxiliary training, development selection, tuning, checkpoint selection, or method changes based on its model performance. This attestation applies only to M0/R_IMG/R_CLIN/XATTN/B2S in the current Frontiers line and is based on the stated training/selection/evaluation records, not download date.
The attestation is recorded in USAGE_HISTORY_CONFIRMATION.json; it does not prove absence of unknown cross-dataset patient/lesion overlap. No other research direction was inspected.

## 2. Cohort and Bounded Overlap Checks

Source: `${R52_PATH_CB55B27527EC}`; image root: `${R52_PATH_6A00E9BD0835}`.
Metadata SHA-256: `471199fb86bcc97cd1521bcc4255bb3f9c563e2ed0ad3ff836392d81a3ed9d3b`.
Candidate 1,011 rows: 252 melanoma and 759 non-melanoma. Included: 1011; melanoma: 252; excluded: 0; prevalence: 0.24925816 (24.9258%).
Only the derm reference in each row was used, with exact case-sensitive directory and extension spelling. FCL and FCl were not merged. All 20 original diagnoses are listed in DIAGNOSIS_DISTRIBUTION.csv. The six prespecified melanoma labels are positive; non-melanoma is not synonymous with benign. No HAM seven-class target remapping or external macro-F1 was performed.
The 9 same-row clinic/derm shared references were each retained once. The 2,013 physical photos were not counted as 2,013 dermoscopic cases.
Original-file SHA-256, unresized RGB-pixel SHA-256 and exact named IDs were compared with the already frozen current-direction HAM manifests (6,982 train, 1,513 validation, 1,520 test-report-only) and the fixed auxiliary 100 image manifest. Exact known training/development/auxiliary identities would have been excluded before predictions.
Observed match counts: `{}`; exclusions: `{}`.
Confirmed within-cohort same-image clusters: 0; resampling clusters: 1011.
There is no verified patient ID or reliable cross-case lesion linkage here. case_id was not treated as a patient identifier. No new perceptual near-duplicate search was performed. Exact matching cannot rule out cropped/recompressed duplicates or undocumented cross-dataset patients. The intervals are case-level, not verified patient-level.
Download-stage file inventory was reused and only the 1,011 selected derm files were checked. No redownload, extraction, whole-project audit or old auxiliary-ID investigation occurred.

## 3. Clinical Inputs and Preprocessing

The unchanged P3 22-dimensional vocabulary and order are in CLINICAL_MAPPING.json. Every age is 0 with age_missing=1. Sex is mapped directly from female/male; no age was inferred from notes, diagnoses or images.
Location: abdomen/acral/back/chest unchanged; lower limbs to lower extremity; upper limbs to upper extremity; genital areas to genital; head neck and buttocks to the unknown category with site_missing=0. All available nonempty sex/site fields have their missing indicators 0.
E_AVAILABLE contains available sex/location with universally missing age. E_NULL is the original structured missing vector, not all zeros. The generated tensors match the original encoder exactly, including the two unsupported nonempty sites.
No diagnosis, checklist attribute, elevation, management, difficulty, case identifier, notes, file path, or ground-truth mask was passed into the model. Only image pixels and the 22 clinical features entered inference.
RGB; short-side resize256 (bilinear); center crop224; original ImageNet mean/std. FP32 eval/inference_mode, TF32 off, batch8, workers2. No calibration, augmentation, ensemble or threshold fitting.
Clinical readout and CMGF were recomputed per input condition. Cached visual features were used only after a deterministic first-eight-case full-forward check of both heads and both conditions. Maximum absolute difference: 0; full-forward fallbacks: 0.

## 4. Fixed Model Identity

M0 is the current FIM FULL, and B2S is the same-protocol simplified reference; neither is a historical B6/B2 or another research-line model. Weights were strictly loaded from the P3 identities without reselecting best.
The original model and region-readout source bytes were retained in the isolated source copy. The P3 candidate-forward implementation and pure metric arithmetic are AST-equivalent after only redirecting their common-module import.

| Family | Seed | Original run | Best epoch | Best SHA-256 |
|---|---:|---|---:|---|
| M0 | 42 | FIM_LPAH_FULL_s42 | 34 | `096913eb3f123747b5dac8b11ca20f0ae594f783dcf8a0398e9390087707dbdd` |
| R_IMG | 42 | R_IMG_s42 | 23 | `f92283cc46bcc45a59e123364c6cea3da8f4e0377cff4a2b8d54458a622c3c13` |
| R_CLIN | 42 | R_CLIN_s42 | 26 | `56af879c23aaa3e741afa0306f55a5e05778c2752a7d699d8ccdb2a85c3cc597` |
| XATTN | 42 | XATTN_s42 | 13 | `2f12c84b23b56a79d99335bf202991549268dc6aeeffcffc18889f854d3dd25e` |
| B2S | 42 | FIM_B2S_s42 | 16 | `7be3a5b604dd64e075f55c1f10dea9ec4ab909d1a461d7ffd8f78049af5345c5` |
| M0 | 2024 | FIM_LPAH_FULL_s2024 | 34 | `d7ddfef5bb3de68cf1d3562a2ad3bd604c454423b963c9fd6c7e2aa568bd5a95` |
| R_IMG | 2024 | R_IMG_s2024 | 14 | `76bb5d4554fcba952ab3cd36a40feda4f29e0f1236443ed7ad97aa54778cb1c4` |
| R_CLIN | 2024 | R_CLIN_s2024 | 32 | `dc9d4567cb001365250df5a46b66e99e32796823ab1e69339300b4b707c801d7` |
| XATTN | 2024 | XATTN_s2024 | 36 | `72e6407e5ca64f46a1c92a439dfb80eb64717f07504ebc6e887446d335dbc757` |
| B2S | 2024 | FIM_B2S_s2024 | 60 | `7cb8b64419359cc4a8dab0a84c91d2e94f817bb7484a2a386f3a8911ad3998ee` |
| M0 | 3407 | FIM_LPAH_FULL_s3407 | 21 | `a8690c9e7b188011f408880b9401faae2878609931705cae81ee429e749df849` |
| R_IMG | 3407 | R_IMG_s3407 | 52 | `4bc4c03e0124ddb73cdb1da0c28cf22a78df6a4183d8e269324288522bae5180` |
| R_CLIN | 3407 | R_CLIN_s3407 | 29 | `5bcdca73e3d273b389cd3086b72b7d11b6b2642ecb0c250ae1be18ec1da0a3fe` |
| XATTN | 3407 | XATTN_s3407 | 19 | `9e5ac8680d75480bc767254644d33404372af67020d239a53084bd5b0bafd86b` |
| B2S | 3407 | FIM_B2S_s3407 | 46 | `7bb760c80638842091a8b44910c3d480ed94bd75cfb6e88c9535bff2c3dcdee1` |

Exact checkpoint paths, configuration identities and original selection metrics are in FIXED_MODEL_IDENTITIES.json and P4_FROZEN_PROTOCOL.json. Per-run model load records identify the actual imported local source. The evaluation-only config copies are in effective_configs/.

## 5. Prespecified Metrics and Main Results

AP is sklearn average precision, not trapezoidal PR area. ROC-AUC, unclipped binary Brier, binary NLL (epsilon=1e-7) and 15 equal-width-bin ECE are reported for each of 15 models, both inputs and both heads (60 rows). Primary score is softmax class MEL index4; the dedicated sigmoid melanoma head is secondary.
Each run is scored separately, then three-seed mean and sample SD (ddof=1) are computed. No concatenation of predictions or averaging of probabilities substitutes for these run means. All models and conditions are retained.

### E_AVAILABLE, seven-class softmax MEL

| Family | AP mean (sample SD) | AUC mean (sample SD) | Brier mean | NLL mean | ECE15 mean |
|---|---:|---:|---:|---:|---:|
| M0 | 0.638385 (0.020920) | 0.830553 (0.011684) | 0.201743 | 2.308684 | 0.201858 |
| R_IMG | 0.652938 (0.048626) | 0.840515 (0.026569) | 0.192007 | 1.875195 | 0.194605 |
| R_CLIN | 0.631497 (0.038318) | 0.819538 (0.025371) | 0.201553 | 2.588923 | 0.201758 |
| XATTN | 0.651371 (0.008218) | 0.828942 (0.002394) | 0.186976 | 1.768568 | 0.186601 |
| B2S | 0.650779 (0.038587) | 0.831946 (0.016121) | 0.192079 | 2.112554 | 0.190229 |

### Primary Paired AP Differences

| Contrast | Mean difference | Seed-difference sample SD | Coverage | Percentile interval |
|---|---:|---:|---:|---|
| R_CLIN_minus_M0 | -0.006887 | 0.051614 | 95% | [-0.027810, 0.012678] |
| R_CLIN_minus_M0 | -0.006887 | 0.051614 | 97.5% | [-0.030510, 0.015152] |
| R_CLIN_minus_XATTN | -0.019874 | 0.031867 | 95% | [-0.042281, 0.002167] |
| R_CLIN_minus_XATTN | -0.019874 | 0.031867 | 97.5% | [-0.045113, 0.005532] |

10,000 valid paired bootstrap draws used PCG64 seed20261017 and ascending numeric representative case_num cluster order. Every draw used the same cluster multiplicities across all families, seeds, conditions and heads. Metrics and same-seed paired differences were calculated first, then averaged across the three seeds.
Single-class draws were rejected and redrawn; observed rejected draws: 0. The two primary comparisons additionally have 97.5% percentile intervals as an approximate Bonferroni adjustment for two comparisons. Supporting AP/AUC comparisons have 95% intervals.
Intervals quantify case/same-image-cluster sampling uncertainty conditional on the three fixed fits; they are neither verified patient-level intervals nor a full interval over training-seed variation. Training-seed sample SD is listed separately. No new seeds, draws beyond the fixed count, or model changes were chosen from significance.
All supporting contrasts, E_NULL results and dedicated-head results are in the CSV tables. Absolute AP values across cohorts were not subtracted as a method improvement; Derm7pt prevalence is reported for this cohort alone.

## 6. Frozen Threshold Transport and Curves

DEVELOPMENT_THRESHOLDS.csv is an unchanged byte-for-byte copy from P3. Each model/seed/head uses its own saved HAM-validation 90% and 95% sensitivity-policy threshold; decisions use score >= threshold.
THRESHOLD_CONFUSION_COUNTS.csv contains all 120 transported policies with TP/FN/FP/TN, sensitivity, specificity, PPV and NPV. Actual external sensitivity is reported, not assumed to remain at 90%/95%. Undefined PPV/NPV remains missing; no model is dropped from a three-seed summary.
All 120 full ROC/PR coordinate files are provided in curve_coordinates/. ROC starts with an infinite threshold; the terminal PR point has no corresponding threshold and is blank. Curves were not smoothed or modified.
REGION_READOUT_DIAGNOSTICS.csv contains aggregate readout diagnostics, not independently measured localization accuracy.

## 7. Verification and Handoff

Frozen manifest rechecked unchanged: 68 entries. Strict best loads:15; finite prediction pairs:15; binary metric rows:60; policies:120; curve files:120; interval rows:34. Weighted/tied AP/AUC bootstrap arithmetic matches sklearn tests and full-sample point estimates.
The archive is an explicit allowlist of code, protocol, mapping, aggregate outputs, complete curves, model/source identities and hashed case fingerprints. It excludes weights, images, raw metadata/archive, per-case predictions, individual clinical records and credentials. Case fingerprints contain hashes and inclusion status only, no diagnosis or clinical values.
Private predictions and bootstrap arrays remain locally under `${R52_PATH_5A0A1C6B8687}` and are not distributed in the ZIP.
No manuscript, original input, shared code, environment, system setting, prior result or other queue was changed. No cloud or remote service was contacted.

### Main Files

- P4_EXTERNAL_REPORT.md: this report.
- P4_EXTERNAL_RESULTS.zip: checked handoff archive.
- P4_PROGRESS.json: final stage status.
- FINAL_VERIFICATION.json: invariant checks; ZIP_VERIFICATION.json: archive readability and member-hash checks.
- P4_FROZEN_PROTOCOL.json, USAGE_HISTORY_CONFIRMATION.json, COHORT_BINDING.json, CLINICAL_MAPPING.json: frozen design and provenance.
- PER_RUN_METRICS.csv, THREE_SEED_SUMMARY.csv, PER_SEED_PAIRED_DIFFERENCES.csv, CASE_CLUSTER_PAIRED_INTERVALS.csv: all results.

Execution stops after this handoff. No tuning or additional experiment is launched.

# Route B: Original OncoDerm Clinical Record Matching

Status: **COMPLETE**. Generated 2026-10-20T23:04:07.857641+08:00.

## Execution
- This is the first Route B run, after accepted A1. No A1 formal profiling or Stage 0 audit was repeated.
- Six original B2/B6 selected-best checkpoints only; 0 new training runs. No FULL/ZERO, regional, TKEEP or other-direction models were run.
- Historical manuscripts, data, weights, source code and results were read-only. No network, cloud, installation or publication actions.
- Scientific protocol: FIM_B_ORIGINAL_RECORD_MATCHING_v1.0_20261020.
- Four-hour window: 2026-10-20T21:41:01+08:00 to 2026-10-21T01:41:01+08:00.
- Actual score-grid units completed: 12/12.
- Final statistical repetitions are reported below, not inferred from scheduled work.

## Main Matching Effects
Common seven-class softmax MEL AP; positive matched-minus-shuffled means higher AP under correct matching.
SIIM comparisons are primary; Derm7pt comparisons are supporting. Results are not changes in the old model or its original performance.

| Cohort | Model | Matched - shuffled AP | Training-seed SD | Mapping SD | Mapping MCSE | 95% interval | 97.5% interval |
|---|---|---:|---:|---:|---:|---|---|
| SIIM | ORIG_B2 | 0.006977 | 0.001800 | 0.002915 | 0.000412 | [0.002683, 0.012791] | [0.001902, 0.013815] |
| SIIM | ORIG_B6 | 0.008372 | 0.002042 | 0.002197 | 0.000311 | [0.005061, 0.013631] | [0.004596, 0.014422] |
| Derm7pt | ORIG_B2 | 0.004525 | 0.002361 | 0.002940 | 0.000416 | [-0.000452, 0.010712] | not prescribed |
| Derm7pt | ORIG_B6 | 0.002202 | 0.002754 | 0.002913 | 0.000412 | [-0.002301, 0.008755] | not prescribed |

Full descriptive endpoints, both heads, all mapping repeats and unchanged operating thresholds are retained in the CSV files. No favorable-result filtering or run deletion was applied.
Matched-minus-null point estimates are provided for algebraic correspondence only. The original matched-minus-null intervals remain in their original sources and are not replaced here.

## Coverage and Estimand
- SIIM: all 33,126 images (584 positives), 2,056 patients. Equal-size patient blocks retain complete age/sex/site/missingness records, including within-patient variation.
- 2,037 patients / 31,638 images (95.5081%) receive a different patient donor. Nineteen singleton-size patients / 1,488 images stay native and remain in every metric denominator. No exchangeable-subset performance analysis.
- Derm7pt: all 1,011 derm cases (252 positives), 1,011 established case/same-image clusters. Donor case changes for every row. These are not verified patient identities.
- Encoded-vector changes differ from donor changes; naturally identical vectors are allowed. Joint record frequencies are preserved, not fields independently permuted.
- E_MATCHED is original SIIM E_REAL / Derm7pt E_AVAILABLE. E_NULL is original structured missingness, not all-zero. Unsupported nonempty sites retain site_missing=0.
- The intervention measures fixed-model dependence on record correspondence. It is not a clinical causal effect, an exact randomization null test, or proof of superiority over other fusion methods.

## Statistical Completion
- SIIM: point mappings 50/50; valid bootstrap draws 2000/2000, each completed draw includes all 50 internal remappings.
- Derm7pt: point mappings 50/50; valid bootstrap draws 2000/2000, each completed draw includes all 50 internal remappings.

- Seed stream: PCG64(SeedSequence([20261020, cohort, phase, draw, repeat, attempt])). Canonical patient/image ordering or numeric case-cluster ordering is frozen.
- Each bootstrap draw samples complete blocks with replacement within fixed original size strata, then rebuilds both recipient rows and donor pool. Duplicate original patients are distinct copy slots; their donor coincidence rate is reported separately.
- The same sample/remapping is used across all six models. Both heads use identical original 50 mappings; bootstrap intervals are prescribed only for the common head, while the dedicated head remains descriptive.
- Per-run metrics are calculated before averaging mappings and then the three fixed seeds. No probability averaging or pooled-seed pseudo-sample.
- Intervals are percentile approximations conditional on fixed model runs, observed size-stratum counts and the 19 fixed SIIM blocks, including finite-K mapping variation. No bootstrap over training seeds.
- Primary SIIM AP differences have 95% and 97.5% intervals; 97.5% is an approximate two-comparison Bonferroni reference. Supporting common-head contrasts have 95% intervals.
- AP is noninterpolated average precision; ROC-AUC uses tie-aware ranking; Brier uses unclipped probabilities; binary NLL clips only log inputs at 1e-7; ECE uses 15 equal-width bins. Undefined threshold statistics are explicitly marked.

## Technical Verification
- Authoritative checkpoint paths, sizes, full SHA-256, configuration identities and selected epochs are recorded in FIXED_MODEL_IDENTITIES.json and SOURCE_MANIFEST.csv.
- Original model modules are copied byte-for-byte into this run's model_code directory. Full strict state loading is used; constructor pretrained=False only prevents downloads.
- FP32, eval/inference_mode, TF32 off; original RGB, resize short side 256, center crop 224 and ImageNet normalization.
- No compatible historical visual feature cache was present in the original B2 cache directory. Each scored image therefore receives one visual pass per model, plus bounded first-eight-image technical checks.
- Every observed clinical type plus E_NULL uses the actual original metadata encoder, gate and prediction heads. Score grids are real forward calculations, not fitted approximations.
- First-eight-image full-forward/head/grid logits checked at atol=1e-5, rtol=1e-4. Matched/null compatibility required probability difference <=2e-6 and AP/AUC difference <=5e-6 against hashed historical FP32 outputs.
- Historical Derm7pt clusters.npy uses encoded cluster indices, whereas COHORT_ALIGNMENT uses case-number representatives. Their exact one-to-one partition correspondence was checked; canonical representative ordering was retained.
- Supplied reference passed 279 synthetic tests. Optimized numerical wrapper was checked against sklearn including ties and against the supplied 50-remapping nested reference.

## Files and Restart
- B_RUN_PROTOCOL.json: pre-inference input/implementation freeze. STATISTICS_IMPLEMENTATION_FREEZE.json: pre-statistics implementation hashes; scientific design unchanged.
- Per-model/cohort COMPLETE receipts and EQUIVALENCE records retain actual timing, compatibility errors and hashes.
- private/: original-to-canonical bindings, finite clinical type vectors, score grids and deterministic bootstrap draw checkpoints. These files are not in the ZIP.
- Inference resumes at saved canonical-order positions; completed units are skipped. Bootstrap resumes from missing draw numbers without rerunning completed draws.
- Delivery contains aggregate tables, original model code/configuration, reproducibility scripts, protocol, QC, this report and final progress only.
- ZIP excludes weights, images, source metadata, individual records/IDs, per-case predictions, clinical vectors, donor maps and score grids.
- No further phase starts automatically. Manuscript integration remains with ChatGPT.

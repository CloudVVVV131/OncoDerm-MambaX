# FIM three-training-seed report

Status: **COMPLETE**

Output: `${R52_PATH_3B164EDAFD4B}`
Reused seed42 source: `${R52_PATH_77BDE2542771}`
Window: 2026-10-13T04:53:12.0286598+08:00 to 2026-10-13T16:53:12.0286598+08:00

## Execution scope
- Exactly FULL/ZERO training seeds2024 and3407 were newly authorized and run sequentially. Seed42 training and inference were not rerun.
- Original HAM split42, fixed100 auxiliary pairs, metadata mapping, preprocessing, model and effective seed42 protocol were reused.
- No cloud connection, other research-line reuse, additional seeds, model search or manuscript/figure changes.
- Completed results are retained regardless of sign or significance. Unfinished runs, if any, remain PARTIAL, not completed.

| Run | Status | Epochs | Best epoch | Sum epoch minutes | Source |
|---|---|---:|---:|---:|---|
| FIM_LPAH_FULL_s42 | TRAINING_COMPLETE | 49 | 34 | 96.1787767569224 | read-only reuse |
| FIM_LPAH_ZERO_s42 | TRAINING_COMPLETE | 71 | 56 | 112.06579685608546 | read-only reuse |
| FIM_LPAH_FULL_s2024 | TRAINING_COMPLETE | 49 | 34 | 86.74411957263946 | new training |
| FIM_LPAH_ZERO_s2024 | TRAINING_COMPLETE | 53 | 38 | 71.1427640835444 | new training |
| FIM_LPAH_FULL_s3407 | TRAINING_COMPLETE | 36 | 21 | 47.55623164971669 | new training |
| FIM_LPAH_ZERO_s3407 | TRAINING_COMPLETE | 37 | 22 | 49.78707025448481 | new training |

## Seed parameterization and fixed training policy
- Training seeds affect Python, NumPy, PyTorch CPU/CUDA RNGs, new-layer initialization, sample order, augmentation, auxiliary sampling, DataLoader, configuration and resume checkpoints.
- Zero-based epoch formulas: classification order=s+epoch; image augmentation=s+epoch*1000003+dataset_index*1009; auxiliary PCG64 SeedSequence([s,epoch,19]); DataLoader generator=s*1000+17.
- HAM split remains42; fim_mask_train100.csv is byte-identical to seed42. No reselection or historical100-ID recovery.
- Each new seed constructs fresh non-pretrained parameters and loads the same original local public backbone. FULL/ZERO use the identical saved initial state within each seed; no trained seed42 state or initial_state_s42.pt is loaded.
- Backbone: timm/convnext_tiny.in12k_ft_in1k, revision aa096f03029c7f0ec052013f64c819b34f8ad790, SHA256 a1aefa409b513cf209b085424eb3efffe4e4a9f511491bc2c12ea35209e6bb95.
- Original224 input; effective classification48 and auxiliary32; microbatch8; workers2. FP16 AMP training, FP32 evaluation, TF32 disabled.
- AdamW: backbone3e-5, other3e-4, weight decay0.05; warmup5/cosine; maximum80epochs, patience15.
- Effective loss: weighted CE(beta0.999, smoothing0); mel BCE0.5; consistency0.05; sparse0.0005; TV0.002; mask Dice+BCE weight0.1 FULL/0 ZERO.
- ZERO retains the same auxiliary training-mode forward and BN exposure; mask gradients alone are disabled. BN/RNG-restoring AMP overflow retry is identical to the final seed42 implementation.
- Select best by HAM validation macro-F1, then seven-class MEL ROC-AUC, then earliest exact tie. No test/external selection.
- Reference YAML is preserved for provenance; effective_config.json records actual effective settings where the original reference contains unused alternatives.

## Evaluation and statistical estimand
- All four new bests must lock before new held-out evaluation. HAM validation/test and SIIM E_REAL/E_NULL use both seven-class softmax-MEL and sigmoid endpoint-MEL heads.
- 90%/95% sensitivity thresholds derive only from each model own HAM validation. No external threshold selection or fitted calibration.
- Seed42 metric/curve tables are copied byte-for-byte; its private predictions are read-only inputs to the authorized shared three-seed bootstrap.
- ALL_METRICS.csv contains individual fits. THREE_SEED_SUMMARY.csv reports the arithmetic mean and sample SD(ddof=1), plus FULL-minus-ZERO paired differences. No prediction concatenation or probability averaging.
- Primary endpoint: SIIM E_REAL seven-class MEL average precision, mean per-seed FULL-minus-ZERO difference. Other heads/inputs/metrics are secondary descriptive.
- 10000 patient-cluster paired bootstrap draws, PCG64 seed20261015, original lexical np.unique patient order. The same patient multiplicities are applied to all seeds and both conditions in each draw.
- 95% percentile intervals are conditional on these three fixed training pairs. They do not incorporate the distribution of unobserved training seeds; observed training-seed sample SD is separately reported.
- Aggregate class metrics remain image-level metrics with patient-cluster resampling, not patient-level diagnosis metrics.
- No new independent localization evaluation is claimed; the frozen auxiliary100 uses the first-batch declared selection and coverage limitations.

## Primary result
| Quantity | Three-seed mean | Training-seed sample SD | Patient-cluster95% interval |
|---|---:|---:|---|
| FULL AP | 0.10419901 | 0.00932775 | [0.08601594, 0.12782709] |
| ZERO AP | 0.10573963 | 0.01016922 | [0.08744717, 0.12876983] |
| FULL_minus_ZERO AP | -0.00154062 | 0.01515610 | [-0.01000002, 0.00710349] |

Primary interval includes zero: True. Report the estimate and interval without selecting runs based on this outcome.

## Verification and delivery
- PARAMETERIZATION_CHECK.json and per-seed SMOKE_TEST files document the narrow pre-training checks. EXECUTION_FREEZE.json/FINAL_INTEGRITY.json track explicitly reused inputs and code.
- FINAL_QC.json checks all six identities, best selection, completed updates, own-validation thresholds, curve-to-metric arithmetic and the per-seed-then-mean summary.
- All new weights, full optimizer/scheduler/scaler/RNG recovery states and predictions remain local in this new directory; they are not in the delivery ZIP.
- ZIP contains aggregate results, curve coordinates, six configurations/history/status records, training logs and necessary code. It excludes weights, images/masks, per-case predictions, clinical records and credentials.
- No subsequent experiment is automatically initiated.

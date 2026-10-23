# P2 Frozen-v1 Three-Seed Training Report

## Status: BLOCKED
New P2 runs with completed training and selected-best validation: **0/6**.
Separately reused: **3 P1 seed42 runs**, no retraining, inference or best change.
Read-only references: **3 current FIM_LPAH_FULL M0 runs**, HAM_val/NATIVE only.
These are not 12 new training runs. All prespecified candidates are retained irrespective of score.

| Run | Origin | Training status | Epochs | Best epoch | Val macro-F1 | Softmax MEL AP |
|---|---|---|---:|---:|---:|---:|
| R_IMG_s42 | reused_P1_no_execution | TRAINING_COMPLETE | 38 | 23 | 0.786216 | 0.724575 |
| R_CLIN_s42 | reused_P1_no_execution | TRAINING_COMPLETE | 41 | 26 | 0.790629 | 0.732862 |
| XATTN_s42 | reused_P1_no_execution | TRAINING_COMPLETE | 28 | 13 | 0.790911 | 0.690365 |
| R_IMG_s2024 | new_P2_training | NOT_STARTED | 0 | None | N/A | N/A |
| R_CLIN_s2024 | new_P2_training | NOT_STARTED | 0 | None | N/A | N/A |
| XATTN_s2024 | new_P2_training | NOT_STARTED | 0 | None | N/A | N/A |
| R_IMG_s3407 | new_P2_training | NOT_STARTED | 0 | None | N/A | N/A |
| R_CLIN_s3407 | new_P2_training | NOT_STARTED | 0 | None | N/A | N/A |
| XATTN_s3407 | new_P2_training | NOT_STARTED | 0 | None | N/A | N/A |
| FIM_LPAH_FULL_s42 | readonly_current_FIM_FULL | TRAINING_COMPLETE | 49 | 34 | 0.808075 | 0.691352 |
| FIM_LPAH_FULL_s2024 | readonly_current_FIM_FULL | TRAINING_COMPLETE | 49 | 34 | 0.798465 | 0.663708 |
| FIM_LPAH_FULL_s3407 | readonly_current_FIM_FULL | TRAINING_COMPLETE | 36 | 21 | 0.786878 | 0.698939 |

## Per-run aggregation
| Model | Head | Metric | n seeds | Mean | Sample SD (ddof=1) |
|---|---|---|---:|---:|---:|
| M0 | seven_class | macro_f1 | 3 | 0.797806 | 0.010614261234330978 |
| M0 | softmax_MEL | average_precision | 3 | 0.684666 | 0.018542350204076966 |
| R_CLIN | seven_class | macro_f1 | 1 | 0.790629 | N/A |
| R_CLIN | softmax_MEL | average_precision | 1 | 0.732862 | N/A |
| R_IMG | seven_class | macro_f1 | 1 | 0.786216 | N/A |
| R_IMG | softmax_MEL | average_precision | 1 | 0.724575 | N/A |
| XATTN | seven_class | macro_f1 | 1 | 0.790911 | N/A |
| XATTN | softmax_MEL | average_precision | 1 | 0.690365 | N/A |

PER_SEED_VALIDATION_METRICS.csv contains both heads' AP/AUC/Brier/NLL and seven-class macro-F1/
Brier/NLL. THREE_SEED_VALIDATION_SUMMARY.csv gives arithmetic mean and sample SD of per-run
metrics, without pooled predictions/probability averaging. PER_SEED_PAIRED_DIFFERENCES.csv and
PAIRED_DIFFERENCE_SUMMARY.csv give R_CLIN-minus-M0/R_IMG/XATTN within each seed, then mean/SD.
Negative differences are favorable for lower-is-better losses. Partial groups show actual n/seeds.
No bootstrap,significance test,CI,threshold selection or calibration fitting was performed.

## Sources and frozen v1
- P1: `${R52_PATH_17A062A20759}`
- Original FIM seed42: `${R52_PATH_77BDE2542771}`
- Original FIM three-seed: `${R52_PATH_3B164EDAFD4B}`
- New output: `${R52_PATH_EE441EEF360A}`

Original region_readout_reference.py and all supplied original model/data/transform source are
byte-identical to P1. Region definitions,detach,residual path,CMGF,XATTN and losses are unchanged.
P2_PROTOCOL.json retains original mechanism/data/settings and specifies the new authorized scope.
P1 protocol is preserved separately; P1_to_P2_*.diff records local integration changes.

Seeds2024/3407 load their corresponding original pre-training initial_state_s{seed}.pt only.
P2_INITIALIZATION_s*.json and EXECUTION_FREEZE.json record original INITIALIZATION metadata,
size,SHA-256 and sources. No trained historical best/last or seed42 state initializes a new seed.
All three same-seed candidates strictly share base tensors. Isolated new-module seeds3024/4407
preserve CPU/CUDA RNG; R_IMG/R_CLIN initial readout hashes match. *_INITIALIZATION_CHECK.json
records base-key identity,actual imported source paths and new-layer hashes. Technical states are discarded.

Python/NumPy/Torch CPU/CUDA use training_seed. Classification order:seed+zero-based epoch;
per-image augmentation:seed+epoch*1000003+dataset_index*1009; auxiliary PCG64:
SeedSequence([seed,epoch,19]); DataLoader generator:seed*1000+17. Config/checkpoints carry the
training seed and fixed split42. Forbidden test/external loader assertions remain enforced.

## Actual protocol and checks
Fixed HAM6982 train/1513 validation,original22-dimensional clinical encoding,unchanged auxiliary100.
Prior P1 pairing/overlap/mechanism checks are reused,not a repeated project audit or old-ID search.
224 train; validation resize256/center224; original augmentations/normalization. Effective48/aux32,
micro8,workers2; AdamW backbone3e-5/other3e-4,wd0.05,warmup5,cosine80,max80,patience15,clip1.
Mask0.1; weighted CE beta0.999/smoothing0,BCE0.5,consistency0.05,sparse0.0005,TV0.002 unchanged.
FP16 AMP train,FP32 validation,TF32 off; overflow restores buffers/RNG and retries the same update.
Best selection:macro-F1,then softmax MEL AUC,then earliest exact tie. AP is logged,not used to
change checkpoint selection,budget or stopping. CUDA_TECHNICAL_CHECK_s*.json records actual
initialization,shape,gradient and full48+32 AMP checks with live two-worker loaders.
Regional validity/area/alpha,eta,gradient norm and training auxiliary Dice are saved per epoch.
Auxiliary Dice is an optimization diagnostic,not an independent localization test. XATTN has no regional alpha.

## Validation and M0
Selected-best HAM_val/NATIVE only:1513 images,164 MEL. Both heads:average_precision_score
(not trapezoidal PR area),ROC-AUC,unclipped Brier,binary NLL clipping[1e-7,1-1e-7]. Seven classes:
macro-F1,sum-over-seven Brier,NLL clipping[1e-7,1]. Float64 metric arrays match P1. Selected-best
F1/AUC/AP must reproduce training validation within1e-7. M0 is current FIM FULL,not ZERO or old B6;
only FULL/HAM_val/NATIVE aggregate rows are consumed. M0_VALIDATION_REFERENCES.json records
precise paths,selected rows and existing best identities. No M0 weights are loaded or inferred anew.

## Timing and incidents
Window:`2026-10-14T17:06:28.8849424+08:00` to `2026-10-15T05:06:28.8849424+08:00`,not P1's deadline.
D free at admission:17.82GiB; planning requirement8GiB.
RUN_IDENTITIES.csv gives completed-epoch time totals and run-wall time; epoch durations include
train/validation but exclude most checkpoint writes. TWO_EPOCH_PROGRESS.json gives early ETAs.
Unfinished runs retain their actual status. Last checkpoints contain optimizer,scheduler,scaler,
Python/NumPy/Torch CPU/CUDA RNG,data position,partial loss/diagnostics,history and best selection.
- TRAINING_ERROR_20261014_172035.json: RuntimeError: bad allocation

## Interpretation and delivery
These are development-validation results across fixed training seeds,not independent external
evidence. Sample SD describes training-seed variation,not a clinical CI. All results are retained.
ZIP contains allowlisted protocol/code/diffs/configs/identities/logs/aggregate metrics/diagnostics.
No weights,images,per-case predictions,clinical rows or credentials. No original manuscript/result/
environment/hibernation/page-file change; no cloud,other direction,simplified-B2 training,test/SIIM
inference or automatic next stage. Return the requested three files and STOP.

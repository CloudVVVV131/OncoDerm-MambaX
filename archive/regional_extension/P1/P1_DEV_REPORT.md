# P1 Development Report

## Actual status: COMPLETE

Completed prescribed training and selected-best validation: 3/3.
All three prespecified candidates are retained below, irrespective of result direction.
No uncompleted run is described as finished. No P2 or external evaluation is authorized or started.

| Candidate | Training status | Epochs completed | Selected epoch | Val macro-F1 | Val MEL AUC | Val MEL AP | AP minus M0 |
|---|---|---:|---:|---:|---:|---:|---:|
| R_IMG_s42 | TRAINING_COMPLETE | 38 | 23 | 0.786216 | 0.930194 | 0.724575 | 0.033223 |
| R_CLIN_s42 | TRAINING_COMPLETE | 41 | 26 | 0.790629 | 0.925643 | 0.732862 | 0.041510 |
| XATTN_s42 | TRAINING_COMPLETE | 28 | 13 | 0.790911 | 0.924547 | 0.690365 | -0.000987 |

Existing M0 FULL_s42: selected epoch 34, macro-F1=0.8080754752858051, MEL AUC=0.9247161402303423, MEL AP=0.6913517174709254. Only original HAM_val/NATIVE aggregate records were extracted; no M0 retraining or new inference.

`VALIDATION_BINARY_METRICS.csv` reports both heads' AP/AUC, binary Brier and binary NLL, and
candidate-minus-M0 differences. `VALIDATION_SUMMARY.csv` additionally reports seven-class macro-F1,
sum-over-seven-classes Brier, multiclass NLL, parameter counts and training times where available.
NLL clips probabilities to [1e-7,1-1e-7] for binary and [1e-7,1] for multiclass, matching the existing
reference statistic. AP is sklearn average_precision_score, not trapezoidal PR area.
Metrics are computed per image on the same 1,513 HAM validation images (164 MEL), without bootstrap,
calibration fitting, threshold selection, confidence intervals or significance claims.

## Sources and isolation

- Original ZIP: `${R52_PATH_594A2E133795}`. CRC passed; original ZIP retained unchanged.
- This output: `${R52_PATH_17A062A20759}`. D free at admission: 31.57 GiB;
  working estimate6 GiB for three best/last checkpoints plus temporary save space.
- Seed42 original input: `${R52_PATH_77BDE2542771}`.
- Final three-seed training reference: `${R52_PATH_3B164EDAFD4B}`.
- Shared initial state: original `initial_state_s42.pt`, SHA-256
  `cb1dc95021bb234e039a1819ab1aca3519c175fc0dc9555b1963c5914d0b8fcc`.
- Only start-ZIP/new-output instructions were redirected to D in the extracted guides.
  C-drive read-only input paths and the scientific protocol were not globally replaced.
- Actual source imports and strict base key maps are recorded in `*_INITIALIZATION_CHECK.json`;
  source hashes and unchanged frozen inputs are listed in `SOURCE_IDENTITY.csv`.
- The candidate reference module and original model module remain byte-identical to the supplied ZIP.
  `TRAINER_INTEGRATION.diff` records the isolated trainer adaptation.

## Preserved training protocol

Training seed42, HAM split42, 6,982 train/1,513 validation and the exact frozen auxiliary100 membership.
Original 22-dimensional clinical encoding, classification augmentation,224 input; validation resize256,
center224. All candidate base tensors strictly match the original pre-training state before new module
initialization. New parameters use isolated seed1042; R_IMG/R_CLIN readout initial states match exactly.
No trained original best/last is used to initialize any candidate.

Classification effective48, auxiliary32, micro8, workers2; original AdamW backbone3e-5/other3e-4,
wd0.05, warmup5, cosine80, max80, patience15 and clip1. FP16 AMP train, FP32 validation, TF32 off.
Class-balanced CE beta0.999 and effective smoothing0; endpoint BCE0.5, consistency0.05,
attention sparse0.0005, TV0.002, auxiliary Dice+BCE0.1 for every candidate. Existing AMP overflow
fallback restores RNG/BN and retries the same effective update instead of skipping it.
Best is selected by validation macro-F1, then seven-class MEL AUC, then earlier exact tie.
AP is logged each epoch but does not replace this checkpoint-selection rule.

## Checks and diagnostics

Supplied nine CPU tests and the recorded CUDA integration gate cover eta0 identity, strict initialization,
one backbone/fuse/mask/metadata call, nonzero-eta gradients, detached predicted mask weights,
live auxiliary-mask gradients, original CMGF, R_IMG metadata-invariant alpha and R_CLIN-responsive alpha,
constant-mask finiteness, weighted-CE microbatch reduction and forbidden-loader assertions.
The new CUDA gate keeps both DataLoader workers alive during complete effective48+aux32 updates.
All technical-test states are discarded before formal training.

Per-epoch `readout_diagnostics.csv` reports region validity/area, alpha mean/within-sample SD,
residual eta/coefficient, gradient norm and auxiliary soft Dice where training occurred.
`SELECTED_BEST_READOUT_DIAGNOSTICS.csv` summarizes selected-best validation readouts where complete.
XATTN has no three-region alpha; these regional measures are not applicable, not zero.
Training auxiliary Dice is an optimization diagnostic, not independent localization performance.
The within-sample SD is descriptive dispersion, not uncertainty across independent training seeds.

## Timing, resource observations and incidents

Fresh active window: `2026-10-14T02:39:32.3106315+08:00` through `2026-10-14T14:39:32.3106315+08:00`.
Original prior-blocked directory remains untouched: `${R52_PATH_453D3E44E9C4}`.
`TWO_EPOCH_PROGRESS.json` records each available early ETA. Epoch durations include training and
validation but exclude most checkpoint write time; the separate run-wall field includes run management.
The specified80-epoch maximum is not shortened to force completion. Early stopping uses patience15.

- No exception was recorded in this invocation.
- `PROCESS_RESUME_20261014_034054.json`: earlier execution process ended without a Python traceback. Resumed from complete epoch6 optimizer/scheduler/scaler/RNG checkpoint at 2026-10-14T03:40:54.7214466+08:00; completed epochs were not rerun. The interrupted uncheckpointed epoch was recovered from the preceding complete checkpoint. Cause of the process exit is unconfirmed; this was not early stopping and the active deadline was not extended.

`RESOURCE_LOG.jsonl` records Windows physical-memory/commit headroom and CUDA allocation during the
live-worker gate. Hibernation/page-file/global settings were not modified and no other process was killed.
No remote/cloud resources were used. Unfinished runs retain their actual status/checkpoint, never a
fabricated best or performance result.

## Interpretation and delivery

This is seed42 development on a repeatedly consulted validation set, not independent held-out evidence.
Differences from the reused M0 reference are descriptive. A favorable validation AP difference is not an
external improvement, a significance finding or an acceptance-probability claim. No candidate or epoch
is removed because of a low score. Decisions about additional seeds/P2 remain with the user.

The ZIP is built from an explicit allowlist of code/configuration, hashes, logs, aggregate metrics,
diagnostics, scoped checks and this report. It excludes weights/optimizer checkpoints, images/masks,
per-example predictions, private membership/clinical rows, credentials and environment secrets.
No HAM test, SIIM, new external cohort, B1/B2 rerun, bootstrap, other research line or manuscript edits.
Return `P1_DEV_REPORT.md`, `P1_DEV_RESULTS.zip`, `P1_PROGRESS.json`, then stop.

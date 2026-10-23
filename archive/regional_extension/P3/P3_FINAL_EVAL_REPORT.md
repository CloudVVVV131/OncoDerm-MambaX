# P3 Fixed-Model Evaluation Report

## Actual status: COMPLETE

Training: 3/3 new B2S runs.
Evaluation: COMPLETE; 15/15 fixed models.
Statistics: COMPLETE; 10000/10,000 accepted patient draws.

## Scope and provenance

P2 authoritative COMPLETE delivery: `${R52_PATH_9574E1D723C3}`.
New isolated P3 directory: `${R52_PATH_2E90ACAFEFFB}`.
Authorized new fits are FIM_B2S seeds42/2024/3407; actually started 3, completed 3. The 12 M0/R_IMG/R_CLIN/XATTN fits remain unchanged.
M0 means current-direction FIM_LPAH_FULL, not ZERO or historical B6.
No other research branch, cloud resource, shared environment/system-setting modification or manuscript edit was used.

## B2S training

Original ConvNeXt-Tiny/CMGF/dual-head model; CSSA, multiscale fusion and lesion-prior auxiliary path disabled.
No auxiliary image read/forward/loss. Unused instantiated mask_head and prev_proj remain present.
This is a simplified-system reference, not a single-factor LPAH ablation.
Original corresponding pre-training state supplies every declared same-name/same-shape shared tensor.
The changed fuse input weight retains seed-initialized shape, never cropped or concatenated from FULL.
B2S_INITIALIZATION_s*.json enumerates transferred, shape-mismatched and FULL-only CSSA keys.
Training: fixed HAM split42,224,effective48/micro8/workers2; AdamW3e-5/3e-4,wd.05,warmup5,cosine80,max80,patience15,clip1.
WeightedCE beta.999 and actual smoothing0; MEL BCE.5,consistency.05,attention sparsity.0005,TV.002.
Selection: validation macro-F1, then softmax MEL AUC, then earlier exact tie. AP never selects best.
FP16 AMP training retries the same effective batch after scale backoff with pre-update buffers/RNG restored; FP32 validation,TF32off.
Lightweight worker modules preserve two workers and original sample/augmentation formulas.
Complete optimizer/scheduler/scaler/RNG/data-position checkpoints are retained privately.

| New run | Status | Completed epochs | Best epoch | Completed epoch seconds |
|---|---|---:|---:|---:|
| FIM_B2S_s42 | TRAINING_COMPLETE | 31 | 16 | 2275.5 |
| FIM_B2S_s2024 | TRAINING_COMPLETE | 75 | 60 | 5237.0 |
| FIM_B2S_s3407 | TRAINING_COMPLETE | 61 | 46 | 4520.1 |

## Fixed-model evaluation protocol

The evaluation gate requires P3_FROZEN_PROTOCOL.json to record all15best hashes, code/input manifests and prespecified analysis before held-out forwarding.
P3 evaluates HAMval1513,HAMtest1520 and previously used SIIM33126 (2056patients,584positive images), not a new independent external cohort.
Required evaluation for all15: fresh uniform P3 predictions,strict loading,FP32/eval/inference_mode,TF32off;RGB resize-short256/center224/ImageNet normalization.
E_REAL uses frozen22-dimensional clinical mapping including age/100 and torso->trunk. E_NULL is structured missing, not an all-zero vector.
No patient ID,target,diagnosis or image ID enters the model clinical vector.
Batch-local feature caching recomputes the clinical readout and CMGF for each condition. Per-model validation checks compare full/cached logits and both heads at atol1e-5,rtol1e-4; failures fall back to full forward.
Validation reproduction is technical verification of the already selected best. No reselection or metric-driven adjustment is allowed.
Validation reproduction discrepancies: [].
Inference status/identity files report actual completed cohorts and per-model elapsed time.

## Metrics and uncertainty

AP uses average_precision_score. AUC,Brier(unclipped),binaryNLL(epsilon1e-7),15-bin ECE are reported for both heads.
HAM also reports seven-class macro-F1,sum-over7 Brier,multiclassNLL and top-labelECE.
Binary ECE bins positive probabilities into15equal-width intervals; populated-bin absolute mean-probability minus observed-rate gaps are weighted by n_bin/N. Empty-bin means/rates remain null. Last bin includes1.
Per-run metrics precede seed means and sampleSD(ddof1). Same-seed differences are averaged; predictions/probabilities are not pooled into an ensemble.
Primary: SIIM E_REAL softmaxMEL AP, R_CLIN-M0 and R_CLIN-XATTN. Other prespecified comparisons/heads/conditions/AUC remain supporting.
Patient-cluster paired bootstrap uses10,000 valid PCG64(seed20261016) draws with lexical patient ordering and common multiplicities across all fits,conditions and heads. Invalid one-class draws are redrawn and counted.
95% percentile intervals are reported for prespecified AP/AUC contrasts. Two primary AP contrasts additionally have97.5%individual intervals as approximate Bonferroni family95%. No p-values.
These intervals condition on three fixed training runs; training-seed variation is separately described.
HAMtest has point values and descriptive3seed summaries only. No extra patient/bootstrap claim is made.
Each model/seed/head selects90%/95%sensitivity thresholds only from its own HAMval, maximizing specificity and taking the largest threshold on ties. p>=threshold defines positive; identical numeric thresholds transport unchanged.
Threshold TP/FN/FP/TN,sensitivity,specificity,PPV,NPV are descriptive, with no external tuning or threshold CI.
Curves are full original coordinates without smoothing; regional readout diagnostics are aggregate model behavior, not a localization test.

## Primary results

| Contrast | Mean AP difference | Training-seed SD | Coverage | Patient interval |
|---|---:|---:|---:|---|
| R_CLIN_minus_M0 | 0.016882 | 0.006242 | 0.95 | [0.008151, 0.025758] |
| R_CLIN_minus_M0 | 0.016882 | 0.006242 | 0.975 | [0.006884, 0.027192] |
| R_CLIN_minus_XATTN | 0.010438 | 0.009994 | 0.95 | [-0.000177, 0.019872] |
| R_CLIN_minus_XATTN | 0.010438 | 0.009994 | 0.975 | [-0.001618, 0.021344] |

## Execution record

Window: `2026-10-15T13:42:54+08:00` to `2026-10-16T01:42:54+08:00`.
Completed-epoch timing includes training and validation but excludes most checkpoint I/O; process interruptions are recorded separately.
The first seed42 attempt encountered Windows shared-memory mapping error1455 duringepoch6. Epoch5 full checkpoint was retained; recovery resumed at epoch6, not epoch1.
No batch/worker reduction, sample removal, bootstrap-count reduction or pagefile/hibernation change was made.
Error records retained: TRAINING_ERROR_20261015_135755.json, TRAINING_ERROR_20261015_140342.json.

BLOCKER_AND_CHECKPOINT.json and earlier delivery folders are historical snapshots. The current P3_PROGRESS.json and this report govern current status; a prior failure does not negate a later completed run.
## Deliverables

P3_FINAL_EVAL_RESULTS.zip contains code/protocols/initialization identities/configurations/logs/aggregate metrics/intervals/curve coordinates.
It excludes weights,raw images,per-example predictions,clinical rows,credentials and .env. Those private assets stay local.
All original P1/P2/FIM files and manuscript remain unchanged. No extra experiment or next method-development stage is started.

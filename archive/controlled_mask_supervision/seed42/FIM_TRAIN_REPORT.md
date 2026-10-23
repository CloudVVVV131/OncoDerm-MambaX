# FIM seed42 paired training report

Overall status: **COMPLETE**

Generated: 2026-10-12T22:16:34.896372+08:00
Output root: `${R52_PATH_77BDE2542771}`

## Scope and execution
- Only FULL and ZERO seed42 were authorized. The other four planned runs were not started.
- Local existing CUDA only. No remote connection, cloud resources, old B6 continuation, or other-direction model reuse.
- Original manuscript, shared source/configuration/environment, raw data, and historical results were not modified.
- This is a new mask-loss ablation under matched auxiliary-image exposure, not historical protocol recovery.

| Run | Status | Completed epochs | Selected epoch | Next batch |
|---|---|---:|---:|---:|
| FIM_LPAH_FULL_s42 | TRAINING_COMPLETE | 49 | 34 | 0 |
| FIM_LPAH_ZERO_s42 | TRAINING_COMPLETE | 71 | 56 | 0 |
| FIM_LPAH_FULL_s2024 | NOT_STARTED | 0 | - | - |
| FIM_LPAH_ZERO_s2024 | NOT_STARTED | 0 | - | - |
| FIM_LPAH_FULL_s3407 | NOT_STARTED | 0 | - | - |
| FIM_LPAH_ZERO_s3407 | NOT_STARTED | 0 | - | - |

### Actual training times
| Run | Sum of completed epoch minutes | Mean seconds/epoch | Wall minutes including restart gaps |
|---|---:|---:|---:|
| FULL | 96.18 | 117.77 | 101.09 |
| ZERO | 112.07 | 94.70 | 112.62 |

### Primary executed result
- SIIM E_REAL softmax MEL AP: FULL 0.101816; ZERO 0.095074.
- FULL minus ZERO AP: 0.006742; paired patient-cluster 95% CI [-0.008257, 0.021462].
- Primary interval includes zero: True. Report this interval rather than an unqualified superiority claim.
- This is a fixed seed42 pair. No across-training-seed conclusion is supported by this first batch.

## Fixed implementation
- Original model source: `${R52_PATH_9A2AE8D21C2F}`
- Model SHA256: `7e737a004f69053821628f84b3da9c44a4e4a208a386fa7322c5c7ff9331a0c5`
- Public base source: `https://huggingface.co/timm/convnext_tiny.in12k_ft_in1k/resolve/main/model.safetensors`
- Public base revision: `aa096f03029c7f0ec052013f64c819b34f8ad790`
- Public base SHA256: `a1aefa409b513cf209b085424eb3efffe4e4a9f511491bc2c12ea35209e6bb95`
- Shared new initialization SHA256: `cb1dc95021bb234e039a1819ab1aca3519c175fc0dc9555b1963c5914d0b8fcc`
- Installed timm resolves convnext_tiny to in12k_ft_in1k. Both conditions use exactly that public ImageNet initialization; no downstream checkpoint is used.
- Original axial-convolution CSSA, multi-scale fusion, 22-dimensional CMGF, dual heads and mask head retained.
- FULL mask loss=0.1; ZERO mask loss=0. Both forward identical auxiliary images in training mode, so BN exposure is matched.
- End-to-end AdamW; backbone LR 3e-5, other LR 3e-4, weight decay 0.05; warmup 5 epochs then cosine, maximum 80, patience 15.
- Selection: validation macro-F1, then seven-class MEL ROC-AUC, then earliest exact tie. No test/SIIM selection.
- Weighted CE beta0.999, no label smoothing; full effective-batch target-weight denominator. Mel BCE0.5, consistency0.05, sparse0.0005, TV0.002.
- Classification effective batch48 (final short batch retained); auxiliary32; actual microbatch and workers recorded in effective_config.json.
- Classification augmentation uses original transforms. Auxiliary image/mask share direct224 resize (bilinear/nearest) and ImageNet normalization; auxiliary clinical vector is explicit all-zero22 per original auxiliary loader.
- AMP training, FP32 evaluation, TF32 off. Overflow retries restore pre-update RNG and BN buffers and reduce scale; no data update is silently skipped.
- AMP reference: https://docs.pytorch.org/docs/2.8/notes/amp_examples.html

## Data and tests
- Original split counts: `{'train': 6982, 'val': 1513, 'test': 1520}`.
- Candidates: 200; eligible: 200; selected: 100.
- See DATA_CHECK.json, SMOKE_TEST.json, MASK_SELECTION_RULE.md and INPUT_SOURCES.json.
- No complete candidate patient/lesion map was available; this is a stated coverage limit, not fabricated non-overlap evidence.

## Outcomes and uncertainty
- Training complete: 2/2; evaluation: COMPLETE; statistics: COMPLETE.
- Any missing phase remains uncompleted. Partial epochs or predictions are not final experimental results.
- When present: METRICS.csv, INTERNAL_MULTICLASS_METRICS.csv, DEVELOPMENT_THRESHOLDS.csv, THRESHOLD_CONFUSION_COUNTS.csv and PAIRED_PATIENT_CLUSTER_CI.csv contain executed results.
- Primary: FULL minus ZERO SIIM E_REAL softmax-MEL average precision. ROC-AUC and other input/head comparisons are secondary.
- CI: 10000 paired patient-cluster draws, PCG64 seed20261015, percentile95%. Conditional on this fixed seed42 pair, not three-seed algorithm uncertainty.
- Thresholds come only from each new model own HAM validation set at90%/95% sensitivity; no fitted calibrator.

## Operational record
- Initial safe-loading incompatibility (TorchVersion metadata) was fixed in the isolated loader without changing initialization bytes.
- An epoch2 FP16 gradient overflow initially halted the strict entrypoint. The retry policy was corrected and continuation uses the completed epoch1 optimizer/scheduler/scaler/RNG checkpoint; discarded incomplete updates are not counted as epochs.
- training.log retains technical failures, continuation events and actual epoch times.
- best.pt contains selected weights/configuration; last.pt contains optimizer/scheduler/scaler/RNG and next epoch/batch for recovery. These remain local and are excluded from the ZIP.

## Delivery exclusions
- No model weights, raw images/masks, per-case clinical records, per-case predictions, credentials or environment secret files are packaged.
- Original figure/manuscript files are untouched. No additional run or research direction is automatically started.

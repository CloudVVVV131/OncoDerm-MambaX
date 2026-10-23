# Bioengineering submission-gap experiments

This supplement implements only the experiments requested in
`Bioengineering_投稿前差距清单与修改方案_OncoDerm-MambaX_2026-08-14.md`.
It preserves the previous B0-B6 evidence and writes new results under
`results/bioengineering_direct_baselines/`.

## Scientific scope

- P0 direct comparison: existing ConvNeXt-Tiny/B1, Swin-Tiny seed 42,
  DermaMamba seed 42, and existing B2/B4/B6 on the identical lesion-grouped
  split (`split_seed=42`, 224 x 224).
- Table 3 uses each architecture's intended melanoma score. Table S5 forces
  the seven-class softmax MEL score for every direct-comparison architecture.
- External thresholds are derived only from each run's internal validation
  predictions and are applied unchanged to SIIM-ISIC 2020. ROC-AUC and PR-AUC
  remain threshold-free. External labels are never used for model selection.
- P1 B6-M changes only the F4 context branch to one official
  `state-spaces/mamba` block. Seed 42 is evaluated first; seeds 2024 and 3407
  run only when the frozen accuracy, Params, FLOPs, and latency gate in
  `B6M_seed42_gate.json` passes.
- P1 paired bootstrap aligns B6 and DermaMamba by external `image_id` and uses
  10,000 paired image-level resamples for ROC-AUC and PR-AUC deltas.
- P1 mask expansion runs only when 200 unique real image-mask pairs exist.
  Otherwise `mask200_status.json` records `not_run_insufficient_pairs`; no pair
  is duplicated or synthesized.
- P2 cross-split sensitivity trains B6 with run seed 42 on an independently
  generated lesion split (`split_seed=2024`). It is reported descriptively and
  never mixed into the fixed-split direct table.

## DermaMamba reproducibility boundary

The paper-linked repository contains architecture source but does not contain
the advertised training script, config, checkpoint, or a detectable LICENSE
file at the locked revision. Its public model source also contains two tensor
shape inconsistencies. Consequently the comparator is named **DermaMamba
(resource-matched architecture port)**, not an exact official reproduction.

The port preserves ResNet-50 dual attention, four public scan orders, the
custom selective-state-space branch, adaptive branch fusion, and classifier.
It omits ABCDE inputs because the locked cohort has no genuine ABCDE variables,
adapts the classifier to seven classes, and uses resource-matched width/depth.
The paper reports a 100-epoch, 10-epoch-warm-up, Adam/CE setting in its final
hyperparameter discussion (despite an AdamW inconsistency elsewhere in the
article); the port therefore uses Adam, cross-entropy, validation Macro-F1
selection, and validation-only early stopping. Project-specific CB-focal and
attention regularization are disabled for this comparator. The one-GPU batch
size and width/depth reduction are declared adaptations, not hidden claims of
an exact author-run reproduction.
All deviations and mechanical fixes are saved in each run's model metadata.
Pinned sources are in `configs/bioengineering/source_lock.json`.

## Full AutoDL run

For a data-only cloud workspace, use the full profile below. It installs the
Python dependencies, trains any missing historical references, trains every
unconditional new experiment, executes the frozen conditional branches, and
runs the independent completion audit. Completed runs and prediction triplets
are reused on restart.

```bash
chmod +x run_bioengineering_supplement.sh run_bioengineering_supplement_full.sh
nohup bash run_bioengineering_supplement_full.sh \
  > autodl_bioengineering_supplement.log 2>&1 &
```

The full profile requires a CUDA-enabled PyTorch image and the prepared
HAM10000/internal, SIIM-ISIC 2020 external, and ISIC mask directories described
by the preflight report. `mamba-ssm==2.3.2.post1` is installed with its official
CUDA selective-scan opt-in and is exercised by a CUDA forward/backward smoke
test before B6-M training.

The runner is restartable: a new run is reusable only after a
`TRAINING_COMPLETE` marker has been written or a complete validation/test/
external prediction triplet already exists. This prevents an interrupted
`last.pth` from being mistaken for a completed experiment.
`TRAIN_MISSING_REFERENCES=0` prevents silent retraining of historical
references. New DermaMamba, B6-M, and cross-split runs are trained when absent.
The full wrapper intentionally sets `TRAIN_MISSING_REFERENCES=1`; override it
only when the historical B1/B2/B4/B6 artifacts must be treated as mandatory.
It also sets `AUTO_PACKAGE_RESULTS=1`, so a full audited result archive and its
SHA-256 file are created in the parent directory after successful completion.

After `completion_audit.json` reports `passed`, create a self-contained result
archive again manually (including the selected checkpoints) with:

```bash
python scripts/23_pack_bioengineering_results.py
```

The packer refuses to archive an incomplete experiment. Raw image datasets are
excluded, while audited tables, predictions, selected run directories,
checkpoints, split files, code, configs, logs, and provenance are retained.

## Required outputs

- `baseline_summary_seed42.csv`
- `external_melanoma_seed42.csv`
- `sevenclass_derived_melanoma_scores.csv`
- `tables/Table3_direct_recent_baselines.csv`
- `tables/TableS5_architecture_fairness.csv`
- `dermamamba_vs_b6_paired_bootstrap.csv`
- `optional/B6M_seed42_gate.json`
- `optional/B6_crosssplit2024_metrics.csv`
- `mask200_status.json`
- `direct_baseline_manifest.json`, `status.json`, and `completion_audit.json`

The final status must be `completed`. A mask status of
`not_run_insufficient_pairs` is a valid, auditable feasibility outcome, not a
completed 200-mask experiment. The runner exits successfully only when
`completion_audit.json` reports `passed`; that audit independently verifies
the fixed split/run seeds, prediction and configuration hashes, validation-only
threshold derivation, direct-table membership, paired-bootstrap budget and
finite-sample p-value method, B6-M gate consistency, cross-split identity, and
the 200-mask feasibility outcome.

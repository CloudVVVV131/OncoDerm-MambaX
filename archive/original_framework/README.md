# original_framework

## Research role
Main 3.3-3.5,4.1-4.3; Table 2; Supplementary Tables S3-S16,S28.

Original B0-B6; B2/B4/B6 repeatability; direct reference systems; separately labelled B6-M/split extension and v12 all6 exploration.

Score/input identity: Original seven-class MEL softmax and distinct dedicated sigmoid head; score choice is series-specific, never pooled across snapshots.

## Retained implementation and order
Historical snapshots: scripts/00_make_splits.py (only historical partition construction), 01_train.py, 02_test.py, 03_eval_melanoma_endpoint.py, 08_external_inference.py, summary scripts 15/16/17; August reference helper and direct-summary implementation remain separate. The complete argparse catalog records exact accepted arguments; no new command parameter is invented.

These are recorded historical entrypoints or static parser-derived invocation forms, **not commands executed by this packaging task**. Do not execute from this read-only review copy. Use a new workspace only after the local inputs, author permissions, execution window and relocated integrity records are resolved. Auto-training/evaluation imports and top-level script effects remain in historical code; do not use `--help` as a safety test.

Actual source files: 289; effective/source configs: 146. Exact definitions, lines, parser arguments, imports and SHA-256 are in `evidence_index/ENTRYPOINT_CATALOG.csv`, `IMPORT_DEPENDENCY_INVENTORY.csv` and `SOURCE_FINGERPRINTS.csv` at candidate root. Prefix entries with this unit's relative path.

## Inputs and outputs
Required, not bundled: Original HAM partitions/metadata and images; original auxiliary membership for LPAH; original fixed checkpoints for evaluation; SIIM images and metadata; provider/pretrained-weight terms.

Retained output names: Historical outputs/runs records, result tables, endpoint metrics and computational profiles; copied effective config/environment records are retained, not regenerated.

Exact original 100-ID auxiliary membership remains unrecovered (accepted previous conclusion; not searched again). Historical initial pretrained revision/SHA not fully recorded. DermaMamba local port requires third-party clearance. Missing licensed inputs/split IDs are not replaced by new data. Current-release helper is not evidence that its implementation ran in July.

## Evidence and validation
Historical completion evidence: retained_run_records/*/{config.yaml,environment.json}; reference_models/BASELINE_CONFIGURATION_INVENTORY.csv; historical snapshot README and archive identities. Copies retain historical status/date, including earlier NOT_STARTED/draft fields where applicable; only the applicable final-stage report indicates completion.

This task performed source/configuration collection and static checks only. No training, weight loading, inference, metric computation, bootstrap, calibration, threshold selection or drawing was performed. Historical scientific completion is not a claim of an end-to-end run from this candidate.

## Environment, path setup and permission
Retained dependency records: original_framework/july_submission/requirements-autodl.txt; original_framework/july_submission/requirements.txt; original_framework/july_stability/requirements-autodl.txt; original_framework/july_stability/requirements.txt; original_framework/v12_exploratory/requirements-autodl.txt; original_framework/v12_exploratory/requirements.txt; original_framework/august_snapshot/requirements-autodl.txt; original_framework/august_snapshot/requirements-bioengineering.txt; original_framework/august_snapshot/requirements.txt; original_framework/local_release/requirements-mamba.txt; original_framework/local_release/requirements-figures.txt; original_framework/retained_run_records/oncoderm_mambax_b0_backbone_only_seed42_20260706_232558/environment.json; original_framework/retained_run_records/oncoderm_mambax_b1_melanoma_head_seed42_20260706_233528/environment.json; original_framework/retained_run_records/oncoderm_mambax_b2_metadata_cmgf_seed42_20260706_234933/environment.json; original_framework/retained_run_records/oncoderm_mambax_b3_cssa_seed42_20260707_000406/environment.json; original_framework/retained_run_records/oncoderm_mambax_b4_lpah_mask_aux_seed42_20260707_002051/environment.json; original_framework/retained_run_records/oncoderm_mambax_b5_cssa_lpah_seed42_20260707_003614/environment.json; original_framework/retained_run_records/oncoderm_mambax_b6_full_seed42_20260707_004712/environment.json; original_framework/reference_models/B1/environment.json; original_framework/reference_models/B2/environment.json; original_framework/reference_models/B4/environment.json; original_framework/reference_models/B6/environment.json; original_framework/reference_models/EfficientNet-B0/environment.json; original_framework/reference_models/Swin-Tiny/environment.json; original_framework/reference_models/DermaMamba port/environment.json; original_framework/retained_stability_runs/oncoderm_mambax_b2_metadata_cmgf_seed2024_20260708_160522/environment.json; original_framework/retained_stability_runs/oncoderm_mambax_b2_metadata_cmgf_seed3407_20260708_230453/environment.json; original_framework/retained_stability_runs/oncoderm_mambax_b2_metadata_cmgf_seed42_20260706_234933/environment.json; original_framework/retained_stability_runs/oncoderm_mambax_b4_lpah_mask_aux_seed2024_20260708_161720/environment.json; original_framework/retained_stability_runs/oncoderm_mambax_b4_lpah_mask_aux_seed3407_20260708_231729/environment.json; original_framework/retained_stability_runs/oncoderm_mambax_b4_lpah_mask_aux_seed42_20260707_002051/environment.json; original_framework/retained_stability_runs/oncoderm_mambax_b6_full_seed2024_20260708_163051/environment.json; original_framework/retained_stability_runs/oncoderm_mambax_b6_full_seed3407_20260708_232735/environment.json; original_framework/retained_stability_runs/oncoderm_mambax_b6_full_seed42_20260707_004712/environment.json

Python absolute path literals were replaced only by required `R52_PATH_*` environment lookups (embedded literals use `expandvars`); their original AST is restored and compared in the static check. Text configs use `${R52_PATH_*}` tokens. `tools/materialize_path_config.py` can create a NEW expanded text-config copy without executing any experiment. `PATH_REQUIREMENTS.csv` has safe binding hints; exact local paths are kept only in the private outer mapping. Output bindings must never target historical folders.

Parent-derived navigation, historical code-hash guards and expired windows are preserved, not silently disabled. They are explicit remaining IO/setup work, not scientific changes. A passed syntax check does not clear these conditions.

License: AUTHOR_CONFIRMATION_REQUIRED for this assembled unit. The pre-existing MIT file belongs only to its recorded local-release subtree; third-party components and data terms are separate. No online publication was attempted.

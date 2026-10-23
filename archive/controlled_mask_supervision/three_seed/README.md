# controlled_mask_supervision/three_seed

## Research role
Main Table 5A; Supplementary Tables S23-S27.

FULL/ZERO new seeds2024/3407 plus read-only reuse of seed42; six total, not six new in this continuation.

Score/input identity: Per-run then seed means/sampleSD; both heads and clinical conditions; no probability ensemble.

## Retained implementation and order
initialize_new_seeds.py -> fim_training.py -> fim_evaluation.py/fim_statistics.py -> three_seed_statistics.py; old finish_pipeline.py deliberately not bundled as an automatic queue driver

These are recorded historical entrypoints or static parser-derived invocation forms, **not commands executed by this packaging task**. Do not execute from this read-only review copy. Use a new workspace only after the local inputs, author permissions, execution window and relocated integrity records are resolved. Auto-training/evaluation imports and top-level script effects remain in historical code; do not use `--help` as a safety test.

Actual source files: 11; effective/source configs: 5. Exact definitions, lines, parser arguments, imports and SHA-256 are in `evidence_index/ENTRYPOINT_CATALOG.csv`, `IMPORT_DEPENDENCY_INVENTORY.csv` and `SOURCE_FINGERPRINTS.csv` at candidate root. Prefix entries with this unit's relative path.

## Inputs and outputs
Required, not bundled: Same split42/auxiliary100; own training-seed initialization; existing seed42 predictions/metrics and patient linkage.

Retained output names: ALL_METRICS.csv, THREE_SEED_SUMMARY.csv, PER_SEED_PAIRED_DIFFERENCES.csv, THREE_SEED_PATIENT_CLUSTER_CI.csv.

Private per-example reuse and initialization files not distributed. Historical WINDOW.json is evidence, not a fresh execution authorization.

## Evidence and validation
Historical completion evidence: FIM_THREE_SEED_REPORT.md; SIX_RUN_IDENTITIES.json; INITIALIZATION_s2024.json; INITIALIZATION_s3407.json; per-run effective_config.json. Copies retain historical status/date, including earlier NOT_STARTED/draft fields where applicable; only the applicable final-stage report indicates completion.

This task performed source/configuration collection and static checks only. No training, weight loading, inference, metric computation, bootstrap, calibration, threshold selection or drawing was performed. Historical scientific completion is not a claim of an end-to-end run from this candidate.

## Environment, path setup and permission
Retained dependency records: No separate complete environment lock located; inspect effective config and history. Unknown versions remain unrecorded.

Python absolute path literals were replaced only by required `R52_PATH_*` environment lookups (embedded literals use `expandvars`); their original AST is restored and compared in the static check. Text configs use `${R52_PATH_*}` tokens. `tools/materialize_path_config.py` can create a NEW expanded text-config copy without executing any experiment. `PATH_REQUIREMENTS.csv` has safe binding hints; exact local paths are kept only in the private outer mapping. Output bindings must never target historical folders.

Parent-derived navigation, historical code-hash guards and expired windows are preserved, not silently disabled. They are explicit remaining IO/setup work, not scientific changes. A passed syntax check does not clear these conditions.

License: AUTHOR_CONFIRMATION_REQUIRED for this assembled unit. The pre-existing MIT file belongs only to its recorded local-release subtree; third-party components and data terms are separate. No online publication was attempted.

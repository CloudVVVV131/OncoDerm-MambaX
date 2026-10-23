# regional_extension/P4

## Research role
Main Table 5B; Supplementary Tables S35-S36.

15 fixed P3 models on Derm7pt; distinct from original_derm7pt six-model study.

Score/input identity: E_AVAILABLE/E_NULL; age missing, sex/location mapped; no arbitrary mapping to seven-class diagnosis; case-level clustered intervals.

## Retained implementation and order
p4_prepare.py -> p4_entry.py -> p4_statistics.py; clinical-dependent readout recomputed for each input condition

These are recorded historical entrypoints or static parser-derived invocation forms, **not commands executed by this packaging task**. Do not execute from this read-only review copy. Use a new workspace only after the local inputs, author permissions, execution window and relocated integrity records are resolved. Auto-training/evaluation imports and top-level script effects remain in historical code; do not use `--help` as a safety test.

Actual source files: 15; effective/source configs: 1. Exact definitions, lines, parser arguments, imports and SHA-256 are in `evidence_index/ENTRYPOINT_CATALOG.csv`, `IMPORT_DEPENDENCY_INVENTORY.csv` and `SOURCE_FINGERPRINTS.csv` at candidate root. Prefix entries with this unit's relative path.

## Inputs and outputs
Required, not bundled: Derm7pt derm column and exact-case paths; P3 model identities and development thresholds; approved usage-history record.

Retained output names: PER_RUN_METRICS.csv, THREE_SEED_SUMMARY.csv, CASE_CLUSTER_PAIRED_INTERVALS.csv, region diagnostics.

No patient-level independence guarantee; private bindings/checkpoints not supplied; do not exchange original-model thresholds with these P3 thresholds.

## Evidence and validation
Historical completion evidence: P4_EXTERNAL_REPORT.md; P4_PROGRESS.json; P4_FROZEN_PROTOCOL.json; FIXED_MODEL_IDENTITIES.json; recorded runtime files. Copies retain historical status/date, including earlier NOT_STARTED/draft fields where applicable; only the applicable final-stage report indicates completion.

This task performed source/configuration collection and static checks only. No training, weight loading, inference, metric computation, bootstrap, calibration, threshold selection or drawing was performed. Historical scientific completion is not a claim of an end-to-end run from this candidate.

## Environment, path setup and permission
Retained dependency records: regional_extension/P4/INFERENCE_RUNTIME.json; regional_extension/P4/STATISTICS_RUNTIME.json

Python absolute path literals were replaced only by required `R52_PATH_*` environment lookups (embedded literals use `expandvars`); their original AST is restored and compared in the static check. Text configs use `${R52_PATH_*}` tokens. `tools/materialize_path_config.py` can create a NEW expanded text-config copy without executing any experiment. `PATH_REQUIREMENTS.csv` has safe binding hints; exact local paths are kept only in the private outer mapping. Output bindings must never target historical folders.

Parent-derived navigation, historical code-hash guards and expired windows are preserved, not silently disabled. They are explicit remaining IO/setup work, not scientific changes. A passed syntax check does not clear these conditions.

License: AUTHOR_CONFIRMATION_REQUIRED for this assembled unit. The pre-existing MIT file belongs only to its recorded local-release subtree; third-party components and data terms are separate. No online publication was attempted.

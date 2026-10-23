# cohort_profiles_A1

## Research role
Main Table1B; Supplementary Tables S40-S41; Methods3.2,3.6.1.

Five cohort groups, clinical descriptive/encoding coverage; no model performance experiment.

Score/input identity: Not applicable; metadata-only descriptive profiling, not patient record output.

## Retained implementation and order
python -B a1_profile.py; accepted-stage0 paths and a1_inputs.json resolved before any future use; not rerun here

These are recorded historical entrypoints or static parser-derived invocation forms, **not commands executed by this packaging task**. Do not execute from this read-only review copy. Use a new workspace only after the local inputs, author permissions, execution window and relocated integrity records are resolved. Auto-training/evaluation imports and top-level script effects remain in historical code; do not use `--help` as a safety test.

Actual source files: 1; effective/source configs: 0. Exact definitions, lines, parser arguments, imports and SHA-256 are in `evidence_index/ENTRYPOINT_CATALOG.csv`, `IMPORT_DEPENDENCY_INVENTORY.csv` and `SOURCE_FINGERPRINTS.csv` at candidate root. Prefix entries with this unit's relative path.

## Inputs and outputs
Required, not bundled: Registered HAM train/validation/test tables, SIIM official metadata and Derm7pt meta.csv; accepted source register and original encoder text.

Retained output names: COHORT_CLINICAL_CHARACTERISTICS.csv, COHORT_FIELD_DISTRIBUTIONS.csv, CLINICAL_ENCODING_COVERAGE.csv; grouped structural design inputs.

Raw clinical tables and source registry dependencies intentionally omitted; group-size aggregates do not turn case_num into patient_id. Prior draft B status is a historical snapshot, superseded by completed B below.

## Evidence and validation
Historical completion evidence: A1_PROFILE_REPORT.md; A1_PROFILE_PROGRESS.json; A1_PROFILE_QC.json; COHORT_SUMMARY_METHOD.md. Copies retain historical status/date, including earlier NOT_STARTED/draft fields where applicable; only the applicable final-stage report indicates completion.

This task performed source/configuration collection and static checks only. No training, weight loading, inference, metric computation, bootstrap, calibration, threshold selection or drawing was performed. Historical scientific completion is not a claim of an end-to-end run from this candidate.

## Environment, path setup and permission
Retained dependency records: No separate complete environment lock located; inspect effective config and history. Unknown versions remain unrecorded.

Python absolute path literals were replaced only by required `R52_PATH_*` environment lookups (embedded literals use `expandvars`); their original AST is restored and compared in the static check. Text configs use `${R52_PATH_*}` tokens. `tools/materialize_path_config.py` can create a NEW expanded text-config copy without executing any experiment. `PATH_REQUIREMENTS.csv` has safe binding hints; exact local paths are kept only in the private outer mapping. Output bindings must never target historical folders.

Parent-derived navigation, historical code-hash guards and expired windows are preserved, not silently disabled. They are explicit remaining IO/setup work, not scientific changes. A passed syntax check does not clear these conditions.

License: AUTHOR_CONFIRMATION_REQUIRED for this assembled unit. The pre-existing MIT file belongs only to its recorded local-release subtree; third-party components and data terms are separate. No online publication was attempted.

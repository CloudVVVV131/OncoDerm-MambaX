# regional_extension/P3

## Research role
Main Table 5B; Supplementary Tables S34-S36.

Three new FIM_B2S fits + frozen 12 M0/R_IMG/R_CLIN/XATTN fits; NOT original B2/B6.

Score/input identity: Both heads; NATIVE/E_REAL/E_NULL; primary SIIM softmax MEL AP, predefined comparisons.

## Retained implementation and order
p3_initialization.py/p3_checks.py -> p3_train_entry.py for original three B2S -> p3_prepare_evaluation.py/p3_eval_entry.py -> p3_statistics.py; historical order only

These are recorded historical entrypoints or static parser-derived invocation forms, **not commands executed by this packaging task**. Do not execute from this read-only review copy. Use a new workspace only after the local inputs, author permissions, execution window and relocated integrity records are resolved. Auto-training/evaluation imports and top-level script effects remain in historical code; do not use `--help` as a safety test.

Actual source files: 20; effective/source configs: 4. Exact definitions, lines, parser arguments, imports and SHA-256 are in `evidence_index/ENTRYPOINT_CATALOG.csv`, `IMPORT_DEPENDENCY_INVENTORY.csv` and `SOURCE_FINGERPRINTS.csv` at candidate root. Prefix entries with this unit's relative path.

## Inputs and outputs
Required, not bundled: Frozen15model identities, HAM train/val/test, SIIM patient metadata, initial states for B2S, reference model source.

Retained output names: PER_RUN_METRICS.csv, THREE_SEED_SUMMARY.csv, PATIENT_CLUSTER_PAIRED_INTERVALS.csv, DEVELOPMENT_THRESHOLDS.csv.

15weights and private bindings/predictions withheld; no actual inference or statistics in current task. Existing progress and freeze files retain historical scope.

## Evidence and validation
Historical completion evidence: P3_FINAL_EVAL_REPORT.md; P3_PROGRESS.json; P3_FROZEN_PROTOCOL.json; FIXED_MODEL_IDENTITIES.json; RUNTIME_VERSIONS.json. Copies retain historical status/date, including earlier NOT_STARTED/draft fields where applicable; only the applicable final-stage report indicates completion.

This task performed source/configuration collection and static checks only. No training, weight loading, inference, metric computation, bootstrap, calibration, threshold selection or drawing was performed. Historical scientific completion is not a claim of an end-to-end run from this candidate.

## Environment, path setup and permission
Retained dependency records: regional_extension/P3/RUNTIME_VERSIONS.json

Python absolute path literals were replaced only by required `R52_PATH_*` environment lookups (embedded literals use `expandvars`); their original AST is restored and compared in the static check. Text configs use `${R52_PATH_*}` tokens. `tools/materialize_path_config.py` can create a NEW expanded text-config copy without executing any experiment. `PATH_REQUIREMENTS.csv` has safe binding hints; exact local paths are kept only in the private outer mapping. Output bindings must never target historical folders.

Parent-derived navigation, historical code-hash guards and expired windows are preserved, not silently disabled. They are explicit remaining IO/setup work, not scientific changes. A passed syntax check does not clear these conditions.

License: AUTHOR_CONFIRMATION_REQUIRED for this assembled unit. The pre-existing MIT file belongs only to its recorded local-release subtree; third-party components and data terms are separate. No online publication was attempted.

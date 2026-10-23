# development_records/P6

## Research role
Supplementary Table S39; token-preserving development section.

FIM_TKEEP_REGION_s42/FIM_TKEEP_GLOBAL_s42; single-seed validation development only.

Score/input identity: Two heads; validation only; no test/SIIM/Derm7pt results inferred for TKEEP.

## Retained implementation and order
p6_prepare.py/p6_checks.py -> p6_entry.py (with p6_training.py and p6_worker_data.py) -> p6_finish.py; p6_resume.py is the retained recovery route

These are recorded historical entrypoints or static parser-derived invocation forms, **not commands executed by this packaging task**. Do not execute from this read-only review copy. Use a new workspace only after the local inputs, author permissions, execution window and relocated integrity records are resolved. Auto-training/evaluation imports and top-level script effects remain in historical code; do not use `--help` as a safety test.

Actual source files: 17; effective/source configs: 3. Exact definitions, lines, parser arguments, imports and SHA-256 are in `evidence_index/ENTRYPOINT_CATALOG.csv`, `IMPORT_DEPENDENCY_INVENTORY.csv` and `SOURCE_FINGERPRINTS.csv` at candidate root. Prefix entries with this unit's relative path.

## Inputs and outputs
Required, not bundled: HAM train/validation, fixed auxiliary100, identical initial shared model and four-slot readout initialization; not trained best initialization.

Retained output names: RUN_SUMMARY.csv, ALL_VALIDATION_BINARY_METRICS.csv, ALL_VALIDATION_MULTICLASS_METRICS.csv, descriptive differences and readout diagnostics.

No extension to other seeds or external evaluation is claimed; initial state and auxiliary IDs not distributed.

## Evidence and validation
Historical completion evidence: P6_DEV_REPORT.md; P6_PROGRESS.json; P6_FROZEN_PROTOCOL.json; P6_INITIALIZATION.json; RUNTIME.json; effective configs. Copies retain historical status/date, including earlier NOT_STARTED/draft fields where applicable; only the applicable final-stage report indicates completion.

This task performed source/configuration collection and static checks only. No training, weight loading, inference, metric computation, bootstrap, calibration, threshold selection or drawing was performed. Historical scientific completion is not a claim of an end-to-end run from this candidate.

## Environment, path setup and permission
Retained dependency records: development_records/P6/RUNTIME.json

Python absolute path literals were replaced only by required `R52_PATH_*` environment lookups (embedded literals use `expandvars`); their original AST is restored and compared in the static check. Text configs use `${R52_PATH_*}` tokens. `tools/materialize_path_config.py` can create a NEW expanded text-config copy without executing any experiment. `PATH_REQUIREMENTS.csv` has safe binding hints; exact local paths are kept only in the private outer mapping. Output bindings must never target historical folders.

Parent-derived navigation, historical code-hash guards and expired windows are preserved, not silently disabled. They are explicit remaining IO/setup work, not scientific changes. A passed syntax check does not clear these conditions.

License: AUTHOR_CONFIRMATION_REQUIRED for this assembled unit. The pre-existing MIT file belongs only to its recorded local-release subtree; third-party components and data terms are separate. No online publication was attempted.

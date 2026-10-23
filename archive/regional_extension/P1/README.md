# regional_extension/P1

## Research role
Main 3.7,Table 5B; Supplementary Tables S34-S36.

R_IMG/R_CLIN/XATTN seed42 v1 development; M0 only read-only FIM FULL validation reference.

Score/input identity: HAM validation NATIVE only; two binary heads plus multiclass selection; no external model selection.

## Retained implementation and order
p1_checks.py -> p1_training.py -> p1_finish.py; region_readout_reference.py and model_code remain separate unchanged scientific sources

These are recorded historical entrypoints or static parser-derived invocation forms, **not commands executed by this packaging task**. Do not execute from this read-only review copy. Use a new workspace only after the local inputs, author permissions, execution window and relocated integrity records are resolved. Auto-training/evaluation imports and top-level script effects remain in historical code; do not use `--help` as a safety test.

Actual source files: 12; effective/source configs: 4. Exact definitions, lines, parser arguments, imports and SHA-256 are in `evidence_index/ENTRYPOINT_CATALOG.csv`, `IMPORT_DEPENDENCY_INVENTORY.csv` and `SOURCE_FINGERPRINTS.csv` at candidate root. Prefix entries with this unit's relative path.

## Inputs and outputs
Required, not bundled: HAM train/validation, fixed auxiliary100, training-before initial_state_s42.pt; reference region module.

Retained output names: VALIDATION_BINARY_METRICS.csv, VALIDATION_SUMMARY.csv, SELECTED_BEST_READOUT_DIAGNOSTICS.csv; per-run configs/history.

Training initialization and auxiliary IDs withheld; prior stage paths/expired window require local setup. Do not call old FIM finish_pipeline as a substitute.

## Evidence and validation
Historical completion evidence: P1_DEV_REPORT.md; P1_PROGRESS.json; P1_INITIALIZATION.json; effective_config.json. Copies retain historical status/date, including earlier NOT_STARTED/draft fields where applicable; only the applicable final-stage report indicates completion.

This task performed source/configuration collection and static checks only. No training, weight loading, inference, metric computation, bootstrap, calibration, threshold selection or drawing was performed. Historical scientific completion is not a claim of an end-to-end run from this candidate.

## Environment, path setup and permission
Retained dependency records: No separate complete environment lock located; inspect effective config and history. Unknown versions remain unrecorded.

Python absolute path literals were replaced only by required `R52_PATH_*` environment lookups (embedded literals use `expandvars`); their original AST is restored and compared in the static check. Text configs use `${R52_PATH_*}` tokens. `tools/materialize_path_config.py` can create a NEW expanded text-config copy without executing any experiment. `PATH_REQUIREMENTS.csv` has safe binding hints; exact local paths are kept only in the private outer mapping. Output bindings must never target historical folders.

Parent-derived navigation, historical code-hash guards and expired windows are preserved, not silently disabled. They are explicit remaining IO/setup work, not scientific changes. A passed syntax check does not clear these conditions.

License: AUTHOR_CONFIRMATION_REQUIRED for this assembled unit. The pre-existing MIT file belongs only to its recorded local-release subtree; third-party components and data terms are separate. No online publication was attempted.

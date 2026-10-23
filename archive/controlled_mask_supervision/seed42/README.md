# controlled_mask_supervision/seed42

## Research role
Main 3.7,4.6,Table 5A; Supplementary Tables S23-S27.

FIM_LPAH_FULL_s42 versus FIM_LPAH_ZERO_s42; separate controlled experiment, not original B6.

Score/input identity: Both heads; HAM NATIVE and SIIM E_REAL/E_NULL; FULL loss0.1 versus ZERO0 with auxiliary stream retained.

## Retained implementation and order
Historical prepare_fim.py -> initialize_fim.py -> fim_training.py -> fim_evaluation.py -> fim_statistics.py. Initializer can download public weights historically; it was not invoked here.

These are recorded historical entrypoints or static parser-derived invocation forms, **not commands executed by this packaging task**. Do not execute from this read-only review copy. Use a new workspace only after the local inputs, author permissions, execution window and relocated integrity records are resolved. Auto-training/evaluation imports and top-level script effects remain in historical code; do not use `--help` as a safety test.

Actual source files: 11; effective/source configs: 3. Exact definitions, lines, parser arguments, imports and SHA-256 are in `evidence_index/ENTRYPOINT_CATALOG.csv`, `IMPORT_DEPENDENCY_INVENTORY.csv` and `SOURCE_FINGERPRINTS.csv` at candidate root. Prefix entries with this unit's relative path.

## Inputs and outputs
Required, not bundled: Fixed HAM split42, fixed fim_mask_train100.csv, same initial_state_s42.pt and INITIALIZATION.json pretrained identity.

Retained output names: FIM_PROGRESS.json, per-run effective_config.json/status.json/training_history.csv, METRICS.csv, intervals and validation-derived thresholds.

Auxiliary member list, initial state and weights withheld. The new controlled auxiliary100 is not a reconstruction of original July auxiliary100.

## Evidence and validation
Historical completion evidence: FIM_TRAIN_REPORT.md; FIM_PROGRESS.json; INITIALIZATION.json; per-run effective_config.json; INPUT_SOURCES.json. Copies retain historical status/date, including earlier NOT_STARTED/draft fields where applicable; only the applicable final-stage report indicates completion.

This task performed source/configuration collection and static checks only. No training, weight loading, inference, metric computation, bootstrap, calibration, threshold selection or drawing was performed. Historical scientific completion is not a claim of an end-to-end run from this candidate.

## Environment, path setup and permission
Retained dependency records: No separate complete environment lock located; inspect effective config and history. Unknown versions remain unrecorded.

Python absolute path literals were replaced only by required `R52_PATH_*` environment lookups (embedded literals use `expandvars`); their original AST is restored and compared in the static check. Text configs use `${R52_PATH_*}` tokens. `tools/materialize_path_config.py` can create a NEW expanded text-config copy without executing any experiment. `PATH_REQUIREMENTS.csv` has safe binding hints; exact local paths are kept only in the private outer mapping. Output bindings must never target historical folders.

Parent-derived navigation, historical code-hash guards and expired windows are preserved, not silently disabled. They are explicit remaining IO/setup work, not scientific changes. A passed syntax check does not clear these conditions.

License: AUTHOR_CONFIRMATION_REQUIRED for this assembled unit. The pre-existing MIT file belongs only to its recorded local-release subtree; third-party components and data terms are separate. No online publication was attempted.

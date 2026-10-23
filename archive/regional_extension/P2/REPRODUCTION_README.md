# regional_extension/P2

## Research role
Main Table 5B; Supplementary Tables S34-S36.

R_IMG/R_CLIN/XATTN seeds2024/3407 plus P1 seed42 reuse; M0 validation reference separate.

Score/input identity: Validation only; both heads; original macro-F1 then MEL-AUC then earlier-epoch best selection.

## Retained implementation and order
p2_prepare.py/p2_checks.py -> p2_continue_lowmem.py (lightweight p2_worker_data.py) -> p2_finish.py. p2_runner.py retained for historical linkage; final low-memory path is the actual continuation route.

These are recorded historical entrypoints or static parser-derived invocation forms, **not commands executed by this packaging task**. Do not execute from this read-only review copy. Use a new workspace only after the local inputs, author permissions, execution window and relocated integrity records are resolved. Auto-training/evaluation imports and top-level script effects remain in historical code; do not use `--help` as a safety test.

Actual source files: 16; effective/source configs: 7. Exact definitions, lines, parser arguments, imports and SHA-256 are in `evidence_index/ENTRYPOINT_CATALOG.csv`, `IMPORT_DEPENDENCY_INVENTORY.csv` and `SOURCE_FINGERPRINTS.csv` at candidate root. Prefix entries with this unit's relative path.

## Inputs and outputs
Required, not bundled: P1 protocol/data; initial_state_s2024.pt and initial_state_s3407.pt; new modules seed+1000; frozen split42.

Retained output names: PER_SEED_VALIDATION_METRICS.csv, THREE_SEED_VALIDATION_SUMMARY.csv, RUN_IDENTITIES.csv.

Original pre-training states, prior run products and member lists withheld; no seed42 replay; old code/source integrity checks require a separately reviewed relocation manifest.

## Evidence and validation
Historical completion evidence: P2_TRAIN_REPORT.md; P2_PROGRESS.json; P2_PROTOCOL.json; effective configs; complete delivery identity recorded privately. Copies retain historical status/date, including earlier NOT_STARTED/draft fields where applicable; only the applicable final-stage report indicates completion.

This task performed source/configuration collection and static checks only. No training, weight loading, inference, metric computation, bootstrap, calibration, threshold selection or drawing was performed. Historical scientific completion is not a claim of an end-to-end run from this candidate.

## Environment, path setup and permission
Retained dependency records: No separate complete environment lock located; inspect effective config and history. Unknown versions remain unrecorded.

Python absolute path literals were replaced only by required `R52_PATH_*` environment lookups (embedded literals use `expandvars`); their original AST is restored and compared in the static check. Text configs use `${R52_PATH_*}` tokens. `tools/materialize_path_config.py` can create a NEW expanded text-config copy without executing any experiment. `PATH_REQUIREMENTS.csv` has safe binding hints; exact local paths are kept only in the private outer mapping. Output bindings must never target historical folders.

Parent-derived navigation, historical code-hash guards and expired windows are preserved, not silently disabled. They are explicit remaining IO/setup work, not scientific changes. A passed syntax check does not clear these conditions.

License: AUTHOR_CONFIRMATION_REQUIRED for this assembled unit. The pre-existing MIT file belongs only to its recorded local-release subtree; third-party components and data terms are separate. No online publication was attempted.

# original_derm7pt

## Research role
Main Tables 3,4A,6; Supplementary Tables S30-S33.

Same six original B2/B6 checkpoints as stage B2, NOT the 15 P3/P4 models.

Score/input identity: prob_mel primary, prob_mel_endpoint secondary; E_AVAILABLE/E_NULL, case/same-image grouping, not proven patient grouping.

## Retained implementation and order
core_prepare.py -> core_entry.py (core_infer.py) -> core_statistics.py; none executed in packaging

These are recorded historical entrypoints or static parser-derived invocation forms, **not commands executed by this packaging task**. Do not execute from this read-only review copy. Use a new workspace only after the local inputs, author permissions, execution window and relocated integrity records are resolved. Auto-training/evaluation imports and top-level script effects remain in historical code; do not use `--help` as a safety test.

Actual source files: 13; effective/source configs: 6. Exact definitions, lines, parser arguments, imports and SHA-256 are in `evidence_index/ENTRYPOINT_CATALOG.csv`, `IMPORT_DEPENDENCY_INVENTORY.csv` and `SOURCE_FINGERPRINTS.csv` at candidate root. Prefix entries with this unit's relative path.

## Inputs and outputs
Required, not bundled: Derm7pt release_v0/images and meta/meta.csv with exact path case; frozen original six models; stage-B2 DEVELOPMENT_THRESHOLDS_ORIGINAL.csv copy.

Retained output names: PER_RUN_METRICS.csv, THREE_RUN_SUMMARY.csv, FIXED_MODEL_INTERVALS.csv, THRESHOLD_COUNTS.csv, THRESHOLD_SUMMARY.csv.

Provider-authorized images/metadata and weights remain external; old ROOT/OUT hierarchy and expired deadline require new local invocation setup. No threshold from P3/P4 can substitute.

## Evidence and validation
Historical completion evidence: ORIGINALCORE_DERM7PT_REPORT.md; ORIGINALCORE_PROGRESS.json; ORIGINALCORE_PROTOCOL.json; FIXED_MODEL_IDENTITIES.json; THRESHOLD_IDENTITY_JOIN.csv. Copies retain historical status/date, including earlier NOT_STARTED/draft fields where applicable; only the applicable final-stage report indicates completion.

This task performed source/configuration collection and static checks only. No training, weight loading, inference, metric computation, bootstrap, calibration, threshold selection or drawing was performed. Historical scientific completion is not a claim of an end-to-end run from this candidate.

## Environment, path setup and permission
Retained dependency records: original_derm7pt/INFERENCE_ENVIRONMENT.json

Python absolute path literals were replaced only by required `R52_PATH_*` environment lookups (embedded literals use `expandvars`); their original AST is restored and compared in the static check. Text configs use `${R52_PATH_*}` tokens. `tools/materialize_path_config.py` can create a NEW expanded text-config copy without executing any experiment. `PATH_REQUIREMENTS.csv` has safe binding hints; exact local paths are kept only in the private outer mapping. Output bindings must never target historical folders.

Parent-derived navigation, historical code-hash guards and expired windows are preserved, not silently disabled. They are explicit remaining IO/setup work, not scientific changes. A passed syntax check does not clear these conditions.

License: AUTHOR_CONFIRMATION_REQUIRED for this assembled unit. The pre-existing MIT file belongs only to its recorded local-release subtree; third-party components and data terms are separate. No online publication was attempted.

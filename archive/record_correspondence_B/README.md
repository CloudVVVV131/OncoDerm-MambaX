# record_correspondence_B

## Research role
Main3.6.2,3.8,4.4.2,Table4B,Figure3B; Supplementary S18,TablesS42-S45.

Six original B2/B6 fixed checkpoints, SIIM equal-sized patient-block reassignment and Derm7pt case/same-image blocks.

Score/input identity: Both heads; matched/shuffled/null; K50,2000 stratified whole-block bootstraps with inner50; no change to six models.

## Retained implementation and order
b_prepare.py -> b_infer.py -> b_statistics.py points --cohort SIIM/Derm7pt -> b_statistics.py bootstrap --cohort SIIM/Derm7pt -> b_statistics.py combine. Slash notation denotes two separate invocations; actual choices are listed in ENTRYPOINT_CATALOG.csv.

These are recorded historical entrypoints or static parser-derived invocation forms, **not commands executed by this packaging task**. Do not execute from this read-only review copy. Use a new workspace only after the local inputs, author permissions, execution window and relocated integrity records are resolved. Auto-training/evaluation imports and top-level script effects remain in historical code; do not use `--help` as a safety test.

Actual source files: 13; effective/source configs: 6. Exact definitions, lines, parser arguments, imports and SHA-256 are in `evidence_index/ENTRYPOINT_CATALOG.csv`, `IMPORT_DEPENDENCY_INVENTORY.csv` and `SOURCE_FINGERPRINTS.csv` at candidate root. Prefix entries with this unit's relative path.

## Inputs and outputs
Required, not bundled: Authorized cohort tables/images, original model identities and clinical encoder; private score grids/donor pools rebuilt by historical scripts, not distributed.

Retained output names: PER_RUN_NATIVE_NULL_METRICS.csv, MATCHING_CONTRASTS.csv, MATCHING_INTERVALS.csv, REASSIGNMENT_METRICS.csv, FROZEN_THRESHOLD_RESULTS.csv.

Score grids, types, donor maps and records excluded. Historical b_common.py deadline preserved; cannot run unchanged in a later time window. Original statistics implementation hash is historical evidence, not a checksum of path-adapted code.

## Evidence and validation
Historical completion evidence: B_CLINICAL_MATCHING_REPORT.md; B_CLINICAL_MATCHING_PROGRESS.json; B_RUN_PROTOCOL.json; STATISTICS_IMPLEMENTATION_FREEZE.json; DELIVERY_QC.json. Copies retain historical status/date, including earlier NOT_STARTED/draft fields where applicable; only the applicable final-stage report indicates completion.

This task performed source/configuration collection and static checks only. No training, weight loading, inference, metric computation, bootstrap, calibration, threshold selection or drawing was performed. Historical scientific completion is not a claim of an end-to-end run from this candidate.

## Environment, path setup and permission
Retained dependency records: record_correspondence_B/ENVIRONMENT.json

Python absolute path literals were replaced only by required `R52_PATH_*` environment lookups (embedded literals use `expandvars`); their original AST is restored and compared in the static check. Text configs use `${R52_PATH_*}` tokens. `tools/materialize_path_config.py` can create a NEW expanded text-config copy without executing any experiment. `PATH_REQUIREMENTS.csv` has safe binding hints; exact local paths are kept only in the private outer mapping. Output bindings must never target historical folders.

Parent-derived navigation, historical code-hash guards and expired windows are preserved, not silently disabled. They are explicit remaining IO/setup work, not scientific changes. A passed syntax check does not clear these conditions.

License: AUTHOR_CONFIRMATION_REQUIRED for this assembled unit. The pre-existing MIT file belongs only to its recorded local-release subtree; third-party components and data terms are separate. No online publication was attempted.

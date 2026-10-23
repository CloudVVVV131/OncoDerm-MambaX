# fixed_clinical_inputs/B1

## Research role
Main Table 2C; Supplementary Tables S18,S28,S29.

B1 analysis stage of saved original predictions, NOT configuration B1 or a new model.

Score/input identity: Both original prediction heads retained with original endpoint labels.

## Retained implementation and order
b1_analysis.py --prepare, then b1_analysis.py --compute (or --resume for its retained state); invocation flags extracted from argparse, not run here

These are recorded historical entrypoints or static parser-derived invocation forms, **not commands executed by this packaging task**. Do not execute from this read-only review copy. Use a new workspace only after the local inputs, author permissions, execution window and relocated integrity records are resolved. Auto-training/evaluation imports and top-level script effects remain in historical code; do not use `--help` as a safety test.

Actual source files: 1; effective/source configs: 1. Exact definitions, lines, parser arguments, imports and SHA-256 are in `evidence_index/ENTRYPOINT_CATALOG.csv`, `IMPORT_DEPENDENCY_INVENTORY.csv` and `SOURCE_FINGERPRINTS.csv` at candidate root. Prefix entries with this unit's relative path.

## Inputs and outputs
Required, not bundled: Original saved predictions, patient linkage, accepted input/freeze identities; not distributed.

Retained output names: DIRECT_POINT_METRICS.csv, REPEATED_RUN_METRICS.csv, REPEATED_RUN_SUMMARY.csv, PAIRWISE_CLUSTERED_CI.csv, FIXED_RUN_MEAN_DIFFERENCE_CI.csv.

Private prediction/ID inputs and local freeze/navigation files are not bundled. Statistical run not repeated.

## Evidence and validation
Historical completion evidence: B1_REPORT.md; B1_STATUS.json; ENVIRONMENT.json; B1_FROZEN_PLAN.yaml. Copies retain historical status/date, including earlier NOT_STARTED/draft fields where applicable; only the applicable final-stage report indicates completion.

This task performed source/configuration collection and static checks only. No training, weight loading, inference, metric computation, bootstrap, calibration, threshold selection or drawing was performed. Historical scientific completion is not a claim of an end-to-end run from this candidate.

## Environment, path setup and permission
Retained dependency records: fixed_clinical_inputs/B1/ENVIRONMENT.json

Python absolute path literals were replaced only by required `R52_PATH_*` environment lookups (embedded literals use `expandvars`); their original AST is restored and compared in the static check. Text configs use `${R52_PATH_*}` tokens. `tools/materialize_path_config.py` can create a NEW expanded text-config copy without executing any experiment. `PATH_REQUIREMENTS.csv` has safe binding hints; exact local paths are kept only in the private outer mapping. Output bindings must never target historical folders.

Parent-derived navigation, historical code-hash guards and expired windows are preserved, not silently disabled. They are explicit remaining IO/setup work, not scientific changes. A passed syntax check does not clear these conditions.

License: AUTHOR_CONFIRMATION_REQUIRED for this assembled unit. The pre-existing MIT file belongs only to its recorded local-release subtree; third-party components and data terms are separate. No online publication was attempted.

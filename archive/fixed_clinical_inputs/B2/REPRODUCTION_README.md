# fixed_clinical_inputs/B2

## Research role
Main 3.6,4.4.1; Tables 3,4A,6; Supplementary Tables S19-S22.

Original B2/B6, training seeds42/2024/3407, original fixed best; NOT B2S, M0 or clean models.

Score/input identity: prob_mel and prob_mel_endpoint; SIIM E_REAL/E_NULL; own HAM-validation thresholds.

## Retained implementation and order
prepare_b2.py -> b2_inference.py technical -> b2_inference.py infer -> b2_statistics.py; --resume is a parser-defined statistical option

These are recorded historical entrypoints or static parser-derived invocation forms, **not commands executed by this packaging task**. Do not execute from this read-only review copy. Use a new workspace only after the local inputs, author permissions, execution window and relocated integrity records are resolved. Auto-training/evaluation imports and top-level script effects remain in historical code; do not use `--help` as a safety test.

Actual source files: 10; effective/source configs: 6. Exact definitions, lines, parser arguments, imports and SHA-256 are in `evidence_index/ENTRYPOINT_CATALOG.csv`, `IMPORT_DEPENDENCY_INVENTORY.csv` and `SOURCE_FINGERPRINTS.csv` at candidate root. Prefix entries with this unit's relative path.

## Inputs and outputs
Required, not bundled: Original six checkpoint identities, HAM validation/split metadata, SIIM source table/images, original 22-dimensional encoder.

Retained output names: B2_METRICS.csv, THREE_RUN_CONDITION_SUMMARY.csv, PAIRED_PATIENT_CLUSTER_CI.csv, DEVELOPMENT_THRESHOLDS.csv.

Fixed weights and local_only inputs intentionally omitted; historical validation checks and provenance dependencies require local binding before any future run.

## Evidence and validation
Historical completion evidence: B2_REPORT.md; B2_STATUS.json; B2_FROZEN_PROTOCOL.json; original_configs/; ENVIRONMENT.json and STATISTICAL_ENVIRONMENT.json. Copies retain historical status/date, including earlier NOT_STARTED/draft fields where applicable; only the applicable final-stage report indicates completion.

This task performed source/configuration collection and static checks only. No training, weight loading, inference, metric computation, bootstrap, calibration, threshold selection or drawing was performed. Historical scientific completion is not a claim of an end-to-end run from this candidate.

## Environment, path setup and permission
Retained dependency records: fixed_clinical_inputs/B2/ENVIRONMENT.json; fixed_clinical_inputs/B2/STATISTICAL_ENVIRONMENT.json

Python absolute path literals were replaced only by required `R52_PATH_*` environment lookups (embedded literals use `expandvars`); their original AST is restored and compared in the static check. Text configs use `${R52_PATH_*}` tokens. `tools/materialize_path_config.py` can create a NEW expanded text-config copy without executing any experiment. `PATH_REQUIREMENTS.csv` has safe binding hints; exact local paths are kept only in the private outer mapping. Output bindings must never target historical folders.

Parent-derived navigation, historical code-hash guards and expired windows are preserved, not silently disabled. They are explicit remaining IO/setup work, not scientific changes. A passed syntax check does not clear these conditions.

License: AUTHOR_CONFIRMATION_REQUIRED for this assembled unit. The pre-existing MIT file belongs only to its recorded local-release subtree; third-party components and data terms are separate. No online publication was attempted.

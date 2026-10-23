# B1 delivery and reproduction

This is the completed 2026-10-11 B1 supplemental CPU analysis, not a new training or inference result.
Read B1_REPORT.md first, then PAIRWISE_CLUSTERED_CI.csv, REPEATED_RUN_SUMMARY.csv and B2_READINESS.md.

## Inputs and runtime

All source paths/hashes are in INPUT_MANIFEST.csv. The extraction used:
`${R52_PATH_B5240D1264E1}`.
The original ZIP is preserved. Private prediction and patient-mapping files remain in their original local locations, not in this package.
Python executable: `${R52_PATH_07EFDCD4B820}`. Exact installed versions are in ENVIRONMENT.json.
No dependencies were installed/upgraded. CPU worker=1 and BLAS threads=1.

## Reproduce only with a separate new directory

Do not run scripts in the completed output directory: writes use exclusive-create mode to prevent overwriting results.
To reproduce on this machine, create a new empty child directory directly under `${R52_PATH_0F70D48282E5}`, and place copies of `prepare_materials.py` and `b1_analysis.py` there. Keep all exact source paths available.
Run in order with the recorded Python: `prepare_materials.py`, `b1_analysis.py --prepare`, `b1_analysis.py --compute`.
The first script only verifies package files and reads Word OOXML. The second freezes hashes before metrics and performs the authorized CPU analysis.
These instructions are documentation only; no second full run was launched. On another machine, path constants and manifest navigation require explicit relocation; never silently substitute inputs.

Formal random stream: NumPy PCG64 seed20261011; exact string patient-ID lexicographic ordering; 2056 patient draws each replicate; 10000 valid replicates. First100 remain included. Separate numeric-validation stream seed20261012.
Local-only `checkpoint_*.npz` files contain bootstrap metric arrays, not model weights or patient predictions. Paired JSON files contain RNG state and hashes. If a future interrupted reproduction needs resumption, use its own `b1_analysis.py --resume`; this completed run needs no resume.

## Meaning of outputs

- Direct comparison: prob_mel only, split42/training seed42, all seven models.
- Repeated runs: B2/B4/B6, three seeds; both heads separately; sample SD uses ddof=1.
- All pairwise intervals nominal95%; two primary AP contrasts also97.5% approximate family adjustment. No p-values.
- Three-run mean CIs are conditional_on_fixed_three_runs, not seed-population inference.
- ROC/PR coordinates have no patient/image IDs. PR columns are recall(x), precision(y); ROC initial threshold is inf; PR last threshold is blank. Full thresholds are curve coordinates, not selected decision thresholds.
- FINAL_SOURCE_INTEGRITY.csv and FINAL_QC.json verify inputs and completed outputs. DELIVERY_MANIFEST_SHA256.csv verifies every payload except itself. ZIP_VERIFICATION.json and B1_RESULTS.zip.sha256 are outside the ZIP to avoid circular hashes.
- REUSE_AND_GAP_REGISTER.csv is static navigation of a separate existing research line, not new scientific metric computation or authorization.

`R3_1_TABLES.json` was an early incomplete working extraction and is not used or delivered. The corrected caption-based extraction is `R3_1_TABLES_BY_CAPTION.json`; the analysis targets MAIN_1, MAIN_2, MAIN_3 and SUPPLEMENTARY_S13.

No manuscripts, original plots/code/configs/results, weights, datasets or remote tasks were changed. No B2/C/D performed. Bootstrap arrays, original per-case predictions, clinical rows, images, credentials and .env files are excluded from the ZIP.

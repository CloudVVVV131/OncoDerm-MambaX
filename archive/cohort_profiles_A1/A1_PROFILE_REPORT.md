# A1_PROFILE_REPORT

## Execution status
**COMPLETE: A1 formal profiles and B_DESIGN_INPUTS. B inference NOT_STARTED; new training 0.**
First A1 launch under 00_START_HERE and master v1.2. Local CPU only. No manuscript edits, environment changes, weights, images, model predictions or performance computations. No bootstrap, donor mappings, threshold selection or calibration. Outputs are new files under `${R52_PATH_EDA0A4B2C0A1}`.

## Five-group descriptive summary
| Group | Rows | Melanoma n (%) | Valid age n | Age median [Q1, Q3], years | Unavailable age/sex/site n |
|---|---:|---:|---:|---|---|
| HAM_train | 6982 | 775 (11.10%) | 6945 | 50 [40, 65] | 37/33/137 |
| HAM_validation | 1513 | 164 (10.84%) | 1502 | 50 [40, 65] | 11/13/53 |
| HAM_internal_test | 1520 | 174 (11.45%) | 1511 | 50 [40, 65] | 9/11/44 |
| SIIM | 33126 | 584 (1.76%) | 33058 | 50 [40, 60] | 68/65/527 |
| Derm7pt | 1011 | 252 (24.93%) | 0 | NA: source column absent | 1011/0/0 |

Full source sex/site categories and missing-reason distributions are included; no top-k truncation. HAM labels are dx=mel; SIIM uses target, not diagnosis unknown; Derm7pt uses the frozen six positive diagnoses. Non-melanoma is not equivalent to benign. Quantiles use finite valid ages and linear interpolation; encoded missing age zero is not an observed age.

## Actual original encoding coverage
- HAM_train site: supported 6845; nonempty unsupported 0; raw missing 137; flag=1 137.
- HAM_validation site: supported 1460; nonempty unsupported 0; raw missing 53; flag=1 53.
- HAM_internal_test site: supported 1476; nonempty unsupported 0; raw missing 44; flag=1 44.
- SIIM site: supported 30245; nonempty unsupported 2354; raw missing 527; flag=1 527.
- Derm7pt site: supported 898; nonempty unsupported 113; raw missing 0; flag=1 0.
Observed raw-missing versus encoder-flag disagreements: 0 across field/row comparisons. Unknown one-hot and missing flag are reported separately. SIIM torso to trunk and Derm7pt unsupported head neck/buttocks rules are unchanged. All 44,152 in-memory encoded records were cross-checked by two metadata-only implementations; no tensors or case vectors were exported.

## SIIM patient structure
2056 explicit patient groups; images per patient: median 12, Q1 5, Q3 22, range 2-115.

- age: 997 patients have >1 observed value/category; 0 have observed/missing mixed rows.
- sex: 0 patients have >1 observed value/category; 0 have observed/missing mixed rows.
- site_raw: 1897 patients have >1 observed value/category; 218 have observed/missing mixed rows.
- site_encoded: 1896 patients have >1 observed value/category; 218 have observed/missing mixed rows.

Patient age summaries use 1056 fully observed, single-valued patients; sex summaries use 2054. Mixed missingness stays explicit. No lesion sites are unified and no first/mode/mean is used to create a patient vector.

Equal-size strata containing at least two patients cover 2037 patients and 31638 images (95.5081%). 19 singleton-size strata account for 1488 images. These are structural candidate-coverage facts only; no cases were excluded or mapped to donors.

Full-vector repetition: 167 distinct vectors; 159 occur in multiple patients. Exact frequency/denominator summaries are supplied for later design.

## Derm7pt structure
OriginalCore binding is reused: 1011 cases, 1011 case/same-image clusters, 16 full-vector types. Nine clinic=derm references remain one case each. No age field or verified patient identity is inferred. No repeated image audit was run.

## Delivery and QC
Formal table: COHORT_CLINICAL_CHARACTERISTICS.csv and its readable five-column TABLE.md; full categories: COHORT_FIELD_DISTRIBUTIONS.csv; encoding: CLINICAL_ENCODING_COVERAGE.csv and CATEGORY_DETAIL.csv; definitions: COHORT_SUMMARY_METHOD.md.
Route B inputs: SIIM_PATIENT_STRUCTURE.csv, CLINICAL_MATCHING_DESIGN_INPUTS.json and CLINICAL_MATCHING_PROTOCOL_DRAFT.md. Draft status is DRAFT_NOT_EXECUTED_NOT_FINAL_FROZEN. ChatGPT still determines exchange algorithm, K/RNG, unequal-block policy, donor/fixed-point rules, uncertainty and comparison family.

Limited QC verifies source identities/unchanged bytes, rows/labels, denominators, missing partitions, accepted counts, independent quantiles/categories/encoding, and group totals. SOURCE_MANIFEST.csv covers only inputs actually read, not a new freeze of all 526 sources. No manuscript/source was modified. Package CRC and member fingerprints are verified in the external DELIVERY_RECEIPT.json.

## Actual blockers and stop
Asset/identity/computation blockers: 0. Pending Route B design decisions are expected next-stage work, not A1 failures. ZIP includes reports, final progress, aggregates, methods and minimal reproducible scripts only. No raw metadata, per-case identifiers/clinical rows, vectors, predictions, images, weights or credentials. Stop here for ChatGPT review; do not start B or manuscript editing.

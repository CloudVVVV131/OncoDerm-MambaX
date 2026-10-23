# Cohort Summary Method

## Scope and sources
First A1 execution, metadata-only, local CPU. Five tables were read into a bounded in-memory cache and shared between A1 and B design summaries. SHA-256, size and modification time were checked before/after processing. Accepted Stage0 source registration was reused, not rehashed in full. R5.1 documents were not opened or edited.

- HAM_train: ham10000_seed42_train.csv; age=age, sex=sex, site=localization, label=dx.
- HAM_validation: ham10000_seed42_val.csv; age=age, sex=sex, site=localization, label=dx.
- HAM_internal_test: ham10000_seed42_test.csv; age=age, sex=sex, site=localization, label=dx.
- SIIM: ISIC_2020_Training_GroundTruth_v2.csv; age=age_approx, sex=sex, site=anatom_site_general_challenge, label=target.
- Derm7pt: meta.csv; age=age, sex=sex, site=location, label=diagnosis.

All source rows are retained. HAM split42 partitions are reported separately; their union is only an integrity check. Derm7pt uses all 1,011 original derm cases and the original case/same-image binding, not clinic photographs as additional cases. There is no new overlap or image audit. Labels are dx=mel (HAM), target=1 (SIIM), and the six positive diagnoses in OriginalCore ORIGINALCORE_PROTOCOL.json#labels (Derm7pt). Original diagnosis/method distributions are exported as aggregates. No labels or clinical values are inferred from images, notes or IDs.

## Missingness and denominators
Mutually exclusive priority: source column absent; stripped blank; case-insensitive NA token; explicit unknown token; numeric parse/nonfinite failure (age only); valid. NA tokens are ['n/a', 'na', 'nan', 'none', 'null']. Explicit unknown tokens are ['not specified', 'unk', 'unknown', 'unspecified']. This reproduces the accepted Stage0 rule in freeze_baseline.py:196-213. Whitespace is stripped for descriptive categories, with no category collapsing beyond that; exact source spellings remain in distributions.

Categorical percentages and unavailable proportions use all image/case rows. percent_of_valid is separate and only applies to valid categories. Missing-reason and source-category rows in COHORT_FIELD_DISTRIBUTIONS.csv are two alternative views, not additive. Source categories marked unknown are counted once in their source-category distribution and once in the independently labelled missing-reason view.

Finite ages only enter median/Q1/Q3 and range. Quantiles use numpy method=linear: index=(n-1)*q with linear interpolation. Age zero remains observed; no imputation, clipping, arbitrary upper-age exclusion or outlier deletion. Derm7pt age quantiles are NA, never computed from encoded zero. NA count fields distinguish undefined quantities from observed zero. Age-summary rows carry valid n and denominator, while value is the age statistic in years.

## Original clinical encoding
Original source: ${R52_PATH_4C5B82098B02}"DEFAULT_SEX_VALUES": [23, 23], "DEFAULT_LOCALIZATION_VALUES": [24, 40], "metadata_dim": [43, 51], "_norm_text": [54, 58], "encode_metadata": [61, 101]}.
Only metadata constants and pure metadata functions were AST-extracted; torch tensor allocation was changed to an equivalent float32 numpy allocation. No project imports, torch, GPU, model, checkpoint or prediction files were used. Every metadata vector was independently checked against a second numpy implementation in memory. No per-case vectors are exported.

The 22 dimensions retain age/100, 3 sex and 15 site one-hots, then age_missing/sex_missing/site_missing. Exact vocabulary/order and E_NULL are included in CLINICAL_MATCHING_DESIGN_INPUTS.json. HAM follows the original pandas CSV handling and encoder defaults. SIIM uses metadata_mapping.json and METADATA_TOKEN_MAPPING.csv (strip/lower/collapse whitespace, torso to trunk). Derm7pt uses CLINICAL_MAPPING.json and core_data.py:21-30: unsupported head neck/buttocks are passed as nonempty unsupported strings, so the one-hot becomes unknown but site_missing remains zero. Missing age is passed as NaN, then age=0 and age_missing=1.

Raw truly_missing, nonempty_unsupported and available_supported form a partition; encoded_unknown and missing_flag_1 are overlapping encoder outcomes and are not additional partition categories. encoded_unknown is NA for scalar age. Unknown-category membership is not equivalent to true missingness. Category-level aggregates and any observed raw-vs-flag differences are explicit. No encoder has been repaired. The original age encoder does not explicitly reject infinity after float conversion; no nonfinite encoded ages occurred in these inputs. Future datasets must not silently inherit an assumption of equivalence.

## Patient and case structure
SIIM groups only by its complete source patient_id, with 2056 groups. Image-weighted and patient-weighted quantities are kept separate. Equal-size-block coverage is merely the proportion in group-size strata containing at least two patients; it is not a donor map, a chosen exchange policy, an exclusion rule or proof of value change.

For each patient and field, count distinct valid values (0,1,>1) crossed with all_missing / values_no_missing / values_plus_missing. Encoded-site validity uses site_missing=0; nonempty unsupported values mapped to unknown remain valid in this encoded representation. Collapsing categories in the encoder does not mean the original sites agree. Sites are never unified within patient.

Patient-level age/sex summaries use only patients with exactly one distinct valid value AND no missing rows for that field. Other patients stay in explicit consistency/missingness states. No first/mode/mean is used to assign a patient vector. Full-vector counts use exact float32 equality across all 22 dimensions including missing flags. Shared-vector proportions separately use distinct vectors, all image rows, all patients and unique patient-vector links as denominators. Patient counts across different vectors are not additive.

Derm7pt reuses OriginalCore COHORT_BINDING.json and its SHA-bound COHORT_ALIGNMENT.csv (case-binding metadata despite the private_predictions directory name, not model predictions). Only membership, source derm/sex/location correspondence and cluster labels are read. There are 1011 existing clusters, not verified patients. No case_id is promoted to patient_id, and no raw image is reread. Row-level identities, vector values and clinical combinations remain private in memory.

## QC, privacy and reproducibility
Independent checks: csv.DictReader versus pandas; Counter versus pandas value_counts for all categories; numpy quantiles versus sorted-value interpolation; AST encoder versus independent numpy encoder; prior missing totals and SIIM mapping counts; cluster/size summations; source hashes before/after.
Python 3.12.14, numpy 2.3.5, pandas 3.0.1. No RNG, statistical tests, model performance, bootstrap, donor mapping or inference is executed. a1_profile.py and a1_inputs.json reproduce these aggregate outputs in a new output folder with the same explicitly named local inputs. Exclusive creation prevents overwriting previous output files.

ZIP uses an explicit whitelist of aggregates/methods/code/QC only. CRC and member SHA-256 are checked and recorded in an external DELIVERY_RECEIPT.json; the ZIP does not contain its own final hash. Input raw metadata, manuscript files, weights, images, predictions, credentials and individual identifiers are not packaged.

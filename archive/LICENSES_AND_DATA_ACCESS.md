# Licenses and Data Access

This is an unpublished local review candidate. No new blanket MIT/GPL/Apache grant is applied. The recorded LICENSE and THIRD_PARTY_NOTICES under `original_framework/local_release/` are preserved for that local historical distribution only. Their sentence excluding the DermaMamba comparator refers to that historical release, not to all sources in this candidate. The August local comparator port is included for author review and remains AUTHOR_CONFIRMATION_REQUIRED before redistribution; its original attribution is retained. A software license does not grant rights to images, metadata, checkpoints or collaborator material.

Provider entrances below are copied from existing access records, not newly fetched or legally reinterpreted:
- HAM10000: https://doi.org/10.7910/DVN/DBW86T ; historical packaging used by preparation code: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000 . Preserve the recorded archive format; no downloading was performed here.
- SIIM-ISIC2020: https://doi.org/10.34970/2020-ds01 ; historical JPEG packaging: https://www.kaggle.com/datasets/cdeotte/jpeg-melanoma-512x512 . Local-release notes record CC BY-NC4.0; author must verify the intended redistribution scope.
- ISIC2018 masks: https://challenge.isic-archive.com/data/ . Auxiliary100 membership and localization200 are different inputs.
- Derm7pt: https://derm.cs.sfu.ca/Download.html ; processing code https://github.com/jeremykawahara/derm7pt is not the image dataset. Existing acquisition report records academic use, CC BY-NC-ND4.0 and no image redistribution. Obtain provider authorization yourself; no login or credential handling is part of this package.
- timm/torchvision pretrained weights: identities/revisions/hashes are retained where recorded. Weight redistribution terms require separate approval; weights are excluded.

## Input schemas (no sample rows)
HAM split tables use image_id, lesion_id, dx, age, sex and localization. lesion_id is not a verified patient identifier. Partition membership/order remain fixed by retained identities; do not regenerate a different split as a substitute.

SIIM metadata uses image_name, patient_id, target, age_approx, sex and anatom_site_general_challenge. Original external evaluation uses 33126 images and 2056 patient groups; a patient may have different lesion sites.

Derm7pt uses case_num, diagnosis, derm, sex and location. There is no age column used as an observed clinical value and no verified patient ID. Preserve FCL/FCl and extension case. Use derm, not clinic; 1011 dermoscopic records, six specified melanoma diagnosis labels, and structured missingness rules in CLINICAL_MAPPING.json. This schema is not a per-case table.

The original22-dimensional encoder and condition-specific mappings are retained in each unit. Diagnosis/IDs are labels/linkage, never model clinical inputs. E_NULL means the structured missing vector, not all zeros.

Split/member manifests, image IDs, individual clinical rows, raw predictions and donor structures are intentionally withheld. File-level fingerprints identify inputs without supplying their member rows. Author must approve each permitted redistribution separately. Do not replace the historical original auxiliary100 with the controlled-study100 or localization200.

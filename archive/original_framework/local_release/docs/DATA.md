# Data Preparation

Obtain medical images, segmentation masks and metadata from the providers below
under their dataset terms. This repository supplies preparation code and split IDs.

## Sources

- HAM10000: [dataset record](https://doi.org/10.7910/DVN/DBW86T) and the
  [kmader packaging used by the preparation code](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000).
- SIIM-ISIC 2020: [official dataset](https://doi.org/10.34970/2020-ds01) and the
  [512-pixel JPEG packaging used for external inference](https://www.kaggle.com/datasets/cdeotte/jpeg-melanoma-512x512).
- ISIC 2018 segmentation images and masks:
  [official challenge downloads and terms](https://challenge.isic-archive.com/data/).

Archive identities and expected checksums are defined in `PUBLIC_SOURCES` in
`scripts/prepare_data.py`. The script accepts previously obtained archives and
verifies them before extraction. Use the specified archive versions and image
resolutions to match the recorded preprocessing.

## Fixed classification partitions

The six released ID manifests preserve the row ordering of the retained split
files. Join them to the metadata that you obtained from the provider:

```bash
python scripts/restore_splits.py --metadata data/raw/HAM10000/HAM10000_metadata.csv
```

| Split seed | Training images | Validation images | Test images |
|---|---:|---:|---:|
| 42 | 6,982 | 1,513 | 1,520 |
| 2024 | 7,013 | 1,515 | 1,487 |

The restore command validates unique image IDs, disjoint lesion groups and
complete coverage, while preserving each partition's membership and row order.
The output `splits/` directory must be new.

## Combined archive preparation

Alternatively, prepare HAM10000, the 33,126-image external cohort, the expanded
200-pair segmentation set, and fixed splits together. Use a new output directory
with sufficient disk space. Download the five archives first, then supply their
local paths to the preparation command:

```bash
python scripts/prepare_data.py --ham-zip downloads/skin-cancer-mnist-ham10000.zip --external-zip downloads/jpeg-melanoma-512x512.zip --validation-images-zip downloads/ISIC2018_Task1-2_Validation_Input.zip --validation-masks-zip downloads/ISIC2018_Task1_Validation_GroundTruth.zip --training-masks-zip downloads/ISIC2018_Task1_Training_GroundTruth.zip
python scripts/prepare_data.py --verify-only
```

This command fetches selected training image entries from the official ISIC 2018
training ZIP using HTTP ranges. The expanded set consists of all 100 validation
pairs and the lexicographically first 100 eligible training pairs. Image/mask
pairing and dimensions are verified. Identical duplicate archive entries are
deduplicated after content-hash verification; conflicting image IDs raise an error.

## Auxiliary supervision versus localization

Principal LPAH configurations expect a separately supplied 100-pair auxiliary
cohort under `data/raw/ISIC_masks_auxiliary/{images,masks}`. The original 100-ID
membership manifest is not included in this release. Matching the historical
auxiliary cohort requires that original membership; the expanded 200-pair set
has a separate preparation path.

The August B6-M and cross-partition configurations use the expanded 200-pair
directory `data/raw/ISIC_masks`. Figure S7 describes localization on this expanded
set using retained B4/B6 models. The evaluation provides descriptive auxiliary-task
localization metrics; a separate held-out segmentation cohort is outside this
protocol. Evaluation of the historical models uses their saved checkpoints.

External inference uses `labels_harmonized.csv` and images under
`data/raw/ISIC_external/`. Unavailable external metadata is encoded through the
implemented missing-value handling. Diagnostic inference accepts images and the
configured metadata; masks are used for auxiliary supervision and localization.

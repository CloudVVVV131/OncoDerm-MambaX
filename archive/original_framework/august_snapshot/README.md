# LesionMamba V0.3 Codebase

This repository implements the code plan for:

**LesionMamba: A Multi-Scale State Space Architecture for Explainable Skin Lesion Classification in Dermoscopic Images**

The implementation follows three local planning documents in the parent folder:

- `LesionMamba_方法论策划_轻量可运行版V0.3.md`
- `LesionMamba_实验设计策划_轻量可运行版V0.3.md`
- `LesionMamba_代码规划策划_轻量可运行版V0.3.md`

## Core Scope

- HAM10000 7-class classification.
- Melanoma vs non-melanoma endpoint from 7-class `prob_mel`.
- Five core baselines: ResNet-50, ConvNeXt-Tiny, EfficientNetV2-S/B0, Swin-Tiny, VMamba-Tiny.
- Proposed model: LesionMamba-Compact.
- A0-A3 ablations.
- Efficiency, calibration, Grad-CAM/internal attention, bootstrap CI.
- Conditional external ISIC inference and mask explainability when data is present.

## Data Layout

Place HAM10000 files at:

```text
data/raw/HAM10000/
  HAM10000_metadata.csv
  images/
```

Optional external files:

```text
data/raw/ISIC_external/labels_harmonized.csv
data/raw/ISIC_masks/
```

## Core Run

```powershell
.\run_all_core.ps1
```

## AutoDL Full Run

On an AutoDL PyTorch/CUDA image, put the data under the relative paths shown
above, then run the complete experiment pipeline with nohup:

```bash
bash -n run_autodl_full.sh
chmod +x run_autodl_full.sh
nohup bash run_autodl_full.sh > autodl_nohup.log 2>&1 &
tail -f autodl_nohup.log
```

The script runs core models, A0-A3 ablations, optional timm baselines,
efficiency, calibration, melanoma endpoint, Grad-CAM/internal attention,
conditional external/mask evaluation, and P2 experiments. It defaults to
`FRESH_RUN=1`, which archives old generated artifacts under `_autodl_archive/`
before starting. Use `FRESH_RUN=0` to keep existing outputs.

Useful switches:

```bash
INSTALL_DEPS=0 nohup bash run_autodl_full.sh > autodl_nohup.log 2>&1 &
RUN_OPTIONAL_BASELINES=0 nohup bash run_autodl_full.sh > autodl_nohup.log 2>&1 &
RUN_P2=0 nohup bash run_autodl_full.sh > autodl_nohup.log 2>&1 &
```

Or run individual scripts:

```bash
python scripts/00_make_splits.py --config configs/defaults.yaml
python scripts/01_train.py --config configs/models/resnet50.yaml
python scripts/01_train.py --config configs/models/lesionmamba_compact.yaml
python scripts/02_test.py --run outputs/runs/<run_id> --split val
python scripts/02_test.py --run outputs/runs/<run_id> --split test
python scripts/09_make_tables_and_figures.py
```

## Optional P2 Runs

```powershell
.\run_optional_p2.ps1
```

This runs robustness-lite, LesionMamba-only loss ablations, 384x384 fine-tuning for the strongest baseline plus LesionMamba when source runs exist, and the top-2 second-seed check.

## Submission V1 Supplement

The submission-track supplement implements **OncoDerm-MambaX** according to
`LesionMamba_补充方法论策划_投稿版V1.0.md` and
`LesionMamba_补充实验设计策划_投稿版V1.0.md` in the parent folder.

It adds:

- B0-B6 ablations from backbone-only to full CSSA + LPAH + CMGF.
- A melanoma endpoint head in addition to 7-class HAM10000 classification.
- Clinical metadata gated fusion for age, sex, and lesion localization.
- Mask auxiliary supervision through ISIC masks without requiring masks at inference.
- External ISIC endpoint validation, calibration, robustness, efficiency, qualitative Grad-CAM/attention, and mask explainability summaries.

Run the full supplement on AutoDL with:

```bash
chmod +x run_submission_v1.sh
nohup bash run_submission_v1.sh > autodl_submission_v1.log 2>&1 &
tail -f autodl_submission_v1.log
```

Useful switches:

```bash
INSTALL_DEPS=0 nohup bash run_submission_v1.sh > autodl_submission_v1.log 2>&1 &
MAX_MASKS=0 nohup bash run_submission_v1.sh > autodl_submission_v1.log 2>&1 &
TRAIN_MODELS=0 TEST_MODELS=1 RUN_ANALYSIS=1 nohup bash run_submission_v1.sh > autodl_submission_v1.log 2>&1 &
```

Primary supplement outputs:

```text
results/submission_v1_ablation_table.csv
results/submission_v1_summary_status.json
results/melanoma_endpoint_table.csv
results/external_test_table.csv
results/calibration_table.csv
results/explainability_mask_table.csv
results/robustness_lite_table.csv
results/efficiency_table.csv
```

## Clinical Scope

The model is intended for computer-aided dermoscopic image classification and decision support. It is not an autonomous diagnostic system.

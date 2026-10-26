# OncoDerm-MambaX

Research software for dermoscopic skin-lesion classification, melanoma endpoint
evaluation, and analysis of internal and external performance.

## Current Release: R5.2

**Start here for the R5.2 source and analysis archive.**

- [Browse the R5.2 source](https://github.com/CloudVVVV131/OncoDerm-MambaX/tree/release/r5.2-source-archive)
- [R5.2 release and downloads](https://github.com/CloudVVVV131/OncoDerm-MambaX/releases/tag/r5.2-source-archive)
- [Download the reviewed source ZIP](https://github.com/CloudVVVV131/OncoDerm-MambaX/releases/download/r5.2-source-archive/OncoDerm_R5_2_Source_Release_Candidate.zip)
- [R5.2 runbook](https://github.com/CloudVVVV131/OncoDerm-MambaX/blob/r5.2-source-archive/RUNBOOK.md)
- [Zenodo version DOI: 10.5281/zenodo.23604529](https://doi.org/10.5281/zenodo.23604529)
- [File-specific license scope](https://github.com/CloudVVVV131/OncoDerm-MambaX/blob/r5.2-source-archive/LICENSE_SCOPE.csv)

R5.2 is preserved on the dedicated `release/r5.2-source-archive` branch and
`r5.2-source-archive` tag. The `main` branch retains the earlier source layout;
its installation instructions and citation below describe that earlier version.
For R5.2, use the source and runbook linked above. The previous
[v0.3.0 release](https://github.com/CloudVVVV131/OncoDerm-MambaX/releases/tag/v0.3.0)
and its history remain available.

---

**Earlier source guide (v0.3.0 / retained main code)**

## Contents

- `lesionmamba/`: OncoDerm-MambaX, timm comparators, data loaders, losses and metrics.
- `configs/runs/`: 17 resolved configurations linked to recorded experiments in
  [the experiment index](configs/experiments.csv).
- `scripts/`: data preparation, training, evaluation, calibration, localization,
  perturbation analysis and profiling.
- `data/manifests/`: image-ID-only HAM10000 partitions for split seeds 42 and 2024.
- `figures/`: portable renderers and aggregate data for Figure 3 and Figures S2-S5/S7.

The Python import namespace is `lesionmamba`. The principal B0-B6 implementation
uses an axial depthwise-convolution token mixer; the optional B6-M configuration
uses the official `mamba-ssm` block. These are distinct implementations.

## Installation

Clone this repository and run commands from its root. Use a separate Python
environment and install a matching PyTorch/torchvision pair for your hardware
from the [official PyTorch distribution](https://pytorch.org/get-started/locally/).

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

Recorded GPU environments and optional B6-M installation are described in
[ENVIRONMENT.md](docs/ENVIRONMENT.md). The principal B0-B6 models use standard
PyTorch operations. Figure rendering can be installed separately:

```bash
python -m pip install -r requirements-figures.txt
python figures/render_figures.py
python figures/render_s7.py
```

## Data and Training

Follow [DATA.md](docs/DATA.md) to obtain the datasets and restore fixed partitions.
The repository contains source code, configurations, split IDs and aggregate
figure data. Obtain images, masks and metadata from the linked dataset providers;
checkpoints and row-level predictions are separate inputs for model evaluation.

Example using the B2 configuration (no auxiliary-mask stream):

```bash
python scripts/01_train.py --config configs/runs/oncoderm_mambax_b2_metadata_cmgf_seed42.yaml
```

The command prints a new `outputs/runs/<run_id>` directory. Use the printed
directory for subsequent commands:

```bash
python scripts/02_test.py --run outputs/runs/<run_id> --split test
python scripts/03_eval_melanoma_endpoint.py --selected-runs outputs/runs/<run_id>
python scripts/05_temperature_scaling.py --selected-runs outputs/runs/<run_id>
python scripts/08_external_inference.py --selected-runs outputs/runs/<run_id>
python scripts/04_measure_efficiency.py --selected-runs outputs/runs/<run_id>
```

Checkpoint selection, calibration and operating thresholds use the validation
partition. External evaluation applies the resulting model and thresholds unchanged.
[REPRODUCTION.md](docs/REPRODUCTION.md) explains score definitions, mask-cohort
requirements, repeatability and the scope of the released analyses.

## Scope and Use

This software supports research on dermoscopic classification and model
evaluation. Record the code revision, configuration, dataset version and runtime
for each experiment. Performance tables describe the evaluated cohorts and
configurations. The source distribution covers OncoDerm-MambaX and timm-backed
comparators; the DermaMamba comparison implementation is outside this distribution.

## Citation and Contributions

Version 0.3.0 is archived on Zenodo with the version-specific DOI
[10.5281/zenodo.22678600](https://doi.org/10.5281/zenodo.22678600).

Use [CITATION.cff](CITATION.cff) for the software citation. Report reproducibility
questions through [Issues](https://github.com/CloudVVVV131/OncoDerm-MambaX/issues)
using reproducible examples and redacted logs.

## License

Original source code is licensed under the [MIT License](LICENSE).
Third-party software, datasets and pretrained weights retain their respective
terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

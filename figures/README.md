# Figure renderers

Run from the repository root:

```bash
python -m pip install -r requirements-figures.txt
python figures/render_figures.py
python figures/render_s7.py
```

Both commands read released aggregate source data and write editable-text SVG,
PDF and 600 dpi PNG to `build/figures/`. Use a new or empty output directory.
Use `--help` to select another output directory for subsequent renders.

| Figure | Released input files in `source_data/` |
| --- | --- |
| Figure 3 | `Figure3_threshold_transport_source.csv`, `Figure3_transport_matrix_source.csv` |
| Figure S2 | `FigureS2_configuration_source.csv`, `FigureS2_direction_aligned_zscores.csv` |
| Figure S3 | `FigureS3_confusion_source.csv`, `FigureS3_perturbation_source.csv` |
| Figure S4 | `FigureS4_curves_source.csv`, `FigureS4_external_systems_source.csv`, `FigureS4_calibration_descriptors_source.csv`, `FigureS4_inference_burden_source.csv`, `FigureS4_member_burden_inputs.csv` |
| Figure S5 | `FigureS5_three_seed_runs_source.csv`, `FigureS5_efficiency_source.csv`, `common_environment_efficiency.csv` |
| Figure S7 | `FigureS7_source_data.csv`, `mask200_table.csv`, `mask200_status.json`, `S7_saliency_id_manifest.csv` |

Companion tables provide aggregate inputs and intermediate descriptors alongside
the tables used directly by the renderers. S7 presents quantitative localization
metrics. Its ID manifest contains public dataset identifiers and stored map hashes.

Axes, statistics and labels follow the retained figure definitions. The renderers
use stored curve coordinates and summary statistics; prediction generation and
statistical estimation are upstream steps. A figure-rendering environment is
separate from the recorded model training environments in `docs/ENVIRONMENT.md`.

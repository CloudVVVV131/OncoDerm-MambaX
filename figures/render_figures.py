"""Render selected frozen paper figures from released aggregate source data."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "source_data"
MAIN = SUPP = Path(__file__).resolve().parents[1] / "build/figures"

def read_csv(path):
    return pd.read_csv(path)

def save_figure(fig, directory, stem):
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "pdf", "png"):
        target = directory / f"{stem}.{extension}"
        if target.exists():
            raise FileExistsError(target)
        fig.savefig(target, dpi=600)
    plt.close(fig)

MM = 1 / 25.4
FULL_WIDTH_MM = 183
COLORS = {'ink': '#202124', 'gray': '#687178', 'mid_gray': '#9AA0A4', 'light_gray': '#DDE2E5', 'grid': '#E8ECEF', 'B0': '#7F8588', 'B1': '#9AA0A4', 'B2': '#0072B2', 'B3': '#8A8F92', 'B4': '#E69F00', 'B5': '#B1B4B6', 'B6': '#D55E00', 'DermaMamba': '#4D5960', 'Swin-Tiny': '#8A8F92', 'all6': '#7A7088', 'teal': '#009E73', 'rose': '#B35C6E', 'blue_fill': '#EAF3F8', 'orange_fill': '#FFF2E5'}
MODEL_LABELS = {'B0': 'B0 backbone', 'B1': 'ConvNeXt-Tiny (B1)', 'B2': 'B2 CMGF', 'B3': 'B3 CSSA', 'B4': 'B4 LPAH', 'B5': 'B5 CSSA+LPAH', 'B6': 'B6 full', 'DermaMamba': 'DermaMamba port', 'Swin-Tiny': 'Swin-Tiny'}

def configure_plotting() -> None:
    mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'], 'font.size': 7.2, 'axes.titlesize': 8.2, 'axes.labelsize': 7.3, 'xtick.labelsize': 6.6, 'ytick.labelsize': 6.6, 'legend.fontsize': 6.4, 'axes.linewidth': 0.75, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7, 'xtick.major.size': 3, 'ytick.major.size': 3, 'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'ps.fonttype': 42})

def clean_axis(ax: plt.Axes, grid: bool=False) -> None:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out', pad=2)
    if grid:
        ax.grid(color=COLORS['grid'], linewidth=0.55, zorder=0)

def panel_label(ax: plt.Axes, label: str, x: float=-0.12, y: float=1.08) -> None:
    ax.text(x, y, label, transform=ax.transAxes, ha='left', va='top', fontsize=10.2, fontweight='bold', color=COLORS['ink'], clip_on=False)

def build_figure3() -> None:
    directory = MAIN / 'Figure3_threshold_transport'
    directory.mkdir(parents=True, exist_ok=True)
    source_path = DATA / 'Figure3_threshold_transport_source.csv'
    data = read_csv(source_path)
    policy_order = ['youden', 'specificity_90', 'specificity_95', 'sensitivity_80']
    policy_labels = {'youden': 'Youden', 'specificity_90': 'Spec 90%', 'specificity_95': 'Spec 95%', 'sensitivity_80': 'Sens 80%'}
    dataset_order = ['val', 'test', 'external']
    lookup = data.set_index(['threshold_policy', 'dataset'])
    rows = [('Validation sensitivity', 'val', 'sensitivity'), ('Internal-test sensitivity', 'test', 'sensitivity'), ('External sensitivity', 'external', 'sensitivity'), ('External specificity', 'external', 'specificity')]
    matrix = np.array([[lookup.loc[(policy, dataset), metric] for policy in policy_order] for _, dataset, metric in rows], dtype=float)
    matrix_source = pd.DataFrame(matrix, index=[r[0] for r in rows], columns=[policy_labels[p] for p in policy_order])
    fig = plt.figure(figsize=(FULL_WIDTH_MM * MM, 78 * MM), layout='constrained')
    grid = fig.add_gridspec(1, 3, width_ratios=[1.17, 1.08, 1.02], wspace=0.22)
    ax_heat = fig.add_subplot(grid[0, 0])
    ax_dumb = fig.add_subplot(grid[0, 1])
    ax_map = fig.add_subplot(grid[0, 2])
    cmap = LinearSegmentedColormap.from_list('transport', ['#F5F7F8', '#B9D9E8', COLORS['B2']])
    image = ax_heat.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect='auto')
    compact_policy_labels = {'youden': 'Youden', 'specificity_90': 'Spec ≥ 0.90', 'specificity_95': 'Spec ≥ 0.95', 'sensitivity_80': 'Sens ≥ 0.80'}
    compact_row_labels = ['Validation sens.', 'Internal-test sens.', 'External sens.', 'External spec.']
    heat_policy_labels = ['Youden', 'Spec ≥\n0.90', 'Spec ≥\n0.95', 'Sens ≥\n0.80']
    ax_heat.set_xticks(range(4), heat_policy_labels)
    ax_heat.set_yticks(range(4), compact_row_labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax_heat.text(j, i, f'{value:.3f}', ha='center', va='center', fontsize=6.5, color='white' if value > 0.66 else COLORS['ink'])
    ax_heat.set_title('Validation-fixed operating policies', loc='left', pad=5, fontweight='bold')
    for spine in ax_heat.spines.values():
        spine.set_visible(False)
    ax_heat.tick_params(length=0)
    panel_label(ax_heat, 'a', x=-0.18)
    y_positions = np.arange(len(policy_order))[::-1]
    for ypos, policy in zip(y_positions, policy_order):
        val = float(lookup.loc[(policy, 'val'), 'sensitivity'])
        external = float(lookup.loc[(policy, 'external'), 'sensitivity'])
        ax_dumb.plot([external, val], [ypos, ypos], color=COLORS['light_gray'], linewidth=2.0, zorder=1)
        ax_dumb.plot(val, ypos, marker='o', markersize=5.2, markerfacecolor='white', markeredgecolor=COLORS['B2'], markeredgewidth=1.2, zorder=3)
        ax_dumb.plot(external, ypos, marker='o', markersize=5.2, color=COLORS['B6'], zorder=3)
        ax_dumb.text(1.225, ypos, f'{external - val:+.3f}', va='center', ha='right', fontsize=6.0, color=COLORS['gray'])
    ax_dumb.set_yticks(y_positions, [compact_policy_labels[p] for p in policy_order])
    ax_dumb.set_xlim(0.05, 1.24)
    ax_dumb.set_ylim(-0.35, 3.52)
    ax_dumb.set_xlabel('Sensitivity')
    ax_dumb.set_title('Sensitivity transport', loc='left', pad=5, fontweight='bold')
    first_val = float(lookup.loc[('youden', 'val'), 'sensitivity'])
    first_external = float(lookup.loc[('youden', 'external'), 'sensitivity'])
    ax_dumb.text(first_external, 3.22, 'ext.', ha='center', va='bottom', fontsize=5.4, color=COLORS['B6'])
    ax_dumb.text(first_val, 3.22, 'val.', ha='center', va='bottom', fontsize=5.4, color=COLORS['B2'])
    ax_dumb.text(1.225, 3.22, 'Δ', ha='right', va='bottom', fontsize=5.4, color=COLORS['gray'])
    clean_axis(ax_dumb, grid=True)
    panel_label(ax_dumb, 'b', x=-0.16)
    external_rows = data[data['dataset'] == 'external'].set_index('threshold_policy').loc[policy_order]
    point_colors = [COLORS['gray'], COLORS['B2'], COLORS['B6'], COLORS['B4']]
    point_numbers = {policy: index for index, policy in enumerate(policy_order, start=1)}
    for (policy, row), color in zip(external_rows.iterrows(), point_colors):
        size = 35 + 230 * float(row['ppv'])
        ax_map.scatter(row['sensitivity'], row['specificity'], s=size, color=color, edgecolor='white', linewidth=0.7, zorder=3)
        ax_map.text(row['sensitivity'], row['specificity'], str(point_numbers[policy]), ha='center', va='center', fontsize=5.2, color='white', fontweight='bold', zorder=4)
    ax_map.set(xlabel='External sensitivity', ylabel='External specificity', xlim=(0.095, 0.19), ylim=(0.987, 0.9982))
    ax_map.set_title('External operating map', loc='left', pad=5, fontweight='bold')
    ax_map.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.3f'))
    clean_axis(ax_map, grid=True)
    panel_label(ax_map, 'c', x=-0.18)
    ax_map.text(0.5, -0.19, '1 Youden · 2 Sp≥.90 · 3 Sp≥.95 · 4 Se≥.80 | area=PPV', transform=ax_map.transAxes, ha='center', va='top', fontsize=5.2, color=COLORS['gray'])
    save_figure(fig, directory, 'Figure3_threshold_transport')

def build_figure_s2() -> None:
    directory = SUPP / 'FigureS2_configuration_specialization'
    directory.mkdir(parents=True, exist_ok=True)
    data = read_csv(DATA / 'FigureS2_configuration_source.csv')
    models = [f'B{i}' for i in range(7)]
    data = data.set_index('Model').loc[models].reset_index()
    module_columns = ['CSSA', 'LPAH', 'CMGF']
    metrics = ['Macro-F1', 'Internal Mel-AUC', 'External Mel-AUC', 'External PR-AUC', 'External Brier', 'External ECE']
    short = ['Macro-F1', 'Internal\nMEL AUC', 'External\nROC AUC', 'External\nPR AUC', 'Brier', 'ECE']
    raw = data[metrics].to_numpy(dtype=float)
    oriented = raw.copy()
    oriented[:, -2:] *= -1
    standardized = (oriented - oriented.mean(axis=0)) / oriented.std(axis=0, ddof=0)
    z_source = pd.DataFrame(standardized, columns=[f'oriented_z_{name}' for name in metrics])
    fig = plt.figure(figsize=(FULL_WIDTH_MM * MM, 83 * MM), layout='constrained')
    grid = fig.add_gridspec(1, 2, width_ratios=[0.72, 2.65], wspace=0.06)
    ax_modules = fig.add_subplot(grid[0, 0])
    ax_metrics = fig.add_subplot(grid[0, 1])
    module_matrix = data[module_columns].astype(int).to_numpy()
    module_cmap = LinearSegmentedColormap.from_list('modules', ['#F4F5F6', '#4C7E93'])
    ax_modules.imshow(module_matrix, cmap=module_cmap, vmin=0, vmax=1, aspect='auto')
    for i in range(len(models)):
        for j in range(len(module_columns)):
            ax_modules.text(j, i, '●' if module_matrix[i, j] else '–', ha='center', va='center', fontsize=7.5, color='white' if module_matrix[i, j] else COLORS['mid_gray'])
    ax_modules.set_xticks(range(3), module_columns, rotation=35, ha='right', rotation_mode='anchor')
    ax_modules.set_yticks(range(7), models)
    ax_modules.set_title('Configuration', loc='left', pad=5, fontweight='bold')
    ax_modules.tick_params(length=0)
    for spine in ax_modules.spines.values():
        spine.set_visible(False)
    panel_label(ax_modules, 'a', x=-0.3)
    cmap = LinearSegmentedColormap.from_list('direction', ['#B96B63', '#F7F7F6', '#367D9A'])
    norm = TwoSlopeNorm(vmin=-2.0, vcenter=0.0, vmax=2.0)
    image = ax_metrics.imshow(standardized, cmap=cmap, norm=norm, aspect='auto')
    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            ax_metrics.text(j, i, f'{raw[i, j]:.3f}', ha='center', va='center', fontsize=6.2, color=COLORS['ink'])
    ax_metrics.set_xticks(range(len(metrics)), short)
    ax_metrics.set_yticks(range(7), [''] * 7)
    ax_metrics.set_title('Direction-aligned within-endpoint standardization', loc='left', pad=5, fontweight='bold')
    ax_metrics.tick_params(length=0)
    for spine in ax_metrics.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, ax=ax_metrics, fraction=0.025, pad=0.015)
    cbar.set_label('Higher = more favorable', rotation=270, rotation_mode='anchor', labelpad=10)
    ax_metrics.text(0.0, -0.12, 'Brier and ECE are sign-reversed before standardization; cell labels show raw values.', transform=ax_metrics.transAxes, fontsize=6.3, color=COLORS['gray'], va='top')
    save_figure(fig, directory, 'FigureS2_configuration_specialization')

def build_figure_s3() -> None:
    directory = SUPP / 'FigureS3_perturbation_and_confusion'
    directory.mkdir(parents=True, exist_ok=True)
    robustness = read_csv(DATA / 'FigureS3_perturbation_source.csv')
    confusion = read_csv(DATA / 'FigureS3_confusion_source.csv')
    robustness = robustness[(robustness['condition'] != 'clean') & robustness['model_id'].isin(['B0', 'B3', 'B6'])].copy()
    condition_order = list(dict.fromkeys(robustness['condition_label'].tolist()))
    model_order = ['B0', 'B3', 'B6']
    macro = robustness.pivot(index='condition_label', columns='model_id', values='macro_f1_drop').reindex(index=condition_order, columns=model_order)
    mel = robustness.pivot(index='condition_label', columns='model_id', values='mel_auc_drop').reindex(index=condition_order, columns=model_order)
    classes = list(dict.fromkeys(confusion['true_class'].tolist()))
    cm = confusion.pivot(index='true_class', columns='predicted_class', values='row_fraction').reindex(index=classes, columns=classes).fillna(0)
    fig = plt.figure(figsize=(FULL_WIDTH_MM * MM, 92 * MM), layout='constrained')
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.28], wspace=0.11)
    ax_macro = fig.add_subplot(grid[0, 0])
    ax_mel = fig.add_subplot(grid[0, 1])
    ax_cm = fig.add_subplot(grid[0, 2])
    drop_cmap = LinearSegmentedColormap.from_list('drop', ['#DDEFE8', '#FAFAF8', '#B95F55'])
    max_abs = max(abs(macro.to_numpy()).max(), abs(mel.to_numpy()).max(), 0.001)
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)
    for ax, matrix, title in ((ax_macro, macro, 'Macro-F1 drop'), (ax_mel, mel, 'MEL AUC drop')):
        image = ax.imshow(matrix.to_numpy(), cmap=drop_cmap, norm=norm, aspect='auto')
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f'{matrix.iloc[i, j]:+.3f}', ha='center', va='center', fontsize=6.1)
        ax.set_xticks(range(3), model_order)
        ax.set_title(title, loc='left', pad=5, fontweight='bold')
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    ax_macro.set_yticks(range(len(condition_order)), [label.replace('Brightness Shift', 'Brightness').replace('Gaussian Blur', 'Blur') for label in condition_order])
    ax_mel.set_yticks(range(len(condition_order)), [''] * len(condition_order))
    panel_label(ax_macro, 'a', x=-0.28)
    cbar = fig.colorbar(image, ax=[ax_macro, ax_mel], fraction=0.025, pad=0.015)
    cbar.set_label('Clean minus perturbed', rotation=270, rotation_mode='anchor', labelpad=10)
    cm_image = ax_cm.imshow(cm.to_numpy(), cmap='Blues', vmin=0, vmax=1, aspect='equal')
    for i in range(len(classes)):
        for j in range(len(classes)):
            value = cm.iloc[i, j]
            count = int(confusion[(confusion['true_class'] == classes[i]) & (confusion['predicted_class'] == classes[j])]['count'].iloc[0])
            ax_cm.text(j, i, f'{value:.2f}\n({count})', ha='center', va='center', fontsize=5.5, color='white' if value > 0.52 else COLORS['ink'])
    ax_cm.set_xticks(range(len(classes)), classes, rotation=45, ha='right', rotation_mode='anchor')
    ax_cm.set_yticks(range(len(classes)), classes)
    ax_cm.set_xlabel('Predicted class')
    ax_cm.set_ylabel('True class')
    ax_cm.set_title('B6 internal-test error structure', loc='left', pad=5, fontweight='bold')
    ax_cm.tick_params(length=0)
    for spine in ax_cm.spines.values():
        spine.set_visible(False)
    fig.colorbar(cm_image, ax=ax_cm, fraction=0.04, pad=0.02, label='Row fraction')
    panel_label(ax_cm, 'b', x=-0.17)
    save_figure(fig, directory, 'FigureS3_perturbation_and_confusion')

def build_figure_s4() -> None:
    directory = SUPP / 'FigureS4_exploratory_all6'
    directory.mkdir(parents=True, exist_ok=True)
    curves = read_csv(DATA / 'FigureS4_curves_source.csv')
    systems = read_csv(DATA / 'FigureS4_external_systems_source.csv')
    efficiency = read_csv(DATA / 'common_environment_efficiency.csv')
    representative_runs = {'B2': 'oncoderm_mambax_b2_metadata_cmgf_seed42_20260706_234933', 'B6': 'oncoderm_mambax_b6_full_seed42_20260707_004712'}
    efficiency = efficiency.set_index('run_id')
    representative = {model: efficiency.loc[run_id] for model, run_id in representative_runs.items()}
    member_rows = []
    for model in ('B2', 'B6'):
        row = representative[model]
        for score_seed in (42, 2024, 3407):
            member_rows.append({'member': f'{model} score seed {score_seed}', 'architecture': model, 'score_seed': score_seed, 'representative_efficiency_run': representative_runs[model], 'stored_parameters': float(row['params']), 'flops': float(row['flops']), 'latency_ms_b1': float(row['latency_ms_b1']), 'derivation_note': 'Representative seed-42 architecture benchmark applied to each score member'})
    member_burden = pd.DataFrame(member_rows)
    all6_params = float(member_burden['stored_parameters'].sum())
    all6_flops = float(member_burden['flops'].sum())
    all6_latency = float(member_burden['latency_ms_b1'].sum())
    burden = pd.DataFrame([{'system': 'B2 single (seed 42)', 'stored_parameters': float(representative['B2']['params']), 'gflops': float(representative['B2']['flops']) / 1000000000.0, 'latency_ms': float(representative['B2']['latency_ms_b1']), 'evaluations': 1}, {'system': 'B6 single (seed 42)', 'stored_parameters': float(representative['B6']['params']), 'gflops': float(representative['B6']['flops']) / 1000000000.0, 'latency_ms': float(representative['B6']['latency_ms_b1']), 'evaluations': 1}, {'system': 'all6 (six members)', 'stored_parameters': all6_params, 'gflops': all6_flops / 1000000000.0, 'latency_ms': all6_latency, 'evaluations': 6}])
    if not np.isclose(all6_params, 196325256, atol=0.5):
        raise AssertionError(f'Unexpected all6 parameter sum: {all6_params}')
    if not np.isclose(all6_flops / 1000000000.0, 29.446781184, atol=1e-09):
        raise AssertionError(f'Unexpected all6 FLOP sum: {all6_flops / 1000000000.0}')
    if not np.isclose(all6_latency, 29.741485, atol=2e-05):
        raise AssertionError(f'Unexpected all6 latency sum: {all6_latency}')
    system_ids = ['b2_seed_ensemble', 'b6_seed_ensemble', 'b2_b6_all6']
    labels = {'b2_seed_ensemble': 'B2, 3-score', 'b6_seed_ensemble': 'B6, 3-score', 'b2_b6_all6': 'all6'}
    colors = {'b2_seed_ensemble': COLORS['B2'], 'b6_seed_ensemble': COLORS['B6'], 'b2_b6_all6': COLORS['all6']}
    system_row_map = {'b2_seed_ensemble': 'B2, 3-seed ensemble', 'b6_seed_ensemble': 'B6, 3-seed ensemble', 'b2_b6_all6': 'B2+B6, 6-model ensemble'}
    system_indexed = systems.set_index('System')
    fig = plt.figure(figsize=(FULL_WIDTH_MM * MM, 116 * MM), layout='constrained')
    outer = fig.add_gridspec(2, 2, height_ratios=[1.28, 0.72], hspace=0.22, wspace=0.2)
    ax_roc = fig.add_subplot(outer[0, 0])
    ax_pr = fig.add_subplot(outer[0, 1])
    ax_cal = fig.add_subplot(outer[1, 0])
    ax_burden = fig.add_subplot(outer[1, 1])
    for sid in system_ids:
        for curve_type, ax in (('ROC', ax_roc), ('PR', ax_pr)):
            subset = curves[(curves['system_id'] == sid) & (curves['curve'] == curve_type)]
            metric_name = 'roc_auc' if curve_type == 'ROC' else 'pr_auc'
            metric = float(system_indexed.loc[system_row_map[sid], metric_name])
            ax.plot(subset['x'], subset['y'], color=colors[sid], linewidth=1.7 if sid != 'b2_b6_all6' else 1.4, label=f'{labels[sid]}  {metric:.3f}')
    ax_roc.plot([0, 1], [0, 1], color=COLORS['light_gray'], linestyle='--', linewidth=0.8)
    ax_roc.set(xlabel='False-positive rate', ylabel='True-positive rate', xlim=(0, 1), ylim=(0, 1.01))
    ax_roc.set_title('Exploratory external ROC', loc='left', pad=5, fontweight='bold')
    ax_roc.legend(loc='lower right', frameon=False)
    clean_axis(ax_roc, grid=True)
    panel_label(ax_roc, 'a')
    prevalence = 584 / 33126
    ax_pr.axhline(prevalence, color=COLORS['mid_gray'], linestyle='--', linewidth=0.8)
    ax_pr.set(xlabel='Recall', ylabel='Precision', xlim=(0, 1), ylim=(0, 0.43))
    ax_pr.set_title('Exploratory external precision-recall', loc='left', pad=5, fontweight='bold')
    ax_pr.legend(loc='upper right', frameon=False)
    clean_axis(ax_pr, grid=True)
    panel_label(ax_pr, 'b')
    external_rows = pd.DataFrame([{'system_id': sid, **system_indexed.loc[system_row_map[sid], ['brier', 'ece', 'nll']].to_dict()} for sid in system_ids])
    calibration_values = external_rows.set_index('system_id').loc[system_ids, ['brier', 'ece', 'nll']].to_numpy(dtype=float)
    calibration_normalized = np.empty_like(calibration_values)
    for column in range(calibration_values.shape[1]):
        values = calibration_values[:, column]
        span = float(values.max() - values.min())
        calibration_normalized[:, column] = 0.5 if span == 0 else (values.max() - values) / span
    descriptor_cmap = LinearSegmentedColormap.from_list('descriptor', ['#F4F5F6', '#B9D9E8', COLORS['B2']])
    ax_cal.imshow(calibration_normalized, cmap=descriptor_cmap, vmin=0, vmax=1, aspect='auto')
    ax_cal.set_xticks(range(3), ['Brier', 'ECE', 'NLL'])
    ax_cal.set_yticks(range(3), ['B2, 3-score', 'B6, 3-score', 'all6'])
    for row in range(3):
        for column in range(3):
            shade = calibration_normalized[row, column]
            ax_cal.text(column, row, f'{calibration_values[row, column]:.3f}', ha='center', va='center', fontsize=6.5, color='white' if shade > 0.68 else COLORS['ink'], fontweight='bold' if shade > 0.92 else 'normal')
    ax_cal.set_title('External calibration descriptors', loc='left', pad=5, fontweight='bold')
    ax_cal.text(1.0, -0.19, 'Exact values; shading is column-normalized (lower is better)', transform=ax_cal.transAxes, ha='right', va='top', fontsize=5.7, color=COLORS['gray'])
    ax_cal.tick_params(length=0)
    for spine in ax_cal.spines.values():
        spine.set_visible(False)
    panel_label(ax_cal, 'c', x=-0.18)
    burden_values = np.column_stack([burden['stored_parameters'].to_numpy(dtype=float) / 1000000.0, burden['gflops'].to_numpy(dtype=float), burden['latency_ms'].to_numpy(dtype=float), burden['evaluations'].to_numpy(dtype=float)])
    burden_normalized = burden_values / burden_values.max(axis=0, keepdims=True)
    burden_cmap = LinearSegmentedColormap.from_list('burden', ['#F4F5F6', '#D9D3DF', COLORS['all6']])
    ax_burden.imshow(burden_normalized, cmap=burden_cmap, vmin=0, vmax=1, aspect='auto')
    ax_burden.set_xticks(range(4), ['Params (M)', 'GFLOPs', 'Latency (ms)', 'Evaluations'])
    ax_burden.set_yticks(range(3), ['B2 single', 'B6 single', 'all6'])
    burden_formats = ['{:.1f}', '{:.1f}', '{:.1f}', '{:.0f}']
    for row in range(3):
        for column in range(4):
            shade = burden_normalized[row, column]
            ax_burden.text(column, row, burden_formats[column].format(burden_values[row, column]), ha='center', va='center', fontsize=6.4, color='white' if shade > 0.72 else COLORS['ink'], fontweight='bold' if row == 2 else 'normal')
    ax_burden.set_title('Inference burden', loc='left', pad=5, fontweight='bold')
    ax_burden.text(1.0, -0.19, 'all6 latency is an arithmetic sequential estimate; not end-to-end', transform=ax_burden.transAxes, ha='right', va='top', fontsize=5.7, color=COLORS['gray'])
    ax_burden.tick_params(length=0)
    for spine in ax_burden.spines.values():
        spine.set_visible(False)
    panel_label(ax_burden, 'd', x=-0.18)
    save_figure(fig, directory, 'FigureS4_exploratory_score_integration')

def build_figure_s5() -> None:
    directory = SUPP / 'FigureS5_repeatability_efficiency'
    directory.mkdir(parents=True, exist_ok=True)
    runs = read_csv(DATA / 'FigureS5_three_seed_runs_source.csv')
    efficiency = read_csv(DATA / 'FigureS5_efficiency_source.csv')
    runs = runs[runs['model_id'].isin(['b2', 'b4', 'b6'])].copy()
    fig = plt.figure(figsize=(FULL_WIDTH_MM * MM, 82 * MM), layout='constrained')
    outer = fig.add_gridspec(1, 2, width_ratios=[1.72, 1.0], wspace=0.18)
    repeat_grid = outer[0, 0].subgridspec(2, 3, height_ratios=[0.18, 1.0], hspace=0.05, wspace=0.16)
    ax_repeat_header = fig.add_subplot(repeat_grid[0, :])
    repeat_axes = [fig.add_subplot(repeat_grid[1, i]) for i in range(3)]
    ax_eff = fig.add_subplot(outer[0, 1])
    model_order = ['b6', 'b2', 'b4']
    model_names = {'b6': 'B6 full', 'b2': 'B2 CMGF', 'b4': 'B4 LPAH'}
    model_colors = {'b6': COLORS['B6'], 'b2': COLORS['B2'], 'b4': COLORS['B4']}
    seed_markers = {42: 'o', 2024: 's', 3407: '^'}
    metric_specs = [('External Mel-AUC', 'External ROC-AUC  ↑', False), ('External PR-AUC', 'External PR-AUC  ↑', False), ('External Brier', 'External Brier  ↓', True)]
    for axis_index, (ax, (column, title, lower_better)) in enumerate(zip(repeat_axes, metric_specs)):
        for ypos, model in enumerate(model_order[::-1]):
            subset = runs[runs['model_id'] == model].sort_values('run_seed')
            values = subset[column].to_numpy(dtype=float)
            mean = values.mean()
            sd = values.std(ddof=1)
            ax.hlines(ypos, mean - sd, mean + sd, color=model_colors[model], linewidth=2.0, zorder=2)
            ax.plot(mean, ypos, marker='D', color=model_colors[model], markersize=4.8, zorder=3)
            offsets = [-0.12, 0, 0.12]
            for offset, row in zip(offsets, subset.itertuples(index=False)):
                seed = int(row.run_seed)
                value = float(getattr(row, column.replace('-', '_').replace(' ', '_'))) if False else float(subset.loc[subset['run_seed'] == seed, column].iloc[0])
                ax.plot(value, ypos + offset, marker=seed_markers[seed], markerfacecolor='white', markeredgecolor=model_colors[model], markeredgewidth=1.0, markersize=4.2, linestyle='none', zorder=4)
        ax.set_yticks(range(3), [model_names[m] for m in model_order[::-1]] if axis_index == 0 else [''] * 3)
        ax.set_title(title, loc='center', pad=4, fontweight='bold')
        ax.set_xlabel('')
        clean_axis(ax, grid=True)
    ax_repeat_header.set_axis_off()
    ax_repeat_header.text(0.0, 0.73, 'a', transform=ax_repeat_header.transAxes, fontsize=10.2, fontweight='bold', va='center')
    ax_repeat_header.text(0.055, 0.73, 'Fixed-split repeatability', transform=ax_repeat_header.transAxes, fontsize=8.2, fontweight='bold', va='center')
    legend_handles = [plt.Line2D([], [], marker=seed_markers[s], markerfacecolor='white', markeredgecolor=COLORS['gray'], linestyle='none', markersize=4.5) for s in seed_markers]
    legend_handles.append(plt.Line2D([], [], marker='D', color=COLORS['gray'], linestyle='none', markersize=4.8))
    ax_repeat_header.legend(legend_handles, ['42', '2024', '3407', 'mean ± s.d.'], loc='center right', bbox_to_anchor=(1.0, 0.73), frameon=False, ncol=4, handletextpad=0.25, columnspacing=0.72, borderaxespad=0)
    label_offsets = {'B0': (-15, 5), 'B1': (-4, -13), 'B2': (5, 5), 'B3': (-2, -13), 'B4': (-14, 6), 'B5': (5, 5), 'B6': (5, 4)}
    for _, row in efficiency.iterrows():
        model = str(row['model_id'])
        color = COLORS.get(model, COLORS['mid_gray']) if model in ('B2', 'B4', 'B6') else COLORS['mid_gray']
        size = 28 + 260 * float(row['pr_auc'])
        latency = float(row['Latency B=1'])
        external_auc = float(row['External Mel-AUC'])
        ax_eff.scatter(latency, external_auc, s=size, color=color, edgecolor='white', linewidth=0.7, zorder=3)
        dx, dy = label_offsets[model]
        ax_eff.annotate(model, (latency, external_auc), xytext=(dx, dy), textcoords='offset points', fontsize=6.3, fontweight='bold' if model in ('B2', 'B4', 'B6') else 'normal')
    ax_eff.set(xlabel='Latency per image, batch=1 (ms)', ylabel='External ROC-AUC', xlim=(3.82, 5.68), ylim=(0.665, 0.817))
    ax_eff.set_title('Efficiency-performance context', loc='left', pad=7, fontweight='bold')
    ax_eff.text(0.02, 0.025, 'Area = external PR-AUC', transform=ax_eff.transAxes, fontsize=5.8, color=COLORS['gray'])
    clean_axis(ax_eff, grid=True)
    panel_label(ax_eff, 'b', x=-0.18, y=1.1)
    save_figure(fig, directory, 'FigureS5_repeatability_and_efficiency')

def main():
    global MAIN, SUPP
    builders = {"3": build_figure3, "S2": build_figure_s2, "S3": build_figure_s3,
                "S4": build_figure_s4, "S5": build_figure_s5}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", choices=list(builders), nargs="+", default=list(builders))
    parser.add_argument("--output-dir", type=Path, default=MAIN)
    args = parser.parse_args()
    if args.output_dir.resolve() == DATA.resolve() or DATA.resolve() in args.output_dir.resolve().parents:
        raise ValueError("Output must not be inside source_data.")
    MAIN = SUPP = args.output_dir
    configure_plotting()
    for figure in args.figure:
        builders[figure]()
        print(f"Rendered Figure {figure} to {args.output_dir}")

if __name__ == "__main__":
    main()

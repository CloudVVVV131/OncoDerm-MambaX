from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import tifffile
import fitz
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from PIL import Image
from pypdf import PdfReader
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


ROOT = Path(__import__('os').environ['R52_PATH_0F70D48282E5'])
V8 = ROOT / "论文投稿图表_30篇标杆重构终版_20260804_v8"
V8_SOURCE = V8 / "source_data"
FULL = ROOT / "oncoderm_bioengineering_full_results_20260817_110459" / "project"
STABILITY = ROOT / "oncoderm_stability_v11_full_results_20260709_012310"
PATIENT_CSV = ROOT / "_figure_source_freeze_20260819" / "ISIC_2020_Training_GroundTruth_v2.csv"
OUT = ROOT / "论文投稿图表_Bioengineering数据图重构_20260819_v8派生版"

MAIN = OUT / "main_figures"
SUPP = OUT / "supplementary_figures"
SOURCE = OUT / "source_data"
SUBMISSION = OUT / "submission_ready"
AUDIT = OUT / "audit"
SCRIPTS = OUT / "scripts"

MM = 1 / 25.4
FULL_WIDTH_MM = 183

COLORS = {
    "ink": "#202124",
    "gray": "#687178",
    "mid_gray": "#9AA0A4",
    "light_gray": "#DDE2E5",
    "grid": "#E8ECEF",
    "B0": "#7F8588",
    "B1": "#9AA0A4",
    "B2": "#0072B2",
    "B3": "#8A8F92",
    "B4": "#E69F00",
    "B5": "#B1B4B6",
    "B6": "#D55E00",
    "DermaMamba": "#4D5960",
    "Swin-Tiny": "#8A8F92",
    "all6": "#7A7088",
    "teal": "#009E73",
    "rose": "#B35C6E",
    "blue_fill": "#EAF3F8",
    "orange_fill": "#FFF2E5",
}

MODEL_LABELS = {
    "B0": "B0 backbone",
    "B1": "ConvNeXt-Tiny (B1)",
    "B2": "B2 CMGF",
    "B3": "B3 CSSA",
    "B4": "B4 LPAH",
    "B5": "B5 CSSA+LPAH",
    "B6": "B6 full",
    "DermaMamba": "DermaMamba port",
    "Swin-Tiny": "Swin-Tiny",
}

PREDICTIONS = {
    "B1": FULL / "outputs" / "predictions" / "oncoderm_mambax_b1_melanoma_head_seed42_20260706_233528_external_predictions.csv",
    "B2": FULL / "outputs" / "predictions" / "oncoderm_mambax_b2_metadata_cmgf_seed42_20260706_234933_external_predictions.csv",
    "B6": FULL / "outputs" / "predictions" / "oncoderm_mambax_b6_full_seed42_20260707_004712_external_predictions.csv",
    "DermaMamba": FULL / "outputs" / "predictions" / "dermamamba_architecture_faithful_seed42_20260817_022043_external_predictions.csv",
    "Swin-Tiny": FULL / "outputs" / "predictions" / "swin_tiny_seed42_20260630_183731_external_predictions.csv",
}


@dataclass(frozen=True)
class FigureSpec:
    figure_id: str
    title: str
    destination: Path
    builder: Callable[[], None]


def configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 7.2,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.3,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
            "legend.fontsize": 6.4,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def prepare_dirs(clean: bool = True) -> None:
    for directory in (MAIN, SUPP, SOURCE, SUBMISSION, AUDIT, SCRIPTS):
        directory.mkdir(parents=True, exist_ok=True)
    if clean:
        for directory in (MAIN, SUPP, SOURCE, SUBMISSION, AUDIT):
            for child in directory.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing source assets:\n" + "\n".join(missing))


def read_csv(path: Path) -> pd.DataFrame:
    require([path])
    return pd.read_csv(path)


def clean_axis(ax: plt.Axes, grid: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", pad=2)
    if grid:
        ax.grid(color=COLORS["grid"], linewidth=0.55, zorder=0)


def panel_label(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.08) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.2,
        fontweight="bold",
        color=COLORS["ink"],
        clip_on=False,
    )


def save_figure(fig: plt.Figure, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    outputs = {
        "pdf": directory / f"{stem}.pdf",
        "svg": directory / f"{stem}.svg",
        "png": directory / f"{stem}.png",
        "tiff": directory / f"{stem}.tiff",
    }
    # Preserve the declared physical canvas; tight bounding boxes can silently
    # expand a nominal 183-mm figure beyond the journal's double-column width.
    fig.savefig(outputs["pdf"])
    fig.savefig(outputs["svg"])
    fig.savefig(outputs["png"], dpi=600)
    plt.close(fig)
    with Image.open(outputs["png"]) as image:
        raster = np.asarray(image.convert("RGB"))
    tifffile.imwrite(
        outputs["tiff"],
        raster,
        photometric="rgb",
        compression="deflate",
        resolution=(600, 600),
        resolutionunit="INCH",
    )
    shutil.copy2(outputs["pdf"], SUBMISSION / outputs["pdf"].name)


def write_figure_readme(directory: Path, text: str) -> None:
    (directory / "README.md").write_text(text.strip() + "\n", encoding="utf-8")


def stem_series(series: pd.Series) -> pd.Series:
    return series.astype(str).map(lambda value: Path(value).stem)


def wilson_interval(successes: float, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return center - radius, center + radius


def risk_deciles(y: np.ndarray, p: np.ndarray, model: str) -> pd.DataFrame:
    frame = pd.DataFrame({"label": y.astype(int), "score": p.astype(float)})
    frame["bin"] = pd.qcut(frame["score"], q=10, duplicates="drop")
    rows = []
    for index, (_, group) in enumerate(frame.groupby("bin", observed=True), start=1):
        observed = float(group["label"].mean())
        low, high = wilson_interval(float(group["label"].sum()), len(group))
        rows.append(
            {
                "model": model,
                "risk_decile": index,
                "n": len(group),
                "n_positive": int(group["label"].sum()),
                "mean_predicted": float(group["score"].mean()),
                "observed_fraction": observed,
                "wilson_low": low,
                "wilson_high": high,
            }
        )
    return pd.DataFrame(rows)


def calibration_glm(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    eps = 1e-12
    clipped = np.clip(p.astype(float), eps, 1 - eps)
    logit = np.log(clipped / (1 - clipped))
    fit = sm.GLM(y.astype(float), sm.add_constant(logit), family=sm.families.Binomial()).fit()
    return float(fit.params[0]), float(fit.params[1])


def grouped_weighted_metrics(
    y: np.ndarray,
    scores: np.ndarray,
    patient_codes: np.ndarray,
    patient_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    batch = patient_counts.shape[0]
    aucs = np.empty(batch, dtype=float)
    aps = np.empty(batch, dtype=float)

    asc = np.argsort(scores, kind="mergesort")
    scores_asc = scores[asc]
    y_asc = y[asc].astype(float)
    codes_asc = patient_codes[asc]
    starts_asc = np.r_[0, np.flatnonzero(np.diff(scores_asc)) + 1]
    weights_asc = patient_counts[:, codes_asc]
    pos = weights_asc * y_asc
    neg = weights_asc * (1 - y_asc)
    pos_group = np.add.reduceat(pos, starts_asc, axis=1)
    neg_group = np.add.reduceat(neg, starts_asc, axis=1)
    cumulative_neg = np.cumsum(neg_group, axis=1) - neg_group
    positives = pos_group.sum(axis=1)
    negatives = neg_group.sum(axis=1)
    numerator = np.sum(pos_group * (cumulative_neg + 0.5 * neg_group), axis=1)
    aucs[:] = numerator / (positives * negatives)

    desc = asc[::-1]
    scores_desc = scores[desc]
    y_desc = y[desc].astype(float)
    codes_desc = patient_codes[desc]
    starts_desc = np.r_[0, np.flatnonzero(np.diff(scores_desc)) + 1]
    weights_desc = patient_counts[:, codes_desc]
    pos_desc = weights_desc * y_desc
    neg_desc = weights_desc * (1 - y_desc)
    pos_group = np.add.reduceat(pos_desc, starts_desc, axis=1)
    neg_group = np.add.reduceat(neg_desc, starts_desc, axis=1)
    cumulative_pos = np.cumsum(pos_group, axis=1)
    cumulative_all = np.cumsum(pos_group + neg_group, axis=1)
    precision = np.divide(cumulative_pos, cumulative_all, out=np.zeros_like(cumulative_pos), where=cumulative_all > 0)
    aps[:] = np.sum((pos_group / positives[:, None]) * precision, axis=1)
    return aucs, aps


def patient_cluster_bootstrap(
    merged: pd.DataFrame,
    iterations: int = 10_000,
    seed: int = 42,
    batch_size: int = 40,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    patients = pd.Index(sorted(merged["patient_id"].astype(str).unique()))
    patient_map = {patient: index for index, patient in enumerate(patients)}
    codes = merged["patient_id"].astype(str).map(patient_map).to_numpy(dtype=int)
    y = merged["label"].to_numpy(dtype=int)
    b6 = merged["score_B6"].to_numpy(dtype=float)
    derma = merged["score_DermaMamba"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    rows: list[pd.DataFrame] = []
    probabilities = np.full(len(patients), 1 / len(patients), dtype=float)
    completed = 0
    while completed < iterations:
        current = min(batch_size, iterations - completed)
        counts = rng.multinomial(len(patients), probabilities, size=current).astype(np.float64)
        auc_b6, ap_b6 = grouped_weighted_metrics(y, b6, codes, counts)
        auc_derma, ap_derma = grouped_weighted_metrics(y, derma, codes, counts)
        rows.append(
            pd.DataFrame(
                {
                    "replicate": np.arange(completed + 1, completed + current + 1),
                    "roc_auc_B6": auc_b6,
                    "roc_auc_DermaMamba": auc_derma,
                    "roc_auc_delta": auc_b6 - auc_derma,
                    "pr_auc_B6": ap_b6,
                    "pr_auc_DermaMamba": ap_derma,
                    "pr_auc_delta": ap_b6 - ap_derma,
                }
            )
        )
        completed += current
    replicates = pd.concat(rows, ignore_index=True)
    point_roc = roc_auc_score(y, b6) - roc_auc_score(y, derma)
    point_pr = average_precision_score(y, b6) - average_precision_score(y, derma)
    summary_rows = []
    for metric, point in (("ROC-AUC", point_roc), ("PR-AUC", point_pr)):
        values = replicates[f"{metric.split('-')[0].lower()}_auc_delta"].to_numpy()
        summary_rows.append(
            {
                "metric": metric,
                "B6_minus_DermaMamba": point,
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
                "bootstrap_unit": "patient",
                "n_patients": len(patients),
                "n_images": len(merged),
                "iterations": iterations,
                "rng": "numpy.default_rng",
                "rng_seed": seed,
            }
        )
    return replicates, pd.DataFrame(summary_rows)


def build_figure2() -> None:
    directory = MAIN / "Figure2_external_evidence"
    directory.mkdir(parents=True, exist_ok=True)
    require([PATIENT_CSV, *PREDICTIONS.values()])
    metadata = read_csv(PATIENT_CSV)
    image_column = "image_name" if "image_name" in metadata.columns else "image_id"
    metadata = metadata[[image_column, "patient_id"]].copy()
    metadata["image_id"] = stem_series(metadata[image_column])
    metadata["patient_id"] = metadata["patient_id"].astype(str)

    curves = []
    row_sources = []
    frames: dict[str, pd.DataFrame] = {}
    expected = {
        "B1": (0.7095113, 0.0844899),
        "Swin-Tiny": (0.7207524, 0.0801000),
        "DermaMamba": (0.7472559, 0.1008085),
        "B2": (0.7985441, 0.1533380),
        "B6": (0.7810361, 0.1249194),
    }
    for model, path in PREDICTIONS.items():
        frame = read_csv(path)
        frame["image_id"] = stem_series(frame["image_id"])
        frame = frame.merge(metadata[["image_id", "patient_id"]], on="image_id", how="left", validate="one_to_one")
        if frame["patient_id"].isna().any():
            raise ValueError(f"Patient linkage failed for {model}")
        if len(frame) != 33126 or int(frame["label"].sum()) != 584:
            raise ValueError(f"Unexpected external cohort for {model}: {len(frame)} rows")
        y = frame["label"].to_numpy(dtype=int)
        p = frame["prob_mel"].to_numpy(dtype=float)
        auc = roc_auc_score(y, p)
        ap = average_precision_score(y, p)
        if not np.allclose((auc, ap), expected[model], atol=5e-5):
            raise AssertionError(f"Uniform-score metric mismatch for {model}: {(auc, ap)}")
        fpr, tpr, _ = roc_curve(y, p)
        recall, precision, _ = precision_recall_curve(y, p)
        curves.append(pd.DataFrame({"model": model, "curve": "ROC", "x": fpr, "y": tpr, "point_metric": auc}))
        curves.append(pd.DataFrame({"model": model, "curve": "PR", "x": recall, "y": precision, "point_metric": ap}))
        row_sources.append(
            frame[["image_id", "patient_id", "label"]]
            .assign(model=model, score_source="prob_mel", score=p)
            [["model", "score_source", "image_id", "patient_id", "label", "score"]]
        )
        frames[model] = frame

    curves_df = pd.concat(curves, ignore_index=True)
    curves_df.to_csv(directory / "Figure2_curves_source.csv", index=False)

    calibration_parts = []
    calibration_summary = []
    calibration_expected = {"B2": (-0.845, 0.132), "B6": (-0.995, 0.158)}
    for model in ("B2", "B6"):
        frame = frames[model]
        y = frame["label"].to_numpy(dtype=int)
        p = frame["prob_mel_endpoint"].to_numpy(dtype=float)
        intercept, slope = calibration_glm(y, p)
        expected_intercept, expected_slope = calibration_expected[model]
        if not np.allclose((intercept, slope), (expected_intercept, expected_slope), atol=8e-4):
            raise AssertionError(f"Calibration mismatch for {model}: {(intercept, slope)}")
        calibration_parts.append(risk_deciles(y, p, model))
        calibration_summary.append(
            {
                "model": model,
                "score_source": "prob_mel_endpoint",
                "calibration_intercept": intercept,
                "calibration_slope": slope,
                "n_images": len(frame),
                "n_positive": int(y.sum()),
            }
        )
        row_sources.append(
            frame[["image_id", "patient_id", "label"]]
            .assign(model=model, score_source="prob_mel_endpoint", score=p)
            [["model", "score_source", "image_id", "patient_id", "label", "score"]]
        )
    calibration_df = pd.concat(calibration_parts, ignore_index=True)
    calibration_summary_df = pd.DataFrame(calibration_summary)
    calibration_df.to_csv(directory / "Figure2_calibration_deciles_source.csv", index=False)
    calibration_summary_df.to_csv(directory / "Figure2_calibration_parameters_source.csv", index=False)
    pd.concat(row_sources, ignore_index=True).to_csv(directory / "Figure2_row_level_score_freeze.csv.gz", index=False, compression="gzip")

    pair = frames["B6"][["image_id", "patient_id", "label", "prob_mel"]].rename(columns={"prob_mel": "score_B6"})
    pair = pair.merge(
        frames["DermaMamba"][["image_id", "label", "prob_mel"]].rename(
            columns={"label": "label_derma", "prob_mel": "score_DermaMamba"}
        ),
        on="image_id",
        validate="one_to_one",
    )
    if not (pair["label"] == pair["label_derma"]).all():
        raise ValueError("B6 and DermaMamba labels do not align")
    pair = pair.drop(columns="label_derma")
    pair.to_csv(directory / "Figure2_patient_linked_pair_source.csv.gz", index=False, compression="gzip")
    replicates, bootstrap_summary = patient_cluster_bootstrap(pair)
    replicates.to_csv(directory / "Figure2_patient_bootstrap_replicates.csv.gz", index=False, compression="gzip")
    bootstrap_summary.to_csv(directory / "Figure2_patient_bootstrap_summary.csv", index=False)

    fig = plt.figure(figsize=(FULL_WIDTH_MM * MM, 126 * MM), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.02, 0.98], wspace=0.18, hspace=0.20)
    ax_roc = fig.add_subplot(grid[0, 0])
    ax_pr = fig.add_subplot(grid[0, 1])
    ax_forest = fig.add_subplot(grid[1, 0])
    ax_cal = fig.add_subplot(grid[1, 1])

    order = ["B6", "B2", "DermaMamba", "B1", "Swin-Tiny"]
    widths = {model: (2.0 if model == "B6" else 1.35 if model == "B2" else 1.0) for model in order}
    zorders = {model: (5 if model == "B6" else 4 if model == "B2" else 2) for model in order}
    for model in reversed(order):
        subset = curves_df[(curves_df["model"] == model) & (curves_df["curve"] == "ROC")]
        metric = subset["point_metric"].iloc[0]
        ax_roc.plot(
            subset["x"], subset["y"], color=COLORS[model], linewidth=widths[model], zorder=zorders[model],
            label=f"{MODEL_LABELS[model]}  {metric:.3f}",
        )
    ax_roc.plot([0, 1], [0, 1], color=COLORS["light_gray"], linewidth=0.8, linestyle="--", zorder=0)
    ax_roc.set(xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1.01))
    ax_roc.set_title("External ROC | uniform seven-class MEL score", loc="left", pad=5, fontweight="bold")
    handles, labels = ax_roc.get_legend_handles_labels()
    mapping = {label.split("  ")[0]: (handle, label) for handle, label in zip(handles, labels)}
    ordered = [mapping[MODEL_LABELS[m]] for m in order]
    ax_roc.legend([x[0] for x in ordered], [x[1] for x in ordered], loc="lower right", frameon=False, handlelength=2.2)
    clean_axis(ax_roc, grid=True)
    panel_label(ax_roc, "a")

    prevalence = frames["B6"]["label"].mean()
    for model in reversed(order):
        subset = curves_df[(curves_df["model"] == model) & (curves_df["curve"] == "PR")]
        metric = subset["point_metric"].iloc[0]
        ax_pr.plot(
            subset["x"], subset["y"], color=COLORS[model], linewidth=widths[model], zorder=zorders[model],
            label=f"{MODEL_LABELS[model]}  {metric:.3f}",
        )
    ax_pr.axhline(prevalence, color=COLORS["mid_gray"], linewidth=0.8, linestyle="--", label=f"Prevalence  {prevalence:.3f}")
    ax_pr.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 0.42))
    ax_pr.set_title("External precision-recall | prevalence 1.76%", loc="left", pad=5, fontweight="bold")
    handles, labels = ax_pr.get_legend_handles_labels()
    named = {label.split("  ")[0]: (handle, label) for handle, label in zip(handles, labels)}
    ordered = [named[MODEL_LABELS[m]] for m in order] + [named["Prevalence"]]
    ax_pr.legend([x[0] for x in ordered], [x[1] for x in ordered], loc="upper right", frameon=False, handlelength=2.2)
    clean_axis(ax_pr, grid=True)
    panel_label(ax_pr, "b")

    y_positions = np.array([1.0, 0.0])
    for ypos, row in zip(y_positions, bootstrap_summary.itertuples(index=False)):
        ax_forest.hlines(ypos, row.ci_low, row.ci_high, color=COLORS["B6"], linewidth=2.0, zorder=2)
        ax_forest.plot(row.B6_minus_DermaMamba, ypos, marker="o", markersize=6.2, color=COLORS["B6"], zorder=3)
        ax_forest.text(
            row.ci_high + 0.0015,
            ypos,
            f"{row.B6_minus_DermaMamba:+.3f} [{row.ci_low:+.3f}, {row.ci_high:+.3f}]",
            ha="left", va="center", fontsize=6.6,
        )
    ax_forest.axvline(0, color=COLORS["ink"], linewidth=0.8)
    ax_forest.set_yticks(y_positions, bootstrap_summary["metric"])
    ax_forest.set_xlim(-0.005, max(0.075, bootstrap_summary["ci_high"].max() + 0.018))
    ax_forest.set_ylim(-0.65, 1.65)
    ax_forest.set_xlabel("B6 minus DermaMamba")
    ax_forest.set_title("Patient-clustered paired bootstrap", loc="left", pad=5, fontweight="bold")
    ax_forest.text(
        0.0, -0.36, "10,000 replicates | 2,056 patients | deterministic seed 42",
        transform=ax_forest.transAxes, color=COLORS["gray"], fontsize=6.2, va="top",
    )
    clean_axis(ax_forest, grid=True)
    panel_label(ax_forest, "c")

    max_axis = 0.18
    ax_cal.plot([0, max_axis], [0, max_axis], color=COLORS["mid_gray"], linestyle="--", linewidth=0.9, zorder=0)
    for model, marker in (("B2", "s"), ("B6", "o")):
        subset = calibration_df[calibration_df["model"] == model]
        yerr = np.vstack(
            [subset["observed_fraction"] - subset["wilson_low"], subset["wilson_high"] - subset["observed_fraction"]]
        )
        ax_cal.errorbar(
            subset["mean_predicted"], subset["observed_fraction"], yerr=yerr,
            color=COLORS[model], marker=marker, markersize=4.1, linewidth=1.15, capsize=1.5,
            label=MODEL_LABELS[model], zorder=3,
        )
    lines = []
    for row in calibration_summary_df.itertuples(index=False):
        lines.append(f"{row.model}: intercept {row.calibration_intercept:.3f}; slope {row.calibration_slope:.3f}")
    ax_cal.text(0.03, 0.96, "\n".join(lines), transform=ax_cal.transAxes, va="top", ha="left", fontsize=6.1, color=COLORS["gray"])
    ax_cal.set(xlabel="Mean predicted risk", ylabel="Observed melanoma fraction", xlim=(0, max_axis), ylim=(0, max_axis))
    ax_cal.set_title("External endpoint calibration", loc="left", pad=5, fontweight="bold")
    ax_cal.legend(loc="lower right", frameon=False)
    clean_axis(ax_cal, grid=True)
    panel_label(ax_cal, "d")

    save_figure(fig, directory, "Figure2_external_melanoma_evidence")
    write_figure_readme(
        directory,
        """
# Figure 2. External melanoma discrimination and patient-clustered comparison under cohort shift

- Panels a-b use the uniform seven-class softmax melanoma probability (`prob_mel`) for all five systems.
- Panel c is recomputed from linked image-level predictions with patient as the bootstrap unit (10,000 replicates; NumPy `default_rng`, seed 42). The historical bootstrap replicate file and RNG seed were not retained, so this package reports the reproducible reanalysis rather than selecting a seed to reproduce a rounded legacy interval.
- Panel d uses each model's dedicated melanoma endpoint (`prob_mel_endpoint`), explicitly separated from the uniform-score comparison. Error bars are Wilson 95% intervals for risk-decile observed fractions.
- Canonical cohort: SIIM-ISIC 2020, 33,126 images, 584 melanomas, 2,056 patients.
""",
    )


def build_figure3() -> None:
    directory = MAIN / "Figure3_threshold_transport"
    directory.mkdir(parents=True, exist_ok=True)
    source_path = V8_SOURCE / "Figure6_threshold_source.csv"
    data = read_csv(source_path)
    data.to_csv(directory / "Figure3_threshold_transport_source.csv", index=False)
    policy_order = ["youden", "specificity_90", "specificity_95", "sensitivity_80"]
    policy_labels = {"youden": "Youden", "specificity_90": "Spec 90%", "specificity_95": "Spec 95%", "sensitivity_80": "Sens 80%"}
    dataset_order = ["val", "test", "external"]
    lookup = data.set_index(["threshold_policy", "dataset"])
    rows = [
        ("Validation sensitivity", "val", "sensitivity"),
        ("Internal-test sensitivity", "test", "sensitivity"),
        ("External sensitivity", "external", "sensitivity"),
        ("External specificity", "external", "specificity"),
    ]
    matrix = np.array([[lookup.loc[(policy, dataset), metric] for policy in policy_order] for _, dataset, metric in rows], dtype=float)
    matrix_source = pd.DataFrame(matrix, index=[r[0] for r in rows], columns=[policy_labels[p] for p in policy_order])
    matrix_source.to_csv(directory / "Figure3_transport_matrix_source.csv")

    fig = plt.figure(figsize=(FULL_WIDTH_MM * MM, 78 * MM), layout="constrained")
    grid = fig.add_gridspec(1, 3, width_ratios=[1.17, 1.08, 1.02], wspace=0.22)
    ax_heat = fig.add_subplot(grid[0, 0])
    ax_dumb = fig.add_subplot(grid[0, 1])
    ax_map = fig.add_subplot(grid[0, 2])

    cmap = LinearSegmentedColormap.from_list("transport", ["#F5F7F8", "#B9D9E8", COLORS["B2"]])
    image = ax_heat.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    compact_policy_labels = {
        "youden": "Youden",
        "specificity_90": "Spec ≥ 0.90",
        "specificity_95": "Spec ≥ 0.95",
        "sensitivity_80": "Sens ≥ 0.80",
    }
    compact_row_labels = ["Validation sens.", "Internal-test sens.", "External sens.", "External spec."]
    heat_policy_labels = ["Youden", "Spec ≥\n0.90", "Spec ≥\n0.95", "Sens ≥\n0.80"]
    ax_heat.set_xticks(range(4), heat_policy_labels)
    ax_heat.set_yticks(range(4), compact_row_labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax_heat.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=6.5, color="white" if value > 0.66 else COLORS["ink"])
    ax_heat.set_title("Validation-fixed operating policies", loc="left", pad=5, fontweight="bold")
    for spine in ax_heat.spines.values():
        spine.set_visible(False)
    ax_heat.tick_params(length=0)
    panel_label(ax_heat, "a", x=-0.18)

    y_positions = np.arange(len(policy_order))[::-1]
    for ypos, policy in zip(y_positions, policy_order):
        val = float(lookup.loc[(policy, "val"), "sensitivity"])
        external = float(lookup.loc[(policy, "external"), "sensitivity"])
        ax_dumb.plot([external, val], [ypos, ypos], color=COLORS["light_gray"], linewidth=2.0, zorder=1)
        ax_dumb.plot(val, ypos, marker="o", markersize=5.2, markerfacecolor="white", markeredgecolor=COLORS["B2"], markeredgewidth=1.2, zorder=3)
        ax_dumb.plot(external, ypos, marker="o", markersize=5.2, color=COLORS["B6"], zorder=3)
        ax_dumb.text(1.225, ypos, f"{external - val:+.3f}", va="center", ha="right", fontsize=6.0, color=COLORS["gray"])
    ax_dumb.set_yticks(y_positions, [compact_policy_labels[p] for p in policy_order])
    ax_dumb.set_xlim(0.05, 1.24)
    ax_dumb.set_ylim(-0.35, 3.52)
    ax_dumb.set_xlabel("Sensitivity")
    ax_dumb.set_title("Sensitivity transport", loc="left", pad=5, fontweight="bold")
    first_val = float(lookup.loc[("youden", "val"), "sensitivity"])
    first_external = float(lookup.loc[("youden", "external"), "sensitivity"])
    ax_dumb.text(first_external, 3.22, "ext.", ha="center", va="bottom", fontsize=5.4, color=COLORS["B6"])
    ax_dumb.text(first_val, 3.22, "val.", ha="center", va="bottom", fontsize=5.4, color=COLORS["B2"])
    ax_dumb.text(1.225, 3.22, "Δ", ha="right", va="bottom", fontsize=5.4, color=COLORS["gray"])
    clean_axis(ax_dumb, grid=True)
    panel_label(ax_dumb, "b", x=-0.16)

    external_rows = data[data["dataset"] == "external"].set_index("threshold_policy").loc[policy_order]
    point_colors = [COLORS["gray"], COLORS["B2"], COLORS["B6"], COLORS["B4"]]
    point_numbers = {policy: index for index, policy in enumerate(policy_order, start=1)}
    for (policy, row), color in zip(external_rows.iterrows(), point_colors):
        size = 35 + 230 * float(row["ppv"])
        ax_map.scatter(row["sensitivity"], row["specificity"], s=size, color=color, edgecolor="white", linewidth=0.7, zorder=3)
        ax_map.text(
            row["sensitivity"], row["specificity"], str(point_numbers[policy]),
            ha="center", va="center", fontsize=5.2, color="white", fontweight="bold", zorder=4,
        )
    ax_map.set(xlabel="External sensitivity", ylabel="External specificity", xlim=(0.095, 0.19), ylim=(0.9870, 0.9982))
    ax_map.set_title("External operating map", loc="left", pad=5, fontweight="bold")
    ax_map.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.3f"))
    clean_axis(ax_map, grid=True)
    panel_label(ax_map, "c", x=-0.18)
    ax_map.text(
        0.5,
        -0.19,
        "1 Youden · 2 Sp≥.90 · 3 Sp≥.95 · 4 Se≥.80 | area=PPV",
        transform=ax_map.transAxes,
        ha="center",
        va="top",
        fontsize=5.2,
        color=COLORS["gray"],
    )

    save_figure(fig, directory, "Figure3_threshold_transport")
    write_figure_readme(
        directory,
        """
# Figure 3. Validation-derived decision thresholds did not transport across cohorts

All values are read directly from the canonical threshold-analysis output. The figure uses the exploratory all6 equal-weight score integration only because that is the score stream on which the four validation-derived policies were defined. No calibration panel was added because a consistent retained row-level stream was not required for this three-panel threshold story.
""",
    )


def build_figure_s2() -> None:
    directory = SUPP / "FigureS2_configuration_specialization"
    directory.mkdir(parents=True, exist_ok=True)
    data = read_csv(V8_SOURCE / "exact_Table2_seed42_ablation.csv")
    data.to_csv(directory / "FigureS2_configuration_source.csv", index=False)
    models = [f"B{i}" for i in range(7)]
    data = data.set_index("Model").loc[models].reset_index()
    module_columns = ["CSSA", "LPAH", "CMGF"]
    metrics = ["Macro-F1", "Internal Mel-AUC", "External Mel-AUC", "External PR-AUC", "External Brier", "External ECE"]
    short = ["Macro-F1", "Internal\nMEL AUC", "External\nROC AUC", "External\nPR AUC", "Brier", "ECE"]
    raw = data[metrics].to_numpy(dtype=float)
    oriented = raw.copy()
    oriented[:, -2:] *= -1
    standardized = (oriented - oriented.mean(axis=0)) / oriented.std(axis=0, ddof=0)
    z_source = pd.DataFrame(standardized, columns=[f"oriented_z_{name}" for name in metrics])
    pd.concat([data[["Model", "Setting"] + module_columns + metrics], z_source], axis=1).to_csv(
        directory / "FigureS2_direction_aligned_zscores.csv", index=False
    )

    fig = plt.figure(figsize=(FULL_WIDTH_MM * MM, 83 * MM), layout="constrained")
    grid = fig.add_gridspec(1, 2, width_ratios=[0.72, 2.65], wspace=0.06)
    ax_modules = fig.add_subplot(grid[0, 0])
    ax_metrics = fig.add_subplot(grid[0, 1])
    module_matrix = data[module_columns].astype(int).to_numpy()
    module_cmap = LinearSegmentedColormap.from_list("modules", ["#F4F5F6", "#4C7E93"])
    ax_modules.imshow(module_matrix, cmap=module_cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(len(models)):
        for j in range(len(module_columns)):
            ax_modules.text(j, i, "●" if module_matrix[i, j] else "–", ha="center", va="center", fontsize=7.5, color="white" if module_matrix[i, j] else COLORS["mid_gray"])
    ax_modules.set_xticks(range(3), module_columns, rotation=35, ha="right", rotation_mode="anchor")
    ax_modules.set_yticks(range(7), models)
    ax_modules.set_title("Configuration", loc="left", pad=5, fontweight="bold")
    ax_modules.tick_params(length=0)
    for spine in ax_modules.spines.values():
        spine.set_visible(False)
    panel_label(ax_modules, "a", x=-0.30)

    cmap = LinearSegmentedColormap.from_list("direction", ["#B96B63", "#F7F7F6", "#367D9A"])
    norm = TwoSlopeNorm(vmin=-2.0, vcenter=0.0, vmax=2.0)
    image = ax_metrics.imshow(standardized, cmap=cmap, norm=norm, aspect="auto")
    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            ax_metrics.text(j, i, f"{raw[i, j]:.3f}", ha="center", va="center", fontsize=6.2, color=COLORS["ink"])
    ax_metrics.set_xticks(range(len(metrics)), short)
    ax_metrics.set_yticks(range(7), [""] * 7)
    ax_metrics.set_title("Direction-aligned within-endpoint standardization", loc="left", pad=5, fontweight="bold")
    ax_metrics.tick_params(length=0)
    for spine in ax_metrics.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, ax=ax_metrics, fraction=0.025, pad=0.015)
    cbar.set_label("Higher = more favorable", rotation=270, rotation_mode="anchor", labelpad=10)
    ax_metrics.text(0.0, -0.12, "Brier and ECE are sign-reversed before standardization; cell labels show raw values.", transform=ax_metrics.transAxes, fontsize=6.3, color=COLORS["gray"], va="top")

    save_figure(fig, directory, "FigureS2_configuration_specialization")
    write_figure_readme(
        directory,
        """
# Supplementary Figure S2. Configuration-level specialization across internal, external, and reliability endpoints

This is a configuration study, not a factorial or causal ablation. Colors are within-column z scores aligned so that higher always means more favorable; Brier and ECE are sign-reversed before standardization. Raw endpoint values remain printed in every cell.
""",
    )


def build_figure_s3() -> None:
    directory = SUPP / "FigureS3_perturbation_and_confusion"
    directory.mkdir(parents=True, exist_ok=True)
    robustness = read_csv(V8_SOURCE / "FigureS3_robustness_source.csv")
    confusion = read_csv(V8_SOURCE / "FigureS6_confusion_matrix_source.csv")
    robustness.to_csv(directory / "FigureS3_perturbation_source.csv", index=False)
    confusion.to_csv(directory / "FigureS3_confusion_source.csv", index=False)
    robustness = robustness[(robustness["condition"] != "clean") & (robustness["model_id"].isin(["B0", "B3", "B6"]))].copy()
    condition_order = list(dict.fromkeys(robustness["condition_label"].tolist()))
    model_order = ["B0", "B3", "B6"]
    macro = robustness.pivot(index="condition_label", columns="model_id", values="macro_f1_drop").reindex(index=condition_order, columns=model_order)
    mel = robustness.pivot(index="condition_label", columns="model_id", values="mel_auc_drop").reindex(index=condition_order, columns=model_order)

    classes = list(dict.fromkeys(confusion["true_class"].tolist()))
    cm = confusion.pivot(index="true_class", columns="predicted_class", values="row_fraction").reindex(index=classes, columns=classes).fillna(0)

    fig = plt.figure(figsize=(FULL_WIDTH_MM * MM, 92 * MM), layout="constrained")
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.28], wspace=0.11)
    ax_macro = fig.add_subplot(grid[0, 0])
    ax_mel = fig.add_subplot(grid[0, 1])
    ax_cm = fig.add_subplot(grid[0, 2])
    drop_cmap = LinearSegmentedColormap.from_list("drop", ["#DDEFE8", "#FAFAF8", "#B95F55"])
    max_abs = max(abs(macro.to_numpy()).max(), abs(mel.to_numpy()).max(), 0.001)
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)
    for ax, matrix, title in ((ax_macro, macro, "Macro-F1 drop"), (ax_mel, mel, "MEL AUC drop")):
        image = ax.imshow(matrix.to_numpy(), cmap=drop_cmap, norm=norm, aspect="auto")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix.iloc[i, j]:+.3f}", ha="center", va="center", fontsize=6.1)
        ax.set_xticks(range(3), model_order)
        ax.set_title(title, loc="left", pad=5, fontweight="bold")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    ax_macro.set_yticks(range(len(condition_order)), [label.replace("Brightness Shift", "Brightness").replace("Gaussian Blur", "Blur") for label in condition_order])
    ax_mel.set_yticks(range(len(condition_order)), [""] * len(condition_order))
    panel_label(ax_macro, "a", x=-0.28)
    cbar = fig.colorbar(image, ax=[ax_macro, ax_mel], fraction=0.025, pad=0.015)
    cbar.set_label("Clean minus perturbed", rotation=270, rotation_mode="anchor", labelpad=10)

    cm_image = ax_cm.imshow(cm.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="equal")
    for i in range(len(classes)):
        for j in range(len(classes)):
            value = cm.iloc[i, j]
            count = int(confusion[(confusion["true_class"] == classes[i]) & (confusion["predicted_class"] == classes[j])]["count"].iloc[0])
            ax_cm.text(j, i, f"{value:.2f}\n({count})", ha="center", va="center", fontsize=5.5, color="white" if value > 0.52 else COLORS["ink"])
    ax_cm.set_xticks(range(len(classes)), classes, rotation=45, ha="right", rotation_mode="anchor")
    ax_cm.set_yticks(range(len(classes)), classes)
    ax_cm.set_xlabel("Predicted class")
    ax_cm.set_ylabel("True class")
    ax_cm.set_title("B6 internal-test error structure", loc="left", pad=5, fontweight="bold")
    ax_cm.tick_params(length=0)
    for spine in ax_cm.spines.values():
        spine.set_visible(False)
    fig.colorbar(cm_image, ax=ax_cm, fraction=0.04, pad=0.02, label="Row fraction")
    panel_label(ax_cm, "b", x=-0.17)

    save_figure(fig, directory, "FigureS3_perturbation_and_confusion")
    write_figure_readme(
        directory,
        """
# Supplementary Figure S3. Controlled perturbation sensitivity and B6 class-level error structure

The perturbation panels are sensitivity characterization, not evidence of universal robustness superiority. Values are clean minus perturbed performance; negative values indicate an observed improvement under that perturbation. The confusion matrix is row-normalized and also reports raw counts.
""",
    )


def build_figure_s4() -> None:
    directory = SUPP / "FigureS4_exploratory_all6"
    directory.mkdir(parents=True, exist_ok=True)
    curves = read_csv(V8_SOURCE / "Figure4_external_curves_source.csv")
    systems = read_csv(V8_SOURCE / "exact_Table4_external_systems.csv")
    efficiency = read_csv(FULL / "results" / "efficiency_table.csv")
    curves.to_csv(directory / "FigureS4_curves_source.csv", index=False)
    systems.to_csv(directory / "FigureS4_external_systems_source.csv", index=False)

    representative_runs = {
        "B2": "oncoderm_mambax_b2_metadata_cmgf_seed42_20260706_234933",
        "B6": "oncoderm_mambax_b6_full_seed42_20260707_004712",
    }
    efficiency = efficiency.set_index("run_id")
    representative = {model: efficiency.loc[run_id] for model, run_id in representative_runs.items()}
    member_rows = []
    for model in ("B2", "B6"):
        row = representative[model]
        for score_seed in (42, 2024, 3407):
            member_rows.append(
                {
                    "member": f"{model} score seed {score_seed}",
                    "architecture": model,
                    "score_seed": score_seed,
                    "representative_efficiency_run": representative_runs[model],
                    "stored_parameters": float(row["params"]),
                    "flops": float(row["flops"]),
                    "latency_ms_b1": float(row["latency_ms_b1"]),
                    "derivation_note": "Representative seed-42 architecture benchmark applied to each score member",
                }
            )
    member_burden = pd.DataFrame(member_rows)
    member_burden.to_csv(directory / "FigureS4_member_burden_inputs.csv", index=False)
    all6_params = float(member_burden["stored_parameters"].sum())
    all6_flops = float(member_burden["flops"].sum())
    all6_latency = float(member_burden["latency_ms_b1"].sum())
    burden = pd.DataFrame(
        [
            {
                "system": "B2 single (seed 42)",
                "stored_parameters": float(representative["B2"]["params"]),
                "gflops": float(representative["B2"]["flops"]) / 1e9,
                "latency_ms": float(representative["B2"]["latency_ms_b1"]),
                "evaluations": 1,
            },
            {
                "system": "B6 single (seed 42)",
                "stored_parameters": float(representative["B6"]["params"]),
                "gflops": float(representative["B6"]["flops"]) / 1e9,
                "latency_ms": float(representative["B6"]["latency_ms_b1"]),
                "evaluations": 1,
            },
            {
                "system": "all6 (six members)",
                "stored_parameters": all6_params,
                "gflops": all6_flops / 1e9,
                "latency_ms": all6_latency,
                "evaluations": 6,
            },
        ]
    )
    if not np.isclose(all6_params, 196325256, atol=0.5):
        raise AssertionError(f"Unexpected all6 parameter sum: {all6_params}")
    if not np.isclose(all6_flops / 1e9, 29.446781184, atol=1e-9):
        raise AssertionError(f"Unexpected all6 FLOP sum: {all6_flops / 1e9}")
    if not np.isclose(all6_latency, 29.741485, atol=2e-5):
        raise AssertionError(f"Unexpected all6 latency sum: {all6_latency}")
    burden.to_csv(directory / "FigureS4_inference_burden_source.csv", index=False)

    system_ids = ["b2_seed_ensemble", "b6_seed_ensemble", "b2_b6_all6"]
    labels = {"b2_seed_ensemble": "B2, 3-score", "b6_seed_ensemble": "B6, 3-score", "b2_b6_all6": "all6"}
    colors = {"b2_seed_ensemble": COLORS["B2"], "b6_seed_ensemble": COLORS["B6"], "b2_b6_all6": COLORS["all6"]}
    system_row_map = {
        "b2_seed_ensemble": "B2, 3-seed ensemble",
        "b6_seed_ensemble": "B6, 3-seed ensemble",
        "b2_b6_all6": "B2+B6, 6-model ensemble",
    }
    system_indexed = systems.set_index("System")

    fig = plt.figure(figsize=(FULL_WIDTH_MM * MM, 116 * MM), layout="constrained")
    outer = fig.add_gridspec(2, 2, height_ratios=[1.28, 0.72], hspace=0.22, wspace=0.20)
    ax_roc = fig.add_subplot(outer[0, 0])
    ax_pr = fig.add_subplot(outer[0, 1])
    ax_cal = fig.add_subplot(outer[1, 0])
    ax_burden = fig.add_subplot(outer[1, 1])

    for sid in system_ids:
        for curve_type, ax in (("ROC", ax_roc), ("PR", ax_pr)):
            subset = curves[(curves["system_id"] == sid) & (curves["curve"] == curve_type)]
            metric_name = "roc_auc" if curve_type == "ROC" else "pr_auc"
            metric = float(system_indexed.loc[system_row_map[sid], metric_name])
            ax.plot(subset["x"], subset["y"], color=colors[sid], linewidth=1.7 if sid != "b2_b6_all6" else 1.4, label=f"{labels[sid]}  {metric:.3f}")
    ax_roc.plot([0, 1], [0, 1], color=COLORS["light_gray"], linestyle="--", linewidth=0.8)
    ax_roc.set(xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1.01))
    ax_roc.set_title("Exploratory external ROC", loc="left", pad=5, fontweight="bold")
    ax_roc.legend(loc="lower right", frameon=False)
    clean_axis(ax_roc, grid=True)
    panel_label(ax_roc, "a")
    prevalence = 584 / 33126
    ax_pr.axhline(prevalence, color=COLORS["mid_gray"], linestyle="--", linewidth=0.8)
    ax_pr.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 0.43))
    ax_pr.set_title("Exploratory external precision-recall", loc="left", pad=5, fontweight="bold")
    ax_pr.legend(loc="upper right", frameon=False)
    clean_axis(ax_pr, grid=True)
    panel_label(ax_pr, "b")

    external_rows = pd.DataFrame(
        [
            {"system_id": sid, **system_indexed.loc[system_row_map[sid], ["brier", "ece", "nll"]].to_dict()}
            for sid in system_ids
        ]
    )
    external_rows.to_csv(directory / "FigureS4_calibration_descriptors_source.csv", index=False)
    calibration_values = external_rows.set_index("system_id").loc[system_ids, ["brier", "ece", "nll"]].to_numpy(dtype=float)
    calibration_normalized = np.empty_like(calibration_values)
    for column in range(calibration_values.shape[1]):
        values = calibration_values[:, column]
        span = float(values.max() - values.min())
        calibration_normalized[:, column] = 0.5 if span == 0 else (values.max() - values) / span
    descriptor_cmap = LinearSegmentedColormap.from_list("descriptor", ["#F4F5F6", "#B9D9E8", COLORS["B2"]])
    ax_cal.imshow(calibration_normalized, cmap=descriptor_cmap, vmin=0, vmax=1, aspect="auto")
    ax_cal.set_xticks(range(3), ["Brier", "ECE", "NLL"])
    ax_cal.set_yticks(range(3), ["B2, 3-score", "B6, 3-score", "all6"])
    for row in range(3):
        for column in range(3):
            shade = calibration_normalized[row, column]
            ax_cal.text(
                column, row, f"{calibration_values[row, column]:.3f}",
                ha="center", va="center", fontsize=6.5,
                color="white" if shade > 0.68 else COLORS["ink"], fontweight="bold" if shade > 0.92 else "normal",
            )
    ax_cal.set_title("External calibration descriptors", loc="left", pad=5, fontweight="bold")
    ax_cal.text(1.0, -0.19, "Exact values; shading is column-normalized (lower is better)", transform=ax_cal.transAxes, ha="right", va="top", fontsize=5.7, color=COLORS["gray"])
    ax_cal.tick_params(length=0)
    for spine in ax_cal.spines.values():
        spine.set_visible(False)
    panel_label(ax_cal, "c", x=-0.18)

    burden_values = np.column_stack(
        [
            burden["stored_parameters"].to_numpy(dtype=float) / 1e6,
            burden["gflops"].to_numpy(dtype=float),
            burden["latency_ms"].to_numpy(dtype=float),
            burden["evaluations"].to_numpy(dtype=float),
        ]
    )
    burden_normalized = burden_values / burden_values.max(axis=0, keepdims=True)
    burden_cmap = LinearSegmentedColormap.from_list("burden", ["#F4F5F6", "#D9D3DF", COLORS["all6"]])
    ax_burden.imshow(burden_normalized, cmap=burden_cmap, vmin=0, vmax=1, aspect="auto")
    ax_burden.set_xticks(range(4), ["Params (M)", "GFLOPs", "Latency (ms)", "Evaluations"])
    ax_burden.set_yticks(range(3), ["B2 single", "B6 single", "all6"])
    burden_formats = ["{:.1f}", "{:.1f}", "{:.1f}", "{:.0f}"]
    for row in range(3):
        for column in range(4):
            shade = burden_normalized[row, column]
            ax_burden.text(
                column, row, burden_formats[column].format(burden_values[row, column]),
                ha="center", va="center", fontsize=6.4,
                color="white" if shade > 0.72 else COLORS["ink"], fontweight="bold" if row == 2 else "normal",
            )
    ax_burden.set_title("Inference burden", loc="left", pad=5, fontweight="bold")
    ax_burden.text(1.0, -0.19, "all6 latency is an arithmetic sequential estimate; not end-to-end", transform=ax_burden.transAxes, ha="right", va="top", fontsize=5.7, color=COLORS["gray"])
    ax_burden.tick_params(length=0)
    for spine in ax_burden.spines.values():
        spine.set_visible(False)
    panel_label(ax_burden, "d", x=-0.18)

    save_figure(fig, directory, "FigureS4_exploratory_score_integration")
    write_figure_readme(
        directory,
        """
# Supplementary Figure S4. Exploratory score integration and associated inference burden

The all6 system is a post-hoc equal-weight score integration with no weights fitted to the external cohort. It requires six model evaluations. The burden estimate follows Supplementary Table S12: the representative seed-42 B2 and B6 architecture measurements are each multiplied by three score members. Latency is therefore an arithmetic sequential estimate, not a measured end-to-end ensemble latency. `FigureS4_member_burden_inputs.csv` records every member and the representative benchmark run used in the derivation.
""",
    )


def build_figure_s5() -> None:
    directory = SUPP / "FigureS5_repeatability_efficiency"
    directory.mkdir(parents=True, exist_ok=True)
    runs = read_csv(STABILITY / "results" / "submission_v1_stability_runs_table.csv")
    efficiency = read_csv(V8_SOURCE / "FigureS2_efficiency_source.csv")
    runs.to_csv(directory / "FigureS5_three_seed_runs_source.csv", index=False)
    efficiency.to_csv(directory / "FigureS5_efficiency_source.csv", index=False)
    runs = runs[runs["model_id"].isin(["b2", "b4", "b6"])].copy()

    fig = plt.figure(figsize=(FULL_WIDTH_MM * MM, 82 * MM), layout="constrained")
    outer = fig.add_gridspec(1, 2, width_ratios=[1.72, 1.0], wspace=0.18)
    repeat_grid = outer[0, 0].subgridspec(2, 3, height_ratios=[0.18, 1.0], hspace=0.05, wspace=0.16)
    ax_repeat_header = fig.add_subplot(repeat_grid[0, :])
    repeat_axes = [fig.add_subplot(repeat_grid[1, i]) for i in range(3)]
    ax_eff = fig.add_subplot(outer[0, 1])
    model_order = ["b6", "b2", "b4"]
    model_names = {"b6": "B6 full", "b2": "B2 CMGF", "b4": "B4 LPAH"}
    model_colors = {"b6": COLORS["B6"], "b2": COLORS["B2"], "b4": COLORS["B4"]}
    seed_markers = {42: "o", 2024: "s", 3407: "^"}
    metric_specs = [
        ("External Mel-AUC", "External ROC-AUC  ↑", False),
        ("External PR-AUC", "External PR-AUC  ↑", False),
        ("External Brier", "External Brier  ↓", True),
    ]
    for axis_index, (ax, (column, title, lower_better)) in enumerate(zip(repeat_axes, metric_specs)):
        for ypos, model in enumerate(model_order[::-1]):
            subset = runs[runs["model_id"] == model].sort_values("run_seed")
            values = subset[column].to_numpy(dtype=float)
            mean = values.mean()
            sd = values.std(ddof=1)
            ax.hlines(ypos, mean - sd, mean + sd, color=model_colors[model], linewidth=2.0, zorder=2)
            ax.plot(mean, ypos, marker="D", color=model_colors[model], markersize=4.8, zorder=3)
            offsets = [-0.12, 0, 0.12]
            for offset, row in zip(offsets, subset.itertuples(index=False)):
                seed = int(row.run_seed)
                value = float(getattr(row, column.replace("-", "_").replace(" ", "_"))) if False else float(subset.loc[subset["run_seed"] == seed, column].iloc[0])
                ax.plot(value, ypos + offset, marker=seed_markers[seed], markerfacecolor="white", markeredgecolor=model_colors[model], markeredgewidth=1.0, markersize=4.2, linestyle="none", zorder=4)
        ax.set_yticks(range(3), [model_names[m] for m in model_order[::-1]] if axis_index == 0 else [""] * 3)
        ax.set_title(title, loc="center", pad=4, fontweight="bold")
        ax.set_xlabel("")
        clean_axis(ax, grid=True)
    ax_repeat_header.set_axis_off()
    ax_repeat_header.text(0.0, 0.73, "a", transform=ax_repeat_header.transAxes, fontsize=10.2, fontweight="bold", va="center")
    ax_repeat_header.text(0.055, 0.73, "Fixed-split repeatability", transform=ax_repeat_header.transAxes, fontsize=8.2, fontweight="bold", va="center")
    legend_handles = [
        plt.Line2D([], [], marker=seed_markers[s], markerfacecolor="white", markeredgecolor=COLORS["gray"], linestyle="none", markersize=4.5)
        for s in seed_markers
    ]
    legend_handles.append(plt.Line2D([], [], marker="D", color=COLORS["gray"], linestyle="none", markersize=4.8))
    ax_repeat_header.legend(
        legend_handles, ["42", "2024", "3407", "mean ± s.d."],
        loc="center right", bbox_to_anchor=(1.0, 0.73), frameon=False, ncol=4,
        handletextpad=0.25, columnspacing=0.72, borderaxespad=0,
    )

    label_offsets = {
        "B0": (-15, 5), "B1": (-4, -13), "B2": (5, 5), "B3": (-2, -13),
        "B4": (-14, 6), "B5": (5, 5), "B6": (5, 4),
    }
    for _, row in efficiency.iterrows():
        model = str(row["model_id"])
        color = COLORS.get(model, COLORS["mid_gray"]) if model in ("B2", "B4", "B6") else COLORS["mid_gray"]
        size = 28 + 260 * float(row["pr_auc"])
        latency = float(row["Latency B=1"])
        external_auc = float(row["External Mel-AUC"])
        ax_eff.scatter(latency, external_auc, s=size, color=color, edgecolor="white", linewidth=0.7, zorder=3)
        dx, dy = label_offsets[model]
        ax_eff.annotate(model, (latency, external_auc), xytext=(dx, dy), textcoords="offset points", fontsize=6.3, fontweight="bold" if model in ("B2", "B4", "B6") else "normal")
    ax_eff.set(xlabel="Latency per image, batch=1 (ms)", ylabel="External ROC-AUC", xlim=(3.82, 5.68), ylim=(0.665, 0.817))
    ax_eff.set_title("Efficiency-performance context", loc="left", pad=7, fontweight="bold")
    ax_eff.text(0.02, 0.025, "Area = external PR-AUC", transform=ax_eff.transAxes, fontsize=5.8, color=COLORS["gray"])
    clean_axis(ax_eff, grid=True)
    panel_label(ax_eff, "b", x=-0.18, y=1.10)

    save_figure(fig, directory, "FigureS5_repeatability_and_efficiency")
    write_figure_readme(
        directory,
        """
# Supplementary Figure S5. Fixed-split training repeatability and computational context

The three runs vary optimization seed while sharing split seed 42. They therefore characterize fixed-split training repeatability and optimization sensitivity, not population-sampling robustness. Efficiency measurements are the retained seed-42 batch-1 measurements.
""",
    )


def endpoint_delta_panel(
    ax: plt.Axes,
    baseline: pd.Series,
    comparison: pd.Series,
    baseline_label: str,
    comparison_label: str,
    title: str,
) -> None:
    columns = ["Macro-F1", "Internal Mel-AUC", "External ROC-AUC", "External PR-AUC"]
    labels = ["Macro-F1", "Internal MEL AUC", "External ROC AUC", "External PR AUC"]
    base_values = np.array([float(baseline[c]) for c in columns])
    comp_values = np.array([float(comparison[c]) for c in columns])
    deltas = comp_values - base_values
    limit = max(abs(deltas).max() * 1.62, 0.040)
    y = np.arange(len(columns))[::-1]
    bar_colors = [COLORS["teal"] if value >= 0 else COLORS["rose"] for value in deltas]
    ax.barh(y, deltas, color=bar_colors, height=0.55)
    ax.axvline(0, color=COLORS["ink"], linewidth=0.8)
    for ypos, delta, base, comp in zip(y, deltas, base_values, comp_values):
        offset = 0.003 if delta >= 0 else -0.003
        ax.text(delta + offset, ypos, f"{delta:+.3f}", ha="left" if delta >= 0 else "right", va="center", fontsize=6.5, fontweight="bold")
        ax.text(-limit * 0.98, ypos - 0.34, f"{baseline_label} {base:.3f}  →  {comparison_label} {comp:.3f}", fontsize=5.7, color=COLORS["gray"], ha="left", va="center")
    ax.set_yticks(y, labels)
    ax.set_xlim(-limit, limit)
    ax.set_xlabel(f"Δ ({comparison_label} minus {baseline_label})")
    ax.set_title(title, loc="left", pad=5, fontweight="bold")
    clean_axis(ax, grid=True)


def build_figure_s6() -> None:
    directory = SUPP / "FigureS6_implementation_partition"
    directory.mkdir(parents=True, exist_ok=True)
    b6m_path = FULL / "results" / "bioengineering_direct_baselines" / "optional" / "B6M_seed42_metrics.csv"
    split_path = FULL / "results" / "bioengineering_direct_baselines" / "optional" / "B6_crosssplit2024_metrics.csv"
    b6m = read_csv(b6m_path)
    split = read_csv(split_path)
    b6m.to_csv(directory / "FigureS6_B6M_source.csv", index=False)
    split.to_csv(directory / "FigureS6_crosssplit_source.csv", index=False)
    baseline = split[split["cohort"] == "fixed_split_seed42"].iloc[0]
    b6m_row = b6m.iloc[0]
    split2024 = split[split["cohort"] == "independent_split_seed2024"].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH_MM * MM, 72 * MM), layout="constrained")
    endpoint_delta_panel(axes[0], baseline, b6m_row, "B6 axial", "B6-M", "Implementation sensitivity")
    panel_label(axes[0], "a", x=-0.20)
    axes[0].text(0.0, -0.20, "B6-M uses the official mamba-ssm block; descriptive comparison only.", transform=axes[0].transAxes, fontsize=6.0, color=COLORS["gray"], va="top")
    endpoint_delta_panel(axes[1], baseline, split2024, "split 42", "split 2024", "Partition sensitivity")
    panel_label(axes[1], "b", x=-0.20)
    axes[1].text(0.0, -0.20, "Model seed fixed at 42; only the lesion-grouped partition changes.", transform=axes[1].transAxes, fontsize=6.0, color=COLORS["gray"], va="top")
    save_figure(fig, directory, "FigureS6_implementation_partition_sensitivity")
    write_figure_readme(
        directory,
        """
# Supplementary Figure S6. Implementation and partition sensitivity of the principal B6 framework

Both panels are descriptive sensitivity analyses. Panel a contrasts the executed axial row/column depthwise Conv1d CSSA backend with a separately trained B6-M implementation using the official `mamba-ssm` block. Panel b holds model seed 42 fixed and changes the lesion-grouped split seed from 42 to 2024. No architecture-selection or superiority claim is inferred.
""",
    )


def build_figure_s7() -> None:
    directory = SUPP / "FigureS7_localization"
    directory.mkdir(parents=True, exist_ok=True)
    table_path = FULL / "results" / "bioengineering_direct_baselines" / "mask200_table.csv"
    data = read_csv(table_path)
    data["model"] = data["run_id"].map(lambda value: "B4" if "_b4_" in value else "B6" if "_b6_" in value else "unknown")
    if set(data["model"]) != {"B4", "B6"}:
        raise ValueError("Could not identify B4/B6 localization rows")
    if not ((data["n"] == 200) & (data["generated_saliency_maps"] == 200) & (data["missing_image_count"] == 0) & (data["saliency_failure_count"] == 0)).all():
        raise ValueError("Localization completeness check failed")
    data.to_csv(directory / "FigureS7_localization_source.csv", index=False)
    data = data.set_index("model").loc[["B4", "B6"]]

    fig = plt.figure(figsize=(FULL_WIDTH_MM * MM, 67 * MM), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.12], hspace=0.12, wspace=0.24)
    axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    ax_footer = fig.add_subplot(grid[1, :])
    metrics = [
        ("saliency_mask_iou", "Saliency-mask IoU", (0, 0.60), ".3f"),
        ("pointing_game_hit_rate", "Pointing-game hit rate", (0, 1.05), ".3f"),
        ("lesion_background_saliency_ratio", "Lesion:background ratio", (0, 39), ".1f"),
    ]
    for index, (ax, (column, title, ylim, fmt)) in enumerate(zip(axes, metrics)):
        values = data[column].to_numpy(dtype=float)
        ax.bar([0, 1], values, color=[COLORS["B4"], COLORS["B6"]], width=0.62)
        for xpos, value in enumerate(values):
            ax.text(xpos, value + (ylim[1] * 0.025), format(value, fmt), ha="center", va="bottom", fontsize=7.0, fontweight="bold")
        ax.set_xticks([0, 1], ["B4", "B6"])
        ax.set_ylim(*ylim)
        ax.set_title(title, loc="center", pad=5, fontweight="bold")
        clean_axis(ax, grid=True)
        if index == 0:
            panel_label(ax, "a", x=-0.22)
        elif index == 1:
            panel_label(ax, "b", x=-0.22)
        else:
            panel_label(ax, "c", x=-0.22)
    ax_footer.set_axis_off()
    ax_footer.text(
        0.5, 0.58,
        "ISIC 2018 | 200 paired masks per model | 200/200 saliency maps generated | no missing images or failures",
        transform=ax_footer.transAxes, ha="center", va="center", fontsize=6.2, color=COLORS["gray"],
    )
    save_figure(fig, directory, "FigureS7_quantitative_localization")
    write_figure_readme(
        directory,
        """
# Supplementary Figure S7. Quantitative lesion-localization characterization

The retained archive supports quantitative reporting for 200 paired masks per model and confirms complete saliency generation. It does not retain the raw dermoscopy images, ground-truth mask files, per-case localization table, prediction-outcome linkage, or deterministic case-selection manifest needed for a defensible TP/TN/FP/FN qualitative montage. Accordingly, no qualitative examples were fabricated or selected from saliency maps alone.
""",
    )


def build_manifest(figure_specs: list[FigureSpec]) -> None:
    inputs = [
        PATIENT_CSV,
        *PREDICTIONS.values(),
        V8_SOURCE / "Figure6_threshold_source.csv",
        V8_SOURCE / "exact_Table2_seed42_ablation.csv",
        V8_SOURCE / "FigureS3_robustness_source.csv",
        V8_SOURCE / "FigureS6_confusion_matrix_source.csv",
        V8_SOURCE / "Figure4_external_curves_source.csv",
        V8_SOURCE / "exact_Table4_external_systems.csv",
        STABILITY / "results" / "submission_v1_stability_runs_table.csv",
        V8_SOURCE / "FigureS2_efficiency_source.csv",
        FULL / "results" / "efficiency_table.csv",
        FULL / "results" / "bioengineering_direct_baselines" / "optional" / "B6M_seed42_metrics.csv",
        FULL / "results" / "bioengineering_direct_baselines" / "optional" / "B6_crosssplit2024_metrics.csv",
        FULL / "results" / "bioengineering_direct_baselines" / "mask200_table.csv",
    ]
    require(inputs)
    rows = []
    for path in inputs:
        rows.append(
            {
                "source_file": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "role": "row-level prediction" if path in PREDICTIONS.values() else "canonical metadata" if path == PATIENT_CSV else "canonical derived source",
            }
        )
    pd.DataFrame(rows).to_csv(SOURCE / "figure_source_manifest.csv", index=False)

    inventory = [
        ("Figure2", "Main text", "Figure2_external_melanoma_evidence", "Direct external evidence, patient-clustered inference and calibration"),
        ("Figure3", "Main text", "Figure3_threshold_transport", "Transport of validation-fixed operating policies"),
        ("FigureS2", "Supplementary", "FigureS2_configuration_specialization", "Configuration specialization; descriptive, non-factorial"),
        ("FigureS3", "Supplementary", "FigureS3_perturbation_and_confusion", "Perturbation sensitivity and B6 error structure"),
        ("FigureS4", "Supplementary", "FigureS4_exploratory_score_integration", "Exploratory post-hoc all6 integration and burden"),
        ("FigureS5", "Supplementary", "FigureS5_repeatability_and_efficiency", "Fixed-split three-seed repeatability and efficiency"),
        ("FigureS6", "Supplementary", "FigureS6_implementation_partition_sensitivity", "Implementation and partition sensitivity"),
        ("FigureS7", "Supplementary", "FigureS7_quantitative_localization", "Quantitative localization only"),
    ]
    pd.DataFrame(inventory, columns=["figure_id", "placement", "file_stem", "scientific_role"]).to_csv(
        OUT / "FINAL_FIGURE_INVENTORY.csv", index=False
    )

    panel_sources = [
        ("Figure2", "a,b", "Uniform prob_mel ROC and PR curves", "; ".join(str(PREDICTIONS[key]) for key in PREDICTIONS)),
        ("Figure2", "c", "B6 minus DermaMamba patient-clustered bootstrap", f"{PREDICTIONS['B6']}; {PREDICTIONS['DermaMamba']}; {PATIENT_CSV}"),
        ("Figure2", "d", "B2 and B6 endpoint calibration", f"{PREDICTIONS['B2']}; {PREDICTIONS['B6']}"),
        ("Figure3", "a-c", "Validation-fixed threshold transport", str(V8_SOURCE / "Figure6_threshold_source.csv")),
        ("FigureS2", "a,b", "B0-B6 modules and endpoint matrix", str(V8_SOURCE / "exact_Table2_seed42_ablation.csv")),
        ("FigureS3", "a", "Controlled perturbation sensitivity", str(V8_SOURCE / "FigureS3_robustness_source.csv")),
        ("FigureS3", "b", "B6 internal-test confusion matrix", str(V8_SOURCE / "FigureS6_confusion_matrix_source.csv")),
        ("FigureS4", "a-c", "Exploratory B2/B6/all6 curves and calibration", f"{V8_SOURCE / 'Figure4_external_curves_source.csv'}; {V8_SOURCE / 'exact_Table4_external_systems.csv'}"),
        ("FigureS4", "d", "Stored inference burden", f"{FULL / 'results' / 'efficiency_table.csv'}; {V8_SOURCE / 'FigureS2_efficiency_source.csv'}"),
        ("FigureS5", "a", "Three-seed fixed-split repeatability", str(STABILITY / "results" / "submission_v1_stability_runs_table.csv")),
        ("FigureS5", "b", "Seed-42 efficiency context", str(V8_SOURCE / "FigureS2_efficiency_source.csv")),
        ("FigureS6", "a", "B6 axial versus B6-M", str(FULL / "results" / "bioengineering_direct_baselines" / "optional" / "B6M_seed42_metrics.csv")),
        ("FigureS6", "b", "Split 42 versus split 2024", str(FULL / "results" / "bioengineering_direct_baselines" / "optional" / "B6_crosssplit2024_metrics.csv")),
        ("FigureS7", "a-c", "Aggregate localization on 200 paired masks", str(FULL / "results" / "bioengineering_direct_baselines" / "mask200_table.csv")),
    ]
    pd.DataFrame(panel_sources, columns=["figure_id", "panel", "evidence", "original_source_assets"]).to_csv(
        SOURCE / "panel_source_map.csv", index=False
    )

    captions = """
# Final figure placement and English captions

## Main text

### Figure 2
**Place in Results immediately after the direct external-comparison paragraph and patient-clustered inferential result.**

**Figure 2. External melanoma discrimination and patient-clustered comparison under cohort shift.** (a,b) Receiver-operating-characteristic and precision-recall curves on SIIM-ISIC 2020 using the uniform seven-class softmax melanoma probability for all systems; the dashed precision-recall reference denotes melanoma prevalence. (c) Paired differences between B6 and the faithful DermaMamba port, with 95% percentile intervals from 10,000 patient-clustered bootstrap replicates. (d) External risk-decile calibration for the dedicated melanoma endpoints of B2 and B6; error bars denote Wilson 95% intervals, and inset text gives logistic calibration intercepts and slopes. The calibration panel uses a distinct, explicitly labelled endpoint score and is not part of the uniform-score ranking comparison.

### Figure 3
**Place in Results §4.5 immediately after the opening statement that validation-derived thresholds did not transport.**

**Figure 3. Transport of validation-derived decision thresholds across cohorts.** (a) Sensitivity on validation, internal test and external data, together with external specificity, for four validation-derived operating policies. (b) Change in sensitivity from validation to external evaluation. (c) External sensitivity-specificity operating points; marker area encodes positive predictive value. All policies were fixed on the validation set before application to the internal test and SIIM-ISIC 2020 cohorts.

## Supplementary material

### Supplementary Figure S2
**Supplementary Figure S2. Configuration-level specialization across internal, external and reliability endpoints.** Module presence and direction-aligned within-endpoint standardized performance for the B0-B6 configuration study. Brier score and expected calibration error were sign-reversed before standardization so that higher color intensity consistently denotes more favorable performance; printed values are the untransformed endpoint estimates. This configuration study is not a factorial or causal ablation.

### Supplementary Figure S3
**Supplementary Figure S3. Controlled perturbation sensitivity and B6 class-level error structure.** (a) Change from clean performance under prespecified blur, brightness and hair-like perturbations for B0, B3 and B6. Positive values denote performance loss. (b) Row-normalized B6 internal-test confusion matrix with raw counts in parentheses. The perturbation results provide sensitivity characterization rather than evidence of universal robustness superiority.

### Supplementary Figure S4
**Supplementary Figure S4. Exploratory score integration and associated inference burden.** (a,b) External ROC and precision-recall curves for the B2 and B6 three-seed score ensembles and their post-hoc equal-weight six-score integration (all6). (c) External Brier score, expected calibration error and negative log-likelihood. (d) Stored parameters, GFLOPs and arithmetic batch-1 latency for one B2 model, one B6 model and all6. No score weights were fitted to the external cohort. All6 requires six model evaluations; latency is an arithmetic sequential estimate, not measured end-to-end latency.

### Supplementary Figure S5
**Supplementary Figure S5. Fixed-split training repeatability and computational context.** (a) External ROC-AUC, precision-recall AUC and Brier score across model seeds 42, 2024 and 3407 for B2, B4 and B6 under the same lesion-grouped split; diamonds and bars denote mean ± s.d. (b) Seed-42 external ROC-AUC versus batch-1 latency across B0-B6, with marker area proportional to external precision-recall AUC. The repeated runs assess fixed-split optimization sensitivity, not population-sampling robustness.

### Supplementary Figure S6
**Supplementary Figure S6. Implementation and partition sensitivity of the principal B6 framework.** (a) Descriptive endpoint changes for an official mamba-ssm B6-M implementation relative to the executed axial directional-context B6. (b) Descriptive endpoint changes after replacing lesion-grouped split seed 42 with split seed 2024 while holding model seed 42 fixed. These analyses are sensitivity characterizations and were not used for post-hoc architecture selection.

### Supplementary Figure S7
**Supplementary Figure S7. Quantitative lesion-localization characterization.** Saliency-mask intersection-over-union, pointing-game hit rate and lesion-to-background saliency ratio for B4 and B6 across 200 paired ISIC 2018 masks per model. All 200 saliency maps were generated for each model, with no missing images or saliency failures. Qualitative cases are not shown because the retained archive does not contain the complete raw-image, mask, outcome-linkage and deterministic case-selection assets required for reproducible case assembly.
"""
    (OUT / "FIGURE_PLACEMENT_AND_CAPTIONS.md").write_text(captions.strip() + "\n", encoding="utf-8")

    gaps = """
# Source-asset gaps and manuscript synchronization items

## Figure 1 and Supplementary Figure S1
These are ChatGPT-led conceptual figures under the supplied division-of-labor guide. Their v8 originals and provenance were located, but this Codex data-figure package does not redesign them.

## Figure 2 historical patient bootstrap interval
The row-level B6 and DermaMamba predictions and official patient mapping are present, so the patient-clustered analysis is fully reproducible. The historical 10,000 bootstrap replicate file and historical RNG seed are absent. This package therefore uses a declared deterministic NumPy seed of 42 and stores every replicate. The manuscript must be synchronized to the newly generated interval if it differs slightly from the rounded legacy interval.

## Supplementary Figure S7 qualitative montage
The archive retains 400 grayscale saliency maps and aggregate localization metrics, but not the complete raw dermoscopy images, ground-truth masks, per-case IoU table, outcome linkage or deterministic TP/TN/FP/FN case-selection manifest. A reproducible qualitative montage cannot be generated from the retained assets. The quantitative 200-pair figure is complete and no cases were guessed or visually cherry-picked.
"""
    (OUT / "SOURCE_ASSET_GAPS.md").write_text(gaps.strip() + "\n", encoding="utf-8")

    manifest_md = "# Figure source manifest\n\n"
    manifest_md += "This package is derived from the frozen v8 figure package and retained experiment archives. No Word-embedded image was used as a numerical source. Every generated data figure reads machine-readable source files listed in `source_data/figure_source_manifest.csv`.\n\n"
    manifest_md += "## Generated figures\n\n"
    for spec in figure_specs:
        manifest_md += f"- **{spec.figure_id}:** {spec.title}\n"
    manifest_md += "\n## Reproduction\n\n"
    manifest_md += __import__('os').path.expandvars("```powershell\n& '${R52_PATH_07EFDCD4B820}' '${R52_PATH_DB971D526188}' --figure all\n```\n")
    (OUT / "MANIFEST.md").write_text(manifest_md, encoding="utf-8")


def build_launchers(figure_ids: list[str]) -> None:
    for figure_id in figure_ids:
        path = SCRIPTS / f"build_{figure_id.lower()}.py"
        code = f'''from pathlib import Path\nimport subprocess\nimport sys\n\nmaster = Path(__file__).with_name("build_bioengineering_data_figures.py")\nraise SystemExit(subprocess.call([sys.executable, str(master), "--figure", "{figure_id}"]))\n'''
        path.write_text(code, encoding="utf-8")
    refresh = SCRIPTS / "refresh_generated_manifest.py"
    refresh.write_text(
        '''from build_bioengineering_data_figures import build_generated_manifest\n\nreport = build_generated_manifest()\nprint(f"manifest_rows={len(report)}")\n''',
        encoding="utf-8",
    )


def expected_outputs(figure_ids: Iterable[str]) -> list[tuple[str, Path, str]]:
    records = [
        ("Figure2", MAIN / "Figure2_external_evidence", "Figure2_external_melanoma_evidence"),
        ("Figure3", MAIN / "Figure3_threshold_transport", "Figure3_threshold_transport"),
        ("FigureS2", SUPP / "FigureS2_configuration_specialization", "FigureS2_configuration_specialization"),
        ("FigureS3", SUPP / "FigureS3_perturbation_and_confusion", "FigureS3_perturbation_and_confusion"),
        ("FigureS4", SUPP / "FigureS4_exploratory_all6", "FigureS4_exploratory_score_integration"),
        ("FigureS5", SUPP / "FigureS5_repeatability_efficiency", "FigureS5_repeatability_and_efficiency"),
        ("FigureS6", SUPP / "FigureS6_implementation_partition", "FigureS6_implementation_partition_sensitivity"),
        ("FigureS7", SUPP / "FigureS7_localization", "FigureS7_quantitative_localization"),
    ]
    selected = set(figure_ids)
    return [record for record in records if record[0] in selected]


def _resolution_value(value: object) -> float:
    if isinstance(value, tuple) and len(value) == 2:
        return float(value[0]) / float(value[1])
    return float(value)


def validate_rasters(figure_ids: Iterable[str]) -> pd.DataFrame:
    rows = []
    for figure_id, directory, stem in expected_outputs(figure_ids):
        png_path = directory / f"{stem}.png"
        tiff_path = directory / f"{stem}.tiff"
        require([png_path, tiff_path])
        with Image.open(png_path) as image:
            png_array = np.array(image.convert("RGB"))
            width, height = image.size
            extrema = image.convert("RGB").getextrema()
        with tifffile.TiffFile(tiff_path) as tiff:
            page = tiff.pages[0]
            tiff_shape = page.shape
            x_resolution = _resolution_value(page.tags["XResolution"].value)
            y_resolution = _resolution_value(page.tags["YResolution"].value)
            resolution_unit = int(page.tags["ResolutionUnit"].value)
            tiff_array = page.asarray()
        passed = (
            width >= 3000
            and height >= 1000
            and any(low < high for low, high in extrema)
            and tuple(tiff_shape[:2]) == (height, width)
            and tiff_array.shape == png_array.shape
            and np.array_equal(tiff_array, png_array)
            and abs(x_resolution - 600.0) < 0.01
            and abs(y_resolution - 600.0) < 0.01
            and resolution_unit == 2
        )
        rows.append(
            {
                "figure_id": figure_id,
                "png_file": str(png_path.relative_to(OUT)),
                "tiff_file": str(tiff_path.relative_to(OUT)),
                "width_px": width,
                "height_px": height,
                "png_nonblank": any(low < high for low, high in extrema),
                "tiff_matches_png_pixels": np.array_equal(tiff_array, png_array),
                "tiff_x_dpi": x_resolution,
                "tiff_y_dpi": y_resolution,
                "tiff_resolution_unit_inch": resolution_unit == 2,
                "passed": passed,
            }
        )
    report = pd.DataFrame(rows)
    report.to_csv(AUDIT / "raster_qa.csv", index=False)
    if report.empty or not report["passed"].all():
        raise AssertionError("Raster QA failed; inspect audit/raster_qa.csv")
    return report


def validate_vectors(figure_ids: Iterable[str]) -> pd.DataFrame:
    rows = []
    for figure_id, directory, stem in expected_outputs(figure_ids):
        pdf_path = directory / f"{stem}.pdf"
        svg_path = directory / f"{stem}.svg"
        submission_path = SUBMISSION / f"{stem}.pdf"
        require([pdf_path, svg_path, submission_path])

        reader = PdfReader(str(pdf_path))
        page = reader.pages[0]
        width_mm = float(page.mediabox.width) * 25.4 / 72.0
        height_mm = float(page.mediabox.height) * 25.4 / 72.0
        extracted_text = "".join(pdf_page.extract_text() or "" for pdf_page in reader.pages)
        pdf_document = fitz.open(pdf_path)
        span_sizes = []
        for pdf_page in pdf_document:
            for block in pdf_page.get_text("dict").get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        if str(span.get("text", "")).strip():
                            span_sizes.append(float(span.get("size", 0.0)))
        pdf_document.close()
        min_font_pt = min(span_sizes) if span_sizes else 0.0

        root = ET.parse(svg_path).getroot()
        svg_text_nodes = sum(1 for element in root.iter() if element.tag.rsplit("}", 1)[-1] == "text")
        svg_image_nodes = sum(1 for element in root.iter() if element.tag.rsplit("}", 1)[-1] == "image")
        passed = (
            len(reader.pages) == 1
            and 80.0 <= width_mm <= 185.5
            and 30.0 <= height_mm <= 170.0
            and len(extracted_text.strip()) >= 20
            and min_font_pt >= 4.99
            and svg_text_nodes >= 5
            and sha256(pdf_path) == sha256(submission_path)
        )
        rows.append(
            {
                "figure_id": figure_id,
                "pdf_file": str(pdf_path.relative_to(OUT)),
                "svg_file": str(svg_path.relative_to(OUT)),
                "pages": len(reader.pages),
                "width_mm": width_mm,
                "height_mm": height_mm,
                "pdf_text_characters": len(extracted_text.strip()),
                "minimum_pdf_font_pt": min_font_pt,
                "svg_text_nodes": svg_text_nodes,
                "svg_image_nodes": svg_image_nodes,
                "submission_pdf_hash_matches": sha256(pdf_path) == sha256(submission_path),
                "passed": passed,
            }
        )
    report = pd.DataFrame(rows)
    report.to_csv(AUDIT / "vector_qa.csv", index=False)
    if report.empty or not report["passed"].all():
        raise AssertionError("Vector QA failed; inspect audit/vector_qa.csv")
    return report


def validate_numeric_contracts(figure_ids: Iterable[str]) -> pd.DataFrame:
    selected = set(figure_ids)
    checks: list[dict[str, object]] = []

    def record(figure_id: str, check: str, observed: object, expected: object, passed: bool) -> None:
        checks.append(
            {
                "figure_id": figure_id,
                "check": check,
                "observed": str(observed),
                "expected": str(expected),
                "passed": bool(passed),
            }
        )

    if "Figure2" in selected:
        directory = MAIN / "Figure2_external_evidence"
        curves = read_csv(directory / "Figure2_curves_source.csv")
        expected_metrics = {
            "B1": (0.7095113, 0.0844899),
            "Swin-Tiny": (0.7207524, 0.0801000),
            "DermaMamba": (0.7472559, 0.1008085),
            "B2": (0.7985441, 0.1533380),
            "B6": (0.7810361, 0.1249194),
        }
        for model, (expected_roc, expected_pr) in expected_metrics.items():
            model_data = curves[curves["model"] == model]
            observed_roc = float(model_data.loc[model_data["curve"] == "ROC", "point_metric"].iloc[0])
            observed_pr = float(model_data.loc[model_data["curve"] == "PR", "point_metric"].iloc[0])
            record("Figure2", f"{model} uniform-score ROC-AUC", observed_roc, expected_roc, abs(observed_roc - expected_roc) <= 5e-5)
            record("Figure2", f"{model} uniform-score PR-AUC", observed_pr, expected_pr, abs(observed_pr - expected_pr) <= 5e-5)
        bootstrap = read_csv(directory / "Figure2_patient_bootstrap_summary.csv")
        record("Figure2", "Patient bootstrap rows", len(bootstrap), 2, len(bootstrap) == 2)
        record("Figure2", "Patient bootstrap units", set(bootstrap["bootstrap_unit"]), {"patient"}, set(bootstrap["bootstrap_unit"]) == {"patient"})
        record("Figure2", "Patient bootstrap cohort", (set(bootstrap["n_patients"]), set(bootstrap["n_images"])), ({2056}, {33126}), set(bootstrap["n_patients"]) == {2056} and set(bootstrap["n_images"]) == {33126})
        record("Figure2", "Patient bootstrap iterations and seed", (set(bootstrap["iterations"]), set(bootstrap["rng_seed"])), ({10000}, {42}), set(bootstrap["iterations"]) == {10000} and set(bootstrap["rng_seed"]) == {42})
        record("Figure2", "Patient bootstrap intervals exclude zero", bool((bootstrap["ci_low"] > 0).all()), True, bool((bootstrap["ci_low"] > 0).all()))
        calibration = read_csv(directory / "Figure2_calibration_parameters_source.csv").set_index("model")
        for model, (expected_intercept, expected_slope) in {"B2": (-0.845, 0.132), "B6": (-0.995, 0.158)}.items():
            observed = (float(calibration.loc[model, "calibration_intercept"]), float(calibration.loc[model, "calibration_slope"]))
            record("Figure2", f"{model} endpoint calibration", observed, (expected_intercept, expected_slope), np.allclose(observed, (expected_intercept, expected_slope), atol=8e-4))

    if "Figure3" in selected:
        directory = MAIN / "Figure3_threshold_transport"
        data = read_csv(directory / "Figure3_threshold_transport_source.csv")
        expected_policies = {"youden", "specificity_90", "specificity_95", "sensitivity_80"}
        expected_datasets = {"val", "test", "external"}
        record("Figure3", "Threshold rows", len(data), 12, len(data) == 12)
        record("Figure3", "Threshold policy set", set(data["threshold_policy"]), expected_policies, set(data["threshold_policy"]) == expected_policies)
        record("Figure3", "Threshold dataset set", set(data["dataset"]), expected_datasets, set(data["dataset"]) == expected_datasets)
        record("Figure3", "Unique policy-dataset pairs", int(data[["threshold_policy", "dataset"]].drop_duplicates().shape[0]), 12, data[["threshold_policy", "dataset"]].drop_duplicates().shape[0] == 12)
        bounded = data[["sensitivity", "specificity", "ppv"]].apply(lambda column: column.between(0, 1)).all().all()
        record("Figure3", "Operating metrics in [0,1]", bool(bounded), True, bool(bounded))

    if "FigureS2" in selected:
        data = read_csv(SUPP / "FigureS2_configuration_specialization" / "FigureS2_configuration_source.csv")
        record("FigureS2", "B0-B6 rows", (len(data), set(data["Model"])), (7, {f"B{i}" for i in range(7)}), len(data) == 7 and set(data["Model"]) == {f"B{i}" for i in range(7)})
        record("FigureS2", "Module indicators binary", set(np.unique(data[["CSSA", "LPAH", "CMGF"]].to_numpy())), {0, 1}, set(np.unique(data[["CSSA", "LPAH", "CMGF"]].to_numpy())).issubset({0, 1}))

    if "FigureS3" in selected:
        directory = SUPP / "FigureS3_perturbation_and_confusion"
        perturbation = read_csv(directory / "FigureS3_perturbation_source.csv")
        confusion = read_csv(directory / "FigureS3_confusion_source.csv")
        record("FigureS3", "Perturbation rows", len(perturbation), 21, len(perturbation) == 21)
        record("FigureS3", "Confusion cells", len(confusion), 49, len(confusion) == 49)
        record("FigureS3", "Confusion sample total", int(confusion["count"].sum()), 1520, int(confusion["count"].sum()) == 1520)
        row_sums = confusion.groupby("true_class")["row_fraction"].sum().to_numpy()
        record("FigureS3", "Confusion row fractions", row_sums.round(8).tolist(), "all 1.0", np.allclose(row_sums, 1.0, atol=1e-8))

    if "FigureS4" in selected:
        directory = SUPP / "FigureS4_exploratory_all6"
        burden = read_csv(directory / "FigureS4_inference_burden_source.csv")
        all6 = burden[burden["system"] == "all6 (six members)"].iloc[0]
        observed = (float(all6["stored_parameters"]), float(all6["gflops"]), float(all6["latency_ms"]), int(all6["evaluations"]))
        expected = (196325256.0, 29.446781184, 29.741485118865967, 6)
        record("FigureS4", "all6 arithmetic inference burden", observed, expected, np.allclose(observed[:3], expected[:3], atol=1e-9) and observed[3] == expected[3])
        systems = read_csv(directory / "FigureS4_calibration_descriptors_source.csv")
        record("FigureS4", "Exploratory score systems", set(systems["system_id"]), {"b2_seed_ensemble", "b6_seed_ensemble", "b2_b6_all6"}, set(systems["system_id"]) == {"b2_seed_ensemble", "b6_seed_ensemble", "b2_b6_all6"})

    if "FigureS5" in selected:
        directory = SUPP / "FigureS5_repeatability_efficiency"
        runs = read_csv(directory / "FigureS5_three_seed_runs_source.csv")
        expected_models = {"b2", "b4", "b6"}
        seed_sets = runs.groupby("model_id")["run_seed"].apply(lambda values: set(map(int, values))).to_dict()
        record("FigureS5", "Three-seed run count", len(runs), 9, len(runs) == 9)
        record("FigureS5", "Three-seed model set", set(runs["model_id"]), expected_models, set(runs["model_id"]) == expected_models)
        record("FigureS5", "Seed sets per model", seed_sets, {model: {42, 2024, 3407} for model in expected_models}, all(seed_sets.get(model) == {42, 2024, 3407} for model in expected_models))
        efficiency = read_csv(directory / "FigureS5_efficiency_source.csv")
        record("FigureS5", "B0-B6 efficiency rows", (len(efficiency), set(efficiency["model_id"])), (7, {f"B{i}" for i in range(7)}), len(efficiency) == 7 and set(efficiency["model_id"]) == {f"B{i}" for i in range(7)})

    if "FigureS6" in selected:
        directory = SUPP / "FigureS6_implementation_partition"
        b6m = read_csv(directory / "FigureS6_B6M_source.csv").iloc[0]
        observed_b6m = (float(b6m["Macro-F1"]), float(b6m["Internal Mel-AUC"]), float(b6m["External ROC-AUC"]), float(b6m["External PR-AUC"]), float(b6m["Latency_ms_b1"]))
        expected_b6m = (0.7849730525322682, 0.9186457105771038, 0.7367498945514459, 0.0995161919734074, 5.602953433990479)
        record("FigureS6", "B6-M canonical endpoints", observed_b6m, expected_b6m, np.allclose(observed_b6m, expected_b6m, atol=1e-12))
        split = read_csv(directory / "FigureS6_crosssplit_source.csv").set_index("cohort")
        observed_split = tuple(float(split.loc["independent_split_seed2024", column]) for column in ["Macro-F1", "Internal Mel-AUC", "External ROC-AUC", "External PR-AUC"])
        expected_split = (0.8180385473497754, 0.9051136363636363, 0.7738832371831128, 0.130182174675528)
        record("FigureS6", "Split-2024 canonical endpoints", observed_split, expected_split, np.allclose(observed_split, expected_split, atol=1e-12))

    if "FigureS7" in selected:
        data = read_csv(SUPP / "FigureS7_localization" / "FigureS7_localization_source.csv").set_index("model")
        record("FigureS7", "Localization model set", set(data.index), {"B4", "B6"}, set(data.index) == {"B4", "B6"})
        complete = bool(((data["n"] == 200) & (data["generated_saliency_maps"] == 200) & (data["missing_image_count"] == 0) & (data["saliency_failure_count"] == 0)).all())
        record("FigureS7", "Localization completeness", complete, True, complete)
        observed = (float(data.loc["B4", "saliency_mask_iou"]), float(data.loc["B6", "saliency_mask_iou"]), float(data.loc["B4", "pointing_game_hit_rate"]), float(data.loc["B6", "pointing_game_hit_rate"]))
        expected = (0.5229188082844071, 0.5232957921396152, 0.955, 0.975)
        record("FigureS7", "Localization canonical endpoints", observed, expected, np.allclose(observed, expected, atol=1e-12))

    report = pd.DataFrame(checks)
    report.to_csv(AUDIT / "numeric_contract_qa.csv", index=False)
    if report.empty or not report["passed"].all():
        raise AssertionError("Numeric contract QA failed; inspect audit/numeric_contract_qa.csv")
    return report


def write_automated_qa_report(raster: pd.DataFrame, vector: pd.DataFrame, numeric: pd.DataFrame) -> None:
    report = f"""# Automated figure QA

Status: **PASS**

- Figures checked: {len(vector)}
- Raster checks passed: {int(raster['passed'].sum())}/{len(raster)}
- Vector checks passed: {int(vector['passed'].sum())}/{len(vector)}
- Numeric contracts passed: {int(numeric['passed'].sum())}/{len(numeric)}
- PDF page contract: one page per figure, width 80-185.5 mm, height no more than 170 mm
- PDF text contract: extractable editable text with a rendered glyph floor of 5 pt
- SVG contract: parseable XML with editable text nodes
- Raster contract: nonblank PNG, pixel-identical 600-dpi TIFF, minimum width 3,000 px
- Submission contract: each submission PDF is byte-identical to its canonical figure PDF

This automated report does not replace the separate panel-by-panel visual review at final physical size. See `MANUAL_VISUAL_QA.md` after that review is completed.
"""
    (AUDIT / "AUTOMATED_QA.md").write_text(report, encoding="utf-8")
    summary = {
        "status": "passed",
        "figures_checked": int(len(vector)),
        "raster_checks_passed": int(raster["passed"].sum()),
        "vector_checks_passed": int(vector["passed"].sum()),
        "numeric_contracts_passed": int(numeric["passed"].sum()),
    }
    (AUDIT / "automated_qa_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def build_generated_manifest() -> pd.DataFrame:
    manifest_path = AUDIT / "generated_file_manifest.csv"
    generated = []
    for path in sorted(OUT.rglob("*")):
        if not path.is_file() or path == manifest_path or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        generated.append({"file": str(path.relative_to(OUT)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    report = pd.DataFrame(generated)
    report.to_csv(manifest_path, index=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--figure",
        default="all",
        choices=["all", "Figure2", "Figure3", "FigureS2", "FigureS3", "FigureS4", "FigureS5", "FigureS6", "FigureS7"],
    )
    args = parser.parse_args()
    configure_plotting()
    prepare_dirs(clean=args.figure == "all")
    specs = [
        FigureSpec("Figure2", "External melanoma discrimination and patient-clustered comparison under cohort shift", MAIN, build_figure2),
        FigureSpec("Figure3", "Transport of validation-derived decision thresholds across cohorts", MAIN, build_figure3),
        FigureSpec("FigureS2", "Configuration-level specialization", SUPP, build_figure_s2),
        FigureSpec("FigureS3", "Controlled perturbation sensitivity and B6 class-level error structure", SUPP, build_figure_s3),
        FigureSpec("FigureS4", "Exploratory score integration and associated inference burden", SUPP, build_figure_s4),
        FigureSpec("FigureS5", "Fixed-split training repeatability and computational context", SUPP, build_figure_s5),
        FigureSpec("FigureS6", "Implementation and partition sensitivity", SUPP, build_figure_s6),
        FigureSpec("FigureS7", "Quantitative lesion-localization characterization", SUPP, build_figure_s7),
    ]
    selected = specs if args.figure == "all" else [spec for spec in specs if spec.figure_id == args.figure]
    for spec in selected:
        print(f"Building {spec.figure_id}: {spec.title}", flush=True)
        spec.builder()
    build_launchers([spec.figure_id for spec in specs])
    build_manifest(specs)
    selected_ids = [spec.figure_id for spec in selected]
    raster_report = validate_rasters(selected_ids)
    vector_report = validate_vectors(selected_ids)
    numeric_report = validate_numeric_contracts(selected_ids)
    write_automated_qa_report(raster_report, vector_report, numeric_report)
    status = {
        "status": "completed",
        "built": selected_ids,
        "png_files_checked": int(len(raster_report)),
        "pdf_svg_files_checked": int(len(vector_report)),
        "numeric_contracts_checked": int(len(numeric_report)),
        "source_manifest_rows": int(len(pd.read_csv(SOURCE / "figure_source_manifest.csv"))),
    }
    (AUDIT / "build_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    generated_manifest = build_generated_manifest()
    status["generated_manifest_rows"] = int(len(generated_manifest))
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()

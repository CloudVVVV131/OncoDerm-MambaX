from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
DEFAULT_TABLE = HERE / "canonical_sources" / "mask200_table.csv"
DEFAULT_STATUS = HERE / "canonical_sources" / "mask200_status.json"
DEFAULT_ID_MANIFEST = HERE / "S7_saliency_id_manifest.csv"

MM = 1 / 25.4
FIGURE_WIDTH_MM = 183
FIGURE_HEIGHT_MM = 76

INK = "#202124"
GRAY = "#687178"
MID_GRAY = "#9AA0A4"
GRID = "#E5E9EC"
B4 = "#E69F00"
B6 = "#D55E00"
B4_FILL = "#FFF4D6"
B6_FILL = "#FDE9E1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_canonical(table_path: Path, status_path: Path) -> tuple[dict[str, dict], dict]:
    if not table_path.is_file() or not status_path.is_file():
        raise FileNotFoundError("The canonical mask200 table or status file is missing.")

    with table_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with status_path.open("r", encoding="utf-8") as handle:
        status = json.load(handle)

    by_model: dict[str, dict] = {}
    for row in rows:
        run_id = row["run_id"]
        if "_b4_" in run_id:
            model = "B4"
        elif "_b6_" in run_id and "crosssplit" not in run_id and "b6m" not in run_id:
            model = "B6"
        else:
            continue
        by_model[model] = {
            "model": model,
            "run_id": run_id,
            "n": int(row["n"]),
            "saliency_mask_iou": float(row["saliency_mask_iou"]),
            "pointing_game_hit_rate": float(row["pointing_game_hit_rate"]),
            "lesion_background_saliency_ratio": float(row["lesion_background_saliency_ratio"]),
            "generated_saliency_maps": int(row["generated_saliency_maps"]),
            "missing_image_count": int(row["missing_image_count"]),
            "saliency_failure_count": int(row["saliency_failure_count"]),
        }

    if set(by_model) != {"B4", "B6"}:
        raise ValueError("The canonical table must contain exactly the final B4 and B6 rows.")
    if status.get("available_image_mask_pairs") != 200 or status.get("evaluated_mask_count") != 200:
        raise ValueError("The canonical status file does not describe the final 200-pair audit.")

    status_runs = {item["run_id"]: item for item in status.get("runs", [])}
    for model, row in by_model.items():
        if row["n"] != 200 or row["generated_saliency_maps"] != 200:
            raise ValueError(f"{model} does not have 200 evaluated pairs and 200 saliency maps.")
        if row["missing_image_count"] != 0 or row["saliency_failure_count"] != 0:
            raise ValueError(f"{model} has missing images or saliency failures.")
        run_status = status_runs.get(row["run_id"])
        if not run_status or run_status.get("status") != "completed":
            raise ValueError(f"{model} is not marked completed in the canonical status file.")
        if int(run_status.get("matched_masks", -1)) != row["n"]:
            raise ValueError(f"{model} table/status matched-mask counts disagree.")
        if int(run_status.get("generated_saliency_maps", -1)) != row["generated_saliency_maps"]:
            raise ValueError(f"{model} table/status saliency-map counts disagree.")
        if int(run_status.get("missing_image_count", -1)) != row["missing_image_count"]:
            raise ValueError(f"{model} table/status missing-image counts disagree.")
        if int(run_status.get("saliency_failure_count", -1)) != row["saliency_failure_count"]:
            raise ValueError(f"{model} table/status saliency-failure counts disagree.")

    return by_model, status


def verify_id_manifest(path: Path) -> int:
    if not path.is_file():
        raise FileNotFoundError("The matched B4/B6 saliency ID manifest is missing.")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"image_id", "b4_saliency_file", "b4_sha256", "b6_saliency_file", "b6_sha256"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("The saliency ID manifest is malformed.")
    ids = [row["image_id"] for row in rows]
    if len(rows) != 200 or len(set(ids)) != 200:
        raise ValueError("The saliency ID manifest must contain 200 unique matched image IDs.")
    hash_pattern = re.compile(r"^[0-9A-Fa-f]{64}$")
    for row in rows:
        if any(not row[field].strip() for field in required):
            raise ValueError(f"The saliency ID manifest has a blank required field for {row['image_id']!r}.")
        if not hash_pattern.fullmatch(row["b4_sha256"]) or not hash_pattern.fullmatch(row["b6_sha256"]):
            raise ValueError(f"The saliency ID manifest has an invalid SHA-256 value for {row['image_id']}.")
    return len(rows)


def write_source_data(
    by_model: dict[str, dict], table_path: Path, status_path: Path, output_dir: Path
) -> Path:
    output = output_dir / "FigureS7_source_data.csv"
    fields = [
        "model_id",
        "run_id",
        "dataset",
        "n_paired_masks",
        "metric",
        "metric_value",
        "metric_unit",
        "generated_saliency_maps",
        "missing_image_count",
        "saliency_failure_count",
        "canonical_table_sha256",
        "canonical_status_sha256",
    ]
    metrics = [
        ("saliency_mask_iou", "proportion"),
        ("pointing_game_hit_rate", "proportion"),
        ("lesion_background_saliency_ratio", "ratio"),
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model in ("B4", "B6"):
            row = by_model[model]
            for metric, unit in metrics:
                writer.writerow(
                    {
                        "model_id": model,
                        "run_id": row["run_id"],
                        "dataset": "ISIC 2018 segmentation image-mask pairs",
                        "n_paired_masks": row["n"],
                        "metric": metric,
                        "metric_value": format(row[metric], ".15g"),
                        "metric_unit": unit,
                        "generated_saliency_maps": row["generated_saliency_maps"],
                        "missing_image_count": row["missing_image_count"],
                        "saliency_failure_count": row["saliency_failure_count"],
                        "canonical_table_sha256": sha256(table_path),
                        "canonical_status_sha256": sha256(status_path),
                    }
                )
    return output


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.titlesize": 7.4,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.7,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.8,
            "ytick.major.size": 0,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def metric_axis(
    ax: plt.Axes,
    values: dict[str, float],
    title: str,
    xlim: tuple[float, float],
    ticks: list[float],
    decimals: int,
    show_model_labels: bool,
) -> None:
    y_positions = {"B4": 1.0, "B6": 0.0}
    colors = {"B4": B4, "B6": B6}
    markers = {"B4": "o", "B6": "D"}

    for model in ("B4", "B6"):
        y = y_positions[model]
        value = values[model]
        ax.hlines(y, xlim[0], value, color=colors[model], linewidth=2.1, alpha=0.32, zorder=2)
        ax.scatter(
            value,
            y,
            s=42,
            marker=markers[model],
            facecolor=colors[model],
            edgecolor="white",
            linewidth=0.65,
            zorder=4,
        )
        span = xlim[1] - xlim[0]
        if value > xlim[0] + 0.82 * span:
            offset, align = (-10, 0), "right"
        else:
            offset, align = (10, 0), "left"
        ax.annotate(
            f"{value:.{decimals}f}",
            xy=(value, y),
            xytext=offset,
            textcoords="offset points",
            ha=align,
            va="center",
            fontsize=6.8,
            fontweight="bold",
            color=INK,
            zorder=5,
        )

    ax.set_xlim(*xlim)
    ax.set_ylim(-0.58, 1.58)
    ax.set_xticks(ticks)
    ax.set_yticks([1.0, 0.0])
    if show_model_labels:
        ax.set_yticklabels(["B4", "B6"], fontweight="bold")
        ax.get_yticklabels()[0].set_color(B4)
        ax.get_yticklabels()[1].set_color(B6)
    else:
        ax.set_yticklabels([])
    ax.set_title(title, loc="left", pad=7, fontweight="bold", color=INK)
    ax.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="x", direction="out", pad=2)
    ax.tick_params(axis="y", pad=5)


def completeness_panel(ax: plt.Axes, by_model: dict[str, dict], matched_ids: int) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    columns = [
        ("Pairs", "n"),
        ("Maps", "generated_saliency_maps"),
        ("Missing", "missing_image_count"),
        ("Failures", "saliency_failure_count"),
    ]
    x_positions = [0.30, 0.50, 0.71, 0.90]
    y_positions = {"B4": 0.62, "B6": 0.37}

    ax.plot([0.05, 0.96], [0.78, 0.78], color=GRID, linewidth=0.8)
    ax.plot([0.05, 0.96], [0.49, 0.49], color=GRID, linewidth=0.65)
    ax.plot([0.05, 0.96], [0.24, 0.24], color=GRID, linewidth=0.8)
    for x, (label, _) in zip(x_positions, columns):
        ax.text(x, 0.84, label, ha="center", va="center", fontsize=6.1, color=GRAY)

    for model, fill, color in (("B4", B4_FILL, B4), ("B6", B6_FILL, B6)):
        y = y_positions[model]
        ax.add_patch(
            plt.Rectangle((0.055, y - 0.085), 0.90, 0.17, facecolor=fill, edgecolor="none", zorder=0)
        )
        ax.text(0.08, y, model, ha="left", va="center", fontsize=7.1, fontweight="bold", color=color)
        row = by_model[model]
        for x, (_, key) in zip(x_positions, columns):
            ax.text(x, y, str(row[key]), ha="center", va="center", fontsize=7.0, fontweight="bold", color=INK)

    ax.text(
        0.50,
        0.12,
        f"Matched B4/B6 saliency IDs: {matched_ids}/200",
        ha="center",
        va="center",
        fontsize=6.1,
        color=GRAY,
    )


def build_figure(by_model: dict[str, dict], matched_ids: int, output_dir: Path) -> list[Path]:
    configure_style()
    fig = plt.figure(figsize=(FIGURE_WIDTH_MM * MM, FIGURE_HEIGHT_MM * MM))
    outer = fig.add_gridspec(
        1,
        2,
        width_ratios=[2.52, 1.05],
        left=0.060,
        right=0.985,
        top=0.78,
        bottom=0.22,
        wspace=0.23,
    )
    metric_grid = outer[0].subgridspec(1, 3, width_ratios=[1.0, 1.0, 1.12], wspace=0.34)
    axes = [fig.add_subplot(metric_grid[0, index]) for index in range(3)]
    audit_ax = fig.add_subplot(outer[1])

    metric_axis(
        axes[0],
        {model: by_model[model]["saliency_mask_iou"] for model in ("B4", "B6")},
        "Saliency-mask IoU",
        (0.0, 1.0),
        [0.0, 0.25, 0.50, 0.75, 1.0],
        3,
        True,
    )
    metric_axis(
        axes[1],
        {model: by_model[model]["pointing_game_hit_rate"] for model in ("B4", "B6")},
        "Pointing-game hit rate",
        (0.0, 1.0),
        [0.0, 0.25, 0.50, 0.75, 1.0],
        3,
        False,
    )
    metric_axis(
        axes[2],
        {model: by_model[model]["lesion_background_saliency_ratio"] for model in ("B4", "B6")},
        "Lesion:background ratio",
        (0.0, 40.0),
        [0, 10, 20, 30, 40],
        1,
        False,
    )
    completeness_panel(audit_ax, by_model, matched_ids)

    fig.text(0.020, 0.925, "a", ha="left", va="top", fontsize=10.2, fontweight="bold", color=INK)
    fig.text(0.060, 0.925, "Aggregate localization profile", ha="left", va="top", fontsize=8.4, fontweight="bold", color=INK)
    fig.text(0.720, 0.925, "b", ha="left", va="top", fontsize=10.2, fontweight="bold", color=INK)
    fig.text(0.757, 0.925, "Audit completeness", ha="left", va="top", fontsize=8.4, fontweight="bold", color=INK)
    fig.text(
        0.060,
        0.095,
        "ISIC 2018; aggregate point estimates from all 200 available image-mask pairs per model.",
        ha="left",
        va="center",
        fontsize=6.1,
        color=GRAY,
    )
    fig.text(
        0.985,
        0.095,
        "No error bars: per-case metric records were not retained.",
        ha="right",
        va="center",
        fontsize=6.1,
        color=GRAY,
    )

    stem = output_dir / "FigureS7_quantitative_localization"
    outputs = [stem.with_suffix(ext) for ext in (".pdf", ".png", ".svg")]
    fig.savefig(outputs[0], metadata={"Title": "Supplementary Figure S7. Quantitative lesion-localization characterization"})
    fig.savefig(outputs[1], dpi=600)
    fig.savefig(outputs[2])
    plt.close(fig)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild Supplementary Figure S7 from canonical localization assets.")
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--id-manifest", type=Path, default=DEFAULT_ID_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()

    by_model, _ = read_canonical(args.table, args.status)
    matched_ids = verify_id_manifest(args.id_manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_path = write_source_data(by_model, args.table, args.status, args.output_dir)
    outputs = build_figure(by_model, matched_ids, args.output_dir)
    print(f"source_data={source_path}")
    for output in outputs:
        print(f"figure={output}")


if __name__ == "__main__":
    main()

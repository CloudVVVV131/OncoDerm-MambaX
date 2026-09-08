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
DEFAULT_TABLE = HERE / "source_data" / "mask200_table.csv"
DEFAULT_STATUS = HERE / "source_data" / "mask200_status.json"
DEFAULT_ID_MANIFEST = HERE / "source_data" / "S7_saliency_id_manifest.csv"

MM = 1 / 25.4
FIGURE_WIDTH_MM = 183
FIGURE_HEIGHT_MM = 55

INK = "#22272B"
GRAY = "#626B72"
AXIS = "#7C858B"
RULE = "#D9DEE2"
B4 = "#B47D18"
B6 = "#B65332"

MODEL_ORDER = ("B4", "B6")
MODEL_Y = {"B4": 1.0, "B6": 0.0}
MODEL_MARKER = {"B4": "o", "B6": "D"}
MODEL_COLOR = {"B4": B4, "B6": B6}


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

    if set(by_model) != set(MODEL_ORDER):
        raise ValueError("The canonical table must contain exactly the final B4 and B6 rows.")
    if status.get("available_image_mask_pairs") != 200 or status.get("evaluated_mask_count") != 200:
        raise ValueError("The status file must describe the 200-pair localization evaluation.")

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

    displayed = {
        "B4": (0.523, 0.955, 34.5),
        "B6": (0.523, 0.975, 10.7),
    }
    for model in MODEL_ORDER:
        row = by_model[model]
        observed = (
            round(row["saliency_mask_iou"], 3),
            round(row["pointing_game_hit_rate"], 3),
            round(row["lesion_background_saliency_ratio"], 1),
        )
        if observed != displayed[model]:
            raise ValueError(f"{model} canonical metrics no longer match the approved displayed values.")

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
            raise ValueError(f"The saliency ID manifest has a blank field for {row['image_id']!r}.")
        if not hash_pattern.fullmatch(row["b4_sha256"]) or not hash_pattern.fullmatch(row["b6_sha256"]):
            raise ValueError(f"The saliency ID manifest has an invalid SHA-256 for {row['image_id']}.")
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
        for model in MODEL_ORDER:
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
            "font.size": 6.8,
            "axes.titlesize": 6.8,
            "axes.labelsize": 6.5,
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.6,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.6,
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
    tick_labels: list[str],
    decimals: int,
    show_model_labels: bool,
) -> None:
    for model in MODEL_ORDER:
        y = MODEL_Y[model]
        value = values[model]
        color = MODEL_COLOR[model]
        ax.hlines(y, xlim[0], value, color=color, linewidth=1.45, alpha=0.58, zorder=2)
        ax.scatter(
            value,
            y,
            s=35,
            marker=MODEL_MARKER[model],
            facecolor=color,
            edgecolor="white",
            linewidth=0.65,
            zorder=4,
        )
        ax.annotate(
            f"{value:.{decimals}f}",
            xy=(value, y),
            xytext=(6.5, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=6.7,
            fontweight="semibold",
            color=INK,
            clip_on=False,
            zorder=5,
        )

    ax.set_xlim(*xlim)
    ax.set_ylim(-0.48, 1.48)
    ax.set_xticks(ticks, labels=tick_labels)
    ax.set_yticks([1.0, 0.0])
    if show_model_labels:
        ax.set_yticklabels(["B4", "B6"], fontweight="semibold")
        for tick, model in zip(ax.get_yticklabels(), MODEL_ORDER):
            tick.set_color(MODEL_COLOR[model])
    else:
        ax.set_yticklabels([])
    ax.set_title(title, loc="center", pad=5.5, fontweight="semibold", color=INK)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.spines["bottom"].set_linewidth(0.65)
    ax.tick_params(axis="x", colors=GRAY, direction="out", pad=2.2)
    ax.tick_params(axis="y", pad=5)


def completeness_panel(ax: plt.Axes, by_model: dict[str, dict], matched_ids: int) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    header_y = 0.77
    row_y = {"B4": 0.57, "B6": 0.37}
    columns = [
        ("Pairs", "n", 0.34),
        ("Maps", "generated_saliency_maps", 0.50),
        ("Missing", "missing_image_count", 0.69),
        ("Failures", "saliency_failure_count", 0.91),
    ]

    ax.text(0.02, header_y, "Model", ha="left", va="center", fontsize=5.25, fontweight="semibold", color=GRAY)
    for label, _, x in columns:
        ax.text(x, header_y, label, ha="center", va="center", fontsize=5.05, fontweight="semibold", color=GRAY)

    ax.plot([0.01, 0.99], [0.69, 0.69], color=RULE, linewidth=0.7)
    ax.plot([0.01, 0.99], [0.47, 0.47], color=RULE, linewidth=0.55)
    ax.plot([0.01, 0.99], [0.27, 0.27], color=RULE, linewidth=0.7)

    for model in MODEL_ORDER:
        y = row_y[model]
        color = MODEL_COLOR[model]
        ax.scatter(
            0.045,
            y,
            s=24,
            marker=MODEL_MARKER[model],
            facecolor=color,
            edgecolor="white",
            linewidth=0.55,
            transform=ax.transAxes,
            zorder=3,
        )
        ax.text(0.10, y, model, ha="left", va="center", fontsize=6.4, fontweight="semibold", color=color)
        row = by_model[model]
        for _, key, x in columns:
            ax.text(x, y, str(row[key]), ha="center", va="center", fontsize=6.25, fontweight="semibold", color=INK)

    ax.text(0.02, 0.12, "Matched saliency IDs", ha="left", va="center", fontsize=5.65, color=GRAY)
    ax.text(
        0.98,
        0.12,
        f"{matched_ids} / 200",
        ha="right",
        va="center",
        fontsize=6.1,
        fontweight="semibold",
        color=INK,
    )


def build_figure(by_model: dict[str, dict], matched_ids: int, output_dir: Path) -> list[Path]:
    configure_style()
    fig = plt.figure(figsize=(FIGURE_WIDTH_MM * MM, FIGURE_HEIGHT_MM * MM))
    outer = fig.add_gridspec(
        1,
        2,
        width_ratios=[3.05, 1.0],
        left=0.052,
        right=0.985,
        top=0.755,
        bottom=0.19,
        wspace=0.16,
    )
    metric_grid = outer[0].subgridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.38)
    axes = [fig.add_subplot(metric_grid[0, index]) for index in range(3)]
    audit_ax = fig.add_subplot(outer[1])

    metric_axis(
        axes[0],
        {model: by_model[model]["saliency_mask_iou"] for model in MODEL_ORDER},
        "Saliency-mask IoU",
        (0.0, 1.0),
        [0.0, 0.5, 1.0],
        ["0", "0.5", "1.0"],
        3,
        True,
    )
    metric_axis(
        axes[1],
        {model: by_model[model]["pointing_game_hit_rate"] for model in MODEL_ORDER},
        "Pointing-game hit rate",
        (0.0, 1.0),
        [0.0, 0.5, 1.0],
        ["0", "0.5", "1.0"],
        3,
        False,
    )
    metric_axis(
        axes[2],
        {model: by_model[model]["lesion_background_saliency_ratio"] for model in MODEL_ORDER},
        "Lesion-to-background saliency ratio",
        (0.0, 40.0),
        [0.0, 20.0, 40.0],
        ["0", "20", "40"],
        1,
        False,
    )
    completeness_panel(audit_ax, by_model, matched_ids)

    panel_b_left = audit_ax.get_position().x0
    fig.text(0.018, 0.925, "a", ha="left", va="top", fontsize=9.2, fontweight="bold", color=INK)
    fig.text(
        0.052,
        0.925,
        "Quantitative lesion localization",
        ha="left",
        va="top",
        fontsize=8.25,
        fontweight="semibold",
        color=INK,
    )
    fig.text(panel_b_left - 0.027, 0.925, "b", ha="left", va="top", fontsize=9.2, fontweight="bold", color=INK)
    fig.text(
        panel_b_left,
        0.925,
        "Evaluation completeness",
        ha="left",
        va="top",
        fontsize=8.0,
        fontweight="semibold",
        color=INK,
    )

    stem = output_dir / "FigureS7_quantitative_localization_refined"
    outputs = [stem.with_suffix(ext) for ext in (".svg", ".pdf", ".png")]
    fig.savefig(outputs[0])
    fig.savefig(
        outputs[1],
        metadata={"Title": "Supplementary Figure S7. Quantitative lesion-localization characterization"},
    )
    fig.savefig(outputs[2], dpi=600)
    plt.close(fig)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the visually refined Supplementary Figure S7.")
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--id-manifest", type=Path, default=DEFAULT_ID_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=HERE.parent / "build/figures/S7")
    args = parser.parse_args()
    if args.output_dir.resolve() == (HERE / "source_data").resolve() or (HERE / "source_data").resolve() in args.output_dir.resolve().parents:
        raise ValueError("Output must not be inside source_data.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Use a new output directory: {args.output_dir}")

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

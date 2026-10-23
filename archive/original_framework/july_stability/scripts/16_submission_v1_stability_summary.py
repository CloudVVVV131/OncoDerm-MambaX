from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from lesionmamba.utils.config import load_yaml
from lesionmamba.utils.io import ensure_dir, read_json, write_json


MODEL_INFO = {
    "b2": {
        "model_name": "oncoderm_mambax_b2_metadata_cmgf",
        "setting": "B2 + CMGF metadata",
    },
    "b4": {
        "model_name": "oncoderm_mambax_b4_lpah_mask_aux",
        "setting": "B4 + LPAH mask aux",
    },
    "b6": {
        "model_name": "oncoderm_mambax_b6_full",
        "setting": "B6 full OncoDerm-MambaX",
    },
}

RUN_RE = re.compile(r"^(?P<model>oncoderm_mambax_b[246].*?)_seed(?P<seed>\d+)_(?P<stamp>.+)$")

INTERNAL_METRICS = [
    "Acc",
    "BAcc",
    "Macro-F1",
    "Macro-AUC",
    "Internal Mel-AUC",
    "Mel-Sens",
]
EXTERNAL_METRICS = [
    "External Mel-AUC",
    "External PR-AUC",
    "External Sensitivity",
    "External Specificity",
    "External Brier",
    "External ECE",
    "External Drop",
]
CALIBRATION_METRICS = [
    "temperature",
    "ece_before",
    "ece_after",
    "brier_before",
    "brier_after",
    "nll_before",
    "nll_after",
]
EXPLAIN_METRICS = [
    "mask_n",
    "saliency_mask_iou",
    "pointing_game_hit_rate",
    "lesion_background_saliency_ratio",
]
EFFICIENCY_METRICS = [
    "Params",
    "FLOPs",
    "model_size_mb",
    "latency_ms_b1",
    "throughput_img_s",
    "peak_mem_mb",
]
SUMMARY_METRICS = INTERNAL_METRICS + EXTERNAL_METRICS + CALIBRATION_METRICS + EXPLAIN_METRICS + EFFICIENCY_METRICS


def optional_table(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def first_value(df: pd.DataFrame, run_id: str, col: str) -> Any:
    if df.empty or "run_id" not in df.columns or col not in df.columns:
        return None
    match = df[df["run_id"] == run_id]
    if match.empty:
        return None
    value = match.iloc[0].get(col)
    if pd.isna(value):
        return None
    return value


def detect_model_key(model_name: str, run_name: str) -> str | None:
    for key, info in MODEL_INFO.items():
        if model_name == info["model_name"] or run_name.startswith(f"{info['model_name']}_seed"):
            return key
    return None


def parse_seed_from_run_name(run_name: str) -> int | None:
    match = RUN_RE.match(run_name)
    if not match:
        return None
    return int(match.group("seed"))


def candidate_runs(runs_dir: Path, model_keys: list[str], seeds: list[int]) -> dict[tuple[str, int], list[Path]]:
    buckets: dict[tuple[str, int], list[Path]] = {(model_key, seed): [] for model_key in model_keys for seed in seeds}
    for run_dir in sorted(runs_dir.glob("*")):
        if not run_dir.is_dir() or not (run_dir / "config.yaml").exists():
            continue
        run_seed = parse_seed_from_run_name(run_dir.name)
        if run_seed is None:
            continue
        try:
            cfg = load_yaml(run_dir / "config.yaml")
        except Exception:
            continue
        model_name = str(cfg.get("model", {}).get("name", ""))
        model_key = detect_model_key(model_name, run_dir.name)
        if model_key is None:
            continue
        if (model_key, run_seed) in buckets:
            buckets[(model_key, run_seed)].append(run_dir)
    return buckets


def latest_run(run_dirs: list[Path]) -> Path | None:
    if not run_dirs:
        return None
    return max(run_dirs, key=lambda p: (p.stat().st_mtime, p.name))


def fallback_internal_metrics(run_dir: Path) -> dict[str, Any]:
    metrics = read_json(run_dir / "metrics_test.json", default={}) or {}
    return {
        "Acc": metrics.get("accuracy"),
        "BAcc": metrics.get("balanced_accuracy"),
        "Macro-F1": metrics.get("macro_f1"),
        "Macro-AUC": metrics.get("macro_auc"),
        "Internal Mel-AUC": metrics.get("mel_auc"),
        "Mel-Sens": metrics.get("mel_sensitivity"),
    }


def build_run_row(
    run_dir: Path,
    model_key: str,
    run_seed: int,
    candidate_count: int,
    tables: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    cfg = load_yaml(run_dir / "config.yaml")
    project_cfg = cfg.get("project", {})
    model_cfg = cfg.get("model", {})
    loss_cfg = cfg.get("loss", {})
    data_cfg = cfg.get("data", {})
    env = read_json(run_dir / "environment.json", default={}) or {}
    model_meta = env.get("model_metadata", {}) if isinstance(env, dict) else {}
    checkpoint_best = run_dir / "checkpoints" / "best_val_macro_f1.pth"
    checkpoint_last = run_dir / "checkpoints" / "last.pth"
    row: dict[str, Any] = {
        "model_id": model_key,
        "Setting": MODEL_INFO[model_key]["setting"],
        "Model": model_cfg.get("name", MODEL_INFO[model_key]["model_name"]),
        "run_seed": run_seed,
        "split_seed": project_cfg.get("split_seed", project_cfg.get("seed")),
        "project_seed": project_cfg.get("seed"),
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "candidate_count_for_model_seed": candidate_count,
        "config_path": str(run_dir / "config.yaml"),
        "checkpoint_best_exists": checkpoint_best.exists(),
        "checkpoint_last_exists": checkpoint_last.exists(),
        "metrics_val_exists": (run_dir / "metrics_val.json").exists(),
        "metrics_test_exists": (run_dir / "metrics_test.json").exists(),
        "CSSA": bool(model_cfg.get("cssa", {}).get("enabled", False)),
        "LPAH_Mask_Aux": bool(model_cfg.get("lesion_prior", {}).get("enabled", False)),
        "CMGF_Metadata": bool(model_cfg.get("metadata_fusion", {}).get("enabled", False)),
        "Melanoma_Head": bool(model_cfg.get("heads", {}).get("melanoma_endpoint", False)),
        "Multi_Task": bool(loss_cfg.get("multitask", False)),
        "Metadata_Input": bool(data_cfg.get("metadata", {}).get("enabled", False)),
        "Mask_Aux_Data": bool(data_cfg.get("mask_aux", {}).get("enabled", False)),
        "Backbone": model_cfg.get("timm_name"),
        "Pretrained": model_cfg.get("pretrained"),
        "SSM_Backend": model_meta.get("effective_ssm_backend"),
        "Inference_Requires_Mask": model_meta.get("inference_requires_mask", False),
    }

    internal_fallback = fallback_internal_metrics(run_dir)
    for src, dst in [
        ("Acc", "Acc"),
        ("BAcc", "BAcc"),
        ("Macro-F1", "Macro-F1"),
        ("Macro-AUC", "Macro-AUC"),
        ("Mel-AUC", "Internal Mel-AUC"),
        ("Mel-Sens", "Mel-Sens"),
    ]:
        row[dst] = first_value(tables["all_metrics"], run_dir.name, src)
        if row[dst] is None:
            row[dst] = internal_fallback.get(dst)

    for src, dst in [
        ("External Mel-AUC", "External Mel-AUC"),
        ("pr_auc", "External PR-AUC"),
        ("sensitivity", "External Sensitivity"),
        ("specificity", "External Specificity"),
        ("brier", "External Brier"),
        ("ece", "External ECE"),
        ("Drop", "External Drop"),
    ]:
        row[dst] = first_value(tables["external"], run_dir.name, src)

    for col in CALIBRATION_METRICS:
        row[col] = first_value(tables["calibration"], run_dir.name, col)

    for src, dst in [
        ("n", "mask_n"),
        ("saliency_mask_iou", "saliency_mask_iou"),
        ("pointing_game_hit_rate", "pointing_game_hit_rate"),
        ("lesion_background_saliency_ratio", "lesion_background_saliency_ratio"),
    ]:
        row[dst] = first_value(tables["explain"], run_dir.name, src)

    for src, dst in [
        ("Params", "Params"),
        ("FLOPs", "FLOPs"),
        ("model_size_mb", "model_size_mb"),
        ("latency_ms_b1", "latency_ms_b1"),
        ("throughput_img_s", "throughput_img_s"),
        ("peak_mem_mb", "peak_mem_mb"),
    ]:
        row[dst] = first_value(tables["all_metrics"], run_dir.name, src)
        if row[dst] is None:
            row[dst] = first_value(tables["efficiency"], run_dir.name, src.lower() if src in {"Params", "FLOPs"} else src)
    return row


def make_mean_std_table(runs_df: pd.DataFrame, model_keys: list[str]) -> pd.DataFrame:
    rows = []
    for model_key in model_keys:
        sub = runs_df[runs_df["model_id"] == model_key].copy()
        row: dict[str, Any] = {
            "model_id": model_key,
            "Setting": MODEL_INFO[model_key]["setting"],
            "n_runs": len(sub),
            "seeds": ";".join(str(x) for x in sorted(pd.to_numeric(sub["run_seed"], errors="coerce").dropna().astype(int).unique())),
        }
        for metric in SUMMARY_METRICS:
            if metric not in sub.columns:
                continue
            values = pd.to_numeric(sub[metric], errors="coerce").dropna()
            row[f"{metric}_n"] = int(values.shape[0])
            row[f"{metric}_mean"] = float(values.mean()) if not values.empty else np.nan
            row[f"{metric}_std"] = float(values.std(ddof=1)) if values.shape[0] > 1 else 0.0 if values.shape[0] == 1 else np.nan
            row[f"{metric}_min"] = float(values.min()) if not values.empty else np.nan
            row[f"{metric}_max"] = float(values.max()) if not values.empty else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def make_vs_baseline_table(runs_df: pd.DataFrame, mean_df: pd.DataFrame, model_keys: list[str]) -> pd.DataFrame:
    rows = []
    key_metrics = [
        "Macro-F1",
        "Macro-AUC",
        "Internal Mel-AUC",
        "Mel-Sens",
        "External Mel-AUC",
        "External PR-AUC",
        "External Sensitivity",
        "External Specificity",
        "External ECE",
        "External Brier",
    ]
    for model_key in model_keys:
        model_runs = runs_df[runs_df["model_id"] == model_key]
        seed42 = model_runs[pd.to_numeric(model_runs["run_seed"], errors="coerce") == 42]
        mean_row = mean_df[mean_df["model_id"] == model_key]
        for metric in key_metrics:
            ref = None
            if not seed42.empty and metric in seed42.columns:
                ref = pd.to_numeric(seed42[metric], errors="coerce").dropna()
                ref = float(ref.iloc[0]) if not ref.empty else None
            mean_value = None
            std_value = None
            n_value = 0
            if not mean_row.empty and f"{metric}_mean" in mean_row.columns:
                raw_mean = mean_row.iloc[0].get(f"{metric}_mean")
                mean_value = None if pd.isna(raw_mean) else float(raw_mean)
                raw_std = mean_row.iloc[0].get(f"{metric}_std")
                std_value = None if pd.isna(raw_std) else float(raw_std)
                raw_n = mean_row.iloc[0].get(f"{metric}_n")
                n_value = 0 if pd.isna(raw_n) else int(raw_n)
            seed_values = {}
            for _, run in model_runs.iterrows():
                seed = run.get("run_seed")
                value = run.get(metric)
                if pd.notna(seed) and pd.notna(value):
                    seed_values[f"seed{int(seed)}_value"] = float(value)
                    if ref is not None:
                        seed_values[f"seed{int(seed)}_delta_vs_seed42"] = float(value) - ref
            rows.append(
                {
                    "model_id": model_key,
                    "Setting": MODEL_INFO[model_key]["setting"],
                    "metric": metric,
                    "seed42_reference": ref,
                    "mean_value": mean_value,
                    "std_value": std_value,
                    "n_values": n_value,
                    "delta_mean_minus_seed42": None if ref is None or mean_value is None else mean_value - ref,
                    **seed_values,
                }
            )
    return pd.DataFrame(rows)


def maybe_write_figures(mean_df: pd.DataFrame, figures_dir: Path) -> list[str]:
    written: list[str] = []
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return written
    ensure_dir(figures_dir)
    for metric, filename in [
        ("External Mel-AUC", "submission_v1_stability_external_auc.png"),
        ("Macro-F1", "submission_v1_stability_macro_f1.png"),
    ]:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        if mean_col not in mean_df.columns:
            continue
        plot_df = mean_df[["model_id", mean_col, std_col]].dropna(subset=[mean_col])
        if plot_df.empty:
            continue
        fig, ax = plt.subplots(figsize=(6.0, 3.6), dpi=160)
        ax.bar(plot_df["model_id"], plot_df[mean_col], yerr=plot_df[std_col].fillna(0.0), capsize=4)
        ax.set_title(f"{metric} mean +/- std")
        ax.set_xlabel("Model")
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        out = figures_dir / filename
        fig.savefig(out)
        plt.close(fig)
        written.append(str(out))
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="outputs/runs")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--figures-dir", default="figures/stability")
    parser.add_argument("--models", nargs="+", default=["b2", "b4", "b6"])
    parser.add_argument("--required-seeds", nargs="+", type=int, default=[42, 2024])
    args = parser.parse_args()

    model_keys = [m.lower() for m in args.models]
    invalid = [m for m in model_keys if m not in MODEL_INFO]
    if invalid:
        raise SystemExit(f"Unknown model ids: {invalid}")

    runs_dir = Path(args.runs_dir)
    results_dir = Path(args.results_dir)
    output_dir = ensure_dir(args.output_dir or results_dir)
    tables = {
        "all_metrics": optional_table(results_dir / "all_runs_metrics_table.csv"),
        "external": optional_table(results_dir / "external_test_table.csv"),
        "calibration": optional_table(results_dir / "calibration_table.csv"),
        "explain": optional_table(results_dir / "explainability_mask_table.csv"),
        "efficiency": optional_table(results_dir / "efficiency_table.csv"),
    }

    buckets = candidate_runs(runs_dir, model_keys, args.required_seeds)
    selected: dict[tuple[str, int], Path] = {}
    missing_runs = []
    found_runs: dict[str, dict[str, str]] = {model_key: {} for model_key in model_keys}
    candidate_counts: dict[tuple[str, int], int] = {}
    for model_key in model_keys:
        for seed in args.required_seeds:
            candidates = buckets.get((model_key, seed), [])
            candidate_counts[(model_key, seed)] = len(candidates)
            chosen = latest_run(candidates)
            if chosen is None:
                missing_runs.append({"model_id": model_key, "run_seed": seed})
            else:
                selected[(model_key, seed)] = chosen
                found_runs[model_key][str(seed)] = chosen.name

    rows = []
    for (model_key, seed), run_dir in sorted(selected.items()):
        rows.append(build_run_row(run_dir, model_key, seed, candidate_counts[(model_key, seed)], tables))
    runs_df = pd.DataFrame(rows)
    if not runs_df.empty:
        runs_df = runs_df.sort_values(["model_id", "run_seed"]).reset_index(drop=True)

    mean_df = make_mean_std_table(runs_df, model_keys) if not runs_df.empty else pd.DataFrame()
    vs_df = make_vs_baseline_table(runs_df, mean_df, model_keys) if not runs_df.empty else pd.DataFrame()

    runs_path = output_dir / "submission_v1_stability_runs_table.csv"
    mean_path = output_dir / "submission_v1_stability_mean_std_table.csv"
    vs_path = output_dir / "submission_v1_stability_vs_baseline_table.csv"
    status_path = output_dir / "submission_v1_stability_status.json"
    runs_df.to_csv(runs_path, index=False)
    mean_df.to_csv(mean_path, index=False)
    vs_df.to_csv(vs_path, index=False)
    figure_paths = maybe_write_figures(mean_df, Path(args.figures_dir))

    split_mismatches = []
    if not runs_df.empty and "split_seed" in runs_df.columns:
        for _, row in runs_df.iterrows():
            try:
                split_seed = int(row.get("split_seed"))
            except Exception:
                split_seed = None
            if split_seed != 42:
                split_mismatches.append({"run_id": row.get("run_id"), "split_seed": row.get("split_seed")})

    metric_missing = []
    important_metrics = ["Macro-F1", "Internal Mel-AUC", "External Mel-AUC", "External PR-AUC"]
    for _, row in runs_df.iterrows():
        missing = [metric for metric in important_metrics if metric not in row or pd.isna(row.get(metric))]
        if missing:
            metric_missing.append({"run_id": row.get("run_id"), "missing_metrics": missing})

    tables_written = [str(runs_path), str(mean_path), str(vs_path), str(status_path), *figure_paths]
    notes = []
    for name, table in tables.items():
        if table.empty:
            notes.append(f"{name} table missing or empty")
    if split_mismatches:
        notes.append("Some selected runs do not use split_seed=42")
    if metric_missing:
        notes.append("Some selected runs are missing important metrics")

    status = {
        "status": "completed" if not missing_runs and not split_mismatches else "completed_or_partial",
        "required_models": model_keys,
        "required_seeds": args.required_seeds,
        "found_runs": found_runs,
        "missing_runs": missing_runs,
        "split_mismatches": split_mismatches,
        "metric_missing": metric_missing,
        "tables_written": tables_written,
        "notes": notes,
    }
    write_json(status, status_path)
    print(f"wrote stability runs={len(runs_df)} missing={len(missing_runs)} status={status['status']}")


if __name__ == "__main__":
    main()


from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from lesionmamba.data.datasets import LABEL_MAP
from lesionmamba.metrics.bootstrap import bootstrap_ci
from lesionmamba.metrics.melanoma_endpoint import binary_stats, endpoint_metrics_from_probs, endpoint_metrics_from_scores
from lesionmamba.utils.config import load_yaml
from lesionmamba.utils.io import ensure_dir, write_json


def iter_run_dirs(base_dir: Path, selected_runs: list[str] | None, run_glob: str | None) -> list[Path]:
    if selected_runs:
        return [Path(run) for run in selected_runs]
    if run_glob:
        return sorted(base_dir.glob(run_glob))
    return sorted(path for path in base_dir.glob("*") if path.is_dir())


def load_pred(run_id: str, split: str):
    base = Path("outputs/predictions")
    labels = np.load(base / f"{run_id}_{split}_labels.npy")
    probs = np.load(base / f"{run_id}_{split}_probs.npy")
    endpoint_path = base / f"{run_id}_{split}_endpoint_probs.npy"
    endpoint_scores = np.load(endpoint_path) if endpoint_path.exists() else probs[:, LABEL_MAP["mel"]]
    return labels, probs, endpoint_scores


def bootstrap_iters_for_run(run_dir: Path) -> int:
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        return 1000
    cfg = load_yaml(cfg_path)
    return int(cfg.get("eval", {}).get("bootstrap_iters", 1000))


def endpoint_ci_columns(labels: np.ndarray, scores: np.ndarray, thresholds: dict[str, float], iters: int) -> dict[str, float]:
    mel_idx = LABEL_MAP["mel"]

    def roc_auc_metric(y: np.ndarray, p: np.ndarray) -> float:
        yy = (y == mel_idx).astype(int)
        if len(np.unique(yy)) < 2:
            return float("nan")
        return float(roc_auc_score(yy, p))

    def sensitivity_metric(y: np.ndarray, p: np.ndarray) -> float:
        yy = (y == mel_idx).astype(int)
        return float(binary_stats(yy, p, thresholds["youden"])["sensitivity"])

    def specificity_metric(y: np.ndarray, p: np.ndarray) -> float:
        yy = (y == mel_idx).astype(int)
        return float(binary_stats(yy, p, thresholds["youden"])["specificity"])

    ci_metrics = {
        "roc_auc": roc_auc_metric,
        "sensitivity": sensitivity_metric,
        "specificity": specificity_metric,
    }
    out: dict[str, float] = {}
    for name, metric_fn in ci_metrics.items():
        ci = bootstrap_ci(labels, scores, metric_fn, iters=iters)
        out[f"{name}_ci_lower"] = ci["ci_lower"]
        out[f"{name}_ci_upper"] = ci["ci_upper"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-runs", default="outputs/runs")
    parser.add_argument("--selected-runs", nargs="*", default=None)
    parser.add_argument("--run-glob", default=None)
    args = parser.parse_args()
    rows = []
    threshold_rows = {}
    for run_dir in iter_run_dirs(Path(args.all_runs), args.selected_runs, args.run_glob):
        if not run_dir.is_dir():
            continue
        try:
            val_labels, val_probs, val_scores = load_pred(run_dir.name, "val")
            test_labels, test_probs, test_scores = load_pred(run_dir.name, "test")
        except FileNotFoundError:
            continue
        _, thresholds = endpoint_metrics_from_scores(val_labels, val_scores)
        metrics, _ = endpoint_metrics_from_scores(test_labels, test_scores, thresholds=thresholds)
        ci_cols = endpoint_ci_columns(test_labels, test_scores, thresholds, bootstrap_iters_for_run(run_dir))
        write_json(thresholds, run_dir / "melanoma_thresholds.json")
        threshold_rows[run_dir.name] = {"source": "validation", **thresholds}
        row = {
            "run_id": run_dir.name,
            **metrics,
            **ci_cols,
            **{f"threshold_{k}": v for k, v in thresholds.items()},
        }
        rows.append(row)
    ensure_dir("results")
    columns = [
        "run_id",
        "roc_auc",
        "roc_auc_ci_lower",
        "roc_auc_ci_upper",
        "pr_auc",
        "sensitivity",
        "sensitivity_ci_lower",
        "sensitivity_ci_upper",
        "specificity",
        "specificity_ci_lower",
        "specificity_ci_upper",
        "sens_at_90_spec",
        "sens_at_95_spec",
        "brier",
        "ece",
        "threshold_youden",
        "threshold_spec90",
        "threshold_spec95",
    ]
    new_df = pd.DataFrame(rows, columns=columns if not rows else None)
    out_path = Path("results/melanoma_endpoint_table.csv")
    merge_existing = bool(args.selected_runs or args.run_glob)
    if merge_existing and out_path.exists() and not new_df.empty:
        old_df = pd.read_csv(out_path)
        if "run_id" in old_df.columns:
            old_df = old_df[~old_df["run_id"].isin(new_df["run_id"])]
        out_df = pd.concat([old_df, new_df], ignore_index=True, sort=False)
    else:
        out_df = new_df
    out_df.to_csv(out_path, index=False)
    write_json(threshold_rows, "results/melanoma_thresholds.json")
    write_json(
        {
            "status": "completed_or_partial",
            "rows": len(out_df),
            "new_or_updated_rows": len(new_df),
            "threshold_source": "validation",
            "merge_existing": merge_existing,
        },
        "results/melanoma_endpoint_status.json",
    )
    print(f"wrote {len(out_df)} endpoint rows; new_or_updated={len(new_df)}")


if __name__ == "__main__":
    main()

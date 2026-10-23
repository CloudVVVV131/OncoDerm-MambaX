from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from lesionmamba.analysis.plots import save_reliability_diagram
from lesionmamba.metrics.calibration import calibration_curve_bins, expected_calibration_error, multiclass_brier
from lesionmamba.utils.config import load_yaml
from lesionmamba.utils.io import ensure_dir, write_json


def iter_run_dirs(base_dir: Path, selected_runs: list[str] | None, run_glob: str | None) -> list[Path]:
    if selected_runs:
        return [Path(run) for run in selected_runs]
    if run_glob:
        return sorted(base_dir.glob(run_glob))
    return sorted(path for path in base_dir.glob("*") if path.is_dir())


def fit_temperature(logits: np.ndarray, labels: np.ndarray, min_temp: float = 0.5, max_temp: float = 5.0) -> float:
    t = torch.nn.Parameter(torch.ones(1))
    x = torch.tensor(logits, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    opt = torch.optim.LBFGS([t], lr=0.05, max_iter=50)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(x / t.clamp(min_temp, max_temp), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(t.detach().clamp(min_temp, max_temp).item())


def softmax(logits: np.ndarray, temp: float = 1.0) -> np.ndarray:
    x = logits / temp
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-runs", default="outputs/runs")
    parser.add_argument("--config", default="configs/eval/calibration.yaml")
    parser.add_argument("--selected-runs", nargs="*", default=None)
    parser.add_argument("--run-glob", default=None)
    args = parser.parse_args()
    cal_cfg = load_yaml(args.config) if Path(args.config).exists() else {}
    ece_bins = int(cal_cfg.get("ece_bins", 15))
    min_temp = float(cal_cfg.get("min_temperature", 0.5))
    max_temp = float(cal_cfg.get("max_temperature", 5.0))
    rows = []
    pred_dir = Path("outputs/predictions")
    for run_dir in iter_run_dirs(Path(args.all_runs), args.selected_runs, args.run_glob):
        if not run_dir.is_dir():
            continue
        try:
            val_logits = np.load(pred_dir / f"{run_dir.name}_val_logits.npy")
            val_labels = np.load(pred_dir / f"{run_dir.name}_val_labels.npy")
            test_logits = np.load(pred_dir / f"{run_dir.name}_test_logits.npy")
            test_labels = np.load(pred_dir / f"{run_dir.name}_test_labels.npy")
        except FileNotFoundError:
            continue
        temp = fit_temperature(val_logits, val_labels, min_temp=min_temp, max_temp=max_temp)
        before = softmax(test_logits, 1.0)
        after = softmax(test_logits, temp)
        row = {
            "run_id": run_dir.name,
            "temperature": temp,
            "ece_bins": ece_bins,
            "temperature_source": "validation",
            "temperature_bounds": f"[{min_temp}, {max_temp}]",
            "ece_before": expected_calibration_error(test_labels, before, n_bins=ece_bins),
            "ece_after": expected_calibration_error(test_labels, after, n_bins=ece_bins),
            "brier_before": multiclass_brier(test_labels, before),
            "brier_after": multiclass_brier(test_labels, after),
            "nll_before": float(F.cross_entropy(torch.tensor(test_logits, dtype=torch.float32), torch.tensor(test_labels)).item()),
            "nll_after": float(
                F.cross_entropy(torch.tensor(test_logits / temp, dtype=torch.float32), torch.tensor(test_labels)).item()
            ),
        }
        rows.append(row)
        write_json(
            {"temperature": temp, "source": "validation", "ece_bins": ece_bins, "bounds": [min_temp, max_temp]},
            run_dir / "temperature.json",
        )
        conf_before, acc_before = calibration_curve_bins(test_labels, before, n_bins=ece_bins)
        conf_after, acc_after = calibration_curve_bins(test_labels, after, n_bins=ece_bins)
        save_reliability_diagram(conf_before, acc_before, Path("figures/calibration") / f"{run_dir.name}_before.png")
        save_reliability_diagram(conf_after, acc_after, Path("figures/calibration") / f"{run_dir.name}_after.png")
    ensure_dir("results")
    columns = [
        "run_id",
        "temperature",
        "ece_bins",
        "temperature_source",
        "temperature_bounds",
        "ece_before",
        "ece_after",
        "brier_before",
        "brier_after",
        "nll_before",
        "nll_after",
    ]
    new_df = pd.DataFrame(rows, columns=columns if not rows else None)
    out_path = Path("results/calibration_table.csv")
    merge_existing = bool(args.selected_runs or args.run_glob)
    if merge_existing and out_path.exists() and not new_df.empty:
        old_df = pd.read_csv(out_path)
        if "run_id" in old_df.columns:
            old_df = old_df[~old_df["run_id"].isin(new_df["run_id"])]
        out_df = pd.concat([old_df, new_df], ignore_index=True, sort=False)
    else:
        out_df = new_df
    out_df.to_csv(out_path, index=False)
    print(f"wrote {len(out_df)} calibration rows; new_or_updated={len(new_df)}")


if __name__ == "__main__":
    main()

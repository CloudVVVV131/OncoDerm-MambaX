from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from lesionmamba.utils.config import load_yaml
from lesionmamba.utils.io import read_json, write_csv


def collect_run_metrics(runs_dir: str | Path = "outputs/runs") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_dir in Path(runs_dir).glob("*"):
        if not run_dir.is_dir():
            continue
        cfg = load_yaml(run_dir / "config.yaml") if (run_dir / "config.yaml").exists() else {}
        metrics = read_json(run_dir / "metrics_test.json", default={})
        env = read_json(run_dir / "environment.json", default={})
        row = {
            "run_id": run_dir.name,
            "model": (env.get("model_metadata") or {}).get(
                "model_name", cfg.get("model", {}).get("name", run_dir.name.split("_seed")[0])
            ),
            "config_path": str(run_dir / "config.yaml"),
            "Acc": metrics.get("accuracy"),
            "BAcc": metrics.get("balanced_accuracy"),
            "Macro-F1": metrics.get("macro_f1"),
            "Macro-AUC": metrics.get("macro_auc"),
            "Mel-AUC": metrics.get("mel_auc"),
            "Mel-Sens": metrics.get("mel_sensitivity"),
        }
        rows.append(row)
    return pd.DataFrame(rows)

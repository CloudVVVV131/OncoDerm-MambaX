from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from lesionmamba.utils.io import ensure_dir, read_json


def main() -> None:
    rows = []
    for run_dir in Path("outputs/runs").glob("lesionmamba_loss_*"):
        metrics = read_json(run_dir / "metrics_test.json", default={})
        rows.append({"run_id": run_dir.name, **metrics})
    ensure_dir("results")
    columns = ["run_id", "accuracy", "balanced_accuracy", "macro_f1", "macro_auc", "mel_auc", "mel_sensitivity"]
    pd.DataFrame(rows, columns=columns if not rows else None).to_csv("results/loss_ablation_table.csv", index=False)
    print(f"wrote loss ablation rows={len(rows)}")


if __name__ == "__main__":
    main()

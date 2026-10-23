from __future__ import annotations

import argparse
from pathlib import Path
import sys
from datetime import datetime, timezone
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from lesionmamba.utils.config import load_yaml
from lesionmamba.utils.io import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optional V1.2 subgroup/robustness helper.")
    parser.add_argument("--config", default="configs/eval/submission_v1_evidence.yaml")
    parser.add_argument("--result-root", default=None)
    return parser.parse_args()


def binary_metrics_at_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = (scores >= threshold).astype(int)
    labels = labels.astype(int)
    tp = float(((pred == 1) & (labels == 1)).sum())
    fp = float(((pred == 1) & (labels == 0)).sum())
    tn = float(((pred == 0) & (labels == 0)).sum())
    fn = float(((pred == 0) & (labels == 1)).sum())
    return {
        "n": float(len(labels)),
        "n_positive": float(labels.sum()),
        "sensitivity": tp / max(tp + fn, 1.0),
        "specificity": tn / max(tn + fp, 1.0),
    }


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    root = Path(args.result_root or config.get("paths", {}).get("result_root", ".")).resolve()
    results_dir = ensure_dir(root / config.get("paths", {}).get("results_dir", "results"))
    status: dict[str, Any] = {
        "status": "completed_with_skips",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "result_root": str(root),
        "tables_written": [],
        "skipped": [],
        "notes": [],
    }

    metadata_path = root / "data/raw/HAM10000/HAM10000_metadata.csv"
    splits_path = root / "splits/ham10000_seed42_test.csv"
    evidence_path = results_dir / "submission_v1_evidence_ensemble_table.csv"
    if not metadata_path.exists() or not splits_path.exists():
        status["skipped"].append(
            {
                "task": "internal_subgroup_analysis",
                "reason": "metadata_or_split_not_found",
                "metadata_path": str(metadata_path),
                "split_path": str(splits_path),
            }
        )
    else:
        # This is a conservative metadata availability audit. Full subgroup scoring
        # needs image_id order exported with test endpoint arrays, which older runs
        # may not contain. The audit still documents usable subgroup fields.
        meta = pd.read_csv(metadata_path)
        split = pd.read_csv(splits_path)
        id_cols = [c for c in ["image_id", "lesion_id"] if c in meta.columns and c in split.columns]
        rows = []
        for field in ["sex", "localization", "anatom_site_general"]:
            if field in meta.columns:
                vc = meta[field].fillna("unknown").astype(str).value_counts()
                for group, n in vc.items():
                    rows.append({"field": field, "group": group, "n_metadata": int(n)})
        if "age" in meta.columns:
            age = pd.to_numeric(meta["age"], errors="coerce")
            rows.append({"field": "age", "group": "non_missing", "n_metadata": int(age.notna().sum())})
            rows.append({"field": "age", "group": "missing", "n_metadata": int(age.isna().sum())})
        out = results_dir / "submission_v1_evidence_internal_subgroup_audit.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        status["tables_written"].append(str(out))
        if not id_cols:
            status["skipped"].append(
                {
                    "task": "internal_subgroup_scored_metrics",
                    "reason": "no_common_id_column_between_metadata_and_split",
                }
            )
        else:
            status["notes"].append(
                "Subgroup metadata audit completed; scored subgroup metrics require exported test image_id order."
            )

    if not evidence_path.exists():
        status["skipped"].append(
            {
                "task": "robustness_inference_only",
                "reason": "P1 robustness requires checkpoints/images and is intentionally not part of P0 evidence pipeline.",
            }
        )
    else:
        status["skipped"].append(
            {
                "task": "robustness_inference_only",
                "reason": "not_run_by_default; run a dedicated GPU inference plan if robustness is needed.",
            }
        )

    write_json(status, results_dir / "submission_v1_evidence_p1_status.json")
    print(f"submission_v1_evidence P1 status={status['status']}")


if __name__ == "__main__":
    main()

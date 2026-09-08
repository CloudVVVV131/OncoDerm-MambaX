from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from lesionmamba.analysis.robustness import brightness_shift, gaussian_blur, hair_occlusion
from lesionmamba.data.datasets import HAM10000Dataset
from lesionmamba.data.splits import split_paths
from lesionmamba.data.transforms import build_transforms
from lesionmamba.engine.checkpoint import load_checkpoint
from lesionmamba.engine.evaluate import predict
from lesionmamba.metrics.classification import multiclass_metrics
from lesionmamba.models.registry import build_model_for_checkpoint
from lesionmamba.utils.config import load_yaml
from lesionmamba.utils.io import ensure_dir, write_json


def build_perturb_transform(config: dict, perturb: Callable | None):
    base = build_transforms(config, "eval")
    if perturb is None:
        return base

    def transform(image):
        return base(perturb(image))

    return transform


def metric_row(run_id: str, condition: str, level: str, metrics: dict) -> dict:
    return {
        "run_id": run_id,
        "condition": condition,
        "level": level,
        "accuracy": metrics.get("accuracy"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "weighted_f1": metrics.get("weighted_f1"),
        "macro_auc": metrics.get("macro_auc"),
        "mel_auc": metrics.get("mel_auc"),
        "mel_sensitivity": metrics.get("mel_sensitivity"),
    }


def add_drop_columns(rows: list[dict]) -> list[dict]:
    clean_by_run = {
        row["run_id"]: row
        for row in rows
        if row.get("condition") == "clean" and row.get("level") == "none"
    }
    for row in rows:
        clean = clean_by_run.get(row["run_id"])
        if not clean or row.get("condition") == "clean":
            row["macro_f1_drop"] = 0.0 if clean else None
            row["mel_auc_drop"] = 0.0 if clean else None
            row["worst_case_drop"] = 0.0 if clean else None
            continue
        macro_drop = None
        mel_drop = None
        if clean.get("macro_f1") is not None and row.get("macro_f1") is not None:
            macro_drop = clean["macro_f1"] - row["macro_f1"]
        if clean.get("mel_auc") is not None and row.get("mel_auc") is not None:
            mel_drop = clean["mel_auc"] - row["mel_auc"]
        drops = [x for x in [macro_drop, mel_drop] if x is not None]
        row["macro_f1_drop"] = macro_drop
        row["mel_auc_drop"] = mel_drop
        row["worst_case_drop"] = max(drops) if drops else None
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-runs", default="outputs/runs")
    parser.add_argument("--config", default="configs/eval/robustness_submission_v1.yaml")
    parser.add_argument("--max-samples", type=int, default=0, help="0 means full test split.")
    args = parser.parse_args()

    robust_cfg = load_yaml(args.config) if Path(args.config).exists() else {}
    target_models = set(robust_cfg.get("models", []))
    rows: list[dict] = []
    statuses: list[dict] = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for run_dir in Path(args.selected_runs).glob("*"):
        if not run_dir.is_dir():
            continue
        try:
            config = load_yaml(run_dir / "config.yaml")
            model_name = str(config.get("model", {}).get("name", run_dir.name))
            if target_models and model_name not in target_models and not any(m in run_dir.name for m in target_models):
                statuses.append({"run_id": run_dir.name, "status": "skipped", "reason": "not_in_robustness_model_list"})
                continue
            test_csv = split_paths(config)["test"]
            ckpt_path = run_dir / "checkpoints" / "best_val_macro_f1.pth"
            if not test_csv.exists():
                statuses.append({"run_id": run_dir.name, "status": "skipped", "reason": f"missing_test_split: {test_csv}"})
                continue
            if not ckpt_path.exists():
                statuses.append({"run_id": run_dir.name, "status": "skipped", "reason": "missing_best_checkpoint"})
                continue

            model = build_model_for_checkpoint(config).to(device).eval()
            model.load_state_dict(load_checkpoint(ckpt_path, map_location=device)["model"], strict=True)
            train_cfg = config.get("train", {})
            batch_size = int(train_cfg.get("batch_size", 64))
            amp = bool(train_cfg.get("amp", True))
            data_cfg = config["data"]

            conditions: list[tuple[str, str, Callable | None]] = [("clean", "none", None)]
            for k in robust_cfg.get("perturbations", {}).get("gaussian_blur", []):
                conditions.append(("gaussian_blur", str(k), lambda img, kk=int(k): gaussian_blur(img, kk)))
            for shift in robust_cfg.get("perturbations", {}).get("brightness_shift", []):
                conditions.append(("brightness_shift", str(shift), lambda img, ss=float(shift): brightness_shift(img, ss)))
            for lines in robust_cfg.get("perturbations", {}).get("hair_occlusion_lines", []):
                conditions.append(("hair_occlusion", str(lines), lambda img, ll=int(lines): hair_occlusion(img, ll)))

            for condition, level, perturb in conditions:
                ds = HAM10000Dataset(
                    test_csv,
                    data_cfg["image_dir"],
                    transform=build_perturb_transform(config, perturb),
                    label_column=data_cfg.get("label_column", "dx"),
                    image_id_column=data_cfg.get("image_id_column", "image_id"),
                    group_column=data_cfg.get("group_column", "lesion_id"),
                    patient_id_column=data_cfg.get("patient_id_column", "patient_id"),
                    metadata_config=data_cfg.get("metadata", {}),
                    input_size=int(data_cfg.get("input_size", 224)),
                )
                if args.max_samples > 0:
                    ds = Subset(ds, list(range(min(args.max_samples, len(ds)))))
                loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
                pred = predict(model, loader, device, amp=amp)
                metrics = multiclass_metrics(pred["labels"], pred["probs"]) if len(pred["labels"]) else {}
                rows.append(metric_row(run_dir.name, condition, level, metrics))
            statuses.append({"run_id": run_dir.name, "status": "completed", "conditions": len(conditions)})
        except Exception as exc:
            statuses.append({"run_id": run_dir.name, "status": "failed", "error": str(exc)})

    ensure_dir("results")
    rows = add_drop_columns(rows)
    columns = [
        "run_id",
        "condition",
        "level",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "macro_auc",
        "mel_auc",
        "mel_sensitivity",
        "macro_f1_drop",
        "mel_auc_drop",
        "worst_case_drop",
    ]
    pd.DataFrame(rows, columns=columns if not rows else None).to_csv("results/robustness_lite_table.csv", index=False)
    write_json(
        {
            "status": "completed_or_partial",
            "rows": len(rows),
            "runs": statuses,
            "config": robust_cfg,
        },
        "results/robustness_lite_status.json",
    )
    print(f"wrote {len(rows)} robustness rows")


if __name__ == "__main__":
    main()

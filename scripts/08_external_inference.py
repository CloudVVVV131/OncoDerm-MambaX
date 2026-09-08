from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from lesionmamba.data.datasets import ExternalEndpointDataset
from lesionmamba.data.transforms import build_transforms
from lesionmamba.engine.checkpoint import load_checkpoint
from lesionmamba.metrics.melanoma_endpoint import endpoint_metrics_from_probs, endpoint_metrics_from_scores
from lesionmamba.models.registry import build_model_for_checkpoint
from lesionmamba.utils.config import load_yaml
from lesionmamba.utils.io import ensure_dir, write_json


def build_summary_row(
    run_id: str,
    threshold_source: str,
    internal_multiclass_auc: float | None,
    external_metrics: dict,
    external_score_source: str,
) -> dict:
    return {
        **external_metrics,
        "run_id": run_id,
        "threshold_source": threshold_source,
        "internal_score_source": "seven_class_softmax_mel",
        "external_score_source": external_score_source,
        "Internal multiclass MEL-AUC": internal_multiclass_auc,
        "External endpoint ROC-AUC": external_metrics.get("roc_auc"),
    }


def merge_summary_rows(old_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    # Upgrade the summary schema without interpreting a cross-head difference.
    old_df = old_df.drop(columns=["Drop"], errors="ignore").copy()
    for old, current in {
        "Internal Mel-AUC": "Internal multiclass MEL-AUC",
        "External Mel-AUC": "External endpoint ROC-AUC",
    }.items():
        if old in old_df:
            if current in old_df:
                old_df[current] = old_df[current].combine_first(old_df[old])
                old_df = old_df.drop(columns=[old])
            else:
                old_df = old_df.rename(columns={old: current})
    if "run_id" in old_df:
        old_df = old_df[~old_df["run_id"].isin(new_df["run_id"])]
    return pd.concat([old_df, new_df], ignore_index=True, sort=False)


def iter_run_dirs(base_dir: Path, selected_runs: list[str] | None, run_glob: str | None) -> list[Path]:
    if selected_runs:
        return [Path(run) for run in selected_runs]
    if run_glob:
        return sorted(base_dir.glob(run_glob))
    return sorted(path for path in base_dir.glob("*") if path.is_dir())


def validation_thresholds(run_dir: Path) -> tuple[dict[str, float], str]:
    threshold_path = run_dir / "melanoma_thresholds.json"
    if threshold_path.exists():
        import json

        thresholds = json.loads(threshold_path.read_text(encoding="utf-8"))
        return {k.replace("threshold_", ""): float(v) for k, v in thresholds.items()}, "saved_validation_thresholds"
    pred_dir = Path("outputs/predictions")
    try:
        val_labels = np.load(pred_dir / f"{run_dir.name}_val_labels.npy")
        val_probs = np.load(pred_dir / f"{run_dir.name}_val_probs.npy")
        endpoint_path = pred_dir / f"{run_dir.name}_val_endpoint_probs.npy"
        if endpoint_path.exists():
            _, thresholds = endpoint_metrics_from_scores(val_labels, np.load(endpoint_path))
        else:
            _, thresholds = endpoint_metrics_from_probs(val_labels, val_probs)
        write_json(thresholds, threshold_path)
        return thresholds, "computed_from_validation_predictions"
    except FileNotFoundError as exc:
        raise FileNotFoundError("Validation predictions or saved validation thresholds are required; no default threshold is substituted.") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-runs", default="outputs/runs")
    parser.add_argument("--selected-runs", nargs="*", default=None)
    parser.add_argument("--run-glob", default=None)
    args = parser.parse_args()
    labels = Path("data/raw/ISIC_external/labels_harmonized.csv")
    ensure_dir("results")
    if not labels.exists():
        write_json(
            {"status": "not_run", "reason": "Missing data/raw/ISIC_external/labels_harmonized.csv"},
            "results/external_test_status.json",
        )
        raise FileNotFoundError(labels)
    image_dir = Path("data/raw/ISIC_external/images")
    rows = []
    pred_dir = ensure_dir("outputs/predictions")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for run_dir in iter_run_dirs(Path(args.all_runs), args.selected_runs, args.run_glob):
        if not run_dir.is_dir() or not (run_dir / "config.yaml").exists():
            continue
        config = load_yaml(run_dir / "config.yaml")
        ckpt_path = run_dir / "checkpoints" / "best_val_macro_f1.pth"
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Validation-selected checkpoint required: {ckpt_path}")
        thresholds, threshold_source = validation_thresholds(run_dir)
        dataset = ExternalEndpointDataset(
            labels,
            image_dir,
            transform=build_transforms(config, "eval"),
            metadata_config=config.get("data", {}).get("metadata", {}),
        )
        loader = DataLoader(dataset, batch_size=int(config["train"].get("batch_size", 64)), shuffle=False, num_workers=0)
        model = build_model_for_checkpoint(config).to(device)
        ckpt = load_checkpoint(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"], strict=True)
        model.eval()
        all_scores, all_labels, out_rows = [], [], []
        external_score_source = None
        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(device)
                metadata = batch.get("metadata")
                if isinstance(metadata, torch.Tensor):
                    metadata = metadata.to(device)
                out = model(images, metadata=metadata)
                logits = out["logits"].float().cpu().numpy()
                probs = np.exp(logits - logits.max(axis=1, keepdims=True))
                probs = probs / probs.sum(axis=1, keepdims=True)
                mel_logits = out.get("mel_logits")
                if isinstance(mel_logits, torch.Tensor):
                    scores = torch.sigmoid(mel_logits).float().cpu().numpy()
                    batch_score_source = "binary_head_sigmoid"
                else:
                    scores = probs[:, 4]
                    batch_score_source = "seven_class_softmax_mel"
                if external_score_source is not None and external_score_source != batch_score_source:
                    raise ValueError("External score source changed between batches.")
                external_score_source = batch_score_source
                all_scores.append(scores)
                all_labels.append(batch["label"].numpy())
                for i in range(len(batch["label"])):
                    out_rows.append(
                        {
                            "image_id": batch["image_id"][i],
                            "label": int(batch["label"][i]),
                            "prob_mel": float(probs[i, 4]),
                            "prob_mel_endpoint": float(scores[i]),
                            "path": batch["path"][i],
                        }
                    )
        scores_arr = np.concatenate(all_scores, axis=0)
        binary_labels = np.concatenate(all_labels, axis=0)
        # Convert binary external labels into 7-class-compatible labels for endpoint helper.
        labels7 = np.where(binary_labels == 1, 4, 5)
        metrics, _ = endpoint_metrics_from_scores(labels7, scores_arr, thresholds=thresholds)
        pd.DataFrame(out_rows).to_csv(pred_dir / f"{run_dir.name}_external_predictions.csv", index=False)
        internal_mel_auc = None
        metrics_path = run_dir / "metrics_test.json"
        if metrics_path.exists():
            import json

            internal_mel_auc = json.loads(metrics_path.read_text(encoding="utf-8")).get("mel_auc")
        rows.append(build_summary_row(
            run_dir.name, threshold_source, internal_mel_auc, metrics, external_score_source
        ))
    if not rows:
        raise ValueError("No external runs were evaluated. Specify existing --selected-runs directories.")
    columns = [
        "run_id",
        "threshold_source",
        "internal_score_source",
        "external_score_source",
        "Internal multiclass MEL-AUC",
        "External endpoint ROC-AUC",
        "roc_auc",
        "pr_auc",
        "sensitivity",
        "specificity",
        "brier",
        "ece",
    ]
    new_df = pd.DataFrame(rows, columns=columns if not rows else None)
    out_path = Path("results/external_test_table.csv")
    merge_existing = bool(args.selected_runs or args.run_glob)
    if merge_existing and out_path.exists() and not new_df.empty:
        old_df = pd.read_csv(out_path)
        out_df = merge_summary_rows(old_df, new_df)
    else:
        out_df = new_df
    out_df.to_csv(out_path, index=False)
    write_json(
        {
            "status": "completed",
            "labels_csv": str(labels),
            "rows": len(out_df),
            "new_or_updated_rows": len(new_df),
            "merge_existing": merge_existing,
        },
        "results/external_test_status.json",
    )
    print(f"wrote external status for {len(out_df)} rows; new_or_updated={len(new_df)}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader
import pandas as pd

from lesionmamba.analysis.attention_export import save_attention_map
from lesionmamba.analysis.gradcam import compute_gradcam, save_gradcam_heatmap
from lesionmamba.data.datasets import HAM10000Dataset
from lesionmamba.data.splits import split_paths
from lesionmamba.data.transforms import build_transforms
from lesionmamba.engine.checkpoint import load_checkpoint
from lesionmamba.models.registry import build_model
from lesionmamba.utils.config import load_yaml
from lesionmamba.utils.io import ensure_dir, write_json


LABEL_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def _append_unique(selected: list[tuple[str, str]], rows: pd.DataFrame, category: str, limit: int = 2) -> None:
    seen = {image_id for image_id, _ in selected}
    for _, row in rows.head(limit).iterrows():
        image_id = str(row["image_id"])
        if image_id not in seen:
            selected.append((image_id, category))
            seen.add(image_id)


def select_representative_samples(run_id: str, test_csv: Path, max_samples: int) -> dict[str, str]:
    pred_path = Path("outputs/predictions") / f"{run_id}_test_predictions.csv"
    if not pred_path.exists():
        return {}
    pred = pd.read_csv(pred_path)
    if pred.empty or "image_id" not in pred.columns:
        return {}
    if test_csv.exists():
        valid_ids = set(pd.read_csv(test_csv)["image_id"].astype(str))
        pred = pred[pred["image_id"].astype(str).isin(valid_ids)].copy()
    if pred.empty:
        return {}
    prob_cols = [f"prob_{name}" for name in LABEL_NAMES if f"prob_{name}" in pred.columns]
    if prob_cols:
        pred["max_prob"] = pred[prob_cols].max(axis=1)
    else:
        pred["max_prob"] = 0.0

    selected: list[tuple[str, str]] = []
    _append_unique(
        selected,
        pred[(pred["true_label_name"] == "mel") & (pred["pred_label_name"] == "mel")].sort_values("prob_mel", ascending=False),
        "melanoma_correct_high_confidence",
        limit=2,
    )
    _append_unique(
        selected,
        pred[(pred["true_label_name"] == "mel") & (pred["pred_label_name"] != "mel")].sort_values("prob_mel", ascending=True),
        "melanoma_false_negative",
        limit=2,
    )
    _append_unique(
        selected,
        pred[(pred["true_label_name"] != "mel") & (pred["pred_label_name"] == "mel")].sort_values("prob_mel", ascending=False),
        "melanoma_false_positive",
        limit=2,
    )
    _append_unique(
        selected,
        pred[pred["true_label_name"].isin(["bcc", "akiec"])].sort_values("max_prob", ascending=False),
        "bcc_or_akiec_example",
        limit=2,
    )
    _append_unique(
        selected,
        pred[(pred["true_label_name"] == "nv") & ((pred["pred_label_name"] != "nv") | (pred["max_prob"] < 0.70))].sort_values(
            "max_prob", ascending=True
        ),
        "nv_confusing_or_low_confidence",
        limit=2,
    )
    _append_unique(
        selected,
        pred[(pred["correct"] == False)].sort_values("max_prob", ascending=True),  # noqa: E712
        "misclassified_low_confidence",
        limit=2,
    )
    _append_unique(selected, pred.sort_values("max_prob", ascending=True), "uncertain_fallback", limit=max_samples)
    return dict(selected[:max_samples])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-runs", default="outputs/runs")
    parser.add_argument("--max-samples", type=int, default=10)
    args = parser.parse_args()
    out_dir = ensure_dir("figures/explainability/qualitative")
    rows = []
    sample_rows = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for run_dir in Path(args.selected_runs).glob("*"):
        if not run_dir.is_dir():
            continue
        gradcam_exported = 0
        exported = 0
        try:
            config = load_yaml(run_dir / "config.yaml")
            data_cfg = config["data"]
            test_csv = split_paths(config)["test"]
            selected_samples = select_representative_samples(run_dir.name, test_csv, args.max_samples)
            ds = HAM10000Dataset(
                test_csv,
                data_cfg["image_dir"],
                transform=build_transforms(config, "eval"),
                label_column=data_cfg.get("label_column", "dx"),
                image_id_column=data_cfg.get("image_id_column", "image_id"),
                group_column=data_cfg.get("group_column", "lesion_id"),
                patient_id_column=data_cfg.get("patient_id_column", "patient_id"),
                metadata_config=data_cfg.get("metadata", {}),
                input_size=int(data_cfg.get("input_size", 224)),
            )
            loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
            model = build_model(config).to(device).eval()
            ckpt_path = run_dir / "checkpoints" / "best_val_macro_f1.pth"
            if ckpt_path.exists():
                model.load_state_dict(load_checkpoint(ckpt_path, map_location=device)["model"], strict=False)
            else:
                rows.append({"run_id": run_dir.name, "status": "skipped", "reason": "missing_best_checkpoint"})
                continue
            for i, batch in enumerate(loader):
                image_id = str(batch["image_id"][0])
                if selected_samples and image_id not in selected_samples:
                    continue
                if not selected_samples and i >= args.max_samples:
                    break
                image = batch["image"].to(device)
                metadata = batch.get("metadata")
                metadata = metadata.to(device) if isinstance(metadata, torch.Tensor) else None
                cam, target_class, target_layer = compute_gradcam(model, image, metadata=metadata)
                save_gradcam_heatmap(cam, out_dir / f"{run_dir.name}_{image_id}_gradcam.png")
                gradcam_exported += 1
                with torch.no_grad():
                    out = model(image, metadata=metadata) if metadata is not None else model(image)
                    probs = torch.softmax(out["logits"].float().cpu(), dim=1)[0]
                    pred_label = int(probs.argmax().item())
                if out.get("attention") is not None:
                    save_attention_map(
                        out["attention"],
                        Path("outputs/attention_maps") / run_dir.name / f"{image_id}.png",
                    )
                    exported += 1
                sample_rows.append(
                    {
                        "run_id": run_dir.name,
                        "image_id": image_id,
                        "original_path": batch["path"][0],
                        "true_label": int(batch["label"][0]),
                        "pred_label": pred_label,
                        "pred_probability": float(probs[pred_label].item()),
                        "sample_category": selected_samples.get(image_id, "sequential_fallback"),
                        "gradcam_path": str(out_dir / f"{run_dir.name}_{image_id}_gradcam.png"),
                        "attention_path": str(Path("outputs/attention_maps") / run_dir.name / f"{image_id}.png")
                        if out.get("attention") is not None
                        else None,
                    }
                )
                if selected_samples and gradcam_exported >= min(args.max_samples, len(selected_samples)):
                    break
        except Exception as exc:
            rows.append({"run_id": run_dir.name, "status": "failed", "error": str(exc)})
            continue
        rows.append(
            {
                "run_id": run_dir.name,
                "status": "completed",
                "gradcam_exported": gradcam_exported,
                "attention_maps_exported": exported,
                "target_layer": target_layer if gradcam_exported else None,
                "last_target_class": target_class if gradcam_exported else None,
            }
        )
    write_json(
        {
            "status": "completed_or_partial",
            "note": "Grad-CAM uses the last Conv2d layer by default; LesionMamba attention maps are exported when the model returns attention.",
            "runs": rows,
        },
        "results/explainability_qualitative_status.json",
    )
    if sample_rows:
        pd.DataFrame(sample_rows).to_csv("results/explainability_qualitative_samples.csv", index=False)
    print(f"wrote explainability status for {len(rows)} runs")


if __name__ == "__main__":
    main()

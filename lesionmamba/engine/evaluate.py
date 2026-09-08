from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lesionmamba.data.datasets import ID_TO_LABEL
from lesionmamba.engine.amp import autocast_context
from lesionmamba.metrics.classification import multiclass_metrics
from lesionmamba.utils.io import ensure_dir, write_json


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    amp: bool = True,
) -> dict[str, Any]:
    model.eval()
    logits_list, labels_list, endpoint_scores_list = [], [], []
    rows = []
    for batch in tqdm(dataloader, desc="predict", leave=False):
        images = batch["image"].to(device)
        metadata = batch.get("metadata")
        if isinstance(metadata, torch.Tensor):
            metadata = metadata.to(device)
        labels = batch["label"].numpy()
        with autocast_context(device, amp):
            out = model(images, metadata=metadata)
            logits = out["logits"].detach().float().cpu().numpy()
            mel_logits = out.get("mel_logits")
            endpoint_scores = (
                torch.sigmoid(mel_logits).detach().float().cpu().numpy()
                if isinstance(mel_logits, torch.Tensor)
                else None
            )
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = probs / probs.sum(axis=1, keepdims=True)
        preds = probs.argmax(axis=1)
        logits_list.append(logits)
        labels_list.append(labels)
        if endpoint_scores is not None:
            endpoint_scores_list.append(endpoint_scores)
        for i in range(len(labels)):
            row = {
                "image_id": batch["image_id"][i],
                "lesion_id": batch["lesion_id"][i],
                "patient_id": batch["patient_id"][i],
                "true_label": int(labels[i]),
                "true_label_name": ID_TO_LABEL[int(labels[i])],
                "pred_label": int(preds[i]),
                "pred_label_name": ID_TO_LABEL[int(preds[i])],
                "correct": bool(preds[i] == labels[i]),
            }
            for cls_idx, cls_name in ID_TO_LABEL.items():
                row[f"prob_{cls_name}"] = float(probs[i, cls_idx])
            row["prob_mel_endpoint"] = (
                float(endpoint_scores[i]) if endpoint_scores is not None else float(probs[i, 4])
            )
            rows.append(row)
    logits_arr = np.concatenate(logits_list, axis=0) if logits_list else np.empty((0, 7))
    labels_arr = np.concatenate(labels_list, axis=0) if labels_list else np.empty((0,), dtype=int)
    probs_arr = np.exp(logits_arr - logits_arr.max(axis=1, keepdims=True))
    probs_arr = probs_arr / probs_arr.sum(axis=1, keepdims=True)
    endpoint_arr = (
        np.concatenate(endpoint_scores_list, axis=0)
        if endpoint_scores_list
        else probs_arr[:, 4] if len(probs_arr) else np.empty((0,), dtype=float)
    )
    return {"logits": logits_arr, "probs": probs_arr, "labels": labels_arr, "endpoint_probs": endpoint_arr, "rows": rows}


def save_predictions(pred: dict[str, Any], run_id: str, split: str, output_dir: str | Path = "outputs/predictions") -> dict[str, str]:
    out_dir = ensure_dir(output_dir)
    paths = {
        "csv": out_dir / f"{run_id}_{split}_predictions.csv",
        "logits": out_dir / f"{run_id}_{split}_logits.npy",
        "probs": out_dir / f"{run_id}_{split}_probs.npy",
        "endpoint_probs": out_dir / f"{run_id}_{split}_endpoint_probs.npy",
        "labels": out_dir / f"{run_id}_{split}_labels.npy",
    }
    pd.DataFrame(pred["rows"]).to_csv(paths["csv"], index=False)
    np.save(paths["logits"], pred["logits"])
    np.save(paths["probs"], pred["probs"])
    np.save(paths["endpoint_probs"], pred["endpoint_probs"])
    np.save(paths["labels"], pred["labels"])
    return {k: str(v) for k, v in paths.items()}


def evaluate_and_save(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    run_id: str,
    split: str,
    run_dir: str | Path,
    amp: bool = True,
) -> dict[str, Any]:
    pred = predict(model, dataloader, device, amp)
    save_predictions(pred, run_id, split)
    metrics = multiclass_metrics(pred["labels"], pred["probs"]) if len(pred["labels"]) else {}
    write_json(metrics, Path(run_dir) / f"metrics_{split}.json")
    return metrics

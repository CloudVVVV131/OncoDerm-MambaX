from __future__ import annotations

import time
from pathlib import Path
from typing import Any
import itertools

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lesionmamba.data.datasets import HAM10000Dataset, LABEL_MAP, SkinLesionMaskDataset, metadata_dim
from lesionmamba.data.splits import split_paths
from lesionmamba.data.transforms import build_transforms
from lesionmamba.engine.amp import autocast_context
from lesionmamba.engine.checkpoint import load_checkpoint, save_checkpoint
from lesionmamba.engine.evaluate import predict, save_predictions
from lesionmamba.losses.attention_regularization import attention_sparse_loss, attention_tv_loss
from lesionmamba.losses.cb_focal import build_classification_loss
from lesionmamba.losses.multitask import OncoDermMultiTaskLoss, use_multitask_loss
from lesionmamba.metrics.classification import multiclass_metrics
from lesionmamba.models.registry import build_model, model_metadata
from lesionmamba.utils.config import save_yaml
from lesionmamba.utils.env import environment_summary
from lesionmamba.utils.io import ensure_dir, write_json
from lesionmamba.utils.logger import CSVLogger, setup_logger
from lesionmamba.utils.seed import set_seed


def class_counts_from_csv(csv_path: str | Path, label_column: str = "dx") -> list[int]:
    df = pd.read_csv(csv_path)
    return [int((df[label_column] == label).sum()) for label in LABEL_MAP]


def build_dataloader(config: dict[str, Any], split: str, shuffle: bool = False) -> DataLoader:
    paths = split_paths(config)
    data_cfg = config["data"]
    metadata_cfg = data_cfg.get("metadata", {})
    if metadata_cfg.get("enabled", False):
        metadata_cfg = dict(metadata_cfg)
        metadata_cfg["metadata_dim"] = metadata_dim(metadata_cfg)
        config.setdefault("data", {}).setdefault("metadata", {}).update({"metadata_dim": metadata_cfg["metadata_dim"]})
    mask_cfg = data_cfg.get("mask_aux", {})
    dataset = HAM10000Dataset(
        csv_path=paths[split],
        image_dir=data_cfg["image_dir"],
        transform=build_transforms(config, "train" if split == "train" else "eval"),
        label_column=data_cfg.get("label_column", "dx"),
        image_id_column=data_cfg.get("image_id_column", "image_id"),
        group_column=data_cfg.get("group_column", "lesion_id"),
        patient_id_column=data_cfg.get("patient_id_column", "patient_id"),
        metadata_config=metadata_cfg,
        mask_config=mask_cfg,
        input_size=int(data_cfg.get("input_size", 224)),
    )
    return DataLoader(
        dataset,
        batch_size=int(config["train"].get("batch_size", 64)),
        shuffle=shuffle,
        num_workers=int(config["train"].get("num_workers", 8)),
        pin_memory=True,
    )


def build_aux_mask_dataloader(config: dict[str, Any]) -> DataLoader | None:
    data_cfg = config.get("data", {})
    mask_cfg = data_cfg.get("mask_aux", {})
    if not bool(mask_cfg.get("enabled", False)):
        return None
    image_dir = Path(mask_cfg.get("image_dir", "data/raw/ISIC_masks/images"))
    mask_dir = Path(mask_cfg.get("mask_dir", "data/raw/ISIC_masks/masks"))
    if not image_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError("Auxiliary mask supervision is enabled, but its image/mask directories are missing.")
    metadata_cfg = data_cfg.get("metadata", {})
    mdim = metadata_dim(metadata_cfg)
    dataset = SkinLesionMaskDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        transform=build_transforms(config, "eval"),
        input_size=int(data_cfg.get("input_size", 224)),
        metadata_dim_value=mdim,
    )
    if len(dataset) == 0:
        raise ValueError("Auxiliary mask supervision is enabled, but no paired masks were found.")
    expected = mask_cfg.get("expected_pairs")
    if expected is not None and len(dataset) != int(expected):
        raise ValueError(f"Expected {expected} auxiliary image-mask pairs; found {len(dataset)}.")
    return DataLoader(
        dataset,
        batch_size=int(mask_cfg.get("batch_size", min(int(config["train"].get("batch_size", 64)), 32))),
        shuffle=True,
        num_workers=int(mask_cfg.get("num_workers", 0)),
        pin_memory=True,
    )


def build_optimizer_scheduler(config: dict[str, Any], model: torch.nn.Module, steps_per_epoch: int):
    train_cfg = config["train"]
    lr = float(train_cfg.get("lr", 3e-4))
    backbone_lr = train_cfg.get("backbone_lr")
    adapter_lr = train_cfg.get("adapter_lr", train_cfg.get("head_lr", lr))
    if backbone_lr is not None and hasattr(model, "backbone"):
        backbone_params = []
        other_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("backbone."):
                backbone_params.append(param)
            else:
                other_params.append(param)
        param_groups = []
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": float(backbone_lr)})
        if other_params:
            param_groups.append({"params": other_params, "lr": float(adapter_lr)})
        if not param_groups:
            param_groups = list(model.parameters())
    else:
        param_groups = model.parameters()
    optimizer_name = str(train_cfg.get("optimizer", "adamw")).lower()
    optimizer_kwargs = {
        "lr": lr,
        "weight_decay": float(train_cfg.get("weight_decay", 0.05)),
    }
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(param_groups, **optimizer_kwargs)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(param_groups, **optimizer_kwargs)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}. Expected 'adamw' or 'adam'.")
    epochs = int(train_cfg.get("epochs", 40))
    warmup = int(train_cfg.get("warmup_epochs", 3))

    def lr_lambda(epoch: int):
        if epoch < warmup:
            return max((epoch + 1) / max(warmup, 1), 1e-6)
        progress = (epoch - warmup) / max(epochs - warmup, 1)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi))).item()

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler


@torch.no_grad()
def evaluate_loss(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: str,
    amp: bool,
) -> float:
    model.eval()
    total = 0.0
    n_seen = 0
    for batch in dataloader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        metadata = batch.get("metadata")
        if isinstance(metadata, torch.Tensor):
            metadata = metadata.to(device)
        with autocast_context(device, amp):
            out = model(images, metadata=metadata)
            if isinstance(criterion, OncoDermMultiTaskLoss):
                details = criterion(out, labels, batch.get("mask"), batch.get("has_mask"))
                loss = details["loss"]
            else:
                loss = criterion(out["logits"], labels)
        bs = images.shape[0]
        total += float(loss.detach()) * bs
        n_seen += bs
    return total / max(n_seen, 1)


def train_one_run(config: dict[str, Any]) -> Path:
    seed = int(config.get("project", {}).get("seed", 42))
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = config.get("model", {}).get("name", "model")
    run_id = f"{model_name}_seed{seed}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = ensure_dir(Path(config.get("project", {}).get("output_dir", "outputs")) / "runs" / run_id)
    logger = setup_logger(run_id, run_dir / "run.log")
    save_yaml(config, run_dir / "config.yaml")
    env = environment_summary()
    env["model_metadata"] = {}
    write_json(env, run_dir / "environment.json")
    train_loader = build_dataloader(config, "train", shuffle=True)
    val_loader = build_dataloader(config, "val", shuffle=False)
    counts = class_counts_from_csv(split_paths(config)["train"], config["data"].get("label_column", "dx"))
    write_json({"label_order": LABEL_MAP, "class_counts": counts}, run_dir / "class_weights.json")
    model = build_model(config).to(device)
    init_checkpoint = config.get("train", {}).get("init_checkpoint")
    if init_checkpoint:
        init_path = Path(init_checkpoint)
        if init_path.exists():
            ckpt = load_checkpoint(init_path, map_location=device)
            model.load_state_dict(ckpt["model"], strict=False)
            logger.info("initialized model weights from %s", init_path)
        else:
            raise FileNotFoundError(f"Requested initialization checkpoint does not exist: {init_path}")
    env["model_metadata"] = model_metadata(model)
    env["train_init_checkpoint"] = init_checkpoint
    write_json(env, run_dir / "environment.json")
    criterion: torch.nn.Module
    if use_multitask_loss(config):
        criterion = OncoDermMultiTaskLoss(config, counts).to(device)
    else:
        criterion = build_classification_loss(config, counts).to(device)
    optimizer, scheduler = build_optimizer_scheduler(config, model, len(train_loader))
    aux_mask_loader = build_aux_mask_dataloader(config)
    aux_mask_iter = itertools.cycle(aux_mask_loader) if aux_mask_loader is not None else None
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config["train"].get("amp", True)) and device == "cuda")
    fields = [
        "epoch",
        "train_loss",
        "train_cls_loss",
        "train_mel_loss",
        "train_mask_aux_loss",
        "train_consistency_loss",
        "train_aux_mask_loss",
        "train_sparse_loss",
        "train_tv_loss",
        "val_loss",
        "val_acc",
        "val_bacc",
        "val_macro_f1",
        "val_macro_auc",
        "val_mel_auc",
        "lr",
        "epoch_time",
        "best_flag",
    ]
    csv_logger = CSVLogger(run_dir / "training_log.csv", fields)
    best_metric = -1.0
    best_tie = -1.0
    best_epoch = -1
    patience = int(config["train"].get("early_stopping_patience", 8))
    epochs = int(config["train"].get("epochs", 40))
    lambda_s = float(config.get("loss", {}).get("lambda_sparse", 0.001))
    lambda_tv = float(config.get("loss", {}).get("lambda_tv", 0.005))
    for epoch in range(epochs):
        start = time.time()
        model.train()
        sums = {"loss": 0.0, "cls": 0.0, "mel": 0.0, "mask": 0.0, "cons": 0.0, "aux_mask": 0.0, "sparse": 0.0, "tv": 0.0}
        n_seen = 0
        for batch in tqdm(train_loader, desc=f"epoch {epoch+1}/{epochs}", leave=False):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            metadata = batch.get("metadata")
            if isinstance(metadata, torch.Tensor):
                metadata = metadata.to(device)
            masks = batch.get("mask")
            has_mask = batch.get("has_mask")
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, bool(config["train"].get("amp", True))):
                out = model(images, metadata=metadata)
                if isinstance(criterion, OncoDermMultiTaskLoss):
                    loss_details = criterion(out, labels, masks, has_mask)
                    cls_loss = loss_details["seven_class"]
                    loss = loss_details["loss"]
                    mel_loss = loss_details["melanoma"]
                    mask_loss = loss_details["mask_aux"]
                    cons_loss = loss_details["consistency"]
                else:
                    cls_loss = criterion(out["logits"], labels)
                    loss = cls_loss
                    mel_loss = loss.new_tensor(0.0)
                    mask_loss = loss.new_tensor(0.0)
                    cons_loss = loss.new_tensor(0.0)
                sparse = attention_sparse_loss(out.get("attention")).to(device)
                tv = attention_tv_loss(out.get("attention")).to(device)
                loss = loss + lambda_s * sparse + lambda_tv * tv
                aux_mask_loss = loss.new_tensor(0.0)
                if aux_mask_iter is not None and isinstance(criterion, OncoDermMultiTaskLoss):
                    aux_batch = next(aux_mask_iter)
                    aux_images = aux_batch["image"].to(device)
                    aux_metadata = aux_batch.get("metadata")
                    if isinstance(aux_metadata, torch.Tensor):
                        aux_metadata = aux_metadata.to(device)
                    aux_out = model(aux_images, metadata=aux_metadata)
                    aux_mask_loss = criterion.mask_auxiliary_loss(
                        aux_out,
                        aux_batch.get("mask"),
                        aux_batch.get("has_mask"),
                    )
                    loss = loss + aux_mask_loss
            scaler.scale(loss).backward()
            if float(config["train"].get("grad_clip", 0)) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["train"].get("grad_clip", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            bs = images.shape[0]
            n_seen += bs
            sums["loss"] += float(loss.detach()) * bs
            sums["cls"] += float(cls_loss.detach()) * bs
            sums["mel"] += float(mel_loss.detach()) * bs
            sums["mask"] += float(mask_loss.detach()) * bs
            sums["cons"] += float(cons_loss.detach()) * bs
            sums["aux_mask"] += float(aux_mask_loss.detach()) * bs
            sums["sparse"] += float(sparse.detach()) * bs
            sums["tv"] += float(tv.detach()) * bs
        scheduler.step()
        amp_enabled = bool(config["train"].get("amp", True))
        val_pred = predict(model, val_loader, device, amp_enabled)
        val_metrics = multiclass_metrics(val_pred["labels"], val_pred["probs"])
        write_json(val_metrics, run_dir / "metrics_val.json")
        val_loss = evaluate_loss(model, val_loader, criterion, device, amp_enabled)
        selection_metric = str(config["train"].get("selection_metric", "macro_f1"))
        tie_metric = str(config["train"].get("selection_tie_breaker", "mel_auc"))
        current = float(val_metrics.get(selection_metric, -1.0))
        tie = float(val_metrics.get(tie_metric, -1.0))
        best_flag = (current > best_metric) or (current == best_metric and tie > best_tie)
        if best_flag:
            best_metric = current
            best_tie = tie
            best_epoch = epoch
            save_checkpoint(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "best_metric": best_metric,
                    "config": config,
                    "label_map": LABEL_MAP,
                    "environment": env,
                },
                run_dir / "checkpoints" / "best_val_macro_f1.pth",
            )
            save_predictions(val_pred, run_id, "val")
            write_json(val_metrics, run_dir / "metrics_val_best.json")
        save_checkpoint(
            {"model": model.state_dict(), "epoch": epoch, "config": config, "label_map": LABEL_MAP, "environment": env},
            run_dir / "checkpoints" / "last.pth",
        )
        csv_logger.log(
            {
                "epoch": epoch + 1,
                "train_loss": sums["loss"] / max(n_seen, 1),
                "train_cls_loss": sums["cls"] / max(n_seen, 1),
                "train_mel_loss": sums["mel"] / max(n_seen, 1),
                "train_mask_aux_loss": sums["mask"] / max(n_seen, 1),
                "train_consistency_loss": sums["cons"] / max(n_seen, 1),
                "train_aux_mask_loss": sums["aux_mask"] / max(n_seen, 1),
                "train_sparse_loss": sums["sparse"] / max(n_seen, 1),
                "train_tv_loss": sums["tv"] / max(n_seen, 1),
                "val_loss": val_loss,
                "val_acc": val_metrics.get("accuracy"),
                "val_bacc": val_metrics.get("balanced_accuracy"),
                "val_macro_f1": val_metrics.get("macro_f1"),
                "val_macro_auc": val_metrics.get("macro_auc"),
                "val_mel_auc": val_metrics.get("mel_auc"),
                "lr": optimizer.param_groups[0]["lr"],
                "epoch_time": time.time() - start,
                "best_flag": best_flag,
            }
        )
        logger.info("epoch=%s macro_f1=%.4f mel_auc=%.4f best=%s", epoch + 1, current, tie, best_flag)
        if epoch - best_epoch >= patience:
            logger.info("early stopping at epoch %s", epoch + 1)
            break
    # Ensure a validation prediction exists even if no improvement branch ran.
    if not (Path("outputs/predictions") / f"{run_id}_val_predictions.csv").exists():
        val_pred = predict(model, val_loader, device, bool(config["train"].get("amp", True)))
        save_predictions(val_pred, run_id, "val")
        write_json(multiclass_metrics(val_pred["labels"], val_pred["probs"]), run_dir / "metrics_val.json")
    (run_dir / "TRAINING_COMPLETE").write_text("completed\n", encoding="utf-8")
    return run_dir

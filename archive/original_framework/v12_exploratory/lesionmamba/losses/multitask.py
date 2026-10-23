from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from lesionmamba.data.datasets import LABEL_MAP
from lesionmamba.losses.cb_focal import build_classification_loss


def dice_bce_loss(mask_logits: torch.Tensor, target: torch.Tensor, has_mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask_logits.shape[-2:] != target.shape[-2:]:
        mask_logits = F.interpolate(mask_logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
    bce = F.binary_cross_entropy_with_logits(mask_logits, target, reduction="none").mean(dim=(1, 2, 3))
    pred = torch.sigmoid(mask_logits)
    intersection = (pred * target).sum(dim=(1, 2, 3))
    denom = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 1.0 - (2.0 * intersection + 1.0) / (denom + 1.0)
    loss = bce + dice
    if has_mask is not None:
        weights = has_mask.float().view(-1).to(loss.device)
        if weights.sum() <= 0:
            return loss.new_tensor(0.0)
        loss = loss * weights
        return loss.sum() / weights.sum().clamp_min(1.0)
    return loss.mean()


class OncoDermMultiTaskLoss(nn.Module):
    def __init__(self, config: dict[str, Any], class_counts: list[int]) -> None:
        super().__init__()
        self.config = config
        loss_cfg = config.get("loss", {})
        self.classification = build_classification_loss(config, class_counts)
        mel_count = max(int(class_counts[LABEL_MAP["mel"]]), 1)
        non_mel_count = max(int(sum(class_counts) - mel_count), 1)
        pos_weight = float(loss_cfg.get("melanoma_pos_weight", non_mel_count / mel_count))
        self.register_buffer("mel_pos_weight", torch.tensor(pos_weight, dtype=torch.float32))
        self.lambda_7 = float(loss_cfg.get("lambda_7class", 1.0))
        self.lambda_mel = float(loss_cfg.get("lambda_melanoma", 0.5))
        self.lambda_mask = float(loss_cfg.get("lambda_mask_aux", 0.1))
        self.lambda_cons = float(loss_cfg.get("lambda_consistency", 0.05))

    def mask_auxiliary_loss(
        self,
        out: dict[str, torch.Tensor | None],
        mask: torch.Tensor | None,
        has_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        mask_logits = out.get("mask_logits")
        if mask_logits is None or mask is None:
            logits = out["logits"]
            assert logits is not None
            return logits.new_tensor(0.0)
        return self.lambda_mask * dice_bce_loss(mask_logits, mask.to(mask_logits.device), has_mask)

    def forward(
        self,
        out: dict[str, torch.Tensor | None],
        labels: torch.Tensor,
        mask: torch.Tensor | None = None,
        has_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        logits = out["logits"]
        assert logits is not None
        labels = labels.to(logits.device)
        seven = self.classification(logits, labels)
        total = self.lambda_7 * seven
        details = {
            "loss": total,
            "seven_class": seven.detach(),
            "melanoma": logits.new_tensor(0.0),
            "mask_aux": logits.new_tensor(0.0),
            "consistency": logits.new_tensor(0.0),
        }

        mel_logits = out.get("mel_logits")
        if mel_logits is not None:
            y_mel = (labels == LABEL_MAP["mel"]).float()
            mel_loss = F.binary_cross_entropy_with_logits(
                mel_logits,
                y_mel,
                pos_weight=self.mel_pos_weight.to(mel_logits.device),
            )
            total = total + self.lambda_mel * mel_loss
            class_mel_prob = torch.softmax(logits, dim=1)[:, LABEL_MAP["mel"]]
            endpoint_prob = torch.sigmoid(mel_logits)
            cons = F.mse_loss(endpoint_prob, class_mel_prob)
            total = total + self.lambda_cons * cons
            details["melanoma"] = mel_loss.detach()
            details["consistency"] = cons.detach()

        mask_loss = self.mask_auxiliary_loss(out, mask, has_mask)
        total = total + mask_loss
        details["mask_aux"] = mask_loss.detach()
        details["loss"] = total
        return details


def use_multitask_loss(config: dict[str, Any]) -> bool:
    model_cfg = config.get("model", {})
    loss_cfg = config.get("loss", {})
    if "multitask" in loss_cfg:
        return bool(loss_cfg.get("multitask", False))
    return bool(loss_cfg.get("multitask", False)) or model_cfg.get("source") == "oncoderm_mambax"

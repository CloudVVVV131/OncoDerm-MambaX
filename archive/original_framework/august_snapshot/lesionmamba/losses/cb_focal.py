from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def effective_num_weights(class_counts: list[int], beta: float = 0.999) -> torch.Tensor:
    counts = torch.tensor(class_counts, dtype=torch.float32)
    denom = (1.0 - torch.pow(torch.tensor(beta), counts)).clamp_min(1e-12)
    weights = (1.0 - beta) / denom
    weights = weights / weights.sum() * len(class_counts)
    return weights


class ClassBalancedFocalLoss(nn.Module):
    def __init__(
        self,
        class_counts: list[int],
        beta: float = 0.999,
        gamma: float = 2.0,
        label_smoothing: float = 0.05,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer("weights", effective_num_weights(class_counts, beta))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.label_smoothing / max(num_classes - 1, 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.label_smoothing)
        focal = torch.pow(1.0 - probs, self.gamma)
        weights = self.weights.to(logits.device).unsqueeze(0)
        return -(true_dist * focal * log_probs * weights).sum(dim=1).mean()


def build_classification_loss(config: dict, class_counts: list[int]) -> nn.Module:
    loss_cfg = config.get("loss", {})
    name = loss_cfg.get("name", "cb_focal")
    if name == "cb_focal":
        return ClassBalancedFocalLoss(
            class_counts,
            beta=float(loss_cfg.get("beta", 0.999)),
            gamma=float(loss_cfg.get("gamma", 2.0)),
            label_smoothing=float(loss_cfg.get("label_smoothing", 0.05)),
        )
    if name == "weighted_ce":
        weight = effective_num_weights(class_counts, float(loss_cfg.get("beta", 0.999)))
        return nn.CrossEntropyLoss(weight=weight)
    if name == "ce":
        return nn.CrossEntropyLoss(label_smoothing=float(loss_cfg.get("label_smoothing", 0.0)))
    raise ValueError(f"Unknown loss: {name}")

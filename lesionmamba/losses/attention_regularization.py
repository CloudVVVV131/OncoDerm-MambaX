from __future__ import annotations

import torch


def attention_sparse_loss(attention: torch.Tensor | None) -> torch.Tensor:
    if attention is None:
        return torch.tensor(0.0)
    return attention.mean()


def attention_tv_loss(attention: torch.Tensor | None) -> torch.Tensor:
    if attention is None:
        return torch.tensor(0.0)
    dh = torch.abs(attention[:, :, 1:, :] - attention[:, :, :-1, :]).mean()
    dw = torch.abs(attention[:, :, :, 1:] - attention[:, :, :, :-1]).mean()
    return dh + dw

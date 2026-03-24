from __future__ import annotations

import torch


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (pred - target) ** 2
    diff = diff[mask]
    if diff.numel() == 0:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    return diff.mean()

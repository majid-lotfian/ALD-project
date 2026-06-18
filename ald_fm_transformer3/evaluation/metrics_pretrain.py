from __future__ import annotations

import torch


def pretrain_metrics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    diff = (pred - target) ** 2
    masked = diff[mask]
    masked_mse = float(masked.mean().cpu()) if masked.numel() > 0 else 0.0
    return {'masked_mse': masked_mse}

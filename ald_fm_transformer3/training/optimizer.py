from __future__ import annotations

import math

import torch
from torch.optim.lr_scheduler import LambdaLR


def build_optimizer(parameters, cfg: dict) -> torch.optim.Optimizer:
    name = cfg.get('name', 'adamw').lower()
    lr = cfg.get('lr', 3e-4)
    wd = cfg.get('weight_decay', 1e-4)
    if name == 'adamw':
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=wd)
    if name == 'adam':
        return torch.optim.Adam(parameters, lr=lr, weight_decay=wd)
    raise ValueError(f'Unsupported optimizer: {name}')


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """
    Linear warmup then cosine decay to min_lr_ratio * base_lr.
    """
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(float(step) / float(warmup_steps), 1e-8)
        if step >= total_steps:
            return min_lr_ratio
        progress = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)

from __future__ import annotations

import torch


def build_optimizer(parameters, cfg: dict):
    name = cfg.get('name', 'adamw').lower()
    lr = cfg.get('lr', 1e-4)
    wd = cfg.get('weight_decay', 1e-4)
    if name == 'adamw':
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=wd)
    if name == 'adam':
        return torch.optim.Adam(parameters, lr=lr, weight_decay=wd)
    raise ValueError(f'Unsupported optimizer: {name}')

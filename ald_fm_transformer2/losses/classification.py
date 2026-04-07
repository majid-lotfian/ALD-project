from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def make_class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    w = 1.0 / counts
    w = w * (n_classes / w.sum())
    return torch.tensor(w, dtype=torch.float32)


def make_ce_loss(class_weights: torch.Tensor | None = None) -> nn.Module:
    if class_weights is None:
        return nn.CrossEntropyLoss()
    return nn.CrossEntropyLoss(weight=class_weights)

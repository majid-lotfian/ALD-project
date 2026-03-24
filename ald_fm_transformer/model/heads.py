from __future__ import annotations

import torch
import torch.nn as nn


class SharedRegressionHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, token_reps: torch.Tensor) -> torch.Tensor:
        return self.net(token_reps).squeeze(-1)


class ClassificationHead(nn.Module):
    def __init__(self, d_model: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, sample_repr: torch.Tensor) -> torch.Tensor:
        return self.fc(sample_repr)

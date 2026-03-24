from __future__ import annotations

import torch
import torch.nn as nn


class SharedValueEncoder(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        layers = []
        in_dim = 1
        for _ in range(max(1, num_layers - 1)):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.GELU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, d_model))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.unsqueeze(-1))


class FeatureTokenizer(nn.Module):
    def __init__(self, num_features: int, d_model: int, value_hidden_dim: int = 64, value_encoder_layers: int = 2, use_cls: bool = True):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model
        self.use_cls = use_cls
        self.value_encoder = SharedValueEncoder(d_model, hidden_dim=value_hidden_dim, num_layers=value_encoder_layers)
        self.feature_embedding = nn.Embedding(num_features, d_model)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_token, std=0.02)
        if use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None

    def forward(self, x: torch.Tensor, masked_positions: torch.Tensor | None = None) -> torch.Tensor:
        bsz, nfeat = x.shape
        device = x.device
        feat_ids = torch.arange(nfeat, device=device).unsqueeze(0).expand(bsz, -1)
        value_emb = self.value_encoder(x)
        feat_emb = self.feature_embedding(feat_ids)
        tokens = value_emb + feat_emb
        if masked_positions is not None:
            tokens = torch.where(masked_positions.unsqueeze(-1), self.mask_token.expand(bsz, nfeat, self.d_model), tokens)
        if self.use_cls:
            cls = self.cls_token.expand(bsz, 1, self.d_model)
            tokens = torch.cat([cls, tokens], dim=1)
        return tokens

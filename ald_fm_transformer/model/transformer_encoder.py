from __future__ import annotations

import torch
import torch.nn as nn

from .token_embeddings import FeatureTokenizer


class TabularTransformerEncoder(nn.Module):
    def __init__(
        self,
        num_features: int,
        d_model: int = 192,
        nhead: int = 6,
        num_layers: int = 4,
        dim_feedforward: int = 384,
        dropout: float = 0.1,
        activation: str = 'gelu',
        layer_norm_eps: float = 1e-5,
        use_cls: bool = True,
        value_hidden_dim: int = 64,
        value_encoder_layers: int = 2,
    ):
        super().__init__()
        self.tokenizer = FeatureTokenizer(
            num_features=num_features,
            d_model=d_model,
            value_hidden_dim=value_hidden_dim,
            value_encoder_layers=value_encoder_layers,
            use_cls=use_cls,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            layer_norm_eps=layer_norm_eps,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.use_cls = use_cls

    def forward(self, x: torch.Tensor, masked_positions: torch.Tensor | None = None) -> torch.Tensor:
        tokens = self.tokenizer(x, masked_positions=masked_positions)
        h = self.encoder(tokens)
        return self.norm(h)

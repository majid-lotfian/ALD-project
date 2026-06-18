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
    """
    token = value_emb(scalar) + feature_emb(feature_id) + group_emb(lipid_class_id)

    group_ids: LongTensor of shape (num_features,) mapping each feature to its lipid
    class group (0=VLCFA_LPC, 1=PC_PE, 2=PLASMALOGEN, 3=CERAMIDE, 4=SM, 5=TG_CE, 6=OTHER).
    If None, no group embedding is added.
    """

    def __init__(
        self,
        num_features: int,
        d_model: int,
        value_hidden_dim: int = 64,
        value_encoder_layers: int = 2,
        use_cls: bool = True,
        num_groups: int = 7,
        group_ids: torch.Tensor | None = None,
    ):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model
        self.use_cls = use_cls

        self.value_encoder = SharedValueEncoder(d_model, hidden_dim=value_hidden_dim, num_layers=value_encoder_layers)
        self.feature_embedding = nn.Embedding(num_features, d_model)
        self.group_embedding = nn.Embedding(num_groups, d_model) if num_groups > 0 else None

        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_token, std=0.02)

        if use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None

        if group_ids is not None:
            self.register_buffer('group_ids', group_ids.long())
        else:
            self.register_buffer('group_ids', None)

    def forward(self, x: torch.Tensor, masked_positions: torch.Tensor | None = None) -> torch.Tensor:
        bsz, nfeat = x.shape
        device = x.device

        feat_ids = torch.arange(nfeat, device=device).unsqueeze(0).expand(bsz, -1)
        value_emb = self.value_encoder(x)
        feat_emb = self.feature_embedding(feat_ids)
        tokens = value_emb + feat_emb

        if self.group_embedding is not None and self.group_ids is not None:
            grp_ids = self.group_ids.unsqueeze(0).expand(bsz, -1)
            tokens = tokens + self.group_embedding(grp_ids)

        if masked_positions is not None:
            mask_expand = masked_positions.unsqueeze(-1).expand_as(tokens)
            tokens = torch.where(mask_expand, self.mask_token.expand(bsz, nfeat, self.d_model), tokens)

        if self.use_cls:
            cls = self.cls_token.expand(bsz, 1, self.d_model)
            tokens = torch.cat([cls, tokens], dim=1)

        return tokens

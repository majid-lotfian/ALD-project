from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MaskingConfig:
    mask_ratio: float
    mask_token_prob: float = 0.80
    random_replace_prob: float = 0.10
    keep_original_prob: float = 0.10


def apply_bert_style_feature_mask(x: torch.Tensor, cfg: MaskingConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if abs(cfg.mask_token_prob + cfg.random_replace_prob + cfg.keep_original_prob - 1.0) > 1e-6:
        raise ValueError('Masking probabilities must sum to 1.')
    bsz, nfeat = x.shape
    device = x.device
    mask = torch.rand((bsz, nfeat), device=device) < cfg.mask_ratio
    empty_rows = mask.sum(dim=1) == 0
    if empty_rows.any():
        rand_col = torch.randint(0, nfeat, (empty_rows.sum(),), device=device)
        mask[empty_rows] = False
        mask[empty_rows, rand_col] = True

    x_corrupt = x.clone()
    masked = mask.nonzero(as_tuple=False)
    if masked.numel() == 0:
        targets = x.clone()
        return x_corrupt, mask, targets

    choices = torch.rand(masked.shape[0], device=device)
    mask_cut = cfg.mask_token_prob
    rand_cut = cfg.mask_token_prob + cfg.random_replace_prob

    random_vals = x[torch.randint(0, bsz, (masked.shape[0],), device=device), masked[:, 1]]

    use_mask_token = choices < mask_cut
    use_random = (choices >= mask_cut) & (choices < rand_cut)
    keep_original = choices >= rand_cut

    x_corrupt[masked[use_mask_token, 0], masked[use_mask_token, 1]] = 0.0
    x_corrupt[masked[use_random, 0], masked[use_random, 1]] = random_vals[use_random]
    x_corrupt[masked[keep_original, 0], masked[keep_original, 1]] = x[masked[keep_original, 0], masked[keep_original, 1]]
    targets = x.clone()
    return x_corrupt, mask, targets

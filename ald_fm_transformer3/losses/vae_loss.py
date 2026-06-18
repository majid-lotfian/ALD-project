from __future__ import annotations

import torch
import torch.nn.functional as F


def vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (total, recon_loss, kl_loss).

    recon_loss: MSE over all features (not just masked) — forces the VAE
    decoder to reconstruct the full lipid profile from the CLS latent.

    kl_loss: KL divergence KL(q(z|x) || p(z)) with p(z) = N(0,I).
    beta controls the trade-off (beta-VAE formulation).
    """
    recon_loss = F.mse_loss(recon, target, reduction='mean')
    kl_loss = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
    total = recon_loss + beta * kl_loss
    return total, recon_loss, kl_loss

from __future__ import annotations

import torch
import torch.nn as nn


class VAEBottleneck(nn.Module):
    """
    VAE on top of the [CLS] token for generative capability.

    Encoder: CLS repr (d_model) -> mu, logvar (latent_dim)
    Decoder: z (latent_dim) -> reconstructed feature vector (num_features)

    Use forward() during training to get (recon, mu, logvar).
    Use generate() at inference to sample unconditionally.
    Use encode_decode() to impute a full profile from a masked input.
    """

    def __init__(self, d_model: int, latent_dim: int, num_features: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_features = num_features

        self.encoder_mu = nn.Linear(d_model, latent_dim)
        self.encoder_logvar = nn.Linear(d_model, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_features),
        )

    def encode(self, cls_repr: torch.Tensor):
        mu = self.encoder_mu(cls_repr)
        logvar = self.encoder_logvar(cls_repr).clamp(-10.0, 4.0)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, cls_repr: torch.Tensor):
        mu, logvar = self.encode(cls_repr)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    @torch.no_grad()
    def generate(self, n_samples: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(n_samples, self.latent_dim, device=device)
        return self.decode(z)

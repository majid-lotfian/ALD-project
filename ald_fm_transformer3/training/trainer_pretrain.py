from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.distributed as dist
from torch.amp import GradScaler, autocast

from evaluation.metrics_pretrain import pretrain_metrics
from losses.masked_regression import masked_mse_loss
from losses.vae_loss import vae_loss
from model.masking import MaskingConfig, apply_bert_style_feature_mask
from training.checkpointing import save_checkpoint
from utils.ddp import all_reduce_mean, is_main
from utils.io import ensure_dir, save_json
from utils.logging_utils import CSVLogger


@dataclass
class PretrainTrainerConfig:
    epochs: int
    amp: bool
    amp_dtype: str
    grad_accum_steps: int
    clip_grad_norm: float
    run_dir: str
    warmup_steps: int = 2000
    vae_lambda_recon: float = 0.5
    vae_beta_kl: float = 1e-3
    kl_beta_start: float = 0.0
    kl_anneal_steps: int = 0
    steps_per_epoch: Optional[int] = None
    val_steps: Optional[int] = None


class PretrainTrainer:
    def __init__(
        self,
        model,
        mask_head,
        vae,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        masking_cfg: MaskingConfig,
        cfg: PretrainTrainerConfig,
        feature_cols,
        norm_stats,
        group_ids,
        config_snapshot: Dict,
        logger,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.model = model
        self.mask_head = mask_head
        self.vae = vae
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.masking_cfg = masking_cfg
        self.cfg = cfg
        self.logger = logger
        self.feature_cols = feature_cols
        self.norm_stats = norm_stats
        self.group_ids = group_ids
        self.config_snapshot = config_snapshot
        self.rank = rank
        self.world_size = world_size
        self.is_main = is_main(rank)
        self.global_step = 0

        if self.is_main:
            self.csv_logger = CSVLogger(
                Path(cfg.run_dir) / 'train_log.csv',
                ['epoch', 'split', 'loss', 'masked_mse', 'recon_loss', 'kl_loss', 'lr'],
            )
            ensure_dir(Path(cfg.run_dir) / 'checkpoints')

        amp_dtype_str = cfg.amp_dtype
        self.scaler = GradScaler('cuda', enabled=cfg.amp and amp_dtype_str == 'fp16')

    def _get_kl_beta(self) -> float:
        if self.cfg.kl_anneal_steps <= 0:
            return self.cfg.vae_beta_kl
        t = min(self.global_step / self.cfg.kl_anneal_steps, 1.0)
        return self.cfg.kl_beta_start + (self.cfg.vae_beta_kl - self.cfg.kl_beta_start) * t

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def _raw_model(self):
        return self.model.module if hasattr(self.model, 'module') else self.model

    def _forward_loss(self, batch, train: bool) -> Dict[str, float]:
        x = batch['x'].to(self.device, non_blocking=True)
        x_corrupt, mask, targets = apply_bert_style_feature_mask(x, self.masking_cfg)

        amp_dtype = torch.bfloat16 if self.cfg.amp_dtype == 'bf16' else torch.float16
        autocast_enabled = self.cfg.amp and self.device.type == 'cuda'

        with autocast('cuda', enabled=autocast_enabled, dtype=amp_dtype):
            h = self.model(x_corrupt, masked_positions=mask)

            # Masked feature prediction loss
            token_reps = h[:, 1:, :] if self._raw_model.use_cls else h
            pred = self.mask_head(token_reps)
            loss_mlm = masked_mse_loss(pred, targets, mask)

            # VAE reconstruction + KL loss on [CLS]
            cls_repr = h[:, 0, :]
            recon, mu, logvar = self.vae(cls_repr)
            kl_beta = self._get_kl_beta()
            _, recon_loss, kl_loss = vae_loss(
                recon, x,
                mu, logvar,
                beta=kl_beta,
            )
            loss_vae = recon_loss + kl_beta * kl_loss

            total_loss = loss_mlm + self.cfg.vae_lambda_recon * loss_vae

        if train:
            scaled = total_loss / self.cfg.grad_accum_steps
            if self.scaler.is_enabled():
                self.scaler.scale(scaled).backward()
            else:
                scaled.backward()

        metrics = pretrain_metrics(pred.detach(), targets.detach(), mask.detach())
        metrics['loss'] = float(total_loss.detach())
        metrics['recon_loss'] = float(recon_loss.detach())
        metrics['kl_loss'] = float(kl_loss.detach())
        return metrics

    def _optimizer_step(self):
        if self.scaler.is_enabled():
            self.scaler.unscale_(self.optimizer)

        if self.cfg.clip_grad_norm > 0:
            params = (
                list(self.model.parameters())
                + list(self.mask_head.parameters())
                + list(self.vae.parameters())
            )
            torch.nn.utils.clip_grad_norm_(params, self.cfg.clip_grad_norm)

        if self.scaler.is_enabled():
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.optimizer.zero_grad(set_to_none=True)
        if self.scheduler is not None:
            self.scheduler.step()
        self.global_step += 1

    def _run_train_epoch(self) -> Dict[str, float]:
        self.model.train()
        self.mask_head.train()
        self.vae.train()
        self.optimizer.zero_grad(set_to_none=True)

        sums = {'loss': 0.0, 'masked_mse': 0.0, 'recon_loss': 0.0, 'kl_loss': 0.0}
        n = 0
        accum_count = 0

        for step, batch in enumerate(self.train_loader, start=1):
            metrics = self._forward_loss(batch, train=True)
            for k in sums:
                sums[k] += metrics.get(k, 0.0)
            n += 1
            accum_count += 1

            if accum_count == self.cfg.grad_accum_steps:
                self._optimizer_step()
                accum_count = 0

            if self.cfg.steps_per_epoch is not None and step >= self.cfg.steps_per_epoch:
                break

        if accum_count > 0:
            self._optimizer_step()

        return {k: v / max(n, 1) for k, v in sums.items()}

    def _run_val_epoch(self) -> Dict[str, float]:
        self.model.eval()
        self.mask_head.eval()
        self.vae.eval()

        sums = {'loss': 0.0, 'masked_mse': 0.0, 'recon_loss': 0.0, 'kl_loss': 0.0}
        n = 0

        with torch.no_grad():
            for step, batch in enumerate(self.val_loader, start=1):
                metrics = self._forward_loss(batch, train=False)
                for k in sums:
                    sums[k] += metrics.get(k, 0.0)
                n += 1
                if self.cfg.val_steps is not None and step >= self.cfg.val_steps:
                    break

        return {k: v / max(n, 1) for k, v in sums.items()}

    def _avg_across_ranks(self, metrics: Dict[str, float]) -> Dict[str, float]:
        if self.world_size == 1:
            return metrics
        t = torch.tensor([metrics[k] for k in sorted(metrics)], dtype=torch.float32, device=self.device)
        all_reduce_mean(t, self.world_size)
        return {k: float(t[i]) for i, k in enumerate(sorted(metrics))}

    def fit(self):
        if self.is_main:
            save_json(self.config_snapshot, Path(self.cfg.run_dir) / 'config_used.json')

        best_val = float('inf')
        best_path = Path(self.cfg.run_dir) / 'checkpoints' / 'best.pt'
        last_path = Path(self.cfg.run_dir) / 'checkpoints' / 'last.pt'

        for epoch in range(1, self.cfg.epochs + 1):
            # Barrier so all ranks start epoch together
            if self.world_size > 1:
                dist.barrier()

            train_m = self._run_train_epoch()
            train_m = self._avg_across_ranks(train_m)

            if self.world_size > 1:
                dist.barrier()

            val_m = self._run_val_epoch()
            val_m = self._avg_across_ranks(val_m)

            current_lr = self.optimizer.param_groups[0]['lr']

            if self.is_main:
                self.csv_logger.log({
                    'epoch': epoch, 'split': 'train',
                    'loss': train_m['loss'], 'masked_mse': train_m['masked_mse'],
                    'recon_loss': train_m['recon_loss'], 'kl_loss': train_m['kl_loss'],
                    'lr': current_lr,
                })
                self.csv_logger.log({
                    'epoch': epoch, 'split': 'val',
                    'loss': val_m['loss'], 'masked_mse': val_m['masked_mse'],
                    'recon_loss': val_m['recon_loss'], 'kl_loss': val_m['kl_loss'],
                    'lr': current_lr,
                })
                self.logger.info(
                    f"Epoch {epoch:03d} | step={self.global_step} | lr={current_lr:.2e} | "
                    f"train_loss={train_m['loss']:.4f}  val_loss={val_m['loss']:.4f}  "
                    f"mlm={val_m['masked_mse']:.4f}  recon={val_m['recon_loss']:.4f}  "
                    f"kl={val_m['kl_loss']:.6f}"
                )

                raw_model = self._raw_model
                raw_head = self.mask_head.module if hasattr(self.mask_head, 'module') else self.mask_head
                raw_vae = self.vae.module if hasattr(self.vae, 'module') else self.vae

                state = {
                    'epoch': epoch,
                    'global_step': self.global_step,
                    'model_state': raw_model.state_dict(),
                    'mask_head_state': raw_head.state_dict(),
                    'vae_state': raw_vae.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'scheduler_state': self.scheduler.state_dict() if self.scheduler else None,
                    'feature_cols': self.feature_cols,
                    'group_ids': self.group_ids.tolist() if self.group_ids is not None else None,
                    'mean': self.norm_stats.mean,
                    'std': self.norm_stats.std,
                    'config': self.config_snapshot,
                }
                save_checkpoint(state, last_path)
                if val_m['loss'] < best_val:
                    best_val = val_m['loss']
                    save_checkpoint(state, best_path)

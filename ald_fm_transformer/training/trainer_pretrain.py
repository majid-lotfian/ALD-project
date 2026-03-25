from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.amp import GradScaler, autocast

from evaluation.metrics_pretrain import pretrain_metrics
from losses.masked_regression import masked_mse_loss
from model.masking import MaskingConfig, apply_bert_style_feature_mask
from training.checkpointing import save_checkpoint
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
    steps_per_epoch: Optional[int] = None
    val_steps: Optional[int] = None


class PretrainTrainer:
    def __init__(
        self,
        model,
        head,
        optimizer,
        train_loader,
        val_loader,
        masking_cfg: MaskingConfig,
        cfg: PretrainTrainerConfig,
        feature_cols,
        norm_stats,
        config_snapshot: Dict,
        logger,
    ):
        self.model = model
        self.head = head
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.masking_cfg = masking_cfg
        self.cfg = cfg
        self.logger = logger
        self.feature_cols = feature_cols
        self.norm_stats = norm_stats
        self.config_snapshot = config_snapshot
        self.csv_logger = CSVLogger(
            Path(cfg.run_dir) / 'train_log.csv',
            ['epoch', 'split', 'loss', 'masked_mse'],
        )
        self.scaler = GradScaler('cuda', enabled=cfg.amp and cfg.amp_dtype == 'fp16')
        ensure_dir(Path(cfg.run_dir) / 'checkpoints')

    @property
    def device(self):
        return next(self.model.parameters()).device

    def _forward_loss(self, batch, train: bool) -> Dict[str, float]:
        x = batch['x'].to(self.device, non_blocking=True)
        x_corrupt, mask, targets = apply_bert_style_feature_mask(x, self.masking_cfg)

        amp_dtype = torch.bfloat16 if self.cfg.amp_dtype == 'bf16' else torch.float16
        autocast_enabled = self.cfg.amp and self.device.type == 'cuda'

        with autocast('cuda', enabled=autocast_enabled, dtype=amp_dtype):
            h = self.model(x_corrupt, masked_positions=mask)
            token_reps = h[:, 1:, :] if self.model.use_cls else h
            pred = self.head(token_reps)
            loss = masked_mse_loss(pred, targets, mask)

        if train:
            loss_for_backward = loss / self.cfg.grad_accum_steps
            if self.scaler.is_enabled():
                self.scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()

        metrics = pretrain_metrics(pred.detach(), targets.detach(), mask.detach())
        metrics['loss'] = float(loss.detach().cpu())
        return metrics

    def _optimizer_step(self):
        if self.scaler.is_enabled():
            self.scaler.unscale_(self.optimizer)

        if self.cfg.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.head.parameters()),
                self.cfg.clip_grad_norm,
            )

        if self.scaler.is_enabled():
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.optimizer.zero_grad(set_to_none=True)

    def _run_train_epoch(self) -> Dict[str, float]:
        self.model.train()
        self.head.train()
        self.optimizer.zero_grad(set_to_none=True)

        sums = {'loss': 0.0, 'masked_mse': 0.0}
        n = 0
        accum_count = 0

        for step, batch in enumerate(self.train_loader, start=1):
            metrics = self._forward_loss(batch, train=True)
            sums['loss'] += metrics['loss']
            sums['masked_mse'] += metrics['masked_mse']
            n += 1
            accum_count += 1

            if accum_count == self.cfg.grad_accum_steps:
                self._optimizer_step()
                accum_count = 0

            if self.cfg.steps_per_epoch is not None and step >= self.cfg.steps_per_epoch:
                break

        if accum_count > 0:
            self._optimizer_step()

        return {
            'loss': sums['loss'] / max(n, 1),
            'masked_mse': sums['masked_mse'] / max(n, 1),
        }

    def _run_val_epoch(self) -> Dict[str, float]:
        self.model.eval()
        self.head.eval()

        sums = {'loss': 0.0, 'masked_mse': 0.0}
        n = 0

        with torch.no_grad():
            for step, batch in enumerate(self.val_loader, start=1):
                metrics = self._forward_loss(batch, train=False)
                sums['loss'] += metrics['loss']
                sums['masked_mse'] += metrics['masked_mse']
                n += 1

                if self.cfg.val_steps is not None and step >= self.cfg.val_steps:
                    break

        return {
            'loss': sums['loss'] / max(n, 1),
            'masked_mse': sums['masked_mse'] / max(n, 1),
        }

    def fit(self):
        best_val = float('inf')
        best_path = Path(self.cfg.run_dir) / 'checkpoints' / 'best.pt'
        last_path = Path(self.cfg.run_dir) / 'checkpoints' / 'last.pt'

        save_json(self.config_snapshot, Path(self.cfg.run_dir) / 'config_used.json')

        for epoch in range(1, self.cfg.epochs + 1):
            train_metrics = self._run_train_epoch()
            self.csv_logger.log({
                'epoch': epoch,
                'split': 'train',
                'loss': train_metrics['loss'],
                'masked_mse': train_metrics['masked_mse'],
            })

            val_metrics = self._run_val_epoch()
            self.csv_logger.log({
                'epoch': epoch,
                'split': 'val',
                'loss': val_metrics['loss'],
                'masked_mse': val_metrics['masked_mse'],
            })

            self.logger.info(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_metrics['loss']:.6f} "
                f"val_loss={val_metrics['loss']:.6f} "
                f"train_masked_mse={train_metrics['masked_mse']:.6f} "
                f"val_masked_mse={val_metrics['masked_mse']:.6f}"
            )

            state = {
                'epoch': epoch,
                'model_state': self.model.state_dict(),
                'head_state': self.head.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'feature_cols': self.feature_cols,
                'mean': self.norm_stats.mean,
                'std': self.norm_stats.std,
                'config': self.config_snapshot,
            }
            save_checkpoint(state, last_path)
            if val_metrics['loss'] < best_val:
                best_val = val_metrics['loss']
                save_checkpoint(state, best_path)
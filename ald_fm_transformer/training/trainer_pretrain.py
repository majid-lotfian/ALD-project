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


class PretrainTrainer:
    def __init__(self, model, head, optimizer, train_loader, val_loader, masking_cfg: MaskingConfig, cfg: PretrainTrainerConfig, feature_cols, norm_stats, config_snapshot: Dict, logger):
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
        self.csv_logger = CSVLogger(Path(cfg.run_dir) / 'train_log.csv', ['epoch', 'split', 'loss', 'masked_mse'])
        self.scaler = GradScaler(enabled=cfg.amp and cfg.amp_dtype == 'fp16')
        ensure_dir(Path(cfg.run_dir) / 'checkpoints')

    def _step(self, batch, train: bool) -> Dict[str, float]:
        x = batch['x'].to(next(self.model.parameters()).device)
        x_corrupt, mask, targets = apply_bert_style_feature_mask(x, self.masking_cfg)
        amp_dtype = torch.bfloat16 if self.cfg.amp_dtype == 'bf16' else torch.float16
        with autocast('cuda', enabled=self.cfg.amp, dtype=amp_dtype):
            h = self.model(x_corrupt, masked_positions=mask)
            token_reps = h[:, 1:, :] if self.model.use_cls else h
            pred = self.head(token_reps)
            loss = masked_mse_loss(pred, targets, mask)
        if train:
            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
        metrics = pretrain_metrics(pred.detach(), targets.detach(), mask.detach())
        metrics['loss'] = float(loss.detach().cpu())
        return metrics

    def fit(self):
        best_val = float('inf')
        best_path = Path(self.cfg.run_dir) / 'checkpoints' / 'best.pt'
        last_path = Path(self.cfg.run_dir) / 'checkpoints' / 'last.pt'

        save_json(self.config_snapshot, Path(self.cfg.run_dir) / 'config_used.json')
        for epoch in range(1, self.cfg.epochs + 1):
            self.model.train(); self.head.train()
            self.optimizer.zero_grad(set_to_none=True)
            train_sums = {'loss': 0.0, 'masked_mse': 0.0}; train_n = 0
            for step, batch in enumerate(self.train_loader, start=1):
                metrics = self._step(batch, train=True)
                train_sums['loss'] += metrics['loss']; train_sums['masked_mse'] += metrics['masked_mse']; train_n += 1
                if step % self.cfg.grad_accum_steps == 0:
                    if self.scaler.is_enabled():
                        self.scaler.unscale_(self.optimizer)
                    if self.cfg.clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(list(self.model.parameters()) + list(self.head.parameters()), self.cfg.clip_grad_norm)
                    if self.scaler.is_enabled():
                        self.scaler.step(self.optimizer); self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
            train_loss = train_sums['loss'] / max(train_n, 1)
            train_mse = train_sums['masked_mse'] / max(train_n, 1)
            self.csv_logger.log({'epoch': epoch, 'split': 'train', 'loss': train_loss, 'masked_mse': train_mse})

            self.model.eval(); self.head.eval()
            val_sums = {'loss': 0.0, 'masked_mse': 0.0}; val_n = 0
            with torch.no_grad():
                for batch in self.val_loader:
                    metrics = self._step(batch, train=False)
                    val_sums['loss'] += metrics['loss']; val_sums['masked_mse'] += metrics['masked_mse']; val_n += 1
            val_loss = val_sums['loss'] / max(val_n, 1)
            val_mse = val_sums['masked_mse'] / max(val_n, 1)
            self.csv_logger.log({'epoch': epoch, 'split': 'val', 'loss': val_loss, 'masked_mse': val_mse})
            self.logger.info(f'Epoch {epoch:03d} | train_loss={train_loss:.6f} val_loss={val_loss:.6f} train_masked_mse={train_mse:.6f} val_masked_mse={val_mse:.6f}')

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
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(state, best_path)

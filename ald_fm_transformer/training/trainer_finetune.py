from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.cuda.amp import GradScaler, autocast

from evaluation.metrics_finetune import compute_classification_metrics
from training.checkpointing import save_checkpoint
from utils.io import save_json
from utils.logging_utils import CSVLogger


@dataclass
class FinetuneTrainerConfig:
    epochs: int
    patience: int
    amp: bool
    amp_dtype: str
    grad_accum_steps: int
    clip_grad_norm: float
    run_dir: str


class FinetuneTrainer:
    def __init__(self, model, head, optimizer, criterion, train_loader, val_loader, cfg: FinetuneTrainerConfig, logger, config_snapshot: Dict):
        self.model = model
        self.head = head
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.logger = logger
        self.scaler = GradScaler(enabled=cfg.amp and cfg.amp_dtype == 'fp16')
        self.csv_logger = CSVLogger(Path(cfg.run_dir) / 'train_log.csv', ['epoch', 'split', 'loss', 'accuracy', 'macro_f1', 'balanced_accuracy'])
        save_json(config_snapshot, Path(cfg.run_dir) / 'config_used.json')

    def _forward_batch(self, batch, train: bool):
        x = batch['x'].to(next(self.model.parameters()).device)
        y = batch['y'].to(next(self.model.parameters()).device)
        amp_dtype = torch.bfloat16 if self.cfg.amp_dtype == 'bf16' else torch.float16
        with autocast('cuda', enabled=self.cfg.amp, dtype=amp_dtype):
            h = self.model(x, masked_positions=None)
            sample_repr = h[:, 0, :] if self.model.use_cls else h.mean(dim=1)
            logits = self.head(sample_repr)
            loss = self.criterion(logits, y)
        if train:
            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
        return loss, logits, y

    def fit(self):
        best_score = -1.0
        bad = 0
        best_path = Path(self.cfg.run_dir) / 'checkpoints' / 'best_macro_f1.pt'
        last_path = Path(self.cfg.run_dir) / 'checkpoints' / 'last.pt'
        for epoch in range(1, self.cfg.epochs + 1):
            self.model.train(); self.head.train()
            self.optimizer.zero_grad(set_to_none=True)
            train_losses = []
            for step, batch in enumerate(self.train_loader, start=1):
                loss, _, _ = self._forward_batch(batch, train=True)
                train_losses.append(float(loss.detach().cpu()))
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
            train_loss = sum(train_losses) / max(len(train_losses), 1)
            self.csv_logger.log({'epoch': epoch, 'split': 'train', 'loss': train_loss, 'accuracy': '', 'macro_f1': '', 'balanced_accuracy': ''})

            self.model.eval(); self.head.eval()
            val_losses = []
            all_logits = []; all_y = []
            with torch.no_grad():
                for batch in self.val_loader:
                    loss, logits, y = self._forward_batch(batch, train=False)
                    val_losses.append(float(loss.detach().cpu()))
                    all_logits.append(logits.detach().cpu())
                    all_y.append(y.detach().cpu())
            y_true = torch.cat(all_y).numpy()
            y_pred = torch.cat(all_logits).argmax(dim=1).numpy()
            metrics = compute_classification_metrics(y_true, y_pred)
            val_loss = sum(val_losses) / max(len(val_losses), 1)
            self.csv_logger.log({'epoch': epoch, 'split': 'val', 'loss': val_loss, **metrics})
            self.logger.info(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} val_loss={val_loss:.6f} acc={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} bal_acc={metrics['balanced_accuracy']:.4f}")
            state = {'epoch': epoch, 'model_state': self.model.state_dict(), 'head_state': self.head.state_dict(), 'optimizer_state': self.optimizer.state_dict(), 'metrics': metrics}
            save_checkpoint(state, last_path)
            if metrics['macro_f1'] > best_score:
                best_score = metrics['macro_f1']
                bad = 0
                save_checkpoint(state, best_path)
            else:
                bad += 1
                if bad >= self.cfg.patience:
                    break

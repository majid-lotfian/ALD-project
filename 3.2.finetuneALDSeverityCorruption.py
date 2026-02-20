#!/usr/bin/env python3
import os
import argparse
import json
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix


# -------------------------
# Model definitions (must match pretrain)
# -------------------------

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm(x)
        h = F.gelu(self.fc1(h))
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.dropout(h)
        return x + h


class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, depth: int, z_dim: int, dropout: float):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim, dropout) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.z_proj = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        h = F.gelu(self.in_proj(x))
        for blk in self.blocks:
            h = blk(h)
        h = self.out_norm(h)
        return self.z_proj(h)


class ClassifierHead(nn.Module):
    def __init__(self, z_dim: int, n_classes: int = 6, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(z_dim),
            nn.Dropout(dropout),
            nn.Linear(z_dim, n_classes),
        )

    def forward(self, z):
        return self.net(z)


# -------------------------
# Utils
# -------------------------
def apply_feature_mask_ft(x: torch.Tensor, mask_ratio: float, mask_value: float = 0.0):
    # x: (B, D) where D = D_omics+1 (sex last)
    B, D = x.shape
    D_omics = D - 1

    mask_omics = (torch.rand((B, D_omics), device=x.device) < mask_ratio)
    # ensure at least one masked per row when mask_ratio>0
    if mask_ratio > 0:
        rows_with_none = ~mask_omics.any(dim=1)
        if rows_with_none.any():
            idx = torch.randint(0, D_omics, (rows_with_none.sum().item(),), device=x.device)
            mask_omics[rows_with_none, idx] = True

    mask = torch.zeros((B, D), dtype=torch.bool, device=x.device)
    mask[:, :D_omics] = mask_omics

    x_masked = x.clone()
    x_masked[mask] = mask_value
    return x_masked, mask

def apply_denoise_corruption_ft(x: torch.Tensor, noise_std: float, dropout_p: float, drop_value: float = 0.0):
    # x: (B, D), sex last untouched by dropout (but can still get noise if you want; we keep it clean)
    B, D = x.shape
    D_omics = D - 1

    x_cor = x.clone()
    drop_mask = torch.zeros((B, D), dtype=torch.bool, device=x.device)

    if noise_std > 0:
        noise = torch.randn((B, D_omics), device=x.device) * noise_std
        x_cor[:, :D_omics] = x_cor[:, :D_omics] + noise

    if dropout_p > 0:
        dm = (torch.rand((B, D_omics), device=x.device) < dropout_p)
        x_cor[:, :D_omics][dm] = drop_value
        drop_mask[:, :D_omics] = dm

    return x_cor, drop_mask


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_fold_stats(X_omics: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = X_omics.mean(axis=0, dtype=np.float64)
    std = X_omics.std(axis=0, ddof=1, dtype=np.float64)
    std = np.maximum(std, 1e-6)
    return mean.astype(np.float32), std.astype(np.float32)

def make_class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    w = 1.0 / counts
    w = w * (n_classes / w.sum())
    return torch.tensor(w, dtype=torch.float32)

@dataclass
class TrainConfig:
    epochs: int
    batch_size: int
    lr_head: float
    lr_enc: float
    weight_decay: float
    patience: int
    freeze_mode: str  # "linear" | "partial" | "full"
    unfreeze_last_blocks: int
    amp: bool
    amp_dtype: str  # "bf16" | "fp16"
    unfreeze_in_proj: bool  # IMPORTANT for your pretrain->finetune shift

def set_freeze_mode(encoder: MLPEncoder, cfg: TrainConfig, depth: int):
    # freeze all
    for p in encoder.parameters():
        p.requires_grad = False

    if cfg.freeze_mode == "linear":
        return

    if cfg.freeze_mode == "full":
        for p in encoder.parameters():
            p.requires_grad = True
        return

    if cfg.freeze_mode == "partial":
        # unfreeze last K residual blocks + out_norm + z_proj
        k = max(0, min(cfg.unfreeze_last_blocks, depth))
        for i in range(depth - k, depth):
            for p in encoder.blocks[i].parameters():
                p.requires_grad = True

        for p in encoder.out_norm.parameters():
            p.requires_grad = True
        for p in encoder.z_proj.parameters():
            p.requires_grad = True

        # KEY FIX: allow adapting to finetune input distribution (indicators are all-zero)
        if cfg.unfreeze_in_proj:
            for p in encoder.in_proj.parameters():
                p.requires_grad = True

        return

    raise ValueError(f"Unknown freeze_mode: {cfg.freeze_mode}")

def batch_iter(X: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool = True):
    n = X.shape[0]
    idx = torch.randperm(n, device=X.device) if shuffle else torch.arange(n, device=X.device)
    for i in range(0, n, batch_size):
        j = idx[i:i+batch_size]
        yield X[j], y[j]


# -------------------------
# Args
# -------------------------

def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--real_csv", type=str, default="./real_data.csv",
                    help="Real dataset CSV.")
    ap.add_argument("--sex_col", type=str, default="sex")
    ap.add_argument("--severity_col", type=str, default="severity")
    ap.add_argument("--id_col", type=str, default="Sample_ID")

    ap.add_argument("--ckpt_path", type=str, required=True,
                    help="Path to pretrained checkpoint ckpt_final_step_*.pt")

    ap.add_argument("--norm_mode", type=str, default="pretrain", choices=["pretrain", "fold"],
                    help="pretrain: use ckpt mean/std for omics; fold: compute mean/std on train fold")

    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)

    ap.add_argument("--freeze_mode", type=str, default="partial", choices=["linear", "partial", "full"])
    ap.add_argument("--unfreeze_last_blocks", type=int, default=2)

    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lr_enc", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-2)

    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16"])

    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", type=str, default="./finetune_runs/severity_cv")

    ap.add_argument("--from_scratch", action="store_true",
                    help="If set, do not load pretrained encoder weights.")

    ap.add_argument("--unfreeze_in_proj", action="store_true",
                    help="If set, allow in_proj to adapt (recommended for your current finetune input).")
    ap.add_argument("--ft_use_corrupt_train", action="store_true",
                    help="If set, apply pretrain-like corruptions during finetune training and pass indicators to encoder.")
    ap.add_argument("--ft_mask_ratio", type=float, default=0.2)
    ap.add_argument("--ft_noise_std", type=float, default=0.02)
    ap.add_argument("--ft_feat_dropout", type=float, default=0.02)

    return ap.parse_args()


# -------------------------
# Main
# -------------------------

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device(args.device)

    # ---- Load checkpoint
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    feature_cols = list(ckpt["feature_cols"])  # omics only (sex excluded)
    mean_pre = np.array(ckpt["mean"], dtype=np.float32)  # (D_omics,)
    std_pre  = np.array(ckpt["std"], dtype=np.float32)   # (D_omics,)

    # ---- Load real data
    df = pd.read_csv(args.real_csv)
    for c in [args.severity_col, args.sex_col, args.id_col]:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in {args.real_csv}")

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Real CSV is missing {len(missing)} feature_cols from checkpoint. Example: {missing[:10]}")

    # y
    y = df[args.severity_col].astype(int).to_numpy()
    n_classes = int(y.max() + 1)
    if n_classes < 2:
        raise ValueError("severity has <2 classes in real data.")
    if n_classes > 6:
        print(f"[WARN] severity has {n_classes} classes; expected <= 6. Proceeding anyway.", flush=True)

    # X: omics + sex(last)
    X_omics = df[feature_cols].astype(np.float32).to_numpy()            # (N, D_omics)
    sex = df[args.sex_col].astype(np.float32).to_numpy().reshape(-1, 1) # (N, 1)
    D_omics = X_omics.shape[1]

    # ---- Encoder dims from checkpoint args
    pre_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    hidden_dim = int(pre_args.get("hidden_dim", 1024))
    depth = int(pre_args.get("depth", 6))
    z_dim = int(pre_args.get("z_dim", 256))
    dropout = float(pre_args.get("dropout", 0.1))

    # Pretrain encoder input: 2*(D_omics + sex)
    in_dim_raw = D_omics + 1
    encoder_in_dim = 2 * in_dim_raw

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_head=args.lr_head,
        lr_enc=args.lr_enc,
        weight_decay=args.weight_decay,
        patience=args.patience,
        freeze_mode=args.freeze_mode,
        unfreeze_last_blocks=args.unfreeze_last_blocks,
        amp=args.amp and device.type == "cuda",
        amp_dtype=args.amp_dtype,
        unfreeze_in_proj=args.unfreeze_in_proj,
    )

    # AMP helpers
    from torch.amp import autocast, GradScaler
    scaler = GradScaler("cuda", enabled=(cfg.amp and cfg.amp_dtype == "fp16"))
    amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bf16" else torch.float16

    # --- Make splits safe with rare classes
    class_counts = np.bincount(y, minlength=n_classes)
    min_count = int(class_counts.min()) if len(class_counts) else 0
    if min_count < args.splits:
        new_splits = max(2, min(args.splits, min_count))
        print(f"[WARN] min class count={min_count} < n_splits={args.splits} -> using n_splits={new_splits}", flush=True)
        n_splits = new_splits
    else:
        n_splits = args.splits

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)

    fold_metrics: List[Dict] = []
    all_cm = np.zeros((n_classes, n_classes), dtype=np.int64)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_omics, y), start=1):
        # --- Fresh models per fold
        encoder = MLPEncoder(
            in_dim=encoder_in_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            z_dim=z_dim,
            dropout=dropout,
        )
        head = ClassifierHead(z_dim=z_dim, n_classes=n_classes, dropout=0.1)

        if not args.from_scratch:
            encoder.load_state_dict(ckpt["encoder"])
            print(f"[Fold {fold}] Loaded pretrained encoder weights.", flush=True)
        else:
            print(f"[Fold {fold}] Using randomly initialized encoder (from scratch).", flush=True)

        encoder.to(device)
        head.to(device)

        set_freeze_mode(encoder, cfg, depth)

        # Optimizer with differential LR
        params = [{"params": head.parameters(), "lr": cfg.lr_head}]
        enc_params = [p for p in encoder.parameters() if p.requires_grad]
        if enc_params:
            params.append({"params": enc_params, "lr": cfg.lr_enc})
        opt = torch.optim.AdamW(params, weight_decay=cfg.weight_decay)

        # Norm stats
        if args.norm_mode == "pretrain":
            mean, std = mean_pre, std_pre
        else:
            mean, std = compute_fold_stats(X_omics[tr_idx])

        # Normalize omics only; append raw sex as last dim
        Xtr = (X_omics[tr_idx] - mean) / std
        Xva = (X_omics[va_idx] - mean) / std
        Xtr = np.concatenate([Xtr, sex[tr_idx]], axis=1).astype(np.float32)  # (Ntr, D_omics+1)
        Xva = np.concatenate([Xva, sex[va_idx]], axis=1).astype(np.float32)

        ytr = y[tr_idx].astype(np.int64)
        yva = y[va_idx].astype(np.int64)

        Xtr_t = torch.from_numpy(Xtr).to(device)
        Xva_t = torch.from_numpy(Xva).to(device)
        ytr_t = torch.from_numpy(ytr).to(device)
        yva_t = torch.from_numpy(yva).to(device)

        # Build indicator channels (no corruption -> zeros)
        #zeros_tr = torch.zeros_like(Xtr_t)
        if args.ft_use_corrupt_train:
            # masked view (indicator = mask)
            x_masked, mask = apply_feature_mask_ft(Xtr_t, mask_ratio=args.ft_mask_ratio, mask_value=0.0)
            Xtr_in = torch.cat([x_masked, mask.float()], dim=1)

            # OPTIONAL alternative: noisy/drop view (indicator = drop_mask)
            # x_noisy, drop_mask = apply_denoise_corruption_ft(Xtr_t, noise_std=args.ft_noise_std,
            #                                                  dropout_p=args.ft_feat_dropout, drop_value=0.0)
            # Xtr_in = torch.cat([x_noisy, drop_mask.float()], dim=1)
        else:
            zeros_tr = torch.zeros_like(Xtr_t)
            Xtr_in = torch.cat([Xtr_t, zeros_tr], dim=1)
        zeros_va = torch.zeros_like(Xva_t)
        #Xtr_in = torch.cat([Xtr_t, zeros_tr], dim=1)  # (Ntr, 2*(D_omics+1))
        Xva_in = torch.cat([Xva_t, zeros_va], dim=1)

        class_w = make_class_weights(ytr, n_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_w)

        best_val = -1.0
        best_state = None
        bad = 0

        for epoch in range(1, cfg.epochs + 1):
            encoder.train()
            head.train()

            for xb, yb in batch_iter(Xtr_in, ytr_t, cfg.batch_size, shuffle=True):
                opt.zero_grad(set_to_none=True)

                with autocast("cuda", dtype=amp_dtype, enabled=cfg.amp):
                    z = encoder(xb)
                    logits = head(z)
                    loss = criterion(logits, yb)

                if cfg.amp and cfg.amp_dtype == "fp16":
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()

            # Validate
            encoder.eval()
            head.eval()
            with torch.no_grad():
                z = encoder(Xva_in)
                logits = head(z)
                pred = torch.argmax(logits, dim=1).detach().cpu().numpy()

            #bal = balanced_accuracy_score(yva, pred)
            labels = list(range(n_classes))

            cm_tmp = confusion_matrix(yva, pred, labels=labels)

            recalls = np.divide(
                np.diag(cm_tmp),
                cm_tmp.sum(axis=1),
                out=np.zeros(n_classes, dtype=np.float64),
                where=(cm_tmp.sum(axis=1) != 0),
            )

            bal = float(recalls.mean())

            f1m = f1_score(
                yva, pred,
                average="macro",
                labels=list(range(n_classes)),
                zero_division=0
            )

            if bal > best_val + 1e-6:
                best_val = bal
                bad = 0
                best_state = {
                    "encoder": {k: v.detach().cpu() for k, v in encoder.state_dict().items()},
                    "head": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                }
            else:
                bad += 1
                if bad >= cfg.patience:
                    break

        # Evaluate best
        encoder.load_state_dict(best_state["encoder"])
        head.load_state_dict(best_state["head"])
        encoder.to(device)
        head.to(device)
        encoder.eval()
        head.eval()

        with torch.no_grad():
            z = encoder(Xva_in)
            logits = head(z)
            pred = torch.argmax(logits, dim=1).cpu().numpy()

        #bal = balanced_accuracy_score(yva, pred)
        labels = list(range(n_classes))

        cm_tmp = confusion_matrix(yva, pred, labels=labels)

        recalls = np.divide(
            np.diag(cm_tmp),
            cm_tmp.sum(axis=1),
            out=np.zeros(n_classes, dtype=np.float64),
            where=(cm_tmp.sum(axis=1) != 0),
        )

        bal = float(recalls.mean())

        f1m = f1_score(
            yva, pred,
            average="macro",
            labels=list(range(n_classes)),
            zero_division=0
        )
        cm = confusion_matrix(yva, pred, labels=list(range(n_classes)))
        all_cm += cm

        fold_metrics.append({
            "fold": fold,
            "balanced_accuracy": float(bal),
            "macro_f1": float(f1m),
            "n_train": int(len(tr_idx)),
            "n_val": int(len(va_idx)),
        })

        print(f"[Fold {fold}] bal_acc={bal:.4f} macro_f1={f1m:.4f} (best bal_acc during train={best_val:.4f})", flush=True)

        # Save fold checkpoint
        torch.save(
            {
                "fold": fold,
                "feature_cols": feature_cols,
                "mean": mean,
                "std": std,
                "encoder": best_state["encoder"],
                "head": best_state["head"],
                "metrics": fold_metrics[-1],
                "args": vars(args),
            },
            os.path.join(args.out_dir, f"best_fold_{fold:02d}.pt")
        )

    # Aggregate
    bal_list = [m["balanced_accuracy"] for m in fold_metrics]
    f1_list = [m["macro_f1"] for m in fold_metrics]

    summary = {
        "norm_mode": args.norm_mode,
        "splits_requested": args.splits,
        "splits_used": n_splits,
        "balanced_accuracy_mean": float(np.mean(bal_list)),
        "balanced_accuracy_std": float(np.std(bal_list, ddof=1)) if len(bal_list) > 1 else 0.0,
        "macro_f1_mean": float(np.mean(f1_list)),
        "macro_f1_std": float(np.std(f1_list, ddof=1)) if len(f1_list) > 1 else 0.0,
        "folds": fold_metrics,
        "confusion_matrix_sum": all_cm.tolist(),
    }

    out_json = os.path.join(args.out_dir, "cv_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== CV Summary ===")
    print(json.dumps({k: summary[k] for k in summary if k.endswith(("_mean", "_std"))}, indent=2))
    print(f"Saved: {out_json}")
    print(f"Saved per-fold best checkpoints in: {args.out_dir}")


if __name__ == "__main__":
    main()

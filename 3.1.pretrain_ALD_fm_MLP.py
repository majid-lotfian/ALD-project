#!/usr/bin/env python3
import os
import math
import time
import glob
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from torch.amp import autocast, GradScaler
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader


# =========================
# Utilities
# =========================

# HPC friendly

def set_cpu_thread_env(max_threads: int = 1):
    os.environ.setdefault("OMP_NUM_THREADS", str(max_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(max_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(max_threads))
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(max_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(max_threads))

def list_csv_files(root_dir: str) -> List[str]:
    pattern = os.path.join(root_dir, "**", "*.csv")
    return sorted(glob.glob(pattern, recursive=True))

def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# =========================
# Global normalization stats (streaming)
# =========================

@dataclass
class RunningStats:
    # Welford algorithm for vector data
    n: int
    mean: np.ndarray
    M2: np.ndarray

    @staticmethod
    def create(d: int) -> "RunningStats":
        return RunningStats(
            n=0,
            mean=np.zeros(d, dtype=np.float64),
            M2=np.zeros(d, dtype=np.float64),
        )

    def update(self, x: np.ndarray):
        """
        x: shape (batch, d) float64
        """
        if x.ndim != 2:
            raise ValueError("RunningStats.update expects 2D array")
        b, d = x.shape
        if d != self.mean.shape[0]:
            raise ValueError("Dim mismatch in RunningStats.update")

        for i in range(b):
            self.n += 1
            delta = x[i] - self.mean
            self.mean += delta / self.n
            delta2 = x[i] - self.mean
            self.M2 += delta * delta2

    def finalize(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.n < 2:
            var = np.ones_like(self.mean)
        else:
            var = self.M2 / (self.n - 1)
        std = np.sqrt(np.maximum(var, 1e-12))
        return self.mean.astype(np.float32), std.astype(np.float32)


def compute_global_norm_stats(
    csv_files: List[str],
    feature_cols: List[str],
    sex_col: str,
    chunk_rows: int,
    max_files: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Compute global mean/std for:
      - lipid features (feature_cols)
      - sex feature appended as last dimension

    Excludes severity and IDs by construction (feature_cols must already exclude them).
    """
    #d = len(feature_cols) + 1  # +1 for sex
    d = len(feature_cols)  # ONLY omics features (exclude sex)
    stats = RunningStats.create(d)

    files = csv_files if max_files is None else csv_files[:max_files]

    for fi, path in enumerate(files):
        if verbose:
            print(f"[{now()}] [NORM] Scanning file {fi+1}/{len(files)}: {path}", flush=True)

        for chunk in pd.read_csv(path, chunksize=chunk_rows):
            if sex_col not in chunk.columns:
                raise ValueError(f"sex_col='{sex_col}' not found in {path}")

            missing = [c for c in feature_cols if c not in chunk.columns]
            if missing:
                raise ValueError(f"Missing feature columns in {path}: {missing[:10]} ... total {len(missing)}")

            '''sex = chunk[sex_col].astype(np.float32).to_numpy().reshape(-1, 1)
            x = chunk[feature_cols].astype(np.float32).to_numpy()

            xb = np.concatenate([x, sex], axis=1).astype(np.float64)
            stats.update(xb)'''
            x = chunk[feature_cols].astype(np.float32).to_numpy()
            stats.update(x.astype(np.float64))

    mean, std = stats.finalize()
    return {"mean": mean, "std": std}


# =========================
# Corruptions: masking and noise
# =========================

'''def apply_feature_mask(
    x: torch.Tensor,
    mask_ratio: float,
    mask_value: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    x: (B, D) normalized features incl sex at last dim.
    We do NOT mask sex by default.
    Returns:
      x_masked, mask (boolean tensor shape (B, D), True where masked)
    """
    B, D = x.shape
    D_lip = D - 1  # last dim is sex

    num_mask = max(1, int(mask_ratio * D_lip))
    mask = torch.zeros((B, D), dtype=torch.bool, device=x.device)

    for b in range(B):
        idx = torch.randperm(D_lip, device=x.device)[:num_mask]
        mask[b, idx] = True

    x_masked = x.clone()
    x_masked[mask] = mask_value
    return x_masked, mask'''

def apply_feature_mask(
    x: torch.Tensor,
    mask_ratio: float,
    mask_value: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    x: (B, D) normalized features incl sex at last dim.
    We do NOT mask sex by default.
    Returns:
      x_masked, mask (boolean tensor shape (B, D), True where masked)
    """
    B, D = x.shape
    D_lip = D - 1  # last dim is sex

    # Sample a Bernoulli mask for lipid dims
    # Guarantee at least 1 masked feature per row (important when mask_ratio small)
    mask_lip = (torch.rand((B, D_lip), device=x.device) < mask_ratio)

    # Ensure each row has at least one True
    rows_with_none = ~mask_lip.any(dim=1)
    if rows_with_none.any():
        # pick one random feature index for those rows
        idx = torch.randint(0, D_lip, (rows_with_none.sum().item(),), device=x.device)
        mask_lip[rows_with_none, idx] = True

    mask = torch.zeros((B, D), dtype=torch.bool, device=x.device)
    mask[:, :D_lip] = mask_lip

    x_masked = x.clone()
    x_masked[mask] = mask_value
    return x_masked, mask


'''def apply_denoise_corruption(
    x: torch.Tensor,
    noise_std: float,
    dropout_p: float,
    drop_value: float = 0.0,
) -> torch.Tensor:
    """
    Mild Gaussian noise + feature dropout on lipid dims only.
    Sex dim is left intact.
    """
    B, D = x.shape
    D_lip = D - 1

    x_cor = x.clone()

    if noise_std > 0:
        noise = torch.randn((B, D_lip), device=x.device) * noise_std
        x_cor[:, :D_lip] = x_cor[:, :D_lip] + noise

    if dropout_p > 0:
        drop_mask = (torch.rand((B, D_lip), device=x.device) < dropout_p)
        x_cor[:, :D_lip][drop_mask] = drop_value

    return x_cor'''

def apply_denoise_corruption(
    x: torch.Tensor,
    noise_std: float,
    dropout_p: float,
    drop_value: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Mild Gaussian noise + feature dropout on lipid dims only.
    Sex dim is left intact.
    Returns:
      x_cor, drop_mask (boolean tensor shape (B, D), True where dropped)
    """
    B, D = x.shape
    D_lip = D - 1

    x_cor = x.clone()
    drop_mask_full = torch.zeros((B, D), dtype=torch.bool, device=x.device)

    if noise_std > 0:
        noise = torch.randn((B, D_lip), device=x.device) * noise_std
        x_cor[:, :D_lip] = x_cor[:, :D_lip] + noise

    if dropout_p > 0:
        drop_mask = (torch.rand((B, D_lip), device=x.device) < dropout_p)
        x_cor[:, :D_lip][drop_mask] = drop_value
        drop_mask_full[:, :D_lip] = drop_mask

    return x_cor, drop_mask_full

# =========================
# Model: Residual MLP encoder + reconstruction head + optional projection head
# =========================

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


class ReconHead(nn.Module):
    def __init__(self, z_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(z_dim, out_dim)

    def forward(self, z):
        return self.fc(z)


class ProjHead(nn.Module):
    def __init__(self, z_dim: int, p_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.GELU(),
            nn.Linear(z_dim, p_dim),
        )

    def forward(self, z):
        return self.net(z)


# =========================
# Contrastive loss (InfoNCE / NT-Xent)
# =========================

def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    z1, z2: (B, P)
    Positive pairs are (z1[i], z2[i]).
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    B = z1.shape[0]

    reps = torch.cat([z1, z2], dim=0)   # (2B, P)
    sim = reps @ reps.T                 # (2B, 2B)
    sim = sim / temperature

    mask = torch.eye(2 * B, device=sim.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e9)

    pos = torch.cat(
        [torch.arange(B, device=sim.device) + B, torch.arange(B, device=sim.device)],
        dim=0
    )

    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    return (-log_prob[torch.arange(2 * B, device=sim.device), pos]).mean()


# =========================
# IterableDataset: streams rows from many CSVs
# =========================

class LipidPretrainIterable(IterableDataset):
    """
    Streams normalized samples from CSV files.
    Yields dict: {"x": tensor(D,)}
    Corruptions are applied on-device in the training step for speed.
    """

    def __init__(
        self,
        csv_files: List[str],
        feature_cols: List[str],
        sex_col: str,
        mean: np.ndarray,
        std: np.ndarray,
        chunk_rows: int,
        seed: int = 123,
    ):
        super().__init__()
        self.csv_files = csv_files
        self.feature_cols = feature_cols
        self.sex_col = sex_col
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.chunk_rows = chunk_rows
        self.seed = seed

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        if worker is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = worker.id, worker.num_workers

        rng = np.random.default_rng(self.seed + worker_id)

        files = self.csv_files[worker_id::num_workers]
        rng.shuffle(files)

        while True:
            for path in files:
                for chunk in pd.read_csv(path, chunksize=self.chunk_rows):
                    if self.sex_col not in chunk.columns:
                        raise ValueError(f"sex_col='{self.sex_col}' not found in {path}")

                    '''sex = chunk[self.sex_col].astype(np.float32).to_numpy().reshape(-1, 1)

                    missing = [c for c in self.feature_cols if c not in chunk.columns]
                    if missing:
                        raise ValueError(f"Missing feature columns in {path}: {missing[:10]} ... total {len(missing)}")

                    x = chunk[self.feature_cols].astype(np.float32).to_numpy()
                    xb = np.concatenate([x, sex], axis=1)  # (N, D)

                    xb = (xb - self.mean) / self.std'''


                    sex = chunk[self.sex_col].astype(np.float32).to_numpy().reshape(-1, 1)

                    missing = [c for c in self.feature_cols if c not in chunk.columns]
                    if missing:
                        raise ValueError(f"Missing feature columns in {path}: {missing[:10]} ... total {len(missing)}")

                    # Omics features only
                    x = chunk[self.feature_cols].astype(np.float32).to_numpy()  # (N, D_omics)

                    # Normalize ONLY omics features (mean/std are shape (D_omics,))
                    x_norm = (x - self.mean) / self.std  # (N, D_omics)

                    # Append raw sex as last dim (NOT normalized)
                    xb = np.concatenate([x_norm, sex], axis=1).astype(np.float32)  # (N, D_omics + 1)

                    for i in range(xb.shape[0]):
                        yield {"x": torch.from_numpy(xb[i]).float()}


# =========================
# Column selection
# =========================

def build_feature_columns(df: pd.DataFrame, sex_col: str, drop_cols: List[str]) -> List[str]:
    cols = [c for c in df.columns if c not in set(drop_cols)]
    cols = [c for c in cols if c != sex_col]
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    return num_cols


# =========================
# Args
# =========================

def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--synthetic_root", type=str, default="./synthetic_data",
                    help="Root directory containing synthetic CSVs in subfolders.")
    ap.add_argument("--out_dir", type=str, default="./pretrain_runs/run1")

    ap.add_argument("--sex_col", type=str, default="sex")
    ap.add_argument("--severity_col", type=str, default="severity")
    ap.add_argument("--id_col", type=str, default="Sample_ID")

    # Data streaming
    ap.add_argument("--chunk_rows", type=int, default=5000)
    ap.add_argument("--num_workers", type=int, default=4)

    # Training steps
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--batch_size", type=int, default=512)

    # Model
    ap.add_argument("--hidden_dim", type=int, default=1024)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--z_dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)

    # SSL objectives
    '''ap.add_argument("--mask_ratio", type=float, default=0.25)
    ap.add_argument("--noise_std", type=float, default=0.10)
    ap.add_argument("--feat_dropout", type=float, default=0.10)'''
    ap.add_argument("--mask_ratio", type=float, default=0.40)
    ap.add_argument("--noise_std", type=float, default=0.05)
    ap.add_argument("--feat_dropout", type=float, default=0.05)

    ap.add_argument("--lambda_mask", type=float, default=1.0)
    ap.add_argument("--lambda_denoise", type=float, default=0.3)
    ap.add_argument("--lambda_contrast", type=float, default=0.1)
    ap.add_argument("--contrast_warmup_steps", type=int, default=10000)
    ap.add_argument("--use_contrastive", action="store_true")
    ap.add_argument("--proj_dim", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.3)

    # Optimization
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=123)

    # Norm stats
    ap.add_argument("--norm_stats_path", type=str, default="",
                    help="If provided and exists, load mean/std from this file instead of recomputing.")
    ap.add_argument("--max_files_for_norm", type=int, default=0,
                    help="If >0, use only first N files to compute norm stats (debug).")

    # Resume
    ap.add_argument("--resume_from", type=str, default="",
                    help="Path to checkpoint (.pt) to resume from. Restores model+optimizer and continues steps.")

    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--save_every", type=int, default=5000)

    # HPC friendly
    ap.add_argument("--amp", action="store_true", help="Enable mixed precision training (AMP).")
    ap.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16"],
                    help="AMP dtype. bf16 recommended on A100/H100; fp16 otherwise.")
    ap.add_argument("--prefetch_factor", type=int, default=4,
                help="Batches prefetched per worker (only if num_workers>0).")
    ap.add_argument("--persistent_workers", action="store_true",
                    help="Keep DataLoader workers alive between iterations.")
    ap.add_argument("--pin_memory", action="store_true",
                    help="Pin CPU memory for faster H2D transfer.")
    ap.add_argument("--cpu_threads", type=int, default=1,
                    help="Max CPU threads per process (prevents BLAS oversubscription).")




    return ap.parse_args()


def seed_worker(worker_id: int):
    # Make numpy/pytorch randomness different across workers
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

# =========================
# Main
# =========================

def main():
    args = parse_args()

    set_cpu_thread_env(args.cpu_threads)
    torch.set_num_threads(args.cpu_threads)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ensure_dir(args.out_dir)
    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    ensure_dir(ckpt_dir)

    # Discover CSVs
    csv_files = list_csv_files(args.synthetic_root)
    if not csv_files:
        raise RuntimeError(f"No CSV files found under {args.synthetic_root}")
    print(f"[{now()}] Found {len(csv_files)} synthetic CSV files under {args.synthetic_root}", flush=True)

    # If resuming, load checkpoint metadata early (feature_cols, mean/std, step, saved args)
    resume_meta = None
    if args.resume_from:
        if not os.path.exists(args.resume_from):
            raise FileNotFoundError(f"--resume_from not found: {args.resume_from}")
        print(f"[{now()}] Resuming requested from: {args.resume_from}", flush=True)
        resume_meta = torch.load(args.resume_from, map_location="cpu")
        if "feature_cols" not in resume_meta or "mean" not in resume_meta or "std" not in resume_meta or "step" not in resume_meta:
            raise ValueError("Resume checkpoint missing required keys: feature_cols/mean/std/step")

    # Infer columns / feature set
    df0 = pd.read_csv(csv_files[0], nrows=10)
    if args.sex_col not in df0.columns:
        raise ValueError(f"sex_col='{args.sex_col}' not found in CSV columns (first file).")

    drop_cols = [args.severity_col, args.id_col]

    if resume_meta is not None:
        feature_cols = list(resume_meta["feature_cols"])
        mean = np.array(resume_meta["mean"], dtype=np.float32)
        std = np.array(resume_meta["std"], dtype=np.float32)

        # Basic sanity check: resumed feature cols exist in first file
        missing = [c for c in feature_cols if c not in df0.columns]
        if missing:
            raise ValueError(f"Resume feature_cols not found in current data schema (first file). Missing: {missing[:10]} ...")

        print(f"[{now()}] Loaded feature_cols ({len(feature_cols)}) and norm stats from resume checkpoint.", flush=True)
    else:
        feature_cols = build_feature_columns(df0, sex_col=args.sex_col, drop_cols=drop_cols)

        # Load/compute normalization stats
        norm_path = args.norm_stats_path if args.norm_stats_path else os.path.join(args.out_dir, "norm_stats.npz")
        if args.norm_stats_path and os.path.exists(args.norm_stats_path):
            data = np.load(args.norm_stats_path)
            mean = data["mean"].astype(np.float32)
            std = data["std"].astype(np.float32)
            print(f"[{now()}] Loaded normalization stats from {args.norm_stats_path}", flush=True)
        elif os.path.exists(norm_path):
            data = np.load(norm_path)
            mean = data["mean"].astype(np.float32)
            std = data["std"].astype(np.float32)
            print(f"[{now()}] Loaded normalization stats from {norm_path}", flush=True)
        else:
            max_files = args.max_files_for_norm if args.max_files_for_norm > 0 else None
            stats = compute_global_norm_stats(
                csv_files=csv_files,
                feature_cols=feature_cols,
                sex_col=args.sex_col,
                chunk_rows=args.chunk_rows,
                max_files=max_files,
                verbose=True,
            )
            mean, std = stats["mean"], stats["std"]
            np.savez(norm_path, mean=mean, std=std)
            print(f"[{now()}] Saved normalization stats to {norm_path}", flush=True)

    in_dim = len(feature_cols) + 1  # +sex
    print(f"[{now()}] Using {len(feature_cols)} numeric lipid features + sex => input dim {in_dim}", flush=True)
    print(f"[{now()}] Excluding columns from SSL: {drop_cols}", flush=True)


    # Feature weights for reconstruction (downweight high-variance features)
    # std is a numpy array of shape (D,)
    feat_w = torch.from_numpy(1.0 / (std + 1e-6)).to(args.device)  # (D_omics,)
    assert feat_w.shape[0] == len(feature_cols)

    # Dataset / Loader (streaming)
    ds = LipidPretrainIterable(
        csv_files=csv_files,
        feature_cols=feature_cols,
        sex_col=args.sex_col,
        mean=mean,
        std=std,
        chunk_rows=args.chunk_rows,
        seed=args.seed,
    )

    '''loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        drop_last=True,
    )'''

    # HPC friendly

    pin = args.pin_memory or args.device.startswith("cuda")

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin,
        drop_last=True,
        persistent_workers=(args.persistent_workers and args.num_workers > 0),
        prefetch_factor=(args.prefetch_factor if args.num_workers > 0 else None),
        worker_init_fn=seed_worker,

    )

    # Model
    encoder = MLPEncoder(
        in_dim= 2 * in_dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        z_dim=args.z_dim,
        dropout=args.dropout,
    )
    #recon = ReconHead(z_dim=args.z_dim, out_dim=in_dim)
    omics_dim = len(feature_cols)
    recon = ReconHead(z_dim=args.z_dim, out_dim=omics_dim)
    proj = ProjHead(z_dim=args.z_dim, p_dim=args.proj_dim) if args.use_contrastive else None

    encoder.to(args.device)
    recon.to(args.device)
    if proj is not None:
        proj.to(args.device)

    params = list(encoder.parameters()) + list(recon.parameters()) + (list(proj.parameters()) if proj is not None else [])
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    use_amp = args.amp and args.device.startswith("cuda")
    #scaler = GradScaler(enabled=use_amp and args.amp_dtype == "fp16")
    scaler = GradScaler("cuda", enabled=use_amp and args.amp_dtype == "fp16")

    # Resume state (model + optimizer + step)
    start_step = 0
    if resume_meta is not None:
        # Load state dicts on the correct device
        ckpt = torch.load(args.resume_from, map_location=args.device)

        encoder.load_state_dict(ckpt["encoder"])
        recon.load_state_dict(ckpt["recon"])
        if proj is not None and ckpt.get("proj") is not None:
            proj.load_state_dict(ckpt["proj"])

        if "opt" in ckpt and ckpt["opt"] is not None:
            opt.load_state_dict(ckpt["opt"])
            if use_amp and args.amp_dtype == "fp16" and ckpt.get("scaler") is not None:
                scaler.load_state_dict(ckpt["scaler"])
        else:
            print(f"[{now()}] [WARN] Optimizer state missing in checkpoint; resuming with fresh optimizer.", flush=True)

        start_step = int(ckpt["step"]) + 1
        print(f"[{now()}] Resumed model/optimizer. Continuing from step {start_step}.", flush=True)

    if start_step >= args.steps:
        print(f"[{now()}] start_step ({start_step}) >= --steps ({args.steps}). Nothing to do. Exiting.", flush=True)
        return

    # Training loop
    print(f"[{now()}] Starting pretraining on device={args.device} for steps [{start_step} .. {args.steps-1}]", flush=True)

    #huber = nn.SmoothL1Loss(reduction="mean")
    huber = nn.SmoothL1Loss(reduction="none")
    step = start_step
    t0 = time.time()

    it = iter(loader)
    while step < args.steps:
        batch = next(it)
        x = batch["x"].to(args.device, non_blocking=True)  # (B, D)

        # Corrupted views
        '''x_masked, mask = apply_feature_mask(x, mask_ratio=args.mask_ratio, mask_value=0.0)
        x_noisy = apply_denoise_corruption(x, noise_std=args.noise_std, dropout_p=args.feat_dropout, drop_value=0.0)

        # Forward
        z_masked = encoder(x_masked)
        z_noisy = encoder(x_noisy)'''

        x_masked, mask = apply_feature_mask(x, mask_ratio=args.mask_ratio, mask_value=0.0)
        x_noisy, drop_mask = apply_denoise_corruption(
            x, noise_std=args.noise_std, dropout_p=args.feat_dropout, drop_value=0.0
        )

        # Indicators as float (B, D)
        mask_f = mask.float()
        drop_f = drop_mask.float()

        # Encoder inputs become (B, 2D)
        x_masked_in = torch.cat([x_masked, mask_f], dim=1)
        x_noisy_in  = torch.cat([x_noisy,  drop_f], dim=1)

        
        # HPC friendly

        amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

        #with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
        with autocast("cuda", dtype=amp_dtype, enabled=use_amp):
        
            z_masked = encoder(x_masked_in)
            z_noisy  = encoder(x_noisy_in)

            xhat_masked = recon(z_masked)
            xhat_noisy = recon(z_noisy)

            # Losses:
            # Mask loss: only masked entries (sex never masked)
            '''L_mask = huber(xhat_masked[mask], x[mask]) if mask.any() else torch.tensor(0.0, device=args.device)

            # Denoise loss: lipid dims only (exclude sex last dim)
            L_denoise = huber(xhat_noisy[:, :-1], x[:, :-1])'''

            # Elementwise errors
            '''err_mask_full = huber(xhat_masked, x)  # (B, D)
            err_denoise_full = huber(xhat_noisy, x)  # (B, D)'''
            x_omics = x[:, :-1]  # (B, D_omics) exclude sex

            err_mask_full = huber(xhat_masked, x_omics)      # (B, D_omics)
            err_denoise_full = huber(xhat_noisy, x_omics)    # (B, D_omics)

            # Weighted mask loss: only masked entries (sex never masked)
            #if mask.any():
            mask_omics = mask[:, :-1]
            if mask_omics.any():
                w_mask = feat_w.unsqueeze(0).expand_as(err_mask_full)  # (B, D)
                L_mask = (err_mask_full[mask_omics] * w_mask[mask_omics]).mean()
            else:
                L_mask = torch.tensor(0.0, device=args.device)

            # Weighted denoise loss: lipid dims only (exclude sex last dim)
            #w_den = feat_w[:-1].unsqueeze(0)  # (1, D-1)
            L_denoise = (err_denoise_full * feat_w.unsqueeze(0)).mean()

            # Contrastive (optional, warmup)
            L_contrast = torch.tensor(0.0, device=args.device)
            lam_c = 0.0
            if args.use_contrastive and proj is not None and step >= args.contrast_warmup_steps:
                p1 = proj(z_masked)
                p2 = proj(z_noisy)
                L_contrast = info_nce_loss(p1, p2, temperature=args.temperature)
                lam_c = args.lambda_contrast

            # Total
            loss = args.lambda_mask * L_mask + args.lambda_denoise * L_denoise + lam_c * L_contrast

        '''opt.zero_grad(set_to_none=True)
        loss.backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)

        opt.step()'''

        # HPC friendly

        opt.zero_grad(set_to_none=True)

        if use_amp and args.amp_dtype == "fp16":
            scaler.scale(loss).backward()

            if args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)

            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)

            opt.step()

        # Logging
        if step % args.log_every == 0:
            dt = time.time() - t0
            sps = (step - start_step + 1) / max(dt, 1e-9)
            rows_per_sec = sps * args.batch_size
            print(
                f"[{now()}] step={step:06d} "
                f"loss={loss.item():.4f} "
                f"Lmask={L_mask.item():.4f} "
                f"Lden={L_denoise.item():.4f} "
                f"Lcon={L_contrast.item():.4f} "
                f"lam_con={lam_c:.3f} "
                f"rows/s={rows_per_sec:.0f}"
                f"sps={sps:.1f}",
                flush=True
            )

        # Checkpointing
        if (step > 0) and (step % args.save_every == 0):
            ckpt_path = os.path.join(ckpt_dir, f"ckpt_step_{step:06d}.pt")
            state = {
                "step": step,
                "args": vars(args),
                "feature_cols": feature_cols,
                "mean": mean,
                "std": std,
                "encoder": encoder.state_dict(),
                "recon": recon.state_dict(),
                "proj": proj.state_dict() if proj is not None else None,
                "opt": opt.state_dict(),
                "scaler": scaler.state_dict() if (use_amp and args.amp_dtype == "fp16") else None,
            }
            torch.save(state, ckpt_path)
            print(f"[{now()}] Saved checkpoint: {ckpt_path}", flush=True)

        step += 1

    final_path = os.path.join(ckpt_dir, f"ckpt_final_step_{args.steps:06d}.pt")
    torch.save(
        {
            "step": args.steps - 1,
            "args": vars(args),
            "feature_cols": feature_cols,
            "mean": mean,
            "std": std,
            "encoder": encoder.state_dict(),
            "recon": recon.state_dict(),
            "proj": proj.state_dict() if proj is not None else None,
            "opt": opt.state_dict(),
        },
        final_path
    )
    print(f"[{now()}] Finished. Final checkpoint saved to {final_path}", flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset

from .preprocessing import NormStats, compute_norm_stats_from_array, normalize_array
from .schema import FeatureSchema


class FinetuneDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return {
            'x': torch.from_numpy(self.x[idx]),
            'y': torch.tensor(self.y[idx], dtype=torch.long),
        }


def load_real_table(real_csv: str, schema: FeatureSchema, severity_col: str) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    df = pd.read_csv(real_csv)
    x = df.reindex(columns=schema.feature_cols).astype(np.float32).to_numpy()
    y = df[severity_col].astype(int).to_numpy()
    return x, y, df


def make_cv_splits(y: np.ndarray, n_splits: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(tr, va) for tr, va in skf.split(np.zeros_like(y), y)]


def stratified_subsample_indices(y: np.ndarray, frac: float, min_per_class: int, seed: int) -> np.ndarray:
    if frac >= 1.0:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    kept = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        n_keep = max(min_per_class, int(round(len(idx) * frac)))
        n_keep = min(n_keep, len(idx))
        chosen = rng.choice(idx, size=n_keep, replace=False)
        kept.extend(chosen.tolist())
    kept = np.array(sorted(kept), dtype=np.int64)
    return kept

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class NormStats:
    mean: np.ndarray
    std: np.ndarray


def compute_norm_stats_from_files(csv_files: Iterable[str | Path], feature_cols: List[str]) -> NormStats:
    total = None
    total_sq = None
    n = 0
    for fp in csv_files:
        df = pd.read_csv(fp, usecols=lambda c: c in set(feature_cols))
        x = df[feature_cols].astype(np.float32).to_numpy()
        if total is None:
            total = x.sum(axis=0, dtype=np.float64)
            total_sq = (x.astype(np.float64) ** 2).sum(axis=0)
        else:
            total += x.sum(axis=0, dtype=np.float64)
            total_sq += (x.astype(np.float64) ** 2).sum(axis=0)
        n += x.shape[0]
    mean = total / max(n, 1)
    var = total_sq / max(n, 1) - mean ** 2
    std = np.sqrt(np.maximum(var, 1e-12))
    return NormStats(mean=mean.astype(np.float32), std=std.astype(np.float32))


def compute_norm_stats_from_array(x: np.ndarray) -> NormStats:
    mean = x.mean(axis=0, dtype=np.float64)
    std = x.std(axis=0, dtype=np.float64)
    std = np.maximum(std, 1e-6)
    return NormStats(mean.astype(np.float32), std.astype(np.float32))


def normalize_array(x: np.ndarray, stats: NormStats) -> np.ndarray:
    return ((x - stats.mean) / stats.std).astype(np.float32)

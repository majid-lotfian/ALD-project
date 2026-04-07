from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


@dataclass
class NormStats:
    mean: np.ndarray
    std: np.ndarray


def compute_norm_stats_from_files(
    csv_files: Iterable[str | Path],
    feature_cols: List[str],
    chunksize: int = 4096,
    max_rows_per_file: Optional[int] = None,
) -> NormStats:
    feature_set = set(feature_cols)

    total = None
    total_sq = None
    n = 0

    for fp in csv_files:
        rows_seen = 0

        reader = pd.read_csv(
            fp,
            usecols=lambda c: c in feature_set,
            chunksize=chunksize,
        )

        for chunk in reader:
            chunk = chunk.reindex(columns=feature_cols)
            if max_rows_per_file is not None:
                remaining = max_rows_per_file - rows_seen
                if remaining <= 0:
                    break
                if len(chunk) > remaining:
                    chunk = chunk.iloc[:remaining].copy()

            x = chunk.astype(np.float32).to_numpy(copy=True)

            if x.size == 0:
                continue

            x64 = x.astype(np.float64, copy=False)

            if total is None:
                total = x64.sum(axis=0)
                total_sq = (x64 ** 2).sum(axis=0)
            else:
                total += x64.sum(axis=0)
                total_sq += (x64 ** 2).sum(axis=0)

            n += x.shape[0]
            rows_seen += x.shape[0]

    if n == 0:
        raise ValueError("No rows found while computing normalization statistics.")

    mean = total / n
    var = total_sq / n - mean ** 2
    std = np.sqrt(np.maximum(var, 1e-12))
    std = np.maximum(std, 1e-6)

    return NormStats(
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
    )


def compute_norm_stats_from_array(x: np.ndarray) -> NormStats:
    mean = x.mean(axis=0, dtype=np.float64)
    std = x.std(axis=0, dtype=np.float64)
    std = np.maximum(std, 1e-6)
    return NormStats(mean.astype(np.float32), std.astype(np.float32))


def normalize_array(x: np.ndarray, stats: NormStats) -> np.ndarray:
    return ((x - stats.mean) / stats.std).astype(np.float32)
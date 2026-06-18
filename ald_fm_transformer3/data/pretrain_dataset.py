from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset, get_worker_info

from .preprocessing import NormStats, normalize_array
from .schema import FeatureSchema


class PretrainCSVIterableDataset(IterableDataset):
    """
    Streaming CSV dataset for pretraining.

    Supports multi-GPU DDP via rank/world_size: each rank receives a disjoint
    subset of files (files[rank::world_size]). Within each rank, DataLoader
    workers further sub-shard by worker.id.
    """

    def __init__(
        self,
        csv_files: Sequence[str | Path],
        schema: FeatureSchema,
        norm_stats: NormStats,
        chunksize: int = 4096,
        max_rows_per_file: Optional[int] = None,
        shuffle_files: bool = False,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
        infinite: bool = False,
    ):
        super().__init__()
        # Each rank gets its own disjoint slice of files
        all_files = [str(fp) for fp in csv_files]
        self.csv_files = all_files[rank::world_size]
        self.schema = schema
        self.norm_stats = norm_stats
        self.chunksize = chunksize
        self.max_rows_per_file = max_rows_per_file
        self.shuffle_files = shuffle_files
        self.seed = seed
        self.rank = rank
        # Cycle through files indefinitely — epoch length controlled by steps_per_epoch.
        # Required for DDP correctness: ranks must never exhaust data at different steps.
        self.infinite = infinite

    def _iter_file_rows(self, fp: str) -> Iterator[Dict[str, torch.Tensor]]:
        rows_yielded = 0
        feature_set = set(self.schema.feature_cols)

        reader = pd.read_csv(
            fp,
            usecols=lambda c: c in feature_set,
            chunksize=self.chunksize,
        )

        for chunk in reader:
            chunk = chunk.reindex(columns=self.schema.feature_cols)

            if chunk.isnull().any().any():
                missing = chunk.columns[chunk.isnull().all(axis=0)].tolist()
                if missing:
                    raise ValueError(f"File {fp} is missing columns required by schema: {missing}")

            x = chunk.astype(np.float32).to_numpy(copy=True)
            x = normalize_array(x, self.norm_stats)

            if not np.isfinite(x).all():
                raise ValueError(f"Non-finite values after normalization in: {fp}")

            for i in range(x.shape[0]):
                yield {'x': torch.from_numpy(x[i])}
                rows_yielded += 1
                if self.max_rows_per_file is not None and rows_yielded >= self.max_rows_per_file:
                    return

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        worker = get_worker_info()

        if worker is None:
            files = list(self.csv_files)
            worker_seed = self.seed + self.rank
        else:
            # Sub-shard within the rank's file list
            files = self.csv_files[worker.id :: worker.num_workers]
            worker_seed = self.seed + self.rank * 1000 + worker.id

        rng = np.random.default_rng(worker_seed + torch.initial_seed() % (2**31))

        if not files:
            return  # worker has no files — yield nothing (guards infinite loop below)

        if self.infinite:
            while True:
                if self.shuffle_files:
                    files = list(files)
                    rng.shuffle(files)
                for fp in files:
                    yield from self._iter_file_rows(fp)
        else:
            if self.shuffle_files:
                files = list(files)
                rng.shuffle(files)
            for fp in files:
                yield from self._iter_file_rows(fp)


def discover_family_csvs(
    root: str | Path,
    input_folders: Sequence[str],
    file_glob: str = '*.csv',
    max_files: Optional[int] = None,
) -> List[str]:
    root = Path(root)
    files: List[str] = []
    for folder in input_folders:
        path = root / folder
        files.extend(sorted(str(p) for p in path.rglob(file_glob)))
    files = [f for f in files if not f.endswith('_metrics.json')]
    if max_files is not None:
        files = files[:max_files]
    return files

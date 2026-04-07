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
    def __init__(
        self,
        csv_files: Sequence[str | Path],
        schema: FeatureSchema,
        norm_stats: NormStats,
        chunksize: int = 4096,
        max_rows_per_file: Optional[int] = None,
        shuffle_files: bool = False,
        seed: int = 42,
    ):
        super().__init__()
        self.csv_files = [str(fp) for fp in csv_files]
        self.schema = schema
        self.norm_stats = norm_stats
        self.chunksize = chunksize
        self.max_rows_per_file = max_rows_per_file
        self.shuffle_files = shuffle_files
        self.seed = seed

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
                missing_cols = chunk.columns[chunk.isnull().all(axis=0)].tolist()
                if missing_cols:
                    raise ValueError(
                        f"File {fp} is missing feature columns required by schema: {missing_cols}"
                    )

            x = chunk.astype(np.float32).to_numpy(copy=True)
            x = normalize_array(x, self.norm_stats)

            if not np.isfinite(x).all():
                raise ValueError(f"Non-finite values found after normalization in file: {fp}")

            for i in range(x.shape[0]):
                yield {"x": torch.from_numpy(x[i])}
                rows_yielded += 1
                if self.max_rows_per_file is not None and rows_yielded >= self.max_rows_per_file:
                    return

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        worker = get_worker_info()

        if worker is None:
            files = list(self.csv_files)
            worker_seed = self.seed + torch.initial_seed()
        else:
            files = self.csv_files[worker.id :: worker.num_workers]
            worker_seed = self.seed + worker.id + torch.initial_seed()

        if self.shuffle_files:
            rng = np.random.default_rng(worker_seed)
            rng.shuffle(files)

        for fp in files:
            yield from self._iter_file_rows(fp)


def discover_family_csvs(
    root: str | Path,
    input_folders: Sequence[str],
    file_glob: str = "*.csv",
    max_files: Optional[int] = None,
) -> List[str]:
    root = Path(root)
    files: List[str] = []
    for folder in input_folders:
        path = root / folder
        files.extend(sorted(str(p) for p in path.rglob(file_glob)))
    files = [f for f in files if not f.endswith("_metrics.json")]
    if max_files is not None:
        files = files[:max_files]
    return files
from __future__ import annotations

import glob
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .preprocessing import NormStats, normalize_array
from .schema import FeatureSchema


class PretrainCSVDataset(Dataset):
    def __init__(
        self,
        csv_files: Sequence[str | Path],
        schema: FeatureSchema,
        norm_stats: NormStats,
        max_rows_per_file: Optional[int] = None,
    ):
        self.schema = schema
        self.norm_stats = norm_stats
        arrays = []
        self.file_index = []
        for fp in csv_files:
            df = pd.read_csv(fp)
            df = df[[c for c in schema.feature_cols if c in df.columns]].copy()
            df = df.reindex(columns=schema.feature_cols)
            if max_rows_per_file is not None:
                df = df.iloc[:max_rows_per_file].copy()
            x = df.astype(np.float32).to_numpy()
            x = normalize_array(x, norm_stats)
            arrays.append(x)
            self.file_index.extend([str(fp)] * x.shape[0])
        self.x = np.concatenate(arrays, axis=0) if arrays else np.zeros((0, len(schema.feature_cols)), dtype=np.float32)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'x': torch.from_numpy(self.x[idx]),
            'index': torch.tensor(idx, dtype=torch.long),
        }


def discover_family_csvs(root: str | Path, input_folders: Sequence[str], file_glob: str = '*.csv', max_files: Optional[int] = None) -> List[str]:
    root = Path(root)
    files: List[str] = []
    for folder in input_folders:
        path = root / folder
        files.extend(sorted(str(p) for p in path.rglob(file_glob)))
    files = [f for f in files if not f.endswith('_metrics.json')]
    if max_files is not None:
        files = files[:max_files]
    return files

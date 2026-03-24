from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd


@dataclass
class FeatureSchema:
    feature_cols: List[str]
    sex_col: Optional[str]
    severity_col: Optional[str]
    id_col: Optional[str]
    exclude_cols: List[str]

    @property
    def token_cols(self) -> List[str]:
        return list(self.feature_cols)


def infer_feature_schema(
    df: pd.DataFrame,
    sex_col: str = 'sex',
    severity_col: str = 'severity',
    id_col: str = 'Sample_ID',
    exclude_cols: Optional[Iterable[str]] = None,
) -> FeatureSchema:
    exclude = set(exclude_cols or [])
    block = {severity_col, id_col, *exclude}
    keep = []
    for c in df.columns:
        if c in block:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            keep.append(c)
    return FeatureSchema(
        feature_cols=keep,
        sex_col=sex_col if sex_col in df.columns else None,
        severity_col=severity_col if severity_col in df.columns else None,
        id_col=id_col if id_col in df.columns else None,
        exclude_cols=list(exclude),
    )


def save_schema(schema: FeatureSchema, path: str | Path) -> None:
    import json
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('w', encoding='utf-8') as f:
        json.dump(schema.__dict__, f, indent=2)


def load_schema(path: str | Path) -> FeatureSchema:
    import json
    with Path(path).open('r', encoding='utf-8') as f:
        data = json.load(f)
    return FeatureSchema(**data)

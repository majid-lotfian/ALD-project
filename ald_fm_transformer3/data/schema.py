from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Lipid class group IDs (must match V11 causal model structure)
# ---------------------------------------------------------------------------
GROUP_VLCFA_LPC  = 0   # lysophospholipids (LPC, LPE, 1/2-acyl-LPC/LPE, LPG)
GROUP_PC_PE      = 1   # diacyl PC and PE (non-ether)
GROUP_PLASMALOGEN = 2  # ether-linked: PC/PE/LPC/LPE O- P- OplusP-, DG O-/P-, TG O-
GROUP_CERAMIDE   = 3   # Cer, HexCer, Hex2Cer, C1P, S1P
GROUP_SM         = 4   # SM, SM4
GROUP_TG_CE      = 5   # TG, CE, DG (non-ether), BMP
GROUP_OTHER      = 6   # sex, PI, PG, PS, CL, PA, and any unmatched column

# Rules are checked in order; first match wins.
# Each rule is (prefix_tuple, group_id).
# More-specific prefixes (O_, P_, OplusP_) must come before generic ones (PC_, PE_, etc.)
_PREFIX_RULES: list[tuple[tuple[str, ...], int]] = [
    # Plasmalogen ether-linked species (check before PC/PE/LPC/DG/TG)
    (('PC_O_', 'PC_P_', 'PC_OplusP_',
      'PE_O_', 'PE_P_', 'PE_OplusP_',
      'LPC_O_', 'LPC_P_', 'LPC_OplusP_',
      'LPE_O_', 'LPE_P_', 'LPE_OplusP_',
      'DG_O_', 'DG_P_',
      'TG_O_'), GROUP_PLASMALOGEN),
    # VLCFA_LPC: lyso-species (LPC, LPE, 1/2-acyl, LPG)
    (('LPC_', '1_acyl_LPC_', '2_acyl_LPC_',
      'LPE_', '1_acyl_LPE_', '2_acyl_LPE_',
      'LPG_'), GROUP_VLCFA_LPC),
    # PC_PE: diacyl phosphatidylcholines and phosphatidylethanolamines
    (('PC_', 'PE_'), GROUP_PC_PE),
    # Ceramide family
    (('Cer_', 'HexCer_', 'Hex2Cer_', 'C1P_', 'S1P_'), GROUP_CERAMIDE),
    # Sphingomyelin
    (('SM_', 'SM4_'), GROUP_SM),
    # Triglycerides, cholesterol esters, diacylglycerols, BMP
    (('TG_', 'CE_', 'DG_', 'BMP_'), GROUP_TG_CE),
]


def _assign_group(col: str) -> int:
    for prefixes, gid in _PREFIX_RULES:
        if any(col.startswith(p) for p in prefixes):
            return gid
    return GROUP_OTHER


def build_group_ids(feature_cols: List[str]) -> torch.Tensor:
    """Return a LongTensor of shape (num_features,) with lipid class group IDs."""
    return torch.tensor([_assign_group(c) for c in feature_cols], dtype=torch.long)


# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------

@dataclass
class FeatureSchema:
    feature_cols: List[str]
    sex_col: Optional[str]
    target_column: Optional[str]
    id_col: Optional[str]
    exclude_cols: List[str]

    @property
    def token_cols(self) -> List[str]:
        return list(self.feature_cols)


def infer_feature_schema(
    df: pd.DataFrame,
    sex_col: str = 'sex',
    target_column: str = 'severity',
    id_col: str = 'Sample_ID',
    exclude_cols: Optional[Iterable[str]] = None,
) -> FeatureSchema:
    exclude = set(exclude_cols or [])
    block = {target_column, id_col, *exclude}
    keep = []
    for c in df.columns:
        if c in block:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            keep.append(c)
    return FeatureSchema(
        feature_cols=keep,
        sex_col=sex_col if sex_col in df.columns else None,
        target_column=target_column if target_column in df.columns else None,
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

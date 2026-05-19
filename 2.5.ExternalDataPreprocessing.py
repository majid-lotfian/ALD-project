"""
External ALD dataset preprocessing — Metabolomics Workbench
============================================================
Sources:
  ST000741  Singh Lab, Henry Ford Health System (2017)
            Untargeted metabolomics of ALD patient-derived fibroblasts
            18 samples: Healthy control / Mild disease / Severe disease
            CC BY 4.0 — doi: 10.21228/M8HH47

  ST000742  Singh Lab / U. Michigan Metabolomics Core (2017)
            Untargeted metabolomics of human post-mortem brain + mouse ABCD1-KO
            38 samples: Human (Healthy control, NLA, PLS, PL) + Mouse (WT, KO-M, ALD-KO)
            CC BY 4.0 — doi: 10.21228/M8HH47

Output
------
  data/external/metabolomics_workbench/ST000741_processed.csv
  data/external/metabolomics_workbench/ST000742_human_processed.csv
  data/external/metabolomics_workbench/ST000742_mouse_processed.csv
  data/external/metabolomics_workbench/external_data_summary.txt

Feature notes
-------------
These are GENERAL untargeted metabolomics datasets (amino acids, nucleotides, lipids, etc.)
with ~300-400 named metabolites, of which 6-30 are lipid-like species.

The naming convention differs from the main ALD plasma lipidomics dataset:
  MW format:  "16:0 LYSO PC"  →  could map to  1_acyl_LPC_16:0 / LPC_16:0
  MW format:  "16:0-18:0 PC"  →  could map to  PC_34:0

A separate partial-feature alignment to the main 1810-lipid space is computed
and saved as *_lipids_aligned.csv for the overlapping species only.
"""

import re
import os
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path("data/external/metabolomics_workbench")
REAL_CSV = Path("data/ALD_lipidomics_merged-updated.csv")

# ── Severity mappings ─────────────────────────────────────────────────────────

ST741_LABEL_MAP = {
    "subject type:healthy control": 0,
    "subject type:mild disease":    1,
    "subject type:severe disease":  2,
}

ST742_HUMAN_LABEL_MAP = {
    "healthy control": 0,   # no neurological involvement
    "nla":             1,   # neurologically asymptomatic (ABCD1 mutation, no symptoms)
    "pls":             2,   # pre-lesion symptomatic (AMN-like)
    "pl":              3,   # progressive lesion (cerebral ALD)
}

ST742_MOUSE_LABEL_MAP = {
    "wt":      0,   # wild-type control
    "ko-m":    1,   # ABCD1 knockout (mild phenotype)
    "ald-ko":  2,   # ALD knockout (severe phenotype)
}

# ── mwTab parser ──────────────────────────────────────────────────────────────

def parse_mwtab(fpath: Path) -> tuple[dict, list, list, dict]:
    """Parse a Metabolomics Workbench mwTab file.

    Returns
    -------
    sample_factors : dict  sample_id → factor string
    sample_ids     : list  ordered sample IDs from data matrix header
    factor_row     : list  per-sample factor labels aligned to sample_ids
    data           : dict  metabolite_name → {sample_id: peak_area_str}
    """
    content = fpath.read_text(encoding="utf-8", errors="replace")

    sample_factors: dict[str, str] = {}
    for line in content.splitlines():
        if line.startswith("SUBJECT_SAMPLE_FACTORS"):
            parts = line.split("\t")
            if len(parts) >= 3:
                sample_factors[parts[1].strip()] = parts[2].strip()

    in_data = False
    header = None
    factor_row: list[str] = []
    data: dict[str, dict[str, str]] = {}

    for line in content.splitlines():
        if "MS_METABOLITE_DATA_START" in line:
            in_data = True
            continue
        if "MS_METABOLITE_DATA_END" in line:
            in_data = False
            continue
        if not in_data:
            continue

        parts = [p.strip() for p in line.split("\t")]

        if header is None:
            header = parts       # ['Samples', sid1, sid2, ...]
            continue

        if parts[0] == "Factors":
            factor_row = parts[1:]
            continue

        met_name = parts[0]
        if not met_name:
            continue

        data[met_name] = {}
        for i, val in enumerate(parts[1:], 1):
            if i < len(header):
                data[met_name][header[i]] = val

    sample_ids = header[1:] if header else []
    return sample_factors, sample_ids, factor_row, data


def mwtab_to_dataframe(
    neg_path: Path,
    pos_path: Path,
    label_map: dict[str, int],
    species_filter: str | None = None,   # "Human" or "Mus musculus" or None
) -> pd.DataFrame:
    """
    Parse one study (negative + positive ion mode) into a sample × metabolite
    DataFrame with a 'severity' label column.

    Duplicate metabolite names across ion modes get a _neg / _pos suffix.
    Values are converted to float (0 for missing).
    """
    sf_neg, ids_neg, fr_neg, data_neg = parse_mwtab(neg_path)
    sf_pos, ids_pos, fr_pos, data_pos = parse_mwtab(pos_path)

    # Build sample metadata: sample_id → {factor_string, species, label}
    def build_meta(sample_factors, sample_ids, factor_row, ion_suffix):
        meta = {}
        for i, sid in enumerate(sample_ids):
            factor_str = factor_row[i] if i < len(factor_row) else sample_factors.get(sid, "")
            meta[sid] = factor_str
        return meta

    meta_neg = build_meta(sf_neg, ids_neg, fr_neg, "neg")
    meta_pos = build_meta(sf_pos, ids_pos, fr_pos, "pos")

    # Unified sample meta (neg and pos should have the same samples)
    all_sids = list(dict.fromkeys(ids_neg + ids_pos))
    all_meta = {sid: meta_neg.get(sid, meta_pos.get(sid, "")) for sid in all_sids}

    # Optionally filter by species
    if species_filter:
        sp_lower = species_filter.lower()
        all_sids = [s for s in all_sids if sp_lower in all_meta[s].lower()]

    # Resolve label
    def resolve_label(factor_str: str) -> int | None:
        fs = factor_str.lower()
        for key, val in label_map.items():
            if key in fs:
                return val
        return None

    # Build rows
    records = []
    for sid in all_sids:
        factor_str = all_meta[sid]
        label = resolve_label(factor_str)
        if label is None:
            continue   # skip unlabelled samples

        row: dict = {"Sample_ID": sid, "severity": label, "factor_string": factor_str}

        for met, values in data_neg.items():
            col = met + "_neg" if met in data_pos else met
            try:
                row[col] = float(values.get(sid, 0) or 0)
            except ValueError:
                row[col] = 0.0

        for met, values in data_pos.items():
            col = met + "_pos" if met in data_neg else met
            try:
                row[col] = float(values.get(sid, 0) or 0)
            except ValueError:
                row[col] = 0.0

        records.append(row)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = df.set_index("Sample_ID")
    df = df.sort_values("severity")
    return df


def log1p_zscore(df: pd.DataFrame, exclude_cols: list[str]) -> pd.DataFrame:
    """Apply log1p + z-score normalisation to all non-excluded numeric columns."""
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    df = df.copy()
    df[feature_cols] = df[feature_cols].clip(lower=0)   # peak areas are non-negative
    df[feature_cols] = np.log1p(df[feature_cols])
    means = df[feature_cols].mean()
    stds  = df[feature_cols].std().replace(0, 1)
    df[feature_cols] = (df[feature_cols] - means) / stds
    return df


# ── Lipid-name alignment to main dataset ─────────────────────────────────────

LIPID_PATTERN = re.compile(
    r"(?P<chain1>\d+:\d+)[-\s]?(?P<chain2>\d+:\d+)?\s*"
    r"(?P<class>LYSO PC|LYSO-PE|LYSO-PI| PC | PE | SM |CER|CERAMIDE|TG |DG |CE )",
    re.IGNORECASE,
)

def mw_name_to_canonical(mw_name: str) -> str | None:
    """
    Attempt to map a Metabolomics Workbench metabolite name to the naming
    convention used in ALD_lipidomics_merged-updated.csv.

    Examples
    --------
    "16:0 LYSO PC"      → "1_acyl_LPC_16:0"
    "16:0-18:0 PC"      → "PC_34:0"
    "24:0 SM (D18:1/24:0)" → "SM_d18:1/24:0"
    """
    s = mw_name.upper()

    # LYSO PC → LPC
    m = re.search(r"(\d+:\d+)\s*LYSO\s*PC", s)
    if m:
        return f"1_acyl_LPC_{m.group(1).lower()}"

    # LYSO PE → LPE
    m = re.search(r"(\d+:\d+)\s*LYSO[-\s]*PE", s)
    if m:
        return f"1_acyl_LPE_{m.group(1).lower()}"

    # X:Y-A:B PC → PC_(X+A):(Y+B)
    m = re.search(r"(\d+):(\d+)-(\d+):(\d+)\s+PC", s)
    if m:
        total_c = int(m.group(1)) + int(m.group(3))
        total_db = int(m.group(2)) + int(m.group(4))
        return f"PC_{total_c}:{total_db}"

    # X:Y-A:B PE → PE_(X+A):(Y+B)
    m = re.search(r"(\d+):(\d+)-(\d+):(\d+)\s+PE", s)
    if m:
        total_c = int(m.group(1)) + int(m.group(3))
        total_db = int(m.group(2)) + int(m.group(4))
        return f"PE_{total_c}:{total_db}"

    # SM (D18:1/X:Y) → SM_d18:1/X:Y
    m = re.search(r"SM\s*\(?D(\d+:\d+)/(\d+:\d+)\)?", s)
    if m:
        return f"SM_d{m.group(1).lower()}/{m.group(2).lower()}"

    # Ceramide (DX:Y/A:B) or CERAMIDE (DX:Y/A:B)
    m = re.search(r"CER\w*\s*\(?D(\d+:\d+)/(\d+:\d+)\)?", s)
    if m:
        return f"Cer_d{m.group(1).lower()}/{m.group(2).lower()}"

    # Galactosyl ceramide → HexCer
    m = re.search(r"GALACTOS\w+\s*CERAMIDE\s*\(?D(\d+:\d+)/(\d+:\d+)\)?", s)
    if m:
        return f"HexCer_d{m.group(1).lower()}/{m.group(2).lower()}"

    return None


def build_aligned_df(df_ext: pd.DataFrame, real_cols: set[str]) -> pd.DataFrame:
    """
    Project external dataset onto the feature columns of the main ALD dataset
    where a name mapping exists.  Unmapped columns are dropped.
    """
    meta_cols = ["severity", "factor_string"]
    mapping = {}   # ext_col → canonical_name
    for col in df_ext.columns:
        if col in meta_cols:
            continue
        # strip ion-mode suffix if present
        base = re.sub(r"_(neg|pos)$", "", col, flags=re.IGNORECASE)
        canonical = mw_name_to_canonical(base)
        if canonical and canonical in real_cols:
            mapping[col] = canonical

    if not mapping:
        return pd.DataFrame()

    feature_cols = list(mapping.keys())
    aligned = df_ext[feature_cols + [c for c in meta_cols if c in df_ext.columns]].copy()
    aligned = aligned.rename(columns=mapping)
    # Average duplicate feature columns after renaming (exclude string meta cols)
    feat_renamed = [mapping[c] for c in feature_cols]
    numeric_part = aligned[feat_renamed].T.groupby(level=0).mean().T
    for col in meta_cols:
        if col in aligned.columns:
            numeric_part[col] = aligned[col]
    return numeric_part


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(BASE, exist_ok=True)

    real_cols = set()
    if REAL_CSV.exists():
        real_cols = set(pd.read_csv(REAL_CSV, nrows=0).columns) - {
            "Sample_ID", "sex", "severity"
        }
        print(f"Main dataset: {len(real_cols)} lipid features loaded for alignment")

    summary_lines = []

    # ── ST000741: fibroblasts ─────────────────────────────────────────────────
    print("\n── ST000741 (fibroblasts) ──")
    df_741 = mwtab_to_dataframe(
        neg_path       = BASE / "ST000741" / "ST000741_AN001155.txt",
        pos_path       = BASE / "ST000741" / "ST000741_AN001156.txt",
        label_map      = ST741_LABEL_MAP,
        species_filter = None,
    )
    if not df_741.empty:
        meta_cols_741 = ["severity", "factor_string"]
        df_741_norm = log1p_zscore(df_741, exclude_cols=meta_cols_741)
        out_path = BASE / "ST000741_processed.csv"
        df_741_norm.to_csv(out_path)
        print(f"  Saved: {out_path}")
        print(f"  Shape: {df_741_norm.shape}  (samples × metabolites+meta)")
        print(f"  Severity distribution: {df_741_norm['severity'].value_counts().to_dict()}")

        # Aligned version
        df_741_aligned = build_aligned_df(df_741, real_cols)
        if not df_741_aligned.empty:
            df_741_aligned.to_csv(BASE / "ST000741_lipids_aligned.csv")
            print(f"  Aligned lipids saved: {df_741_aligned.shape[1] - 2} features mapped to main dataset")
        else:
            print("  No lipid features could be aligned to main dataset naming convention")

        summary_lines.append(
            f"ST000741 fibroblasts: {df_741_norm.shape[0]} samples, "
            f"{df_741_norm.shape[1] - 2} metabolite features, "
            f"labels: {dict(sorted(df_741_norm['severity'].value_counts().to_dict().items()))}"
        )

    # ── ST000742: brain + mouse ────────────────────────────────────────────────
    print("\n── ST000742 human brain ──")
    df_742h = mwtab_to_dataframe(
        neg_path       = BASE / "ST000742" / "ST000742_AN001157.txt",
        pos_path       = BASE / "ST000742" / "ST000742_AN001158.txt",
        label_map      = ST742_HUMAN_LABEL_MAP,
        species_filter = "Homo sapiens",
    )
    if not df_742h.empty:
        meta_cols_742 = ["severity", "factor_string"]
        df_742h_norm = log1p_zscore(df_742h, exclude_cols=meta_cols_742)
        out_path = BASE / "ST000742_human_processed.csv"
        df_742h_norm.to_csv(out_path)
        print(f"  Saved: {out_path}")
        print(f"  Shape: {df_742h_norm.shape}")
        print(f"  Severity distribution: {df_742h_norm['severity'].value_counts().to_dict()}")

        df_742h_aligned = build_aligned_df(df_742h, real_cols)
        if not df_742h_aligned.empty:
            df_742h_aligned.to_csv(BASE / "ST000742_human_lipids_aligned.csv")
            print(f"  Aligned lipids: {df_742h_aligned.shape[1] - 2} features")
        else:
            print("  No lipid features aligned")

        summary_lines.append(
            f"ST000742 human brain: {df_742h_norm.shape[0]} samples, "
            f"{df_742h_norm.shape[1] - 2} metabolite features, "
            f"labels: {dict(sorted(df_742h_norm['severity'].value_counts().to_dict().items()))}"
        )

    print("\n── ST000742 mouse brain ──")
    df_742m = mwtab_to_dataframe(
        neg_path       = BASE / "ST000742" / "ST000742_AN001157.txt",
        pos_path       = BASE / "ST000742" / "ST000742_AN001158.txt",
        label_map      = ST742_MOUSE_LABEL_MAP,
        species_filter = "Mus musculus",
    )
    if not df_742m.empty:
        df_742m_norm = log1p_zscore(df_742m, exclude_cols=["severity", "factor_string"])
        out_path = BASE / "ST000742_mouse_processed.csv"
        df_742m_norm.to_csv(out_path)
        print(f"  Saved: {out_path}")
        print(f"  Shape: {df_742m_norm.shape}")
        print(f"  Severity distribution: {df_742m_norm['severity'].value_counts().to_dict()}")
        summary_lines.append(
            f"ST000742 mouse brain: {df_742m_norm.shape[0]} samples, "
            f"{df_742m_norm.shape[1] - 2} metabolite features, "
            f"labels: {dict(sorted(df_742m_norm['severity'].value_counts().to_dict().items()))}"
        )

    # ── Summary ──────────────────────────────────────────────────────────────
    summary_path = BASE / "external_data_summary.txt"
    with open(summary_path, "w") as f:
        f.write("External ALD Datasets — Preprocessing Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write("Sources: Metabolomics Workbench ST000741, ST000742\n")
        f.write("Singh Lab / Henry Ford Health System / U. Michigan, 2017\n")
        f.write("License: CC BY 4.0  |  DOI: 10.21228/M8HH47\n\n")
        f.write("Data type: Untargeted LC-MS metabolomics\n")
        f.write("Tissues: Fibroblast cell culture (ST000741), Post-mortem brain (ST000742)\n\n")
        f.write("NOTE: Feature space mismatch with main plasma lipidomics dataset.\n")
        f.write("These datasets contain general metabolomics (~300-400 metabolites),\n")
        f.write("of which only 6-30 are lipid-like species.\n")
        f.write("Direct feature-level merging with the 1810-lipid plasma dataset\n")
        f.write("is not possible — use the *_lipids_aligned.csv files for partial\n")
        f.write("overlap, or use these datasets independently (separate encoder).\n\n")
        f.write("Severity label mapping:\n")
        f.write("  ST000741: 0=Healthy control, 1=Mild disease, 2=Severe disease\n")
        f.write("  ST000742 human: 0=Healthy control, 1=NLA (asymptomatic),\n")
        f.write("                  2=PLS (AMN-like), 3=PL (cerebral ALD)\n")
        f.write("  ST000742 mouse: 0=WT, 1=KO-M, 2=ALD-KO\n\n")
        f.write("Processed files:\n")
        for line in summary_lines:
            f.write(f"  {line}\n")
    print(f"\nSummary written to {summary_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()

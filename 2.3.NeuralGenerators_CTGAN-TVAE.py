"""
STEP 2.3 — Neural Generators for Tabular Data (CTGAN + TVAE, SDV ≥ 1.28)
========================================================================

This script:
  • Loads the canonical ALD lipidomics dataset (CSV).
  • Builds SDV Metadata (all columns explicitly set to numerical).
  • Trains CTGAN and TVAE synthesizers (SDV single_table).
  • Generates multiple synthetic tables per model.
  • Saves each table + basic fidelity metrics (KS, CorrΔ).

Requirements:
  pip install "sdv>=1.28" pandas numpy scipy torch
"""

import os
import json
import time
import random
import numpy as np
import pandas as pd
from scipy import stats

# ---- SDV (for SDV 1.28) ----
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer

# Optional: PyTorch seeding for CTGAN/TVAE
try:
    import torch
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False

# --------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------
INPUT_FILE = "./data/ALD_lipidomics_merged.csv"
OUT_DIR = "./synthetic_data/neural_sdv128"
os.makedirs(OUT_DIR, exist_ok=True)

# How many rows per synthetic table
N_SAMPLES = 20000

# How many tables per model (change this later to scale up)
N_TABLES = 100

# Base seed (we’ll use BASE_SEED + table_index)
BASE_SEED = 2025

# Training knobs (adjust as you scale)
CTGAN_PARAMS = dict(
    epochs=300,             # increase for higher fidelity later
    batch_size=64,          # must be a multiple of pac
    pac=1,                  # avoid PAC assertion on small data
    generator_dim=(128, 128),
    discriminator_dim=(128, 128),
    verbose=True
)

TVAE_PARAMS = dict(
    epochs=300,
    batch_size=256,
    embedding_dim=128,
    compress_dims=(256, 256),
    decompress_dims=(256, 256),
    l2scale=1e-5,
    verbose=True
)

# --------------------------------------------------------------------
# LOAD DATA
# --------------------------------------------------------------------
real_df = pd.read_csv(INPUT_FILE)

# Keep only numeric columns (your canonical file is numeric already)
real_df = real_df.select_dtypes(include=[np.number]).copy()
assert real_df.shape[1] > 0, "No numeric columns found in the merged dataset."
real_df = real_df.astype(float)

print(f"Loaded dataset shape: {real_df.shape}")

# --------------------------------------------------------------------
# BUILD METADATA (ALL COLUMNS AS NUMERICAL)
# --------------------------------------------------------------------
# 1) Detect metadata from dataframe
metadata = Metadata.detect_from_dataframe(
    data=real_df,
    table_name="ald_lipidomics"
)

# 2) Convert all columns to numerical sdtype explicitly (SDV 1.28-supported route)
meta_dict = metadata.to_dict()
cols_meta = meta_dict["tables"]["ald_lipidomics"]["columns"]
for col in cols_meta.keys():
    cols_meta[col]["sdtype"] = "numerical"

# 3) Reload metadata from the edited dict
metadata = Metadata.load_from_dict(meta_dict)

print("Metadata prepared: all columns set to 'numerical' sdtype.")

# --------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------
def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    if HAVE_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def ks_fidelity(real, synth):
    """Average Kolmogorov–Smirnov distance over shared numeric columns."""
    ds = []
    common = real.columns.intersection(synth.columns)
    for c in common:
        try:
            d, _ = stats.ks_2samp(real[c], synth[c])
            ds.append(d)
        except Exception:
            pass
    return float(np.nanmean(ds)) if ds else float("nan")

def corr_similarity(real, synth):
    """Mean absolute difference between correlation matrices."""
    common = real.columns.intersection(synth.columns)
    r = real[common].corr()
    s = synth[common].corr()
    return float((r.sub(s).abs().values).mean())

def save_outputs(prefix, df, metrics):
    csv_path = os.path.join(OUT_DIR, f"{prefix}.csv")
    json_path = os.path.join(OUT_DIR, f"{prefix}_metrics.json")
    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ {prefix}  |  KS={metrics['ks_distance']:.4f}  CorrΔ={metrics['corr_diff']:.4f}")

# --------------------------------------------------------------------
# TRAIN & SAMPLE — CTGAN (SDV 1.28)
# --------------------------------------------------------------------
for t in range(N_TABLES):
    seed = BASE_SEED + t
    set_global_seed(seed)
    print(f"\n[CTGAN] Table {t+1}/{N_TABLES}  (seed={seed})")
    start = time.time()

    ctgan = CTGANSynthesizer(metadata, **CTGAN_PARAMS)
    ctgan.fit(real_df)
    synth = ctgan.sample(num_rows=N_SAMPLES)

    metrics = {
        "model": "CTGAN",
        "seed": seed,
        "n_samples": N_SAMPLES,
        "ks_distance": ks_fidelity(real_df, synth),
        "corr_diff": corr_similarity(real_df, synth),
        "train_seconds": round(time.time() - start, 2)
    }
    save_outputs(f"ctgan_seed{seed:03d}", synth, metrics)

# --------------------------------------------------------------------
# TRAIN & SAMPLE — TVAE (SDV 1.28)
# --------------------------------------------------------------------
for t in range(N_TABLES):
    seed = BASE_SEED + 10_000 + t   # separate seed range for TVAE
    set_global_seed(seed)
    print(f"\n[TVAE] Table {t+1}/{N_TABLES}  (seed={seed})")
    start = time.time()

    tvae = TVAESynthesizer(metadata, **TVAE_PARAMS)
    tvae.fit(real_df)
    synth = tvae.sample(num_rows=N_SAMPLES)

    metrics = {
        "model": "TVAE",
        "seed": seed,
        "n_samples": N_SAMPLES,
        "ks_distance": ks_fidelity(real_df, synth),
        "corr_diff": corr_similarity(real_df, synth),
        "train_seconds": round(time.time() - start, 2)
    }
    save_outputs(f"tvae_seed{seed:03d}", synth, metrics)

print(f"\n✅ Synthetic generation complete. Outputs in: {OUT_DIR}")

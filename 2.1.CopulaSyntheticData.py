"""
STEP 2.1 — Copula-based Synthetic Data Generation
=================================================

Fits Gaussian and Vine Copula models to the canonical ALD lipidomics
dataset and generates large-scale synthetic tables.

Dependencies:
    pip install sdv copulas pyvinecopulib scipy pandas numpy
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from sdv.tabular import GaussianCopula
from copulas.multivariate import VineCopula

# --------------------------------------------------------------------
# 1. CONFIG
# --------------------------------------------------------------------
INPUT_FILE = "./data/ALD_lipidomics_merged.csv"
OUT_DIR = "./synthetic_data/copula"
os.makedirs(OUT_DIR, exist_ok=True)

N_SAMPLES = 10_000       # rows per synthetic table
N_REPEATS = 3            # repeat with different random seeds

# --------------------------------------------------------------------
# 2. LOAD CANONICAL DATA
# --------------------------------------------------------------------
real_df = pd.read_csv(INPUT_FILE)

# drop non-numeric columns like Sample_ID if present
real_df = real_df.select_dtypes(include=[np.number]).copy()

print(f"Loaded dataset shape: {real_df.shape}")

# --------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# --------------------------------------------------------------------
def ks_fidelity(real, synth):
    """Average Kolmogorov-Smirnov distance over all numeric columns."""
    distances = []
    for c in real.columns:
        try:
            d, _ = stats.ks_2samp(real[c], synth[c])
            distances.append(d)
        except Exception:
            continue
    return float(np.nanmean(distances))

def corr_similarity(real, synth):
    """Mean absolute difference between correlation matrices."""
    r_corr = real.corr()
    s_corr = synth.corr()
    common = r_corr.columns.intersection(s_corr.columns)
    diff = (r_corr.loc[common, common] - s_corr.loc[common, common]).abs().values
    return float(np.nanmean(diff))

def save_table_and_metrics(name, synth_df, metrics):
    path_csv = os.path.join(OUT_DIR, f"{name}.csv")
    path_json = os.path.join(OUT_DIR, f"{name}_metrics.json")
    synth_df.to_csv(path_csv, index=False)
    with open(path_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Saved {name}  |  KS={metrics['ks_distance']:.4f}, CorrΔ={metrics['corr_diff']:.4f}")

# --------------------------------------------------------------------
# 4. GAUSSIAN COPULA
# --------------------------------------------------------------------
for seed in range(N_REPEATS):
    print(f"\nTraining GaussianCopula (seed={seed})")
    gc = GaussianCopula(default_distribution="norm", random_state=seed)
    gc.fit(real_df)
    synth_gc = gc.sample(num_rows=N_SAMPLES)

    metrics = {
        "model": "GaussianCopula",
        "seed": seed,
        "n_samples": N_SAMPLES,
        "ks_distance": ks_fidelity(real_df, synth_gc),
        "corr_diff": corr_similarity(real_df, synth_gc),
    }
    save_table_and_metrics(f"gaussian_copula_seed{seed:03d}", synth_gc, metrics)

# --------------------------------------------------------------------
# 5. VINE COPULA
# --------------------------------------------------------------------
for seed in range(N_REPEATS):
    print(f"\nTraining VineCopula (seed={seed})")
    np.random.seed(seed)
    vc = VineCopula("center")          # can use 'regular', 'drawable', 'center'
    vc.fit(real_df)
    synth_vc = vc.sample(N_SAMPLES)

    metrics = {
        "model": "VineCopula",
        "seed": seed,
        "n_samples": N_SAMPLES,
        "ks_distance": ks_fidelity(real_df, synth_vc),
        "corr_diff": corr_similarity(real_df, synth_vc),
    }
    save_table_and_metrics(f"vine_copula_seed{seed:03d}", synth_vc, metrics)

# --------------------------------------------------------------------
# 6. SUMMARY
# --------------------------------------------------------------------
print("\n✅ Synthetic generation complete.")
print(f"Synthetic tables and metrics stored in: {OUT_DIR}")

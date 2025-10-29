"""
STEP 2.1 — Copula-based Synthetic Data (Efficient Multi-Table Generator, SDV ≥ 1.28)
====================================================================================

This version:
  • Keeps all features (or top-K variance features if specified)
  • Splits large tables into manageable feature blocks
  • Fits Gaussian Copula synthesizers (docs-verified)
  • Generates multiple synthetic tables per fitted model (fast diversity)
  • Computes simple fidelity metrics for quality tracking

Recommended for CPU nodes (e.g., Snellius 'rome' or 'genoa' partitions)

Requirements:
  pip install "sdv>=1.28" pandas numpy scipy copulas
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer

# Optional vine copula (rarely needed)
try:
    from copulas.multivariate import VineCopula
    HAVE_VINE = True
except Exception:
    HAVE_VINE = False

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------
INPUT_FILE = "./data/ALD_lipidomics_merged.csv"
OUT_DIR = "./synthetic_data/copula"
os.makedirs(OUT_DIR, exist_ok=True)

# Sampling parameters
N_SAMPLES = 20000          # rows per synthetic table
N_REPEATS = 20              # how many times to fit (distinct copula structures)
TABLES_PER_MODEL = 25       # how many tables to sample from each fitted copula
RUN_VINE = False            # vine copula disabled by default (too slow)

# Performance controls
TOP_K_FEATURES = None       # set to e.g. 400 to keep only top-variance features
BLOCK_SIZE = 200            # features per block (smaller = faster, lower memory)

# ---------------------------------------------------------------------
# LOAD & PREPARE DATA
# ---------------------------------------------------------------------
real_df = pd.read_csv(INPUT_FILE)
real_df = real_df.select_dtypes(include=[np.number]).copy()
assert real_df.shape[1] > 0, "No numeric columns found."

# Optionally keep top-K features by variance
if TOP_K_FEATURES is not None and TOP_K_FEATURES < real_df.shape[1]:
    variances = real_df.var().sort_values(ascending=False)
    top_features = variances.index[:TOP_K_FEATURES]
    real_df = real_df[top_features]

n_samples, n_features = real_df.shape
print(f"Loaded dataset: {n_samples} samples × {n_features} features")

# ---------------------------------------------------------------------
# METRICS HELPERS
# ---------------------------------------------------------------------
def ks_fidelity(real, synth):
    """Average KS distance across columns"""
    dists = []
    common = real.columns.intersection(synth.columns)
    for c in common:
        try:
            d, _ = stats.ks_2samp(real[c], synth[c])
            dists.append(d)
        except Exception:
            pass
    return float(np.nanmean(dists)) if dists else float("nan")

def corr_similarity(real, synth):
    """Mean absolute difference between correlation matrices"""
    common = real.columns.intersection(synth.columns)
    r = real[common].corr()
    s = synth[common].corr()
    return float((r.sub(s).abs().values).mean())

def save_outputs(name, df, metrics):
    csv_path = os.path.join(OUT_DIR, f"{name}.csv")
    json_path = os.path.join(OUT_DIR, f"{name}_metrics.json")
    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"💾 Saved {name}.csv  |  KS={metrics['ks_distance']:.4f}  CorrΔ={metrics['corr_diff']:.4f}")

# ---------------------------------------------------------------------
# BLOCKED GAUSSIAN COPULA GENERATION
# ---------------------------------------------------------------------
n_blocks = int(np.ceil(n_features / BLOCK_SIZE))
print(f"Splitting {n_features} features into {n_blocks} blocks of ≤{BLOCK_SIZE} each.\n")

for seed in range(N_REPEATS):
    np.random.seed(seed)
    print(f"[SDV] Fit {seed+1}/{N_REPEATS} (seed={seed})")

    # Cache fitted block models for re-sampling
    fitted_blocks = []

    for b in range(n_blocks):
        subset_cols = real_df.columns[b*BLOCK_SIZE:(b+1)*BLOCK_SIZE]
        subset = real_df[subset_cols]
        print(f"  ⏳ Fitting block {b+1}/{n_blocks} ({subset.shape[1]} features)...")

        metadata = Metadata.detect_from_dataframe(subset, table_name=f"block_{b+1}")
        gc = GaussianCopulaSynthesizer(
            metadata,
            enforce_min_max_values=False,
            enforce_rounding=False
        )
        try:
            gc.fit(subset)
            fitted_blocks.append((b, gc, subset.columns))
            print(f"    ✅ Block {b+1}/{n_blocks} fitted.")
        except Exception as e:
            print(f"    ⚠️ Block {b+1} failed to fit: {e}")

    # Generate multiple tables from this fitted model
    for rep in range(TABLES_PER_MODEL):
        synth_blocks = []
        for b, gc, cols in fitted_blocks:
            synth_block = gc.sample(num_rows=N_SAMPLES)
            synth_block.columns = cols
            synth_blocks.append(synth_block)

        # Merge all synthetic blocks by column
        synth_gc = pd.concat(synth_blocks, axis=1).iloc[:N_SAMPLES, :]

        metrics = {
            "model": "GaussianCopulaSynthesizer",
            "seed": seed,
            "rep": rep,
            "n_samples": N_SAMPLES,
            "ks_distance": ks_fidelity(real_df.iloc[:, :len(synth_gc.columns)], synth_gc),
            "corr_diff": corr_similarity(real_df.iloc[:, :len(synth_gc.columns)], synth_gc),
        }

        prefix = f"gaussian_copula_seed{seed:03d}_rep{rep:02d}"
        save_outputs(prefix, synth_gc, metrics)

    print(f"✅ Finished all {TABLES_PER_MODEL} tables for seed {seed}.\n")

# ---------------------------------------------------------------------
# OPTIONAL: Vine Copula (rarely used for large feature sets)
# ---------------------------------------------------------------------
if RUN_VINE and HAVE_VINE:
    for seed in range(N_REPEATS):
        np.random.seed(seed)
        print(f"[copulas] VineCopula (seed={seed})")
        try:
            vc = VineCopula("center")
            vc.fit(real_df)
            synth_vc = pd.DataFrame(vc.sample(N_SAMPLES), columns=real_df.columns)
            metrics = {
                "model": "VineCopula",
                "seed": seed,
                "n_samples": N_SAMPLES,
                "ks_distance": ks_fidelity(real_df, synth_vc),
                "corr_diff": corr_similarity(real_df, synth_vc),
            }
            save_outputs(f"vine_copula_seed{seed:03d}", synth_vc, metrics)
        except Exception as e:
            print(f"⚠️ Vine copula failed: {e}")

print(f"\n✅ All synthetic generation complete. Outputs in: {OUT_DIR}")

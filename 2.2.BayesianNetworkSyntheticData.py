"""
STEP 2.2a — Bayesian Network Synthetic Data Generation (Memory-Optimized)
=========================================================================
Trains several manageable Bayesian Networks on subsets of the canonical
ALD lipidomics dataset, each on quantile-binned, top-variance features.

Designed to run comfortably on a 32 GB RAM workstation.

Requirements:
    pip install pomegranate==0.14.8 pandas numpy scipy scikit-learn
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import KBinsDiscretizer
from pomegranate import BayesianNetwork

# --------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------
INPUT_FILE = "./data/ALD_lipidomics_merged.csv"
OUT_DIR = "./synthetic_data/bayesian_network_optimized"
os.makedirs(OUT_DIR, exist_ok=True)

# Controls memory footprint
TOP_K_FEATURES = 1800     # keep only the top-variance features
N_BINS = 32              # discretization bins per feature
BLOCK_SIZE = 200         # features per BN (split into blocks)
N_SAMPLES = 20000       # rows per synthetic table
N_TABLES = 500            # number of synthetic tables to generate
BASE_SEED = 42

# --------------------------------------------------------------------
# LOAD DATA
# --------------------------------------------------------------------
real_df = pd.read_csv(INPUT_FILE)
real_df = real_df.select_dtypes(include=[np.number]).copy()
assert real_df.shape[1] > 0, "No numeric columns found."



# Reduce to top-K by variance (most informative features)
variances = real_df.var().sort_values(ascending=False)
selected_cols = variances.head(TOP_K_FEATURES).index
#real_df = real_df[selected_cols]   #uncomment this line if you want to use the TOP_K_FEATURES

print(f"Loaded dataset: {real_df.shape[0]} samples × {real_df.shape[1]} features")
print(f"Using top {TOP_K_FEATURES} variance features.")

# --------------------------------------------------------------------
# PREPROCESS — QUANTILE BINNING
# --------------------------------------------------------------------
disc = KBinsDiscretizer(n_bins=N_BINS, encode="ordinal", strategy="quantile")
real_disc = pd.DataFrame(
    disc.fit_transform(real_df),
    columns=real_df.columns
)
print(f"Discretized data into {N_BINS} quantile bins per feature.")

# --------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------
def ks_fidelity(real, synth):
    ds = []
    for c in real.columns.intersection(synth.columns):
        try:
            d, _ = stats.ks_2samp(real[c], synth[c])
            ds.append(d)
        except Exception:
            pass
    return float(np.nanmean(ds)) if ds else float("nan")

def corr_similarity(real, synth):
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
    print(f"✅ {name}  |  KS={metrics['ks_distance']:.4f}  CorrΔ={metrics['corr_diff']:.4f}")

# --------------------------------------------------------------------
# TRAIN SMALL BNs ON FEATURE BLOCKS
# --------------------------------------------------------------------
feature_blocks = [
    real_disc.columns[i:i + BLOCK_SIZE]
    for i in range(0, real_disc.shape[1], BLOCK_SIZE)
]

print(f"Splitting {real_disc.shape[1]} features into {len(feature_blocks)} blocks of ≤{BLOCK_SIZE}.")

for table_idx in range(N_TABLES):
    np.random.seed(BASE_SEED + table_idx)
    print(f"\nGenerating synthetic table {table_idx+1}/{N_TABLES}")

    synthetic_blocks = []

    for bi, cols in enumerate(feature_blocks):
        subset = real_disc[cols]
        print(f"  Fitting BN block {bi+1}/{len(feature_blocks)} on {len(cols)} features...")

        # Train Bayesian Network on block
        model = BayesianNetwork.from_samples(
            subset.values,
            algorithm="chow-liu",
            state_names=subset.columns,
            pseudocount=1e-3
        )

        synth_subset = pd.DataFrame(model.sample(N_SAMPLES, min_prob=1e-8), columns=subset.columns)
        synthetic_blocks.append(synth_subset)

    # Merge all block samples horizontally
    synth_full = pd.concat(synthetic_blocks, axis=1)
    synth_full = synth_full[real_disc.columns]  # ensure original order

    # Evaluate fidelity
    metrics = {
        "model": "BayesianNetwork-blocked",
        "seed": int(BASE_SEED + table_idx),
        "table_index": int(table_idx),
        "n_samples": int(N_SAMPLES),
        "n_blocks": len(feature_blocks),
        "features_per_block": BLOCK_SIZE,
        "ks_distance": ks_fidelity(real_disc, synth_full),
        "corr_diff": corr_similarity(real_disc, synth_full),
    }

    save_outputs(f"bn_table{table_idx:03d}", synth_full, metrics)

# --------------------------------------------------------------------
# SUMMARY
# --------------------------------------------------------------------
print(f"\n✅ Generated {N_TABLES} synthetic tables under memory constraints.")
print(f"Output directory: {OUT_DIR}")

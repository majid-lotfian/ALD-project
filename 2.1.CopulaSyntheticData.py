"""
STEP 2.1 — Copula-based Synthetic Data (Docs-verified, SDV ≥ 1.28)

This script:
  1) Loads your canonical single-table dataset (CSV).
  2) Builds SDV Metadata from a pandas.DataFrame (docs API).
  3) Fits SDV's GaussianCopulaSynthesizer.
  4) Samples synthetic data and writes it to disk.
  5) (Optional) Also fits a vine copula using the 'copulas' library.

Requirements:
  pip install "sdv>=1.28" pandas numpy scipy
  # Optional for vine:
  pip install copulas
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats

# --- SDV (docs-backed) ---
from sdv.metadata import Metadata                       # docs show using Metadata.detect_from_dataframe
from sdv.single_table import GaussianCopulaSynthesizer  # docs page for GaussianCopulaSynthesizer

# --- Optional: vine copula via 'copulas' (separate library with its own docs) ---
try:
    from copulas.multivariate import VineCopula  # documented in Copulas
    HAVE_VINE = True
except Exception:
    HAVE_VINE = False

# ---------------------------
# Config
# ---------------------------
INPUT_FILE = "./data/ALD_lipidomics_merged.csv"
OUT_DIR = "./synthetic_data/copula_docs_verified"
os.makedirs(OUT_DIR, exist_ok=True)

N_SAMPLES = 10_000
N_REPEATS = 3
RUN_VINE = True  # set False to skip the optional copulas-based vine generator

# ---------------------------
# Load data
# ---------------------------
real_df = pd.read_csv(INPUT_FILE)
# Keep only numeric columns (follows typical practice for copula modeling on continuous vars;
# SDV will internally transform other sdtypes if present, but your canonical set is numeric already)
real_df = real_df.select_dtypes(include=[np.number]).copy()
assert real_df.shape[1] > 0, "No numeric columns found."

# ---------------------------
# Build Metadata from a DataFrame  (docs API)
# ---------------------------
# SDV docs: Metadata.detect_from_dataframe(data=<DataFrame>, table_name=<str>) → Metadata
metadata = Metadata.detect_from_dataframe(
    data=real_df,
    table_name="ald_lipidomics"
)

# ---------------------------
# Helpers: basic fidelity metrics
# ---------------------------
def ks_fidelity(real, synth):
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

# ---------------------------
# SDV Gaussian Copula (docs-backed)
# ---------------------------
for seed in range(N_REPEATS):
    print(f"\n[SDV] GaussianCopulaSynthesizer (seed={seed})")

    # set seeds globally before constructing or fitting the synthesizer
    import random
    random.seed(seed)
    np.random.seed(seed)

    gc = GaussianCopulaSynthesizer(
        metadata,
        enforce_min_max_values=False,
        enforce_rounding=False
    )

    gc.fit(real_df)
    synth_gc = gc.sample(num_rows=N_SAMPLES)

    metrics = {
        "model": "GaussianCopulaSynthesizer",
        "seed": seed,
        "n_samples": N_SAMPLES,
        "ks_distance": ks_fidelity(real_df, synth_gc),
        "corr_diff": corr_similarity(real_df, synth_gc),
    }
    save_outputs(f"gaussian_copula_seed{seed:03d}", synth_gc, metrics)

# ---------------------------
# Optional: Vine Copula (via 'copulas' library docs)
# ---------------------------
if RUN_VINE:
    if not HAVE_VINE:
        print("⚠️ 'copulas' not installed; skipping vine copula. Install with: pip install copulas")
    else:
        for seed in range(N_REPEATS):
            print(f"\n[copulas] VineCopula (seed={seed})")
            np.random.seed(seed)
            vc = VineCopula("center")   # Copulas supports vine types incl. 'center', 'regular', 'drawable'
            vc.fit(real_df)
            synth_vc = vc.sample(N_SAMPLES)
            synth_vc = pd.DataFrame(synth_vc, columns=real_df.columns)

            metrics = {
                "model": "VineCopula (copulas)",
                "seed": seed,
                "n_samples": N_SAMPLES,
                "ks_distance": ks_fidelity(real_df, synth_vc),
                "corr_diff": corr_similarity(real_df, synth_vc),
            }
            save_outputs(f"vine_copula_seed{seed:03d}", synth_vc, metrics)

print("\n✅ Done. Outputs in:", OUT_DIR)

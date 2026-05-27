#!/usr/bin/env python3
"""
Enhanced Fig 1 — Principled lipid selection for marginal distribution validation.

Replaces the arbitrary "first 2 per class" selection in 2.6.SyntheticDataValidation.py
with two scientifically grounded strategies displayed in a 3×4 grid (12 panels):

  Rows 1–2  (8 panels) — Task-specific
      Top 8 severity-associated lipids: 1 per class (6, class diversity) +
      top 2 additional overall scorers.  Ranked by |Spearman ρ| with disease
      severity in the real data.  These are the lipids whose distributional
      fidelity matters most for ALD severity prediction and biomarker discovery.

  Row 3  (4 panels) — Task-agnostic
      Top 4 lipids by variance in the real data (not already shown above).
      High-variance lipids define the dominant axes of lipidomic variation
      and determine whether the synthetic data is broadly useful for
      clustering, dimensionality reduction, and foundation-model pre-training.

Usage:
    python 2.6b.SyntheticDataValidation_PrincipledFig1.py
    python 2.6b.SyntheticDataValidation_PrincipledFig1.py --smoke-test
"""

import argparse
import tempfile
import warnings
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde, ks_2samp, spearmanr

warnings.filterwarnings("ignore")

# ── Publication style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
})

REAL_C    = "#2166AC"
SYN_C     = "#D6604D"
BAND_A    = 0.20
META_COLS = ["Sample_ID", "sex", "severity"]

BASE_DIR  = Path(__file__).parent
REAL_PATH = BASE_DIR / "data" / "ALD_lipidomics_merged-updated.csv"
SYN_DIR   = BASE_DIR / "synthetic_data" / "causal_knowledge_v8"
FIG_DIR   = BASE_DIR / "figures" / "synthetic_validation_v8"


# ── Shared utilities ──────────────────────────────────────────────────────────
def lipid_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in META_COLS]


def factor_groups(cols: list) -> dict:
    plasmalogen_pfx = ("PE_O_", "PE_P_", "PC_O_", "PC_P_", "LPE_O_", "LPC_O_")
    vlcfa_pfx       = ("1_acyl_LPC_", "2_acyl_LPC_", "LPC_", "LPE_")
    sm_pfx          = ("SM_", "SM4_")
    cer_pfx         = ("Cer_", "HexCer_", "Hex2Cer_", "C1P_", "S1P_")
    tg_pfx          = ("TG_", "TG_O_", "DG_", "CE_")
    groups = {g: [] for g in ["VLCFA / LPC", "PC / PE", "Plasmalogen",
                               "Ceramide", "SM", "TG / CE"]}
    for c in cols:
        if any(c.startswith(p) for p in plasmalogen_pfx):
            groups["Plasmalogen"].append(c)
        elif any(c.startswith(p) for p in vlcfa_pfx):
            groups["VLCFA / LPC"].append(c)
        elif any(c.startswith(p) for p in sm_pfx):
            groups["SM"].append(c)
        elif any(c.startswith(p) for p in cer_pfx):
            groups["Ceramide"].append(c)
        elif any(c.startswith(p) for p in tg_pfx):
            groups["TG / CE"].append(c)
        else:
            groups["PC / PE"].append(c)
    return groups


def load_real(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_world(path: Path, cols: list = None) -> pd.DataFrame:
    return pd.read_csv(path, usecols=cols) if cols else pd.read_csv(path)


def world_paths(syn_dir: Path) -> list:
    return sorted(syn_dir.glob("ALD_world_*.csv"))


def sample_worlds(wpaths: list, n: int, seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(wpaths), size=min(n, len(wpaths)), replace=False)
    return [wpaths[i] for i in idx]


def short_label(col: str, maxlen: int = 16) -> str:
    return col if len(col) <= maxlen else col[:maxlen] + "…"


# ── Principled selection helpers ──────────────────────────────────────────────
def pick_severity_reps(real: pd.DataFrame, groups: dict,
                        n_class: int = 1, n_extra: int = 2) -> list:
    """Select severity-associated lipids in two passes.

    Pass 1 — class diversity: top `n_class` per lipid class by |Spearman ρ|
              with severity (ensures every class is represented).
    Pass 2 — global top-ups: `n_extra` additional highest-|ρ| lipids from
              anywhere, not already selected.

    Returns list of (col, group_name, rho) ordered by descending |ρ|.
    """
    all_scored = []
    for grp, cols in groups.items():
        for c in cols:
            if c not in real.columns:
                continue
            r, _ = spearmanr(real[c].fillna(0), real["severity"])
            if np.isfinite(r):
                all_scored.append((abs(r), r, c, grp))

    all_scored.sort(reverse=True)

    # Pass 1: 1 per class
    selected = {}      # col → (abs_rho, rho, grp)
    class_done = set()
    for abs_r, r, c, grp in all_scored:
        if grp not in class_done:
            selected[c] = (abs_r, r, grp)
            class_done.add(grp)
        if len(class_done) == len(groups):
            break

    # Pass 2: top-up globally
    for abs_r, r, c, grp in all_scored:
        if len(selected) >= len(groups) * n_class + n_extra:
            break
        if c not in selected:
            selected[c] = (abs_r, r, grp)

    result = [(c, grp, r) for c, (_, r, grp) in selected.items()]
    result.sort(key=lambda x: abs(x[2]), reverse=True)
    return result


def pick_variance_reps(real: pd.DataFrame, groups: dict,
                        n: int = 4, exclude: set = None) -> list:
    """Top `n` lipids by variance in real data, skipping `exclude` set.

    Returns list of (col, group_name, variance).
    """
    exclude = exclude or set()
    lcols   = lipid_cols(real)
    scored  = []
    for c in lcols:
        if c in exclude or c not in real.columns:
            continue
        var = real[c].var()
        if np.isfinite(var):
            grp = next((g for g, cols in groups.items() if c in cols), "PC / PE")
            scored.append((var, c, grp))
    scored.sort(reverse=True)
    return [(c, grp, var) for var, c, grp in scored[:n]]


# ── Figure ─────────────────────────────────────────────────────────────────────
def fig1_principled(real: pd.DataFrame, wpaths: list, fig_dir: Path,
                     n_worlds: int = 30):
    lcols  = lipid_cols(real)
    groups = factor_groups(lcols)

    # ── Selection ─────────────────────────────────────────────────────────────
    sev_reps = pick_severity_reps(real, groups, n_class=1, n_extra=2)   # 8 lipids
    sev_cols = [c for c, _, _ in sev_reps]
    sev_rho  = {c: rho for c, _, rho in sev_reps}
    sev_grp  = {c: grp for c, grp, _ in sev_reps}

    var_reps = pick_variance_reps(real, groups, n=4, exclude=set(sev_cols))  # 4 lipids
    var_cols = [c for c, _, _ in var_reps]
    var_var  = {c: v   for c, _, v in var_reps}
    var_grp  = {c: grp for c, grp, _ in var_reps}

    # Spearman ρ for variance-panel annotation (informational only)
    var_rho = {}
    for c in var_cols:
        r, _ = spearmanr(real[c].fillna(0), real["severity"])
        var_rho[c] = r if np.isfinite(r) else 0.0

    all_cols = sev_cols + var_cols  # 12 total

    # ── x-grids ───────────────────────────────────────────────────────────────
    x_grids = {}
    for c in all_cols:
        lo, hi = real[c].quantile(0.005), real[c].quantile(0.995)
        pad = 0.4 * (hi - lo)
        x_grids[c] = np.linspace(lo - pad, hi + pad, 250)

    # ── Accumulate synthetic KDEs ─────────────────────────────────────────────
    syn_kdes = {c: [] for c in all_cols}
    for wp in sample_worlds(wpaths, n_worlds):
        world = load_world(wp, cols=all_cols)
        for c in all_cols:
            if c not in world.columns:
                continue
            vals = world[c].dropna().values
            if len(vals) < 10:
                continue
            try:
                syn_kdes[c].append(gaussian_kde(vals, bw_method="scott")(x_grids[c]))
            except Exception:
                pass

    ks_world_path = sample_worlds(wpaths, 1)[0]

    # ── Layout: 3 rows × 4 cols ───────────────────────────────────────────────
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    axes = axes.flatten()

    def _draw_panel(ax, c, bg_color):
        ax.set_facecolor(bg_color)
        xg = x_grids[c]

        kde_r = gaussian_kde(real[c].dropna().values, bw_method="scott")
        ax.plot(xg, kde_r(xg), color=REAL_C, lw=2.0, zorder=4)

        if syn_kdes[c]:
            arr  = np.array(syn_kdes[c])
            mean = arr.mean(axis=0)
            lo95 = np.percentile(arr, 2.5, axis=0)
            hi95 = np.percentile(arr, 97.5, axis=0)
            ax.plot(xg, mean, color=SYN_C, lw=2.0, zorder=4)
            ax.fill_between(xg, lo95, hi95, color=SYN_C, alpha=BAND_A)

            syn_sample = load_world(ks_world_path, cols=[c])[c].dropna()
            ks, _ = ks_2samp(real[c].dropna(), syn_sample)
            ax.text(0.97, 0.95, f"KS = {ks:.3f}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=7, color="#444")

        ax.set_xlabel("z-score", fontsize=7)
        ax.set_ylabel("Density", fontsize=7)

    # ── Rows 1–2: severity-associated ─────────────────────────────────────────
    for i, c in enumerate(sev_cols):
        ax = axes[i]
        _draw_panel(ax, c, bg_color="#EEF4FB")
        rho = sev_rho[c]
        sign = "+" if rho >= 0 else "−"
        ax.text(0.03, 0.95, f"ρ = {sign}{abs(rho):.2f}",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=7.5, color=REAL_C, fontweight="bold")
        ax.set_title(f"{sev_grp[c]}  |  {short_label(c, 14)}", fontsize=8, pad=3)

    # ── Row 3: high-variance ───────────────────────────────────────────────────
    for j, c in enumerate(var_cols):
        ax = axes[8 + j]
        _draw_panel(ax, c, bg_color="#F4F8EE")
        rho = var_rho[c]
        sign = "+" if rho >= 0 else "−"
        ax.text(0.03, 0.95, f"ρ = {sign}{abs(rho):.2f}",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=7.5, color="#666")
        ax.set_title(f"{var_grp[c]}  |  {short_label(c, 14)}", fontsize=8, pad=3)

    fig.tight_layout(rect=[0, 0.05, 1, 0.93])

    # ── Section labels ────────────────────────────────────────────────────────
    fig.text(0.01, 0.935,
             "A   Task-specific — Severity-associated lipids"
             "  (top 8 by |ρ(lipid, severity)|; 1 per class + 2 global top-ups)",
             fontsize=9, fontweight="bold", color="#2166AC", va="top")
    fig.text(0.01, 0.375,
             "B   Task-agnostic — High-variance lipids"
             "  (top 4 by variance; broadest lipidomic coverage for clustering / pre-training)",
             fontsize=9, fontweight="bold", color="#4DAC26", va="top")

    # Dashed separator between rows 2 and 3
    line = plt.Line2D([0.01, 0.99], [0.375, 0.375],
                      transform=fig.transFigure, color="#bbb", lw=1.2, ls="--")
    fig.add_artist(line)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = [
        Line2D([0], [0], color=REAL_C, lw=2,  label="Real data"),
        Line2D([0], [0], color=SYN_C,  lw=2,  label=f"Synthetic (mean, n={n_worlds} worlds)"),
        mpatches.Patch(color=SYN_C,    alpha=0.35, label="Synthetic 95% CI"),
        mpatches.Patch(color="#EEF4FB", edgecolor="#aaa", label="Section A — task-specific"),
        mpatches.Patch(color="#F4F8EE", edgecolor="#aaa", label="Section B — task-agnostic"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, 0.0), frameon=False, fontsize=7.5)

    fig.suptitle(
        "Marginal Distributions: Real vs. Synthetic  (Principled Lipid Selection)",
        fontsize=12, fontweight="bold", y=0.99)

    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"fig1b_marginal_principled.{ext}")
    plt.close(fig)
    print("  ✓ Fig 1b — Principled marginal distributions saved")


# ── Smoke-test data ───────────────────────────────────────────────────────────
def make_smoke_data(tmp_dir: Path, n_worlds: int = 5, n_samples: int = 300):
    rng = np.random.default_rng(42)
    prefixes = {
        "LPC_": 8, "1_acyl_LPC_": 4, "PC_": 12, "PE_": 8,
        "PE_O_": 4, "Cer_d18:1/": 4, "SM_d18:1/": 4, "TG_": 6, "CE_": 4,
    }
    lipid_names = []
    for pref, cnt in prefixes.items():
        for j in range(cnt):
            lipid_names.append(f"{pref}{16 + j}:0")

    real_n = 184
    real_df = pd.DataFrame(
        rng.standard_normal((real_n, len(lipid_names))), columns=lipid_names
    )
    real_df.insert(0, "Sample_ID", [f"real_{i}" for i in range(real_n)])
    real_df.insert(1, "sex",       rng.integers(0, 2, real_n))
    real_df.insert(2, "severity",  rng.integers(0, 5, real_n))

    real_path = tmp_dir / "data" / "ALD_lipidomics_merged.csv"
    real_path.parent.mkdir(parents=True, exist_ok=True)
    real_df.to_csv(real_path, index=False)

    syn_dir = tmp_dir / "synthetic_data" / "causal_knowledge_v8"
    syn_dir.mkdir(parents=True, exist_ok=True)
    for w in range(1, n_worlds + 1):
        sdf = pd.DataFrame(
            rng.standard_normal((n_samples, len(lipid_names))), columns=lipid_names
        )
        sdf.insert(0, "Sample_ID", [f"SYN_{k:04d}" for k in range(n_samples)])
        sdf.insert(1, "sex",       rng.integers(0, 2, n_samples))
        sdf.insert(2, "severity",  rng.integers(0, 5, n_samples))
        sdf.to_csv(syn_dir / f"ALD_world_{w:03d}.csv", index=False)

    return real_path, syn_dir


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Enhanced Fig 1 with principled lipid selection")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--n-worlds",   type=int, default=30)
    args = parser.parse_args()

    real_path = REAL_PATH
    syn_dir   = SYN_DIR
    fig_dir   = FIG_DIR

    if args.smoke_test:
        tmp = Path(tempfile.mkdtemp())
        real_path, syn_dir = make_smoke_data(tmp)
        fig_dir = tmp / "figures" / "validation"
        print(f"[SMOKE TEST] Fake data in: {tmp}")

    print("Loading real data …")
    real   = load_real(real_path)
    wpaths = world_paths(syn_dir)
    print(f"  Real: {real.shape[0]} samples × {real.shape[1]} columns")
    print(f"  Synthetic: {len(wpaths)} worlds in {syn_dir}")
    if not wpaths:
        raise FileNotFoundError(f"No ALD_world_*.csv files in {syn_dir}")

    fig1_principled(real, wpaths, fig_dir, n_worlds=args.n_worlds)
    print(f"\nDone. Figure saved to: {fig_dir}")


if __name__ == "__main__":
    main()

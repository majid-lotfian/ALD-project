#!/usr/bin/env python3
"""
Synthetic Data Validation Suite — ALD Lipidomics  (v2)

Changes vs 2.6.SyntheticDataValidation.py
------------------------------------------
Fig 1 — Principled lipid selection (3 rows × 4 cols, 12 panels):
  Rows 1–2  Task-specific  : top 8 severity-associated lipids ranked by
            |Spearman ρ| with severity (1 per class for diversity + 2 global
            top-ups).  Fidelity of these lipids matters most for severity
            prediction and biomarker discovery.
  Row 3     Task-agnostic  : top 4 lipids by variance (not already shown).
            High-variance lipids drive clustering, dimensionality reduction,
            and foundation-model pre-training quality.

Figs 2–7 — unchanged from 2.6.

Usage:
    python 2.6.2.SyntheticDataValidation.py              # full run
    python 2.6.2.SyntheticDataValidation.py --smoke-test
    python 2.6.2.SyntheticDataValidation.py --figs 1 3 5
"""

import argparse
import tempfile
import warnings
from pathlib import Path

from typing import Dict
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde, ks_2samp, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

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


# ── Utilities ─────────────────────────────────────────────────────────────────
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


def pick_reps(groups: dict, n_per_group: int = 3, existing: list = None) -> list:
    reps = []
    for cols in groups.values():
        candidates = [c for c in cols if existing is None or c in existing]
        reps.extend(candidates[:n_per_group])
    return reps


def pick_severity_reps(real: pd.DataFrame, groups: dict,
                        n_class: int = 1, n_extra: int = 2) -> list:
    """Top severity-associated lipids: 1 per class + n_extra global top-ups.

    Returns list of (col, group_name, rho) sorted by descending |rho|.
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

    selected = {}
    class_done = set()
    for abs_r, r, c, grp in all_scored:
        if grp not in class_done:
            selected[c] = (abs_r, r, grp)
            class_done.add(grp)
        if len(class_done) == len(groups):
            break

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
    """Top n lipids by variance in real data, skipping excluded set.

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


def pick_variance_stratified_reps(real: pd.DataFrame, groups: dict,
                                   n_class: int = 1, n_extra: int = 2) -> list:
    """Top variance lipids: n_class per class + n_extra global top-ups.

    Mirrors pick_severity_reps but ranks by variance instead of |Spearman ρ|.
    Returns list of (col, group_name, variance) sorted by descending variance.
    """
    all_scored = []
    for grp, cols in groups.items():
        for c in cols:
            if c not in real.columns:
                continue
            var = real[c].var()
            if np.isfinite(var):
                all_scored.append((var, c, grp))
    all_scored.sort(reverse=True)

    selected  = {}
    class_done = set()
    for var, c, grp in all_scored:
        if grp not in class_done:
            selected[c] = (var, grp)
            class_done.add(grp)
        if len(class_done) == len(groups):
            break

    for var, c, grp in all_scored:
        if len(selected) >= len(groups) * n_class + n_extra:
            break
        if c not in selected:
            selected[c] = (var, grp)

    result = [(c, grp, var) for c, (var, grp) in selected.items()]
    result.sort(key=lambda x: x[2], reverse=True)
    return result


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


# ── Figure 1 — Marginal distributions (principled selection) ──────────────────
def fig1_marginal_distributions(real: pd.DataFrame, wpaths: list, fig_dir: Path,
                                 n_worlds: int = 30):
    """3 rows × 4 cols — principled lipid selection.

    Rows 1–2  Task-agnostic  (green tint): top 8 by variance (1 per class + 2 global).
    Row 3     Task-specific  (blue tint): top 4 by |Spearman ρ| with severity (not shown above).
    Each panel annotated with KS statistic and real-data Spearman ρ.
    """
    lcols  = lipid_cols(real)
    groups = factor_groups(lcols)

    # Rows 1–2: top 8 by variance, stratified (1 per class + 2 global top-ups)
    var_reps = pick_variance_stratified_reps(real, groups, n_class=1, n_extra=2)
    var_cols = [c for c, _, _ in var_reps]
    var_grp  = {c: grp for c, grp, _ in var_reps}

    var_rho = {}
    for c in var_cols:
        r, _ = spearmanr(real[c].fillna(0), real["severity"])
        var_rho[c] = r if np.isfinite(r) else 0.0

    # Row 3: top 4 by |Spearman ρ| with severity, not already shown above
    var_set = set(var_cols)
    all_sev_scored = []
    for grp, cols in groups.items():
        for c in cols:
            if c in var_set or c not in real.columns:
                continue
            r, _ = spearmanr(real[c].fillna(0), real["severity"])
            if np.isfinite(r):
                all_sev_scored.append((abs(r), r, c, grp))
    all_sev_scored.sort(reverse=True)
    sev_cols = [c for _, _, c, _ in all_sev_scored[:4]]
    sev_rho  = {c: r  for _, r, c, _ in all_sev_scored[:4]}
    sev_grp  = {c: grp for _, _, c, grp in all_sev_scored[:4]}

    all_cols = var_cols + sev_cols

    x_grids = {}
    for c in all_cols:
        lo, hi = real[c].quantile(0.005), real[c].quantile(0.995)
        pad = 0.4 * (hi - lo)
        x_grids[c] = np.linspace(lo - pad, hi + pad, 250)

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

    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    axes = axes.flatten()

    def _draw_panel(ax, c, bg_color, rho, rho_color):
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

        sign = "+" if rho >= 0 else "−"
        ax.text(0.03, 0.95, f"ρ = {sign}{abs(rho):.2f}", transform=ax.transAxes,
                ha="left", va="top", fontsize=7.5, color=rho_color,
                fontweight="bold" if rho_color == REAL_C else "normal")
        ax.set_xlabel("z-score", fontsize=7)
        ax.set_ylabel("Density", fontsize=7)

    for i, c in enumerate(var_cols):
        _draw_panel(axes[i], c, "#F4F8EE", var_rho[c], "#555")
        axes[i].set_title(f"{var_grp[c]}  |  {short_label(c, 14)}", fontsize=8, pad=3)

    for j, c in enumerate(sev_cols):
        _draw_panel(axes[8 + j], c, "#EEF4FB", sev_rho[c], REAL_C)
        axes[8 + j].set_title(f"{sev_grp[c]}  |  {short_label(c, 14)}", fontsize=8, pad=3)

    fig.tight_layout(rect=[0, 0.05, 1, 0.93])

    fig.text(0.01, 0.935,
             "A   Task-agnostic — top 8 high-variance lipids"
             "  (1 per class + 2 global top-ups, broadest lipidomic coverage for pre-training)",
             fontsize=9, fontweight="bold", color="#4DAC26", va="top")
    fig.text(0.01, 0.375,
             "B   Task-specific — top 4 severity-associated lipids"
             "  (not shown above, ranked by |ρ(lipid, severity)|)",
             fontsize=9, fontweight="bold", color="#2166AC", va="top")

    line = plt.Line2D([0.01, 0.99], [0.375, 0.375],
                      transform=fig.transFigure, color="#bbb", lw=1.2, ls="--")
    fig.add_artist(line)

    legend_handles = [
        Line2D([0], [0], color=REAL_C, lw=2, label="Real data"),
        Line2D([0], [0], color=SYN_C,  lw=2,
               label=f"Synthetic (mean, n={n_worlds} worlds)"),
        mpatches.Patch(color=SYN_C,    alpha=0.35, label="Synthetic 95% CI"),
        mpatches.Patch(color="#F4F8EE", edgecolor="#aaa",
                       label="Section A — task-agnostic"),
        mpatches.Patch(color="#EEF4FB", edgecolor="#aaa",
                       label="Section B — task-specific"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, 0.0), frameon=False, fontsize=7.5)
    fig.suptitle("Marginal Distributions: Real vs. Synthetic Lipidomics Data",
                 fontsize=12, fontweight="bold", y=0.99)

    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"fig1_marginal_distributions.{ext}")
    plt.close(fig)
    print("  ✓ Fig 1 — Marginal distributions (principled selection)")


# ── Figure 2 — Categorical distributions (sex, severity) ──────────────────────
def fig2_categorical_distributions(real: pd.DataFrame, wpaths: list, fig_dir: Path,
                                    n_worlds: int = 100):
    sex_rates  = []
    sev_props  = {s: [] for s in range(5)}

    for wp in sample_worlds(wpaths, n_worlds):
        w = load_world(wp, cols=["sex", "severity"])
        sex_rates.append(w["sex"].mean())
        vc = w["severity"].value_counts(normalize=True)
        for s in range(5):
            sev_props[s].append(vc.get(s, 0.0))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    bar_w = 0.35

    ax = axes[0]
    cats   = ["Female (0)", "Male (1)"]
    real_v = [1 - real["sex"].mean(), real["sex"].mean()]
    syn_v  = [1 - np.mean(sex_rates), np.mean(sex_rates)]
    male_lo, male_hi = np.percentile(sex_rates, [2.5, 97.5])
    syn_lo = [1 - male_hi, male_lo]
    syn_hi = [1 - male_lo, male_hi]
    x = np.arange(2)
    ax.bar(x - bar_w / 2, real_v, bar_w, color=REAL_C, label="Real", zorder=3)
    ax.bar(x + bar_w / 2, syn_v,  bar_w, color=SYN_C,
           label=f"Synthetic (n={n_worlds} worlds)", zorder=3)
    ax.errorbar(x + bar_w / 2, syn_v,
                yerr=[np.maximum(0, np.array(syn_v) - np.array(syn_lo)),
                      np.maximum(0, np.array(syn_hi) - np.array(syn_v))],
                fmt="none", color="#333", capsize=4, lw=1.5, zorder=5)
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_ylabel("Proportion"); ax.set_ylim(0, 1)
    ax.set_title("A   Sex Distribution", pad=6)
    ax.legend(frameon=False)

    ax = axes[1]
    sev_labels = [f"Sev. {s}" for s in range(5)]
    real_sv = [np.mean(real["severity"] == s) for s in range(5)]
    syn_sv  = [np.mean(sev_props[s]) for s in range(5)]
    syn_slo = [np.percentile(sev_props[s], 2.5)  for s in range(5)]
    syn_shi = [np.percentile(sev_props[s], 97.5) for s in range(5)]
    x = np.arange(5)
    ax.bar(x - bar_w / 2, real_sv, bar_w, color=REAL_C, label="Real", zorder=3)
    ax.bar(x + bar_w / 2, syn_sv,  bar_w, color=SYN_C,
           label=f"Synthetic (n={n_worlds} worlds)", zorder=3)
    ax.errorbar(x + bar_w / 2, syn_sv,
                yerr=[np.maximum(0, np.array(syn_sv) - np.array(syn_slo)),
                      np.maximum(0, np.array(syn_shi) - np.array(syn_sv))],
                fmt="none", color="#333", capsize=4, lw=1.5, zorder=5)
    ax.set_xticks(x); ax.set_xticklabels(sev_labels, rotation=25, ha="right")
    ax.set_ylabel("Proportion")
    ax.set_title("B   Disease Severity Distribution", pad=6)
    ax.legend(frameon=False)

    fig.suptitle("Categorical Variable Distributions: Real vs. Synthetic",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"fig2_categorical_distributions.{ext}")
    plt.close(fig)
    print("  ✓ Fig 2 — Categorical distributions")


# ── Figure 3 — Correlation heatmaps ───────────────────────────────────────────
def fig3_correlation_structure(real: pd.DataFrame, wpaths: list, fig_dir: Path,
                                n_worlds: int = 50, n_per_group: int = 7):
    lcols  = lipid_cols(real)
    groups = factor_groups(lcols)

    # Selection per class: top 5 by variance (task-agnostic) +
    # top 2 by |Spearman ρ| with severity (task-specific), deduplicated.
    # Lipids appended group-by-group to preserve natural block structure.
    n_var_per = max(1, n_per_group - 2)   # 5 when n_per_group=7
    n_sev_per = n_per_group - n_var_per   # 2

    sel                  = []
    group_sizes          = []
    group_names_ordered  = []
    for grp_name, grp_cols in groups.items():
        avail = [c for c in grp_cols if c in lcols]
        if not avail:
            continue

        var_scored = sorted(
            [(real[c].var() if np.isfinite(real[c].var()) else 0.0, c) for c in avail],
            reverse=True
        )
        var_picked = [c for _, c in var_scored[:n_var_per]]
        var_set    = set(var_picked)

        sev_scored = []
        for c in avail:
            if c in var_set:
                continue
            valid = real[[c, "severity"]].dropna()
            rho, _ = spearmanr(valid[c], valid["severity"])
            sev_scored.append((abs(rho) if not np.isnan(rho) else 0.0, c))
        sev_scored.sort(reverse=True)
        sev_picked = [c for _, c in sev_scored[:n_sev_per]]

        picked = var_picked + sev_picked
        sel.extend(picked)
        group_sizes.append(len(picked))
        group_names_ordered.append(grp_name)

    # Block boundaries and label midpoints (in heatmap cell coordinates)
    boundaries = list(np.cumsum(group_sizes)[:-1])
    midpoints  = []
    cum = 0
    for sz in group_sizes:
        midpoints.append(cum + sz / 2)
        cum += sz

    real_corr = real[sel].corr().values

    # Systematic world sampling: evenly spaced indices guarantee representative
    # coverage of the hyperparameter space rather than a random cluster.
    step = max(1, len(wpaths) // n_worlds)
    sampled_paths = wpaths[::step][:n_worlds]

    stack = []
    for wp in sampled_paths:
        world = load_world(wp, cols=sel)
        available = [c for c in sel if c in world.columns]
        if len(available) < 5:
            continue
        wc = world[available].sample(min(2000, len(world)), random_state=42)
        stack.append(wc.corr().values)

    syn_corr_mean = np.nanmean(stack, axis=0) if stack else np.zeros_like(real_corr)
    diff_mat      = real_corr - syn_corr_mean
    labels        = [short_label(c, 14) for c in sel]

    # Distinct color per group for boundary labels
    GRP_COLORS = ["#2166AC", "#4DAC26", "#D01C8B", "#F1A340", "#998EC3", "#E08214"]

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.5))
    hkw = dict(xticklabels=labels, yticklabels=labels, linewidths=0,
               square=True, cbar_kws={"shrink": 0.65})

    sns.heatmap(real_corr,     ax=axes[0], vmin=-1, vmax=1, cmap="RdBu_r",
                cbar_kws={**hkw["cbar_kws"], "label": "Pearson r"},
                **{k: v for k, v in hkw.items() if k != "cbar_kws"})
    axes[0].set_title("A   Real Data", pad=8)

    sns.heatmap(syn_corr_mean, ax=axes[1], vmin=-1, vmax=1, cmap="RdBu_r",
                cbar_kws={**hkw["cbar_kws"], "label": "Pearson r"},
                **{k: v for k, v in hkw.items() if k != "cbar_kws"})
    axes[1].set_title(f"B   Synthetic (mean, n={len(sampled_paths)} worlds)", pad=8)

    sns.heatmap(diff_mat,      ax=axes[2], vmin=-0.4, vmax=0.4, cmap="RdBu_r",
                cbar_kws={**hkw["cbar_kws"], "label": "Real − Synthetic"},
                **{k: v for k, v in hkw.items() if k != "cbar_kws"})
    axes[2].set_title("C   Difference (Real − Synthetic)", pad=8)

    # Overlay: dashed group boundaries + group name at each diagonal block centre
    for ax in axes:
        for b in boundaries:
            ax.axhline(b, color="black", lw=1.0, ls="--", alpha=0.5)
            ax.axvline(b, color="black", lw=1.0, ls="--", alpha=0.5)
        ax.tick_params(axis="x", rotation=90, labelsize=6)
        ax.tick_params(axis="y", rotation=0,  labelsize=6)

    for ax in (axes[0], axes[1]):
        for mid, name, col in zip(midpoints, group_names_ordered, GRP_COLORS):
            ax.text(mid, mid, name, ha="center", va="center", fontsize=6,
                    fontweight="bold", color="white",
                    bbox=dict(boxstyle="round,pad=0.15", fc=col, ec="none", alpha=0.75))

    fig.suptitle("Correlation Structure: Real vs. Synthetic Lipidomics Data",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"fig3_correlation_structure.{ext}")
    plt.close(fig)
    print("  ✓ Fig 3 — Correlation structure")


# ── Figure 4 — KS statistics per lipid group ──────────────────────────────────
def fig4_ks_statistics(real: pd.DataFrame, wpaths: list, fig_dir: Path,
                        n_worlds: int = 50, cols_per_group: int = 20):
    lcols  = lipid_cols(real)
    groups = factor_groups(lcols)

    records = []
    for wp in sample_worlds(wpaths, n_worlds):
        world   = load_world(wp)
        world_s = world.sample(min(500, len(world)), random_state=42)
        for grp_name, grp_cols in groups.items():
            ks_vals = []
            for c in grp_cols[:cols_per_group]:
                if c not in world_s.columns or c not in real.columns:
                    continue
                try:
                    stat, _ = ks_2samp(real[c].dropna(), world_s[c].dropna())
                    ks_vals.append(stat)
                except Exception:
                    pass
            if ks_vals:
                records.append({"Lipid Group": grp_name,
                                 "KS Statistic": float(np.mean(ks_vals)),
                                 "n_lipids": len(ks_vals)})

    df_ks  = pd.DataFrame(records)
    order  = (df_ks.groupby("Lipid Group")["KS Statistic"]
              .median().sort_values().index.tolist())
    palette = {g: plt.cm.Set2(i / max(len(order) - 1, 1)) for i, g in enumerate(order)}

    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.violinplot(data=df_ks, x="Lipid Group", y="KS Statistic", order=order,
                   palette=palette, inner=None, cut=0, alpha=0.70, ax=ax, linewidth=1.2)
    sns.stripplot(data=df_ks, x="Lipid Group", y="KS Statistic", order=order,
                  palette=palette, size=3.5, alpha=0.55, jitter=True, ax=ax, zorder=3)

    for i, grp in enumerate(order):
        med = df_ks[df_ks["Lipid Group"] == grp]["KS Statistic"].median()
        ax.text(i, med + 0.005, f"{med:.3f}", ha="center", va="bottom",
                fontsize=7, fontweight="bold", color="#222")

    ax.axhline(0.05, color="#777", lw=1.3, ls="--", label="KS = 0.05 reference")
    ax.set_xlabel("Lipid Class Group")
    ax.set_ylabel(f"Mean KS Statistic per World (n ≤ {cols_per_group} lipids/group)")
    ax.set_title("KS Statistics by Lipid Class — Lower Is Better", pad=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    ax.legend(frameon=False)
    ax.text(0.98, 0.97, f"Each point = one synthetic world (n={n_worlds})",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7.5, color="#555", style="italic")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"fig4_ks_statistics.{ext}")
    plt.close(fig)
    print("  ✓ Fig 4 — KS statistics")


# ── Figure 5 — Propensity score / discriminative fidelity ─────────────────────
def fig5_propensity_score(real: pd.DataFrame, wpaths: list, fig_dir: Path,
                           n_worlds: int = 30, n_features: int = 50):
    lcols  = lipid_cols(real)
    groups = factor_groups(lcols)

    # Top 35 by variance, stratified (~6 per class) — task-agnostic majority
    n_var = 35
    n_sev = n_features - n_var   # 15

    n_per  = max(1, n_var // len(groups))
    var_sel: list = []
    for grp_cols in groups.values():
        avail = [c for c in grp_cols if c in lcols]
        scored_var = sorted(
            [(real[c].var() if np.isfinite(real[c].var()) else 0.0, c) for c in avail],
            reverse=True
        )
        var_sel.extend(c for _, c in scored_var[:n_per])

    if len(var_sel) < n_var:
        all_var = sorted(
            [(real[c].var() if np.isfinite(real[c].var()) else 0.0, c) for c in lcols],
            reverse=True
        )
        var_set = set(var_sel)
        for _, c in all_var:
            if c not in var_set:
                var_sel.append(c)
            if len(var_sel) >= n_var:
                break
    var_set = set(var_sel)

    # Top 15 by |Spearman ρ| with severity, not already selected — task-specific
    sev_scored = []
    for c in lcols:
        if c in var_set:
            continue
        r, _ = spearmanr(real[c].fillna(0), real["severity"])
        if np.isfinite(r):
            sev_scored.append((abs(r), c))
    sev_scored.sort(reverse=True)
    sel = var_sel + [c for _, c in sev_scored[:n_sev]]

    real_X = real[sel].fillna(0).values

    sampled   = sample_worlds(wpaths, n_worlds)
    aucs      = []
    base_fpr  = np.linspace(0, 1, 200)
    tpr_list  = []
    prop_real_all = []
    prop_syn_all  = []

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_roc  = axes[0]
    ax_prop = axes[1]

    for i, wp in enumerate(sampled):
        world    = load_world(wp, cols=sel)
        world    = world[[c for c in sel if c in world.columns]].fillna(0)
        n_common = min(len(real_X), len(world))
        rng      = np.random.default_rng(i)
        ri = rng.choice(len(real_X), n_common, replace=False)
        si = rng.choice(len(world),  n_common, replace=False)
        X  = np.vstack([real_X[ri], world.values[si]])
        y  = np.concatenate([np.zeros(n_common), np.ones(n_common)])

        scaler = StandardScaler()
        Xs     = scaler.fit_transform(X)
        clf    = LogisticRegression(max_iter=500, C=0.1, random_state=42)
        clf.fit(Xs, y)
        prob   = clf.predict_proba(Xs)[:, 1]

        fpr, tpr, _ = roc_curve(y, prob)
        auc = roc_auc_score(y, prob)
        aucs.append(auc)
        tpr_list.append(np.interp(base_fpr, fpr, tpr))
        ax_roc.plot(fpr, tpr, color=SYN_C, alpha=0.25, lw=0.8)
        prop_real_all.append(prob[y == 0])
        prop_syn_all.append(prob[y == 1])

    mean_tpr = np.mean(tpr_list, axis=0)
    std_tpr  = np.std(tpr_list, axis=0)
    ax_roc.plot(base_fpr, mean_tpr, color=SYN_C, lw=2.5,
                label=f"Mean ROC (AUC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f})")
    ax_roc.fill_between(base_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                         color=SYN_C, alpha=0.15)
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random classifier (AUC = 0.50)")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("A   ROC Curve — Real vs. Synthetic", pad=8)
    ax_roc.legend(frameon=False, loc="lower right")
    ax_roc.text(0.03, 0.97, "AUC ≈ 0.50 → data indistinguishable",
                transform=ax_roc.transAxes, ha="left", va="top",
                fontsize=7.5, color="#555", style="italic")

    p_real = np.concatenate(prop_real_all)
    p_syn  = np.concatenate(prop_syn_all)
    xg     = np.linspace(0, 1, 250)
    for arr, color, label in [(p_real, REAL_C, "Real"), (p_syn, SYN_C, "Synthetic")]:
        try:
            kde_v = gaussian_kde(arr, bw_method=0.1)(xg)
            ax_prop.plot(xg, kde_v, color=color, lw=2.2, label=label)
            ax_prop.fill_between(xg, kde_v, color=color, alpha=0.18)
        except Exception:
            pass
    ax_prop.set_xlabel("Propensity Score  P̂(synthetic)")
    ax_prop.set_ylabel("Density")
    ax_prop.set_title("B   Propensity Score Overlap", pad=8)
    ax_prop.legend(frameon=False)
    ax_prop.text(0.5, 0.97, "Ideal: overlapping distributions centred at 0.5",
                 transform=ax_prop.transAxes, ha="center", va="top",
                 fontsize=7.5, color="#555", style="italic")

    fig.suptitle("Discriminative Fidelity: Can a Classifier Distinguish Real from Synthetic?",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"fig5_propensity_score.{ext}")
    plt.close(fig)
    print("  ✓ Fig 5 — Propensity score / discriminative fidelity")


# ── Figure 6 — Forest plot of clinical associations ───────────────────────────
def fig6_clinical_effects(real: pd.DataFrame, wpaths: list, fig_dir: Path,
                           n_worlds: int = 100, n_top: int = 10):
    """Forest plot of mean lipid–severity Spearman ρ per lipid class.

    For each of the 6 lipid classes, selects the top n_top lipids by
    |Spearman ρ| with severity in the real data, then reports:
      - Real:      mean ρ across those n_top lipids (diamond marker)
      - Synthetic: mean ρ per world → 95% CI across n_worlds worlds (bar + dot)

    Using n_top lipids instead of a single arbitrary representative makes the
    class-level estimate robust to individual outliers.
    """
    lcols  = lipid_cols(real)
    groups = factor_groups(lcols)

    # Select top n_top per class by |Spearman ρ| with severity
    class_lipids: Dict[str, list] = {}
    for grp, cols in groups.items():
        avail = [c for c in cols if c in lcols and c in real.columns]
        scored = []
        for c in avail:
            r, _ = spearmanr(real[c].fillna(0), real["severity"])
            if np.isfinite(r):
                scored.append((abs(r), c))
        scored.sort(reverse=True)
        class_lipids[grp] = [c for _, c in scored[:n_top]]

    all_needed = list({c for cols in class_lipids.values() for c in cols})

    # Real: |ρ|-weighted mean across top-n_top lipids per class.
    # Weights = |real ρ| so the strongest signals dominate the class estimate.
    real_mean_rho: Dict[str, float] = {}
    class_weights: Dict[str, Dict[str, float]] = {}   # grp -> {col: weight}
    for grp, cols in class_lipids.items():
        rhos, weights = [], []
        for c in cols:
            r, _ = spearmanr(real[c].fillna(0), real["severity"])
            if np.isfinite(r):
                rhos.append(r)
                weights.append(abs(r))
        if rhos:
            w = np.array(weights)
            w = w / w.sum() if w.sum() > 0 else np.ones(len(w)) / len(w)
            real_mean_rho[grp] = float(np.dot(w, rhos))
            class_weights[grp] = {c: float(wi) for c, wi in zip(cols, w)}
        else:
            real_mean_rho[grp] = 0.0
            class_weights[grp] = {}

    # Synthetic: per-world |ρ|-weighted mean using same weights as real data.
    # Weights are fixed from real data so the same lipids dominate in both.
    syn_mean_rho: Dict[str, list] = {grp: [] for grp in groups}
    for wp in sample_worlds(wpaths, n_worlds, seed=7):
        world = load_world(wp, cols=all_needed + ["severity"])
        world = world.sample(min(500, len(world)), random_state=42)
        for grp, cols in class_lipids.items():
            rhos, weights = [], []
            for c in cols:
                if c not in world.columns or c not in class_weights[grp]:
                    continue
                r, _ = spearmanr(world[c].fillna(0), world["severity"])
                if np.isfinite(r):
                    rhos.append(r)
                    weights.append(class_weights[grp][c])
            if rhos:
                w = np.array(weights)
                w = w / w.sum() if w.sum() > 0 else np.ones(len(w)) / len(w)
                syn_mean_rho[grp].append(float(np.dot(w, rhos)))

    grp_names = list(groups.keys())
    n = len(grp_names)
    fig, ax = plt.subplots(figsize=(9, n * 0.95 + 2))

    for i, grp in enumerate(grp_names):
        yr = float(n - 1 - i)

        sv = syn_mean_rho[grp]
        if sv:
            lo, hi = np.percentile(sv, [2.5, 97.5])
            mn = float(np.mean(sv))
            ax.hlines(yr, lo, hi, color=SYN_C, lw=2.5, alpha=0.8, zorder=3)
            ax.scatter(mn, yr, color=SYN_C, s=55, zorder=4,
                       label=f"Synthetic mean ρ (95% CI, n={n_worlds} worlds)"
                             if i == 0 else "")

        rv = real_mean_rho[grp]
        ax.scatter(rv, yr, color=REAL_C, marker="D", s=80, zorder=5,
                   label=f"Real |ρ|-weighted mean (top {n_top}/class)" if i == 0 else "")

        n_used = len(class_lipids[grp])
        ax.text(-1.07, yr, f"{grp}  (n={n_used})",
                va="center", ha="left", fontsize=8)

    ax.axvline(0, color="#aaa", lw=1.2, ls="--", zorder=2)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-0.8, n - 0.2)
    ax.set_yticks([])
    ax.set_xlabel("|ρ|-weighted mean Spearman ρ  (lipids vs. disease severity)")
    ax.set_title(
        "Clinical Effect Preservation: Weighted Mean Lipid–Severity Associations\n"
        f"per Lipid Class (top {n_top} by |ρ|, weighted by |ρ|) — Real vs. Synthetic",
        pad=8)
    ax.legend(frameon=False, loc="lower right")
    ax.text(0.01, 0.97,
            "Horizontal lines = 95% CI across worlds   ◆ = real mean ρ",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=6.5, color="#666", style="italic")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"fig6_clinical_effects.{ext}")
    plt.close(fig)
    print("  ✓ Fig 6 — Forest plot / clinical effects")


# ── Figure 7 — World variability (violin) ─────────────────────────────────────
def fig7_world_variability(real: pd.DataFrame, wpaths: list, fig_dir: Path,
                            n_worlds: int = 200):
    """Violin plots of summary statistics across synthetic worlds.

    Demographic panels: male prevalence, mean severity.
    Lipid panels: std of the highest-variance lipid per class.
      - Representative = top lipid by variance in real data (principled, not
        arbitrary first column).
      - Std tracked instead of mean: after z-score calibration all lipid means
        are ~0 by construction, so mean panels are uninformative. Std captures
        whether the generator reproduces the real spread for each class.
    """
    lcols  = lipid_cols(real)
    groups = factor_groups(lcols)

    # Top lipid per class by variance in real data
    reps: Dict[str, str] = {}
    for grp, cols in groups.items():
        avail = [c for c in cols if c in lcols and c in real.columns]
        if not avail:
            continue
        best = max(avail, key=lambda c: real[c].var() if np.isfinite(real[c].var()) else 0.0)
        reps[grp] = best

    stat_defs = [
        ("Male prevalence (%)",
         lambda w: w["sex"].mean() * 100 if "sex" in w.columns else np.nan),
        ("Mean severity",
         lambda w: w["severity"].mean() if "severity" in w.columns else np.nan),
    ]
    for grp, col in reps.items():
        stat_defs.append(
            (f"Std {grp}\n({short_label(col, 14)})",
             (lambda c: lambda w: w[c].std() if c in w.columns else np.nan)(col))
        )

    needed_cols = list(reps.values()) + ["sex", "severity"]
    world_stats = {name: [] for name, _ in stat_defs}

    for wp in sample_worlds(wpaths, n_worlds, seed=13):
        world = load_world(wp, cols=[c for c in needed_cols])
        world = world.sample(min(1000, len(world)), random_state=42)
        for name, fn in stat_defs:
            try:
                val = fn(world)
                if not np.isnan(val):
                    world_stats[name].append(float(val))
            except Exception:
                pass

    real_stats = {}
    for name, fn in stat_defs:
        try:
            real_stats[name] = float(fn(real))
        except Exception:
            real_stats[name] = np.nan

    n_stats = len(stat_defs)
    ncols   = 4
    nrows   = int(np.ceil(n_stats / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3.8))
    axes = axes.flatten()

    for i, (name, _) in enumerate(stat_defs):
        ax   = axes[i]
        vals = world_stats[name]
        if not vals:
            ax.set_visible(False)
            continue

        vp = ax.violinplot(vals, positions=[0], showmedians=True,
                           showextrema=True, widths=0.6)
        for body in vp["bodies"]:
            body.set_facecolor(SYN_C); body.set_alpha(0.55); body.set_edgecolor("none")
        for part in ("cmedians", "cmins", "cmaxes", "cbars"):
            if part in vp:
                vp[part].set_color(SYN_C); vp[part].set_lw(1.6)

        rval = real_stats.get(name, np.nan)
        if not np.isnan(rval):
            ax.axhline(rval, color=REAL_C, lw=2.5, zorder=5,
                       label=f"Real: {rval:.2f}")
            ax.legend(frameon=False, fontsize=7.5, loc="upper right")

        if vals:
            med = np.median(vals)
            ax.text(0.55, med, f" {med:.2f}",
                    transform=ax.get_yaxis_transform(),
                    ha="left", va="center", fontsize=7, color=SYN_C)

        ax.set_xticks([0])
        ax.set_xticklabels(["Synthetic\nworlds"], fontsize=8)
        ax.set_title(name, fontsize=8.5, pad=4)

    for j in range(n_stats, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"World Variability: Summary Statistics Across {n_worlds} Synthetic Worlds",
        fontsize=12, fontweight="bold")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"fig7_world_variability.{ext}")
    plt.close(fig)
    print("  ✓ Fig 7 — World variability (violin)")


# ── Smoke test data ───────────────────────────────────────────────────────────
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

    real_n  = 184
    real_df = pd.DataFrame(
        rng.standard_normal((real_n, len(lipid_names))), columns=lipid_names)
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
            rng.standard_normal((n_samples, len(lipid_names))), columns=lipid_names)
        sdf.insert(0, "Sample_ID", [f"SYN_{k:04d}" for k in range(n_samples)])
        sdf.insert(1, "sex",       rng.integers(0, 2, n_samples))
        sdf.insert(2, "severity",  rng.integers(0, 5, n_samples))
        sdf.to_csv(syn_dir / f"ALD_world_{w:03d}.csv", index=False)

    return real_path, syn_dir


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ALD Synthetic Data Validation v2")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--figs", nargs="+", type=int, default=list(range(1, 8)),
                        help="Which figures to generate (default: all)")
    args = parser.parse_args()

    real_path = REAL_PATH
    syn_dir   = SYN_DIR
    fig_dir   = FIG_DIR

    if args.smoke_test:
        tmp = Path(tempfile.mkdtemp())
        real_path, syn_dir = make_smoke_data(tmp)
        fig_dir = tmp / "figures" / "validation"
        print(f"[SMOKE TEST] Using fake data in: {tmp}")

    fig_dir.mkdir(parents=True, exist_ok=True)

    print("Loading real data …")
    real   = load_real(real_path)
    wpaths = world_paths(syn_dir)
    print(f"  Real data : {real.shape[0]} samples × {real.shape[1]} columns")
    print(f"  Synthetic : {len(wpaths)} worlds found in {syn_dir}")
    if not wpaths:
        raise FileNotFoundError(f"No ALD_world_*.csv files in {syn_dir}")

    smoke = args.smoke_test
    dispatch = {
        1: lambda: fig1_marginal_distributions(real, wpaths, fig_dir,
                       n_worlds=5 if smoke else 30),
        2: lambda: fig2_categorical_distributions(real, wpaths, fig_dir,
                       n_worlds=5 if smoke else 100),
        3: lambda: fig3_correlation_structure(real, wpaths, fig_dir,
                       n_worlds=5 if smoke else 30),
        4: lambda: fig4_ks_statistics(real, wpaths, fig_dir,
                       n_worlds=5 if smoke else 50),
        5: lambda: fig5_propensity_score(real, wpaths, fig_dir,
                       n_worlds=3 if smoke else 30),
        6: lambda: fig6_clinical_effects(real, wpaths, fig_dir,
                       n_worlds=5 if smoke else 100),
        7: lambda: fig7_world_variability(real, wpaths, fig_dir,
                       n_worlds=5 if smoke else 200),
    }

    print(f"\nGenerating figures → {fig_dir}\n")
    for fn in sorted(args.figs):
        if fn in dispatch:
            dispatch[fn]()

    print(f"\nDone. All figures saved to:\n  {fig_dir}")


if __name__ == "__main__":
    main()

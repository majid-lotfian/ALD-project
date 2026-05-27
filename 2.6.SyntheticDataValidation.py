#!/usr/bin/env python3
"""
Synthetic Data Validation Suite — ALD Lipidomics
Publication-quality figures comparing synthetic vs. real data.

Usage:
    python 2.6.SyntheticDataValidation.py              # full run (500 worlds)
    python 2.6.SyntheticDataValidation.py --smoke-test # quick test on fake data
    python 2.6.SyntheticDataValidation.py --figs 1 3 5 # specific figures only
"""

import argparse
import tempfile
import warnings
from pathlib import Path

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

REAL_C  = "#2166AC"   # steel blue  — real data
SYN_C   = "#D6604D"   # muted red   — synthetic data
BAND_A  = 0.20        # alpha for 95% CI shading
META_COLS = ["Sample_ID", "sex", "severity"]

# ── Default paths (overridden by smoke-test) ──────────────────────────────────
BASE_DIR  = Path(__file__).parent
REAL_PATH = BASE_DIR / "data" / "ALD_lipidomics_merged-updated.csv"
SYN_DIR   = BASE_DIR / "synthetic_data" / "causal_knowledge_v7"
FIG_DIR   = BASE_DIR / "figures" / "synthetic_validation_v7"


# ── Utilities ─────────────────────────────────────────────────────────────────
def lipid_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in META_COLS]


def factor_groups(cols: list) -> dict:
    """Assign lipid columns to their causal factor group."""
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


# ── Figure 1 — Marginal distributions (KDE) ───────────────────────────────────
def fig1_marginal_distributions(real: pd.DataFrame, wpaths: list, fig_dir: Path,
                                 n_worlds: int = 30):
    lcols  = lipid_cols(real)
    groups = factor_groups(lcols)
    reps   = pick_reps(groups, n_per_group=2, existing=lcols)[:12]

    # x-grids per column
    x_grids = {}
    for c in reps:
        lo, hi = real[c].quantile(0.005), real[c].quantile(0.995)
        pad = 0.4 * (hi - lo)
        x_grids[c] = np.linspace(lo - pad, hi + pad, 250)

    # Accumulate synthetic KDEs across sampled worlds
    syn_kdes = {c: [] for c in reps}
    for wp in sample_worlds(wpaths, n_worlds):
        world = load_world(wp, cols=reps)
        for c in reps:
            if c not in world.columns:
                continue
            vals = world[c].dropna().values
            if len(vals) < 10:
                continue
            try:
                syn_kdes[c].append(gaussian_kde(vals, bw_method="scott")(x_grids[c]))
            except Exception:
                pass

    ncols = 4
    nrows = int(np.ceil(len(reps) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 2.8))
    axes = axes.flatten()

    for i, c in enumerate(reps):
        ax = axes[i]
        xg = x_grids[c]

        # Real KDE
        kde_r = gaussian_kde(real[c].dropna().values, bw_method="scott")
        ax.plot(xg, kde_r(xg), color=REAL_C, lw=2.0, zorder=4)

        # Synthetic: mean ± 95% band
        if syn_kdes[c]:
            arr  = np.array(syn_kdes[c])
            mean = arr.mean(axis=0)
            lo95 = np.percentile(arr, 2.5, axis=0)
            hi95 = np.percentile(arr, 97.5, axis=0)
            ax.plot(xg, mean, color=SYN_C, lw=2.0, zorder=4)
            ax.fill_between(xg, lo95, hi95, color=SYN_C, alpha=BAND_A)
            # KS annotation
            syn_sample = load_world(sample_worlds(wpaths, 1)[0], cols=[c])[c].dropna()
            ks, _ = ks_2samp(real[c].dropna(), syn_sample)
            ax.text(0.97, 0.95, f"KS = {ks:.3f}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=7, color="#444")

        ax.set_title(short_label(c), fontsize=8, pad=3)
        ax.set_xlabel("z-score", fontsize=7)
        ax.set_ylabel("Density", fontsize=7)

    for j in range(len(reps), len(axes)):
        axes[j].set_visible(False)

    legend_handles = [
        Line2D([0], [0], color=REAL_C, lw=2, label="Real data"),
        Line2D([0], [0], color=SYN_C,  lw=2, label=f"Synthetic (mean, n={n_worlds} worlds)"),
        mpatches.Patch(color=SYN_C, alpha=0.35, label="Synthetic 95% CI"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.01), frameon=False)
    fig.suptitle("Marginal Distributions: Real vs. Synthetic Lipidomics Data",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"fig1_marginal_distributions.{ext}")
    plt.close(fig)
    print("  ✓ Fig 1 — Marginal distributions")


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

    # Panel A — Sex
    ax = axes[0]
    cats  = ["Female (0)", "Male (1)"]
    real_v = [1 - real["sex"].mean(), real["sex"].mean()]
    syn_v  = [1 - np.mean(sex_rates), np.mean(sex_rates)]
    # CI for male proportion (female is complement)
    male_lo, male_hi = np.percentile(sex_rates, [2.5, 97.5])
    syn_lo = [1 - male_hi, male_lo]
    syn_hi = [1 - male_lo, male_hi]
    x = np.arange(2)
    ax.bar(x - bar_w / 2, real_v, bar_w, color=REAL_C, label="Real", zorder=3)
    ax.bar(x + bar_w / 2, syn_v,  bar_w, color=SYN_C,  label=f"Synthetic (n={n_worlds} worlds)", zorder=3)
    ax.errorbar(x + bar_w / 2, syn_v,
                yerr=[np.maximum(0, np.array(syn_v) - np.array(syn_lo)),
                      np.maximum(0, np.array(syn_hi) - np.array(syn_v))],
                fmt="none", color="#333", capsize=4, lw=1.5, zorder=5)
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_ylabel("Proportion"); ax.set_ylim(0, 1)
    ax.set_title("A   Sex Distribution", pad=6)
    ax.legend(frameon=False)

    # Panel B — Severity
    ax = axes[1]
    sev_labels = [f"Sev. {s}" for s in range(5)]
    real_sv = [np.mean(real["severity"] == s) for s in range(5)]
    syn_sv  = [np.mean(sev_props[s]) for s in range(5)]
    syn_slo = [np.percentile(sev_props[s], 2.5)  for s in range(5)]
    syn_shi = [np.percentile(sev_props[s], 97.5) for s in range(5)]
    x = np.arange(5)
    ax.bar(x - bar_w / 2, real_sv, bar_w, color=REAL_C, label="Real", zorder=3)
    ax.bar(x + bar_w / 2, syn_sv,  bar_w, color=SYN_C,  label=f"Synthetic (n={n_worlds} worlds)", zorder=3)
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
                                n_worlds: int = 30, n_vars: int = 42):
    lcols  = lipid_cols(real)
    groups = factor_groups(lcols)
    n_per  = max(1, n_vars // len(groups))
    sel    = []
    for cols in groups.values():
        sel.extend([c for c in cols if c in lcols][:n_per])
    sel = sel[:n_vars]

    real_corr = real[sel].corr().values

    stack = []
    for wp in sample_worlds(wpaths, n_worlds):
        world = load_world(wp, cols=sel)
        available = [c for c in sel if c in world.columns]
        if len(available) < 5:
            continue
        wc = world[available].sample(min(2000, len(world)), random_state=42)
        stack.append(wc.corr().values)

    syn_corr_mean = np.nanmean(stack, axis=0) if stack else np.zeros_like(real_corr)
    diff_mat      = real_corr - syn_corr_mean

    labels = [short_label(c, 14) for c in sel]

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.5))
    hkw = dict(xticklabels=labels, yticklabels=labels, linewidths=0,
               square=True, cbar_kws={"shrink": 0.65})

    sns.heatmap(real_corr,     ax=axes[0], vmin=-1, vmax=1, cmap="RdBu_r",
                cbar_kws={**hkw["cbar_kws"], "label": "Pearson r"}, **{k: v for k, v in hkw.items() if k != "cbar_kws"})
    axes[0].set_title("A   Real Data", pad=8)

    sns.heatmap(syn_corr_mean, ax=axes[1], vmin=-1, vmax=1, cmap="RdBu_r",
                cbar_kws={**hkw["cbar_kws"], "label": "Pearson r"}, **{k: v for k, v in hkw.items() if k != "cbar_kws"})
    axes[1].set_title(f"B   Synthetic (mean, n={n_worlds} worlds)", pad=8)

    sns.heatmap(diff_mat,      ax=axes[2], vmin=-0.4, vmax=0.4, cmap="RdBu_r",
                cbar_kws={**hkw["cbar_kws"], "label": "Real − Synthetic"}, **{k: v for k, v in hkw.items() if k != "cbar_kws"})
    axes[2].set_title("C   Difference (Real − Synthetic)", pad=8)

    for ax in axes:
        ax.tick_params(axis="x", rotation=90, labelsize=6)
        ax.tick_params(axis="y", rotation=0,  labelsize=6)

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
        world = load_world(wp)
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

    df_ks = pd.DataFrame(records)
    order = (df_ks.groupby("Lipid Group")["KS Statistic"]
             .median().sort_values().index.tolist())
    palette = {g: plt.cm.Set2(i / max(len(order) - 1, 1)) for i, g in enumerate(order)}

    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.violinplot(data=df_ks, x="Lipid Group", y="KS Statistic", order=order,
                   palette=palette, inner=None, cut=0, alpha=0.70, ax=ax, linewidth=1.2)
    sns.stripplot(data=df_ks, x="Lipid Group", y="KS Statistic", order=order,
                  palette=palette, size=3.5, alpha=0.55, jitter=True, ax=ax, zorder=3)

    # Median annotations
    for i, grp in enumerate(order):
        med = df_ks[df_ks["Lipid Group"] == grp]["KS Statistic"].median()
        ax.text(i, med + 0.005, f"{med:.3f}", ha="center", va="bottom", fontsize=7,
                fontweight="bold", color="#222")

    ax.axhline(0.05, color="#777", lw=1.3, ls="--", label="KS = 0.05 reference")
    ax.set_xlabel("Lipid Class Group")
    ax.set_ylabel(f"Mean KS Statistic per World (n ≤ {cols_per_group} lipids/group)")
    ax.set_title("KS Statistics by Lipid Class — Lower Is Better", pad=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    ax.legend(frameon=False)
    ax.text(0.98, 0.97,
            f"Each point = one synthetic world (n={n_worlds})",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7.5, color="#555", style="italic")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"fig4_ks_statistics.{ext}")
    plt.close(fig)
    print("  ✓ Fig 4 — KS statistics")


# ── Figure 5 — Propensity score / discriminative fidelity ─────────────────────
def fig5_propensity_score(real: pd.DataFrame, wpaths: list, fig_dir: Path,
                           n_worlds: int = 10, n_features: int = 50):
    lcols  = lipid_cols(real)
    groups = factor_groups(lcols)
    n_per  = max(1, n_features // len(groups))
    sel    = []
    for cols in groups.values():
        sel.extend([c for c in cols if c in lcols][:n_per])
    sel = sel[:n_features]

    real_X = real[sel].fillna(0).values
    real_y = np.zeros(len(real_X))

    sampled = sample_worlds(wpaths, n_worlds)
    aucs    = []

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_roc  = axes[0]
    ax_prop = axes[1]

    # Mean ROC accumulator
    base_fpr  = np.linspace(0, 1, 200)
    tpr_list  = []

    prop_real_all = []
    prop_syn_all  = []

    for i, wp in enumerate(sampled):
        world = load_world(wp, cols=sel)
        world = world[[c for c in sel if c in world.columns]].fillna(0)
        # equalise sizes
        n_common = min(len(real_X), len(world))
        rng = np.random.default_rng(i)
        ri  = rng.choice(len(real_X), n_common, replace=False)
        si  = rng.choice(len(world),  n_common, replace=False)
        X   = np.vstack([real_X[ri], world.values[si]])
        y   = np.concatenate([np.zeros(n_common), np.ones(n_common)])

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        clf = LogisticRegression(max_iter=500, C=0.1, random_state=42)
        clf.fit(Xs, y)
        prob = clf.predict_proba(Xs)[:, 1]

        fpr, tpr, _ = roc_curve(y, prob)
        auc = roc_auc_score(y, prob)
        aucs.append(auc)
        tpr_list.append(np.interp(base_fpr, fpr, tpr))

        # light individual curves
        ax_roc.plot(fpr, tpr, color=SYN_C, alpha=0.25, lw=0.8)

        prop_real_all.append(prob[y == 0])
        prop_syn_all.append(prob[y == 1])

    # Mean ROC
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
    ax_roc.text(0.97, 0.06,
                "AUC ≈ 0.50 → data indistinguishable",
                transform=ax_roc.transAxes, ha="right", va="bottom",
                fontsize=7.5, color="#555", style="italic")

    # Propensity density (pooled across worlds)
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
    ax_prop.text(0.5, 0.97,
                 "Ideal: overlapping distributions centred at 0.5",
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
                           n_worlds: int = 100):
    lcols    = lipid_cols(real)
    groups   = factor_groups(lcols)
    reps     = {}
    for grp, cols in groups.items():
        candidates = [c for c in cols if c in lcols]
        if candidates:
            reps[grp] = candidates[0]

    biomarkers = list(reps.values())

    # Real Spearman ρ with severity
    real_rho = {}
    real_pval = {}
    for c in biomarkers:
        r, p = spearmanr(real[c].fillna(0), real["severity"])
        real_rho[c]  = r
        real_pval[c] = p

    # Synthetic ρ distribution
    syn_rho = {c: [] for c in biomarkers}
    for wp in sample_worlds(wpaths, n_worlds, seed=7):
        needed = biomarkers + ["severity"]
        world  = load_world(wp, cols=[c for c in needed])
        world  = world.sample(min(500, len(world)), random_state=42)
        for c in biomarkers:
            if c not in world.columns:
                continue
            r, _ = spearmanr(world[c].fillna(0), world["severity"])
            syn_rho[c].append(r)

    # Draw forest plot (horizontal)
    n = len(biomarkers)
    fig, ax = plt.subplots(figsize=(9, n * 0.95 + 2))
    cmap = plt.cm.Set2(np.linspace(0, 1, n))

    for i, c in enumerate(biomarkers):
        grp = next(g for g, v in reps.items() if v == c)
        yr  = float(n - 1 - i)   # y position (top = first)

        # Synthetic CI bar
        sv = syn_rho[c]
        if sv:
            lo, hi = np.percentile(sv, [2.5, 97.5])
            mn = float(np.mean(sv))
            ax.hlines(yr, lo, hi, color=SYN_C, lw=2.5, alpha=0.8, zorder=3)
            ax.scatter(mn, yr, color=SYN_C, s=55, zorder=4,
                       label="Synthetic mean (95% CI)" if i == 0 else "")

        # Real diamond
        rv = real_rho[c]
        ax.scatter(rv, yr, color=REAL_C, marker="D", s=80, zorder=5,
                   label="Real data" if i == 0 else "")

        # Significance stars
        p = real_pval[c]
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        if star:
            ax.text(rv, yr + 0.32, star, ha="center", va="bottom",
                    fontsize=8, color=REAL_C, fontweight="bold")

        # Row label
        label = f"{grp}  —  {short_label(c, 20)}"
        ax.text(-1.07, yr, label, va="center", ha="left", fontsize=8)

    ax.axvline(0, color="#aaa", lw=1.2, ls="--", zorder=2)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-0.8, n - 0.2)
    ax.set_yticks([])
    ax.set_xlabel("Spearman ρ  (lipid vs. disease severity)")
    ax.set_title("Clinical Effect Preservation: Lipid–Severity Associations\nacross Real and Synthetic Data",
                 pad=8)
    ax.legend(frameon=False, loc="lower right")
    ax.text(0.99, 0.01,
            "Horizontal lines = 95% CI across worlds   ◆ = real estimate  * p<0.05  ** p<0.01  *** p<0.001",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color="#666", style="italic")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"fig6_clinical_effects.{ext}")
    plt.close(fig)
    print("  ✓ Fig 6 — Forest plot / clinical effects")


# ── Figure 7 — World variability (violin) ─────────────────────────────────────
def fig7_world_variability(real: pd.DataFrame, wpaths: list, fig_dir: Path,
                            n_worlds: int = 200):
    lcols  = lipid_cols(real)
    groups = factor_groups(lcols)
    reps   = {}
    for grp, cols in groups.items():
        candidates = [c for c in cols if c in lcols]
        if candidates:
            reps[grp] = candidates[0]

    # Define summary statistics — store as (display_name, fn_of_world_df)
    stat_defs = [
        ("Male prevalence (%)",   lambda w: w["sex"].mean() * 100 if "sex" in w.columns else np.nan),
        ("Mean severity",         lambda w: w["severity"].mean() if "severity" in w.columns else np.nan),
    ]
    for grp, col in reps.items():
        # capture col in default arg to avoid closure issues
        stat_defs.append((f"Mean {grp}\n({short_label(col, 14)})",
                          (lambda c: lambda w: w[c].mean() if c in w.columns else np.nan)(col)))

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

        vp = ax.violinplot(vals, positions=[0], showmedians=True, showextrema=True, widths=0.6)
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

        # percentile annotation
        if vals:
            med = np.median(vals)
            ax.text(0.5, 0.02, f"Median: {med:.2f}",
                    transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=7, color=SYN_C)

        ax.set_xticks([0])
        ax.set_xticklabels(["Synthetic\nworlds"], fontsize=8)
        ax.set_title(name, fontsize=8.5, pad=4)

    for j in range(n_stats, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"World Variability: Summary Statistics Across {n_worlds} Synthetic Worlds",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"fig7_world_variability.{ext}")
    plt.close(fig)
    print("  ✓ Fig 7 — World variability (violin)")


# ── Smoke test data ───────────────────────────────────────────────────────────
def make_smoke_data(tmp_dir: Path, n_worlds: int = 5, n_samples: int = 300):
    """Fake data with the same column schema as ALD lipidomics."""
    rng = np.random.default_rng(42)
    prefixes = {
        "LPC_": 8, "1_acyl_LPC_": 4, "PC_": 12, "PE_": 8,
        "PE_O_": 4, "Cer_d18:1/": 4, "SM_d18:1/": 4, "TG_": 6, "CE_": 4,
    }
    lipid_names = []
    for pref, cnt in prefixes.items():
        for j in range(cnt):
            lipid_names.append(f"{pref}{16 + j}:0")

    all_cols = META_COLS + lipid_names

    real_n = 184
    real_df = pd.DataFrame(
        rng.standard_normal((real_n, len(lipid_names))), columns=lipid_names
    )
    real_df.insert(0, "Sample_ID", [f"real_{i}" for i in range(real_n)])
    real_df.insert(1, "sex",      rng.integers(0, 2, real_n))
    real_df.insert(2, "severity", rng.integers(0, 5, real_n))

    real_path = tmp_dir / "data" / "ALD_lipidomics_merged.csv"
    real_path.parent.mkdir(parents=True, exist_ok=True)
    real_df.to_csv(real_path, index=False)

    syn_dir = tmp_dir / "synthetic_data" / "causal_knowledge_v5"
    syn_dir.mkdir(parents=True, exist_ok=True)
    for w in range(1, n_worlds + 1):
        sdf = pd.DataFrame(
            rng.standard_normal((n_samples, len(lipid_names))), columns=lipid_names
        )
        sdf.insert(0, "Sample_ID", [f"SYN_{k:04d}" for k in range(n_samples)])
        sdf.insert(1, "sex",      rng.integers(0, 2, n_samples))
        sdf.insert(2, "severity", rng.integers(0, 5, n_samples))
        sdf.to_csv(syn_dir / f"ALD_world_{w:03d}.csv", index=False)

    return real_path, syn_dir


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ALD Synthetic Data Validation")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Quick run on fake data to verify the code")
    parser.add_argument("--figs", nargs="+", type=int, default=list(range(1, 8)),
                        help="Which figures to generate, e.g. --figs 1 3 5 (default: all)")
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

    print(f"Loading real data …")
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
                       n_worlds=3 if smoke else 10),
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

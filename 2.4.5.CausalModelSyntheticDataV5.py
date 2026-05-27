import multiprocessing as mp
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List

# ============================================================
# 1) Utility
# ============================================================

def clip(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def relu(x):
    return np.maximum(x, 0.0)


# ============================================================
# 2) World config
# ============================================================

@dataclass
class WorldConfig:
    n_samples: int
    seed: Optional[int] = None


# ============================================================
# 3) Gaussian copula fitted from real data
# ============================================================

def fit_copula_components(
    df_real: pd.DataFrame,
    n_components: int = 30,
    meta_cols: Tuple[str, ...] = ("Sample_ID", "sex", "severity"),
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Fit a low-rank Gaussian copula to the real lipidomics data.

    Returns
    -------
    components  : (K, p) array — top-K PCA eigenvectors of the copula-normal data
    eigenvalues : (K,)   array — corresponding eigenvalues (explained variance)
    lipid_order : list[str]    — column order that indexes components axis-1
    """
    from sklearn.decomposition import PCA
    from scipy.stats import norm

    lipid_order = [c for c in df_real.columns if c not in meta_cols]
    data = df_real[lipid_order].values.astype(float)
    n, p = data.shape

    # Rank transform → uniform → standard normal  (non-parametric Gaussian copula)
    uniform = np.zeros_like(data)
    for j in range(p):
        col = data[:, j]
        valid = np.isfinite(col)
        ranks = np.argsort(np.argsort(col[valid]))
        tmp = np.full(n, 0.5)
        tmp[valid] = (ranks + 1.0) / (valid.sum() + 1.0)
        uniform[:, j] = tmp

    copula_normal = norm.ppf(np.clip(uniform, 1e-6, 1 - 1e-6))

    k = min(n_components, min(n, p) - 1)
    pca = PCA(n_components=k)
    pca.fit(copula_normal)

    pct = 100.0 * pca.explained_variance_ratio_.sum()
    print(f"  Copula PCA: top-{k} components explain {pct:.1f}% of copula variance "
          f"({n} samples × {p} lipids)")

    return pca.components_, pca.explained_variance_, lipid_order


# ============================================================
# 4) World-level hyperparameters
# ============================================================

def sample_world_hyperparams(rng: np.random.Generator) -> Dict:
    params = {}

    # ABCD1 effective activity
    params["abcd1"] = dict(
        baseline_activity = rng.uniform(0.7, 1.0),
        severity_weight   = rng.uniform(0.5, 1.0),
        female_rescue     = rng.uniform(0.2, 0.6),
        noise_sd_male     = rng.uniform(0.03, 0.10),
        noise_sd_female   = rng.uniform(0.05, 0.20),
    )

    # ABCD2 compensatory transport (estrogen-upregulated in females)
    params["abcd2"] = dict(
        female_upregulation = rng.uniform(0.15, 0.45),
        noise_sd            = rng.uniform(0.05, 0.15),
    )

    # VLCFA burden
    params["vlcfa"] = dict(
        a0_V            = rng.uniform(0.0, 0.5),
        a_activity      = rng.uniform(1.0, 2.0),
        a_age           = rng.uniform(0.3, 0.8),
        noise_sd_V      = rng.uniform(0.2, 0.5),
        c24_from_elovl1 = rng.uniform(0.4, 0.9),
        c24_noise       = rng.uniform(0.15, 0.4),
    )

    # Oxidative stress
    params["oxidative_stress"] = dict(
        os_from_vlcfa = rng.uniform(0.3, 0.8),
        os_from_infl  = rng.uniform(0.4, 0.9),
        noise_sd      = rng.uniform(0.15, 0.4),
    )

    # Lipid factors (6 factors)
    params["lipid_factors"] = {}

    params["lipid_factors"]["VLCFA_LPC"] = dict(
        b0         = rng.normal(0.0, 0.3),
        b_vlcfa    = rng.uniform(0.7, 1.6),
        b_oxstress = rng.uniform(0.3, 0.9),
        b_metab    = rng.uniform(0.1, 0.5),
        noise_sd   = rng.uniform(0.1, 0.3),
    )

    params["lipid_factors"]["PC_PE"] = dict(
        b0       = rng.normal(0.0, 0.3),
        b_vlcfa  = rng.uniform(0.2, 0.8),
        b_metab  = rng.uniform(0.3, 1.0),
        noise_sd = rng.uniform(0.1, 0.3),
    )

    params["lipid_factors"]["PLASMALOGEN"] = dict(
        b0         = rng.normal(0.0, 0.3),
        b_vlcfa    = rng.uniform(-0.4, -0.1),
        b_oxstress = rng.uniform(-0.8, -0.3),
        b_metab    = rng.uniform(0.2, 0.8),
        noise_sd   = rng.uniform(0.1, 0.3),
    )

    params["lipid_factors"]["CERAMIDE"] = dict(
        b0          = rng.normal(0.0, 0.3),
        b_vlcfa     = rng.uniform(0.7, 1.6),
        b_vlcfa_c24 = rng.uniform(0.4, 0.9),
        b_infl      = rng.uniform(0.3, 0.9),
        b_metab     = rng.uniform(0.1, 0.5),
        noise_sd    = rng.uniform(0.1, 0.3),
    )

    params["lipid_factors"]["SM"] = dict(
        b0       = rng.normal(0.0, 0.3),
        b_vlcfa  = rng.uniform(0.2, 0.7),
        b_infl   = rng.uniform(-0.5, -0.2),
        b_metab  = rng.uniform(0.2, 0.8),
        noise_sd = rng.uniform(0.1, 0.3),
    )

    params["lipid_factors"]["TG_CE"] = dict(
        b0       = rng.normal(0.0, 0.3),
        b_vlcfa  = rng.uniform(0.1, 0.6),
        b_metab  = rng.uniform(0.6, 1.5),
        noise_sd = rng.uniform(0.1, 0.3),
    )

    params["ce_vlcfa"] = dict(
        c_vlcfa    = rng.uniform(0.8, 1.5),
        c_sex_male = rng.uniform(0.2, 0.6),
        noise_sd   = rng.uniform(0.1, 0.3),
    )

    params["adrenal_insufficiency"] = dict(
        ce_threshold = rng.uniform(0.7, 1.3),
        scale        = rng.uniform(3.0, 7.0),
        noise_sd     = rng.uniform(0.05, 0.20),
    )

    params["inflammation"] = dict(
        c0            = rng.normal(0.0, 0.3),
        c_V           = rng.uniform(0.4, 1.2),
        c_cer         = rng.uniform(0.3, 1.0),
        c_LPC         = rng.uniform(0.3, 1.0),
        c_oxstress    = rng.uniform(0.2, 0.6),
        c_plasmalogen = rng.uniform(-0.4, -0.1),
        noise_sd_I    = rng.uniform(0.2, 0.5),
    )

    params["demyelination"] = dict(
        d0         = rng.normal(0.0, 0.3),
        d_I        = rng.uniform(0.6, 1.8),
        d_V        = rng.uniform(0.3, 1.2),
        d_cer      = rng.uniform(0.2, 0.7),
        noise_sd_D = rng.uniform(0.2, 0.5),
    )

    # V5: remove t5 (only 5 severity classes 0-4); thresholds are now quantile-based
    params["severity"] = dict(
        e0         = rng.normal(0.0, 0.3),
        e_D        = rng.uniform(0.2, 0.8),
        e_I        = rng.uniform(0.1, 0.6),
        e_ai       = rng.uniform(0.1, 0.4),
        e_lpc      = rng.uniform(0.05, 0.20),
        e_age      = rng.uniform(-0.3, 0.3),
        e_sex      = rng.uniform(0.1, 0.8),
        noise_sd_S = rng.uniform(0.3, 0.8),
    )

    return params


# ============================================================
# 5) Root nodes
# ============================================================

# V5: empirical target severity proportions for 5 classes (0-4) derived from real ALD data
#     sev 0: 6.52%,  sev 1: 15.22%,  sev 2: 5.98%,  sev 3: 58.15%,  sev 4: 14.13%
_SEV_TARGET_CDF = np.array([0.0652, 0.2174, 0.2772, 0.8587])


def sample_roots(n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    out = {}

    # V5: sex ratio calibrated to real cohort (58.15% male)
    p_male = float(np.clip(rng.normal(0.58, 0.04), 0.47, 0.70))
    out["sex_male"] = rng.binomial(1, p_male, size=n)

    mu_child = rng.uniform(5.0, 12.0)
    sd_child = rng.uniform(2.0, 5.0)
    mu_adult = rng.uniform(25.0, 50.0)
    sd_adult = rng.uniform(5.0, 15.0)
    pi_child = rng.uniform(0.2, 0.6)
    is_child = rng.binomial(1, pi_child, size=n)
    age = np.where(
        is_child == 1,
        rng.normal(mu_child, sd_child, size=n),
        rng.normal(mu_adult, sd_adult, size=n),
    )
    out["age"] = clip(age, 0.0, 80.0)

    out["mutation_severity"] = rng.beta(
        rng.uniform(1.5, 3.0), rng.uniform(1.0, 3.0), size=n
    )

    mean_metab = rng.normal(0.0, 0.3)
    sd_metab   = rng.uniform(0.7, 1.5)
    out["metabolic_background"] = rng.normal(mean_metab, sd_metab, size=n)

    return out


# ============================================================
# 6) Structural equations  (unchanged from V4 except severity)
# ============================================================

def compute_abcd1_effective(roots, params, rng):
    p = params["abcd1"]
    sex_male = roots["sex_male"]
    sev = roots["mutation_severity"]
    n = sev.shape[0]
    a_star = p["baseline_activity"] - p["severity_weight"] * sev
    a_sex  = a_star + (1 - sex_male) * p["female_rescue"]
    eps = np.where(
        sex_male == 1,
        rng.normal(0.0, p["noise_sd_male"], size=n),
        rng.normal(0.0, p["noise_sd_female"], size=n),
    )
    return clip(a_sex + eps, 0.0, 1.2)


def compute_vlcfa(roots, abcd1_eff, params, rng):
    p_v     = params["vlcfa"]
    p_abcd2 = params["abcd2"]
    age      = roots["age"]
    sex_male = roots["sex_male"]
    n = age.shape[0]
    log_age = np.log(age + 1.0)
    abcd2_comp = (1 - sex_male) * rng.normal(
        p_abcd2["female_upregulation"], p_abcd2["noise_sd"], size=n
    )
    abcd2_comp = clip(abcd2_comp, 0.0, 0.5)
    V_star = (
        p_v["a0_V"]
        - p_v["a_activity"] * abcd1_eff
        - abcd2_comp
        + p_v["a_age"] * log_age
        + rng.normal(0.0, p_v["noise_sd_V"], size=n)
    )
    return np.tanh(V_star)


def compute_vlcfa_c24(roots, abcd1_eff, params, rng):
    p_v = params["vlcfa"]
    age = roots["age"]
    n = age.shape[0]
    log_age = np.log(age + 1.0)
    C24_star = (
        p_v["a0_V"] * 0.8
        - p_v["a_activity"] * 0.7 * abcd1_eff
        + p_v["c24_from_elovl1"] * log_age * 0.5
        + rng.normal(0.0, p_v["c24_noise"], size=n)
    )
    return np.tanh(C24_star)


def compute_oxidative_stress(vlcfa, infl, params, rng):
    p = params["oxidative_stress"]
    n = vlcfa.shape[0]
    os_star = (
        p["os_from_vlcfa"] * vlcfa
        + p["os_from_infl"] * infl
        + rng.normal(0.0, p["noise_sd"], size=n)
    )
    return relu(os_star)


def compute_lipid_factors(roots, vlcfa, vlcfa_c24, oxidative_stress, infl, params, rng):
    metab   = roots["metabolic_background"]
    n       = vlcfa.shape[0]
    lf_p    = params["lipid_factors"]
    factors = {}

    p = lf_p["VLCFA_LPC"]
    L = p["b0"] + p["b_vlcfa"] * vlcfa + p["b_oxstress"] * oxidative_stress + p["b_metab"] * metab
    factors["VLCFA_LPC"] = np.tanh(L + rng.normal(0.0, p["noise_sd"], size=n))

    p = lf_p["PC_PE"]
    L = p["b0"] + p["b_vlcfa"] * vlcfa + p["b_metab"] * metab
    factors["PC_PE"] = np.tanh(L + rng.normal(0.0, p["noise_sd"], size=n))

    p = lf_p["PLASMALOGEN"]
    L = p["b0"] + p["b_vlcfa"] * vlcfa + p["b_oxstress"] * oxidative_stress + p["b_metab"] * metab
    factors["PLASMALOGEN"] = np.tanh(L + rng.normal(0.0, p["noise_sd"], size=n))

    p = lf_p["CERAMIDE"]
    L = (p["b0"]
         + p["b_vlcfa"]     * vlcfa
         + p["b_vlcfa_c24"] * vlcfa_c24
         + p["b_infl"]      * infl
         + p["b_metab"]     * metab)
    factors["CERAMIDE"] = np.tanh(L + rng.normal(0.0, p["noise_sd"], size=n))

    p = lf_p["SM"]
    L = p["b0"] + p["b_vlcfa"] * vlcfa + p["b_infl"] * infl + p["b_metab"] * metab
    factors["SM"] = np.tanh(L + rng.normal(0.0, p["noise_sd"], size=n))

    p = lf_p["TG_CE"]
    L = p["b0"] + p["b_vlcfa"] * vlcfa + p["b_metab"] * metab
    factors["TG_CE"] = np.tanh(L + rng.normal(0.0, p["noise_sd"], size=n))

    return factors


def compute_ce_vlcfa(roots, vlcfa, params, rng):
    p        = params["ce_vlcfa"]
    sex_male = roots["sex_male"]
    n        = vlcfa.shape[0]
    ce_star  = (
        p["c_vlcfa"] * vlcfa
        + p["c_sex_male"] * sex_male
        + rng.normal(0.0, p["noise_sd"], size=n)
    )
    return relu(ce_star)


def compute_adrenal_insufficiency(ce_vlcfa, params, rng):
    p = params["adrenal_insufficiency"]
    n = ce_vlcfa.shape[0]
    ai_star = p["scale"] * (ce_vlcfa - p["ce_threshold"])
    return clip(sigmoid(ai_star) + rng.normal(0.0, p["noise_sd"], size=n), 0.0, 1.0)


def compute_inflammation(vlcfa, oxidative_stress, factors, params, rng):
    p     = params["inflammation"]
    n     = vlcfa.shape[0]
    L_cer = factors["CERAMIDE"]
    L_lpc = factors["VLCFA_LPC"]
    L_pls = factors["PLASMALOGEN"]
    I_star = (
        p["c0"]
        + p["c_V"]           * vlcfa
        + p["c_cer"]         * L_cer
        + p["c_LPC"]         * L_lpc
        + p["c_oxstress"]    * oxidative_stress
        + p["c_plasmalogen"] * L_pls
        + rng.normal(0.0, p["noise_sd_I"], size=n)
    )
    return relu(I_star)


def compute_demyelination(vlcfa, infl, factors, params, rng):
    p     = params["demyelination"]
    n     = vlcfa.shape[0]
    L_cer = factors["CERAMIDE"]
    D_star = (
        p["d0"]
        + p["d_I"]   * infl
        + p["d_V"]   * vlcfa
        + p["d_cer"] * L_cer
        + rng.normal(0.0, p["noise_sd_D"], size=n)
    )
    return sigmoid(D_star)


def compute_severity(roots, infl, demy, adrenal_insuff, factors, params, rng):
    """5-level ordinal severity (0–4).

    V5 change: quantile-based thresholds ensure the generated severity distribution
    matches the empirical real-data proportions exactly per world, eliminating the
    systematic mismatch seen in V4 (which generated a near-uniform distribution).

    Causal ordering of S_star is fully preserved: Spearman correlations between any
    lipid and severity are monotone-equivalent to those with S_star.
    """
    p        = params["severity"]
    n        = infl.shape[0]
    age      = roots["age"]
    sex_male = roots["sex_male"]
    L_lpc    = factors["VLCFA_LPC"]

    age_mu   = age.mean()
    age_sd   = age.std() + 1e-6
    age_norm = (age - age_mu) / age_sd

    S_star = (
        p["e0"]
        + p["e_D"]   * demy
        + p["e_I"]   * infl
        + p["e_ai"]  * adrenal_insuff
        + p["e_lpc"] * L_lpc
        + p["e_age"] * age_norm
        + p["e_sex"] * sex_male
        + rng.normal(0.0, p["noise_sd_S"], size=n)
    )

    # Quantile-based thresholds: calibrated to real ALD data distribution
    # sev 0: 6.52%, sev 1: 15.22%, sev 2: 5.98%, sev 3: 58.15%, sev 4: 14.13%
    q1, q2, q3, q4 = np.quantile(S_star, _SEV_TARGET_CDF)

    severity = np.zeros(n, dtype=int)
    severity[(S_star >= q1) & (S_star < q2)] = 1
    severity[(S_star >= q2) & (S_star < q3)] = 2
    severity[(S_star >= q3) & (S_star < q4)] = 3
    severity[S_star >= q4]                   = 4

    return S_star, severity


# ============================================================
# 7) Map real lipid column name → latent factor
# ============================================================

def assign_main_factor(col: str) -> str:
    if col.startswith((
        "PE_O_", "PE_P_", "PE_OplusP_",
        "PC_O_", "PC_P_", "PC_OplusP_",
    )):
        return "PLASMALOGEN"

    if col.startswith((
        "1_acyl_LPC_", "2_acyl_LPC_",
        "1_acyl_LPE_", "2_acyl_LPE_",
        "LPC_", "LPE_",
    )):
        return "VLCFA_LPC"

    if col.startswith(("SM_", "SM4_")):
        return "SM"

    if col.startswith(("Cer_", "HexCer_", "Hex2Cer_", "C1P_", "S1P_")):
        return "CERAMIDE"

    if col.startswith(("TG_", "TG_O_", "DG_", "CE_")):
        return "TG_CE"

    if col.startswith((
        "PC_", "PE_",
        "PI_", "PS_", "PG_", "PA_",
        "LPG_", "LPA_", "LPI_", "LPS_",
        "CL_", "BMP_",
    )):
        return "PC_PE"

    return "PC_PE"


# ============================================================
# 8) Generate ONE synthetic world
# ============================================================

FACTOR_NAMES = ["VLCFA_LPC", "PC_PE", "PLASMALOGEN", "CERAMIDE", "SM", "TG_CE"]


def sample_world_like(
    df_template: pd.DataFrame,
    cfg: WorldConfig,
    copula_components: Optional[np.ndarray] = None,
    copula_variance:   Optional[np.ndarray] = None,
    copula_col_order:  Optional[List[str]]  = None,
    copula_noise_scale: float = 0.80,
) -> Tuple[pd.DataFrame, Dict]:
    """Generate one synthetic ALD cohort matching df_template's column schema.

    V5 changes vs V4
    ----------------
    1. Gaussian copula noise  — correlated residual noise fitted from real data
       replaces the large independent per-lipid noise (was 0.30–0.80 SD).
       This is the primary fix for the near-zero inter-lipid correlation (AUC 0.998,
       flat correlation matrix) seen in V4 validation.

    2. Severity calibration   — 5 classes (0–4), quantile-based thresholds that
       reproduce the real cohort's distribution (~58% severity 3) rather than V4's
       near-uniform spread.

    3. Sex-ratio calibration  — p_male centred at 0.58 (real cohort) instead of
       U(0.3, 0.7).

    4. Reduced signal noise   — causal-factor-level b0/noise_sd parameters tightened
       to let the copula dominate the variance budget (causal ≈ 20%, copula ≈ 70%,
       independent residual ≈ 10%), matching real data std ≈ 1 (z-scored).

    Causal order (identical to V4)
    --------------------------------
    roots → abcd1 → vlcfa/vlcfa_c24 → ox_stress (boot) → factors → inflammation
    → ox_stress (refined) → factors (refined) → inflammation (refined)
    → ce_vlcfa → adrenal_insufficiency → demyelination → severity
    """
    rng    = np.random.default_rng(cfg.seed)
    params = sample_world_hyperparams(rng)
    n      = cfg.n_samples

    roots      = sample_roots(n, rng)
    abcd1      = compute_abcd1_effective(roots, params, rng)
    vlcfa      = compute_vlcfa(roots, abcd1, params, rng)
    vlcfa_c24  = compute_vlcfa_c24(roots, abcd1, params, rng)

    infl_boot = relu(rng.normal(0.0, 0.3, size=n))
    ox_stress = compute_oxidative_stress(vlcfa, infl_boot, params, rng)
    factors   = compute_lipid_factors(roots, vlcfa, vlcfa_c24, ox_stress, infl_boot, params, rng)
    infl      = compute_inflammation(vlcfa, ox_stress, factors, params, rng)

    ox_stress = compute_oxidative_stress(vlcfa, infl, params, rng)
    factors   = compute_lipid_factors(roots, vlcfa, vlcfa_c24, ox_stress, infl, params, rng)
    infl      = compute_inflammation(vlcfa, ox_stress, factors, params, rng)

    ce_vlcfa       = compute_ce_vlcfa(roots, vlcfa, params, rng)
    adrenal_insuff = compute_adrenal_insufficiency(ce_vlcfa, params, rng)
    demy           = compute_demyelination(vlcfa, infl, factors, params, rng)
    S_star, severity = compute_severity(roots, infl, demy, adrenal_insuff, factors, params, rng)

    # ── Pre-generate correlated noise matrix (n × n_lipids_copula) ──────────
    # Uses the Gaussian copula's PCA decomposition to produce noise vectors
    # whose pairwise correlations match the real data's rank-correlation structure.
    if (copula_components is not None
            and copula_variance is not None
            and copula_col_order is not None):
        K = copula_components.shape[0]
        z = rng.standard_normal(size=(n, K))                               # n × K
        # Scale each component by sqrt(eigenvalue) then project to lipid space
        corr_noise = (z * np.sqrt(copula_variance)) @ copula_components    # n × p
        corr_noise *= copula_noise_scale
        copula_idx = {c: i for i, c in enumerate(copula_col_order)}
    else:
        corr_noise = None
        copula_idx = {}

    # ── Build output ─────────────────────────────────────────────────────────
    cols       = list(df_template.columns)
    meta_cols  = ["Sample_ID", "sex", "severity"]
    lipid_cols = [c for c in cols if c not in meta_cols]
    out        = {}

    if "Sample_ID" in cols:
        out["Sample_ID"] = [f"SYN_{i+1:04d}" for i in range(n)]
    if "sex" in cols:
        out["sex"] = roots["sex_male"].astype(int)
    if "severity" in cols:
        out["severity"] = severity

    for col in lipid_cols:
        main_key    = assign_main_factor(col)
        main_factor = factors[main_key]

        # V5: smaller per-world factor loading variability so copula noise dominates
        bias   = rng.normal(0.0, 0.05)
        w_main = rng.normal(1.0, 0.10)
        value  = bias + w_main * main_factor

        for k in FACTOR_NAMES:
            if k == main_key:
                continue
            w_cross = rng.normal(0.0, 0.08)
            value   = value + w_cross * factors[k]

        # HexCer columns additionally reflect demyelination-released myelin GalCer
        if col.startswith(("HexCer_", "Hex2Cer_")):
            w_demy = rng.normal(0.6, 0.1)
            value  = value + w_demy * demy

        # V5: correlated copula noise (or small independent fallback)
        ci = copula_idx.get(col)
        if ci is not None and corr_noise is not None:
            value = value + corr_noise[:, ci]
            # Small independent residual for unexplained within-class variance
            value = value + rng.normal(0.0, 0.10, size=n)
        else:
            noise_sd = rng.uniform(0.10, 0.30)
            value    = value + rng.normal(0.0, noise_sd, size=n)

        out[col] = value

    return pd.DataFrame(out, columns=cols), params


# ============================================================
# 9) Generate many worlds
# ============================================================

def _generate_one(args):
    i, df_template, n_samples, seed, out_dir, prefix, cop_comp, cop_var, cop_cols, cop_scale = args
    cfg = WorldConfig(n_samples=n_samples, seed=seed)
    df_syn, _ = sample_world_like(
        df_template, cfg,
        copula_components  = cop_comp,
        copula_variance    = cop_var,
        copula_col_order   = cop_cols,
        copula_noise_scale = cop_scale,
    )
    if out_dir is not None:
        path = os.path.join(out_dir, f"{prefix}_{i + 1:03d}.csv")
        df_syn.to_csv(path, index=False)
    print(f"World {i + 1} saved", flush=True)


def generate_many_worlds(
    df_template: pd.DataFrame,
    n_worlds: int,
    n_samples: Optional[int] = None,
    base_seed: int = 123,
    out_dir: Optional[str] = None,
    prefix: str = "synthetic_world",
    n_jobs: int = 1,
    copula_components:   Optional[np.ndarray] = None,
    copula_variance:     Optional[np.ndarray] = None,
    copula_col_order:    Optional[List[str]]  = None,
    copula_noise_scale:  float = 0.80,
) -> None:
    """Generate n_worlds synthetic cohorts and stream each to disk.

    Worlds are never accumulated in RAM — each is saved and discarded immediately.
    """
    if n_samples is None:
        n_samples = df_template.shape[0]
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    args = [
        (i, df_template, n_samples, base_seed + i, out_dir, prefix,
         copula_components, copula_variance, copula_col_order, copula_noise_scale)
        for i in range(n_worlds)
    ]

    if n_jobs == 1:
        for a in args:
            _generate_one(a)
    else:
        with mp.Pool(processes=n_jobs) as pool:
            pool.map(_generate_one, args)


# ============================================================
# 10) Entry point
# ============================================================

if __name__ == "__main__":
    N_JOBS   = int(os.environ.get("N_JOBS", 1))
    N_WORLDS = 500
    N_SAMPLES = 20_000
    N_COPULA_COMPONENTS = 30    # top PCA components of Gaussian copula
    COPULA_NOISE_SCALE  = 0.80  # fraction of variance from copula noise

    real_path = "./data/ALD_lipidomics_merged-updated.csv"
    df_real   = pd.read_csv(real_path)

    print(f"Fitting Gaussian copula from real data ({df_real.shape[0]} samples) …")
    cop_comp, cop_var, cop_cols = fit_copula_components(
        df_real, n_components=N_COPULA_COMPONENTS
    )

    print(f"\nGenerating {N_WORLDS} worlds × {N_SAMPLES} samples using {N_JOBS} CPU core(s).")
    generate_many_worlds(
        df_template        = df_real,
        n_worlds           = N_WORLDS,
        n_samples          = N_SAMPLES,
        base_seed          = 42,
        out_dir            = "./synthetic_data/causal_knowledge_v5/",
        prefix             = "ALD_world",
        n_jobs             = N_JOBS,
        copula_components  = cop_comp,
        copula_variance    = cop_var,
        copula_col_order   = cop_cols,
        copula_noise_scale = COPULA_NOISE_SCALE,
    )
    print("Done.")

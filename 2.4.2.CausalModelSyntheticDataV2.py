import numpy as np
import pandas as pd
from dataclasses import dataclass
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
# 3) World-level hyperparameters (full Option 2 SCM)
# ============================================================

def sample_world_hyperparams(rng: np.random.Generator) -> Dict:
    params = {}

    # ABCD1_effective_activity
    params["abcd1"] = dict(
        baseline_activity = rng.uniform(0.7, 1.0),
        severity_weight   = rng.uniform(0.5, 1.0),
        female_rescue     = rng.uniform(0.2, 0.6),
        noise_sd_male     = rng.uniform(0.03, 0.10),
        noise_sd_female   = rng.uniform(0.05, 0.20),
    )

    # VLCFA_burden
    params["vlcfa"] = dict(
        a0_V       = rng.uniform(0.0, 0.5),
        a_activity = rng.uniform(1.0, 2.0),
        a_age      = rng.uniform(0.3, 0.8),
        noise_sd_V = rng.uniform(0.2, 0.5),
    )

    # Lipid factors (VLCFA_LPC, PC_PE, CER_SM, TG_CE)
    params["lipid_factors"] = {}

    params["lipid_factors"]["VLCFA_LPC"] = dict(
        b0       = rng.normal(0.0, 0.5),
        b1       = rng.uniform(0.8, 1.8),
        b2       = rng.uniform(0.2, 1.0),
        noise_sd = rng.uniform(0.1, 0.4),
    )
    params["lipid_factors"]["PC_PE"] = dict(
        b0       = rng.normal(0.0, 0.5),
        b1       = rng.uniform(0.3, 1.0),
        b2       = rng.uniform(0.3, 1.0),
        noise_sd = rng.uniform(0.1, 0.4),
    )
    params["lipid_factors"]["CER_SM"] = dict(
        b0       = rng.normal(0.0, 0.5),
        b1       = rng.uniform(0.7, 1.7),
        b2       = rng.uniform(0.1, 1.0),
        noise_sd = rng.uniform(0.1, 0.4),
    )
    params["lipid_factors"]["TG_CE"] = dict(
        b0       = rng.normal(0.0, 0.5),
        b1       = rng.uniform(0.1, 0.8),
        b2       = rng.uniform(0.7, 1.7),
        noise_sd = rng.uniform(0.1, 0.4),
    )

    # Inflammation
    params["inflammation"] = dict(
        c0         = rng.normal(0.0, 0.5),
        c_V        = rng.uniform(0.5, 1.5),
        c_cer      = rng.uniform(0.3, 1.2),
        c_LPC      = rng.uniform(0.3, 1.2),
        noise_sd_I = rng.uniform(0.2, 0.5),
    )

    # Demyelination
    params["demyelination"] = dict(
        d0         = rng.normal(0.0, 0.5),
        d_I        = rng.uniform(0.6, 1.8),
        d_V        = rng.uniform(0.3, 1.2),
        noise_sd_D = rng.uniform(0.2, 0.5),
    )

    # Severity (latent score + thresholds) → 6 levels
    params["severity"] = dict(
    	e0         = rng.normal(0.0, 0.5),

    	
    	e_D        = rng.uniform(0.2, 0.8),   
    	e_I        = rng.uniform(0.1, 0.6),   

    	e_age      = rng.uniform(-0.3, 0.3),  
    	e_sex      = rng.uniform(0.1, 0.8),   
    	noise_sd_S = rng.uniform(0.3, 0.8),

    	
    	t1 = rng.normal(-1.5, 0.3),
    	t2 = rng.normal(-0.5, 0.3),
    	t3 = rng.normal( 0.5, 0.3),
    	t4 = rng.normal( 1.5, 0.3),
    	t5 = rng.normal( 2.5, 0.3),
    )

    return params


# ============================================================
# 4) Root nodes (sex, age, mutation_severity, metabolic_background)
# ============================================================

def sample_roots(n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    out = {}

    # Sex (0=female, 1=male)
    p_male = rng.uniform(0.3, 0.7)
    sex_male = rng.binomial(1, p_male, size=n)
    out["sex_male"] = sex_male

    # Age mixture
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
    age = clip(age, 0.0, 80.0)
    out["age"] = age

    # Mutation severity
    alpha_mut = rng.uniform(1.5, 3.0)
    beta_mut  = rng.uniform(1.0, 3.0)
    mutation_severity = rng.beta(alpha_mut, beta_mut, size=n)
    out["mutation_severity"] = mutation_severity

    # Metabolic background
    mean_metab = rng.normal(0.0, 0.5)
    sd_metab   = rng.uniform(0.7, 1.5)
    metab_bg   = rng.normal(mean_metab, sd_metab, size=n)
    out["metabolic_background"] = metab_bg

    return out


# ============================================================
# 5) Structural equations
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
    abcd1_eff = clip(a_sex + eps, 0.0, 1.2)
    return abcd1_eff


def compute_vlcfa(roots, abcd1_eff, params, rng):
    p = params["vlcfa"]
    age = roots["age"]
    n = age.shape[0]

    log_age = np.log(age + 1.0)
    V_star = (
        p["a0_V"]
        - p["a_activity"] * abcd1_eff
        + p["a_age"] * log_age
        + rng.normal(0.0, p["noise_sd_V"], size=n)
    )
    vlcfa = np.tanh(V_star)
    return vlcfa


def compute_lipid_factors(roots, vlcfa, params, rng):
    metab = roots["metabolic_background"]
    n = vlcfa.shape[0]
    lf_params = params["lipid_factors"]

    factors = {}
    for key, hp in lf_params.items():
        L_star = (
            hp["b0"]
            + hp["b1"] * vlcfa
            + hp["b2"] * metab
            + rng.normal(0.0, hp["noise_sd"], size=n)
        )
        factors[key] = np.tanh(L_star)
    return factors


def compute_inflammation(vlcfa, factors, params, rng):
    p = params["inflammation"]
    n = vlcfa.shape[0]

    L_cer = factors["CER_SM"]
    L_lpc = factors["VLCFA_LPC"]

    I_star = (
        p["c0"]
        + p["c_V"] * vlcfa
        + p["c_cer"] * L_cer
        + p["c_LPC"] * L_lpc
        + rng.normal(0.0, p["noise_sd_I"], size=n)
    )
    infl = relu(I_star)
    return infl


def compute_demyelination(vlcfa, infl, params, rng):
    p = params["demyelination"]
    n = vlcfa.shape[0]

    D_star = (
        p["d0"]
        + p["d_I"] * infl
        + p["d_V"] * vlcfa
        + rng.normal(0.0, p["noise_sd_D"], size=n)
    )
    demy = sigmoid(D_star)
    return demy


# ============================================================
# 6) Updated severity equation (6 classes)
# ============================================================

def compute_severity(roots, infl, demy, params, rng):
    p = params["severity"]
    n = infl.shape[0]

    age = roots["age"]
    sex_male = roots["sex_male"]

    # Normalize age inside each world
    age_mu = age.mean()
    age_sd = age.std() + 1e-6
    age_norm = (age - age_mu) / age_sd

    # Latent severity score
    S_star = (
        p["e0"]
        + p["e_D"] * demy
        + p["e_I"] * infl
        + p["e_age"] * age_norm
        + p["e_sex"] * sex_male
        + rng.normal(0.0, p["noise_sd_S"], size=n)
    )

    # Five thresholds → 6-level ordinal severity
    t1, t2, t3, t4, t5 = p["t1"], p["t2"], p["t3"], p["t4"], p["t5"]

    severity = np.zeros(n, dtype=int)
    severity[(S_star >= t1) & (S_star < t2)] = 1
    severity[(S_star >= t2) & (S_star < t3)] = 2
    severity[(S_star >= t3) & (S_star < t4)] = 3
    severity[(S_star >= t4) & (S_star < t5)] = 4
    severity[S_star >= t5]                  = 5

    return S_star, severity


# ============================================================
# 7) Map real lipid name → latent factor
# ============================================================

def assign_main_factor(col: str) -> str:

    # Lysophospholipids → VLCFA_LPC factor
    if (
        col.startswith("1_acyl_LPC_")
        or col.startswith("2_acyl_LPC_")
        or col.startswith("1_acyl_LPE_")
        or col.startswith("2_acyl_LPE_")
        or col.startswith("LPC_")
        or col.startswith("LPE_")
    ):
        return "VLCFA_LPC"

    # Glycerophospholipids → PC_PE factor
    if col.startswith((
        "PC_", "PC_O_", "PC_OplusP_",
        "PE_", "PE_O_", "PE_OplusP_",
        "PI_", "PS_", "PG_", "PA_",
        "LPG_", "LPA_", "LPI_", "LPS_",
        "CL_", "BMP_"
    )):
        return "PC_PE"

    # Sphingolipids → CER_SM factor
    if col.startswith((
        "SM_", "SM4_",
        "Cer_", "HexCer_", "Hex2Cer_",
        "C1P_", "S1P_"
    )):
        return "CER_SM"

    # Neutral lipids → TG_CE factor
    if col.startswith(("TG_", "TG_O_", "DG_", "CE_")):
        return "TG_CE"

    return "PC_PE"


# ============================================================
# 8) Generate ONE synthetic world
# ============================================================

def sample_world_like(
    df_template: pd.DataFrame,
    cfg: WorldConfig,
) -> Tuple[pd.DataFrame, Dict]:

    rng = np.random.default_rng(cfg.seed)
    params = sample_world_hyperparams(rng)
    n = cfg.n_samples

    # Internal SCM
    roots   = sample_roots(n, rng)
    abcd1   = compute_abcd1_effective(roots, params, rng)
    vlcfa   = compute_vlcfa(roots, abcd1, params, rng)
    factors = compute_lipid_factors(roots, vlcfa, params, rng)
    infl    = compute_inflammation(vlcfa, factors, params, rng)
    demy    = compute_demyelination(vlcfa, infl, params, rng)
    S_star, severity = compute_severity(roots, infl, demy, params, rng)

    cols = list(df_template.columns)
    out = {}

    # Meta columns
    meta_cols = ["Sample_ID", "sex", "severity"]
    lipid_cols = [c for c in cols if c not in meta_cols]

    if "Sample_ID" in cols:
        out["Sample_ID"] = [f"SYN_{i+1:04d}" for i in range(n)]

    if "sex" in cols:
        out["sex"] = roots["sex_male"].astype(int)

    if "severity" in cols:
        out["severity"] = severity  # already 0–5

    # Lipid columns
    for col in lipid_cols:
        main_factor_key = assign_main_factor(col)
        main_factor = factors[main_factor_key]

        bias   = rng.normal(0.0, 0.3)
        w_main = rng.normal(1.0, 0.2)
        value  = bias + w_main * main_factor

        for k, f in factors.items():
            if k == main_factor_key:
                continue
            w_cross = rng.normal(0.0, 0.1)
            value   = value + w_cross * f

        noise_sd = rng.uniform(0.3, 0.8)
        value = value + rng.normal(0.0, noise_sd, size=n)

        out[col] = value

    df_syn = pd.DataFrame(out, columns=cols)
    return df_syn, params


# ============================================================
# 9) Generate MANY synthetic worlds
# ============================================================

def generate_many_worlds(
    df_template: pd.DataFrame,
    n_worlds: int,
    n_samples: Optional[int] = None,
    base_seed: int = 123,
    out_dir: Optional[str] = None,
    prefix: str = "synthetic_world",
) -> List[pd.DataFrame]:

    if n_samples is None:
        n_samples = df_template.shape[0]

    dfs = []
    for i in range(n_worlds):
        print(f"World {i+1}/{n_worlds} is generated!")

        seed = base_seed + i
        cfg = WorldConfig(n_samples=n_samples, seed=seed)
        df_syn, params = sample_world_like(df_template, cfg)
        dfs.append(df_syn)

        if out_dir is not None:
            path = f"{out_dir}/{prefix}_{i+1:03d}.csv"
            df_syn.to_csv(path, index=False)

    return dfs


# ============================================================
# 10) Example usage
# ============================================================

if __name__ == "__main__":
    real_path = "./data/ALD_lipidomics_merged.csv"
    df_real = pd.read_csv(real_path)

    # Generate a single synthetic world
    #cfg = WorldConfig(n_samples=df_real.shape[0], seed=42)
    #df_syn_one, world_params = sample_world_like(df_real, cfg)
    #print("One synthetic world:", df_syn_one.shape)

    # Generate many worlds
    worlds = generate_many_worlds(
        df_template=df_real,
        n_worlds=500,
        n_samples=20000,
        base_seed=42,
        out_dir="./synthetic_data/causal_knowledge/",
        prefix="ALD_world"
    )
    print("Generated", len(worlds), "worlds.")

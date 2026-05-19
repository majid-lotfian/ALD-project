import multiprocessing as mp
import os
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
# 3) World-level hyperparameters
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
    # Ref: Fourcade et al., Hum Mol Genet 2003 (PMID 12915447)
    params["abcd2"] = dict(
        female_upregulation = rng.uniform(0.15, 0.45),
        noise_sd            = rng.uniform(0.05, 0.15),
    )

    # VLCFA burden: C26:0 primary, C24:0 intermediate
    # Ref: Kemp et al., Nat Rev Neurol 2012 (PMID 22965358)
    params["vlcfa"] = dict(
        a0_V              = rng.uniform(0.0, 0.5),
        a_activity        = rng.uniform(1.0, 2.0),
        a_age             = rng.uniform(0.3, 0.8),
        noise_sd_V        = rng.uniform(0.2, 0.5),
        c24_from_elovl1   = rng.uniform(0.4, 0.9),   # ELOVL1 elongates C22→C24
        c24_noise         = rng.uniform(0.15, 0.4),
    )

    # Oxidative stress (ROS from VLCFA + inflammation amplification)
    # Ref: Fourcade et al., Hum Mol Genet 2008 (PMID 18628207)
    params["oxidative_stress"] = dict(
        os_from_vlcfa = rng.uniform(0.3, 0.8),
        os_from_infl  = rng.uniform(0.4, 0.9),
        noise_sd      = rng.uniform(0.15, 0.4),
    )

    # Lipid factors (6 factors)
    params["lipid_factors"] = {}

    # 1. VLCFA_LPC: lysophospholipids driven by VLCFA (direct) + oxidative stress (iPLA2)
    # Ref: Engelen/Kemp group 2020-2024; Lands cycle disruption via PLA2G6
    params["lipid_factors"]["VLCFA_LPC"] = dict(
        b0         = rng.normal(0.0, 0.5),
        b_vlcfa    = rng.uniform(0.7, 1.6),
        b_oxstress = rng.uniform(0.3, 0.9),   # iPLA2 activated by ROS → LPC 26:0 from PC
        b_metab    = rng.uniform(0.1, 0.5),
        noise_sd   = rng.uniform(0.1, 0.4),
    )

    # 2. PC_PE: standard (non-ether) glycerophospholipids
    params["lipid_factors"]["PC_PE"] = dict(
        b0       = rng.normal(0.0, 0.5),
        b_vlcfa  = rng.uniform(0.2, 0.8),    # VLCFA incorporation into PC/PE
        b_metab  = rng.uniform(0.3, 1.0),
        noise_sd = rng.uniform(0.1, 0.4),
    )

    # 3. PLASMALOGEN: ether phospholipids (PE-p/PC-p) depleted by oxidative stress
    # Vinyl-ether bonds act as ROS antioxidant sinks → consumed by elevated ROS in ALD
    # Ref: Brites et al., Biochim Biophys Acta 2009 (PMID 19328222)
    params["lipid_factors"]["PLASMALOGEN"] = dict(
        b0         = rng.normal(0.0, 0.5),
        b_vlcfa    = rng.uniform(-0.4, -0.1),   # weak membrane disruption
        b_oxstress = rng.uniform(-0.8, -0.3),   # ROS strongly depletes vinyl-ether bonds
        b_metab    = rng.uniform(0.2, 0.8),     # baseline peroxisomal synthesis (positive)
        noise_sd   = rng.uniform(0.1, 0.4),
    )

    # 4. CERAMIDE: very-long-chain ceramides elevated by CerS2/CerS3 incorporating C24:0/C26:0
    # Also elevated by inflammation via nSMase (SM→Cer hydrolysis)
    # Ref: Cutler et al., Ann Neurol 2002 (PMID 12112068); Hein et al., J Neurochem 2008 (PMID 18298656)
    # NEW (V4): CerS2 uses C24:0-CoA as dominant brain substrate (Hama 2010, PMID 20184944)
    params["lipid_factors"]["CERAMIDE"] = dict(
        b0          = rng.normal(0.0, 0.5),
        b_vlcfa     = rng.uniform(0.7, 1.6),   # CerS3: C26:0-CoA → Cer(d18:1/26:0)
        b_vlcfa_c24 = rng.uniform(0.4, 0.9),   # CerS2: C24:0-CoA → Cer(d18:1/24:0)
        b_infl      = rng.uniform(0.3, 0.9),   # nSMase/aSMase activated by inflammation
        b_metab     = rng.uniform(0.1, 0.5),
        noise_sd    = rng.uniform(0.1, 0.4),
    )

    # 5. SM: sphingomyelin — inversely regulated by inflammation (sphingomyelinase activation)
    # SM species with VLCFA tails elevated; normal SM depleted by inflammation
    # Ref: Hein et al., J Neurochem 2008 (PMID 18298656)
    params["lipid_factors"]["SM"] = dict(
        b0       = rng.normal(0.0, 0.5),
        b_vlcfa  = rng.uniform(0.2, 0.7),    # VLCFA-containing SM species accumulate
        b_infl   = rng.uniform(-0.5, -0.2),  # nSMase depletes SM → Cer (negative effect)
        b_metab  = rng.uniform(0.2, 0.8),
        noise_sd = rng.uniform(0.1, 0.4),
    )

    # 6. TG_CE: neutral lipids (TG + standard cholesterol esters)
    params["lipid_factors"]["TG_CE"] = dict(
        b0       = rng.normal(0.0, 0.5),
        b_vlcfa  = rng.uniform(0.1, 0.6),
        b_metab  = rng.uniform(0.6, 1.5),
        noise_sd = rng.uniform(0.1, 0.4),
    )

    # VLCFA-containing cholesterol esters (adrenal cortex accumulation)
    # CE(24:0) and CE(26:0) accumulate → adrenal lipid droplet expansion → cortisol deficiency
    # Ref: Moser et al., Ann Neurol 1980 (PMID 6776400); Powers & Schaumburg 1974
    params["ce_vlcfa"] = dict(
        c_vlcfa    = rng.uniform(0.8, 1.5),
        c_sex_male = rng.uniform(0.2, 0.6),   # males more severely affected in adrenal glands
        noise_sd   = rng.uniform(0.1, 0.3),
    )

    # NEW (V4): Adrenal Insufficiency — threshold on accumulated CE_VLCFA
    # Expanded adrenal lipid droplets impair StAR-mediated cholesterol import →
    # cortisol/aldosterone deficiency in ~70% of males before neurological symptoms.
    # Threshold at CE_VLCFA ≈ 1.0 (Moser 1980); manifests as independent severity axis.
    # Ref: Moser et al., Ann Neurol 1980 (PMID 6776400); GeneReviews 2023
    params["adrenal_insufficiency"] = dict(
        ce_threshold = rng.uniform(0.7, 1.3),   # CE_VLCFA level triggering dysfunction
        scale        = rng.uniform(3.0, 7.0),   # sharpness of sigmoid threshold
        noise_sd     = rng.uniform(0.05, 0.20),
    )

    # Inflammation (updated V4: add plasmalogen anti-inflammatory feedback)
    # LPC activates microglia via GPR132/TLR; ceramide via PP2A/JNK; ROS directly.
    # Plasmalogen depletion removes anti-inflammatory buffer → amplifies neuroinflammation.
    # Ref: Fourcade et al., Hum Mol Genet 2008 (PMID 18628207); Brites 2009 (PMID 19328222)
    params["inflammation"] = dict(
        c0            = rng.normal(0.0, 0.5),
        c_V           = rng.uniform(0.4, 1.2),
        c_cer         = rng.uniform(0.3, 1.0),
        c_LPC         = rng.uniform(0.3, 1.0),   # LPC pro-inflammatory signal (GPR132)
        c_oxstress    = rng.uniform(0.2, 0.6),
        c_plasmalogen = rng.uniform(-0.4, -0.1),  # NEW: anti-inflammatory buffer (negative)
        noise_sd_I    = rng.uniform(0.2, 0.5),
    )

    # Demyelination (VLCFA + inflammation + ceramide-mediated oligodendrocyte apoptosis)
    # Ref: Hama 2010 (PMID 20184944); Engelen group 2020-2024
    params["demyelination"] = dict(
        d0         = rng.normal(0.0, 0.5),
        d_I        = rng.uniform(0.6, 1.8),
        d_V        = rng.uniform(0.3, 1.2),
        d_cer      = rng.uniform(0.2, 0.7),   # ceramide-mediated oligodendrocyte apoptosis
        noise_sd_D = rng.uniform(0.2, 0.5),
    )

    # Severity (6 levels; V4: adrenal insufficiency node + LPC direct path)
    # NEW (V4): e_ai replaces e_ce (now severity input is adrenal insufficiency, not raw CE_VLCFA)
    # NEW (V4): e_lpc — validated NBS biomarker LPC26_0 is an independent severity predictor
    # Ref: Billington 2025 (NBS cohort); GeneReviews 2023
    params["severity"] = dict(
        e0         = rng.normal(0.0, 0.5),
        e_D        = rng.uniform(0.2, 0.8),
        e_I        = rng.uniform(0.1, 0.6),
        e_ai       = rng.uniform(0.1, 0.4),   # adrenal insufficiency → severity
        e_lpc      = rng.uniform(0.05, 0.20), # LPC26_0 → severity (NBS biomarker)
        e_age      = rng.uniform(-0.3, 0.3),
        e_sex      = rng.uniform(0.1, 0.8),
        noise_sd_S = rng.uniform(0.3, 0.8),
        t1         = rng.normal(-1.5, 0.3),
        t2         = rng.normal(-0.5, 0.3),
        t3         = rng.normal( 0.5, 0.3),
        t4         = rng.normal( 1.5, 0.3),
        t5         = rng.normal( 2.5, 0.3),
    )

    return params


# ============================================================
# 4) Root nodes
# ============================================================

def sample_roots(n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    out = {}

    p_male = rng.uniform(0.3, 0.7)
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

    mean_metab = rng.normal(0.0, 0.5)
    sd_metab   = rng.uniform(0.7, 1.5)
    out["metabolic_background"] = rng.normal(mean_metab, sd_metab, size=n)

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
    return clip(a_sex + eps, 0.0, 1.2)


def compute_vlcfa(roots, abcd1_eff, params, rng):
    """C26:0 VLCFA burden.

    Female sex reduces net VLCFA via estrogen-driven ABCD2 upregulation,
    providing partial compensatory peroxisomal transport.
    Ref: Fourcade et al., Hum Mol Genet 2003 (PMID 12915447)
    """
    p_v     = params["vlcfa"]
    p_abcd2 = params["abcd2"]
    age      = roots["age"]
    sex_male = roots["sex_male"]
    n = age.shape[0]

    log_age = np.log(age + 1.0)

    # ABCD2 compensation active in females (sex_male == 0)
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
    """C24:0 VLCFA as ELOVL1-driven intermediate.

    C24:0/C22:0 ratio used alongside C26:0/C22:0 as ALD diagnostic biomarker.
    In V4 this is wired into ceramide computation (CerS2 substrate).
    Ref: Kemp et al., Nat Rev Neurol 2012 (PMID 22965358)
    """
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
    """Oxidative stress from VLCFA-driven peroxisomal dysfunction and inflammation.

    VLCFA directly generates ROS via incomplete beta-oxidation byproducts.
    Inflammation amplifies oxidative burden via microglial NADPH oxidase.
    Ref: Fourcade et al., Hum Mol Genet 2008 (PMID 18628207)
    """
    p = params["oxidative_stress"]
    n = vlcfa.shape[0]
    os_star = (
        p["os_from_vlcfa"] * vlcfa
        + p["os_from_infl"] * infl
        + rng.normal(0.0, p["noise_sd"], size=n)
    )
    return relu(os_star)


def compute_lipid_factors(roots, vlcfa, vlcfa_c24, oxidative_stress, infl, params, rng):
    """Compute 6 lipid latent factors with literature-grounded structural equations.

    V4 change: CERAMIDE now takes vlcfa_c24 as an additional input representing
    CerS2 (CERS2) incorporating C24:0-CoA → Cer(d18:1/24:0), the dominant brain
    ceramide elongase. Previously vlcfa_c24 was computed but unused.
    Ref: Hama 2010, Biochim Biophys Acta (PMID 20184944)
    """
    metab    = roots["metabolic_background"]
    n        = vlcfa.shape[0]
    lf_p     = params["lipid_factors"]
    factors  = {}

    p = lf_p["VLCFA_LPC"]
    L = p["b0"] + p["b_vlcfa"] * vlcfa + p["b_oxstress"] * oxidative_stress + p["b_metab"] * metab
    factors["VLCFA_LPC"] = np.tanh(L + rng.normal(0.0, p["noise_sd"], size=n))

    p = lf_p["PC_PE"]
    L = p["b0"] + p["b_vlcfa"] * vlcfa + p["b_metab"] * metab
    factors["PC_PE"] = np.tanh(L + rng.normal(0.0, p["noise_sd"], size=n))

    p = lf_p["PLASMALOGEN"]
    L = p["b0"] + p["b_vlcfa"] * vlcfa + p["b_oxstress"] * oxidative_stress + p["b_metab"] * metab
    factors["PLASMALOGEN"] = np.tanh(L + rng.normal(0.0, p["noise_sd"], size=n))

    # V4: CerS2 (C24:0) + CerS3 (C26:0) contributions to ceramide
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
    """VLCFA-containing cholesterol esters in adrenal cortex.

    CE(24:0) and CE(26:0) accumulate as poor substrates for HSL/LIPA.
    Foamy adrenocortical cells → impaired StAR-mediated cholesterol import
    → cortisol/aldosterone deficiency (adrenal insufficiency in ~70% of male ALD).
    Males more severely affected (adrenal gland androgen receptor effects).
    Ref: Moser et al., Ann Neurol 1980 (PMID 6776400)
    """
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
    """Adrenal insufficiency from accumulated VLCFA cholesterol esters.

    NEW in V4: explicit threshold-based intermediate between ce_vlcfa and severity.

    Expanded adrenal lipid droplets impair StAR-mediated cholesterol import once
    CE_VLCFA exceeds a threshold (~1.0). Manifests clinically as cortisol/aldosterone
    deficiency in ~70% of males, often preceding neurological symptoms by years.
    Ref: Moser et al., Ann Neurol 1980 (PMID 6776400); GeneReviews 2023
    """
    p = params["adrenal_insufficiency"]
    n = ce_vlcfa.shape[0]
    ai_star = p["scale"] * (ce_vlcfa - p["ce_threshold"])
    return clip(sigmoid(ai_star) + rng.normal(0.0, p["noise_sd"], size=n), 0.0, 1.0)


def compute_inflammation(vlcfa, oxidative_stress, factors, params, rng):
    """Neuroinflammation driven by VLCFA, LPC (microglial GPR132/TLR), ceramide, ROS,
    and modulated by plasmalogen anti-inflammatory buffer.

    V4 change: plasmalogen depletion removes the anti-inflammatory vinyl-ether buffer,
    amplifying neuroinflammation (negative coefficient on PLASMALOGEN factor).
    This completes the ROS → plasmalogen depletion → inflammation amplification loop.
    Ref: Fourcade et al., Hum Mol Genet 2008 (PMID 18628207); Brites 2009 (PMID 19328222)
    """
    p     = params["inflammation"]
    n     = vlcfa.shape[0]
    L_cer = factors["CERAMIDE"]
    L_lpc = factors["VLCFA_LPC"]
    L_pls = factors["PLASMALOGEN"]   # anti-inflammatory buffer (negative effect)

    I_star = (
        p["c0"]
        + p["c_V"]            * vlcfa
        + p["c_cer"]          * L_cer
        + p["c_LPC"]          * L_lpc
        + p["c_oxstress"]     * oxidative_stress
        + p["c_plasmalogen"]  * L_pls
        + rng.normal(0.0, p["noise_sd_I"], size=n)
    )
    return relu(I_star)


def compute_demyelination(vlcfa, infl, factors, params, rng):
    """Demyelination from inflammation + direct VLCFA toxicity + ceramide-mediated
    oligodendrocyte apoptosis.

    Ref: Hama 2010 (PMID 20184944); ceramide-mediated apoptosis in ALD oligodendrocytes
    """
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
    """6-level ordinal severity.

    V4 changes:
    - adrenal_insuff replaces raw ce_vlcfa: severity now goes through the explicit
      adrenal insufficiency threshold, aligning with clinical staging (Moser 1980).
    - VLCFA_LPC factor added as direct severity input: LPC26_0 is an independent
      validated predictor of severe/cerebral ALD phenotype in NBS cohorts.
      Ref: Billington 2025 (NBS cohort); GeneReviews 2023
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

    t1, t2, t3, t4, t5 = p["t1"], p["t2"], p["t3"], p["t4"], p["t5"]
    severity = np.zeros(n, dtype=int)
    severity[(S_star >= t1) & (S_star < t2)] = 1
    severity[(S_star >= t2) & (S_star < t3)] = 2
    severity[(S_star >= t3) & (S_star < t4)] = 3
    severity[(S_star >= t4) & (S_star < t5)] = 4
    severity[S_star >= t5]                   = 5

    return S_star, severity


# ============================================================
# 6) Map real lipid column name → latent factor (6 factors)
# ============================================================

def assign_main_factor(col: str) -> str:
    """Map a real lipid column to one of 6 latent factors."""
    # Ether phospholipids (plasmalogens) — vinyl-ether bonds consumed by ROS
    if col.startswith((
        "PE_O_", "PE_P_", "PE_OplusP_",
        "PC_O_", "PC_P_", "PC_OplusP_",
    )):
        return "PLASMALOGEN"

    # Lysophospholipids (VLCFA-LPC / iPLA2-mediated biomarkers)
    if col.startswith((
        "1_acyl_LPC_", "2_acyl_LPC_",
        "1_acyl_LPE_", "2_acyl_LPE_",
        "LPC_", "LPE_",
    )):
        return "VLCFA_LPC"

    # Sphingomyelin (separate from ceramide; inversely regulated by nSMase)
    if col.startswith(("SM_", "SM4_")):
        return "SM"

    # Ceramides, glycosphingolipids, ceramide metabolites (VLCFA-containing)
    if col.startswith(("Cer_", "HexCer_", "Hex2Cer_", "C1P_", "S1P_")):
        return "CERAMIDE"

    # Neutral lipids (TG + CE)
    if col.startswith(("TG_", "TG_O_", "DG_", "CE_")):
        return "TG_CE"

    # Standard glycerophospholipids (non-ether PC/PE + PI/PS/PG/PA/CL/BMP)
    if col.startswith((
        "PC_", "PE_",
        "PI_", "PS_", "PG_", "PA_",
        "LPG_", "LPA_", "LPI_", "LPS_",
        "CL_", "BMP_",
    )):
        return "PC_PE"

    return "PC_PE"


# ============================================================
# 7) Generate ONE synthetic world matching df_template schema
# ============================================================

FACTOR_NAMES = ["VLCFA_LPC", "PC_PE", "PLASMALOGEN", "CERAMIDE", "SM", "TG_CE"]


def sample_world_like(
    df_template: pd.DataFrame,
    cfg: WorldConfig,
) -> Tuple[pd.DataFrame, Dict]:
    """Generate one synthetic ALD cohort with the same column schema as df_template.

    Causal order (V4):
      roots → abcd1_effective → vlcfa (+ vlcfa_c24)
           → oxidative_stress (bootstrap) → factors → inflammation
           → oxidative_stress (refined) → factors (refined) → inflammation (refined)
           → ce_vlcfa → adrenal_insufficiency → demyelination → severity

    V4 additions vs V3:
      1. Plasmalogen → Inflammation feedback (anti-inflammatory buffer)
      2. VLCFA_C24 → CERAMIDE factor (CerS2 C24:0 substrate)
      3. Adrenal Insufficiency as explicit threshold-based intermediate node
      4. LPC26_0 (VLCFA_LPC factor) as direct severity input
      5. HexCer_/Hex2Cer_ columns receive extra contribution from demyelination
         (myelin GalCer release into plasma during active demyelination)
    """
    rng    = np.random.default_rng(cfg.seed)
    params = sample_world_hyperparams(rng)
    n      = cfg.n_samples

    roots      = sample_roots(n, rng)
    abcd1      = compute_abcd1_effective(roots, params, rng)
    vlcfa      = compute_vlcfa(roots, abcd1, params, rng)
    vlcfa_c24  = compute_vlcfa_c24(roots, abcd1, params, rng)

    # Bootstrap: initial zero inflammation for first oxidative_stress estimate
    infl_boot = relu(rng.normal(0.0, 0.3, size=n))
    ox_stress = compute_oxidative_stress(vlcfa, infl_boot, params, rng)
    factors   = compute_lipid_factors(roots, vlcfa, vlcfa_c24, ox_stress, infl_boot, params, rng)
    infl      = compute_inflammation(vlcfa, ox_stress, factors, params, rng)

    # One refinement step to resolve circular VLCFA → ROS → LPC → infl dependency
    ox_stress = compute_oxidative_stress(vlcfa, infl, params, rng)
    factors   = compute_lipid_factors(roots, vlcfa, vlcfa_c24, ox_stress, infl, params, rng)
    infl      = compute_inflammation(vlcfa, ox_stress, factors, params, rng)

    ce_vlcfa       = compute_ce_vlcfa(roots, vlcfa, params, rng)
    adrenal_insuff = compute_adrenal_insufficiency(ce_vlcfa, params, rng)
    demy           = compute_demyelination(vlcfa, infl, factors, params, rng)
    S_star, severity = compute_severity(roots, infl, demy, adrenal_insuff, factors, params, rng)

    # ---- Build output ----
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

        bias   = rng.normal(0.0, 0.3)
        w_main = rng.normal(1.0, 0.2)
        value  = bias + w_main * main_factor

        for k in FACTOR_NAMES:
            if k == main_key:
                continue
            w_cross = rng.normal(0.0, 0.08)
            value   = value + w_cross * factors[k]

        # HexCer columns additionally reflect demyelination-released myelin GalCer.
        # Active demyelination releases HexCer (galactosylceramide) into plasma —
        # a reverse causal path making HexCer a biomarker of lesion activity.
        # Ref: Engelen 2024 (plasma lipidomics 2020-2024)
        if col.startswith(("HexCer_", "Hex2Cer_")):
            w_demy = rng.normal(0.6, 0.1)
            value  = value + w_demy * demy

        noise_sd = rng.uniform(0.3, 0.8)
        value    = value + rng.normal(0.0, noise_sd, size=n)
        out[col] = value

    return pd.DataFrame(out, columns=cols), params


# ============================================================
# 8) Generate many worlds
# ============================================================

def _generate_one(args):
    """Worker function for parallel generation — one world per call."""
    i, df_template, n_samples, seed, out_dir, prefix = args
    cfg = WorldConfig(n_samples=n_samples, seed=seed)
    df_syn, _ = sample_world_like(df_template, cfg)
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
) -> None:
    """Generate n_worlds synthetic cohorts and stream each to disk.

    Worlds are never accumulated in RAM — each is saved and discarded immediately.
    Use n_jobs > 1 to parallelize across CPU cores (each world is independent).
    """
    if n_samples is None:
        n_samples = df_template.shape[0]
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    args = [
        (i, df_template, n_samples, base_seed + i, out_dir, prefix)
        for i in range(n_worlds)
    ]

    if n_jobs == 1:
        for a in args:
            _generate_one(a)
    else:
        with mp.Pool(processes=n_jobs) as pool:
            pool.map(_generate_one, args)


# ============================================================
# 9) Entry point
# ============================================================

if __name__ == "__main__":
    N_JOBS   = int(os.environ.get("N_JOBS", 1))   # set via: N_JOBS=32 python ...
    N_WORLDS = 500
    N_SAMPLES = 20_000

    real_path = "./data/ALD_lipidomics_merged.csv"
    df_real   = pd.read_csv(real_path)

    print(f"Generating {N_WORLDS} worlds × {N_SAMPLES} samples using {N_JOBS} CPU core(s).")
    generate_many_worlds(
        df_template = df_real,
        n_worlds    = N_WORLDS,
        n_samples   = N_SAMPLES,
        base_seed   = 42,
        out_dir     = "./synthetic_data/causal_knowledge_v4/",
        prefix      = "ALD_world",
        n_jobs      = N_JOBS,
    )
    print("Done.")

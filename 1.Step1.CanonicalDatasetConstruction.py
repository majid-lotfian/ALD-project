"""
STEP 1 — Canonical ALD Lipidomics Dataset Construction
======================================================

Creates the unified dataset and metadata artifacts used by all
synthetic-data generators and foundation-model pretraining.
"""

import pandas as pd
import numpy as np
import yaml, json, pickle, os
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer

# --------------------------------------------------------------------
# 1. CONFIG
# --------------------------------------------------------------------
RAW_MALE_FILE = "/data/43856_2024_605_MOESM4_ESM-lipidomics_data_males.xlsx"
RAW_FEMALE_FILE = "/data/43856_2024_605_MOESM6_ESM-lipidomics_data_females.xlsx"
MALE_SHEET = "lipidomics_data_males"
FEMALE_SHEET = "lipidomics_data_females"

OUT_DIR = "./data"
os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------------------------------------------------
# 2. LOAD & MERGE RAW DATA
# --------------------------------------------------------------------
def load_dataset(path, sheet, sex_label):
    df = pd.read_excel(path, sheet_name=sheet)
    df["sex"] = sex_label
    return df

male_df = load_dataset(RAW_MALE_FILE, MALE_SHEET, "male")
female_df = load_dataset(RAW_FEMALE_FILE, FEMALE_SHEET, "female")
df = pd.concat([male_df, female_df], ignore_index=True)

# --------------------------------------------------------------------
# 3. CLEAN COLUMN NAMES
# --------------------------------------------------------------------
def clean_columns(df):
    df = df.rename(columns=lambda c: str(c).strip()
                   .replace("(", "_")
                   .replace(")", "")
                   .replace(" ", "_")
                   .replace("-", "_")
                   .replace("+", "plus")
                   .replace("/", "_div_"))
    return df

df = clean_columns(df)

# --------------------------------------------------------------------
# 4. DETECT TYPES
# --------------------------------------------------------------------
id_cols = [c for c in df.columns if "sample" in c.lower()]
cat_cols = [c for c in df.columns if c.lower() in ["severity", "sex"]]
num_cols = [c for c in df.columns if c not in id_cols + cat_cols]

# Keep only numeric + categorical of interest
numeric_df = df[num_cols].apply(pd.to_numeric, errors="coerce")
cat_df = df[cat_cols].copy()

# --------------------------------------------------------------------
# 5. HANDLE MISSINGNESS
# --------------------------------------------------------------------
missing_mask = numeric_df.isna().astype(int)
with open(os.path.join(OUT_DIR, "ald_missingmask.pkl"), "wb") as f:
    pickle.dump(missing_mask, f)

imputer = KNNImputer(n_neighbors=5, weights="distance")
numeric_imputed = pd.DataFrame(
    imputer.fit_transform(numeric_df),
    columns=numeric_df.columns
)

# --------------------------------------------------------------------
# 6. NORMALIZE NUMERIC FEATURES (log1p + z-score)
# --------------------------------------------------------------------
numeric_logged = np.log1p(np.maximum(numeric_imputed, 0))
scaler = StandardScaler()
numeric_scaled = pd.DataFrame(
    scaler.fit_transform(numeric_logged),
    columns=numeric_logged.columns
)

# --------------------------------------------------------------------
# 7. ENCODE CATEGORICALS
# --------------------------------------------------------------------
encoders = {}
for c in cat_df.columns:
    le = LabelEncoder()
    cat_df[c] = le.fit_transform(cat_df[c].astype(str))
    encoders[c] = dict(zip(le.classes_, le.transform(le.classes_)))

with open(os.path.join(OUT_DIR, "ald_label_encoders.yaml"), "w") as f:
    yaml.dump(encoders, f)

# --------------------------------------------------------------------
# 8. REASSEMBLE CLEANED DATAFRAME
# --------------------------------------------------------------------
canonical_df = pd.concat([cat_df, numeric_scaled], axis=1)
canonical_df.insert(0, "Sample_ID", df[id_cols[0]].values)

# --------------------------------------------------------------------
# 9. COMPUTE STATISTICAL FINGERPRINT
# --------------------------------------------------------------------
stats_dict = {}
for col in numeric_scaled.columns:
    x = numeric_scaled[col].values
    stats_dict[col] = {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "skew": float(stats.skew(x, nan_policy="omit")),
        "kurt": float(stats.kurtosis(x, nan_policy="omit")),
        "missing_rate": float(missing_mask[col].mean())
    }

# pairwise correlations (top-100 features only to keep JSON small)
corr_subset = numeric_scaled.corr().iloc[:100, :100]
stats_dict["pairwise_corr_top100"] = corr_subset.to_dict()

with open(os.path.join(OUT_DIR, "ald_stats.json"), "w") as f:
    json.dump(stats_dict, f, indent=2)

# --------------------------------------------------------------------
# 10. SAVE SCHEMA + DATA
# --------------------------------------------------------------------
schema = {
    "n_samples": int(len(canonical_df)),
    "n_features": int(len(num_cols)),
    "categorical_features": cat_df.columns.tolist(),
    "numeric_features": num_cols,
    "transformations": {
        "numeric": "log1p + zscore",
        "categorical": "LabelEncoder"
    },
    "imputation": "KNNImputer(k=5, distance)",
}

with open(os.path.join(OUT_DIR, "ald_schema.yaml"), "w") as f:
    yaml.dump(schema, f)

canonical_df.to_csv(os.path.join(OUT_DIR, "ALD_lipidomics_merged.csv"), index=False)

print(f"✅ Canonical dataset saved in {OUT_DIR}")
print(f"Shape: {canonical_df.shape}")
print(f"Columns: {canonical_df.columns[:10].tolist()} ...")

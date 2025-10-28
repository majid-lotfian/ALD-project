🧩 Applying the ALD Synthetic Data Pipeline to Other Datasets

This repository’s synthetic-data pipeline is domain-agnostic — you can reuse it for any structured tabular dataset, not just lipidomics or ALD research.

However, there are a few important nuances you should be aware of when adapting it.

🧩 1️⃣ What parts are 100% reusable

The following stages are completely generic — they work with any structured (tabular) data:

| Stage                                          | Description                                                                                    | Notes                                                               |
| ---------------------------------------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| **Step 1 — Canonical dataset construction**    | Cleans column names, detects numeric/categorical, imputes missing values, normalizes, encodes. | Works for any table that’s in “row = sample, col = feature” format. |
| **Step 2.1 — Copula generator**                | Uses Gaussian (or vine) copulas to model correlations.                                         | Works on any continuous + categorical mix.                          |
| **Step 2.2 — Bayesian network generator**      | Learns probabilistic structure among variables.                                                | Works if feature count and memory fit your hardware.                |
| **Step 2.3 — Neural generators (CTGAN, TVAE)** | Deep generative models for complex dependencies.                                               | SDV synthesizers are domain-agnostic — you just need `Metadata`.    |
| **Step 2.4 — Causal / Knowledge-based generator** | Integrates domain or expert-defined causal relationships into the generation process. | Optional layer — define causal graphs or pathways (e.g., with CausalNex or pgmpy). |

✅ These steps will run unchanged on:

Clinical lab panels

Genomic feature matrices

Finance, insurance, or manufacturing data

Any other structured dataset (wide or narrow)

🧠 Optional: Domain-knowledge layer

If your field includes known causal or mechanistic relationships — for example, metabolic pathways, clinical dependencies, or process flows — you can encode that structure manually and generate synthetic data that respect those relationships.  
This is handled in **Step 2.4**, which uses simple causal graphs or structural-equation models to complement the purely data-driven generators.


🧠 2️⃣ What you must adapt per dataset

There are only a few places that depend on the data domain or format:

| Part                                            | Why it matters                                                  | How to adapt                                                                        |
| ----------------------------------------------- | --------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| **Missing-data handling (KNN imputer, Step 1)** | Some datasets may have categorical missingness, not numeric.    | Use `SimpleImputer(strategy="most_frequent")` for categoricals.                     |
| **Normalization**                               | `log1p` is specific to lipidomics / positive-valued features.   | Replace with `StandardScaler` or `MinMaxScaler` if your features include negatives. |
| **Label encoding**                              | Some domains have text categories, not sex/severity.            | Replace with `OneHotEncoder` or leave them as `category` dtype.                     |
| **Metadata construction (Step 2.3)**            | SDV needs correct `sdtype` (numerical/categorical/datetime/id). | Use `Metadata.detect_from_dataframe()` and optionally adjust sdtypes.               |
| **Feature count (BN step)**                     | Large tables require different block sizes.                     | Tune `TOP_K_FEATURES` and `BLOCK_SIZE` according to available RAM.                  |
| **Causal structure (Step 2.4)** | Optional expert-defined relationships between variables. | Provide your own causal graph (edges or YAML) describing known dependencies; otherwise skip this step. |


So apart from those preprocessing details, the logic and training flow stay identical.

⚙️ 3️⃣ The only assumption: tabular structure

All models here assume:

Each row = an independent observation/sample.

Each column = a feature/variable.

The table fits in memory (or can be chunked).

If that’s true, you can use it on:

Proteomics, transcriptomics

EHR / patient data

Financial transactions (after aggregation)

IoT / sensor readings

Socio-economic datasets

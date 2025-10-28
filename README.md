# ALD-project

🧬 ALD Foundation Model: Synthetic Data Pretraining Pipeline

ALD-FM is an open research project to build a foundation model for Adrenoleukodystrophy (ALD) and related lipidomics disorders.
It focuses on scalable synthetic data generation and representation learning from high-dimensional biochemical tables.

🚀 Project Overview

Modern foundation models need millions of samples — but ALD datasets are small and sparse.
This project builds a synthetic data engine that expands real lipidomics datasets into a massive, biologically consistent corpus for pretraining.

Pipeline highlights:

🔧 1. Prepare your dataset

Format your data as a single CSV or Excel sheet where:

Each row = one sample or observation

Each column = one feature or variable

Ensure columns have unique, clean names.

🧼 2. Canonical dataset construction

Use 1.CanonicalDatasetConstruction.py to:

Impute missing values

Normalize numeric features (log1p or z-score)

Encode categorical features
Modify the imputation or normalization methods if your data contain negatives or text columns.

🧮 3. Synthetic data generation

You can choose any or all of:

Copula-based models (2.1) — preserves correlations

Bayesian networks (2.2a) — learns conditional dependencies

CTGAN / TVAE (2.3) — learns nonlinear, multimodal patterns

All models operate on any clean numeric or mixed-type DataFrame.

⚙️ 4. Parameters to adjust

TOP_K_FEATURES, BLOCK_SIZE → control memory

N_TABLES, N_SAMPLES → control scale of synthetic generation

Metadata sdtypes (numerical, categorical) → control how SDV models each feature

💾 5. Outputs

Each step saves:

Synthetic tables (.csv)

Metrics (_metrics.json)

Optional schema and statistics (.yaml, .json)

These outputs can be used for model pretraining or simulation tasks in any domain.

🧠 Example use cases

Biomedical omics (proteomics, metabolomics, transcriptomics)

Clinical EHR data

Financial time-independent tables

Industrial sensor datasets

Social science / demographic data

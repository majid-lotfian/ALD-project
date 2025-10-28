🧩 Causal Model Definition & Simulation Pipeline
ALD-FM Synthetic Data Pipeline — Step 2.4: Domain-Knowledge-Based Generation


🧭 Overview

This subprocess lets clinicians and domain experts describe biological causal relationships in an intuitive, spreadsheet-friendly way.
The system then automatically converts those relationships into YAML configuration files and uses them to generate synthetic datasets that obey those causal mechanisms.

🧩 Workflow Summary

| Stage                           | Actor                     | Output                                  | Description                                                                     |
| ------------------------------- | ------------------------- | --------------------------------------- | ------------------------------------------------------------------------------- |
| 1️⃣ Define causal relationships | Clinician / domain expert | `causal_relations.xlsx` or `.csv`       | Each row describes how one or more causes affect an outcome variable.           |
| 2️⃣ Parse causal table → YAML   | Parser script             | `configs/causal_models/*.yaml`          | Converts human-readable table into machine-readable causal graph configuration. |
| 3️⃣ Simulate data from YAML     | Simulator script          | `synthetic_data/causal_knowledge/*.csv` | Generates synthetic tables following the specified causal structure.            |
| 4️⃣ QC & merge                  | Data scientist            | `qc_report.json`, combined corpus       | Optional validation and integration with Copula/BN/Neural synthetic data.       |



🧠 Step 1 — Define Causal Relationships (Clinician Input)

Create or edit a spreadsheet named
inputs/causal_relations.xlsx (or .csv) with the following columns:

| Cause(s)                             | Effect                | Direction                               | Type                                      | Condition                            | Strength                     | Comment                          |
| ------------------------------------ | --------------------- | --------------------------------------- | ----------------------------------------- | ------------------------------------ | ---------------------------- | -------------------------------- |
| One or more causes (comma-separated) | The variable affected | `positive` or `negative` (list allowed) | `linear`, `logistic`, `interaction`, etc. | Optional logic, e.g. `if Sex = male` | `weak`, `moderate`, `strong` | Free-text biological explanation |


Example

| Cause(s)         | Effect        | Direction | Type     | Condition     | Strength | Comment                                    |
| ---------------- | ------------- | --------- | -------- | ------------- | -------- | ------------------------------------------ |
| ABCD1_loss       | Peroxi_BetaOx | negative  | linear   | —             | strong   | ABCD1 loss reduces β-oxidation             |
| Peroxi_BetaOx    | VLCFA_C26_0   | negative  | linear   | —             | strong   | Low oxidation → high VLCFA                 |
| ELOVL1_activity  | VLCFA_C26_0   | positive  | linear   | —             | moderate | Elongase synthesizes VLCFA                 |
| VLCFA_C26_0, Age | LPC26_0       | positive  | additive | if Sex = male | weak     | More C26:0 & age increase LPC26:0 in males |


🧩 Rules of thumb

Use concise biochemical statements; one row = one rule.

For multiple causes, separate them by commas.

For conditional cases, use plain phrases like if Sex = male.

“Direction” controls the sign of effect; “Strength” controls magnitude.

⚙️ Step 2 — Generate YAML Causal Models

Run the parser:

parse_causal_table.py

The parser will:
1. Read each row.
2. Detect causes, effects, directions, and strengths.
3. Create numeric coefficients automatically (e.g., strong → 0.9, moderate → 0.6, weak → 0.3; sign from direction).
4. Combine rows for the same effect into one node.
5. Save one YAML file per logical group.


Example auto-generated YAML:
'''
model_name: ald_vlcfa
n_samples: 5000
output_dir: synthetic_data/causal_knowledge

nodes:
  ABCD1_loss:
    type: bernoulli
    params: {p: 0.5}

  Peroxi_BetaOx:
    type: linear
    parents: [ABCD1_loss]
    coef: {intercept: 1.0, ABCD1_loss: -0.9}
    noise_sd: 0.3

  VLCFA_C26_0:
    type: linear
    parents: [Peroxi_BetaOx, ELOVL1_activity]
    coef: {intercept: 0.2, Peroxi_BetaOx: -0.9, ELOVL1_activity: 0.6}
    noise_sd: 0.25

  LPC26_0:
    type: linear
    parents: [VLCFA_C26_0, Age]
    coef: {intercept: 0.1, VLCFA_C26_0: 0.3, Age: 0.1}
    noise_sd: 0.2
    condition: "Sex == 'male'"
'''

💾 Step 3 — Generate Synthetic Tables

Run the simulator:

simulate_causal_tables.py

The script will:
Scan configs/causal_models/ for all .yaml files.
For each YAML:
  Build the directed graph (DAG).
  Topologically sort nodes.
  Simulate exogenous and endogenous variables according to the equations.
  Save results to:
    synthetic_data/causal_knowledge/<model_name>_v1.csv

Each row = one simulated patient/sample following the biological causal logic.


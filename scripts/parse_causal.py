import pandas as pd
import yaml
from pathlib import Path

# ============================================================
# Configuration
# ============================================================

STRENGTH_MAP = {'weak': 0.3, 'moderate': 0.6, 'strong': 0.9}

DEFAULT_PRIORS = {
    "Age": {"type": "truncated_normal", "params": {"mean": 35, "sd": 12, "low": 0, "high": 100}},
    "Sex": {"type": "bernoulli", "params": {"p": 0.5}},
}

# ============================================================
# Helper functions
# ============================================================

def parse_strength_or_coef(value, direction_sign):
    """Interpret coefficient as numeric or qualitative."""
    try:
        return float(value)
    except (TypeError, ValueError):
        level = str(value).lower().strip()
        base = STRENGTH_MAP.get(level, 0.6)
        return direction_sign * base

def parse_direction(text):
    if isinstance(text, str) and 'neg' in text.lower():
        return -1
    return 1

# ============================================================
# Core conversion
# ============================================================

def build_yaml_from_csv(csv_path, out_dir):
    df = pd.read_csv(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes = {}
    global_refs = set()

    for _, row in df.iterrows():
        cause = str(row['Cause']).strip()
        effect = str(row['Effect']).strip()
        if not cause or not effect:
            continue

        direction_sign = parse_direction(row.get('Direction', 'positive'))
        coef_value = parse_strength_or_coef(row.get('Coefficient_or_Level', 'moderate'), direction_sign)
        node_type = str(row.get('Type', 'linear')).strip()
        condition = str(row.get('Condition', '')).strip() or None
        refkey = str(row.get('RefKey', '')).strip() or None
        refnote = str(row.get('RefNote', '')).strip() or None

        if effect not in nodes:
            nodes[effect] = {
                'type': node_type,
                'parents': [],
                'coef': {'intercept': 0.0},
                'noise_sd': 0.25
            }
            if condition:
                nodes[effect]['condition'] = condition
            nodes[effect]['refs'] = []

        nodes[effect]['parents'].append(cause)
        nodes[effect]['coef'][cause] = coef_value
        if refkey:
            nodes[effect]['refs'].append({'RefKey': refkey, 'RefNote': refnote})
            global_refs.add(refkey)

    # Define exogenous priors
    for node_name in list(nodes.keys()):
        for parent in nodes[node_name]['parents']:
            if parent not in nodes:
                if parent in DEFAULT_PRIORS:
                    nodes[parent] = DEFAULT_PRIORS[parent]
                else:
                    nodes[parent] = {'type': 'normal', 'params': {'mean': 0.0, 'sd': 1.0}}

    import math

    # Automatic n_samples heuristic (scaled for biomedical DAGs)
    E = len(df)  # number of causal rules (edges)
    K = sum(1 for n in nodes.values() if 'parents' in n)  # endogenous nodes
    auto_n = int(round(1000 * math.sqrt(E + K)))
    n_samples = max(100, min(auto_n, 10000))  # clamp between 100 and 10k


    yaml_spec = {
        'model_name': Path(csv_path).stem,
        'n_samples': n_samples,
        'output_dir': 'synthetic_data/causal_knowledge',
        'model_refs': sorted(list(global_refs)),
        'nodes': nodes
    }

    out_path = out_dir / f"{Path(csv_path).stem}.yaml"
    with open(out_path, "w") as f:
        yaml.dump(yaml_spec, f, sort_keys=False)

    print(f"✅ YAML written to {out_path} (n_samples={n_samples})")

# ============================================================
# Main
# ============================================================

def main(input_dir="inputs", output_dir="configs/causal_models"):
    input_dir = Path(input_dir)
    csv_files = list(input_dir.glob("*.csv"))
    if not csv_files:
        print("⚠️ No CSV files found in inputs/.")
        return

    for csv_path in csv_files:
        print(f"📄 Processing {csv_path.name} ...")
        build_yaml_from_csv(csv_path, output_dir)

    print("\n✅ All CSVs converted to YAML successfully.")

if __name__ == "__main__":
    main()

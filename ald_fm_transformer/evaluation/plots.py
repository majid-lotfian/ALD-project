from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_learning_curve(csv_path: str, out_path: str, metric: str = 'loss'):
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(8, 5))
    for split in sorted(df['split'].unique()):
        sub = df[df['split'] == split]
        plt.plot(sub['epoch'], sub[metric], label=split)
    plt.xlabel('epoch')
    plt.ylabel(metric)
    plt.legend()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

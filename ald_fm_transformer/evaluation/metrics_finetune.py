from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    labels = list(range(int(max(y_true.max(), y_pred.max())) + 1))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    recalls = np.divide(
        np.diag(cm),
        cm.sum(axis=1),
        out=np.zeros(len(labels), dtype=np.float64),
        where=(cm.sum(axis=1) != 0),
    )
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'balanced_accuracy': float(recalls.mean()),
    }

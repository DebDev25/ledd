"""Metrics: AUC, F1, accuracy, average precision, plus deletion/insertion AUC."""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def classification_metrics(logits: Sequence[float], labels: Sequence[int],
                           threshold: float = 0.5) -> Dict[str, float]:
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 f1_score, roc_auc_score)

    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    probs = _sigmoid(logits)
    preds = (probs >= threshold).astype(int)

    out = {
        "acc": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "n": int(len(labels)),
        "pos_rate": float(labels.mean()) if len(labels) else 0.0,
    }
    if len(np.unique(labels)) > 1:      # AUC undefined on a single-class split
        out["auc"] = float(roc_auc_score(labels, probs))
        out["ap"] = float(average_precision_score(labels, probs))
    else:
        out["auc"] = float("nan")
        out["ap"] = float("nan")
    return out


def per_generator_metrics(logits, labels, generators) -> Dict[str, Dict[str, float]]:
    out = {}
    gens = np.asarray(generators)
    for g in sorted(set(gens.tolist())):
        m = gens == g
        out[str(g)] = classification_metrics(np.asarray(logits)[m], np.asarray(labels)[m])
    return out


def curve_auc(values: Sequence[float]) -> float:
    """Normalised area under a monotone step curve (used by deletion/insertion)."""
    v = np.asarray(values, dtype=np.float64)
    if len(v) < 2:
        return float("nan")
    return float(np.trapz(v, dx=1.0 / (len(v) - 1)))

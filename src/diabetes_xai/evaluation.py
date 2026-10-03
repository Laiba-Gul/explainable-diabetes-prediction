"""Metrics, bootstrap confidence intervals and validation-only threshold selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRICS = [
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "roc_auc",
    "pr_auc",
    "mcc",
    "brier",
    "log_loss",
]


def metrics_at(y_true, proba, threshold: float = 0.5) -> dict[str, float]:
    """Full metric bundle for positive-class probabilities at a threshold."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.clip(np.asarray(proba, dtype=float).ravel(), 1e-7, 1 - 1e-7)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "mcc": matthews_corrcoef(y_true, pred),
        "brier": brier_score_loss(y_true, proba),
        "log_loss": log_loss(y_true, proba, labels=[0, 1]),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def choose_threshold(y_val, proba_val, policy: str = "f1") -> float:
    """Pick a decision threshold on VALIDATION data only.

    policies: ``fixed`` (0.5), ``f1`` (max F1), ``youden`` (max sens+spec-1),
    ``accuracy`` (max accuracy).
    """
    if policy == "fixed":
        return 0.5
    y_val = np.asarray(y_val).astype(int)
    proba_val = np.asarray(proba_val, dtype=float)
    grid = np.linspace(0.01, 0.99, 981)
    best_t, best_s = 0.5, -np.inf
    for t in grid:
        pred = proba_val >= t
        if policy == "f1":
            s = f1_score(y_val, pred, zero_division=0)
        elif policy == "accuracy":
            s = accuracy_score(y_val, pred)
        elif policy == "youden":
            tp = np.sum(pred & (y_val == 1))
            tn = np.sum(~pred & (y_val == 0))
            s = tp / max(1, (y_val == 1).sum()) + tn / max(1, (y_val == 0).sum()) - 1
        else:
            raise ValueError(f"Unknown threshold policy '{policy}'")
        if s > best_s:
            best_t, best_s = float(t), s
    return round(best_t, 4)


def bootstrap_ci(
    y_true,
    proba,
    threshold: float = 0.5,
    n_boot: int = 1000,
    metrics=("roc_auc", "pr_auc", "f1", "mcc", "recall"),
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, tuple[float, float]]:
    """Percentile bootstrap CIs for selected metrics on a fixed test set."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    n = len(y_true)
    draws: dict[str, list[float]] = {m: [] for m in metrics}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, pp = y_true[idx], proba[idx]
        if yt.min() == yt.max():
            continue
        pred = pp >= threshold
        for m in metrics:
            if m == "roc_auc":
                draws[m].append(roc_auc_score(yt, pp))
            elif m == "pr_auc":
                draws[m].append(average_precision_score(yt, pp))
            elif m == "f1":
                draws[m].append(f1_score(yt, pred, zero_division=0))
            elif m == "mcc":
                draws[m].append(matthews_corrcoef(yt, pred))
            elif m == "recall":
                draws[m].append(recall_score(yt, pred, zero_division=0))
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {m: (float(np.percentile(v, lo)), float(np.percentile(v, hi))) for m, v in draws.items() if v}


def results_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    order = [c for c in ["model", "family", "split"] if c in df.columns]
    rest = [c for c in df.columns if c not in order]
    return df[order + rest]

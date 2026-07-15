"""Classification metrics for baseline reversal models."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from apex_lab.models.calibration import calibration_metrics


def evaluate_binary_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    *,
    calibration_bins: int = 10,
) -> dict[str, Any]:
    """Return the baseline classification metric suite."""
    y_true_array = np.asarray(y_true, dtype=int)
    y_pred_array = np.asarray(y_pred, dtype=int)
    y_prob_array = np.asarray(y_prob, dtype=float)

    roc_auc: float | None
    pr_auc: float | None

    try:
        roc_auc = float(roc_auc_score(y_true_array, y_prob_array))
    except ValueError:
        roc_auc = None

    try:
        pr_auc = float(average_precision_score(y_true_array, y_prob_array))
    except ValueError:
        pr_auc = None

    cm = confusion_matrix(y_true_array, y_pred_array, labels=[0, 1])

    return {
        "precision": float(precision_score(y_true_array, y_pred_array, zero_division=0)),
        "recall": float(recall_score(y_true_array, y_pred_array, zero_division=0)),
        "f1": float(f1_score(y_true_array, y_pred_array, zero_division=0)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm.astype(int).tolist(),
        "calibration": calibration_metrics(y_true_array, y_prob_array, n_bins=calibration_bins),
    }

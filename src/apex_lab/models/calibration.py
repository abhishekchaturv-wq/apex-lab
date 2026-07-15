"""Calibration helpers for probabilistic classifiers."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


def calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute calibration diagnostics from true labels and probabilities."""
    y_true_array = np.asarray(y_true, dtype=int)
    y_prob_array = np.asarray(y_prob, dtype=float)

    if y_true_array.size == 0:
        return {
            "brier_score": None,
            "n_bins": n_bins,
            "curve": {
                "mean_predicted_probability": [],
                "fraction_of_positives": [],
            },
        }

    frac_pos, mean_pred = calibration_curve(y_true_array, y_prob_array, n_bins=n_bins, strategy="uniform")

    return {
        "brier_score": float(brier_score_loss(y_true_array, y_prob_array)),
        "n_bins": n_bins,
        "curve": {
            "mean_predicted_probability": mean_pred.astype(float).tolist(),
            "fraction_of_positives": frac_pos.astype(float).tolist(),
        },
    }

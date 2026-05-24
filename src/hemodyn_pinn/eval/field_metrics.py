"""Scalar accuracy metrics for field comparisons.

All functions accept flat or multi-dimensional arrays and ravel internally.
Reference values use RMS normalisation for NRMSE (consistent with the BHPO
objective in bhpo/objective.py).
"""

from __future__ import annotations

import numpy as np


def nrmse(pred: np.ndarray, ref: np.ndarray) -> float:
    """Root-mean-square error normalised by RMS of the reference field.

    Parameters
    ----------
    pred, ref:
        Arrays of equal shape (any layout; ravelled internally).

    Returns
    -------
    float
        NRMSE ∈ [0, ∞).  0 = perfect; 1 ≈ error comparable to reference scale.
    """
    diff = pred.ravel() - ref.ravel()
    rms_ref = float(np.sqrt(np.mean(ref.ravel() ** 2)))
    return float(np.sqrt(np.mean(diff ** 2))) / (rms_ref + 1e-12)


def r2(pred: np.ndarray, ref: np.ndarray) -> float:
    """Coefficient of determination R².

    Returns
    -------
    float
        R² ∈ (-∞, 1].  1 = perfect; 0 = no better than the mean; <0 = worse.
    """
    p, r = pred.ravel(), ref.ravel()
    ss_res = float(np.sum((p - r) ** 2))
    ss_tot = float(np.sum((r - r.mean()) ** 2))
    return 1.0 - ss_res / (ss_tot + 1e-12)


def mae(pred: np.ndarray, ref: np.ndarray) -> float:
    """Mean absolute error in the same units as the inputs."""
    return float(np.mean(np.abs(pred.ravel() - ref.ravel())))


def compute_field_metrics(pred: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    """Compute NRMSE, R², and MAE in one call.

    Parameters
    ----------
    pred, ref:
        Predicted and reference arrays (same shape).

    Returns
    -------
    dict with keys ``nrmse``, ``r2``, ``mae``.
    """
    return {
        "nrmse": nrmse(pred, ref),
        "r2": r2(pred, ref),
        "mae": mae(pred, ref),
    }

"""WSS-specific accuracy metrics.

Extends field_metrics with clinically relevant WSS summaries:
- peak WSS relative error (relevant for rupture-risk biomarkers)
- mean pointwise relative error
- descriptive statistics (mean, max, p95) for both prediction and reference
"""

from __future__ import annotations

import numpy as np

from hemodyn_pinn.eval.field_metrics import compute_field_metrics


def peak_wss_error(pred: np.ndarray, ref: np.ndarray) -> float:
    """Relative error in peak (maximum) WSS.

    Returns
    -------
    float
        |max(pred) − max(ref)| / max(ref).
    """
    return float(abs(float(pred.max()) - float(ref.max())) / (float(ref.max()) + 1e-12))


def mean_relative_error(pred: np.ndarray, ref: np.ndarray) -> float:
    """Mean pointwise relative error: mean(|pred − ref| / (|ref| + ε))."""
    return float(np.mean(np.abs(pred - ref) / (np.abs(ref) + 1e-12)))


def compute_wss_metrics(pred: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    """Full WSS metric suite.

    Parameters
    ----------
    pred:
        (W,) PINN WSS magnitudes [Pa] on all wall faces.
    ref:
        (W,) CFD WSS magnitudes [Pa] on the same faces.

    Returns
    -------
    dict with keys: nrmse, r2, mae, peak_wss_error, mean_relative_error,
    pred_mean_pa, pred_max_pa, pred_p95_pa,
    ref_mean_pa, ref_max_pa, ref_p95_pa.
    """
    metrics = compute_field_metrics(pred, ref)
    metrics.update({
        "peak_wss_error": peak_wss_error(pred, ref),
        "mean_relative_error": mean_relative_error(pred, ref),
        "pred_mean_pa": float(pred.mean()),
        "pred_max_pa": float(pred.max()),
        "pred_p95_pa": float(np.percentile(pred, 95)),
        "ref_mean_pa": float(ref.mean()),
        "ref_max_pa": float(ref.max()),
        "ref_p95_pa": float(np.percentile(ref, 95)),
    })
    return metrics

"""Bland–Altman agreement statistics and intraclass correlation coefficient.

Reference: Bland & Altman (1986), Lancet.
ICC formula: one-way random effects model, ICC(1,1).
"""

from __future__ import annotations

import numpy as np


def bland_altman_stats(pred: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    """Compute Bland–Altman agreement statistics.

    Parameters
    ----------
    pred:
        (W,) predicted values (PINN WSS [Pa]).
    ref:
        (W,) reference values (CFD WSS [Pa]).

    Returns
    -------
    dict with keys:
        bias_pa          — mean difference (pred − ref)
        std_pa           — standard deviation of differences
        loa_upper_pa     — upper 95 % limit of agreement (bias + 1.96 SD)
        loa_lower_pa     — lower 95 % limit of agreement (bias − 1.96 SD)
        pct_within_loa   — % of points within LoA
    """
    diffs = pred.ravel() - ref.ravel()
    bias = float(np.mean(diffs))
    std_d = float(np.std(diffs, ddof=1))
    loa_upper = bias + 1.96 * std_d
    loa_lower = bias - 1.96 * std_d
    within = float(np.mean((diffs >= loa_lower) & (diffs <= loa_upper)) * 100.0)
    return {
        "bias_pa": bias,
        "std_pa": std_d,
        "loa_upper_pa": loa_upper,
        "loa_lower_pa": loa_lower,
        "pct_within_loa": within,
    }


def icc_one_way(pred: np.ndarray, ref: np.ndarray) -> float:
    """Intraclass correlation coefficient ICC(1,1) — one-way random effects.

    Treats each wall face as a subject rated by two "raters" (CFD and PINN).
    The ICC(1,1) formula is:

        ICC = (MS_b − MS_w) / (MS_b + MS_w)

    where MS_b = between-subjects mean square, MS_w = within-subjects mean square.

    Returns
    -------
    float
        ICC(1,1) ∈ [−1, 1].  >0.9 = excellent, 0.75–0.9 = good.
    """
    data = np.column_stack([pred.ravel(), ref.ravel()])   # (n, 2)
    n = data.shape[0]
    grand_mean = data.mean()
    row_means = data.mean(axis=1)                          # (n,)
    col_means = data.mean(axis=0)                          # (2,)

    # Between-subjects SS and MS
    ss_b = 2.0 * float(np.sum((row_means - grand_mean) ** 2))
    ms_b = ss_b / (n - 1)

    # Within-subjects SS and MS (residual after removing subject effect)
    ss_w = float(np.sum((data - row_means[:, None]) ** 2))
    ms_w = ss_w / n          # k=2 observations per subject

    return float((ms_b - ms_w) / (ms_b + ms_w + 1e-12))

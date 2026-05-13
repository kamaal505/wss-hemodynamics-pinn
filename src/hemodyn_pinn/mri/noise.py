"""Additive Gaussian noise model for synthetic 4D-flow MRI.

The noise model is:
    u_MRI = u_voxel + eta,    eta ~ N(0, sigma_v^2 I)

where sigma_v = VENC / VNR (velocity noise ratio).
"""

from __future__ import annotations

import numpy as np


def sigma_from_vnr(venc: float, vnr: float) -> float:
    """Compute noise standard deviation from VENC and VNR.

    Parameters
    ----------
    venc:
        Velocity encoding in m/s.  Set to (slightly above) the peak velocity
        magnitude in the domain.
    vnr:
        Velocity-to-noise ratio (dimensionless, > 0).  Higher = less noise.

    Returns
    -------
    float
        Noise standard deviation in m/s.
    """
    if vnr <= 0.0:
        raise ValueError(f"VNR must be positive, got {vnr}")
    return venc / vnr


def add_gaussian_noise(
    signal: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add independent Gaussian noise to each velocity component.

    Parameters
    ----------
    signal:
        (..., 3) array of velocity values in m/s.
    sigma:
        Noise standard deviation in m/s.  If 0.0, returns an unmodified copy.
    rng:
        NumPy random Generator from :func:`~hemodyn_pinn.utils.seeds.make_numpy_rng`.

    Returns
    -------
    np.ndarray
        (..., 3) array with the same dtype as *signal*.
    """
    if sigma < 0.0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")
    if sigma == 0.0:
        return signal.copy()
    noise = rng.standard_normal(signal.shape) * sigma
    return (signal + noise).astype(signal.dtype)

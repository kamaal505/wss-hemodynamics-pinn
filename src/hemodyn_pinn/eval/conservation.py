"""Flow-rate conservation diagnostics for PINN velocity fields.

Checks that the PINN respects mass conservation by comparing the count-weighted
mean normal velocity at the inlet and outlet faces.  Face areas are not stored
in the NPZ, so this is a proxy rather than an absolute flow rate; the ratio
is meaningful and used for Figure 9 diagnostics.
"""

from __future__ import annotations

import numpy as np


def flow_rate_conservation_error(
    u_inlet_ms: np.ndarray,
    n_inlet: np.ndarray,
    u_outlet_ms: np.ndarray,
    n_outlet: np.ndarray,
) -> dict[str, float]:
    """Estimate mass-conservation error from face-centroid velocities.

    Uses count-weighted mean normal velocity as a proxy for flow rate (valid
    when face areas are approximately uniform within each opening).

    Parameters
    ----------
    u_inlet_ms:
        (N_in, 3) velocity vectors [m/s] at inlet face centroids.
    n_inlet:
        (N_in, 3) outward unit normals at inlet faces.
        (Outward = pointing away from the fluid domain, i.e. opposite to flow.)
    u_outlet_ms:
        (N_out, 3) velocity vectors [m/s] at outlet face centroids.
    n_outlet:
        (N_out, 3) outward unit normals at outlet faces.
        (Outward = pointing away from the fluid domain, same direction as flow.)

    Returns
    -------
    dict with keys:
        q_in_proxy    — mean(u · n) at inlet (positive = inflow)
        q_out_proxy   — mean(u · n) at outlet (positive = outflow)
        conservation_error — |q_in − q_out| / q_in  (ideal = 0)
    """
    # Dot product at each face
    un_in = (u_inlet_ms * n_inlet).sum(axis=1)       # (N_in,)
    un_out = (u_outlet_ms * n_outlet).sum(axis=1)    # (N_out,)

    # Inlet outward normal opposes flow → negate so positive = inflow
    q_in = float(abs(np.mean(un_in)))
    q_out = float(abs(np.mean(un_out)))

    error = abs(q_in - q_out) / (q_in + 1e-12)

    return {
        "q_in_proxy": q_in,
        "q_out_proxy": q_out,
        "conservation_error": error,
    }


def z_profile_flow_rate(
    predict_fn,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_values: np.ndarray,
    n_samples: int = 500,
    rng_seed: int = 0,
) -> dict[str, np.ndarray]:
    """Estimate flow rate Q(z) along z-axis cross-sections.

    Samples a uniform grid of (x, y) points within [x_min, x_max] × [y_min, y_max]
    at each z value and averages u_z.  The result is proportional to Q(z) up to
    the unknown cross-sectional area.

    Parameters
    ----------
    predict_fn:
        Callable(x_nondim: np.ndarray) → (u_ms: np.ndarray, p_pa: np.ndarray).
        Must accept (N, 3) non-dimensional coordinates.
    x_min, x_max, y_min, y_max:
        Bounding box of the cross-sectional plane [non-dimensional units].
    z_values:
        (K,) non-dimensional z values at which to estimate Q.
    n_samples:
        Number of (x, y) sample points per cross-section.
    rng_seed:
        Seed for reproducible sampling.

    Returns
    -------
    dict with keys:
        z_nondim   — (K,) z positions
        q_proxy    — (K,) mean u_z at each cross-section (proxy for Q)
    """
    rng = np.random.default_rng(rng_seed)
    q_proxy = np.zeros(len(z_values))

    xs = rng.uniform(x_min, x_max, n_samples)
    ys = rng.uniform(y_min, y_max, n_samples)

    for k, z in enumerate(z_values):
        zs = np.full(n_samples, z)
        pts_nondim = np.column_stack([xs, ys, zs]).astype(np.float32)
        u_ms, _ = predict_fn(pts_nondim)
        q_proxy[k] = float(np.mean(u_ms[:, 2]))   # u_z component

    return {"z_nondim": np.asarray(z_values), "q_proxy": q_proxy}

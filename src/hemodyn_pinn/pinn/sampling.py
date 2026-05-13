"""Collocation and boundary point sampling utilities.

All functions operate in non-dimensional coordinates.  They return NumPy
arrays; conversion to tensors is done in the trainer.
"""

from __future__ import annotations

import numpy as np

from hemodyn_pinn.pinn.networks import L_SCALE, U_SCALE
from hemodyn_pinn.utils.seeds import PINN_COLLOC_SEED, make_numpy_rng


def to_nondim_coords(x_m: np.ndarray) -> np.ndarray:
    """Convert SI coordinates (m) to non-dimensional coordinates."""
    return x_m / L_SCALE


def to_nondim_velocity(u_ms: np.ndarray) -> np.ndarray:
    """Convert SI velocity (m/s) to non-dimensional velocity."""
    return u_ms / U_SCALE


def sample_collocation(
    interior_pts_nondim: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw a random subset of interior points as collocation points.

    Parameters
    ----------
    interior_pts_nondim:
        (M, 3) non-dimensional interior coordinates (mesh nodes).
    n:
        Number of collocation points to sample.
    rng:
        NumPy random generator (fresh per epoch — do NOT reuse across epochs).

    Returns
    -------
    np.ndarray
        (min(n, M), 3) sampled collocation points.
    """
    M = interior_pts_nondim.shape[0]
    n = min(n, M)
    idx = rng.choice(M, size=n, replace=False)
    return interior_pts_nondim[idx]


def sample_wall_bc(
    wall_pts_nondim: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw a random subset of wall centroid points.

    Parameters
    ----------
    wall_pts_nondim:
        (W, 3) non-dimensional wall-centroid coordinates.
    n:
        Number of wall BC points to sample.
    rng:
        NumPy random generator.

    Returns
    -------
    np.ndarray
        (min(n, W), 3) sampled wall points.
    """
    W = wall_pts_nondim.shape[0]
    n = min(n, W)
    idx = rng.choice(W, size=n, replace=False)
    return wall_pts_nondim[idx]


def make_pressure_anchor(outlet_pts_nondim: np.ndarray) -> np.ndarray:
    """Return a single outlet point for pressure anchoring.

    Picks the centroid closest to the mean of the outlet face.

    Parameters
    ----------
    outlet_pts_nondim:
        (K, 3) non-dimensional outlet face centroids.

    Returns
    -------
    np.ndarray
        (1, 3) anchor point.
    """
    mean_pt = outlet_pts_nondim.mean(axis=0, keepdims=True)
    dists = np.linalg.norm(outlet_pts_nondim - mean_pt, axis=1)
    idx = int(np.argmin(dists))
    return outlet_pts_nondim[[idx]]


def epoch_rng(base_seed: int, epoch: int) -> np.random.Generator:
    """Return a deterministic RNG for a given epoch.

    Using epoch-dependent seeds ensures collocation points differ each
    epoch while remaining reproducible across runs.

    Parameters
    ----------
    base_seed:
        Base seed (e.g. PINN_COLLOC_SEED).
    epoch:
        Current epoch index (0-based).
    """
    return make_numpy_rng(base_seed + epoch)

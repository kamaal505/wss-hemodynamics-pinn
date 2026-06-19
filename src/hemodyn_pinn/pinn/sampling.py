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


def sample_collocation_biased(
    interior_pts_nondim: np.ndarray,
    near_wall_pts_nondim: np.ndarray,
    n: int,
    wall_bias_frac: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw collocation points with a fraction concentrated near the wall.

    ``wall_bias_frac`` of the n points are drawn from ``near_wall_pts_nondim``
    (a pre-computed subset of interior nodes close to the vessel wall); the
    remainder come from the full interior pool.  Overlap between the two pools
    is acceptable — duplicate physics residual points are harmless.

    Parameters
    ----------
    interior_pts_nondim:
        (M, 3) full interior collocation pool.
    near_wall_pts_nondim:
        (K, 3) near-wall subset of interior nodes (K ≤ M).
    n:
        Total number of collocation points requested.
    wall_bias_frac:
        Fraction in [0, 1) drawn from the near-wall pool.
    rng:
        NumPy random generator (fresh per epoch).

    Returns
    -------
    np.ndarray
        (n, 3) collocation points.
    """
    K = near_wall_pts_nondim.shape[0]
    M = interior_pts_nondim.shape[0]

    n_near = min(int(n * wall_bias_frac), K)
    n_bulk = min(n - n_near, M)

    idx_near = rng.choice(K, size=n_near, replace=False)
    idx_bulk = rng.choice(M, size=n_bulk, replace=False)

    return np.concatenate(
        [near_wall_pts_nondim[idx_near], interior_pts_nondim[idx_bulk]], axis=0
    )


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


def select_inlet_targets(
    source: str,
    inlet_centroids_m: np.ndarray,
    voxel_centers_m: np.ndarray,
    voxel_velocity_ms: np.ndarray,
    node_points_m: "np.ndarray | None" = None,
    node_velocity_ms: "np.ndarray | None" = None,
    band_m: float = 2.0e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Build inlet-plane target points and velocities for the inflow constraint.

    The inflow term pins the flow magnitude, which is otherwise unpinned for
    linear homogeneous Stokes flow (see ``losses.inlet_loss``).

    Parameters
    ----------
    source:
        ``"mri"`` (honest, default): use the synthetic-MRI voxels whose
        centres fall within ``band_m`` of the inlet plane — in real 4D-flow
        MRI the inlet velocity is simply measured, so this is not
        ground-truth leakage.
        ``"cfd"`` (diagnostic upper bound): use the CFD nodal velocity at the
        mesh node nearest each inlet face centroid.  Leaks the ground-truth
        inlet profile — for diagnostics only.
        ``"none"``: returns empty arrays (no inflow term).
    inlet_centroids_m:
        (I, 3) inlet face centroids in metres.
    voxel_centers_m, voxel_velocity_ms:
        (V, 3) MRI voxel centres (metres) and velocities (m/s).
    node_points_m, node_velocity_ms:
        (N, 3) full-mesh node coordinates (metres) and velocities (m/s),
        required only when ``source == "cfd"``.
    band_m:
        Distance threshold from the inlet plane for the ``"mri"`` source.

    Returns
    -------
    x_inlet_m : np.ndarray
        (K, 3) inlet target coordinates in metres.
    u_inlet_ms : np.ndarray
        (K, 3) inlet target velocities in m/s.
    """
    if source == "none" or inlet_centroids_m.shape[0] == 0:
        return (
            np.empty((0, 3), dtype=float),
            np.empty((0, 3), dtype=float),
        )

    from scipy.spatial import cKDTree

    if source == "mri":
        tree = cKDTree(inlet_centroids_m)
        dists, _ = tree.query(voxel_centers_m)
        mask = dists <= band_m
        return voxel_centers_m[mask], voxel_velocity_ms[mask]

    if source == "cfd":
        if node_points_m is None or node_velocity_ms is None:
            raise ValueError("source='cfd' requires node_points_m and node_velocity_ms.")
        tree = cKDTree(node_points_m)
        _, idx = tree.query(inlet_centroids_m)
        return inlet_centroids_m, node_velocity_ms[idx]

    raise ValueError(f"Unknown inlet source '{source}'. Use 'mri', 'cfd', or 'none'.")


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

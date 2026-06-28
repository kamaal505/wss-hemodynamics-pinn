"""Dense velocity interpolant from sparse MRI voxels (anti-collapse prior).

Steady Stokes is linear and homogeneous, so u ≡ 0 is an exact zero-residual
solution.  The sparse MRI data (~445 voxels) is too thin to hold the field
magnitude against the physics residual, which rewards shrinkage everywhere it
is not anchored.  This module builds a *dense* velocity field by interpolating
the sparse voxels over the whole interior, to be used as:

    1. a warm-up supervised target (data-only curriculum), and
    2. a supervised term whose weight decays as physics takes over.

A dense non-zero magnitude target at every collocation point makes the trivial
collapse structurally impossible early in training.

The interpolant is built **only** from the measured MRI voxels — the same data
already available to the reconstruction — so it introduces no CFD ground-truth
leakage.

"Cubic spline" on scattered 3D points is not well defined (true splines need a
grid); the scattered-data analogue is a radial-basis-function interpolant
(thin-plate / cubic kernel), with a linear-barycentric fallback for robustness.
"""

from __future__ import annotations

import logging
from typing import Callable, Literal

import numpy as np

log = logging.getLogger(__name__)

InterpMethod = Literal["rbf", "linear"]


def build_velocity_interpolant(
    x_data_nondim: np.ndarray,
    u_obs_nondim: np.ndarray,
    method: InterpMethod = "rbf",
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a dense velocity interpolant from sparse MRI voxels.

    Parameters
    ----------
    x_data_nondim:
        (N_d, 3) non-dimensional voxel-centre coordinates.
    u_obs_nondim:
        (N_d, 3) non-dimensional observed velocities at those voxels.
    method:
        ``"rbf"`` (default) — scipy ``RBFInterpolator`` (thin-plate spline),
        the scattered-data analogue of a cubic spline.  ``"linear"`` —
        ``LinearNDInterpolator`` (barycentric), faster and monotone but only
        C0.  Both fall back to nearest-neighbour outside the convex hull.

    Returns
    -------
    Callable[[np.ndarray], np.ndarray]
        ``f(x_query_nondim) -> (M, 3)`` velocity predictions.
    """
    from scipy.interpolate import NearestNDInterpolator

    x = np.asarray(x_data_nondim, dtype=float)
    u = np.asarray(u_obs_nondim, dtype=float)
    nearest = NearestNDInterpolator(x, u)

    if method == "rbf":
        from scipy.interpolate import RBFInterpolator
        # thin-plate spline: smooth, no shape parameter to tune; neighbors caps
        # the linear system size on large voxel sets.
        n_neighbors = min(64, x.shape[0])
        rbf = RBFInterpolator(x, u, kernel="thin_plate_spline", neighbors=n_neighbors)

        def _interp(xq: np.ndarray) -> np.ndarray:
            xq = np.asarray(xq, dtype=float)
            return rbf(xq)

        return _interp

    if method == "linear":
        from scipy.interpolate import LinearNDInterpolator
        lin = LinearNDInterpolator(x, u)

        def _interp(xq: np.ndarray) -> np.ndarray:
            xq = np.asarray(xq, dtype=float)
            out = lin(xq)
            # Fill NaNs (queries outside the convex hull) with nearest-neighbour.
            bad = ~np.isfinite(out).all(axis=1)
            if bad.any():
                out[bad] = nearest(xq[bad])
            return out

        return _interp

    raise ValueError(f"Unknown interpolant method '{method}'. Use 'rbf' or 'linear'.")


def dense_aux_targets(
    interpolant: Callable[[np.ndarray], np.ndarray],
    interior_pts_nondim: np.ndarray,
    wall_pts_nondim: np.ndarray | None = None,
    chunk_size: int = 20_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the interpolant at interior nodes; inject no-slip at the wall.

    Parameters
    ----------
    interpolant:
        Callable from :func:`build_velocity_interpolant`.
    interior_pts_nondim:
        (M, 3) non-dimensional interior collocation pool.
    wall_pts_nondim:
        (W, 3) non-dimensional wall centroids.  When provided, these points are
        appended with **zero** target velocity so the dense prior respects the
        no-slip boundary condition instead of fighting it.
    chunk_size:
        Interior points evaluated per call.  An RBF interpolant solves a local
        linear system per query; evaluating ~10^5 mesh nodes at once is a memory
        spike, so the evaluation is streamed in chunks (no point is dropped).

    Returns
    -------
    x_aux : np.ndarray
        (M + W, 3) auxiliary target coordinates.
    u_aux : np.ndarray
        (M + W, 3) auxiliary target velocities (zero at the wall rows).
    """
    interior = np.asarray(interior_pts_nondim, dtype=float)
    M = interior.shape[0]
    cs = max(1, int(chunk_size))
    if M <= cs:
        u_interior = np.asarray(interpolant(interior), dtype=float)
    else:
        parts = [
            np.asarray(interpolant(interior[s:s + cs]), dtype=float)
            for s in range(0, M, cs)
        ]
        u_interior = np.concatenate(parts, axis=0)
    u_interior = np.nan_to_num(u_interior, nan=0.0, posinf=0.0, neginf=0.0)

    if wall_pts_nondim is None or wall_pts_nondim.shape[0] == 0:
        log.info("Dense aux targets: %d interior pts (no wall rows)", interior.shape[0])
        return interior, u_interior

    wall = np.asarray(wall_pts_nondim, dtype=float)
    u_wall = np.zeros_like(wall)
    x_aux = np.concatenate([interior, wall], axis=0)
    u_aux = np.concatenate([u_interior, u_wall], axis=0)
    log.info(
        "Dense aux targets: %d interior + %d wall (no-slip) = %d pts",
        interior.shape[0], wall.shape[0], x_aux.shape[0],
    )
    return x_aux, u_aux

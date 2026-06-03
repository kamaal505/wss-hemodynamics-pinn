"""Unsigned wall-distance function for the hard no-slip SDF constraint.

Computes the distance from any point in the domain to the nearest wall face
centroid using a KD-tree over the wall point cloud.  The result is strictly
non-negative (zero on the wall, positive inside the lumen) and is used to
multiply the network's raw velocity output so that no-slip is satisfied by
construction.

The SDF values are pre-computed as fixed floating-point scalars and are NOT
part of the PyTorch computation graph.  This means the autograd chain through
the physics residual ignores the ∂SDF/∂x term, which introduces an O(δ)
approximation error near the wall (where δ is the boundary-layer thickness).
For interior collocation points this error is negligible.  The principal
benefit — exact no-slip at the wall and well-conditioned WSS gradients — does
not require a differentiable SDF.  See notes/implementation_fixes_E_A_B_D.md.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree


class WallSDF:
    """Approximate unsigned distance from arbitrary lumen points to the wall.

    Parameters
    ----------
    wall_pts_m:
        (W, 3) wall face centroid coordinates in SI metres.  These are the
        AnXplore mesh wall centroids extracted from ``postprocess.py``.
    """

    def __init__(self, wall_pts_m: np.ndarray) -> None:
        self._tree = KDTree(wall_pts_m)

    def __call__(self, query_pts_m: np.ndarray) -> np.ndarray:
        """Return distances (m) from each query point to the nearest wall face.

        Parameters
        ----------
        query_pts_m:
            (N, 3) query coordinates in SI metres.

        Returns
        -------
        np.ndarray
            (N,) float32 distances in metres.  Zero for wall points, positive
            for interior/exterior points.
        """
        dists, _ = self._tree.query(query_pts_m)
        return dists.astype(np.float32)

    def nondim(self, query_pts_nondim: np.ndarray, l_scale: float) -> np.ndarray:
        """Return non-dimensional distances for non-dimensional query coords.

        Parameters
        ----------
        query_pts_nondim:
            (N, 3) non-dimensional coordinates (x / l_scale).
        l_scale:
            Length scale used for non-dimensionalisation (metres).

        Returns
        -------
        np.ndarray
            (N,) float32 non-dimensional distances.
        """
        return self(query_pts_nondim * l_scale) / l_scale

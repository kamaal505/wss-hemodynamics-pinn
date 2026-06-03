"""Tests for hemodyn_pinn.geometry.sdf.WallSDF (Fix A — hard no-slip constraint).

All tests run on CPU; no mesh files required.
"""

from __future__ import annotations

import numpy as np
import pytest

from hemodyn_pinn.geometry.sdf import WallSDF


@pytest.fixture()
def wall_pts() -> np.ndarray:
    """20 wall points on the z = 0 plane, x/y in [0, 1]."""
    rng = np.random.default_rng(0)
    pts = rng.uniform(0.0, 1.0, (20, 3)).astype(np.float32)
    pts[:, 2] = 0.0
    return pts


# ---------------------------------------------------------------------------
# __call__
# ---------------------------------------------------------------------------


class TestWallSDFCall:
    def test_output_shape(self, wall_pts: np.ndarray) -> None:
        sdf = WallSDF(wall_pts)
        q = np.random.default_rng(1).uniform(0, 1, (10, 3)).astype(np.float32)
        assert sdf(q).shape == (10,)

    def test_output_dtype_float32(self, wall_pts: np.ndarray) -> None:
        sdf = WallSDF(wall_pts)
        assert sdf(wall_pts[:3]).dtype == np.float32

    def test_wall_points_have_zero_distance(self, wall_pts: np.ndarray) -> None:
        """Every point in the wall cloud is its own nearest neighbour → dist = 0."""
        sdf = WallSDF(wall_pts)
        d = sdf(wall_pts)
        assert np.allclose(d, 0.0, atol=1e-7)

    def test_interior_point_positive(self, wall_pts: np.ndarray) -> None:
        """A point far from the wall must have strictly positive distance."""
        sdf = WallSDF(wall_pts)
        far_pt = np.array([[100.0, 100.0, 100.0]], dtype=np.float32)
        assert float(sdf(far_pt)[0]) > 0.0

    def test_all_distances_nonneg(self, wall_pts: np.ndarray) -> None:
        sdf = WallSDF(wall_pts)
        q = np.random.default_rng(2).uniform(-1.0, 2.0, (50, 3)).astype(np.float32)
        assert np.all(sdf(q) >= 0.0)

    def test_single_point_query(self, wall_pts: np.ndarray) -> None:
        sdf = WallSDF(wall_pts)
        q = np.array([[0.5, 0.5, 1.0]], dtype=np.float32)
        d = sdf(q)
        assert d.shape == (1,)
        assert float(d[0]) > 0.0


# ---------------------------------------------------------------------------
# nondim
# ---------------------------------------------------------------------------


class TestWallSDFNondim:
    def test_nondim_wall_points_zero(self, wall_pts: np.ndarray) -> None:
        """Wall points queried in non-dim space must return distance ≈ 0."""
        l_scale = 0.01628
        wall_m = wall_pts * l_scale
        sdf = WallSDF(wall_m)
        d = sdf.nondim(wall_pts, l_scale)
        assert np.allclose(d, 0.0, atol=1e-6)

    def test_nondim_output_shape(self, wall_pts: np.ndarray) -> None:
        l_scale = 0.01628
        sdf = WallSDF(wall_pts * l_scale)
        q = np.random.default_rng(3).uniform(0, 1, (8, 3)).astype(np.float32)
        assert sdf.nondim(q, l_scale).shape == (8,)

    def test_nondim_consistent_with_call(self, wall_pts: np.ndarray) -> None:
        """nondim(x̂, L) must equal call(x̂ * L) / L pointwise."""
        l_scale = 0.01628
        sdf = WallSDF(wall_pts * l_scale)
        q_nd = np.array([[0.3, 0.7, 0.5]], dtype=np.float32)
        d_nd = sdf.nondim(q_nd, l_scale)
        d_si = sdf(q_nd * l_scale) / l_scale
        assert np.allclose(d_nd, d_si, atol=1e-6)

"""Tests for hemodyn_pinn.pinn.interpolant (dense MRI interpolant prior).

The interpolant builds a dense velocity field from the sparse MRI voxels to
supply a non-zero magnitude target everywhere (anti-collapse, Fixes J–L).
Tests run on CPU and require only scipy/numpy.
"""

from __future__ import annotations

import numpy as np
import pytest

from hemodyn_pinn.pinn.interpolant import (
    build_velocity_interpolant,
    dense_aux_targets,
)


def _sparse(seed: int = 0, n: int = 60):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, (n, 3))
    u = rng.normal(0, 0.1, (n, 3))
    return x, u


@pytest.mark.parametrize("method", ["rbf", "linear"])
class TestBuildInterpolant:
    def test_reproduces_samples(self, method: str) -> None:
        """Interpolant evaluated at the sample points returns ~ the samples."""
        x, u = _sparse()
        interp = build_velocity_interpolant(x, u, method=method)
        u_at = interp(x)
        assert u_at.shape == u.shape
        # rbf with neighbors is an interpolant → near-exact at the nodes;
        # linear barycentric is exact at the nodes.
        assert np.allclose(u_at, u, atol=1e-4)

    def test_query_shape(self, method: str) -> None:
        x, u = _sparse()
        interp = build_velocity_interpolant(x, u, method=method)
        q = np.random.default_rng(3).uniform(-1, 1, (25, 3))
        out = interp(q)
        assert out.shape == (25, 3)
        assert np.isfinite(out).all()


class TestUnknownMethod:
    def test_raises(self) -> None:
        x, u = _sparse()
        with pytest.raises(ValueError, match="Unknown interpolant method"):
            build_velocity_interpolant(x, u, method="bogus")  # type: ignore[arg-type]


class TestDenseAuxTargets:
    def test_wall_rows_are_zero(self) -> None:
        x, u = _sparse()
        interp = build_velocity_interpolant(x, u, method="linear")
        interior = np.random.default_rng(5).uniform(-1, 1, (200, 3))
        wall = np.random.default_rng(6).uniform(-1, 1, (40, 3))
        x_aux, u_aux = dense_aux_targets(interp, interior, wall)
        assert x_aux.shape[0] == interior.shape[0] + wall.shape[0]
        assert np.allclose(u_aux[-wall.shape[0]:], 0.0)

    def test_no_wall_argument(self) -> None:
        x, u = _sparse()
        interp = build_velocity_interpolant(x, u, method="linear")
        interior = np.random.default_rng(7).uniform(-1, 1, (50, 3))
        x_aux, u_aux = dense_aux_targets(interp, interior, None)
        assert x_aux.shape == (50, 3)
        assert u_aux.shape == (50, 3)
        assert np.isfinite(u_aux).all()

    def test_targets_finite(self) -> None:
        """Queries outside the convex hull must be filled (no NaNs)."""
        x, u = _sparse()
        interp = build_velocity_interpolant(x, u, method="linear")
        # interior pts deliberately span beyond the data hull
        interior = np.random.default_rng(8).uniform(-3, 3, (120, 3))
        x_aux, u_aux = dense_aux_targets(interp, interior, None)
        assert np.isfinite(u_aux).all()

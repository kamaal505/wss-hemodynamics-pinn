"""Tests for sampling utilities and Fix E (wall-biased collocation fraction = 0.4).

Covers:
- sample_collocation: shape, cap at pool size, no duplicates.
- sample_collocation_biased: total count, near-wall fraction, reproducibility.
- epoch_rng: same epoch ↔ same sample; different epochs ↔ different samples.
"""

from __future__ import annotations

import numpy as np
import pytest

from hemodyn_pinn.pinn.sampling import (
    epoch_rng,
    sample_collocation,
    sample_collocation_biased,
    sample_wall_bc,
)


@pytest.fixture()
def interior() -> np.ndarray:
    return np.random.default_rng(0).uniform(0.0, 1.0, (200, 3)).astype(np.float32)


@pytest.fixture()
def near_wall(interior: np.ndarray) -> np.ndarray:
    return interior[:40]


# ---------------------------------------------------------------------------
# sample_collocation
# ---------------------------------------------------------------------------


class TestSampleCollocation:
    def test_output_shape(self, interior: np.ndarray) -> None:
        pts = sample_collocation(interior, n=50, rng=np.random.default_rng(1))
        assert pts.shape == (50, 3)

    def test_capped_at_pool_size(self, interior: np.ndarray) -> None:
        pts = sample_collocation(interior, n=10_000, rng=np.random.default_rng(2))
        assert pts.shape[0] == len(interior)

    def test_no_duplicate_rows(self, interior: np.ndarray) -> None:
        pts = sample_collocation(interior, n=50, rng=np.random.default_rng(3))
        unique = np.unique(pts, axis=0)
        assert len(unique) == len(pts)

    def test_rows_from_pool(self, interior: np.ndarray) -> None:
        pts = sample_collocation(interior, n=30, rng=np.random.default_rng(4))
        pool_set = set(map(tuple, interior.tolist()))
        for row in pts:
            assert tuple(row.tolist()) in pool_set


# ---------------------------------------------------------------------------
# sample_wall_bc
# ---------------------------------------------------------------------------


class TestSampleWallBC:
    def test_output_shape(self, interior: np.ndarray) -> None:
        pts = sample_wall_bc(interior[:50], n=20, rng=np.random.default_rng(5))
        assert pts.shape == (20, 3)

    def test_capped_at_pool_size(self, interior: np.ndarray) -> None:
        pts = sample_wall_bc(interior[:10], n=1_000, rng=np.random.default_rng(6))
        assert pts.shape[0] == 10


# ---------------------------------------------------------------------------
# sample_collocation_biased (Fix E)
# ---------------------------------------------------------------------------


class TestSampleCollocationBiased:
    def test_total_row_count(
        self, interior: np.ndarray, near_wall: np.ndarray
    ) -> None:
        pts = sample_collocation_biased(
            interior, near_wall, n=80, wall_bias_frac=0.4,
            rng=np.random.default_rng(10),
        )
        assert pts.shape == (80, 3)

    def test_near_wall_fraction_honoured(
        self, interior: np.ndarray, near_wall: np.ndarray
    ) -> None:
        """The first int(n * frac) rows must come from the near-wall pool."""
        n, frac = 100, 0.4
        pts = sample_collocation_biased(
            interior, near_wall, n=n, wall_bias_frac=frac,
            rng=np.random.default_rng(11),
        )
        n_near = int(n * frac)
        near_set = set(map(tuple, near_wall.tolist()))
        for row in pts[:n_near]:
            assert tuple(row.tolist()) in near_set

    def test_reproducible_with_same_seed(
        self, interior: np.ndarray, near_wall: np.ndarray
    ) -> None:
        kwargs = dict(n=60, wall_bias_frac=0.4)
        pts1 = sample_collocation_biased(
            interior, near_wall, **kwargs, rng=np.random.default_rng(99)
        )
        pts2 = sample_collocation_biased(
            interior, near_wall, **kwargs, rng=np.random.default_rng(99)
        )
        assert np.allclose(pts1, pts2)

    def test_zero_bias_same_as_uniform(
        self, interior: np.ndarray, near_wall: np.ndarray
    ) -> None:
        """wall_bias_frac=0 should draw everything from the interior pool."""
        pts = sample_collocation_biased(
            interior, near_wall, n=50, wall_bias_frac=0.0,
            rng=np.random.default_rng(12),
        )
        pool_set = set(map(tuple, interior.tolist()))
        for row in pts:
            assert tuple(row.tolist()) in pool_set


# ---------------------------------------------------------------------------
# epoch_rng
# ---------------------------------------------------------------------------


class TestEpochRng:
    def test_same_epoch_same_sample(self, interior: np.ndarray) -> None:
        pts1 = sample_collocation(interior, 20, epoch_rng(1000, 5))
        pts2 = sample_collocation(interior, 20, epoch_rng(1000, 5))
        assert np.allclose(pts1, pts2)

    def test_different_epochs_differ(self, interior: np.ndarray) -> None:
        pts1 = sample_collocation(interior, 20, epoch_rng(1000, 5))
        pts2 = sample_collocation(interior, 20, epoch_rng(1000, 6))
        assert not np.allclose(pts1, pts2)

    def test_different_base_seeds_differ(self, interior: np.ndarray) -> None:
        pts1 = sample_collocation(interior, 20, epoch_rng(1000, 0))
        pts2 = sample_collocation(interior, 20, epoch_rng(2000, 0))
        assert not np.allclose(pts1, pts2)

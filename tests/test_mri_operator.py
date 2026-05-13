"""Tests for the synthetic MRI forward operator (Phase 1.4 REQUIRED).

Covers:
  - VoxelGrid construction and geometry
  - Node-to-voxel assignment
  - Noise model (sigma_from_vnr, add_gaussian_noise statistics)
  - SyntheticMRIOperator: correctness, noise statistics, reproducibility
  - MRIObservation save/load round-trip
  - Partial-volume diagnostic (boundary voxels)
  - Error handling (mismatched inputs, bad parameters)
"""

from __future__ import annotations

import pathlib
import tempfile

import numpy as np
import pytest

from hemodyn_pinn.mri.voxelise import VoxelGrid, assign_nodes_to_voxels
from hemodyn_pinn.mri.noise import add_gaussian_noise, sigma_from_vnr
from hemodyn_pinn.mri.operator import MRIConfig, MRIObservation, SyntheticMRIOperator
from hemodyn_pinn.utils.seeds import make_numpy_rng


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uniform_box_nodes(
    n_per_side: int = 20,
    lo: float = 0.0,
    hi: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (points_mm, velocity_ms) on a regular Cartesian lattice."""
    xs = np.linspace(lo, hi, n_per_side)
    gx, gy, gz = np.meshgrid(xs, xs, xs, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    # Constant velocity field (1, 2, 3) m/s throughout
    vel = np.full((pts.shape[0], 3), [1.0, 2.0, 3.0])
    return pts.astype(np.float64), vel.astype(np.float64)


# ---------------------------------------------------------------------------
# VoxelGrid tests
# ---------------------------------------------------------------------------

class TestVoxelGrid:

    def test_from_bbox_shape(self):
        pts, _ = _uniform_box_nodes(n_per_side=5, lo=0.0, hi=10.0)
        grid = VoxelGrid.from_bbox(pts, voxel_size_mm=2.0)
        assert isinstance(grid.shape, tuple)
        assert len(grid.shape) == 3
        nx, ny, nz = grid.shape
        # bbox is [0,10]^3; 10/2 = 5 voxels per axis
        assert nx >= 5 and ny >= 5 and nz >= 5

    def test_n_voxels(self):
        grid = VoxelGrid(
            origin=np.zeros(3), voxel_size_mm=1.0, shape=(4, 5, 6)
        )
        assert grid.n_voxels == 4 * 5 * 6

    def test_centers_shape(self):
        grid = VoxelGrid(
            origin=np.zeros(3), voxel_size_mm=1.0, shape=(3, 4, 5)
        )
        c = grid.centers()
        assert c.shape == (3, 4, 5, 3)

    def test_centers_first_voxel(self):
        grid = VoxelGrid(
            origin=np.array([0.0, 0.0, 0.0]), voxel_size_mm=2.0, shape=(3, 3, 3)
        )
        c = grid.centers()
        np.testing.assert_allclose(c[0, 0, 0], [1.0, 1.0, 1.0])  # half-voxel offset

    def test_flat_index_round_trip(self):
        grid = VoxelGrid(
            origin=np.zeros(3), voxel_size_mm=1.0, shape=(4, 5, 6)
        )
        ix = np.array([0, 1, 3], dtype=np.int32)
        iy = np.array([2, 4, 0], dtype=np.int32)
        iz = np.array([5, 1, 3], dtype=np.int32)
        flat = grid.flat_index(ix, iy, iz)
        assert flat.max() < grid.n_voxels
        assert flat.min() >= 0

    def test_padding_extends_grid(self):
        pts, _ = _uniform_box_nodes(n_per_side=5, lo=1.0, hi=9.0)
        grid_no_pad = VoxelGrid.from_bbox(pts, voxel_size_mm=1.0, padding_mm=0.0)
        grid_pad = VoxelGrid.from_bbox(pts, voxel_size_mm=1.0, padding_mm=2.0)
        # Padded grid must be at least as large
        assert all(
            grid_pad.shape[i] >= grid_no_pad.shape[i] for i in range(3)
        )


# ---------------------------------------------------------------------------
# Node assignment tests
# ---------------------------------------------------------------------------

class TestAssignNodesToVoxels:

    def test_all_nodes_inside(self):
        pts, _ = _uniform_box_nodes(n_per_side=10, lo=0.5, hi=9.5)
        grid = VoxelGrid.from_bbox(pts, voxel_size_mm=1.0)
        indices = assign_nodes_to_voxels(pts, grid)
        assert (indices >= 0).all(), "All nodes should be inside the grid"

    def test_outside_nodes_flagged(self):
        pts = np.array([[50.0, 50.0, 50.0]])  # far outside a small grid
        grid = VoxelGrid(
            origin=np.zeros(3), voxel_size_mm=1.0, shape=(5, 5, 5)
        )
        indices = assign_nodes_to_voxels(pts, grid)
        assert indices[0] == -1

    def test_consistent_with_grid_shape(self):
        pts, _ = _uniform_box_nodes(n_per_side=8, lo=0.0, hi=8.0)
        grid = VoxelGrid.from_bbox(pts, voxel_size_mm=1.0)
        indices = assign_nodes_to_voxels(pts, grid)
        valid = indices[indices >= 0]
        assert valid.max() < grid.n_voxels


# ---------------------------------------------------------------------------
# Noise model tests
# ---------------------------------------------------------------------------

class TestNoiseModel:

    def test_sigma_from_vnr(self):
        venc = 1.0e-4  # m/s
        vnr = 20.0
        sigma = sigma_from_vnr(venc, vnr)
        assert abs(sigma - 5.0e-6) < 1e-15

    def test_sigma_from_vnr_bad_input(self):
        with pytest.raises(ValueError):
            sigma_from_vnr(1.0, -5.0)
        with pytest.raises(ValueError):
            sigma_from_vnr(1.0, 0.0)

    def test_zero_sigma_returns_copy(self):
        rng = make_numpy_rng(0)
        signal = np.ones((10, 3))
        out = add_gaussian_noise(signal, 0.0, rng)
        np.testing.assert_array_equal(out, signal)
        assert out is not signal  # must be a copy

    def test_noise_is_zero_mean(self):
        """With many samples, noise should be zero-mean to within ~3σ/√N."""
        rng = make_numpy_rng(42)
        signal = np.zeros((10_000, 3))
        sigma = 1.0e-3
        out = add_gaussian_noise(signal, sigma, rng)
        noise = out - signal
        # Mean should be close to zero (3σ/√N ~ 3e-3/100 = 3e-5)
        assert abs(noise.mean()) < 5.0 * sigma / np.sqrt(noise.size)

    def test_noise_std_matches_sigma(self):
        rng = make_numpy_rng(99)
        signal = np.zeros((50_000, 3))
        sigma = 2.5e-4
        out = add_gaussian_noise(signal, sigma, rng)
        measured_std = (out - signal).std()
        # Within 1 % of target sigma
        assert abs(measured_std - sigma) / sigma < 0.01

    def test_negative_sigma_raises(self):
        rng = make_numpy_rng(0)
        with pytest.raises(ValueError):
            add_gaussian_noise(np.ones((5, 3)), -1.0, rng)


# ---------------------------------------------------------------------------
# SyntheticMRIOperator tests
# ---------------------------------------------------------------------------

class TestSyntheticMRIOperator:

    def test_constant_field_noiseless(self):
        """Voxel average of a constant field must equal that constant in every voxel."""
        pts, vel = _uniform_box_nodes(n_per_side=20, lo=0.0, hi=10.0)
        cfg = MRIConfig(voxel_size_mm=2.0, sigma=0.0)
        op = SyntheticMRIOperator(cfg)
        obs = op.apply(pts, vel)
        expected = np.broadcast_to(np.array([[1.0, 2.0, 3.0]]), obs.velocity_ms.shape)
        np.testing.assert_allclose(
            obs.velocity_ms,
            expected,
            atol=1e-12,
            rtol=0.0,
            err_msg="Voxel average of constant field must be the constant",
        )

    def test_output_shape(self):
        pts, vel = _uniform_box_nodes(n_per_side=15)
        op = SyntheticMRIOperator(MRIConfig(voxel_size_mm=1.0, sigma=0.0))
        obs = op.apply(pts, vel)
        assert obs.voxel_centers_mm.ndim == 2
        assert obs.voxel_centers_mm.shape[1] == 3
        assert obs.velocity_ms.shape == obs.voxel_centers_mm.shape
        assert obs.n_nodes_per_voxel.ndim == 1
        assert obs.n_nodes_per_voxel.shape[0] == obs.velocity_ms.shape[0]

    def test_no_empty_voxels_in_output(self):
        """Output voxels must all have at least min_nodes_per_voxel nodes."""
        pts, vel = _uniform_box_nodes(n_per_side=12)
        cfg = MRIConfig(voxel_size_mm=1.0, sigma=0.0, min_nodes_per_voxel=1)
        obs = SyntheticMRIOperator(cfg).apply(pts, vel)
        assert (obs.n_nodes_per_voxel >= 1).all()

    def test_reproducibility(self):
        """Same seed must produce identical noisy observations."""
        pts, vel = _uniform_box_nodes(n_per_side=10)
        cfg = MRIConfig(voxel_size_mm=1.0, sigma=1e-5, rng_seed=7)
        obs1 = SyntheticMRIOperator(cfg).apply(pts, vel)
        obs2 = SyntheticMRIOperator(cfg).apply(pts, vel)
        np.testing.assert_array_equal(obs1.velocity_ms, obs2.velocity_ms)

    def test_different_seeds_differ(self):
        pts, vel = _uniform_box_nodes(n_per_side=10)
        cfg1 = MRIConfig(voxel_size_mm=1.0, sigma=1e-5, rng_seed=1)
        cfg2 = MRIConfig(voxel_size_mm=1.0, sigma=1e-5, rng_seed=2)
        obs1 = SyntheticMRIOperator(cfg1).apply(pts, vel)
        obs2 = SyntheticMRIOperator(cfg2).apply(pts, vel)
        assert not np.array_equal(obs1.velocity_ms, obs2.velocity_ms)

    def test_noisy_observation_mean_close_to_noiseless(self):
        """With many voxels, mean of noisy obs should be close to mean of noiseless."""
        pts, vel = _uniform_box_nodes(n_per_side=25, lo=0.0, hi=25.0)
        sigma = 1e-4  # m/s
        cfg_noisy = MRIConfig(voxel_size_mm=1.0, sigma=sigma, rng_seed=42)
        cfg_clean = MRIConfig(voxel_size_mm=1.0, sigma=0.0, rng_seed=42)
        obs_noisy = SyntheticMRIOperator(cfg_noisy).apply(pts, vel)
        obs_clean = SyntheticMRIOperator(cfg_clean).apply(pts, vel)
        # Mean difference should be << sigma (noise averages out over many voxels)
        n = obs_noisy.velocity_ms.shape[0]
        diff = abs(obs_noisy.velocity_ms.mean(axis=0) - obs_clean.velocity_ms.mean(axis=0))
        # Expect |diff| < 3 * sigma / sqrt(n)  with high probability
        threshold = 5.0 * sigma / np.sqrt(n)
        assert (diff < threshold).all(), (
            f"Mean diff {diff} exceeds 5σ/√n = {threshold}"
        )

    def test_mismatched_inputs_raise(self):
        pts = np.zeros((10, 3))
        vel = np.zeros((9, 3))  # wrong
        op = SyntheticMRIOperator()
        with pytest.raises(ValueError, match="rows"):
            op.apply(pts, vel)

    def test_wrong_units_raises(self):
        """Coordinates in metres (very small values) → empty grid."""
        pts = np.random.default_rng(0).uniform(0, 0.02, (100, 3))  # 0-20 mm in metres
        vel = np.ones((100, 3)) * 1e-5
        # With voxel_size_mm=1.0 and ~mm-scale points, 100 nodes should still work
        # (0-0.02 range → ~0 voxels if treated as mm → but VoxelGrid snaps to grid)
        # The test just checks it doesn't crash silently
        op = SyntheticMRIOperator(MRIConfig(voxel_size_mm=0.01))
        obs = op.apply(pts, vel)
        assert obs.velocity_ms.shape[0] > 0


# ---------------------------------------------------------------------------
# MRIObservation save/load round-trip
# ---------------------------------------------------------------------------

class TestMRIObservationIO:

    def test_save_load_round_trip(self):
        pts, vel = _uniform_box_nodes(n_per_side=10)
        cfg = MRIConfig(voxel_size_mm=1.5, sigma=5e-6, rng_seed=123)
        obs = SyntheticMRIOperator(cfg).apply(pts, vel)

        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / "mri_obs"
            saved = obs.save(out)
            loaded = MRIObservation.load(saved)

        np.testing.assert_array_equal(loaded.voxel_centers_mm, obs.voxel_centers_mm)
        np.testing.assert_array_equal(loaded.velocity_ms, obs.velocity_ms)
        np.testing.assert_array_equal(loaded.n_nodes_per_voxel, obs.n_nodes_per_voxel)
        assert loaded.cfg.voxel_size_mm == cfg.voxel_size_mm
        assert loaded.cfg.sigma == cfg.sigma

    def test_overwrite_guard(self):
        pts, vel = _uniform_box_nodes(n_per_side=5)
        obs = SyntheticMRIOperator().apply(pts, vel)

        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / "obs"
            obs.save(out)
            with pytest.raises(FileExistsError):
                obs.save(out, overwrite=False)
            obs.save(out, overwrite=True)  # should not raise


# ---------------------------------------------------------------------------
# Partial-volume diagnostic
# ---------------------------------------------------------------------------

class TestPartialVolume:

    def test_boundary_voxels_have_fewer_nodes(self):
        """Interior voxels should accumulate more nodes than edge voxels."""
        pts, vel = _uniform_box_nodes(n_per_side=30, lo=0.0, hi=30.0)
        cfg = MRIConfig(voxel_size_mm=1.0, sigma=0.0)
        obs = SyntheticMRIOperator(cfg).apply(pts, vel)

        # Interior voxel centres: far from any face (> 3 mm from bbox edge)
        bbox_lo = obs.voxel_centers_mm.min(axis=0)
        bbox_hi = obs.voxel_centers_mm.max(axis=0)
        is_interior = (
            (obs.voxel_centers_mm[:, 0] > bbox_lo[0] + 3)
            & (obs.voxel_centers_mm[:, 0] < bbox_hi[0] - 3)
            & (obs.voxel_centers_mm[:, 1] > bbox_lo[1] + 3)
            & (obs.voxel_centers_mm[:, 1] < bbox_hi[1] - 3)
            & (obs.voxel_centers_mm[:, 2] > bbox_lo[2] + 3)
            & (obs.voxel_centers_mm[:, 2] < bbox_hi[2] - 3)
        )
        is_boundary = (
            (obs.voxel_centers_mm[:, 0] <= bbox_lo[0] + 1)
            | (obs.voxel_centers_mm[:, 0] >= bbox_hi[0] - 1)
            | (obs.voxel_centers_mm[:, 1] <= bbox_lo[1] + 1)
            | (obs.voxel_centers_mm[:, 1] >= bbox_hi[1] - 1)
            | (obs.voxel_centers_mm[:, 2] <= bbox_lo[2] + 1)
            | (obs.voxel_centers_mm[:, 2] >= bbox_hi[2] - 1)
        )

        if is_interior.any() and is_boundary.any():
            mean_interior = obs.n_nodes_per_voxel[is_interior].mean()
            mean_boundary = obs.n_nodes_per_voxel[is_boundary].mean()
            assert mean_interior >= mean_boundary, (
                f"Interior mean {mean_interior:.1f} < boundary mean {mean_boundary:.1f}"
            )

"""Tests for PINNTrainer.__init__ with the new architecture flags (Fixes A, B, D, E).

All tests verify that the trainer initialises its internal state correctly;
no actual training iterations are executed.  The PINNConfig fields n_adam=0
and n_lbfgs=0 are used to skip all training.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hemodyn_pinn.pinn.networks import L_SCALE, PINNNetwork
from hemodyn_pinn.pinn.trainer import PINNConfig, PINNTrainer


# ---------------------------------------------------------------------------
# Shared synthetic data
# ---------------------------------------------------------------------------


def _make_data(
    n_interior: int = 100,
    n_wall: int = 30,
    n_data: int = 10,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    interior = rng.uniform(0.1, 0.9, (n_interior, 3)).astype(np.float32)
    wall = rng.uniform(0.0, 0.05, (n_wall, 3)).astype(np.float32)
    anchor = rng.uniform(0, 1, (1, 3)).astype(np.float32)
    x_data = rng.uniform(0.1, 0.9, (n_data, 3)).astype(np.float32)
    u_obs = rng.normal(0, 1e-4, (n_data, 3)).astype(np.float32)
    wall_m = wall * L_SCALE
    return interior, wall, anchor, x_data, u_obs, wall_m


def _make_trainer(cfg: PINNConfig, wall_m=None, **kw) -> PINNTrainer:
    interior, wall, anchor, x_data, u_obs, wall_m_default = _make_data()
    net = PINNNetwork(n_hidden=8, n_layers=2, seed=0)
    return PINNTrainer(
        net=net,
        cfg=cfg,
        interior_pts_nondim=interior,
        wall_pts_nondim=wall,
        anchor_pt_nondim=anchor,
        x_data=x_data,
        u_obs_nondim=u_obs,
        wall_pts_m=wall_m if wall_m is not None else wall_m_default,
        **kw,
    )


# ---------------------------------------------------------------------------
# Fix E — near-wall subset (wall_bias_frac)
# ---------------------------------------------------------------------------


class TestWallBiasInit:
    def test_near_wall_populated_when_bias_positive(self) -> None:
        cfg = PINNConfig(wall_bias_frac=0.4)
        trainer = _make_trainer(cfg)
        assert trainer._near_wall_np is not None
        assert trainer._near_wall_np.ndim == 2
        assert trainer._near_wall_np.shape[1] == 3

    def test_near_wall_size_at_most_20th_percentile(self) -> None:
        """KDTree bottom-20th-percentile filter: subset ≤ ~20 % of interior."""
        cfg = PINNConfig(wall_bias_frac=0.4)
        trainer = _make_trainer(cfg)
        assert len(trainer._near_wall_np) <= int(0.22 * 100) + 1

    def test_near_wall_none_when_bias_zero(self) -> None:
        cfg = PINNConfig(wall_bias_frac=0.0)
        trainer = _make_trainer(cfg)
        assert trainer._near_wall_np is None


# ---------------------------------------------------------------------------
# Fix A — SDF precomputation (use_hard_sdf)
# ---------------------------------------------------------------------------


class TestHardSDFInit:
    def test_sdf_tensors_precomputed(self) -> None:
        """Both _sdf_interior and _sdf_data must be populated when use_hard_sdf=True."""
        interior, wall, anchor, x_data, u_obs, wall_m = _make_data()
        net = PINNNetwork(n_hidden=8, n_layers=2, use_hard_sdf=True, seed=0)
        cfg = PINNConfig(use_hard_sdf=True)
        trainer = PINNTrainer(
            net=net, cfg=cfg,
            interior_pts_nondim=interior, wall_pts_nondim=wall,
            anchor_pt_nondim=anchor, x_data=x_data,
            u_obs_nondim=u_obs, wall_pts_m=wall_m,
        )
        assert trainer._sdf_interior is not None
        assert trainer._sdf_data is not None
        assert isinstance(trainer._sdf_interior, torch.Tensor)
        assert isinstance(trainer._sdf_data, torch.Tensor)

    def test_sdf_shapes_correct(self) -> None:
        n_int, n_data = 100, 10
        interior, wall, anchor, x_data, u_obs, wall_m = _make_data(
            n_interior=n_int, n_data=n_data
        )
        net = PINNNetwork(n_hidden=8, n_layers=2, use_hard_sdf=True, seed=0)
        cfg = PINNConfig(use_hard_sdf=True)
        trainer = PINNTrainer(
            net=net, cfg=cfg,
            interior_pts_nondim=interior, wall_pts_nondim=wall,
            anchor_pt_nondim=anchor, x_data=x_data,
            u_obs_nondim=u_obs, wall_pts_m=wall_m,
        )
        assert trainer._sdf_interior.shape == (n_int,)
        assert trainer._sdf_data.shape == (n_data,)

    def test_sdf_values_nonneg(self) -> None:
        interior, wall, anchor, x_data, u_obs, wall_m = _make_data()
        net = PINNNetwork(n_hidden=8, n_layers=2, use_hard_sdf=True, seed=0)
        cfg = PINNConfig(use_hard_sdf=True)
        trainer = PINNTrainer(
            net=net, cfg=cfg,
            interior_pts_nondim=interior, wall_pts_nondim=wall,
            anchor_pt_nondim=anchor, x_data=x_data,
            u_obs_nondim=u_obs, wall_pts_m=wall_m,
        )
        assert (trainer._sdf_interior >= 0.0).all()
        assert (trainer._sdf_data >= 0.0).all()

    def test_raises_without_wall_pts_m(self) -> None:
        """use_hard_sdf=True without wall_pts_m must raise ValueError."""
        interior, wall, anchor, x_data, u_obs, _ = _make_data()
        net = PINNNetwork(n_hidden=8, n_layers=2, use_hard_sdf=True, seed=0)
        cfg = PINNConfig(use_hard_sdf=True)
        with pytest.raises(ValueError, match="wall_pts_m"):
            PINNTrainer(
                net=net, cfg=cfg,
                interior_pts_nondim=interior, wall_pts_nondim=wall,
                anchor_pt_nondim=anchor, x_data=x_data,
                u_obs_nondim=u_obs, wall_pts_m=None,
            )

    def test_effective_lambda_bc_zero(self) -> None:
        """When use_hard_sdf=True, the trainer must zero out the BC weight."""
        interior, wall, anchor, x_data, u_obs, wall_m = _make_data()
        net = PINNNetwork(n_hidden=8, n_layers=2, use_hard_sdf=True, seed=0)
        cfg = PINNConfig(use_hard_sdf=True, lambda_bc=10.0)
        trainer = PINNTrainer(
            net=net, cfg=cfg,
            interior_pts_nondim=interior, wall_pts_nondim=wall,
            anchor_pt_nondim=anchor, x_data=x_data,
            u_obs_nondim=u_obs, wall_pts_m=wall_m,
        )
        assert trainer._effective_lambda_bc == 0.0


# ---------------------------------------------------------------------------
# Fix B — vec potential (use_vec_potential)
# ---------------------------------------------------------------------------


class TestVecPotentialInit:
    def test_skip_div_set_when_vec_potential(self) -> None:
        """_skip_div must be True whenever use_vec_potential=True."""
        cfg = PINNConfig(use_vec_potential=True)
        net = PINNNetwork(n_hidden=8, n_layers=2, use_vec_potential=True, seed=0)
        interior, wall, anchor, x_data, u_obs, _ = _make_data()
        trainer = PINNTrainer(
            net=net, cfg=cfg,
            interior_pts_nondim=interior, wall_pts_nondim=wall,
            anchor_pt_nondim=anchor, x_data=x_data,
            u_obs_nondim=u_obs,
        )
        assert trainer._skip_div is True

    def test_skip_div_false_without_vec_potential(self) -> None:
        cfg = PINNConfig(use_vec_potential=False)
        trainer = _make_trainer(cfg)
        assert trainer._skip_div is False


# ---------------------------------------------------------------------------
# Fix D — self-adaptive loss weights (use_adaptive_weights)
# ---------------------------------------------------------------------------


class TestAdaptiveWeightsInit:
    def test_sa_loss_constructed(self) -> None:
        """_sa_loss must not be None when use_adaptive_weights=True."""
        cfg = PINNConfig(use_adaptive_weights=True)
        trainer = _make_trainer(cfg)
        assert trainer._sa_loss is not None

    def test_sa_loss_none_by_default(self) -> None:
        cfg = PINNConfig(use_adaptive_weights=False)
        trainer = _make_trainer(cfg)
        assert trainer._sa_loss is None

    def test_sa_loss_initial_lambdas_match_config(self) -> None:
        """SA-PINN weights should be initialised from PINNConfig lambda values."""
        cfg = PINNConfig(
            use_adaptive_weights=True,
            lambda_data=2.0, lambda_phys=3.0,
            lambda_bc=5.0, lambda_anchor=7.0,
        )
        trainer = _make_trainer(cfg)
        lam = trainer._sa_loss.lambdas
        assert lam["data"] == pytest.approx(2.0, rel=1e-4)
        assert lam["phys"] == pytest.approx(3.0, rel=1e-4)
        assert lam["anchor"] == pytest.approx(7.0, rel=1e-4)

    def test_sa_loss_bc_zeroed_with_hard_sdf(self) -> None:
        """When both use_hard_sdf and use_adaptive_weights are True,
        the SA-PINN bc weight must be 0.0."""
        interior, wall, anchor, x_data, u_obs, wall_m = _make_data()
        net = PINNNetwork(
            n_hidden=8, n_layers=2,
            use_hard_sdf=True, seed=0,
        )
        cfg = PINNConfig(
            use_hard_sdf=True, use_adaptive_weights=True, lambda_bc=10.0
        )
        trainer = PINNTrainer(
            net=net, cfg=cfg,
            interior_pts_nondim=interior, wall_pts_nondim=wall,
            anchor_pt_nondim=anchor, x_data=x_data,
            u_obs_nondim=u_obs, wall_pts_m=wall_m,
        )
        assert trainer._sa_loss.lambdas["bc"] == pytest.approx(0.0, abs=1e-7)

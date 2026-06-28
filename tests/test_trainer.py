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


# ---------------------------------------------------------------------------
# Fixes J–L — curriculum warm-up + decaying dense-interpolant aux weight
# ---------------------------------------------------------------------------


def _make_aux(n_aux: int = 80, seed: int = 1):
    rng = np.random.default_rng(seed)
    x_aux = rng.uniform(0.1, 0.9, (n_aux, 3)).astype(np.float32)
    u_aux = rng.normal(0, 1e-4, (n_aux, 3)).astype(np.float32)
    return x_aux, u_aux


class TestCurriculumAux:
    def test_target_rms_precomputed(self) -> None:
        cfg = PINNConfig()
        trainer = _make_trainer(cfg)
        _, _, _, _, u_obs, _ = _make_data()
        expected = float(np.sqrt(np.mean(np.sum(u_obs ** 2, axis=1))))
        assert trainer._target_rms == pytest.approx(expected, rel=1e-5)

    def test_aux_flags_and_tensors(self) -> None:
        x_aux, u_aux = _make_aux()
        cfg = PINNConfig(lambda_aux=5.0, lambda_mag_floor=1.0)
        trainer = _make_trainer(cfg, x_aux_nondim=x_aux, u_aux_nondim=u_aux)
        assert trainer._use_aux is True
        assert trainer._use_mag_floor is True
        assert trainer.x_aux is not None and trainer.x_aux.shape == (80, 3)

    def test_aux_disabled_when_weight_zero(self) -> None:
        x_aux, u_aux = _make_aux()
        cfg = PINNConfig(lambda_aux=0.0)
        trainer = _make_trainer(cfg, x_aux_nondim=x_aux, u_aux_nondim=u_aux)
        assert trainer._use_aux is False

    def test_aux_weight_decays_to_zero(self) -> None:
        x_aux, u_aux = _make_aux()
        cfg = PINNConfig(lambda_aux=5.0, aux_decay_frac=0.5, n_adam=100)
        trainer = _make_trainer(cfg, x_aux_nondim=x_aux, u_aux_nondim=u_aux)
        assert trainer._aux_weight(0) == pytest.approx(5.0, rel=1e-6)
        assert trainer._aux_weight(25) == pytest.approx(2.5, rel=1e-3)
        assert trainer._aux_weight(50) == pytest.approx(0.0, abs=1e-6)
        assert trainer._aux_weight(100) == pytest.approx(0.0, abs=1e-6)

    def test_warmup_zeroes_physics(self) -> None:
        """During warm-up the physics term must not contribute to the total.

        Aux/mag-floor are disabled here so the physics contribution is not
        swamped by the (relative) aux loss on the tiny synthetic velocities.
        """
        interior, wall, anchor, x_data, u_obs, wall_m = _make_data()
        net = PINNNetwork(n_hidden=8, n_layers=2, seed=0)
        # Isolate physics (zero the other terms) so the warm-up effect is exact
        # and not lost to float32 cancellation against a huge relative data loss.
        cfg = PINNConfig(n_adam=4, n_lbfgs=0, n_warmup=2,
                         lambda_data=0.0, lambda_phys=1.0, lambda_bc=0.0,
                         lambda_anchor=0.0, lambda_aux=0.0, lambda_mag_floor=0.0)
        trainer = PINNTrainer(
            net=net, cfg=cfg,
            interior_pts_nondim=interior, wall_pts_nondim=wall,
            anchor_pt_nondim=anchor, x_data=x_data, u_obs_nondim=u_obs,
        )
        x_c, x_w, sdf_c = trainer._sample_batch(0)
        total_warm, bd = trainer._compute_loss(x_c, x_w, sdf_c, lambda_phys_eff=0.0)
        total_phys, _ = trainer._compute_loss(x_c, x_w, sdf_c, lambda_phys_eff=1.0)
        # Physics off ⇒ total is ~0; physics on ⇒ total = loss_phys > 0.
        assert float(total_warm.detach()) == pytest.approx(0.0, abs=1e-6)
        assert float(total_phys.detach()) > 0.0
        assert "loss_aux" in bd and "loss_mag_floor" in bd

    def test_aux_batch_streams_subset(self) -> None:
        """Per-epoch aux batch is capped at n_aux and varies across epochs."""
        x_aux, u_aux = _make_aux(n_aux=300)
        cfg = PINNConfig(lambda_aux=5.0, n_aux=64)
        trainer = _make_trainer(cfg, x_aux_nondim=x_aux, u_aux_nondim=u_aux)
        b0 = trainer._aux_batch(0)
        b1 = trainer._aux_batch(1)
        assert b0 is not None and b0[0].shape == (64, 3)
        assert b0[1].shape == (64, 3)
        # different epoch → different subset (resampled)
        assert not torch.equal(b0[0], b1[0])

    def test_aux_batch_uses_all_when_cap_large(self) -> None:
        x_aux, u_aux = _make_aux(n_aux=50)
        cfg = PINNConfig(lambda_aux=5.0, n_aux=0)   # 0 → use all
        trainer = _make_trainer(cfg, x_aux_nondim=x_aux, u_aux_nondim=u_aux)
        b = trainer._aux_batch(0)
        assert b is not None and b[0].shape[0] == 50

    def test_aux_batch_none_when_disabled(self) -> None:
        cfg = PINNConfig(lambda_aux=0.0, n_aux=64)
        trainer = _make_trainer(cfg)
        assert trainer._aux_batch(0) is None

    def test_short_run_completes(self) -> None:
        x_aux, u_aux = _make_aux()
        interior, wall, anchor, x_data, u_obs, wall_m = _make_data()
        net = PINNNetwork(n_hidden=8, n_layers=2, seed=0)
        cfg = PINNConfig(n_adam=6, n_lbfgs=0, n_warmup=2, checkpoint_every=1000,
                         lambda_aux=5.0, aux_decay_frac=0.5, lambda_mag_floor=1.0)
        trainer = PINNTrainer(
            net=net, cfg=cfg,
            interior_pts_nondim=interior, wall_pts_nondim=wall,
            anchor_pt_nondim=anchor, x_data=x_data, u_obs_nondim=u_obs,
            x_aux_nondim=x_aux, u_aux_nondim=u_aux,
        )
        trainer.run_adam()
        assert len(trainer.history) == 6
        assert "loss_aux" in trainer.history[0]

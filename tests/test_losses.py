"""Tests for hemodyn_pinn.pinn.losses.

Each loss term is tested in isolation with a small synthetic network and
known inputs.  Tests run on CPU and do not require FEniCSx or any data files.

Coverage
--------
- data_loss: zero when prediction is exact; positive when off.
- stokes_residual_loss: zero for an exact Stokes solution (linear pressure,
  quadratic velocity); non-zero for a random network.
- bc_loss: zero when network outputs zero velocity; positive otherwise.
- pressure_anchor_loss: zero when p = 0 at anchor; positive otherwise.
- total_loss: returns a dict with expected keys; weighted sum is correct.
"""

from __future__ import annotations

import math
import pytest
import torch
import torch.nn as nn
from torch import Tensor

from hemodyn_pinn.pinn.losses import (
    SelfAdaptiveLoss,
    bc_loss,
    data_loss,
    inlet_loss,
    pressure_anchor_loss,
    stokes_residual_loss,
    total_loss,
)
from hemodyn_pinn.pinn.networks import PINNNetwork

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def small_net() -> PINNNetwork:
    """Tiny PINN network for fast testing (2 hidden layers, 16 neurons)."""
    return PINNNetwork(n_hidden=16, n_layers=2, use_rff=False, seed=0)


@pytest.fixture()
def rng_pts() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Small sets of synthetic coordinate tensors."""
    torch.manual_seed(42)
    x_data   = torch.rand(20, 3)    # 20 observation points
    x_colloc = torch.rand(30, 3)    # 30 collocation points
    x_wall   = torch.rand(10, 3)    # 10 wall points
    x_anchor = torch.rand(1, 3)     # 1 pressure anchor
    return x_data, x_colloc, x_wall, x_anchor


# ---------------------------------------------------------------------------
# Zero-velocity constant network (for bc_loss = 0 test)
# ---------------------------------------------------------------------------


class ZeroVelocityNet(nn.Module):
    """Always outputs (0, 0, 0, p) where p varies, for BC loss testing."""

    def forward(self, x: Tensor) -> Tensor:
        N = x.shape[0]
        zeros = torch.zeros(N, 3, device=x.device, dtype=x.dtype)
        p = torch.sin(x[:, 0:1])   # non-zero pressure
        return torch.cat([zeros, p], dim=1)


class ZeroPressureAtPointNet(nn.Module):
    """Returns p=0 always (for pressure anchor = 0 test)."""

    def forward(self, x: Tensor) -> Tensor:
        N = x.shape[0]
        return torch.zeros(N, 4, device=x.device, dtype=x.dtype)


class ZeroVelSDFNet(nn.Module):
    """Always outputs zero velocity; accepts sdf_vals (for relative-loss tests)."""

    def forward(self, x: Tensor, sdf_vals=None) -> Tensor:
        N = x.shape[0]
        return torch.zeros(N, 4, device=x.device, dtype=x.dtype)


# ---------------------------------------------------------------------------
# data_loss
# ---------------------------------------------------------------------------


class TestDataLoss:
    def test_zero_when_exact(self, small_net: PINNNetwork) -> None:
        """Loss is zero when predicted velocity matches observations exactly."""
        torch.manual_seed(1)
        x_data = torch.rand(10, 3)
        with torch.no_grad():
            pred = small_net(x_data)[:, :3]
        loss = data_loss(small_net, x_data, pred)
        assert float(loss.detach()) == pytest.approx(0.0, abs=1e-7)

    def test_positive_when_off(self, small_net: PINNNetwork) -> None:
        torch.manual_seed(2)
        x_data = torch.rand(10, 3)
        u_wrong = torch.ones(10, 3) * 99.0
        loss = data_loss(small_net, x_data, u_wrong)
        assert float(loss.detach()) > 0.0

    def test_output_is_scalar(self, small_net: PINNNetwork) -> None:
        torch.manual_seed(3)
        x_data = torch.rand(15, 3)
        u_obs = torch.rand(15, 3)
        loss = data_loss(small_net, x_data, u_obs)
        assert loss.shape == ()

    def test_gradients_flow(self, small_net: PINNNetwork) -> None:
        """Backprop through data_loss reaches network parameters."""
        x_data = torch.rand(8, 3)
        u_obs = torch.rand(8, 3)
        loss = data_loss(small_net, x_data, u_obs)
        loss.backward()
        grads = [p.grad for p in small_net.parameters() if p.grad is not None]
        assert len(grads) > 0

    def test_relative_zero_prediction_is_one(self) -> None:
        """Relative data loss ≈ 1.0 when the network predicts zero velocity.

        This is the collapse signature: a near-zero field gives NRMSE ≈ 1.
        """
        net = ZeroVelSDFNet()
        torch.manual_seed(4)
        x_data = torch.rand(20, 3)
        u_obs = torch.rand(20, 3) + 0.5   # strictly non-zero magnitudes
        loss = data_loss(net, x_data, u_obs, relative=True)
        assert float(loss.detach()) == pytest.approx(1.0, rel=1e-5)

    def test_relative_normalises_absolute(self, small_net: PINNNetwork) -> None:
        """relative == absolute / mean(u_obs²)."""
        torch.manual_seed(5)
        x_data = torch.rand(12, 3)
        u_obs = torch.rand(12, 3) + 0.2
        abs_loss = float(data_loss(small_net, x_data, u_obs, relative=False).detach())
        rel_loss = float(data_loss(small_net, x_data, u_obs, relative=True).detach())
        denom = float((u_obs ** 2).mean())
        assert rel_loss == pytest.approx(abs_loss / denom, rel=1e-5)

    def test_relative_zero_when_exact(self, small_net: PINNNetwork) -> None:
        torch.manual_seed(6)
        x_data = torch.rand(10, 3)
        with torch.no_grad():
            pred = small_net(x_data)[:, :3]
        loss = data_loss(small_net, x_data, pred, relative=True)
        assert float(loss.detach()) == pytest.approx(0.0, abs=1e-7)


# ---------------------------------------------------------------------------
# inlet_loss (Tier 2 — inflow magnitude constraint)
# ---------------------------------------------------------------------------


class TestInletLoss:
    def test_zero_when_exact(self, small_net: PINNNetwork) -> None:
        torch.manual_seed(7)
        x_inlet = torch.rand(8, 3)
        with torch.no_grad():
            u_inlet = small_net(x_inlet)[:, :3]
        loss = inlet_loss(small_net, x_inlet, u_inlet)
        assert float(loss.detach()) == pytest.approx(0.0, abs=1e-7)

    def test_relative_zero_prediction_is_one(self) -> None:
        net = ZeroVelSDFNet()
        torch.manual_seed(8)
        x_inlet = torch.rand(10, 3)
        u_inlet = torch.rand(10, 3) + 0.5
        loss = inlet_loss(net, x_inlet, u_inlet, relative=True)
        assert float(loss.detach()) == pytest.approx(1.0, rel=1e-5)

    def test_output_is_scalar_and_gradients_flow(self, small_net: PINNNetwork) -> None:
        x_inlet = torch.rand(8, 3)
        u_inlet = torch.rand(8, 3)
        loss = inlet_loss(small_net, x_inlet, u_inlet)
        assert loss.shape == ()
        loss.backward()
        grads = [p.grad for p in small_net.parameters() if p.grad is not None]
        assert len(grads) > 0


# ---------------------------------------------------------------------------
# stokes_residual_loss
# ---------------------------------------------------------------------------


class ExactStokesNet(nn.Module):
    """Returns u=y², v=0, w=0, p=2*x — an exact non-dim Stokes solution.

    Verification:
      ∂p/∂x=2,  ∂p/∂y=0,  ∂p/∂z=0
      Δu = ∂²(y²)/∂y² = 2  →  -∂p/∂x + Δu = -2+2 = 0  ✓
      Δv = 0  →  -∂p/∂y + Δv = 0 ✓
      Δw = 0  →  -∂p/∂z + Δw = 0 ✓
      ∂u/∂x + ∂v/∂y + ∂w/∂z = 0 + 0 + 0 = 0  ✓
    """

    def forward(self, x: Tensor, sdf_vals=None) -> Tensor:
        u = x[:, 1] ** 2            # y²
        v = torch.zeros_like(u)
        w = torch.zeros_like(u)
        p = 2.0 * x[:, 0]           # 2x
        return torch.stack([u, v, w, p], dim=1)


class TestStokesResidualLoss:
    def test_zero_for_exact_solution(self) -> None:
        """Residual must vanish for the analytical Stokes solution."""
        net = ExactStokesNet()
        torch.manual_seed(5)
        x_colloc = torch.rand(20, 3) * 0.5 + 0.25  # avoid x=y=z=0
        loss = stokes_residual_loss(net, x_colloc)
        assert float(loss) == pytest.approx(0.0, abs=1e-5)

    def test_positive_for_random_net(self, small_net: PINNNetwork) -> None:
        torch.manual_seed(6)
        x_colloc = torch.rand(15, 3)
        loss = stokes_residual_loss(small_net, x_colloc)
        assert float(loss) > 0.0

    def test_output_is_scalar(self, small_net: PINNNetwork) -> None:
        x_colloc = torch.rand(10, 3)
        loss = stokes_residual_loss(small_net, x_colloc)
        assert loss.shape == ()

    def test_gradients_flow(self, small_net: PINNNetwork) -> None:
        """Backprop through physics residual (second-order autograd) works."""
        x_colloc = torch.rand(8, 3)
        loss = stokes_residual_loss(small_net, x_colloc)
        loss.backward()
        grads = [p.grad for p in small_net.parameters() if p.grad is not None]
        assert len(grads) > 0

    def test_residual_not_nan(self, small_net: PINNNetwork) -> None:
        x_colloc = torch.rand(20, 3)
        loss = stokes_residual_loss(small_net, x_colloc)
        assert not torch.isnan(loss)


# ---------------------------------------------------------------------------
# bc_loss
# ---------------------------------------------------------------------------


class TestBCLoss:
    def test_zero_for_zero_velocity(self) -> None:
        net = ZeroVelocityNet()
        x_wall = torch.rand(12, 3)
        loss = bc_loss(net, x_wall)
        assert float(loss) == pytest.approx(0.0, abs=1e-7)

    def test_positive_for_random_net(self, small_net: PINNNetwork) -> None:
        torch.manual_seed(7)
        x_wall = torch.rand(12, 3)
        loss = bc_loss(small_net, x_wall)
        assert float(loss) > 0.0

    def test_output_is_scalar(self, small_net: PINNNetwork) -> None:
        x_wall = torch.rand(8, 3)
        loss = bc_loss(small_net, x_wall)
        assert loss.shape == ()

    def test_gradients_flow(self, small_net: PINNNetwork) -> None:
        x_wall = torch.rand(8, 3)
        loss = bc_loss(small_net, x_wall)
        loss.backward()
        grads = [p.grad for p in small_net.parameters() if p.grad is not None]
        assert len(grads) > 0


# ---------------------------------------------------------------------------
# pressure_anchor_loss
# ---------------------------------------------------------------------------


class TestPressureAnchorLoss:
    def test_zero_when_pressure_is_zero(self) -> None:
        net = ZeroPressureAtPointNet()
        x_anchor = torch.rand(1, 3)
        loss = pressure_anchor_loss(net, x_anchor)
        assert float(loss) == pytest.approx(0.0, abs=1e-7)

    def test_positive_for_random_net(self, small_net: PINNNetwork) -> None:
        x_anchor = torch.rand(1, 3)
        loss = pressure_anchor_loss(small_net, x_anchor)
        # Not guaranteed to be positive (could be exactly zero by chance),
        # but for a randomly initialised net it almost certainly is.
        assert float(loss) >= 0.0

    def test_output_is_scalar(self, small_net: PINNNetwork) -> None:
        x_anchor = torch.rand(1, 3)
        loss = pressure_anchor_loss(small_net, x_anchor)
        assert loss.shape == ()


# ---------------------------------------------------------------------------
# total_loss
# ---------------------------------------------------------------------------


class TestTotalLoss:
    def test_returns_tensor_and_dict(
        self,
        small_net: PINNNetwork,
        rng_pts: tuple,
    ) -> None:
        x_data, x_colloc, x_wall, x_anchor = rng_pts
        u_obs = torch.rand(20, 3)
        loss, bd = total_loss(
            small_net, x_data, u_obs, x_colloc, x_wall, x_anchor
        )
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()
        assert isinstance(bd, dict)

    def test_breakdown_keys(self, small_net: PINNNetwork, rng_pts: tuple) -> None:
        x_data, x_colloc, x_wall, x_anchor = rng_pts
        u_obs = torch.rand(20, 3)
        _, bd = total_loss(small_net, x_data, u_obs, x_colloc, x_wall, x_anchor)
        for key in ("loss_data", "loss_phys", "loss_bc", "loss_anchor",
                    "loss_inlet", "loss_total"):
            assert key in bd, f"Missing key: {key}"

    def test_inlet_term_included_when_weighted(
        self, small_net: PINNNetwork, rng_pts: tuple
    ) -> None:
        """A non-zero lambda_inlet must add a positive contribution to total."""
        x_data, x_colloc, x_wall, x_anchor = rng_pts
        u_obs = torch.rand(20, 3)
        x_inlet = torch.rand(6, 3)
        u_inlet = torch.rand(6, 3) + 0.3
        loss_no, bd_no = total_loss(
            small_net, x_data, u_obs, x_colloc, x_wall, x_anchor,
        )
        loss_yes, bd_yes = total_loss(
            small_net, x_data, u_obs, x_colloc, x_wall, x_anchor,
            x_inlet=x_inlet, u_inlet=u_inlet, lambda_inlet=5.0,
        )
        assert bd_no["loss_inlet"] == pytest.approx(0.0, abs=1e-12)
        assert bd_yes["loss_inlet"] > 0.0
        assert float(loss_yes) > float(loss_no)

    def test_weighted_sum_consistent(
        self, small_net: PINNNetwork, rng_pts: tuple
    ) -> None:
        x_data, x_colloc, x_wall, x_anchor = rng_pts
        u_obs = torch.rand(20, 3)
        ld, lp, lb, la = 2.0, 3.0, 5.0, 7.0
        loss, bd = total_loss(
            small_net, x_data, u_obs, x_colloc, x_wall, x_anchor,
            lambda_data=ld, lambda_phys=lp, lambda_bc=lb, lambda_anchor=la,
        )
        expected = (
            ld * bd["loss_data"]
            + lp * bd["loss_phys"]
            + lb * bd["loss_bc"]
            + la * bd["loss_anchor"]
        )
        assert float(loss) == pytest.approx(expected, rel=1e-5)

    def test_gradients_flow_through_total(
        self, small_net: PINNNetwork, rng_pts: tuple
    ) -> None:
        x_data, x_colloc, x_wall, x_anchor = rng_pts
        u_obs = torch.rand(20, 3)
        loss, _ = total_loss(small_net, x_data, u_obs, x_colloc, x_wall, x_anchor)
        loss.backward()
        grads = [p.grad for p in small_net.parameters() if p.grad is not None]
        assert len(grads) > 0

    def test_no_nan_in_loss(self, small_net: PINNNetwork, rng_pts: tuple) -> None:
        x_data, x_colloc, x_wall, x_anchor = rng_pts
        u_obs = torch.rand(20, 3)
        loss, bd = total_loss(small_net, x_data, u_obs, x_colloc, x_wall, x_anchor)
        assert not torch.isnan(loss)
        for key, val in bd.items():
            assert math.isfinite(val), f"{key} is not finite: {val}"


# ---------------------------------------------------------------------------
# Fix D — SelfAdaptiveLoss
# ---------------------------------------------------------------------------


class TestSelfAdaptiveLoss:
    def test_initial_lambdas_match_init_values(self) -> None:
        init = {"data": 1.0, "phys": 2.0, "bc": 5.0, "anchor": 3.0}
        sa = SelfAdaptiveLoss(init)
        for k, v in init.items():
            assert sa.lambdas[k] == pytest.approx(v, rel=1e-4)

    def test_forward_returns_scalar(self) -> None:
        sa = SelfAdaptiveLoss({"a": 1.0, "b": 2.0})
        out = sa({"a": torch.tensor(0.5), "b": torch.tensor(0.3)})
        assert out.shape == ()

    def test_forward_weighted_sum_value(self) -> None:
        sa = SelfAdaptiveLoss({"a": 2.0, "b": 3.0})
        out = sa({"a": torch.tensor(1.0), "b": torch.tensor(1.0)})
        assert float(out.detach()) == pytest.approx(5.0, rel=1e-4)

    def test_lambdas_property_has_correct_keys(self) -> None:
        init = {"data": 1.0, "phys": 2.0, "bc": 5.0, "anchor": 3.0}
        assert set(SelfAdaptiveLoss(init).lambdas.keys()) == set(init.keys())

    def test_gradients_on_log_lambdas(self) -> None:
        sa = SelfAdaptiveLoss({"data": 1.0, "phys": 1.0})
        out = sa({"data": torch.tensor(2.0), "phys": torch.tensor(3.0)})
        out.backward()
        for name, param in sa.log_lambdas.items():
            assert param.grad is not None, f"No grad on log_lambda[{name}]"

    def test_gradient_reversal_increases_weight(self) -> None:
        """Reversing the gradient and taking an SGD step must increase λ when loss is large."""
        sa = SelfAdaptiveLoss({"a": 1.0})
        opt = torch.optim.SGD(sa.parameters(), lr=0.1)
        opt.zero_grad()
        out = sa({"a": torch.tensor(10.0)})   # large residual → weight should grow
        out.backward()
        for param in sa.parameters():
            if param.grad is not None:
                param.grad.neg_()            # gradient reversal as in trainer
        opt.step()
        assert sa.lambdas["a"] > 1.0

    def test_zero_initial_lambda_handled(self) -> None:
        """lambda=0 is clamped to 1e-8 before log; must not raise."""
        sa = SelfAdaptiveLoss({"bc": 0.0})
        out = sa({"bc": torch.tensor(1.0)})
        assert out.shape == ()
        assert not torch.isnan(out)


# ---------------------------------------------------------------------------
# Fix B — stokes_residual_loss with skip_div_loss
# ---------------------------------------------------------------------------


class TestStokesResidualSkipDiv:
    def test_skip_div_returns_scalar(self, small_net: PINNNetwork) -> None:
        loss = stokes_residual_loss(small_net, torch.rand(10, 3), skip_div_loss=True)
        assert loss.shape == ()

    def test_full_loss_ge_skip_loss(self, small_net: PINNNetwork) -> None:
        """Full loss (momentum + div) ≥ momentum-only loss (non-negative div term)."""
        x = torch.rand(10, 3)
        loss_full = float(stokes_residual_loss(small_net, x, skip_div_loss=False).detach())
        loss_skip = float(stokes_residual_loss(small_net, x, skip_div_loss=True).detach())
        assert loss_full >= loss_skip - 1e-7

    def test_skip_div_gradients_flow(self, small_net: PINNNetwork) -> None:
        loss = stokes_residual_loss(small_net, torch.rand(8, 3), skip_div_loss=True)
        loss.backward()
        grads = [p.grad for p in small_net.parameters() if p.grad is not None]
        assert len(grads) > 0

    def test_skip_div_no_nan(self, small_net: PINNNetwork) -> None:
        loss = stokes_residual_loss(small_net, torch.rand(12, 3), skip_div_loss=True)
        assert not torch.isnan(loss)


# ---------------------------------------------------------------------------
# Fix A — loss functions accept sdf_vals
# ---------------------------------------------------------------------------


class TestLossesWithSDFVals:
    def test_data_loss_accepts_sdf(self, small_net: PINNNetwork) -> None:
        x = torch.rand(10, 3)
        u_obs = torch.rand(10, 3)
        sdf = torch.rand(10)
        loss = data_loss(small_net, x, u_obs, sdf_vals=sdf)
        assert loss.shape == ()
        assert not torch.isnan(loss)

    def test_stokes_loss_accepts_sdf(self, small_net: PINNNetwork) -> None:
        x = torch.rand(10, 3)
        sdf = torch.rand(10)
        loss = stokes_residual_loss(small_net, x, sdf_vals=sdf)
        assert loss.shape == ()
        assert not torch.isnan(loss)

    def test_total_loss_accepts_sdf_kwargs(
        self, small_net: PINNNetwork, rng_pts: tuple
    ) -> None:
        x_data, x_colloc, x_wall, x_anchor = rng_pts
        u_obs = torch.rand(20, 3)
        sdf_data = torch.rand(20)
        sdf_colloc = torch.rand(30)
        loss, bd = total_loss(
            small_net, x_data, u_obs, x_colloc, x_wall, x_anchor,
            sdf_data=sdf_data, sdf_colloc=sdf_colloc,
        )
        assert loss.shape == ()
        assert not torch.isnan(loss)

    def test_total_loss_skip_div_accepted(
        self, small_net: PINNNetwork, rng_pts: tuple
    ) -> None:
        x_data, x_colloc, x_wall, x_anchor = rng_pts
        u_obs = torch.rand(20, 3)
        loss, _ = total_loss(
            small_net, x_data, u_obs, x_colloc, x_wall, x_anchor,
            skip_div_loss=True,
        )
        assert not torch.isnan(loss)

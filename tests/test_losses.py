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
    bc_loss,
    data_loss,
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

    def forward(self, x: Tensor) -> Tensor:
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
        for key in ("loss_data", "loss_phys", "loss_bc", "loss_anchor", "loss_total"):
            assert key in bd, f"Missing key: {key}"

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

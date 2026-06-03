"""Tests for hemodyn_pinn.pinn.inference.

Verifies the WSS-from-autograd pipeline (CLAUDE.md §3.5):
    D_θ = ½(∇u_θ + (∇u_θ)ᵀ)
    τ_w = 2μ [D_θ · n̂]_tangential

All tests run on CPU using small synthetic networks; no data files required.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from hemodyn_pinn.pinn.inference import (
    compute_velocity_jacobian,
    compute_wss,
    compute_wss_auto,
    compute_wss_hard_sdf,
    predict_velocity_field,
)
from hemodyn_pinn.pinn.networks import L_SCALE, MU, U_SCALE, PINNNetwork


# ---------------------------------------------------------------------------
# Analytic networks for ground-truth checks
# ---------------------------------------------------------------------------


class LinearVelocityNet(nn.Module):
    """u = (A x, 0, 0, 0) — linear shear in x-direction.

    ∇u_0 = (A, 0, 0);  ∇u_1 = ∇u_2 = 0.
    D = ½(∇u + (∇u)ᵀ);  D[0,0]=A, all others 0.
    For n̂ = (0, 1, 0):
      D·n̂ = column 1 of D = 0 vector → τ = 0
    For n̂ = (1, 0, 0):
      D·n̂ = (A, 0, 0) → normal component = A → τ_tangential = 0
    For n̂ = (0, 0, 1):
      D·n̂ = (0, 0, 0) → τ = 0
    """

    def __init__(self, A: float = 3.0) -> None:
        super().__init__()
        self.A = A

    def forward(self, x: Tensor) -> Tensor:
        u = self.A * x[:, 0]
        zeros = torch.zeros_like(u)
        return torch.stack([u, zeros, zeros, zeros], dim=1)


class ShearFlowNet(nn.Module):
    """Couette-like flow: u = y, v = 0, w = 0, p = 0.

    ∂u/∂y = 1 — the only non-zero gradient.
    D[0,1] = D[1,0] = 0.5.

    For n̂ = (0, 1, 0)  (wall normal pointing in y-direction):
      D·n̂ = (0.5, 0, 0)
      Normal component = 0
      τ_tangential = (0.5, 0, 0)
      |τ| = 0.5  →  WSS_Pa = 2μ(U/L) · 0.5 = μ(U/L)
    """

    def forward(self, x: Tensor) -> Tensor:
        u = x[:, 1]           # y-component
        zeros = torch.zeros_like(u)
        return torch.stack([u, zeros, zeros, zeros], dim=1)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def small_net() -> PINNNetwork:
    return PINNNetwork(n_hidden=16, n_layers=2, use_rff=False, seed=0)


@pytest.fixture()
def wall_setup() -> tuple[Tensor, Tensor]:
    """Returns (x_wall_nondim, wall_normals) for 5 synthetic wall points."""
    torch.manual_seed(42)
    x_wall = torch.rand(5, 3)
    normals = torch.zeros(5, 3)
    normals[:, 1] = 1.0    # all normals point in y-direction
    return x_wall, normals


# ---------------------------------------------------------------------------
# compute_velocity_jacobian
# ---------------------------------------------------------------------------


class TestComputeVelocityJacobian:
    def test_shape(self, small_net: PINNNetwork, wall_setup: tuple) -> None:
        x_wall, _ = wall_setup
        J = compute_velocity_jacobian(small_net, x_wall)
        assert J.shape == (5, 3, 3)

    def test_known_jacobian_linear_net(self) -> None:
        """Jacobian of u=(Ax, 0, 0) should be diag(A, 0, 0) in first row."""
        A = 2.5
        net = LinearVelocityNet(A=A)
        x = torch.tensor([[1.0, 2.0, 3.0]])
        J = compute_velocity_jacobian(net, x)   # (1, 3, 3)
        assert J.shape == (1, 3, 3)
        # Row 0 of J (gradient of u_0 = Ax): should be (A, 0, 0)
        assert J[0, 0, 0].item() == pytest.approx(A, abs=1e-5)
        assert J[0, 0, 1].item() == pytest.approx(0.0, abs=1e-5)
        assert J[0, 0, 2].item() == pytest.approx(0.0, abs=1e-5)
        # Rows 1, 2 (gradient of v=0, w=0): all zeros
        assert J[0, 1, :].abs().max().item() == pytest.approx(0.0, abs=1e-5)
        assert J[0, 2, :].abs().max().item() == pytest.approx(0.0, abs=1e-5)

    def test_no_nan(self, small_net: PINNNetwork, wall_setup: tuple) -> None:
        x_wall, _ = wall_setup
        J = compute_velocity_jacobian(small_net, x_wall)
        assert not torch.isnan(J).any()


# ---------------------------------------------------------------------------
# compute_wss
# ---------------------------------------------------------------------------


class TestComputeWSS:
    def test_output_shapes(self, small_net: PINNNetwork, wall_setup: tuple) -> None:
        x_wall, normals = wall_setup
        tau, tau_mag = compute_wss(small_net, x_wall, normals)
        assert tau.shape == (5, 3)
        assert tau_mag.shape == (5,)

    def test_wss_magnitude_nonnegative(
        self, small_net: PINNNetwork, wall_setup: tuple
    ) -> None:
        x_wall, normals = wall_setup
        _, tau_mag = compute_wss(small_net, x_wall, normals)
        assert (tau_mag >= 0).all()

    def test_wss_tangential_couette(self) -> None:
        """WSS magnitude for Couette flow (u=y) with n=(0,1,0) is μU/L."""
        net = ShearFlowNet()
        x_wall = torch.tensor([[0.5, 0.0, 0.5]])   # y=0 wall
        normals = torch.tensor([[0.0, 1.0, 0.0]])

        tau, tau_mag = compute_wss(net, x_wall, normals)

        # τ_tangential = 2μ(U/L) · 0.5 = μU/L
        expected_pa = MU * U_SCALE / L_SCALE
        assert float(tau_mag[0]) == pytest.approx(expected_pa, rel=1e-4)

    def test_wss_normal_component_zero(self) -> None:
        """WSS must be purely tangential — no component along n̂."""
        net = ShearFlowNet()
        x_wall = torch.tensor([[0.5, 0.0, 0.5], [0.3, 0.0, 0.8]])
        normals = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])

        tau, _ = compute_wss(net, x_wall, normals)

        # τ · n̂ should be zero (tangential projection removes normal component)
        normal_dot = (tau * normals).sum(dim=1)
        assert normal_dot.abs().max().item() == pytest.approx(0.0, abs=1e-5)

    def test_wss_no_nan(self, small_net: PINNNetwork, wall_setup: tuple) -> None:
        x_wall, normals = wall_setup
        tau, tau_mag = compute_wss(small_net, x_wall, normals)
        assert not torch.isnan(tau).any()
        assert not torch.isnan(tau_mag).any()

    def test_wss_units_are_pa(self) -> None:
        """Order-of-magnitude check: WSS for Couette flow is ~μU/L (Pa)."""
        net = ShearFlowNet()
        x_wall = torch.rand(10, 3)
        normals = torch.zeros(10, 3)
        normals[:, 1] = 1.0
        _, tau_mag = compute_wss(net, x_wall, normals)

        # Non-dim shear rate = 0.5, so WSS = 2μ(U/L)(0.5) = μU/L
        expected_order = MU * U_SCALE / L_SCALE   # ≈ 4.3e-6 Pa
        assert float(tau_mag.mean()) == pytest.approx(expected_order, rel=1e-4)


# ---------------------------------------------------------------------------
# predict_velocity_field
# ---------------------------------------------------------------------------


class TestPredictVelocityField:
    def test_output_shapes(self, small_net: PINNNetwork) -> None:
        x = torch.rand(12, 3)
        u_ms, p_pa = predict_velocity_field(small_net, x)
        assert u_ms.shape == (12, 3)
        assert p_pa.shape == (12,)

    def test_no_gradients_tracked(self, small_net: PINNNetwork) -> None:
        """predict_velocity_field runs under no_grad."""
        x = torch.rand(5, 3)
        u_ms, p_pa = predict_velocity_field(small_net, x)
        assert not u_ms.requires_grad
        assert not p_pa.requires_grad

    def test_dimensional_scaling(self) -> None:
        """Output should scale with U_SCALE and pressure scale."""

        class UnitNet(nn.Module):
            """Always outputs (1, 1, 1, 1) in non-dim."""

            def forward(self, x: Tensor, sdf_vals=None) -> Tensor:
                return torch.ones(x.shape[0], 4)

        net = UnitNet()
        x = torch.rand(3, 3)
        u_ms, p_pa = predict_velocity_field(net, x)

        assert u_ms.mean().item() == pytest.approx(U_SCALE, rel=1e-6)
        p_scale = MU * U_SCALE / L_SCALE
        assert p_pa.mean().item() == pytest.approx(p_scale, rel=1e-6)


# ---------------------------------------------------------------------------
# Analytic networks for hard-SDF WSS tests
# ---------------------------------------------------------------------------


class _ConstantTangentialNet(nn.Module):
    """Raw velocity = (1, 0, 0) — purely tangential to n = (0, 1, 0).

    Expected hard-SDF WSS: |τ| = μ(U/L) * |(1,0,0) − 0*(0,1,0)| = μU/L.
    """

    use_hard_sdf: bool = True

    def forward_raw(self, x: Tensor) -> Tensor:
        N = x.shape[0]
        return torch.cat([
            torch.ones(N, 1),
            torch.zeros(N, 1),
            torch.zeros(N, 1),
            torch.zeros(N, 1),
        ], dim=1)

    def forward(self, x: Tensor, sdf_vals=None) -> Tensor:
        return self.forward_raw(x)


class _NormalAlignedNet(nn.Module):
    """Raw velocity = (0, 1, 0) — parallel to n = (0, 1, 0).

    Expected hard-SDF WSS: tangential component = 0 → |τ| = 0.
    """

    use_hard_sdf: bool = True

    def forward_raw(self, x: Tensor) -> Tensor:
        N = x.shape[0]
        return torch.cat([
            torch.zeros(N, 1),
            torch.ones(N, 1),
            torch.zeros(N, 1),
            torch.zeros(N, 1),
        ], dim=1)

    def forward(self, x: Tensor, sdf_vals=None) -> Tensor:
        return self.forward_raw(x)


# ---------------------------------------------------------------------------
# Fix A — compute_wss_hard_sdf
# ---------------------------------------------------------------------------


class TestComputeWSSHardSDF:
    def test_output_shapes(self, wall_setup: tuple) -> None:
        net = PINNNetwork(n_hidden=16, n_layers=2, use_hard_sdf=True, seed=0)
        x_wall, normals = wall_setup
        tau, mag = compute_wss_hard_sdf(net, x_wall, normals)
        assert tau.shape == (5, 3)
        assert mag.shape == (5,)

    def test_magnitude_nonneg(self, wall_setup: tuple) -> None:
        net = PINNNetwork(n_hidden=16, n_layers=2, use_hard_sdf=True, seed=0)
        x_wall, normals = wall_setup
        _, mag = compute_wss_hard_sdf(net, x_wall, normals)
        assert (mag >= 0).all()

    def test_no_nan(self, wall_setup: tuple) -> None:
        net = PINNNetwork(n_hidden=16, n_layers=2, use_hard_sdf=True, seed=0)
        x_wall, normals = wall_setup
        tau, mag = compute_wss_hard_sdf(net, x_wall, normals)
        assert not torch.isnan(tau).any()
        assert not torch.isnan(mag).any()

    def test_known_tangential_velocity(self) -> None:
        """u_raw = (1,0,0), n = (0,1,0) → |WSS| = μ(U/L) exactly."""
        net = _ConstantTangentialNet()
        x_wall = torch.rand(4, 3)
        normals = torch.zeros(4, 3); normals[:, 1] = 1.0
        _, mag = compute_wss_hard_sdf(net, x_wall, normals)
        expected = MU * U_SCALE / L_SCALE
        assert float(mag.mean()) == pytest.approx(expected, rel=1e-5)

    def test_normal_velocity_gives_zero_wss(self) -> None:
        """u_raw parallel to n → tangential component = 0 → WSS = 0."""
        net = _NormalAlignedNet()
        x_wall = torch.rand(4, 3)
        normals = torch.zeros(4, 3); normals[:, 1] = 1.0
        _, mag = compute_wss_hard_sdf(net, x_wall, normals)
        assert mag.abs().max().item() == pytest.approx(0.0, abs=1e-7)

    def test_wss_purely_tangential(self) -> None:
        """The WSS vector must have zero normal component."""
        net = PINNNetwork(n_hidden=16, n_layers=2, use_hard_sdf=True, seed=2)
        x_wall, normals = torch.rand(8, 3), torch.zeros(8, 3)
        normals[:, 1] = 1.0
        tau, _ = compute_wss_hard_sdf(net, x_wall, normals)
        normal_dot = (tau * normals).sum(dim=1)
        assert normal_dot.abs().max().item() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Fix A — compute_wss_auto dispatch
# ---------------------------------------------------------------------------


class TestComputeWSSAuto:
    def test_dispatches_to_hard_sdf_path(self, wall_setup: tuple) -> None:
        """With use_hard_sdf=True, auto must agree with compute_wss_hard_sdf."""
        net = PINNNetwork(n_hidden=16, n_layers=2, use_hard_sdf=True, seed=0)
        x_wall, normals = wall_setup
        _, mag_auto   = compute_wss_auto(net, x_wall, normals)
        _, mag_direct = compute_wss_hard_sdf(net, x_wall, normals)
        assert torch.allclose(mag_auto, mag_direct, atol=1e-7)

    def test_dispatches_to_standard_path(self, wall_setup: tuple) -> None:
        """Without use_hard_sdf, auto must agree with the standard autograd path."""
        net = PINNNetwork(n_hidden=16, n_layers=2, use_hard_sdf=False, seed=0)
        x_wall, normals = wall_setup
        _, mag_auto   = compute_wss_auto(net, x_wall, normals)
        _, mag_direct = compute_wss(net, x_wall, normals)
        assert torch.allclose(mag_auto, mag_direct, atol=1e-7)

    def test_output_shapes(self) -> None:
        net = PINNNetwork(n_hidden=16, n_layers=2, seed=0)
        x_wall = torch.rand(7, 3)
        normals = torch.nn.functional.normalize(torch.rand(7, 3), dim=1)
        tau, mag = compute_wss_auto(net, x_wall, normals)
        assert tau.shape == (7, 3)
        assert mag.shape == (7,)

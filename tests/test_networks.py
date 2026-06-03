"""Tests for PINNNetwork and submodules (Fixes A, B).

Covers:
- MLP: activation guard, output shape.
- RFFEncoder: output dimension, determinism.
- PINNNetwork (base): output shape for all activation choices, no-ReLU invariant.
- Fix A (use_hard_sdf): zero velocity at wall, pressure unaffected, forward_raw bypass,
  scaling identity u_forward = sdf * u_raw.
- Fix B (use_vec_potential): output shape, exact div-free property, _curl shape.
- Combined A+B: correct shape, zero velocity when sdf=0.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from hemodyn_pinn.pinn.networks import MLP, PINNNetwork, RFFEncoder


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class TestMLP:
    def test_unknown_activation_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown activation"):
            MLP(in_dim=3, activation="relu")  # type: ignore[arg-type]

    def test_output_shape(self) -> None:
        mlp = MLP(in_dim=3, out_dim=4, n_hidden=16, n_layers=2, activation="tanh")
        assert mlp(torch.rand(7, 3)).shape == (7, 4)


# ---------------------------------------------------------------------------
# RFFEncoder
# ---------------------------------------------------------------------------


class TestRFFEncoder:
    def test_out_dim_property(self) -> None:
        enc = RFFEncoder(n_features=64)
        assert enc.out_dim == 128

    def test_output_shape(self) -> None:
        enc = RFFEncoder(n_features=32)
        assert enc(torch.rand(10, 3)).shape == (10, 64)

    def test_deterministic_with_same_seed(self) -> None:
        x = torch.rand(5, 3)
        enc1 = RFFEncoder(n_features=16, seed=42)
        enc2 = RFFEncoder(n_features=16, seed=42)
        assert torch.allclose(enc1(x), enc2(x))

    def test_different_seeds_differ(self) -> None:
        x = torch.rand(5, 3)
        enc1 = RFFEncoder(n_features=32, seed=1)
        enc2 = RFFEncoder(n_features=32, seed=2)
        assert not torch.allclose(enc1(x), enc2(x))


# ---------------------------------------------------------------------------
# PINNNetwork — base
# ---------------------------------------------------------------------------


class TestPINNNetworkBase:
    def test_output_shape_default(self) -> None:
        net = PINNNetwork(n_hidden=16, n_layers=2, seed=0)
        assert net(torch.rand(12, 3)).shape == (12, 4)

    def test_output_shape_with_rff(self) -> None:
        net = PINNNetwork(n_hidden=16, n_layers=2, use_rff=True, rff_features=32, seed=0)
        assert net(torch.rand(8, 3)).shape == (8, 4)

    @pytest.mark.parametrize("act", ["tanh", "swish", "gelu"])
    def test_supported_activations(self, act: str) -> None:
        net = PINNNetwork(n_hidden=16, n_layers=2, activation=act, seed=0)  # type: ignore[arg-type]
        assert net(torch.rand(5, 3)).shape == (5, 4)

    def test_no_relu_in_backbone(self) -> None:
        net = PINNNetwork(n_hidden=32, n_layers=4, seed=0)
        for mod in net.modules():
            assert not isinstance(mod, nn.ReLU), "ReLU must not appear in PINN backbone"


# ---------------------------------------------------------------------------
# Fix A — hard SDF no-slip
# ---------------------------------------------------------------------------


class TestHardSDF:
    def test_zero_velocity_when_sdf_zero(self) -> None:
        """Velocity output must be exactly 0 when sdf_vals = 0."""
        net = PINNNetwork(n_hidden=16, n_layers=2, use_hard_sdf=True, seed=0)
        x = torch.rand(8, 3)
        out = net(x, sdf_vals=torch.zeros(8))
        assert torch.allclose(out[:, :3], torch.zeros(8, 3), atol=1e-7)

    def test_pressure_not_affected_by_sdf(self) -> None:
        """The pressure column (index 3) must be identical for sdf=0 and sdf=1."""
        net = PINNNetwork(n_hidden=16, n_layers=2, use_hard_sdf=True, seed=0)
        x = torch.rand(6, 3)
        p_zero = net(x, sdf_vals=torch.zeros(6))[:, 3]
        p_ones = net(x, sdf_vals=torch.ones(6))[:, 3]
        assert torch.allclose(p_zero, p_ones, atol=1e-6)

    def test_velocity_scales_linearly_with_sdf(self) -> None:
        """u_forward(x, sdf=d) must equal d * u_raw(x) componentwise."""
        net = PINNNetwork(n_hidden=16, n_layers=2, use_hard_sdf=True, seed=0)
        x = torch.rand(5, 3)
        d = torch.rand(5) * 2.0
        out_sdf = net(x, sdf_vals=d)
        u_raw = net.forward_raw(x)[:, :3]
        expected = u_raw * d.unsqueeze(-1)
        assert torch.allclose(out_sdf[:, :3], expected, atol=1e-6)

    def test_forward_raw_bypasses_sdf(self) -> None:
        """forward_raw must match the plain forward of an identical network without SDF."""
        net_sdf = PINNNetwork(n_hidden=16, n_layers=2, use_hard_sdf=True, seed=7)
        net_plain = PINNNetwork(n_hidden=16, n_layers=2, use_hard_sdf=False, seed=7)
        x = torch.rand(5, 3)
        assert torch.allclose(net_sdf.forward_raw(x), net_plain(x), atol=1e-6)


# ---------------------------------------------------------------------------
# Fix B — vector potential (divergence-free)
# ---------------------------------------------------------------------------


class TestVecPotential:
    def test_output_shape(self) -> None:
        net = PINNNetwork(n_hidden=16, n_layers=2, use_vec_potential=True, seed=0)
        assert net(torch.rand(8, 3)).shape == (8, 4)

    def test_divergence_free(self) -> None:
        """div(u) = ∂u/∂x + ∂v/∂y + ∂w/∂z must be ≈ 0 for a vec-potential network."""
        net = PINNNetwork(n_hidden=16, n_layers=2, use_vec_potential=True, seed=1)
        net.eval()
        x = torch.rand(12, 3, requires_grad=True)
        out = net(x)
        u, v, w = out[:, 0], out[:, 1], out[:, 2]
        (du,) = torch.autograd.grad(u.sum(), x, create_graph=False, retain_graph=True)
        (dv,) = torch.autograd.grad(v.sum(), x, create_graph=False, retain_graph=True)
        (dw,) = torch.autograd.grad(w.sum(), x, create_graph=False, retain_graph=True)
        div = du[:, 0] + dv[:, 1] + dw[:, 2]
        assert div.abs().max().item() == pytest.approx(0.0, abs=1e-5)

    def test_curl_shape(self) -> None:
        """_curl must return (N, 3)."""
        net = PINNNetwork(n_hidden=16, n_layers=2, use_vec_potential=True, seed=0)
        x = torch.rand(8, 3, requires_grad=True)
        A = net._mlp_out(x)[:, :3]
        assert net._curl(A, x).shape == (8, 3)

    def test_no_nan_in_output(self) -> None:
        net = PINNNetwork(n_hidden=16, n_layers=2, use_vec_potential=True, seed=0)
        out = net(torch.rand(10, 3))
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# Combined A + B
# ---------------------------------------------------------------------------


class TestCombinedSDFAndVecPotential:
    def test_output_shape(self) -> None:
        net = PINNNetwork(
            n_hidden=16, n_layers=2,
            use_hard_sdf=True, use_vec_potential=True, seed=0,
        )
        out = net(torch.rand(6, 3), sdf_vals=torch.rand(6) * 0.5)
        assert out.shape == (6, 4)

    def test_zero_velocity_when_sdf_zero(self) -> None:
        """Even with vec-potential, SDF=0 must yield zero velocity."""
        net = PINNNetwork(
            n_hidden=16, n_layers=2,
            use_hard_sdf=True, use_vec_potential=True, seed=0,
        )
        out = net(torch.rand(5, 3), sdf_vals=torch.zeros(5))
        assert torch.allclose(out[:, :3], torch.zeros(5, 3), atol=1e-7)

    def test_no_nan(self) -> None:
        net = PINNNetwork(
            n_hidden=16, n_layers=2,
            use_hard_sdf=True, use_vec_potential=True, seed=0,
        )
        out = net(torch.rand(8, 3), sdf_vals=torch.rand(8))
        assert not torch.isnan(out).any()

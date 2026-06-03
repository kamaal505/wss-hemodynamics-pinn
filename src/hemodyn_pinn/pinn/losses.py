"""PINN loss terms for steady Stokes flow reconstruction.

Non-dimensional Stokes equations (with x̂ = x/L, û = u/U, p̂ = p·L/(μU)):
    -∇̂p̂ + Δ̂û = 0   (momentum)
    ∇̂·û = 0          (continuity)

All inputs are non-dimensional tensors on the same device as the network.

New in this version:
    - stokes_residual_loss accepts sdf_vals (precomputed SDF at collocation
      points) and skip_div_loss (for vector-potential networks where
      div(u) = 0 identically).
    - SelfAdaptiveLoss implements the McClenny & Braga-Neto (2023) self-adaptive
      loss weight scheme (Fix D): weights are learnable parameters optimised
      in the opposite direction to the network (gradient reversal performed by
      the trainer via param.grad.neg_()).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Autograd helpers
# ---------------------------------------------------------------------------


def _grad(scalar_field: Tensor, x: Tensor) -> Tensor:
    """Gradient of a scalar field w.r.t. x via autograd."""
    (g,) = torch.autograd.grad(
        scalar_field.sum(),
        x,
        create_graph=True,
        retain_graph=True,
    )
    return g


# ---------------------------------------------------------------------------
# Individual loss terms
# ---------------------------------------------------------------------------


def data_loss(
    net: nn.Module,
    x_data: Tensor,
    u_obs: Tensor,
    sdf_vals: Optional[Tensor] = None,
) -> Tensor:
    """MSE between network velocity and MRI observations.

    Parameters
    ----------
    net:
        PINNNetwork.
    x_data:
        (N_d, 3) non-dimensional voxel-centre coordinates.
    u_obs:
        (N_d, 3) non-dimensional observed velocity.
    sdf_vals:
        (N_d,) non-dimensional SDF at data points.  Passed through to
        net.forward() when use_hard_sdf is active so that the predicted
        velocity includes the SDF scaling.
    """
    out = net(x_data, sdf_vals=sdf_vals)
    u_pred = out[:, :3]
    return ((u_pred - u_obs) ** 2).mean()


def stokes_residual_loss(
    net: nn.Module,
    x_colloc: Tensor,
    sdf_vals: Optional[Tensor] = None,
    skip_div_loss: bool = False,
) -> Tensor:
    """Stokes momentum + continuity residual loss at collocation points.

    Parameters
    ----------
    net:
        PINNNetwork.
    x_colloc:
        (N_r, 3) non-dimensional collocation coordinates.
    sdf_vals:
        (N_r,) SDF values at collocation points.  When use_hard_sdf is
        active, the network multiplies velocity by these values.  The
        autograd chain treats sdf_vals as a fixed constant (not part of
        the graph), which is an accepted approximation for interior points
        where the ∇SDF correction is small.
    skip_div_loss:
        If True, the continuity residual is skipped.  Set this when
        use_vec_potential=True because div(curl(A)) = 0 identically.
        Also set when use_hard_sdf=True AND use_vec_potential=True,
        because div(SDF · curl(A)) = ∇SDF · curl(A) ≠ 0 and penalising
        it would fight the hard constraint.
    """
    x = x_colloc.detach().requires_grad_(True)
    out = net(x, sdf_vals=sdf_vals)
    u, v, w, p = out[:, 0], out[:, 1], out[:, 2], out[:, 3]

    du = _grad(u, x)
    dv = _grad(v, x)
    dw = _grad(w, x)
    dp = _grad(p, x)

    lap_u = (
        _grad(du[:, 0], x)[:, 0]
        + _grad(du[:, 1], x)[:, 1]
        + _grad(du[:, 2], x)[:, 2]
    )
    lap_v = (
        _grad(dv[:, 0], x)[:, 0]
        + _grad(dv[:, 1], x)[:, 1]
        + _grad(dv[:, 2], x)[:, 2]
    )
    lap_w = (
        _grad(dw[:, 0], x)[:, 0]
        + _grad(dw[:, 1], x)[:, 1]
        + _grad(dw[:, 2], x)[:, 2]
    )

    R_x = -dp[:, 0] + lap_u
    R_y = -dp[:, 1] + lap_v
    R_z = -dp[:, 2] + lap_w
    loss_mom = (R_x ** 2 + R_y ** 2 + R_z ** 2).mean()

    if skip_div_loss:
        return loss_mom

    div_u = du[:, 0] + dv[:, 1] + dw[:, 2]
    loss_div = (div_u ** 2).mean()
    return loss_mom + loss_div


def bc_loss(
    net: nn.Module,
    x_wall: Tensor,
) -> Tensor:
    """No-slip BC loss on the vessel wall (soft constraint).

    When use_hard_sdf=True this loss evaluates to nearly zero by construction
    (SDF = 0 at wall points → network output is zero) and its weight
    lambda_bc is set to 0.0 in the trainer.  The function is kept for
    interface consistency.
    """
    out = net(x_wall)   # no sdf_vals: wall SDF = 0 enforced by hard constraint
    u_wall = out[:, :3]
    return (u_wall ** 2).mean()


def pressure_anchor_loss(
    net: nn.Module,
    x_anchor: Tensor,
) -> Tensor:
    """Pin pressure to zero at one outlet centroid (gauge fixing)."""
    out = net(x_anchor)
    p_anchor = out[:, 3]
    return (p_anchor ** 2).mean()


# ---------------------------------------------------------------------------
# Self-adaptive loss weights (Fix D)
# ---------------------------------------------------------------------------


class SelfAdaptiveLoss(nn.Module):
    """Learnable loss-weight module (McClenny & Braga-Neto, 2023, JCP).

    Each weight is parameterised as exp(log_λ_k) so it stays positive.
    The trainer performs gradient *reversal* on these parameters after
    each backward pass (param.grad.neg_()) so that they are maximised
    while the network parameters are minimised.  Net effect: terms with
    high residual automatically receive higher weight.

    Parameters
    ----------
    init_lambdas:
        Initial values for each loss weight, keyed by loss name.
        Typically {'data': λ_d, 'phys': λ_p, 'bc': λ_bc, 'anchor': λ_a}.
    """

    def __init__(self, init_lambdas: dict[str, float]) -> None:
        super().__init__()
        self.log_lambdas = nn.ParameterDict({
            k: nn.Parameter(torch.tensor(math.log(max(v, 1e-8)), dtype=torch.float32))
            for k, v in init_lambdas.items()
        })

    def forward(self, losses: dict[str, Tensor]) -> Tensor:
        """Compute weighted sum Σ_k exp(log_λ_k) · L_k.

        Parameters
        ----------
        losses:
            Dict of scalar loss tensors keyed by the same names as
            init_lambdas.
        """
        return sum(
            torch.exp(self.log_lambdas[k]) * v
            for k, v in losses.items()
        )

    @property
    def lambdas(self) -> dict[str, float]:
        """Current weight values (detached, for logging)."""
        return {
            k: float(torch.exp(v).detach())
            for k, v in self.log_lambdas.items()
        }


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------


def total_loss(
    net: nn.Module,
    x_data: Tensor,
    u_obs: Tensor,
    x_colloc: Tensor,
    x_wall: Tensor,
    x_anchor: Tensor,
    lambda_data: float = 1.0,
    lambda_phys: float = 1.0,
    lambda_bc: float = 10.0,
    lambda_anchor: float = 10.0,
    sdf_data: Optional[Tensor] = None,
    sdf_colloc: Optional[Tensor] = None,
    skip_div_loss: bool = False,
) -> tuple[Tensor, dict[str, float]]:
    """Compute the full PINN loss and return a breakdown dict.

    Parameters
    ----------
    net:
        PINNNetwork.
    x_data, u_obs:
        MRI observation locations and (non-dim) velocities.
    x_colloc:
        Collocation points for the physics residual.
    x_wall:
        Wall boundary points for the no-slip condition.
    x_anchor:
        Single outlet point for pressure anchoring.
    lambda_data, lambda_phys, lambda_bc, lambda_anchor:
        Fixed loss weights (used when SelfAdaptiveLoss is not active).
    sdf_data:
        (N_d,) SDF values at data points (Fix A).
    sdf_colloc:
        (N_r,) SDF values at collocation points (Fix A).
    skip_div_loss:
        Skip the divergence residual (Fix B or combined A+B).

    Returns
    -------
    total : Tensor
    breakdown : dict
    """
    L_data   = data_loss(net, x_data, u_obs, sdf_vals=sdf_data)
    L_phys   = stokes_residual_loss(
        net, x_colloc, sdf_vals=sdf_colloc, skip_div_loss=skip_div_loss
    )
    L_bc     = bc_loss(net, x_wall)
    L_anchor = pressure_anchor_loss(net, x_anchor)

    total = (
        lambda_data   * L_data
        + lambda_phys * L_phys
        + lambda_bc   * L_bc
        + lambda_anchor * L_anchor
    )
    breakdown = {
        "loss_data":   float(L_data.detach()),
        "loss_phys":   float(L_phys.detach()),
        "loss_bc":     float(L_bc.detach()),
        "loss_anchor": float(L_anchor.detach()),
        "loss_total":  float(total.detach()),
    }
    return total, breakdown

"""PINN loss terms for steady Stokes flow reconstruction.

Non-dimensional Stokes equations (with x̂ = x/L, û = u/U, p̂ = p·L/(μU)):
    -∇̂p̂ + Δ̂û = 0   (momentum)
    ∇̂·û = 0          (continuity)

All inputs are non-dimensional tensors on the same device as the network.
"""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn as nn


# ---------------------------------------------------------------------------
# Autograd helpers
# ---------------------------------------------------------------------------


def _grad(scalar_field: Tensor, x: Tensor) -> Tensor:
    """Compute gradient of a scalar field w.r.t. x using autograd.

    Parameters
    ----------
    scalar_field:
        (N,) tensor whose sum's gradient w.r.t. x is taken.  Because
        batch points are independent through the MLP, summing and
        differentiating is equivalent to pointwise differentiation.
    x:
        (N, 3) input tensor with requires_grad=True.

    Returns
    -------
    Tensor
        (N, 3) gradient at each point.
    """
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
) -> Tensor:
    """Mean-squared error between network velocity and MRI observations.

    Parameters
    ----------
    net:
        PINNNetwork.
    x_data:
        (N_d, 3) non-dimensional voxel-centre coordinates.
    u_obs:
        (N_d, 3) non-dimensional observed velocity (û, v̂, ŵ).

    Returns
    -------
    Tensor
        Scalar MSE loss.
    """
    out = net(x_data)       # (N_d, 4)
    u_pred = out[:, :3]     # (N_d, 3)  velocity only
    return ((u_pred - u_obs) ** 2).mean()


def stokes_residual_loss(
    net: nn.Module,
    x_colloc: Tensor,
) -> Tensor:
    """Stokes momentum + continuity residual loss at collocation points.

    Computes the non-dimensional Stokes residual:
        R_mom = -∇̂p̂ + Δ̂û  (3 components)
        R_div = ∇̂·û        (scalar)

    Uses second-order autograd.  x_colloc must NOT have requires_grad
    set before this call — we set it internally.

    Parameters
    ----------
    net:
        PINNNetwork.
    x_colloc:
        (N_r, 3) non-dimensional collocation coordinates.

    Returns
    -------
    Tensor
        Scalar mean residual loss (momentum + continuity combined).
    """
    x = x_colloc.detach().requires_grad_(True)
    out = net(x)                  # (N_r, 4)
    u, v, w, p = out[:, 0], out[:, 1], out[:, 2], out[:, 3]

    # First-order gradients of velocity and pressure
    du = _grad(u, x)   # (N_r, 3)
    dv = _grad(v, x)
    dw = _grad(w, x)
    dp = _grad(p, x)   # (N_r, 3)  ∇̂p̂

    # Continuity residual: ∂û/∂x̂ + ∂v̂/∂ŷ + ∂ŵ/∂ẑ
    div_u = du[:, 0] + dv[:, 1] + dw[:, 2]

    # Laplacian of each velocity component via diagonal of Hessian.
    # grad(du[:,j].sum(), x)[:,j] gives ∂²u/∂x_j² at each point.
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

    # Momentum residual: -∇̂p̂ + Δ̂û = 0
    R_x = -dp[:, 0] + lap_u
    R_y = -dp[:, 1] + lap_v
    R_z = -dp[:, 2] + lap_w

    loss_mom = (R_x ** 2 + R_y ** 2 + R_z ** 2).mean()
    loss_div = (div_u ** 2).mean()
    return loss_mom + loss_div


def bc_loss(
    net: nn.Module,
    x_wall: Tensor,
) -> Tensor:
    """No-slip boundary condition loss on the vessel wall.

    Enforces û = 0 at wall points (soft constraint).

    Parameters
    ----------
    net:
        PINNNetwork.
    x_wall:
        (N_b, 3) non-dimensional wall-centroid coordinates.

    Returns
    -------
    Tensor
        Scalar MSE loss.
    """
    out = net(x_wall)
    u_wall = out[:, :3]    # velocity only
    return (u_wall ** 2).mean()


def pressure_anchor_loss(
    net: nn.Module,
    x_anchor: Tensor,
) -> Tensor:
    """Pin pressure to zero at a single outlet point.

    Prevents pressure drift (gauge freedom of the Stokes equations).
    x_anchor should contain exactly one point — a centroid on the outlet face.

    Parameters
    ----------
    net:
        PINNNetwork.
    x_anchor:
        (1, 3) non-dimensional anchor coordinate.

    Returns
    -------
    Tensor
        Scalar squared pressure at the anchor point.
    """
    out = net(x_anchor)    # (1, 4)
    p_anchor = out[:, 3]   # (1,)
    return (p_anchor ** 2).mean()


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
        Loss weights.

    Returns
    -------
    total : Tensor
        Weighted sum, differentiable for backprop.
    breakdown : dict
        Scalar float values for logging (detached from graph).
    """
    L_data   = data_loss(net, x_data, u_obs)
    L_phys   = stokes_residual_loss(net, x_colloc)
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

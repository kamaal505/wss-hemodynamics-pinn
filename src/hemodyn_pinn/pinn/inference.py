"""WSS inference from a trained PINNNetwork via autograd.

The WSS computation follows CLAUDE.md §3.5:

    D_θ(x_w) = ½(∇u_θ + (∇u_θ)ᵀ)|_{x_w}
    τ_w = 2μ [D_θ · n̂]_tangential

where the tangential projection removes the normal component:
    [D · n̂]_tangential = D · n̂ - (n̂ · D · n̂) n̂

All computations use autograd — no finite differences.

Inputs and outputs
------------------
- Network operates in non-dimensional coordinates (x̂, û, p̂).
- Wall normals are dimensionless unit vectors.
- WSS is returned in Pa (SI) after dimensional reconstruction.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from hemodyn_pinn.pinn.networks import L_SCALE, U_SCALE, MU


def compute_velocity_jacobian(
    net: nn.Module,
    x_wall: Tensor,
) -> Tensor:
    """Compute ∇û at wall points via autograd.

    Parameters
    ----------
    net:
        Trained PINNNetwork.
    x_wall:
        (W, 3) non-dimensional wall coordinates.

    Returns
    -------
    Tensor
        (W, 3, 3) Jacobian J where J[i, a, b] = ∂û_a/∂x̂_b at point i.
    """
    x = x_wall.detach().requires_grad_(True)
    out = net(x)           # (W, 4)
    u_vec = out[:, :3]     # (W, 3)

    W = x.shape[0]
    J = torch.zeros(W, 3, 3, device=x.device, dtype=x.dtype)

    for a in range(3):
        (grad_a,) = torch.autograd.grad(
            u_vec[:, a].sum(),
            x,
            create_graph=False,
            retain_graph=True,
        )
        J[:, a, :] = grad_a   # ∂û_a/∂x̂_b for b=0,1,2

    return J   # (W, 3, 3)


def compute_wss(
    net: nn.Module,
    x_wall_nondim: Tensor,
    wall_normals: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compute wall shear stress (WSS) from a trained PINN.

    Parameters
    ----------
    net:
        Trained PINNNetwork.
    x_wall_nondim:
        (W, 3) non-dimensional wall-centroid coordinates.
    wall_normals:
        (W, 3) outward unit normals at the wall centroids.

    Returns
    -------
    tau_pa : Tensor
        (W, 3) WSS vectors in Pa (SI).
    tau_mag_pa : Tensor
        (W,) WSS magnitudes in Pa.
    """
    J = compute_velocity_jacobian(net, x_wall_nondim)  # (W, 3, 3) non-dim

    # Rate-of-strain tensor D̂ = ½(J + Jᵀ), shape (W, 3, 3)
    D_hat = 0.5 * (J + J.transpose(1, 2))

    n = wall_normals.to(x_wall_nondim.device)    # (W, 3)

    # Traction vector: D̂ · n̂, shape (W, 3)
    Dn = torch.einsum("wab,wb->wa", D_hat, n)

    # Normal component: (n̂ · D̂ · n̂) n̂, shape (W, 3)
    normal_mag = (Dn * n).sum(dim=1, keepdim=True)   # (W, 1)
    tau_hat_tangential = Dn - normal_mag * n          # (W, 3)

    # Dimensional WSS: τ = 2μ(U/L) · τ̂_tangential
    dim_scale = 2.0 * MU * U_SCALE / L_SCALE
    tau_pa = dim_scale * tau_hat_tangential

    tau_mag_pa = tau_pa.norm(dim=1)

    return tau_pa, tau_mag_pa


def predict_velocity_field(
    net: nn.Module,
    x_nondim: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return dimensional velocity and pressure at arbitrary points.

    Parameters
    ----------
    net:
        Trained PINNNetwork.
    x_nondim:
        (N, 3) non-dimensional query coordinates.

    Returns
    -------
    u_ms : Tensor
        (N, 3) velocity in m/s.
    p_pa : Tensor
        (N,) pressure in Pa.
    """
    with torch.no_grad():
        out = net(x_nondim)   # (N, 4)
    u_ms = out[:, :3] * U_SCALE
    p_pa = out[:, 3] * (MU * U_SCALE / L_SCALE)
    return u_ms, p_pa

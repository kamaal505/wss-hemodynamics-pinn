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
    relative: bool = True,
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
    relative:
        If True, normalise the MSE by the mean squared observation
        magnitude.  This makes the loss scale-invariant (a zero prediction
        gives ≈ 1.0, a perfect fit gives 0.0) so that lambda_data is
        comparable to the physics residual regardless of the tiny absolute
        velocity scale (U_SCALE = 2e-5 m/s).  Prevents the magnitude
        collapse to the trivial u ≡ 0 Stokes solution.
    """
    out = net(x_data, sdf_vals=sdf_vals)
    u_pred = out[:, :3]
    mse = ((u_pred - u_obs) ** 2).mean()
    if relative:
        return mse / ((u_obs ** 2).mean() + 1e-12)
    return mse


def inlet_loss(
    net: nn.Module,
    x_inlet: Tensor,
    u_inlet: Tensor,
    sdf_vals: Optional[Tensor] = None,
    relative: bool = True,
) -> Tensor:
    """Inflow constraint: fit the (measured) velocity at the inlet plane.

    Steady Stokes flow is linear and homogeneous, so u ≡ 0 is an exact
    zero-residual solution.  Without a velocity/flux condition at an open
    boundary the field magnitude is unpinned and collapses toward zero.
    This term anchors the magnitude by matching the network velocity to a
    known inlet velocity (sourced from the near-inlet MRI voxels or, for a
    diagnostic upper bound, the CFD inlet profile).

    Identical in form to ``data_loss`` but kept separate so it carries its
    own weight (lambda_inlet) and participates in best-model selection.
    """
    out = net(x_inlet, sdf_vals=sdf_vals)
    u_pred = out[:, :3]
    mse = ((u_pred - u_inlet) ** 2).mean()
    if relative:
        return mse / ((u_inlet ** 2).mean() + 1e-12)
    return mse


def aux_data_loss(
    net: nn.Module,
    x_aux: Tensor,
    u_aux: Tensor,
    sdf_vals: Optional[Tensor] = None,
    relative: bool = True,
) -> Tensor:
    """Supervised misfit to a dense interpolant of the sparse MRI voxels.

    The dense target (see ``pinn.interpolant``) provides a non-zero velocity
    magnitude at every interior collocation point, so the network cannot
    collapse to the trivial u ≡ 0 Stokes solution while this term is active.
    Used as a curriculum warm-up prior and then with a decaying weight as the
    physics residual takes over.

    Identical in form to ``data_loss``; kept separate so it carries its own
    (decaying) weight and breakdown entry.
    """
    out = net(x_aux, sdf_vals=sdf_vals)
    u_pred = out[:, :3]
    mse = ((u_pred - u_aux) ** 2).mean()
    if relative:
        return mse / ((u_aux ** 2).mean() + 1e-12)
    return mse


def magnitude_floor_loss(
    net: nn.Module,
    x_colloc: Tensor,
    target_rms: float,
    sdf_vals: Optional[Tensor] = None,
) -> Tensor:
    """One-sided penalty when the predicted velocity RMS falls below a floor.

    ``target_rms`` is the RMS magnitude of the observed (non-dimensional)
    velocity.  The penalty is ``relu(target_rms − rms(|u_pred|))²`` evaluated
    over the collocation batch: it is zero once the field magnitude is healthy,
    so a correctly-scaled solution is never biased, while a collapsing field is
    pushed back up globally (not just at the sparse data points).
    """
    out = net(x_colloc, sdf_vals=sdf_vals)
    u_pred = out[:, :3]
    pred_rms = torch.sqrt((u_pred ** 2).sum(dim=1).mean() + 1e-12)
    deficit = torch.relu(target_rms - pred_rms)
    return deficit ** 2


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
    relative_data: bool = True,
    x_inlet: Optional[Tensor] = None,
    u_inlet: Optional[Tensor] = None,
    sdf_inlet: Optional[Tensor] = None,
    lambda_inlet: float = 0.0,
    x_aux: Optional[Tensor] = None,
    u_aux: Optional[Tensor] = None,
    sdf_aux: Optional[Tensor] = None,
    lambda_aux: float = 0.0,
    target_rms: Optional[float] = None,
    lambda_mag_floor: float = 0.0,
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
    relative_data:
        Use the scale-invariant (relative) form for the data and inlet
        misfit terms.  Recommended True to avoid magnitude collapse.
    x_inlet, u_inlet, sdf_inlet:
        Inlet-plane coordinates, target velocities, and SDF values for the
        inflow constraint.  When ``x_inlet`` is None or ``lambda_inlet`` is
        0 the inlet term is omitted.
    lambda_inlet:
        Weight on the inflow constraint.
    x_aux, u_aux, sdf_aux:
        Dense interpolant target coordinates, velocities, and SDF values for
        the auxiliary anti-collapse supervision.  Omitted when ``x_aux`` is
        None or ``lambda_aux`` is 0.  ``lambda_aux`` is normally the *decayed*
        weight supplied by the trainer (it ramps to 0 as physics takes over).
    lambda_aux:
        Weight on the dense interpolant term (current, post-decay value).
    target_rms, lambda_mag_floor:
        RMS of the observed non-dim velocity and the weight of the one-sided
        magnitude-floor penalty on the collocation batch.

    Returns
    -------
    total : Tensor
    breakdown : dict
    """
    L_data   = data_loss(net, x_data, u_obs, sdf_vals=sdf_data, relative=relative_data)
    L_phys   = stokes_residual_loss(
        net, x_colloc, sdf_vals=sdf_colloc, skip_div_loss=skip_div_loss
    )
    L_bc     = bc_loss(net, x_wall)
    L_anchor = pressure_anchor_loss(net, x_anchor)

    if x_inlet is not None and lambda_inlet > 0.0:
        L_inlet = inlet_loss(
            net, x_inlet, u_inlet, sdf_vals=sdf_inlet, relative=relative_data
        )
    else:
        L_inlet = torch.zeros((), device=x_data.device, dtype=x_data.dtype)

    if x_aux is not None and lambda_aux > 0.0:
        L_aux = aux_data_loss(
            net, x_aux, u_aux, sdf_vals=sdf_aux, relative=relative_data
        )
    else:
        L_aux = torch.zeros((), device=x_data.device, dtype=x_data.dtype)

    if target_rms is not None and lambda_mag_floor > 0.0:
        L_mag = magnitude_floor_loss(
            net, x_colloc, target_rms, sdf_vals=sdf_colloc
        )
    else:
        L_mag = torch.zeros((), device=x_data.device, dtype=x_data.dtype)

    total = (
        lambda_data   * L_data
        + lambda_phys * L_phys
        + lambda_bc   * L_bc
        + lambda_anchor * L_anchor
        + lambda_inlet * L_inlet
        + lambda_aux  * L_aux
        + lambda_mag_floor * L_mag
    )
    breakdown = {
        "loss_data":   float(L_data.detach()),
        "loss_phys":   float(L_phys.detach()),
        "loss_bc":     float(L_bc.detach()),
        "loss_anchor": float(L_anchor.detach()),
        "loss_inlet":  float(L_inlet.detach()),
        "loss_aux":    float(L_aux.detach()),
        "loss_mag_floor": float(L_mag.detach()),
        "loss_total":  float(total.detach()),
    }
    return total, breakdown

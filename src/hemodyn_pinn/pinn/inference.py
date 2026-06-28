"""WSS inference from a trained PINNNetwork via autograd.

Standard path (use_hard_sdf=False):
    Differentiates the network output û w.r.t. x̂ at wall points.  The
    velocity Jacobian is formed, the rate-of-strain tensor assembled, and the
    tangential traction extracted.

Hard-SDF path (use_hard_sdf=True):
    At wall points SDF = 0, so differentiating the SDF-scaled output through
    the computation graph gives zero.  Instead we use the analytical result
    derived from the product rule:

        u(x) = d(x) · u_net(x),   d(x_wall) = 0,  ∇d|_wall = −n̂_out

    At wall:  ∂u_a/∂x_b = (∂d/∂x_b) · u_net_a = −n_b · u_net_a(x_wall)
    Strain:   D_ab = −½(n_b u_net_a + n_a u_net_b)
    Traction: (D·n̂)_a = −½(u_net_a + n_a (u_net·n̂))
    Normal:   n̂·D·n̂   = −(u_net·n̂)
    Tangential traction = −½ u_net_tangential

    WSS = 2μ(U/L) · [D·n̂]_tang = −μ(U/L) · u_net_tangential

    The sign is a convention (direction of traction on the fluid); the
    magnitude |WSS| = μ(U/L) · |u_net_tang| is what we report.

    This requires only a forward pass (net.forward_raw) at wall points —
    no autograd through the wall.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch import Tensor

from hemodyn_pinn.pinn.networks import L_SCALE, U_SCALE, MU

log = logging.getLogger(__name__)


def compute_velocity_jacobian(
    net: nn.Module,
    x_wall: Tensor,
) -> Tensor:
    """Compute ∇û at wall points via autograd (standard path only).

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
    out = net(x)           # (W, 4) — standard forward (no SDF at wall = 0)
    u_vec = out[:, :3]

    W = x.shape[0]
    J = torch.zeros(W, 3, 3, device=x.device, dtype=x.dtype)

    for a in range(3):
        (grad_a,) = torch.autograd.grad(
            u_vec[:, a].sum(),
            x,
            create_graph=False,
            retain_graph=True,
        )
        J[:, a, :] = grad_a

    return J


def compute_wss(
    net: nn.Module,
    x_wall_nondim: Tensor,
    wall_normals: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compute WSS using the standard autograd path.

    Use this when use_hard_sdf=False.  When use_hard_sdf=True call
    compute_wss_hard_sdf instead.

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

    D_hat = 0.5 * (J + J.transpose(1, 2))
    n = wall_normals.to(x_wall_nondim.device)

    Dn = torch.einsum("wab,wb->wa", D_hat, n)
    normal_mag = (Dn * n).sum(dim=1, keepdim=True)
    tau_hat_tangential = Dn - normal_mag * n

    dim_scale = 2.0 * MU * U_SCALE / L_SCALE
    tau_pa = dim_scale * tau_hat_tangential
    tau_mag_pa = tau_pa.norm(dim=1)

    return tau_pa, tau_mag_pa


def compute_wss_hard_sdf(
    net: nn.Module,
    x_wall_nondim: Tensor,
    wall_normals: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compute WSS using the analytical hard-SDF formula (Fix A).

    Requires net.use_hard_sdf=True.  Does NOT call autograd on wall points.
    Instead evaluates the raw network output (before SDF scaling) and applies
    the analytical result:

        |τ_w| = μ (U/L) |û_net − (û_net · n̂) n̂|

    Parameters
    ----------
    net:
        Trained PINNNetwork with use_hard_sdf=True.
    x_wall_nondim:
        (W, 3) non-dimensional wall-centroid coordinates.
    wall_normals:
        (W, 3) outward unit normals at the wall centroids.

    Returns
    -------
    tau_pa : Tensor
        (W, 3) WSS vectors in Pa.
    tau_mag_pa : Tensor
        (W,) WSS magnitudes in Pa.
    """
    with torch.no_grad():
        raw = net.forward_raw(x_wall_nondim)   # (W, 4) before SDF
    u_net = raw[:, :3]                          # (W, 3) non-dim

    n = wall_normals.to(x_wall_nondim.device)   # (W, 3)
    u_net_n = (u_net * n).sum(dim=1, keepdim=True)   # normal component (W, 1)
    u_net_tang = u_net - u_net_n * n                  # tangential component (W, 3)

    # From derivation: τ_w = −μ(U/L) · û_net_tang  (sign = convention)
    # Magnitude: |τ_w| = μ(U/L) · |û_net_tang|
    dim_scale = MU * U_SCALE / L_SCALE   # note: no factor 2 (already folded in)
    tau_pa = dim_scale * u_net_tang
    tau_mag_pa = tau_pa.norm(dim=1)

    return tau_pa, tau_mag_pa


def compute_wss_auto(
    net: nn.Module,
    x_wall_nondim: Tensor,
    wall_normals: Tensor,
) -> tuple[Tensor, Tensor]:
    """Dispatch to the appropriate WSS computation based on network flags.

    Uses the hard-SDF analytical formula when net.use_hard_sdf=True;
    otherwise uses the standard autograd Jacobian path.
    """
    use_hard = getattr(net, "use_hard_sdf", False)
    if use_hard:
        return compute_wss_hard_sdf(net, x_wall_nondim, wall_normals)
    return compute_wss(net, x_wall_nondim, wall_normals)


def _is_oom_error(exc: BaseException) -> bool:
    """True if `exc` looks like an out-of-memory error (CUDA/MPS/CPU)."""
    if exc.__class__.__name__ == "OutOfMemoryError":
        return True
    msg = str(exc).lower()
    return (
        "out of memory" in msg
        or "can't allocate" in msg
        or "cuda error: out of memory" in msg
        or "mps backend out of memory" in msg
    )


def compute_wss_batched(
    net: nn.Module,
    x_wall_nondim: Tensor,
    wall_normals: Tensor,
    batch_size: int = 4096,
    min_batch_size: int = 64,
) -> tuple[Tensor, Tensor]:
    """WSS over many wall points, computed in memory-bounded chunks.

    The standard autograd path forms a per-point velocity Jacobian and holds the
    graph for three backward passes; doing this for every wall face at once is
    the dominant out-of-memory risk during evaluation (e.g. inside a BHPO
    trial).  This wrapper streams the wall points in batches and, if a batch
    still triggers an OOM, **halves the batch size and retries** down to
    ``min_batch_size`` rather than abandoning the computation.  No wall face is
    ever dropped.

    Parameters
    ----------
    net:
        Trained PINNNetwork.
    x_wall_nondim:
        (W, 3) non-dimensional wall-centroid coordinates.
    wall_normals:
        (W, 3) outward unit normals.
    batch_size:
        Initial points per chunk; adaptively reduced on OOM.
    min_batch_size:
        Floor below which an OOM is re-raised (genuinely insufficient memory).

    Returns
    -------
    tau_pa : Tensor
        (W, 3) WSS vectors in Pa.
    tau_mag_pa : Tensor
        (W,) WSS magnitudes in Pa.
    """
    W = x_wall_nondim.shape[0]
    tau_chunks: list[Tensor] = []
    mag_chunks: list[Tensor] = []

    start = 0
    bs = max(min_batch_size, int(batch_size))
    while start < W:
        end = min(start + bs, W)
        xb = x_wall_nondim[start:end]
        nb = wall_normals[start:end]
        try:
            tau_b, mag_b = compute_wss_auto(net, xb, nb)
            tau_chunks.append(tau_b.detach())
            mag_chunks.append(mag_b.detach())
            start = end
        except (RuntimeError, MemoryError) as exc:
            if not _is_oom_error(exc) or bs <= min_batch_size:
                raise
            _free_memory(x_wall_nondim.device)
            bs = max(min_batch_size, bs // 2)
            log.warning(
                "WSS batch OOM at %d points; retrying with batch_size=%d", end - start, bs
            )

    return torch.cat(tau_chunks, dim=0), torch.cat(mag_chunks, dim=0)


def _free_memory(device: "torch.device") -> None:
    """Release cached allocator memory for the active accelerator."""
    import gc
    gc.collect()
    try:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()
    except Exception:
        pass


def predict_velocity_field(
    net: nn.Module,
    x_nondim: Tensor,
    sdf_vals: "Optional[Tensor]" = None,
) -> tuple[Tensor, Tensor]:
    """Return dimensional velocity and pressure at arbitrary points.

    Parameters
    ----------
    net:
        Trained PINNNetwork.
    x_nondim:
        (N, 3) non-dimensional query coordinates.
    sdf_vals:
        (N,) non-dimensional SDF values.  Pass when use_hard_sdf=True so
        the velocity includes the SDF scaling.

    Returns
    -------
    u_ms : Tensor
        (N, 3) velocity in m/s.
    p_pa : Tensor
        (N,) pressure in Pa.
    """
    from typing import Optional  # noqa: PLC0415 — local import avoids cycle
    with torch.no_grad():
        out = net(x_nondim, sdf_vals=sdf_vals)
    u_ms = out[:, :3] * U_SCALE
    p_pa = out[:, 3] * (MU * U_SCALE / L_SCALE)
    return u_ms, p_pa

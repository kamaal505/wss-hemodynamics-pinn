"""PINN network architectures for Stokes flow reconstruction.

All modules operate in non-dimensional coordinates:
    x̂ = x_m / L_SCALE,  û = u_ms / U_SCALE,  p̂ = p_Pa * L_SCALE / (MU * U_SCALE)

Network output is always (û, v̂, ŵ, p̂) — four scalars per query point.

Supported activations: "tanh", "swish", "gelu".  NEVER ReLU (breaks second-order autograd).

Architecture flags (set in PINNNetwork):
    use_hard_sdf      (Fix A) — multiply velocity output by precomputed SDF values
                                so no-slip is satisfied by construction.
                                Requires sdf_vals to be passed to forward().
    use_vec_potential (Fix B) — network outputs vector potential A = (Ax, Ay, Az, p)
                                and computes u = curl(A) via autograd, guaranteeing
                                div(u) = 0 identically.  Requires requires_grad=True
                                on x (handled automatically inside forward()).
                                NOTE: the Stokes Laplacian through A requires 3rd-order
                                autograd — significantly slower per step but may need
                                fewer total iterations.

Both flags may be enabled simultaneously: u = SDF · curl(A).  In that case:
    - no-slip is exact (SDF = 0 at wall)
    - div(u) ≠ 0 in general (∇SDF · curl(A) ≠ 0), so skip_div_loss should be True
    - WSS is computed via the analytical hard-constraint formula in inference.py
"""

from __future__ import annotations

import math
from typing import Literal, Optional

import torch
import torch.nn as nn
from torch import Tensor

from hemodyn_pinn.utils.seeds import RFF_SEED, PINN_TRAIN_SEED

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "tanh": nn.Tanh,
    "swish": nn.SiLU,
    "gelu": nn.GELU,
}
ActivationName = Literal["tanh", "swish", "gelu"]

# Physical scales (SI)
L_SCALE: float = 0.01628   # m
U_SCALE: float = 2e-5      # m/s
MU: float = 3.5e-3         # Pa·s


def nondim_coords(x_m: Tensor) -> Tensor:
    return x_m / L_SCALE


def nondim_velocity(u_ms: Tensor) -> Tensor:
    return u_ms / U_SCALE


def dim_velocity(u_hat: Tensor) -> Tensor:
    return u_hat * U_SCALE


def dim_wss(tau_hat: Tensor) -> Tensor:
    """Convert non-dimensional WSS to SI (Pa).

    Non-dim WSS scale = MU * U_SCALE / L_SCALE.
    """
    return tau_hat * (MU * U_SCALE / L_SCALE)


# ---------------------------------------------------------------------------
# Random Fourier Features encoder
# ---------------------------------------------------------------------------


class RFFEncoder(nn.Module):
    """Random Fourier Features positional encoder.

    Replaces raw (x, y, z) with
        φ(r) = [sin(2π B r), cos(2π B r)] ∈ ℝ^{2D}
    where B ∈ ℝ^{D×3} has entries B_ij ~ N(0, σ²).

    B is fixed (not trained).  σ controls the highest representable frequency.
    """

    def __init__(
        self,
        n_features: int = 128,
        sigma: float = 1.0,
        seed: int = RFF_SEED,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.sigma = sigma
        generator = torch.Generator()
        generator.manual_seed(seed)
        B = torch.randn(n_features, 3, generator=generator) * sigma
        self.register_buffer("B", B)

    @property
    def out_dim(self) -> int:
        return 2 * self.n_features

    def forward(self, x: Tensor) -> Tensor:
        """Encode spatial coordinates.

        Parameters
        ----------
        x:
            (N, 3) non-dimensional coordinates.

        Returns
        -------
        Tensor
            (N, 2D) encoded features.
        """
        proj = (2 * math.pi * x) @ self.B.T   # (N, D)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


# ---------------------------------------------------------------------------
# MLP backbone
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    """Fully-connected network with configurable smooth activations."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 4,
        n_hidden: int = 128,
        n_layers: int = 4,
        activation: ActivationName = "tanh",
    ) -> None:
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{activation}'. Choose from {list(_ACTIVATIONS)}."
            )
        act_cls = _ACTIVATIONS[activation]
        layers: list[nn.Module] = [nn.Linear(in_dim, n_hidden), act_cls()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_hidden, n_hidden), act_cls()]
        layers.append(nn.Linear(n_hidden, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# PINN network
# ---------------------------------------------------------------------------


class PINNNetwork(nn.Module):
    """Coordinate-based PINN for steady Stokes flow.

    Input:  x̂ = (x̂, ŷ, ẑ) — non-dimensional coordinates.
    Output: (û, v̂, ŵ, p̂)  — non-dimensional velocity + pressure.

    Parameters
    ----------
    n_hidden:
        Neurons per hidden layer.
    n_layers:
        Number of hidden layers.
    use_rff:
        Whether to prepend a Random Fourier Features encoder.
    rff_features:
        D — number of RFF frequency samples (ignored if use_rff=False).
    rff_sigma:
        Bandwidth σ for the RFF encoder (ignored if use_rff=False).
    activation:
        Activation: "tanh", "swish", or "gelu".
    seed:
        Seed for deterministic weight initialisation.
    use_hard_sdf:
        Fix A — if True, velocity output is multiplied pointwise by sdf_vals
        passed to forward(), enforcing u = 0 at the wall by construction.
        When enabled, bc_loss becomes structurally zero and lambda_bc should
        be set to 0 in the trainer.
    use_vec_potential:
        Fix B — if True, the MLP outputs a vector potential A = (Ax, Ay, Az, p)
        and the physical velocity is u = curl(A) computed via autograd.
        div(u) = 0 identically (div·curl = 0).  Requires 3rd-order autograd
        for the Stokes momentum residual — slower but structurally div-free.
    """

    def __init__(
        self,
        n_hidden: int = 128,
        n_layers: int = 4,
        use_rff: bool = False,
        rff_features: int = 128,
        rff_sigma: float = 1.0,
        activation: ActivationName = "tanh",
        seed: int = PINN_TRAIN_SEED,
        use_hard_sdf: bool = False,
        use_vec_potential: bool = False,
    ) -> None:
        super().__init__()
        self.use_rff = use_rff
        self.use_hard_sdf = use_hard_sdf
        self.use_vec_potential = use_vec_potential

        if use_rff:
            self.encoder: Optional[RFFEncoder] = RFFEncoder(
                n_features=rff_features, sigma=rff_sigma
            )
            in_dim = self.encoder.out_dim
        else:
            self.encoder = None
            in_dim = 3

        torch.manual_seed(seed)
        self.mlp = MLP(
            in_dim=in_dim, out_dim=4, n_hidden=n_hidden,
            n_layers=n_layers, activation=activation,
        )

    # ------------------------------------------------------------------
    # Curl helper (Fix B)
    # ------------------------------------------------------------------

    def _curl(self, A: Tensor, x: Tensor) -> Tensor:
        """Compute curl(A) via autograd.

        Parameters
        ----------
        A:
            (N, 3) vector potential components — raw MLP output first 3 cols.
        x:
            (N, 3) input coordinates with requires_grad=True.

        Returns
        -------
        Tensor
            (N, 3) velocity = (∂Az/∂y − ∂Ay/∂z,  ∂Ax/∂z − ∂Az/∂x,
                                ∂Ay/∂x − ∂Ax/∂y).

        Notes
        -----
        create_graph=True is always used so that higher-order derivatives
        (Laplacian in the Stokes residual) can be computed by subsequent
        backward passes.  This makes training slower but is required for
        physics correctness.
        """
        grads = []
        for i in range(3):
            (g,) = torch.autograd.grad(
                A[:, i].sum(), x,
                create_graph=True, retain_graph=True,
            )
            grads.append(g)   # grads[i][:, j] = ∂A_i/∂x_j

        # curl components
        u = grads[2][:, 1] - grads[1][:, 2]   # ∂Az/∂y − ∂Ay/∂z
        v = grads[0][:, 2] - grads[2][:, 0]   # ∂Ax/∂z − ∂Az/∂x
        w = grads[1][:, 0] - grads[0][:, 1]   # ∂Ay/∂x − ∂Ax/∂y

        return torch.stack([u, v, w], dim=-1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _mlp_out(self, x: Tensor) -> Tensor:
        """Run encoder + MLP, returning raw (N, 4) output."""
        h = self.encoder(x) if self.use_rff else x
        return self.mlp(h)

    def forward(
        self,
        x: Tensor,
        sdf_vals: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass returning (û, v̂, ŵ, p̂).

        Parameters
        ----------
        x:
            (N, 3) non-dimensional coordinates.
        sdf_vals:
            (N,) non-dimensional SDF distances.  Required when
            use_hard_sdf=True is active.  Ignored otherwise.

        Returns
        -------
        Tensor
            (N, 4) — columns: û, v̂, ŵ, p̂.
        """
        if self.use_vec_potential:
            # Curl requires gradients w.r.t. x.  Detach and re-enable so
            # this forward pass is self-contained (data_loss, bc_loss callers
            # do not need to manage requires_grad).
            if not x.requires_grad:
                x = x.detach().requires_grad_(True)
            raw = self._mlp_out(x)          # (N, 4) = (Ax, Ay, Az, p)
            A = raw[:, :3]
            p = raw[:, 3:4]
            u_vel = self._curl(A, x)        # (N, 3)  — div-free by identity
            out = torch.cat([u_vel, p], dim=-1)
        else:
            out = self._mlp_out(x)          # (N, 4) = (u, v, w, p)

        if self.use_hard_sdf and sdf_vals is not None:
            # Hard no-slip: u(x) = SDF(x) · u_net(x).  Pressure unchanged.
            d = sdf_vals.unsqueeze(-1)      # (N, 1)
            out = torch.cat([out[:, :3] * d, out[:, 3:4]], dim=-1)

        return out

    def forward_raw(self, x: Tensor) -> Tensor:
        """Forward pass WITHOUT SDF multiplication — used for WSS inference.

        When use_hard_sdf=True, WSS at wall points is proportional to the
        tangential component of the raw (pre-SDF) network output; see
        inference.compute_wss_hard_sdf.  This method bypasses the SDF
        multiplication regardless of use_hard_sdf.

        Returns (N, 4) same as forward().
        """
        if self.use_vec_potential:
            if not x.requires_grad:
                x = x.detach().requires_grad_(True)
            raw = self._mlp_out(x)
            A = raw[:, :3]
            p = raw[:, 3:4]
            u_vel = self._curl(A, x)
            return torch.cat([u_vel, p], dim=-1)
        return self._mlp_out(x)

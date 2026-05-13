"""PINN network architectures for Stokes flow reconstruction.

All modules operate in non-dimensional coordinates:
    x̂ = x_m / L_SCALE,  û = u_ms / U_SCALE,  p̂ = p_Pa * L_SCALE / (MU * U_SCALE)

Network output is always (û, v̂, ŵ, p̂) — four scalars per query point.

Activations: tanh throughout.  NEVER ReLU (breaks second-order autograd).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from hemodyn_pinn.utils.seeds import RFF_SEED, PINN_TRAIN_SEED

# Physical scales (SI)
L_SCALE: float = 0.01628   # m   (parent-artery diameter, from CLAUDE.md §6)
U_SCALE: float = 2e-5      # m/s (inlet velocity giving Re ≈ 0.10)
MU: float = 3.5e-3         # Pa·s


def nondim_coords(x_m: Tensor) -> Tensor:
    """Convert SI coordinates (m) to non-dimensional coordinates."""
    return x_m / L_SCALE


def nondim_velocity(u_ms: Tensor) -> Tensor:
    """Convert SI velocity (m/s) to non-dimensional velocity."""
    return u_ms / U_SCALE


def dim_velocity(u_hat: Tensor) -> Tensor:
    """Convert non-dimensional velocity to SI (m/s)."""
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

    Parameters
    ----------
    n_features:
        D — number of frequency samples.  Output dim = 2D.
    sigma:
        Bandwidth of the Gaussian frequency distribution.
    seed:
        RNG seed for reproducible B.
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
        self.register_buffer("B", B)   # (D, 3) — not a parameter, not trained

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
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # (N, 2D)


# ---------------------------------------------------------------------------
# MLP backbone
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    """Fully-connected network with tanh activations.

    Parameters
    ----------
    in_dim:
        Input feature dimension.
    out_dim:
        Output dimension (4 for PINN: u, v, w, p).
    n_hidden:
        Number of neurons per hidden layer.
    n_layers:
        Number of hidden layers.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 4,
        n_hidden: int = 128,
        n_layers: int = 4,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, n_hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_hidden, n_hidden), nn.Tanh()]
        layers.append(nn.Linear(n_hidden, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# PINN network (main entry point)
# ---------------------------------------------------------------------------


class PINNNetwork(nn.Module):
    """Coordinate-based PINN for steady Stokes flow.

    Input:  x̂ = (x̂, ŷ, ẑ) — non-dimensional coordinates.
    Output: (û, v̂, ŵ, p̂)  — non-dimensional velocity components + pressure.

    Parameters
    ----------
    n_hidden:
        Neurons per hidden layer.
    n_layers:
        Number of hidden layers.
    use_rff:
        Whether to prepend a Random Fourier Features encoder.
    rff_features:
        D (number of RFF frequency samples; ignored if use_rff=False).
    rff_sigma:
        Bandwidth σ for the RFF encoder (ignored if use_rff=False).
    seed:
        Seed for deterministic weight initialisation.
    """

    def __init__(
        self,
        n_hidden: int = 128,
        n_layers: int = 4,
        use_rff: bool = False,
        rff_features: int = 128,
        rff_sigma: float = 1.0,
        seed: int = PINN_TRAIN_SEED,
    ) -> None:
        super().__init__()
        self.use_rff = use_rff

        if use_rff:
            self.encoder: Optional[RFFEncoder] = RFFEncoder(
                n_features=rff_features, sigma=rff_sigma
            )
            in_dim = self.encoder.out_dim
        else:
            self.encoder = None
            in_dim = 3

        torch.manual_seed(seed)
        self.mlp = MLP(in_dim=in_dim, out_dim=4, n_hidden=n_hidden, n_layers=n_layers)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x:
            (N, 3) non-dimensional coordinates.

        Returns
        -------
        Tensor
            (N, 4) — columns: û, v̂, ŵ, p̂.
        """
        h = self.encoder(x) if self.use_rff else x
        return self.mlp(h)

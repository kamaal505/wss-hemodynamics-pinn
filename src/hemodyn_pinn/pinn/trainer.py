"""PINN training loop: Adam → L-BFGS with checkpointing.

Training recipe (CLAUDE.md §3.6):
- Optimiser: Adam (first phase) → L-BFGS (refinement).
- LR schedule: cosine annealing 1e-3 → 1e-6 over Adam phase.
- Collocation points: resampled each epoch from interior_pts_nondim.
- Pressure anchor: fixed outlet point where p̂ = 0.
- Checkpointing: save every `checkpoint_every` epochs; keep best val loss.

New in this version:
    - use_hard_sdf     (Fix A): precomputes SDF at interior + data points;
                                sets lambda_bc = 0 automatically.
    - use_vec_potential (Fix B): sets skip_div_loss = True automatically.
    - use_adaptive_weights (Fix D): wraps loss in SelfAdaptiveLoss and
                                    reverses the gradient on weight params.
"""

from __future__ import annotations

import dataclasses
import logging
import pathlib
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from hemodyn_pinn.pinn.losses import SelfAdaptiveLoss, total_loss
from hemodyn_pinn.pinn.sampling import (
    epoch_rng,
    sample_collocation,
    sample_collocation_biased,
    sample_wall_bc,
)
from hemodyn_pinn.utils.seeds import PINN_COLLOC_SEED

log = logging.getLogger(__name__)


@dataclasses.dataclass
class PINNConfig:
    """Hyperparameters for the PINN training loop.

    Parameters
    ----------
    n_colloc:
        Collocation points per epoch (physics residual).
    n_wall_bc:
        Wall boundary points per epoch (no-slip).
    n_adam:
        Number of Adam iterations (epochs).
    n_lbfgs:
        Maximum L-BFGS iterations (refinement phase).
    lr_adam:
        Initial Adam learning rate.
    lr_lbfgs:
        L-BFGS learning rate (usually 1.0).
    lambda_data, lambda_phys, lambda_bc, lambda_anchor:
        Initial loss weights.  When use_adaptive_weights=True these become
        initial values for the SelfAdaptiveLoss parameters.
        When use_hard_sdf=True, lambda_bc is overridden to 0.0 internally.
    checkpoint_every:
        Save checkpoint every this many Adam epochs.
    device:
        PyTorch device string.
    colloc_seed:
        Base seed for per-epoch collocation sampling.
    wall_bias_frac:
        Fraction of collocation points drawn from the near-wall interior
        subset.  0.0 = uniform; 0.4 recommended for WSS accuracy.
    use_hard_sdf:
        Fix A — enforce no-slip exactly via SDF multiplication.
    use_vec_potential:
        Fix B — use vector-potential parameterisation (div-free by identity).
    use_adaptive_weights:
        Fix D — self-adaptive loss weights (gradient reversal on λ params).
    sa_weight_lr:
        Learning rate for the SelfAdaptiveLoss parameters (only used when
        use_adaptive_weights=True).
    """

    n_colloc: int = 10_000
    n_wall_bc: int = 2_000
    n_adam: int = 50_000
    n_lbfgs: int = 5_000
    lr_adam: float = 1e-3
    lr_lbfgs: float = 1.0
    lambda_data: float = 1.0
    lambda_phys: float = 1.0
    lambda_bc: float = 10.0
    lambda_anchor: float = 1.0
    checkpoint_every: int = 1_000
    device: str = "cpu"
    colloc_seed: int = PINN_COLLOC_SEED
    wall_bias_frac: float = 0.0
    # Architecture / physics flags
    use_hard_sdf: bool = False
    use_vec_potential: bool = False
    use_adaptive_weights: bool = False
    sa_weight_lr: float = 1e-3
    # Magnitude-collapse fixes
    relative_data: bool = True
    lambda_inlet: float = 0.0


class PINNTrainer:
    """Manages Adam → L-BFGS training of a PINNNetwork.

    Parameters
    ----------
    net:
        PINNNetwork to train.
    cfg:
        Training hyperparameters.
    interior_pts_nondim:
        (M, 3) non-dim interior points for collocation sampling.
    wall_pts_nondim:
        (W, 3) non-dim wall centroids for BC sampling.
    anchor_pt_nondim:
        (1, 3) non-dim outlet anchor for pressure = 0.
    x_data:
        (N_d, 3) non-dim MRI voxel centres.
    u_obs_nondim:
        (N_d, 3) non-dim observed velocities.
    wall_pts_m:
        (W, 3) wall centroids in SI metres.  Required when
        cfg.use_hard_sdf=True to build the WallSDF tree.
    out_dir:
        Directory for checkpoints and loss history.
    x_inlet_nondim, u_inlet_nondim:
        (K, 3) non-dim inlet-plane coordinates and target velocities for the
        inflow magnitude constraint.  Active only when cfg.lambda_inlet > 0.
    """

    def __init__(
        self,
        net: nn.Module,
        cfg: PINNConfig,
        interior_pts_nondim: np.ndarray,
        wall_pts_nondim: np.ndarray,
        anchor_pt_nondim: np.ndarray,
        x_data: np.ndarray,
        u_obs_nondim: np.ndarray,
        wall_pts_m: Optional[np.ndarray] = None,
        out_dir: Optional[pathlib.Path | str] = None,
        x_inlet_nondim: Optional[np.ndarray] = None,
        u_inlet_nondim: Optional[np.ndarray] = None,
    ) -> None:
        self.net = net
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.net.to(self.device)

        def _t(arr: np.ndarray) -> Tensor:
            return torch.tensor(arr, dtype=torch.float32, device=self.device)

        self.interior_pts = _t(interior_pts_nondim)
        self.wall_pts = _t(wall_pts_nondim)
        self.anchor_pt = _t(anchor_pt_nondim)
        self.x_data = _t(x_data)
        self.u_obs = _t(u_obs_nondim)

        # Inlet observations for the inflow magnitude constraint (Tier 2)
        self.x_inlet: Optional[Tensor] = None
        self.u_inlet: Optional[Tensor] = None
        self._x_inlet_np: Optional[np.ndarray] = None
        if (
            x_inlet_nondim is not None
            and u_inlet_nondim is not None
            and x_inlet_nondim.shape[0] > 0
        ):
            self.x_inlet = _t(x_inlet_nondim)
            self.u_inlet = _t(u_inlet_nondim)
            self._x_inlet_np = x_inlet_nondim

        self._interior_np = interior_pts_nondim
        self._wall_np = wall_pts_nondim

        # ------------------------------------------------------------------
        # Pre-compute near-wall interior subset (Fix E: wall_bias_frac)
        # ------------------------------------------------------------------
        self._near_wall_np: Optional[np.ndarray] = None
        if cfg.wall_bias_frac > 0.0:
            try:
                from scipy.spatial import KDTree
                tree = KDTree(wall_pts_nondim)
                dists, _ = tree.query(interior_pts_nondim)
                threshold = float(np.percentile(dists, 20))
                mask = dists <= threshold
                self._near_wall_np = interior_pts_nondim[mask]
                log.info(
                    "Wall-biased sampling: %d near-wall pts (%.0f%% of interior, d≤%.4f)",
                    int(mask.sum()), 100.0 * mask.mean(), threshold,
                )
            except ImportError:
                log.warning("scipy not found; wall_bias_frac ignored.")

        # ------------------------------------------------------------------
        # Pre-compute SDF values (Fix A: use_hard_sdf)
        # ------------------------------------------------------------------
        self._sdf_interior: Optional[Tensor] = None
        self._sdf_data: Optional[Tensor] = None
        self._sdf_inlet: Optional[Tensor] = None

        if cfg.use_hard_sdf:
            if wall_pts_m is None:
                raise ValueError(
                    "wall_pts_m is required when cfg.use_hard_sdf=True."
                )
            from hemodyn_pinn.geometry.sdf import WallSDF
            from hemodyn_pinn.pinn.networks import L_SCALE
            sdf_fn = WallSDF(wall_pts_m)

            # Interior collocation points (non-dim → m → distance → non-dim)
            interior_sdf = sdf_fn.nondim(interior_pts_nondim, L_SCALE)
            self._sdf_interior = torch.tensor(
                interior_sdf, dtype=torch.float32, device=self.device
            )

            # MRI data points
            data_sdf = sdf_fn.nondim(x_data, L_SCALE)
            self._sdf_data = torch.tensor(
                data_sdf, dtype=torch.float32, device=self.device
            )

            # Inlet target points (open boundary — non-zero SDF away from wall)
            if self._x_inlet_np is not None:
                inlet_sdf = sdf_fn.nondim(self._x_inlet_np, L_SCALE)
                self._sdf_inlet = torch.tensor(
                    inlet_sdf, dtype=torch.float32, device=self.device
                )
            log.info(
                "SDF pre-computed.  Interior min/mean/max: %.4f / %.4f / %.4f (nondim)",
                float(interior_sdf.min()),
                float(interior_sdf.mean()),
                float(interior_sdf.max()),
            )

        # ------------------------------------------------------------------
        # Self-adaptive loss weights (Fix D)
        # ------------------------------------------------------------------
        self._sa_loss: Optional[SelfAdaptiveLoss] = None
        effective_lambda_bc = 0.0 if cfg.use_hard_sdf else cfg.lambda_bc

        self._use_inlet = self.x_inlet is not None and cfg.lambda_inlet > 0.0

        if cfg.use_adaptive_weights:
            sa_init = {
                "data":   cfg.lambda_data,
                "phys":   cfg.lambda_phys,
                "bc":     effective_lambda_bc,
                "anchor": cfg.lambda_anchor,
            }
            if self._use_inlet:
                sa_init["inlet"] = cfg.lambda_inlet
            self._sa_loss = SelfAdaptiveLoss(sa_init)
            self._sa_loss.to(self.device)
            log.info("SelfAdaptiveLoss initialised: %s", self._sa_loss.lambdas)

        self._effective_lambda_bc = effective_lambda_bc

        # skip div loss when vec-potential is active OR when both A+B are combined
        self._skip_div = cfg.use_vec_potential

        self.history: list[dict] = []
        self.best_loss: float = float("inf")
        self.best_state: Optional[dict] = None

        self.out_dir: Optional[pathlib.Path] = (
            pathlib.Path(out_dir) if out_dir is not None else None
        )
        if self.out_dir is not None:
            self.out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_batch(self, epoch: int) -> tuple[Tensor, Tensor, Optional[Tensor]]:
        """Sample collocation and wall points for one epoch.

        Returns (x_colloc, x_wall, sdf_colloc_or_None).
        """
        rng = epoch_rng(self.cfg.colloc_seed, epoch)

        if self.cfg.wall_bias_frac > 0.0 and self._near_wall_np is not None:
            x_c_np = sample_collocation_biased(
                self._interior_np,
                self._near_wall_np,
                self.cfg.n_colloc,
                self.cfg.wall_bias_frac,
                rng,
            )
        else:
            x_c_np = sample_collocation(self._interior_np, self.cfg.n_colloc, rng)
        x_w_np = sample_wall_bc(self._wall_np, self.cfg.n_wall_bc, rng)

        x_c = torch.tensor(x_c_np, dtype=torch.float32, device=self.device)
        x_w = torch.tensor(x_w_np, dtype=torch.float32, device=self.device)

        # SDF values for the sampled collocation indices
        sdf_c: Optional[Tensor] = None
        if self.cfg.use_hard_sdf and self._sdf_interior is not None:
            # Re-compute which interior indices were sampled to fetch SDF
            # The sampled points are a subset — look them up by matching
            # (exact match is safe because interior_np is a fixed pool)
            idx = _find_indices(self._interior_np, x_c_np)
            sdf_c = self._sdf_interior[idx]

        return x_c, x_w, sdf_c

    def _compute_loss(
        self, x_colloc: Tensor, x_wall: Tensor, sdf_colloc: Optional[Tensor]
    ) -> tuple[Tensor, dict[str, float]]:
        cfg = self.cfg

        if self._sa_loss is not None:
            from hemodyn_pinn.pinn.losses import (
                data_loss as _dl,
                inlet_loss as _il,
                stokes_residual_loss as _pl,
                bc_loss as _bc,
                pressure_anchor_loss as _al,
            )
            L_data   = _dl(self.net, self.x_data, self.u_obs,
                           sdf_vals=self._sdf_data, relative=cfg.relative_data)
            L_phys   = _pl(self.net, x_colloc,
                           sdf_vals=sdf_colloc, skip_div_loss=self._skip_div)
            L_bc     = _bc(self.net, x_wall)
            L_anchor = _al(self.net, self.anchor_pt)

            sa_terms = {"data": L_data, "phys": L_phys, "bc": L_bc, "anchor": L_anchor}
            L_inlet = None
            if self._use_inlet:
                L_inlet = _il(self.net, self.x_inlet, self.u_inlet,
                              sdf_vals=self._sdf_inlet, relative=cfg.relative_data)
                sa_terms["inlet"] = L_inlet

            total = self._sa_loss(sa_terms)
            lam = self._sa_loss.lambdas
            breakdown = {
                "loss_data":      float(L_data.detach()),
                "loss_phys":      float(L_phys.detach()),
                "loss_bc":        float(L_bc.detach()),
                "loss_anchor":    float(L_anchor.detach()),
                "loss_inlet":     float(L_inlet.detach()) if L_inlet is not None else 0.0,
                "loss_total":     float(total.detach()),
                "lambda_data":    lam["data"],
                "lambda_phys":    lam["phys"],
                "lambda_bc":      lam["bc"],
                "lambda_anchor":  lam["anchor"],
            }
        else:
            total, breakdown = total_loss(
                self.net,
                x_data=self.x_data,
                u_obs=self.u_obs,
                x_colloc=x_colloc,
                x_wall=x_wall,
                x_anchor=self.anchor_pt,
                lambda_data=cfg.lambda_data,
                lambda_phys=cfg.lambda_phys,
                lambda_bc=self._effective_lambda_bc,
                lambda_anchor=cfg.lambda_anchor,
                sdf_data=self._sdf_data,
                sdf_colloc=sdf_colloc,
                skip_div_loss=self._skip_div,
                relative_data=cfg.relative_data,
                x_inlet=self.x_inlet if self._use_inlet else None,
                u_inlet=self.u_inlet if self._use_inlet else None,
                sdf_inlet=self._sdf_inlet,
                lambda_inlet=cfg.lambda_inlet,
            )
        return total, breakdown

    def _save_checkpoint(self, tag: str) -> None:
        if self.out_dir is None:
            return
        path = self.out_dir / f"checkpoint_{tag}.pt"
        state = {"state_dict": self.net.state_dict(), "history": self.history}
        if self._sa_loss is not None:
            state["sa_loss_state"] = self._sa_loss.state_dict()
        torch.save(state, path)
        log.debug("Checkpoint saved: %s", path)

    def _selection_metric(self, breakdown: dict) -> float:
        """Metric used to pick the best checkpoint.

        Selecting by total loss rewards the trivial near-zero Stokes solution
        (it has the lowest physics + total loss).  Instead we select by the
        observation misfit — data fit plus, if active, the inflow fit — so a
        collapsed field (high data loss) is never saved as "best".
        """
        metric = breakdown.get("loss_data", breakdown["loss_total"])
        if self._use_inlet:
            metric = metric + breakdown.get("loss_inlet", 0.0)
        return metric

    def _maybe_update_best(self, breakdown: dict) -> None:
        metric = self._selection_metric(breakdown)
        if metric < self.best_loss:
            self.best_loss = metric
            self.best_state = {
                k: v.cpu().clone() for k, v in self.net.state_dict().items()
            }

    # ------------------------------------------------------------------
    # Optimiser construction
    # ------------------------------------------------------------------

    def _build_adam(self) -> torch.optim.Optimizer:
        cfg = self.cfg
        param_groups: list[dict] = [
            {"params": list(self.net.parameters()), "lr": cfg.lr_adam}
        ]
        if self._sa_loss is not None:
            param_groups.append({
                "params": list(self._sa_loss.parameters()),
                "lr": cfg.sa_weight_lr,
            })
        return torch.optim.Adam(param_groups)

    # ------------------------------------------------------------------
    # Adam phase
    # ------------------------------------------------------------------

    def run_adam(self) -> None:
        """Run the Adam optimisation phase with cosine-annealing LR."""
        cfg = self.cfg
        optimizer = self._build_adam()
        # Cosine schedule applies only to the network group (group 0).
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.n_adam,
            eta_min=1e-6,
        )

        self.net.train()
        if self._sa_loss is not None:
            self._sa_loss.train()

        for epoch in range(cfg.n_adam):
            x_c, x_w, sdf_c = self._sample_batch(epoch)

            optimizer.zero_grad()
            loss, breakdown = self._compute_loss(x_c, x_w, sdf_c)
            loss.backward()

            # Fix D: reverse gradients on SA weight params so they are maximised
            if self._sa_loss is not None:
                for param in self._sa_loss.parameters():
                    if param.grad is not None:
                        param.grad.neg_()

            optimizer.step()
            scheduler.step()

            breakdown["epoch"] = epoch
            breakdown["phase"] = "adam"
            breakdown["lr"] = float(scheduler.get_last_lr()[0])
            self.history.append(breakdown)
            self._maybe_update_best(breakdown)

            if (epoch + 1) % cfg.checkpoint_every == 0:
                log.info(
                    "Adam [%d/%d]  total=%.4e  data=%.4e  phys=%.4e  bc=%.4e",
                    epoch + 1, cfg.n_adam,
                    breakdown["loss_total"],
                    breakdown["loss_data"],
                    breakdown["loss_phys"],
                    breakdown["loss_bc"],
                )
                self._save_checkpoint(f"adam_{epoch+1:07d}")

        self._save_checkpoint("adam_final")

    # ------------------------------------------------------------------
    # L-BFGS phase
    # ------------------------------------------------------------------

    def run_lbfgs(self) -> None:
        """Run L-BFGS refinement after Adam."""
        cfg = self.cfg
        optimizer = torch.optim.LBFGS(
            self.net.parameters(),
            lr=cfg.lr_lbfgs,
            max_iter=20,
            line_search_fn="strong_wolfe",
        )

        rng = epoch_rng(cfg.colloc_seed, cfg.n_adam)
        x_c_np = sample_collocation(self._interior_np, cfg.n_colloc, rng)
        x_w_np = sample_wall_bc(self._wall_np, cfg.n_wall_bc, rng)
        x_c = torch.tensor(x_c_np, dtype=torch.float32, device=self.device)
        x_w = torch.tensor(x_w_np, dtype=torch.float32, device=self.device)

        # SDF for the fixed L-BFGS batch
        sdf_c: Optional[Tensor] = None
        if self.cfg.use_hard_sdf and self._sdf_interior is not None:
            idx = _find_indices(self._interior_np, x_c_np)
            sdf_c = self._sdf_interior[idx]

        self.net.train()
        lbfgs_iter = [0]

        def closure() -> Tensor:
            optimizer.zero_grad()
            loss, breakdown = self._compute_loss(x_c, x_w, sdf_c)
            loss.backward()
            breakdown["epoch"] = cfg.n_adam + lbfgs_iter[0]
            breakdown["phase"] = "lbfgs"
            breakdown["lr"] = cfg.lr_lbfgs
            self.history.append(breakdown)
            self._maybe_update_best(breakdown)
            lbfgs_iter[0] += 1
            if lbfgs_iter[0] % 100 == 0:
                log.info(
                    "L-BFGS [%d/%d]  total=%.4e  phys=%.4e",
                    lbfgs_iter[0], cfg.n_lbfgs,
                    breakdown["loss_total"], breakdown["loss_phys"],
                )
            return loss

        outer_steps = max(1, cfg.n_lbfgs // 20)
        for _ in range(outer_steps):
            if lbfgs_iter[0] >= cfg.n_lbfgs:
                break
            optimizer.step(closure)

        self._save_checkpoint("lbfgs_final")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def fit(self) -> None:
        """Run the full Adam → L-BFGS training pipeline."""
        log.info("Starting Adam phase (%d epochs)…", self.cfg.n_adam)
        self.run_adam()
        log.info(
            "Adam complete.  Best loss=%.4e.  Starting L-BFGS (%d iters)…",
            self.best_loss, self.cfg.n_lbfgs,
        )
        self.run_lbfgs()

        if self.best_state is not None:
            self.net.load_state_dict(self.best_state)
            log.info("Best parameters restored (loss=%.4e).", self.best_loss)

        if self.out_dir is not None:
            state = {"state_dict": self.net.state_dict(), "history": self.history}
            if self._sa_loss is not None:
                state["sa_loss_state"] = self._sa_loss.state_dict()
            torch.save(state, self.out_dir / "best_model.pt")
            log.info("Best model saved to %s/best_model.pt", self.out_dir)


# ---------------------------------------------------------------------------
# Index lookup helper
# ---------------------------------------------------------------------------


def _find_indices(pool: np.ndarray, sampled: np.ndarray) -> np.ndarray:
    """Return integer indices of `sampled` rows within `pool`.

    Uses a hash of (x, y, z) floats reinterpreted as uint32 triples for O(N)
    lookup without allocating a full KD-tree.
    """
    pool_view = pool.view(np.uint32).reshape(len(pool), -1)
    samp_view = sampled.view(np.uint32).reshape(len(sampled), -1)
    # Build a dict from row-bytes → index
    lookup = {row.tobytes(): i for i, row in enumerate(pool_view)}
    return np.array([lookup[row.tobytes()] for row in samp_view], dtype=np.int64)

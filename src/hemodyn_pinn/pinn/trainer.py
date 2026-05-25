"""PINN training loop: Adam → L-BFGS with checkpointing.

Training recipe (CLAUDE.md §3.6):
- Optimiser: Adam (first phase) → L-BFGS (refinement).
- LR schedule: cosine annealing 1e-3 → 1e-6 over Adam phase.
- Collocation points: resampled each epoch from interior_pts_nondim.
- Pressure anchor: fixed outlet point where p̂ = 0.
- Checkpointing: save every `checkpoint_every` epochs; keep best val loss.
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

from hemodyn_pinn.pinn.losses import total_loss
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
    lambda_data:
        Weight for the data loss term.
    lambda_phys:
        Weight for the Stokes residual loss.
    lambda_bc:
        Weight for the no-slip BC loss.
    lambda_anchor:
        Weight for the pressure-anchor loss.
    checkpoint_every:
        Save checkpoint every this many Adam epochs.
    device:
        PyTorch device string ("cpu", "cuda", "cuda:0", etc.).
    colloc_seed:
        Base seed for per-epoch collocation sampling.
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
    lambda_anchor: float = 10.0
    checkpoint_every: int = 1_000
    device: str = "cpu"
    colloc_seed: int = PINN_COLLOC_SEED
    wall_bias_frac: float = 0.0
    # Fraction of collocation points drawn from a near-wall interior subset.
    # 0.0 = uniform (default); 0.3 means 30 % from nearest-wall nodes.
    # The near-wall pool is the bottom-20th-percentile by min-distance-to-wall,
    # pre-computed once in PINNTrainer.__init__ using a scipy KDTree.


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
    out_dir:
        Directory for checkpoints and loss history.  Created if absent.
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
        out_dir: Optional[pathlib.Path | str] = None,
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

        self._interior_np = interior_pts_nondim
        self._wall_np = wall_pts_nondim

        # Pre-compute near-wall interior subset for biased collocation sampling.
        self._near_wall_np: Optional[np.ndarray] = None
        if cfg.wall_bias_frac > 0.0:
            try:
                from scipy.spatial import KDTree  # noqa: PLC0415
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
                log.warning(
                    "scipy not found; wall_bias_frac ignored, using uniform sampling."
                )

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

    def _sample_batch(self, epoch: int) -> tuple[Tensor, Tensor]:
        """Sample collocation and wall points for one epoch."""
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
        return x_c, x_w

    def _compute_loss(
        self, x_colloc: Tensor, x_wall: Tensor
    ) -> tuple[Tensor, dict[str, float]]:
        return total_loss(
            self.net,
            x_data=self.x_data,
            u_obs=self.u_obs,
            x_colloc=x_colloc,
            x_wall=x_wall,
            x_anchor=self.anchor_pt,
            lambda_data=self.cfg.lambda_data,
            lambda_phys=self.cfg.lambda_phys,
            lambda_bc=self.cfg.lambda_bc,
            lambda_anchor=self.cfg.lambda_anchor,
        )

    def _save_checkpoint(self, tag: str) -> None:
        if self.out_dir is None:
            return
        path = self.out_dir / f"checkpoint_{tag}.pt"
        torch.save(
            {
                "state_dict": self.net.state_dict(),
                "history": self.history,
            },
            path,
        )
        log.debug("Checkpoint saved: %s", path)

    def _maybe_update_best(self, loss_val: float) -> None:
        if loss_val < self.best_loss:
            self.best_loss = loss_val
            self.best_state = {
                k: v.cpu().clone() for k, v in self.net.state_dict().items()
            }

    # ------------------------------------------------------------------
    # Adam phase
    # ------------------------------------------------------------------

    def run_adam(self) -> None:
        """Run the Adam optimisation phase with cosine-annealing LR."""
        cfg = self.cfg
        optimizer = torch.optim.Adam(self.net.parameters(), lr=cfg.lr_adam)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.n_adam,
            eta_min=1e-6,
        )

        self.net.train()
        for epoch in range(cfg.n_adam):
            x_c, x_w = self._sample_batch(epoch)

            optimizer.zero_grad()
            loss, breakdown = self._compute_loss(x_c, x_w)
            loss.backward()
            optimizer.step()
            scheduler.step()

            breakdown["epoch"] = epoch
            breakdown["phase"] = "adam"
            breakdown["lr"] = float(scheduler.get_last_lr()[0])
            self.history.append(breakdown)

            self._maybe_update_best(breakdown["loss_total"])

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

        # Fix a single collocation batch for L-BFGS (closure must be deterministic)
        rng = epoch_rng(cfg.colloc_seed, cfg.n_adam)   # epoch after Adam
        x_c_np = sample_collocation(self._interior_np, cfg.n_colloc, rng)
        x_w_np = sample_wall_bc(self._wall_np, cfg.n_wall_bc, rng)
        x_c = torch.tensor(x_c_np, dtype=torch.float32, device=self.device)
        x_w = torch.tensor(x_w_np, dtype=torch.float32, device=self.device)

        self.net.train()
        lbfgs_iter = [0]

        def closure() -> Tensor:
            optimizer.zero_grad()
            loss, breakdown = self._compute_loss(x_c, x_w)
            loss.backward()

            breakdown["epoch"] = cfg.n_adam + lbfgs_iter[0]
            breakdown["phase"] = "lbfgs"
            breakdown["lr"] = cfg.lr_lbfgs
            self.history.append(breakdown)
            self._maybe_update_best(breakdown["loss_total"])
            lbfgs_iter[0] += 1

            if lbfgs_iter[0] % 100 == 0:
                log.info(
                    "L-BFGS [%d/%d]  total=%.4e  phys=%.4e",
                    lbfgs_iter[0], cfg.n_lbfgs,
                    breakdown["loss_total"],
                    breakdown["loss_phys"],
                )
            return loss

        # L-BFGS runs multiple inner iterations per step; outer loop
        # controls the total budget.
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
        """Run the full Adam → L-BFGS training pipeline.

        After training, the best-seen parameters are restored to the network.
        """
        log.info("Starting Adam phase (%d epochs)…", self.cfg.n_adam)
        self.run_adam()

        log.info(
            "Adam complete.  Best loss=%.4e.  Starting L-BFGS (%d iters)…",
            self.best_loss,
            self.cfg.n_lbfgs,
        )
        self.run_lbfgs()

        if self.best_state is not None:
            self.net.load_state_dict(self.best_state)
            log.info("Best parameters restored (loss=%.4e).", self.best_loss)

        if self.out_dir is not None:
            torch.save(
                {"state_dict": self.net.state_dict(), "history": self.history},
                self.out_dir / "best_model.pt",
            )
            log.info("Best model saved to %s/best_model.pt", self.out_dir)

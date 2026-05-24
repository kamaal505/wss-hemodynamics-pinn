"""BHPO objective function: WSS NRMSE on held-out wall faces.

The ``BHPOObjective`` class is initialised once with all fixed data (interior
points, wall geometry, MRI observations) and then called repeatedly by the
skopt search loop.  Each call trains a fresh PINN for ``n_adam_trial`` Adam
steps plus a short L-BFGS refinement, then evaluates WSS NRMSE on the 20 %
held-out wall faces that were withheld from BC training.

Design notes
------------
- ``__call__`` accepts a *raw skopt params list* so the instance can be passed
  directly as ``func`` to ``gp_minimize``.
- L-BFGS uses ``n_lbfgs_trial`` (default 500) — enough to refine Adam's
  solution without dominating trial wall-time.
- Failed trials return 2.0 (a conservative penalty above the worst realistic
  NRMSE of ~1.6 seen in the baseline run) so the GP learns to avoid that region.
- WSS computation uses autograd (``compute_wss`` from inference.py); do NOT
  wrap in ``torch.no_grad()``.
"""

from __future__ import annotations

import csv
import logging
import pathlib
import time
from datetime import datetime
from typing import Optional

import numpy as np
import torch

from hemodyn_pinn.bhpo.space import decode_params
from hemodyn_pinn.pinn.inference import compute_wss
from hemodyn_pinn.pinn.networks import PINNNetwork
from hemodyn_pinn.pinn.trainer import PINNConfig, PINNTrainer
from hemodyn_pinn.utils.seeds import BHPO_VAL_SEED

log = logging.getLogger(__name__)

# Penalty returned for trials that crash or produce NaN.
_FAILURE_PENALTY: float = 2.0

# CSV column order for the trial log.
_LOG_COLUMNS = [
    "trial_id", "wss_nrmse", "elapsed_s", "timestamp",
    "lambda_phys", "lambda_bc", "n_layers", "n_hidden",
    "activation", "use_rff", "rff_sigma", "lr_adam",
    "n_colloc", "wall_bias_frac",
]


class BHPOObjective:
    """Callable objective for the GP-based BHPO search.

    Parameters
    ----------
    interior_pts_nondim:
        (M, 3) non-dimensional interior collocation pool.
    wall_pts_nondim:
        (W, 3) non-dimensional wall-centroid coordinates (vessel wall only).
    wall_normals:
        (W, 3) outward unit normals at the wall centroids.
    cfd_wss_pa:
        (W,) CFD WSS magnitudes [Pa] at the wall centroids.
    anchor_pt_nondim:
        (1, 3) pressure-anchor coordinate (one outlet centroid).
    x_data:
        (N_d, 3) non-dimensional MRI voxel centres.
    u_obs_nondim:
        (N_d, 3) non-dimensional observed velocities.
    val_frac:
        Fraction of wall faces held out for WSS NRMSE evaluation.
    n_adam_trial:
        Adam iterations per BHPO trial.
    n_lbfgs_trial:
        L-BFGS iterations per BHPO trial (short refinement only).
    device:
        PyTorch device string.
    trial_log_path:
        Optional path for a CSV per-trial log (created on first call).
    val_seed:
        Seed for the reproducible train/val wall-face split.
    """

    def __init__(
        self,
        interior_pts_nondim: np.ndarray,
        wall_pts_nondim: np.ndarray,
        wall_normals: np.ndarray,
        cfd_wss_pa: np.ndarray,
        anchor_pt_nondim: np.ndarray,
        x_data: np.ndarray,
        u_obs_nondim: np.ndarray,
        val_frac: float = 0.2,
        n_adam_trial: int = 20_000,
        n_lbfgs_trial: int = 500,
        device: str = "cpu",
        trial_log_path: Optional[pathlib.Path | str] = None,
        val_seed: int = BHPO_VAL_SEED,
    ) -> None:
        # ── Train / validation split of wall faces ──────────────────────────
        rng = np.random.default_rng(val_seed)
        n_wall = wall_pts_nondim.shape[0]
        idx = rng.permutation(n_wall)
        n_val = max(1, int(n_wall * val_frac))
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]

        self._train_wall_pts = wall_pts_nondim[train_idx]   # (W_train, 3)
        self._val_wall_pts = wall_pts_nondim[val_idx]       # (W_val, 3)
        self._val_normals = wall_normals[val_idx]           # (W_val, 3)
        self._val_cfd_wss = cfd_wss_pa[val_idx]            # (W_val,)

        log.info(
            "BHPO val split: %d train / %d val wall faces (seed=%d)",
            train_idx.shape[0], val_idx.shape[0], val_seed,
        )

        # ── Fixed data ───────────────────────────────────────────────────────
        self._interior_pts = interior_pts_nondim
        self._anchor_pt = anchor_pt_nondim
        self._x_data = x_data
        self._u_obs = u_obs_nondim

        self.n_adam_trial = n_adam_trial
        self.n_lbfgs_trial = n_lbfgs_trial
        self.device = device

        self._trial_count = 0
        self._trial_log_path: Optional[pathlib.Path] = (
            pathlib.Path(trial_log_path) if trial_log_path is not None else None
        )
        self._log_initialised = False

    # ------------------------------------------------------------------
    # skopt interface
    # ------------------------------------------------------------------

    def __call__(self, params: list) -> float:
        """Train one PINN trial and return WSS NRMSE on held-out faces.

        Parameters
        ----------
        params:
            Raw skopt parameter list (decoded internally via ``decode_params``).
        """
        self._trial_count += 1
        trial_id = self._trial_count
        hp = decode_params(params)

        t0 = time.perf_counter()
        try:
            nrmse = self._run_trial(hp)
        except Exception as exc:
            log.warning("Trial %d failed: %s", trial_id, exc, exc_info=True)
            nrmse = _FAILURE_PENALTY
        elapsed = time.perf_counter() - t0

        log.info(
            "Trial %3d | WSS-NRMSE=%.4f | %.0f s | "
            "λ_phys=%.3g λ_bc=%.3g act=%s rff=%s lr=%.2e n_c=%d bias=%.2f",
            trial_id, nrmse, elapsed,
            hp["lambda_phys"], hp["lambda_bc"], hp["activation"],
            hp["use_rff"], hp["lr_adam"], hp["n_colloc"], hp["wall_bias_frac"],
        )
        self._log_trial(trial_id, hp, nrmse, elapsed)
        return nrmse

    # ------------------------------------------------------------------
    # Trial execution
    # ------------------------------------------------------------------

    def _run_trial(self, hp: dict) -> float:
        net = PINNNetwork(
            n_hidden=hp["n_hidden"],
            n_layers=hp["n_layers"],
            use_rff=hp["use_rff"],
            rff_features=128,
            rff_sigma=hp["rff_sigma"],
            activation=hp["activation"],
        )
        cfg = PINNConfig(
            n_colloc=hp["n_colloc"],
            n_wall_bc=2_000,
            n_adam=self.n_adam_trial,
            n_lbfgs=self.n_lbfgs_trial,
            lr_adam=hp["lr_adam"],
            lr_lbfgs=1.0,
            lambda_data=1.0,
            lambda_phys=hp["lambda_phys"],
            lambda_bc=hp["lambda_bc"],
            lambda_anchor=1.0,
            # Disable periodic file I/O during BHPO — set threshold beyond run length.
            checkpoint_every=self.n_adam_trial + 1,
            device=self.device,
            wall_bias_frac=hp["wall_bias_frac"],
        )
        trainer = PINNTrainer(
            net=net,
            cfg=cfg,
            interior_pts_nondim=self._interior_pts,
            wall_pts_nondim=self._train_wall_pts,
            anchor_pt_nondim=self._anchor_pt,
            x_data=self._x_data,
            u_obs_nondim=self._u_obs,
            out_dir=None,  # no checkpointing during BHPO
        )
        trainer.fit()

        # ── WSS NRMSE on held-out validation faces ───────────────────────────
        net.eval()
        device = torch.device(self.device)
        x_val = torch.tensor(self._val_wall_pts, dtype=torch.float32, device=device)
        n_val = torch.tensor(self._val_normals, dtype=torch.float32, device=device)

        # compute_wss uses autograd internally — must NOT be inside no_grad().
        _, pinn_wss = compute_wss(net, x_val, n_val)
        pinn_wss_np = pinn_wss.detach().cpu().numpy()

        diff = pinn_wss_np - self._val_cfd_wss
        rms_cfd = float(np.sqrt(np.mean(self._val_cfd_wss ** 2)))
        nrmse = float(np.sqrt(np.mean(diff ** 2))) / (rms_cfd + 1e-12)

        # Explicit cleanup helps with long BHPO runs on CPU.
        del net, trainer, pinn_wss, x_val, n_val

        if not np.isfinite(nrmse):
            return _FAILURE_PENALTY
        return nrmse

    # ------------------------------------------------------------------
    # Trial logging
    # ------------------------------------------------------------------

    def _log_trial(
        self,
        trial_id: int,
        hp: dict,
        nrmse: float,
        elapsed: float,
    ) -> None:
        if self._trial_log_path is None:
            return

        self._trial_log_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self._log_initialised and not self._trial_log_path.exists()
        self._log_initialised = True

        row = {
            "trial_id":      trial_id,
            "wss_nrmse":     round(nrmse, 6),
            "elapsed_s":     round(elapsed, 1),
            "timestamp":     datetime.utcnow().isoformat(timespec="seconds"),
            **hp,
        }
        with open(self._trial_log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_LOG_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

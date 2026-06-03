"""BHPO objective function: WSS NRMSE on held-out wall faces.

The ``BHPOObjective`` class is initialised once with all fixed data and then
called by the Optuna (primary) or skopt (legacy) search loop.  Each call
trains a fresh PINN for ``n_adam_trial`` Adam steps plus a short L-BFGS
refinement, then evaluates WSS NRMSE on the 20 % held-out wall faces.

Architecture flags (use_hard_sdf, use_vec_potential, use_adaptive_weights)
are fixed at construction time — the BHPO searches over hyperparameters
within a single architecture family.

WSS computation routes:
    use_hard_sdf=False → compute_wss  (standard autograd Jacobian)
    use_hard_sdf=True  → compute_wss_hard_sdf (analytical formula, faster)
"""

from __future__ import annotations

import csv
import logging
import pathlib
import time
from datetime import datetime
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

from hemodyn_pinn.bhpo.space import decode_params, optuna_suggest
from hemodyn_pinn.pinn.inference import compute_wss_auto
from hemodyn_pinn.pinn.networks import PINNNetwork
from hemodyn_pinn.pinn.trainer import PINNConfig, PINNTrainer
from hemodyn_pinn.utils.seeds import BHPO_VAL_SEED

if TYPE_CHECKING:
    import optuna

log = logging.getLogger(__name__)

_FAILURE_PENALTY: float = 2.0

_LOG_COLUMNS = [
    "trial_id", "wss_nrmse", "elapsed_s", "timestamp",
    "lambda_phys", "lambda_bc", "n_layers", "n_hidden",
    "activation", "use_rff", "rff_sigma", "lr_adam",
    "n_colloc", "wall_bias_frac",
]


class BHPOObjective:
    """Callable objective for BHPO search (Optuna and skopt compatible).

    Parameters
    ----------
    interior_pts_nondim:
        (M, 3) non-dimensional interior collocation pool.
    wall_pts_nondim:
        (W, 3) non-dimensional wall-centroid coordinates.
    wall_normals:
        (W, 3) outward unit normals at the wall centroids.
    cfd_wss_pa:
        (W,) CFD WSS magnitudes [Pa] at the wall centroids.
    anchor_pt_nondim:
        (1, 3) pressure-anchor coordinate.
    x_data:
        (N_d, 3) non-dimensional MRI voxel centres.
    u_obs_nondim:
        (N_d, 3) non-dimensional observed velocities.
    wall_pts_m:
        (W, 3) wall centroids in SI metres.  Required when use_hard_sdf=True.
    val_frac:
        Fraction of wall faces held out for WSS NRMSE evaluation.
    n_adam_trial:
        Adam iterations per BHPO trial.
    n_lbfgs_trial:
        L-BFGS iterations per BHPO trial.
    device:
        PyTorch device string.  "auto" triggers MPS/CUDA/CPU detection.
    trial_log_path:
        Optional CSV per-trial log.
    val_seed:
        Seed for the train/val wall-face split.
    use_hard_sdf:
        Fix A flag, forwarded to PINNNetwork and PINNTrainer.
    use_vec_potential:
        Fix B flag, forwarded to PINNNetwork.
    use_adaptive_weights:
        Fix D flag, forwarded to PINNTrainer.
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
        wall_pts_m: Optional[np.ndarray] = None,
        val_frac: float = 0.2,
        n_adam_trial: int = 10_000,
        n_lbfgs_trial: int = 200,
        device: str = "auto",
        trial_log_path: Optional[pathlib.Path | str] = None,
        val_seed: int = BHPO_VAL_SEED,
        use_hard_sdf: bool = False,
        use_vec_potential: bool = False,
        use_adaptive_weights: bool = False,
    ) -> None:
        from hemodyn_pinn.bhpo.search import auto_device
        self.device = auto_device() if device == "auto" else device
        log.info("BHPO device: %s", self.device)

        # Architecture flags fixed for this search run
        self.use_hard_sdf = use_hard_sdf
        self.use_vec_potential = use_vec_potential
        self.use_adaptive_weights = use_adaptive_weights

        if use_hard_sdf and wall_pts_m is None:
            raise ValueError("wall_pts_m required when use_hard_sdf=True.")
        self._wall_pts_m = wall_pts_m

        # Train / validation split
        rng = np.random.default_rng(val_seed)
        n_wall = wall_pts_nondim.shape[0]
        idx = rng.permutation(n_wall)
        n_val = max(1, int(n_wall * val_frac))
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]

        self._train_wall_pts = wall_pts_nondim[train_idx]
        self._val_wall_pts   = wall_pts_nondim[val_idx]
        self._val_normals    = wall_normals[val_idx]
        self._val_cfd_wss    = cfd_wss_pa[val_idx]

        # Wall points in metres for SDF (only val subset needed for inference,
        # full wall needed for SDF precomputation inside the trainer)
        self._train_wall_m: Optional[np.ndarray] = None
        if wall_pts_m is not None:
            self._train_wall_m = wall_pts_m[train_idx]

        log.info(
            "BHPO val split: %d train / %d val wall faces (seed=%d)",
            train_idx.shape[0], val_idx.shape[0], val_seed,
        )

        self._interior_pts = interior_pts_nondim
        self._anchor_pt = anchor_pt_nondim
        self._x_data = x_data
        self._u_obs = u_obs_nondim

        self.n_adam_trial = n_adam_trial
        self.n_lbfgs_trial = n_lbfgs_trial

        self._trial_count = 0
        self._trial_log_path: Optional[pathlib.Path] = (
            pathlib.Path(trial_log_path) if trial_log_path is not None else None
        )
        self._log_initialised = False

    # ------------------------------------------------------------------
    # Optuna interface (primary)
    # ------------------------------------------------------------------

    def optuna_objective(self, trial: "optuna.Trial") -> float:
        """Optuna objective: sample HPs, train PINN, return WSS NRMSE."""
        self._trial_count += 1
        trial_id = self._trial_count
        hp = optuna_suggest(trial)

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
    # Legacy skopt interface (fallback)
    # ------------------------------------------------------------------

    def __call__(self, params: list) -> float:
        """skopt interface: accepts a raw parameter list."""
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
            use_hard_sdf=self.use_hard_sdf,
            use_vec_potential=self.use_vec_potential,
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
            checkpoint_every=self.n_adam_trial + 1,
            device=self.device,
            wall_bias_frac=hp["wall_bias_frac"],
            use_hard_sdf=self.use_hard_sdf,
            use_vec_potential=self.use_vec_potential,
            use_adaptive_weights=self.use_adaptive_weights,
        )
        trainer = PINNTrainer(
            net=net,
            cfg=cfg,
            interior_pts_nondim=self._interior_pts,
            wall_pts_nondim=self._train_wall_pts,
            anchor_pt_nondim=self._anchor_pt,
            x_data=self._x_data,
            u_obs_nondim=self._u_obs,
            wall_pts_m=self._train_wall_m,
            out_dir=None,
        )
        trainer.fit()

        # WSS NRMSE on held-out validation faces
        net.eval()
        device = torch.device(self.device)
        x_val = torch.tensor(self._val_wall_pts, dtype=torch.float32, device=device)
        n_val = torch.tensor(self._val_normals,  dtype=torch.float32, device=device)

        _, pinn_wss = compute_wss_auto(net, x_val, n_val)
        pinn_wss_np = pinn_wss.detach().cpu().numpy()

        diff = pinn_wss_np - self._val_cfd_wss
        rms_cfd = float(np.sqrt(np.mean(self._val_cfd_wss ** 2)))
        nrmse = float(np.sqrt(np.mean(diff ** 2))) / (rms_cfd + 1e-12)

        del net, trainer, pinn_wss, x_val, n_val
        # Explicit cache clear helps on MPS between trials
        if self.device == "mps":
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

        return nrmse if np.isfinite(nrmse) else _FAILURE_PENALTY

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
            "trial_id":  trial_id,
            "wss_nrmse": round(nrmse, 6),
            "elapsed_s": round(elapsed, 1),
            "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
            **hp,
        }
        with open(self._trial_log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_LOG_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

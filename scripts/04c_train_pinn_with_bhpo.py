"""Retrain the PINN with the winning hyperparameters from the BHPO search.

The winning HPs are always read from data/bhpo_runs/<bhpo.source_case_id>/
(default: caseC).  The trained model is saved under data/checkpoints/ with a
run ID that encodes the geometry, voxel config, and _bhpo suffix so it is
clearly distinguished from the fixed-HP baseline runs.

Usage:
    # Retrain on caseC with BHPO-winning HPs:
    python scripts/04c_train_pinn_with_bhpo.py geometry=caseC

    # Retrain on caseA using the same BHPO result:
    python scripts/04c_train_pinn_with_bhpo.py geometry=caseA

    # Override the retrain budget from the CLI:
    python scripts/04c_train_pinn_with_bhpo.py geometry=caseC \\
        bhpo.n_adam_full=100000 bhpo.n_lbfgs_full=10000

Requirements:
    data/bhpo_runs/<source_case_id>/best_params.json — from 04b_bhpo_search.py
    data/cfd/<case_id>/solution.npz                  — from 02_run_cfd.py
    data/synthetic_mri/<case_id>/<voxel>/mri_obs.npz — from 03_generate_synthetic_mri.py

Output:
    data/checkpoints/<case_id>_<voxel_tag>_bhpo/
        checkpoint_adam_*.pt   — periodic Adam snapshots
        checkpoint_adam_final.pt
        checkpoint_lbfgs_final.pt
        best_model.pt
        loss_history.npy
        bhpo_params.json       — copy of the winning HP dict (provenance record)
"""

from __future__ import annotations

import json
import logging
import pathlib
import sys
import time

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root / "src"))


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    import torch
    from hemodyn_pinn.bhpo.search import load_best_params
    from hemodyn_pinn.cfd.postprocess import load_solution
    from hemodyn_pinn.geometry.anxplore_loader import load_anxplore
    from hemodyn_pinn.mri.operator import MRIObservation
    from hemodyn_pinn.pinn.networks import PINNNetwork
    from hemodyn_pinn.pinn.sampling import (
        make_pressure_anchor,
        select_inlet_targets,
        to_nondim_coords,
        to_nondim_velocity,
    )
    from hemodyn_pinn.pinn.trainer import PINNConfig, PINNTrainer

    case_id = cfg.geometry.case_id

    # ── Load winning hyperparameters from BHPO ───────────────────────────────
    bhpo_source = str(cfg.bhpo.source_case_id)
    bhpo_dir = _root / "data" / "bhpo_runs" / bhpo_source
    best_hp = load_best_params(bhpo_dir)
    log.info("Loaded BHPO HPs from %s:\n  %s", bhpo_dir,
             "  ".join(f"{k}={v}" for k, v in best_hp.items()))

    # ── Load CFD solution ────────────────────────────────────────────────────
    cfd_dir = _root / cfg.output.base_dir / case_id
    npz_path = cfd_dir / "solution.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"CFD solution not found: {npz_path}\n"
            "Run scripts/02_run_cfd.py in WSL first."
        )

    log.info("Loading CFD solution: %s", npz_path)
    sol = load_solution(npz_path)

    # ── Mesh for collocation pool ────────────────────────────────────────────
    vtk_path = _root / cfg.geometry.vtk_path
    log.info("Loading mesh: %s", vtk_path)
    mesh = load_anxplore(vtk_path, case_id)
    interior_pts_nondim = to_nondim_coords(mesh.points_m)

    # ── Wall points and pressure anchor ─────────────────────────────────────
    wall_centroids_m = sol["wall_centroids_m"]
    wall_mask = sol["wall_mask"].astype(bool)
    wall_pts_nondim = to_nondim_coords(wall_centroids_m[wall_mask])

    outlet_mask = sol["outlet_mask"].astype(bool)
    outlet_pts_nondim = to_nondim_coords(wall_centroids_m[outlet_mask])
    anchor_nondim = make_pressure_anchor(outlet_pts_nondim)

    log.info(
        "Collocation pool: %d nodes | wall BC: %d pts",
        interior_pts_nondim.shape[0], wall_pts_nondim.shape[0],
    )

    # ── Load MRI observations ────────────────────────────────────────────────
    voxel_tag = f"voxel_{cfg.mri.voxel_size_mm}mm".replace(".", "p")
    mri_path = _root / cfg.output.mri_base_dir / case_id / voxel_tag / "mri_obs.npz"
    if not mri_path.exists():
        raise FileNotFoundError(
            f"MRI observation not found: {mri_path}\n"
            "Run scripts/03_generate_synthetic_mri.py first."
        )

    log.info("Loading MRI observations: %s", mri_path)
    mri_obs = MRIObservation.load(mri_path)
    x_data_nondim = to_nondim_coords(mri_obs.voxel_centers_mm * 1e-3)
    u_obs_nondim = to_nondim_velocity(mri_obs.velocity_ms)
    log.info("MRI observations: %d voxels", x_data_nondim.shape[0])

    # ── Inflow magnitude constraint (Tier 2) ─────────────────────────────────
    inlet_mask = sol["inlet_mask"].astype(bool)
    inlet_centroids_m = wall_centroids_m[inlet_mask]
    inlet_source = str(cfg.training.get("inlet_source", "none"))
    x_inlet_m, u_inlet_ms = select_inlet_targets(
        source=inlet_source,
        inlet_centroids_m=inlet_centroids_m,
        voxel_centers_m=mri_obs.voxel_centers_mm * 1e-3,
        voxel_velocity_ms=mri_obs.velocity_ms,
        node_points_m=mesh.points_m,
        node_velocity_ms=sol["velocity"],
        band_m=float(cfg.training.get("inlet_band_mm", 2.0)) * 1e-3,
    )
    x_inlet_nondim = to_nondim_coords(x_inlet_m)
    u_inlet_nondim = to_nondim_velocity(u_inlet_ms)
    log.info("Inlet targets (%s): %d points", inlet_source, x_inlet_nondim.shape[0])

    # ── Dense interpolant supervision (anti-collapse, Fixes J–L) ──────────────
    from hemodyn_pinn.pinn.interpolant import (
        build_velocity_interpolant,
        dense_aux_targets,
    )
    # Full retrain uses the training.* curriculum knobs (they scale with the
    # full n_adam budget); the bhpo.* curriculum knobs apply to the short trials.
    lambda_aux = float(cfg.training.get("lambda_aux", cfg.bhpo.get("lambda_aux", 0.0)))
    if lambda_aux > 0.0:
        interp = build_velocity_interpolant(
            x_data_nondim, u_obs_nondim,
            method=str(cfg.training.get("interp_method", "rbf")),
        )
        x_aux_nondim, u_aux_nondim = dense_aux_targets(
            interp, interior_pts_nondim, wall_pts_nondim
        )
    else:
        x_aux_nondim = u_aux_nondim = None

    # ── Architecture flags (must match the BHPO run) ─────────────────────────
    use_hard_sdf         = bool(cfg.bhpo.get("use_hard_sdf", False))
    use_vec_potential    = bool(cfg.bhpo.get("use_vec_potential", False))
    use_adaptive_weights = bool(cfg.bhpo.get("use_adaptive_weights", False))

    # ── Build network with BHPO-winning architecture ─────────────────────────
    net = PINNNetwork(
        n_hidden=best_hp["n_hidden"],
        n_layers=best_hp["n_layers"],
        use_rff=best_hp["use_rff"],
        rff_features=128,
        rff_sigma=best_hp["rff_sigma"],
        activation=best_hp["activation"],
        use_hard_sdf=use_hard_sdf,
        use_vec_potential=use_vec_potential,
    )
    n_params = sum(p.numel() for p in net.parameters())
    log.info(
        "Network: %d params | layers=%d hidden=%d act=%s rff=%s "
        "hard_sdf=%s vec_pot=%s adaptive=%s",
        n_params, best_hp["n_layers"], best_hp["n_hidden"],
        best_hp["activation"], best_hp["use_rff"],
        use_hard_sdf, use_vec_potential, use_adaptive_weights,
    )

    # ── Configure full retrain ───────────────────────────────────────────────
    # Resolve device the same way as the BHPO search.
    from hemodyn_pinn.bhpo.search import auto_device
    retrain_device = str(cfg.bhpo.device)
    if retrain_device == "auto":
        retrain_device = auto_device()

    train_cfg = PINNConfig(
        n_colloc=best_hp["n_colloc"],
        n_wall_bc=2_000,
        n_adam=int(cfg.bhpo.n_adam_full),
        n_lbfgs=int(cfg.bhpo.n_lbfgs_full),
        lr_adam=best_hp["lr_adam"],
        lr_lbfgs=1.0,
        # Fall back to the config default (not 1.0) so a best_params.json that
        # predates the lambda_data search dimension still avoids collapse.
        lambda_data=best_hp.get("lambda_data", float(cfg.training.get("lambda_data", 10.0))),
        lambda_phys=best_hp["lambda_phys"],
        lambda_bc=best_hp["lambda_bc"],
        lambda_anchor=1.0,
        checkpoint_every=int(cfg.training.checkpoint_every),
        device=retrain_device,
        wall_bias_frac=best_hp["wall_bias_frac"],
        use_hard_sdf=use_hard_sdf,
        use_vec_potential=use_vec_potential,
        use_adaptive_weights=use_adaptive_weights,
        relative_data=bool(cfg.training.get("relative_data", True)),
        lambda_inlet=float(cfg.training.get("lambda_inlet", 0.0)),
        lambda_aux=lambda_aux,
        aux_decay_frac=float(cfg.training.get("aux_decay_frac", 0.5)),
        lambda_mag_floor=float(cfg.training.get("lambda_mag_floor", 0.0)),
        n_warmup=int(cfg.training.get("n_warmup", 0)),
        interp_method=str(cfg.training.get("interp_method", "rbf")),
        n_aux=int(cfg.training.get("n_aux", 4096)),
    )

    run_id = f"{case_id}_{voxel_tag}_bhpo"
    out_dir = _root / cfg.output.pinn_base_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save provenance: which HP dict and architecture flags this retrain used.
    # 05_evaluate.py reads these back to reconstruct an identical PINNNetwork.
    with open(out_dir / "bhpo_params.json", "w") as f:
        json.dump(
            {
                "source_case_id": bhpo_source,
                "best_hp": best_hp,
                "use_hard_sdf": use_hard_sdf,
                "use_vec_potential": use_vec_potential,
                "use_adaptive_weights": use_adaptive_weights,
            },
            f, indent=2,
        )

    wall_pts_m = sol["wall_centroids_m"][sol["wall_mask"].astype(bool)]
    trainer = PINNTrainer(
        net=net,
        cfg=train_cfg,
        interior_pts_nondim=interior_pts_nondim,
        wall_pts_nondim=wall_pts_nondim,
        anchor_pt_nondim=anchor_nondim,
        x_data=x_data_nondim,
        u_obs_nondim=u_obs_nondim,
        wall_pts_m=wall_pts_m if use_hard_sdf else None,
        out_dir=out_dir,
        x_inlet_nondim=x_inlet_nondim,
        u_inlet_nondim=u_inlet_nondim,
    )

    # ── Train ────────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    trainer.fit()
    elapsed = time.perf_counter() - t0
    log.info(
        "Training complete in %.1f s.  Best loss=%.4e",
        elapsed, trainer.best_loss,
    )

    # ── Save loss history ────────────────────────────────────────────────────
    history_arr = {k: [d[k] for d in trainer.history] for k in trainer.history[0]}
    np.save(out_dir / "loss_history.npy", history_arr)
    log.info("Loss history saved to %s/loss_history.npy", out_dir)


if __name__ == "__main__":
    main()

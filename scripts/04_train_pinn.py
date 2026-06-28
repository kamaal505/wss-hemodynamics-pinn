"""Train the PINN on one AnXplore case.

Requires:
  data/cfd/<case_id>/solution.npz   — from scripts/02_run_cfd.py (WSL)
  data/synthetic_mri/<case_id>/<voxel_config>/mri_obs.npz — from 03_generate_synthetic_mri.py

Usage (Windows or Linux):
  python scripts/04_train_pinn.py geometry=caseC model=pinn_base

Override hyperparameters on the CLI:
  python scripts/04_train_pinn.py geometry=caseC \\
      training.n_adam=10000 training.device=cuda model=pinn_rff

Output:
  data/checkpoints/<case_id>/<run_id>/
    checkpoint_adam_*.pt
    best_model.pt
    loss_history.npy
"""

from __future__ import annotations

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
    from hemodyn_pinn.cfd.postprocess import load_solution
    from hemodyn_pinn.mri.operator import MRIObservation
    from hemodyn_pinn.pinn.networks import PINNNetwork, nondim_coords, nondim_velocity
    from hemodyn_pinn.pinn.sampling import (
        make_pressure_anchor,
        select_inlet_targets,
        to_nondim_coords,
        to_nondim_velocity,
    )
    from hemodyn_pinn.pinn.trainer import PINNConfig, PINNTrainer

    case_id = cfg.geometry.case_id

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

    # Full-mesh node coordinates for collocation sampling (mm → m → non-dim)
    # sol["velocity"] shape: (N, 3) — gives us N node count.
    # The mesh nodes are loaded from the AnXplore VTK to get coordinates.
    vtk_path = _root / cfg.geometry.vtk_path
    from hemodyn_pinn.geometry.anxplore_loader import load_anxplore
    mesh = load_anxplore(vtk_path, case_id)
    interior_pts_nondim = to_nondim_coords(mesh.points_m)  # (N, 3) non-dim

    # Wall points (wall centroid coords in m)
    wall_centroids_m = sol["wall_centroids_m"]             # (W, 3)
    wall_mask = sol["wall_mask"].astype(bool)
    wall_pts_m = wall_centroids_m[wall_mask]               # keep vessel wall only
    wall_pts_nondim = to_nondim_coords(wall_pts_m)

    # Pressure anchor: pick a centroid on the outlet face
    outlet_mask = sol["outlet_mask"].astype(bool)
    outlet_pts_m = wall_centroids_m[outlet_mask]
    anchor_nondim = make_pressure_anchor(to_nondim_coords(outlet_pts_m))

    log.info(
        "Collocation pool: %d nodes  |  wall BC: %d pts  |  outlet anchor: 1 pt",
        interior_pts_nondim.shape[0],
        wall_pts_nondim.shape[0],
    )

    # ── Load MRI observations ────────────────────────────────────────────────
    voxel_tag = f"voxel_{cfg.mri.voxel_size_mm:.1f}mm".replace(".", "p")
    # Normalise: e.g. 1.0 → "voxel_1p0mm"
    voxel_tag = f"voxel_{cfg.mri.voxel_size_mm}mm".replace(".", "p")
    mri_dir = _root / cfg.output.mri_base_dir / case_id / voxel_tag
    mri_path = mri_dir / "mri_obs.npz"
    if not mri_path.exists():
        raise FileNotFoundError(
            f"MRI observation not found: {mri_path}\n"
            "Run scripts/03_generate_synthetic_mri.py first."
        )

    log.info("Loading MRI observations: %s", mri_path)
    mri_obs = MRIObservation.load(mri_path)

    # Voxel centres: mm → m → non-dim
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
    lambda_aux = float(cfg.training.get("lambda_aux", 0.0))
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

    # ── Build network ────────────────────────────────────────────────────────
    net = PINNNetwork(
        n_hidden=cfg.model.n_hidden,
        n_layers=cfg.model.n_layers,
        use_rff=cfg.model.use_rff,
        rff_features=cfg.model.rff_features,
        rff_sigma=cfg.model.rff_sigma,
        activation=cfg.model.activation,
        use_hard_sdf=cfg.model.use_hard_sdf,
        use_vec_potential=cfg.model.use_vec_potential,
    )
    n_params = sum(p.numel() for p in net.parameters())
    log.info(
        "Network: %d parameters  use_rff=%s  activation=%s  "
        "use_hard_sdf=%s  use_vec_potential=%s",
        n_params, cfg.model.use_rff, cfg.model.activation,
        cfg.model.use_hard_sdf, cfg.model.use_vec_potential,
    )

    # ── Configure training ───────────────────────────────────────────────────
    train_cfg = PINNConfig(
        n_colloc=cfg.training.n_colloc,
        n_wall_bc=cfg.training.n_wall_bc,
        n_adam=cfg.training.n_adam,
        n_lbfgs=cfg.training.n_lbfgs,
        lr_adam=cfg.training.lr_adam,
        lr_lbfgs=cfg.training.lr_lbfgs,
        lambda_data=cfg.training.lambda_data,
        lambda_phys=cfg.training.lambda_phys,
        lambda_bc=cfg.training.lambda_bc,
        lambda_anchor=cfg.training.lambda_anchor,
        checkpoint_every=cfg.training.checkpoint_every,
        device=cfg.training.device,
        wall_bias_frac=float(cfg.training.wall_bias_frac),
        use_hard_sdf=cfg.model.use_hard_sdf,
        use_vec_potential=cfg.model.use_vec_potential,
        use_adaptive_weights=cfg.training.use_adaptive_weights,
        sa_weight_lr=float(cfg.training.sa_weight_lr),
        relative_data=bool(cfg.training.get("relative_data", True)),
        lambda_inlet=float(cfg.training.get("lambda_inlet", 0.0)),
        lambda_aux=lambda_aux,
        aux_decay_frac=float(cfg.training.get("aux_decay_frac", 0.5)),
        lambda_mag_floor=float(cfg.training.get("lambda_mag_floor", 0.0)),
        n_warmup=int(cfg.training.get("n_warmup", 0)),
        interp_method=str(cfg.training.get("interp_method", "rbf")),
        n_aux=int(cfg.training.get("n_aux", 4096)),
    )

    run_id = f"{case_id}_{voxel_tag}_adam{cfg.training.n_adam}"
    out_dir = _root / cfg.output.pinn_base_dir / run_id

    trainer = PINNTrainer(
        net=net,
        cfg=train_cfg,
        interior_pts_nondim=interior_pts_nondim,
        wall_pts_nondim=wall_pts_nondim,
        anchor_pt_nondim=anchor_nondim,
        x_data=x_data_nondim,
        u_obs_nondim=u_obs_nondim,
        wall_pts_m=wall_pts_m if cfg.model.use_hard_sdf else None,
        out_dir=out_dir,
        x_inlet_nondim=x_inlet_nondim,
        u_inlet_nondim=u_inlet_nondim,
        x_aux_nondim=x_aux_nondim,
        u_aux_nondim=u_aux_nondim,
    )

    # ── Train ────────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    trainer.fit()
    elapsed = time.perf_counter() - t0
    log.info("Training complete in %.1f s.  Best loss=%.4e", elapsed, trainer.best_loss)

    # ── Save loss history ────────────────────────────────────────────────────
    history_arr = {k: [d[k] for d in trainer.history] for k in trainer.history[0]}
    np.save(out_dir / "loss_history.npy", history_arr)
    log.info("Loss history saved to %s/loss_history.npy", out_dir)


if __name__ == "__main__":
    main()

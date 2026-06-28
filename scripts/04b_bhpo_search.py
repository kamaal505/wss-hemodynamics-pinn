"""Bayesian Hyperparameter Optimisation for the PINN — search phase.

Must be run on caseC (the reference geometry for BHPO).  The winning
hyperparameters are saved to data/bhpo_runs/caseC/ and consumed by
04c_train_pinn_with_bhpo.py.

Usage:
    python scripts/04b_bhpo_search.py geometry=caseC
    python scripts/04b_bhpo_search.py geometry=caseC bhpo.n_calls=30
    python scripts/04b_bhpo_search.py geometry=caseC bhpo.device=mps
    python scripts/04b_bhpo_search.py geometry=caseC bhpo.use_hard_sdf=true
    python scripts/04b_bhpo_search.py geometry=caseC \\
        bhpo.use_hard_sdf=true bhpo.use_vec_potential=true

Requirements:
    data/cfd/caseC/solution.npz
    data/synthetic_mri/caseC/voxel_*/mri_obs.npz

Output:
    data/bhpo_runs/caseC/
        best_params.json   — winning HP dict (read by 04c)
        convergence.json   — per-trial objective trace
        trial_log.csv      — one row per evaluated HP configuration
        optuna_study.db    — Optuna SQLite study (resumable)
"""

from __future__ import annotations

import logging
import pathlib
import sys
import time

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root / "src"))


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    from hemodyn_pinn.bhpo import BHPOObjective, run_bhpo_search
    from hemodyn_pinn.cfd.postprocess import load_solution
    from hemodyn_pinn.geometry.anxplore_loader import load_anxplore
    from hemodyn_pinn.mri.operator import MRIObservation
    from hemodyn_pinn.pinn.sampling import (
        make_pressure_anchor,
        select_inlet_targets,
        to_nondim_coords,
        to_nondim_velocity,
    )

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

    # ── Mesh for interior collocation pool ──────────────────────────────────
    vtk_path = _root / cfg.geometry.vtk_path
    log.info("Loading mesh: %s", vtk_path)
    mesh = load_anxplore(vtk_path, case_id)
    interior_pts_nondim = to_nondim_coords(mesh.points_m)

    # ── Wall geometry and CFD WSS ────────────────────────────────────────────
    wall_centroids_m = sol["wall_centroids_m"]
    wall_normals     = sol["wall_normals"]
    wss_magnitudes   = sol["wss_magnitudes"]
    wall_mask        = sol["wall_mask"].astype(bool)

    wall_pts_m       = wall_centroids_m[wall_mask]
    wall_normals_w   = wall_normals[wall_mask]
    wss_pa_w         = wss_magnitudes[wall_mask]
    wall_pts_nondim  = to_nondim_coords(wall_pts_m)

    outlet_mask       = sol["outlet_mask"].astype(bool)
    outlet_pts_nondim = to_nondim_coords(wall_centroids_m[outlet_mask])
    anchor_nondim     = make_pressure_anchor(outlet_pts_nondim)

    # ── Load MRI observations ────────────────────────────────────────────────
    voxel_tag = f"voxel_{cfg.mri.voxel_size_mm}mm".replace(".", "p")
    mri_path = _root / cfg.output.mri_base_dir / case_id / voxel_tag / "mri_obs.npz"
    if not mri_path.exists():
        raise FileNotFoundError(
            f"MRI observation not found: {mri_path}\n"
            "Run scripts/03_generate_synthetic_mri.py first."
        )
    log.info("Loading MRI observations: %s", mri_path)
    mri_obs       = MRIObservation.load(mri_path)
    x_data_nondim = to_nondim_coords(mri_obs.voxel_centers_mm * 1e-3)
    u_obs_nondim  = to_nondim_velocity(mri_obs.velocity_ms)

    # ── Inflow magnitude constraint (Tier 2) ─────────────────────────────────
    inlet_mask        = sol["inlet_mask"].astype(bool)
    inlet_centroids_m = wall_centroids_m[inlet_mask]
    inlet_source      = str(cfg.training.get("inlet_source", "none"))
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

    # ── Dense interpolant supervision (anti-collapse, Fixes J–L) ──────────────
    from hemodyn_pinn.pinn.interpolant import (
        build_velocity_interpolant,
        dense_aux_targets,
    )
    lambda_aux = float(cfg.bhpo.get("lambda_aux", 0.0))
    if lambda_aux > 0.0:
        interp = build_velocity_interpolant(
            x_data_nondim, u_obs_nondim,
            method=str(cfg.bhpo.get("interp_method", "rbf")),
        )
        x_aux_nondim, u_aux_nondim = dense_aux_targets(
            interp, interior_pts_nondim, wall_pts_nondim
        )
    else:
        x_aux_nondim = u_aux_nondim = None

    log.info(
        "Data summary: %d interior pts | %d wall pts | %d MRI voxels | "
        "%d inlet pts (%s)",
        interior_pts_nondim.shape[0],
        wall_pts_nondim.shape[0],
        x_data_nondim.shape[0],
        x_inlet_nondim.shape[0],
        inlet_source,
    )

    # ── Architecture flags from config ───────────────────────────────────────
    use_hard_sdf         = bool(cfg.bhpo.get("use_hard_sdf", False))
    use_vec_potential    = bool(cfg.bhpo.get("use_vec_potential", False))
    use_adaptive_weights = bool(cfg.bhpo.get("use_adaptive_weights", False))
    backend              = str(cfg.bhpo.get("backend", "auto"))

    # ── Build BHPO objective ─────────────────────────────────────────────────
    out_dir = _root / "data" / "bhpo_runs" / case_id
    out_dir.mkdir(parents=True, exist_ok=True)

    objective = BHPOObjective(
        interior_pts_nondim=interior_pts_nondim,
        wall_pts_nondim=wall_pts_nondim,
        wall_normals=wall_normals_w,
        cfd_wss_pa=wss_pa_w,
        anchor_pt_nondim=anchor_nondim,
        x_data=x_data_nondim,
        u_obs_nondim=u_obs_nondim,
        wall_pts_m=wall_pts_m if use_hard_sdf else None,
        val_frac=float(cfg.bhpo.val_frac),
        n_adam_trial=int(cfg.bhpo.n_adam_trial),
        n_lbfgs_trial=int(cfg.bhpo.n_lbfgs_trial),
        device=str(cfg.bhpo.device),
        trial_log_path=out_dir / "trial_log.csv",
        use_hard_sdf=use_hard_sdf,
        use_vec_potential=use_vec_potential,
        use_adaptive_weights=use_adaptive_weights,
        x_inlet_nondim=x_inlet_nondim if x_inlet_nondim.shape[0] > 0 else None,
        u_inlet_nondim=u_inlet_nondim if x_inlet_nondim.shape[0] > 0 else None,
        lambda_inlet=float(cfg.training.get("lambda_inlet", 0.0)),
        relative_data=bool(cfg.training.get("relative_data", True)),
        x_aux_nondim=x_aux_nondim,
        u_aux_nondim=u_aux_nondim,
        lambda_aux=lambda_aux,
        aux_decay_frac=float(cfg.bhpo.get("aux_decay_frac", 0.5)),
        lambda_mag_floor=float(cfg.bhpo.get("lambda_mag_floor", 0.0)),
        n_warmup=int(cfg.bhpo.get("n_warmup", 0)),
        n_aux=int(cfg.bhpo.get("n_aux", 4096)),
        wss_batch_size=int(cfg.bhpo.get("wss_batch_size", 4096)),
    )

    log.info(
        "Starting BHPO: %d trials (%d random) on %s | "
        "backend=%s hard_sdf=%s vec_pot=%s adaptive=%s",
        cfg.bhpo.n_calls, cfg.bhpo.n_initial_points, case_id,
        backend, use_hard_sdf, use_vec_potential, use_adaptive_weights,
    )
    t0 = time.perf_counter()
    result, best_hp = run_bhpo_search(
        objective_fn=objective,
        n_calls=int(cfg.bhpo.n_calls),
        n_initial_points=int(cfg.bhpo.n_initial_points),
        out_dir=out_dir,
        backend=backend,
    )
    elapsed = time.perf_counter() - t0

    log.info(
        "BHPO finished in %.0f s (%.1f s/trial avg).",
        elapsed, elapsed / int(cfg.bhpo.n_calls),
    )
    if objective.n_oom_trials or objective.n_error_trials:
        log.warning(
            "BHPO trial failures: %d OOM (retried then penalised), %d error. "
            "See status column in %s/trial_log.csv.",
            objective.n_oom_trials, objective.n_error_trials, out_dir,
        )
    log.info("Best params saved to %s/best_params.json", out_dir)


if __name__ == "__main__":
    main()

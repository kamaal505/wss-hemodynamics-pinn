"""Bayesian Hyperparameter Optimisation for the PINN — search phase.

Must be run on caseC (the reference geometry for BHPO).  The winning
hyperparameters are saved to data/bhpo_runs/caseC/ and consumed by
04c_train_pinn_with_bhpo.py.

Usage:
    python scripts/04b_bhpo_search.py geometry=caseC
    python scripts/04b_bhpo_search.py geometry=caseC bhpo.n_calls=30
    python scripts/04b_bhpo_search.py geometry=caseC bhpo.device=cuda

Requirements:
    data/cfd/caseC/solution.npz         — from 02_run_cfd.py
    data/synthetic_mri/caseC/voxel_*/mri_obs.npz — from 03_generate_synthetic_mri.py

Output:
    data/bhpo_runs/caseC/
        best_params.json   — winning HP dict (read by 04c)
        result.pkl         — full skopt OptimizeResult
        convergence.json   — per-trial objective trace
        trial_log.csv      — one row per evaluated HP configuration
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

    # ── Load mesh for interior collocation pool ──────────────────────────────
    vtk_path = _root / cfg.geometry.vtk_path
    log.info("Loading mesh: %s", vtk_path)
    mesh = load_anxplore(vtk_path, case_id)
    interior_pts_nondim = to_nondim_coords(mesh.points_m)  # (N, 3)

    # ── Wall geometry and CFD WSS ────────────────────────────────────────────
    wall_centroids_m = sol["wall_centroids_m"]     # (W, 3)
    wall_normals = sol["wall_normals"]             # (W, 3)
    wss_magnitudes = sol["wss_magnitudes"]         # (W,) Pa
    wall_mask = sol["wall_mask"].astype(bool)

    wall_pts_m = wall_centroids_m[wall_mask]
    wall_normals_w = wall_normals[wall_mask]
    wss_pa_w = wss_magnitudes[wall_mask]
    wall_pts_nondim = to_nondim_coords(wall_pts_m)

    # Pressure anchor: one outlet centroid, p̂ = 0.
    outlet_mask = sol["outlet_mask"].astype(bool)
    outlet_pts_nondim = to_nondim_coords(wall_centroids_m[outlet_mask])
    anchor_nondim = make_pressure_anchor(outlet_pts_nondim)

    # ── Load MRI observations ────────────────────────────────────────────────
    voxel_tag = f"voxel_{cfg.mri.voxel_size_mm:.1f}mm".replace(".", "p")
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

    log.info(
        "Data summary: %d interior pts | %d wall pts | %d MRI voxels",
        interior_pts_nondim.shape[0],
        wall_pts_nondim.shape[0],
        x_data_nondim.shape[0],
    )

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
        val_frac=float(cfg.bhpo.val_frac),
        n_adam_trial=int(cfg.bhpo.n_adam_trial),
        n_lbfgs_trial=int(cfg.bhpo.n_lbfgs_trial),
        device=str(cfg.bhpo.device),
        trial_log_path=out_dir / "trial_log.csv",
    )

    # ── Run search ───────────────────────────────────────────────────────────
    log.info(
        "Starting BHPO search: %d calls (%d initial random) on %s",
        cfg.bhpo.n_calls, cfg.bhpo.n_initial_points, case_id,
    )
    t0 = time.perf_counter()
    result, best_hp = run_bhpo_search(
        objective_fn=objective,
        n_calls=int(cfg.bhpo.n_calls),
        n_initial_points=int(cfg.bhpo.n_initial_points),
        out_dir=out_dir,
    )
    elapsed = time.perf_counter() - t0

    log.info(
        "BHPO finished in %.0f s (%.1f s/trial avg).",
        elapsed, elapsed / int(cfg.bhpo.n_calls),
    )
    log.info("Best WSS-NRMSE = %.4f", float(result.fun))
    log.info("Best params saved to %s/best_params.json", out_dir)


if __name__ == "__main__":
    main()

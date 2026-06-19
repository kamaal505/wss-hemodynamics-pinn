"""Diagnose PINN magnitude collapse — the unit of measurement for the fixes.

Steady Stokes is linear and homogeneous, so u ≡ 0 is an exact zero-residual
solution.  When the data term is under-weighted the network collapses toward
that trivial solution and WSS is grossly underestimated (the original symptom:
NRMSE > 0.8 regardless of voxel resolution).  This script quantifies the
collapse so each fix can be measured.

It loads a trained checkpoint (architecture reconstructed exactly as
05_evaluate.py does), the CFD ground truth, and the MRI observations, then
reports:
  * predicted/CFD velocity-magnitude ratio at the MRI voxels (≈ 1 is healthy,
    ≪ 1 means collapse),
  * predicted/CFD WSS-magnitude ratio + WSS NRMSE at the wall faces,
  * each per-term loss at the saved weights.

Usage:
    # Current fixed-HP baseline run:
    python scripts/00_diagnose_collapse.py geometry=caseC eval.run_suffix=adam50000 model=pinn_base

    # BHPO-retrained run (default suffix):
    python scripts/00_diagnose_collapse.py geometry=caseC
"""

from __future__ import annotations

import json
import logging
import pathlib
import sys

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
    from hemodyn_pinn.eval.field_metrics import nrmse
    from hemodyn_pinn.geometry.anxplore_loader import load_anxplore
    from hemodyn_pinn.mri.operator import MRIObservation
    from hemodyn_pinn.pinn.inference import (
        compute_wss_auto,
        predict_velocity_field,
    )
    from hemodyn_pinn.pinn.losses import (
        bc_loss,
        data_loss,
        inlet_loss,
        pressure_anchor_loss,
        stokes_residual_loss,
    )
    from hemodyn_pinn.pinn.networks import L_SCALE, PINNNetwork
    from hemodyn_pinn.pinn.sampling import (
        make_pressure_anchor,
        select_inlet_targets,
        to_nondim_coords,
        to_nondim_velocity,
    )

    case_id = cfg.geometry.case_id
    voxel_tag = f"voxel_{cfg.mri.voxel_size_mm}mm".replace(".", "p")
    run_id = f"{case_id}_{voxel_tag}_{cfg.eval.run_suffix}"

    ckpt_dir = _root / cfg.output.pinn_base_dir / run_id
    ckpt_path = ckpt_dir / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\nRun 04/04c training first."
        )
    log.info("Checkpoint: %s", ckpt_path)

    # ── Reconstruct network (same logic as 05_evaluate.py) ───────────────────
    bhpo_json = ckpt_dir / "bhpo_params.json"
    if bhpo_json.exists():
        rec = json.loads(bhpo_json.read_text())
        hp = rec["best_hp"]
        use_hard_sdf = bool(rec.get("use_hard_sdf", False))
        use_vec_potential = bool(rec.get("use_vec_potential", False))
        net = PINNNetwork(
            n_hidden=int(hp["n_hidden"]), n_layers=int(hp["n_layers"]),
            use_rff=bool(hp["use_rff"]), rff_features=128,
            rff_sigma=float(hp.get("rff_sigma", 1.0)),
            activation=str(hp.get("activation", "tanh")),
            use_hard_sdf=use_hard_sdf, use_vec_potential=use_vec_potential,
        )
    else:
        use_hard_sdf = bool(cfg.model.use_hard_sdf)
        use_vec_potential = bool(cfg.model.use_vec_potential)
        net = PINNNetwork(
            n_hidden=cfg.model.n_hidden, n_layers=cfg.model.n_layers,
            use_rff=cfg.model.use_rff, rff_features=cfg.model.rff_features,
            rff_sigma=cfg.model.rff_sigma, activation=cfg.model.activation,
            use_hard_sdf=use_hard_sdf, use_vec_potential=use_vec_potential,
        )

    device = torch.device(cfg.eval.device)
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    net.load_state_dict(ckpt["state_dict"])
    net.to(device)
    net.eval()

    # ── Load CFD + MRI ────────────────────────────────────────────────────────
    sol = load_solution(_root / cfg.output.base_dir / case_id / "solution.npz")
    wall_mask = sol["wall_mask"].astype(bool)
    wall_centroids_m = sol["wall_centroids_m"][wall_mask]
    wall_normals = sol["wall_normals"][wall_mask]
    ref_wss_pa = sol["wss_magnitudes"][wall_mask]

    mesh = load_anxplore(_root / cfg.geometry.vtk_path, case_id)

    mri_path = _root / cfg.output.mri_base_dir / case_id / voxel_tag / "mri_obs.npz"
    mri_obs = MRIObservation.load(mri_path)
    x_data_nondim = to_nondim_coords(mri_obs.voxel_centers_mm * 1e-3)
    u_obs_nondim = to_nondim_velocity(mri_obs.velocity_ms)

    # SDF for hard-SDF de-scaling of interior/voxel predictions
    sdf_fn = None
    if use_hard_sdf:
        from hemodyn_pinn.geometry.sdf import WallSDF
        sdf_fn = WallSDF(wall_centroids_m)

    def _sdf_t(pts_nondim: np.ndarray):
        if sdf_fn is None:
            return None
        return torch.tensor(
            sdf_fn.nondim(pts_nondim, L_SCALE), dtype=torch.float32, device=device
        )

    # ── Velocity-magnitude ratio at the MRI voxels ───────────────────────────
    x_data_t = torch.tensor(x_data_nondim, dtype=torch.float32, device=device)
    u_pred_ms, _ = predict_velocity_field(net, x_data_t, sdf_vals=_sdf_t(x_data_nondim))
    u_pred_ms = u_pred_ms.cpu().numpy()
    u_obs_ms = mri_obs.velocity_ms

    def _rms(a):
        return float(np.sqrt(np.mean(np.sum(a ** 2, axis=1))))

    vel_ratio = _rms(u_pred_ms) / (_rms(u_obs_ms) + 1e-30)

    # ── WSS-magnitude ratio + NRMSE at the wall ──────────────────────────────
    wall_nondim = to_nondim_coords(wall_centroids_m)
    wss_pred = []
    bs = int(cfg.eval.wss_batch_size)
    for s in range(0, wall_nondim.shape[0], bs):
        e = min(s + bs, wall_nondim.shape[0])
        xb = torch.tensor(wall_nondim[s:e], dtype=torch.float32, device=device)
        nb = torch.tensor(wall_normals[s:e], dtype=torch.float32, device=device)
        _, tau = compute_wss_auto(net, xb, nb)
        wss_pred.append(tau.detach().cpu().numpy())
    wss_pred = np.concatenate(wss_pred)

    wss_ratio = float(np.sqrt(np.mean(wss_pred ** 2))) / (
        float(np.sqrt(np.mean(ref_wss_pa ** 2))) + 1e-30
    )
    wss_nrmse = nrmse(wss_pred, ref_wss_pa)

    # ── Per-term losses at the saved weights ─────────────────────────────────
    rng = np.random.default_rng(0)
    interior_nondim = to_nondim_coords(mesh.points_m)
    n_c = min(5000, interior_nondim.shape[0])
    colloc = interior_nondim[rng.choice(interior_nondim.shape[0], n_c, replace=False)]
    colloc_t = torch.tensor(colloc, dtype=torch.float32, device=device)
    wall_bc_t = torch.tensor(wall_nondim, dtype=torch.float32, device=device)
    outlet_mask = sol["outlet_mask"].astype(bool)
    anchor = make_pressure_anchor(to_nondim_coords(sol["wall_centroids_m"][outlet_mask]))
    anchor_t = torch.tensor(anchor, dtype=torch.float32, device=device)

    u_obs_t = torch.tensor(u_obs_nondim, dtype=torch.float32, device=device)
    l_data_rel = float(data_loss(net, x_data_t, u_obs_t, sdf_vals=_sdf_t(x_data_nondim), relative=True).detach())
    l_data_abs = float(data_loss(net, x_data_t, u_obs_t, sdf_vals=_sdf_t(x_data_nondim), relative=False).detach())
    l_phys = float(stokes_residual_loss(net, colloc_t, sdf_vals=_sdf_t(colloc), skip_div_loss=use_vec_potential).detach())
    l_bc = float(bc_loss(net, wall_bc_t).detach())
    l_anchor = float(pressure_anchor_loss(net, anchor_t).detach())

    inlet_centroids_m = sol["wall_centroids_m"][sol["inlet_mask"].astype(bool)]
    xi_m, ui_ms = select_inlet_targets(
        source=str(cfg.training.get("inlet_source", "mri")),
        inlet_centroids_m=inlet_centroids_m,
        voxel_centers_m=mri_obs.voxel_centers_mm * 1e-3,
        voxel_velocity_ms=mri_obs.velocity_ms,
        node_points_m=mesh.points_m,
        node_velocity_ms=sol["velocity"],
        band_m=float(cfg.training.get("inlet_band_mm", 2.0)) * 1e-3,
    )
    l_inlet = None
    if xi_m.shape[0] > 0:
        xi = to_nondim_coords(xi_m)
        xi_t = torch.tensor(xi, dtype=torch.float32, device=device)
        ui_t = torch.tensor(to_nondim_velocity(ui_ms), dtype=torch.float32, device=device)
        l_inlet = float(inlet_loss(net, xi_t, ui_t, sdf_vals=_sdf_t(xi), relative=True).detach())

    # ── Report ────────────────────────────────────────────────────────────────
    bar = "=" * 66
    print(f"\n{bar}\nCOLLAPSE DIAGNOSTIC - {run_id}\n{bar}")
    print(f"  velocity ratio (RMS |u_pred| / RMS |u_obs|) : {vel_ratio:8.4f}   (1.0 healthy, <<1 collapsed)")
    print(f"  WSS ratio      (RMS pred / RMS CFD)         : {wss_ratio:8.4f}")
    print(f"  WSS NRMSE                                   : {wss_nrmse:8.4f}   (>0.8 = collapsed)")
    print("-" * 66)
    print(f"  loss_data (relative) : {l_data_rel:.4e}   (~1.0 means predicting ~nothing)")
    print(f"  loss_data (absolute) : {l_data_abs:.4e}")
    print(f"  loss_phys            : {l_phys:.4e}")
    print(f"  loss_bc              : {l_bc:.4e}")
    print(f"  loss_anchor          : {l_anchor:.4e}")
    if l_inlet is not None:
        print(f"  loss_inlet (relative): {l_inlet:.4e}   ({xi_m.shape[0]} inlet pts)")
    print(bar)
    verdict = "COLLAPSED" if (wss_ratio < 0.5 or wss_nrmse > 0.8) else "HEALTHY"
    print(f"  VERDICT: {verdict}\n{bar}\n")


if __name__ == "__main__":
    main()

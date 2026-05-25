"""Evaluate a trained PINN checkpoint against CFD ground truth.

Loads the best checkpoint from a training run, computes WSS on all vessel-wall
faces via autograd, and saves a full metric suite to data/results/<run_id>/.

Usage:
    # Evaluate the BHPO-retrained model for caseC, 1 mm voxel (default suffix):
    python scripts/05_evaluate.py geometry=caseC

    # Evaluate a fixed-HP baseline run:
    python scripts/05_evaluate.py geometry=caseC eval.run_suffix=adam50000 model=pinn_base

    # Evaluate caseA:
    python scripts/05_evaluate.py geometry=caseA

    # Override voxel config:
    python scripts/05_evaluate.py geometry=caseC mri=voxel_2p0mm

Requirements:
    data/checkpoints/<run_id>/best_model.pt    — from 04c (or 04) training
    data/cfd/<case_id>/solution.npz            — from 02_run_cfd.py (WSL)

Output (data/results/<run_id>/):
    eval_metrics.json      — all scalar metrics (NRMSE, R², MAE, BA stats, ICC,
                             conservation error, descriptive WSS statistics)
    wss_comparison.npz     — per-face arrays: pinn_wss_pa, ref_wss_pa,
                             diff_pa, wall_centroids_m, wall_normals
    loss_curve.pdf         — training loss curves (if loss_history.npy present)
    wss_scatter.pdf        — PINN vs CFD scatter
    bland_altman.pdf       — Bland–Altman agreement plot
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


# ── Batched WSS computation ────────────────────────────────────────────────────

def _compute_wss_batched(
    net,
    wall_pts_nondim: np.ndarray,
    wall_normals: np.ndarray,
    batch_size: int,
    device,
) -> np.ndarray:
    """Compute WSS magnitudes in batches to avoid autograd OOM on large walls.

    Returns
    -------
    np.ndarray
        (W,) WSS magnitudes [Pa].
    """
    import torch
    from hemodyn_pinn.pinn.inference import compute_wss

    net.eval()
    wss_all = []
    W = wall_pts_nondim.shape[0]

    for start in range(0, W, batch_size):
        end = min(start + batch_size, W)
        x_b = torch.tensor(wall_pts_nondim[start:end], dtype=torch.float32,
                           device=device)
        n_b = torch.tensor(wall_normals[start:end], dtype=torch.float32,
                           device=device)
        _, tau_mag = compute_wss(net, x_b, n_b)
        wss_all.append(tau_mag.detach().cpu().numpy())

    return np.concatenate(wss_all, axis=0)   # (W,)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    import torch
    from hemodyn_pinn.cfd.postprocess import load_solution
    from hemodyn_pinn.eval import (
        compute_wss_metrics,
        bland_altman_stats,
        icc_one_way,
        flow_rate_conservation_error,
    )
    from hemodyn_pinn.pinn.inference import predict_velocity_field
    from hemodyn_pinn.pinn.networks import PINNNetwork
    from hemodyn_pinn.pinn.sampling import to_nondim_coords

    case_id = cfg.geometry.case_id
    voxel_tag = f"voxel_{cfg.mri.voxel_size_mm}mm".replace(".", "p")
    run_id = f"{case_id}_{voxel_tag}_{cfg.eval.run_suffix}"

    # ── Locate checkpoint ───────────────────────────────────────────────────
    ckpt_dir = _root / cfg.output.pinn_base_dir / run_id
    ckpt_path = ckpt_dir / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Run the appropriate training script first (04c or 04)."
        )
    log.info("Checkpoint: %s", ckpt_path)

    # ── Determine network architecture ──────────────────────────────────────
    bhpo_json = ckpt_dir / "bhpo_params.json"
    if bhpo_json.exists():
        best_hp = json.loads(bhpo_json.read_text())["best_hp"]
        log.info("Architecture from BHPO: %s", best_hp)
        net = PINNNetwork(
            n_hidden=int(best_hp["n_hidden"]),
            n_layers=int(best_hp["n_layers"]),
            use_rff=bool(best_hp["use_rff"]),
            rff_features=128,
            rff_sigma=float(best_hp.get("rff_sigma", 1.0)),
            activation=str(best_hp.get("activation", "tanh")),
        )
    else:
        log.info("Architecture from model config: n_hidden=%d n_layers=%d",
                 cfg.model.n_hidden, cfg.model.n_layers)
        net = PINNNetwork(
            n_hidden=cfg.model.n_hidden,
            n_layers=cfg.model.n_layers,
            use_rff=cfg.model.use_rff,
            rff_features=cfg.model.rff_features,
            rff_sigma=cfg.model.rff_sigma,
            activation=cfg.model.activation,
        )

    # ── Load checkpoint weights ─────────────────────────────────────────────
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    net.load_state_dict(ckpt["state_dict"])
    device = torch.device(cfg.eval.device)
    net.to(device)
    net.eval()
    n_params = sum(p.numel() for p in net.parameters())
    log.info("Loaded network: %d parameters", n_params)

    # ── Load CFD solution ───────────────────────────────────────────────────
    cfd_dir = _root / cfg.output.base_dir / case_id
    npz_path = cfd_dir / "solution.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"CFD solution not found: {npz_path}\n"
            "Run scripts/02_run_cfd.py in WSL first."
        )
    log.info("Loading CFD solution: %s", npz_path)
    sol = load_solution(npz_path)

    wall_mask = sol["wall_mask"].astype(bool)
    inlet_mask = sol["inlet_mask"].astype(bool)
    outlet_mask = sol["outlet_mask"].astype(bool)

    wall_centroids_m = sol["wall_centroids_m"][wall_mask]   # (W, 3) m
    wall_normals = sol["wall_normals"][wall_mask]           # (W, 3)
    ref_wss_pa = sol["wss_magnitudes"][wall_mask]           # (W,) Pa
    log.info("Wall faces: %d | inlet: %d | outlet: %d",
             wall_mask.sum(), inlet_mask.sum(), outlet_mask.sum())

    # ── Compute PINN WSS on all wall faces ──────────────────────────────────
    wall_pts_nondim = to_nondim_coords(wall_centroids_m)

    log.info("Computing PINN WSS on %d wall faces (batch=%d)...",
             wall_pts_nondim.shape[0], cfg.eval.wss_batch_size)
    t0 = time.perf_counter()
    pinn_wss_pa = _compute_wss_batched(
        net,
        wall_pts_nondim,
        wall_normals,
        batch_size=int(cfg.eval.wss_batch_size),
        device=device,
    )
    elapsed_wss = time.perf_counter() - t0
    log.info("WSS inference: %.1f s", elapsed_wss)

    # ── Conservation check ──────────────────────────────────────────────────
    import torch as _torch

    def _predict_fn(pts_nondim: np.ndarray):
        x = _torch.tensor(pts_nondim, dtype=_torch.float32, device=device)
        u_ms, p_pa = predict_velocity_field(net, x)
        return u_ms.cpu().numpy(), p_pa.cpu().numpy()

    inlet_centroids_m = sol["wall_centroids_m"][inlet_mask]
    outlet_centroids_m = sol["wall_centroids_m"][outlet_mask]
    inlet_normals = sol["wall_normals"][inlet_mask]
    outlet_normals = sol["wall_normals"][outlet_mask]

    u_inlet, _ = _predict_fn(to_nondim_coords(inlet_centroids_m))
    u_outlet, _ = _predict_fn(to_nondim_coords(outlet_centroids_m))

    cons = flow_rate_conservation_error(
        u_inlet, inlet_normals, u_outlet, outlet_normals
    )
    log.info(
        "Conservation: Q_in=%.3e  Q_out=%.3e  error=%.3f",
        cons["q_in_proxy"], cons["q_out_proxy"], cons["conservation_error"],
    )

    # ── Compute all metrics ─────────────────────────────────────────────────
    wss_metrics = compute_wss_metrics(pinn_wss_pa, ref_wss_pa)
    ba_stats = bland_altman_stats(pinn_wss_pa, ref_wss_pa)
    icc = icc_one_way(pinn_wss_pa, ref_wss_pa)

    log.info(
        "WSS metrics: NRMSE=%.4f  R²=%.4f  MAE=%.4e Pa  "
        "peak_err=%.4f  mean_rel_err=%.4f",
        wss_metrics["nrmse"], wss_metrics["r2"],
        wss_metrics["mae"], wss_metrics["peak_wss_error"],
        wss_metrics["mean_relative_error"],
    )
    log.info(
        "Bland–Altman: bias=%.3e Pa  LoA=[%.3e, %.3e]  within=%.1f%%  ICC=%.4f",
        ba_stats["bias_pa"],
        ba_stats["loa_lower_pa"], ba_stats["loa_upper_pa"],
        ba_stats["pct_within_loa"], icc,
    )

    # ── Build full metrics dict ─────────────────────────────────────────────
    metrics = {
        "run_id": run_id,
        "case_id": case_id,
        "voxel_tag": voxel_tag,
        "n_wall_faces": int(wall_mask.sum()),
        "wss_inference_s": round(elapsed_wss, 2),
        "n_params": n_params,
        **wss_metrics,
        **{f"ba_{k}": v for k, v in ba_stats.items()},
        "icc_one_way": icc,
        **{f"cons_{k}": v for k, v in cons.items()},
    }

    # ── Save outputs ────────────────────────────────────────────────────────
    out_dir = _root / cfg.output.results_base_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "eval_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    log.info("Metrics saved: %s", metrics_path)

    np.savez_compressed(
        out_dir / "wss_comparison.npz",
        pinn_wss_pa=pinn_wss_pa,
        ref_wss_pa=ref_wss_pa,
        diff_pa=pinn_wss_pa - ref_wss_pa,
        wall_centroids_m=wall_centroids_m,
        wall_normals=wall_normals,
    )
    log.info("WSS arrays saved: %s/wss_comparison.npz", out_dir)

    # ── Generate quick-look plots ───────────────────────────────────────────
    try:
        from hemodyn_pinn.viz.plots_2d import (
            plot_loss_curves, plot_wss_scatter, plot_bland_altman,
        )

        # Loss curves (if history is available in the checkpoint)
        history_path = ckpt_dir / "loss_history.npy"
        if history_path.exists():
            history_raw = np.load(str(history_path), allow_pickle=True).item()
            # Reconstruct list-of-dicts from the saved dict-of-lists
            keys = list(history_raw.keys())
            n = len(history_raw[keys[0]])
            history = [{k: history_raw[k][i] for k in keys} for i in range(n)]
            plot_loss_curves(history, out_path=out_dir / "loss_curve.pdf")
            log.info("Loss curve saved.")

        plot_wss_scatter(
            pinn_wss_pa, ref_wss_pa,
            out_path=out_dir / "wss_scatter.pdf",
            title=f"{case_id} | {voxel_tag} | NRMSE={wss_metrics['nrmse']:.3f}",
        )
        plot_bland_altman(
            pinn_wss_pa, ref_wss_pa,
            out_path=out_dir / "bland_altman.pdf",
            title=f"{case_id} | {voxel_tag}",
        )
        log.info("Quick-look plots saved to %s/", out_dir)
    except Exception as exc:
        log.warning("Could not generate plots: %s", exc)

    log.info("Evaluation complete for run '%s'.", run_id)
    log.info(
        "Summary — NRMSE=%.4f | R²=%.4f | peak_err=%.4f | cons_err=%.4f",
        metrics["nrmse"], metrics["r2"],
        metrics["peak_wss_error"], metrics["cons_conservation_error"],
    )


if __name__ == "__main__":
    main()

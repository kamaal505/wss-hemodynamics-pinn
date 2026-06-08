"""Aggregate evaluation results across all (geometry, voxel) trained runs.

Discovers every checkpoint under data/checkpoints/ whose run_id matches the
pattern ``<case_id>_voxel_*_<suffix>`` and runs the evaluation pipeline
(05_evaluate.py logic) on each one.  Results are collected into a single
sweep_results.csv (and per-run eval_metrics.json) under data/results/.

This script is designed to be run *after* training is complete for all desired
(geometry, voxel) combinations.  It does NOT launch new training jobs.

Typical workflow:
    # 1. Train for all voxel configs on caseC and caseA:
    for mri in voxel_0p5mm voxel_1p0mm voxel_1p5mm voxel_2p0mm voxel_2p5mm; do
        python scripts/04c_train_pinn_with_bhpo.py geometry=caseC mri=$mri
        python scripts/04c_train_pinn_with_bhpo.py geometry=caseA mri=$mri
    done

    # 2. Aggregate results:
    python scripts/06_run_sweep.py

    # 3. Generate figures from aggregated results:
    python scripts/07_make_all_figures.py

Usage:
    python scripts/06_run_sweep.py
    python scripts/06_run_sweep.py sweep.run_suffix=bhpo sweep.device=cuda
    python scripts/06_run_sweep.py sweep.case_ids=[caseC,caseA,caseR]

Output:
    data/results/sweep_results.csv   — one row per (case_id, voxel_tag) combo
    data/results/<run_id>/           — per-run eval output (as in 05_evaluate.py)
"""

from __future__ import annotations

import csv
import json
import logging
import pathlib
import sys
import time

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root / "src"))

# Voxel config tags (must match configs/mri/*.yaml filenames)
_ALL_VOXEL_TAGS = [
    "voxel_0p5mm",
    "voxel_1p0mm",
    "voxel_1p5mm",
    "voxel_2p0mm",
    "voxel_2p5mm",
]

# Voxel sizes in mm for CSV / heatmap axis
_VOXEL_MM = {
    "voxel_0p5mm": 0.5,
    "voxel_1p0mm": 1.0,
    "voxel_1p5mm": 1.5,
    "voxel_2p0mm": 2.0,
    "voxel_2p5mm": 2.5,
}

_CSV_COLUMNS = [
    "run_id", "case_id", "voxel_tag", "voxel_mm",
    "nrmse", "r2", "mae",
    "peak_wss_error", "mean_relative_error",
    "pred_mean_pa", "ref_mean_pa",
    "pred_max_pa", "ref_max_pa",
    "ba_bias_pa", "ba_loa_upper_pa", "ba_loa_lower_pa", "ba_pct_within_loa",
    "icc_one_way",
    "cons_conservation_error",
    "wss_inference_s",
    "n_wall_faces",
]


def _run_evaluation(
    case_id: str,
    voxel_tag: str,
    run_suffix: str,
    device: str,
    wss_batch_size: int,
    results_dir: pathlib.Path,
    cfd_base: pathlib.Path,
    ckpt_base: pathlib.Path,
) -> dict | None:
    """Evaluate one (case_id, voxel_tag) run.  Returns metrics dict or None."""
    import torch
    from hemodyn_pinn.cfd.postprocess import load_solution
    from hemodyn_pinn.eval import (
        compute_wss_metrics,
        bland_altman_stats,
        icc_one_way,
        flow_rate_conservation_error,
    )
    from hemodyn_pinn.pinn.inference import compute_wss_auto, predict_velocity_field
    from hemodyn_pinn.pinn.networks import PINNNetwork
    from hemodyn_pinn.pinn.sampling import to_nondim_coords

    run_id = f"{case_id}_{voxel_tag}_{run_suffix}"
    ckpt_dir = ckpt_base / run_id
    ckpt_path = ckpt_dir / "best_model.pt"

    if not ckpt_path.exists():
        log.warning("Checkpoint not found, skipping: %s", ckpt_path)
        return None

    out_dir = results_dir / run_id
    metrics_path = out_dir / "eval_metrics.json"
    if metrics_path.exists():
        log.info("Cached result found, loading: %s", metrics_path)
        return json.loads(metrics_path.read_text())

    # ── Load architecture ────────────────────────────────────────────────────
    bhpo_json = ckpt_dir / "bhpo_params.json"
    if bhpo_json.exists():
        _saved = json.loads(bhpo_json.read_text())
        best_hp = _saved["best_hp"]
        _arch = _saved.get("arch_flags", {})
        net = PINNNetwork(
            n_hidden=int(best_hp["n_hidden"]),
            n_layers=int(best_hp["n_layers"]),
            use_rff=bool(best_hp["use_rff"]),
            rff_features=128,
            rff_sigma=float(best_hp.get("rff_sigma", 1.0)),
            activation=str(best_hp.get("activation", "tanh")),
            use_hard_sdf=bool(_arch.get("use_hard_sdf", False)),
            use_vec_potential=bool(_arch.get("use_vec_potential", False)),
        )
    else:
        # Default pinn_base architecture (fixed-HP runs)
        net = PINNNetwork(n_hidden=128, n_layers=4, use_rff=False, activation="tanh")

    dev = torch.device(device)
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    net.load_state_dict(ckpt["state_dict"])
    net.to(dev)
    net.eval()

    # ── Load CFD solution ────────────────────────────────────────────────────
    npz_path = cfd_base / case_id / "solution.npz"
    if not npz_path.exists():
        log.warning("CFD solution not found, skipping: %s", npz_path)
        return None
    sol = load_solution(npz_path)

    wall_mask = sol["wall_mask"].astype(bool)
    inlet_mask = sol["inlet_mask"].astype(bool)
    outlet_mask = sol["outlet_mask"].astype(bool)

    wall_centroids_m = sol["wall_centroids_m"][wall_mask]
    wall_normals = sol["wall_normals"][wall_mask]
    ref_wss_pa = sol["wss_magnitudes"][wall_mask]

    # ── Batched WSS inference ────────────────────────────────────────────────
    wall_pts_nondim = to_nondim_coords(wall_centroids_m)

    t0 = time.perf_counter()
    pinn_wss_parts = []
    for start in range(0, wall_pts_nondim.shape[0], wss_batch_size):
        end = min(start + wss_batch_size, wall_pts_nondim.shape[0])
        x_b = torch.tensor(wall_pts_nondim[start:end], dtype=torch.float32, device=dev)
        n_b = torch.tensor(wall_normals[start:end], dtype=torch.float32, device=dev)
        _, tau_mag = compute_wss_auto(net, x_b, n_b)
        pinn_wss_parts.append(tau_mag.detach().cpu().numpy())
    import numpy as np
    pinn_wss_pa = np.concatenate(pinn_wss_parts, axis=0)
    elapsed = time.perf_counter() - t0

    # ── Conservation check ───────────────────────────────────────────────────
    def _pred(pts):
        x = torch.tensor(pts, dtype=torch.float32, device=dev)
        u, p = predict_velocity_field(net, x)
        return u.cpu().numpy(), p.cpu().numpy()

    u_in, _ = _pred(to_nondim_coords(sol["wall_centroids_m"][inlet_mask]))
    u_out, _ = _pred(to_nondim_coords(sol["wall_centroids_m"][outlet_mask]))
    cons = flow_rate_conservation_error(
        u_in, sol["wall_normals"][inlet_mask],
        u_out, sol["wall_normals"][outlet_mask],
    )

    # ── Metrics ──────────────────────────────────────────────────────────────
    wss_m = compute_wss_metrics(pinn_wss_pa, ref_wss_pa)
    ba_m = bland_altman_stats(pinn_wss_pa, ref_wss_pa)
    icc = icc_one_way(pinn_wss_pa, ref_wss_pa)

    metrics = {
        "run_id": run_id,
        "case_id": case_id,
        "voxel_tag": voxel_tag,
        "voxel_mm": _VOXEL_MM.get(voxel_tag, float("nan")),
        "n_wall_faces": int(wall_mask.sum()),
        "wss_inference_s": round(elapsed, 2),
        "n_params": sum(p.numel() for p in net.parameters()),
        **wss_m,
        **{f"ba_{k}": v for k, v in ba_m.items()},
        "icc_one_way": icc,
        **{f"cons_{k}": v for k, v in cons.items()},
    }

    # ── Save per-run artefacts ────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(
        out_dir / "wss_comparison.npz",
        pinn_wss_pa=pinn_wss_pa,
        ref_wss_pa=ref_wss_pa,
        diff_pa=pinn_wss_pa - ref_wss_pa,
        wall_centroids_m=wall_centroids_m,
        wall_normals=wall_normals,
    )
    log.info(
        "%s | %s → NRMSE=%.4f R²=%.4f peak_err=%.4f",
        case_id, voxel_tag,
        metrics["nrmse"], metrics["r2"], metrics["peak_wss_error"],
    )
    return metrics


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    # ── Discover which case IDs have CFD solutions available ────────────────
    cfd_base = _root / cfg.output.base_dir
    ckpt_base = _root / cfg.output.pinn_base_dir
    results_dir = _root / cfg.output.results_base_dir

    # Use the geometry from the Hydra config as a default, but sweep all
    # case_ids that have both a CFD solution and at least one checkpoint.
    # The user can extend this list by passing sweep.case_ids=[caseC,caseA]
    # via the CLI (with +sweep.case_ids=...).
    default_case_ids = [cfg.geometry.case_id]
    # Check for any other case_id directories with solution.npz
    if cfd_base.exists():
        for subdir in sorted(cfd_base.iterdir()):
            if (subdir / "solution.npz").exists():
                cid = subdir.name
                if cid not in default_case_ids:
                    default_case_ids.append(cid)

    run_suffix = cfg.eval.run_suffix
    device = cfg.eval.device
    wss_batch_size = int(cfg.eval.wss_batch_size)

    log.info(
        "Sweep: case_ids=%s | voxel_tags=%s | suffix=%s",
        default_case_ids, _ALL_VOXEL_TAGS, run_suffix,
    )

    # ── Run evaluation for all (case_id, voxel_tag) combos ─────────────────
    all_metrics: list[dict] = []

    for case_id in default_case_ids:
        for voxel_tag in _ALL_VOXEL_TAGS:
            try:
                result = _run_evaluation(
                    case_id=case_id,
                    voxel_tag=voxel_tag,
                    run_suffix=run_suffix,
                    device=device,
                    wss_batch_size=wss_batch_size,
                    results_dir=results_dir,
                    cfd_base=cfd_base,
                    ckpt_base=ckpt_base,
                )
                if result is not None:
                    all_metrics.append(result)
            except Exception as exc:
                log.warning(
                    "Error evaluating %s / %s: %s",
                    case_id, voxel_tag, exc, exc_info=True,
                )

    if not all_metrics:
        log.warning("No results found. Check that checkpoints exist under %s/", ckpt_base)
        return

    # ── Write sweep_results.csv ─────────────────────────────────────────────
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "sweep_results.csv"
    columns = [c for c in _CSV_COLUMNS if c in all_metrics[0]]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_metrics)

    log.info(
        "Sweep complete: %d runs evaluated.  Results: %s",
        len(all_metrics), csv_path,
    )

    # ── Print compact summary table ─────────────────────────────────────────
    header = f"{'run_id':<40} {'NRMSE':>7} {'R²':>7} {'peak_err':>9}"
    log.info(header)
    log.info("-" * len(header))
    for m in sorted(all_metrics, key=lambda x: x.get("nrmse", 99)):
        log.info(
            "%-40s %7.4f %7.4f %9.4f",
            m["run_id"], m.get("nrmse", float("nan")),
            m.get("r2", float("nan")), m.get("peak_wss_error", float("nan")),
        )


if __name__ == "__main__":
    main()

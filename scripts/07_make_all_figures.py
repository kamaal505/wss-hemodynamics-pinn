"""Generate all paper figures from saved evaluation results.

Reads from data/results/ (produced by 05_evaluate.py and 06_run_sweep.py)
and writes figures to paper/figures/.

Each figure matches the spec in docs/project_knowledge/04_research_artefacts_and_figures.tex.
Functions are independent so individual figures can be regenerated without
rerunning the full script.  Run with --fig=3 etc. to produce a single figure:

    python scripts/07_make_all_figures.py          # all available figures
    python scripts/07_make_all_figures.py +fig=8   # only Fig 8

Output format: PDF (vector) or 600 dpi PNG (CLAUDE.md §8).

Figure index:
    Fig 3  — CFD ground-truth WSS map + velocity streamlines (matplotlib)
    Fig 5  — PINN vs CFD WSS side-by-side surface maps (PyVista, optional)
    Fig 6  — Sensitivity heatmap (voxel size × VNR) from sweep_results.csv
    Fig 8  — Bland–Altman and WSS scatter
    Fig 9  — Conservation diagnostic (Q proxy along z)
    Fig B  — BHPO convergence trace (from trial_log.csv)
    Fig L  — Training loss curves per run
"""

from __future__ import annotations

import csv
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

_FIGURES_DIR = _root / "paper" / "figures"


# ── Figure 3: CFD ground-truth WSS distribution ───────────────────────────────

def fig3_cfd_ground_truth(sol: dict, case_id: str, out_dir: pathlib.Path) -> None:
    """WSS histogram + surface summary from the CFD solution (matplotlib only)."""
    from hemodyn_pinn.viz.plots_2d import apply_paper_style
    import matplotlib.pyplot as plt

    apply_paper_style()
    wall_mask = sol["wall_mask"].astype(bool)
    wss = sol["wss_magnitudes"][wall_mask]
    pcts = np.percentile(wss, [5, 25, 50, 75, 95])

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))

    # (a) WSS histogram
    ax = axes[0]
    ax.hist(wss * 1e3, bins=60, color="steelblue", edgecolor="none", density=True)
    for pct, label in zip(pcts[[1, 2, 3]], ["P25", "P50", "P75"]):
        ax.axvline(pct * 1e3, color="black", lw=0.8, ls="--")
        ax.text(pct * 1e3, ax.get_ylim()[1] * 0.9, label, ha="center", fontsize=7)
    ax.set_xlabel("WSS [mPa]")
    ax.set_ylabel("Density")
    ax.set_title(f"{case_id} — CFD WSS distribution")

    # (b) Velocity speed histogram
    speed = np.linalg.norm(sol["velocity"], axis=1)
    ax = axes[1]
    ax.hist(speed * 1e3, bins=60, color="steelblue", edgecolor="none", density=True)
    ax.set_xlabel("Speed [mm/s]")
    ax.set_ylabel("Density")
    ax.set_title(f"{case_id} — nodal velocity distribution")

    fig.tight_layout()
    out_path = out_dir / f"fig3_cfd_truth_{case_id}.pdf"
    fig.savefig(str(out_path))
    log.info("Fig 3 saved: %s", out_path)


# ── Figure 5: PINN vs CFD surface comparison (PyVista) ───────────────────────

def fig5_wss_comparison(
    npz_path: pathlib.Path,
    run_id: str,
    out_dir: pathlib.Path,
) -> None:
    """Side-by-side WSS surface render: CFD vs PINN."""
    try:
        from hemodyn_pinn.viz.render_3d import render_wss_comparison, render_wss_error
    except ImportError as exc:
        log.warning("Fig 5 skipped (pyvista not available): %s", exc)
        return

    data = np.load(str(npz_path))
    wall_pts = data["wall_centroids_m"]
    ref_wss = data["ref_wss_pa"]
    pinn_wss = data["pinn_wss_pa"]

    render_wss_comparison(
        wall_pts, ref_wss, pinn_wss,
        out_path=out_dir / f"fig5_wss_comparison_{run_id}.png",
    )
    render_wss_error(
        wall_pts, ref_wss, pinn_wss,
        out_path=out_dir / f"fig5_wss_error_{run_id}.png",
    )
    log.info("Fig 5 saved for run %s", run_id)


# ── Figure 6: Sensitivity heatmap ─────────────────────────────────────────────

def fig6_sensitivity_heatmap(
    sweep_csv: pathlib.Path,
    out_dir: pathlib.Path,
) -> None:
    """NRMSE heatmap over (voxel size × VNR) — Fig 6 from doc 04."""
    if not sweep_csv.exists():
        log.warning("Fig 6 skipped: sweep_results.csv not found (%s)", sweep_csv)
        return

    from hemodyn_pinn.viz.plots_2d import plot_sensitivity_heatmap

    # Load sweep results
    rows: list[dict] = []
    with open(sweep_csv, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        log.warning("Fig 6 skipped: sweep_results.csv is empty")
        return

    # Group by case_id
    case_ids = sorted(set(r["case_id"] for r in rows))
    voxel_mms = sorted(set(float(r["voxel_mm"]) for r in rows if r["voxel_mm"]))

    for case_id in case_ids:
        case_rows = [r for r in rows if r["case_id"] == case_id]
        if not case_rows:
            continue

        # Build NRMSE matrix: rows = voxel_size, cols = VNR (placeholder = 1
        # since VNR is not swept in the current pipeline — will be extended)
        # For now produce a 1-column heatmap per case.
        nrmse_vals = []
        vx_vals = []
        for r in sorted(case_rows, key=lambda x: float(x.get("voxel_mm", 99))):
            try:
                vx_vals.append(float(r["voxel_mm"]))
                nrmse_vals.append(float(r["nrmse"]))
            except (ValueError, KeyError):
                continue

        if not nrmse_vals:
            continue

        # Single-VNR heatmap (column vector)
        nrmse_mat = np.array(nrmse_vals).reshape(-1, 1)

        plot_sensitivity_heatmap(
            voxel_sizes_mm=vx_vals,
            vnr_values=[30],     # placeholder until VNR sweep is run
            nrmse_matrix=nrmse_mat,
            out_path=out_dir / f"fig6_heatmap_{case_id}.pdf",
            title=f"{case_id} — WSS NRMSE vs voxel size",
        )
        log.info("Fig 6 saved for %s (%d voxel configs)", case_id, len(vx_vals))


# ── Figure 8: Bland–Altman and scatter ────────────────────────────────────────

def fig8_bland_altman(
    npz_path: pathlib.Path,
    run_id: str,
    out_dir: pathlib.Path,
    metrics: dict,
) -> None:
    """Scatter + Bland–Altman for WSS agreement — Fig 8 from doc 04."""
    from hemodyn_pinn.viz.plots_2d import plot_wss_scatter, plot_bland_altman

    data = np.load(str(npz_path))
    pred = data["pinn_wss_pa"]
    ref = data["ref_wss_pa"]

    title = (
        f"{metrics.get('case_id', run_id)} | {metrics.get('voxel_tag', '')} | "
        f"NRMSE={metrics.get('nrmse', 0):.3f}"
    )

    plot_wss_scatter(
        pred, ref,
        out_path=out_dir / f"fig8_scatter_{run_id}.pdf",
        title=title,
    )
    plot_bland_altman(
        pred, ref,
        out_path=out_dir / f"fig8_ba_{run_id}.pdf",
        title=title,
    )
    log.info("Fig 8 saved for run %s", run_id)


# ── Figure 9: Conservation diagnostic ────────────────────────────────────────

def fig9_conservation(
    run_metrics: dict,
    out_dir: pathlib.Path,
) -> None:
    """Conservation error bar chart across all evaluated runs — Fig 9."""
    from hemodyn_pinn.viz.plots_2d import apply_paper_style
    import matplotlib.pyplot as plt

    if not run_metrics:
        return

    apply_paper_style()
    labels = [m["run_id"].replace("_bhpo", "") for m in run_metrics]
    errors = [m.get("cons_conservation_error", float("nan")) for m in run_metrics]

    fig, ax = plt.subplots(figsize=(max(5.0, len(labels) * 0.7), 3.0))
    bars = ax.bar(range(len(labels)), errors, color="steelblue", edgecolor="none")
    ax.axhline(0.05, color="firebrick", lw=0.8, ls="--", label="5% threshold")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Conservation error |Q_in − Q_out| / Q_in")
    ax.set_title("Mass conservation error across runs")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out_path = out_dir / "fig9_conservation.pdf"
    fig.savefig(str(out_path))
    log.info("Fig 9 saved: %s", out_path)


# ── Figure B: BHPO convergence ─────────────────────────────────────────────────

def figB_bhpo_convergence(
    bhpo_dir: pathlib.Path,
    out_dir: pathlib.Path,
) -> None:
    """BHPO GP-minimum convergence trace from trial_log.csv."""
    trial_log = bhpo_dir / "trial_log.csv"
    if not trial_log.exists():
        log.info("Fig B skipped: %s not found", trial_log)
        return

    from hemodyn_pinn.viz.plots_2d import plot_bhpo_convergence

    objectives = []
    with open(trial_log, newline="") as f:
        for row in csv.DictReader(f):
            try:
                objectives.append(float(row["wss_nrmse"]))
            except (KeyError, ValueError):
                pass

    if not objectives:
        return

    plot_bhpo_convergence(
        objectives,
        out_path=out_dir / "figB_bhpo_convergence.pdf",
    )
    log.info("Fig B saved: BHPO convergence (%d trials)", len(objectives))


# ── Figure L: Loss curves ──────────────────────────────────────────────────────

def figL_loss_curves(
    ckpt_dir: pathlib.Path,
    run_id: str,
    out_dir: pathlib.Path,
) -> None:
    """Training loss curves from loss_history.npy."""
    from hemodyn_pinn.viz.plots_2d import plot_loss_curves

    history_path = ckpt_dir / "loss_history.npy"
    if not history_path.exists():
        log.info("Fig L skipped for %s: no loss_history.npy", run_id)
        return

    history_raw = np.load(str(history_path), allow_pickle=True).item()
    keys = list(history_raw.keys())
    n = len(history_raw[keys[0]])
    history = [{k: history_raw[k][i] for k in keys} for i in range(n)]

    plot_loss_curves(
        history,
        out_path=out_dir / f"figL_loss_{run_id}.pdf",
    )
    log.info("Fig L saved for run %s", run_id)


# ── Main ──────────────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    results_base = _root / cfg.output.results_base_dir
    ckpt_base = _root / cfg.output.pinn_base_dir
    cfd_base = _root / cfg.output.base_dir
    bhpo_base = _root / "data" / "bhpo_runs"

    out_dir = _FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Discover all per-run result directories ──────────────────────────────
    run_dirs = sorted(results_base.glob("*/eval_metrics.json")) if results_base.exists() else []

    if not run_dirs:
        log.warning("No evaluation results found under %s/", results_base)
        log.warning("Run scripts/05_evaluate.py and/or 06_run_sweep.py first.")

    all_run_metrics: list[dict] = []
    for metrics_path in run_dirs:
        try:
            metrics = json.loads(metrics_path.read_text())
            all_run_metrics.append(metrics)
        except Exception as exc:
            log.warning("Could not read %s: %s", metrics_path, exc)

    log.info("Found %d evaluated runs.", len(all_run_metrics))

    # ── Fig 3: CFD ground truth (one per case_id found) ─────────────────────
    case_ids_done: set[str] = set()
    for metrics in all_run_metrics:
        case_id = metrics.get("case_id", "")
        if case_id and case_id not in case_ids_done:
            npz = cfd_base / case_id / "solution.npz"
            if npz.exists():
                try:
                    from hemodyn_pinn.cfd.postprocess import load_solution
                    sol = load_solution(npz)
                    fig3_cfd_ground_truth(sol, case_id, out_dir)
                    case_ids_done.add(case_id)
                except Exception as exc:
                    log.warning("Fig 3 failed for %s: %s", case_id, exc)

    # ── Fig 5: WSS comparison maps (one per run) ─────────────────────────────
    for metrics in all_run_metrics:
        run_id = metrics.get("run_id", "")
        npz_path = results_base / run_id / "wss_comparison.npz"
        if npz_path.exists():
            try:
                fig5_wss_comparison(npz_path, run_id, out_dir)
            except Exception as exc:
                log.warning("Fig 5 failed for %s: %s", run_id, exc)

    # ── Fig 6: Sensitivity heatmap ───────────────────────────────────────────
    try:
        fig6_sensitivity_heatmap(results_base / "sweep_results.csv", out_dir)
    except Exception as exc:
        log.warning("Fig 6 failed: %s", exc)

    # ── Fig 8: Bland–Altman (one per run) ────────────────────────────────────
    for metrics in all_run_metrics:
        run_id = metrics.get("run_id", "")
        npz_path = results_base / run_id / "wss_comparison.npz"
        if npz_path.exists():
            try:
                fig8_bland_altman(npz_path, run_id, out_dir, metrics)
            except Exception as exc:
                log.warning("Fig 8 failed for %s: %s", run_id, exc)

    # ── Fig 9: Conservation ──────────────────────────────────────────────────
    try:
        fig9_conservation(all_run_metrics, out_dir)
    except Exception as exc:
        log.warning("Fig 9 failed: %s", exc)

    # ── Fig B: BHPO convergence (one per bhpo_runs dir) ──────────────────────
    if bhpo_base.exists():
        for bhpo_case_dir in sorted(bhpo_base.iterdir()):
            try:
                figB_bhpo_convergence(bhpo_case_dir, out_dir)
            except Exception as exc:
                log.warning("Fig B failed for %s: %s", bhpo_case_dir.name, exc)

    # ── Fig L: Loss curves (one per run that has loss_history.npy) ───────────
    for metrics in all_run_metrics:
        run_id = metrics.get("run_id", "")
        ckpt_dir = ckpt_base / run_id
        if (ckpt_dir / "loss_history.npy").exists():
            try:
                figL_loss_curves(ckpt_dir, run_id, out_dir)
            except Exception as exc:
                log.warning("Fig L failed for %s: %s", run_id, exc)

    # ── Summary ──────────────────────────────────────────────────────────────
    produced = sorted(out_dir.glob("fig*.pdf")) + sorted(out_dir.glob("fig*.png"))
    log.info(
        "Done.  %d figure files written to %s/",
        len(produced), out_dir,
    )
    for p in produced:
        log.info("  %s", p.name)


if __name__ == "__main__":
    main()

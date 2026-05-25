"""2D paper-quality plots for the hemodyn-pinn project.

All functions return a matplotlib Figure and optionally save it.
Call apply_paper_style() once at the start of a script before calling these.

Colormaps: viridis (sequential), RdBu_r (diverging). NEVER jet.
Format: vector PDF or 600 dpi PNG (CLAUDE.md §8).
"""

from __future__ import annotations

import pathlib

import numpy as np
import matplotlib.pyplot as plt

from hemodyn_pinn.viz.style import apply_paper_style


# ── Loss curves ───────────────────────────────────────────────────────────────


def plot_loss_curves(
    history: list[dict],
    out_path: pathlib.Path | str | None = None,
) -> plt.Figure:
    """Plot training loss curves (total + components) on a log scale.

    Parameters
    ----------
    history:
        List of per-epoch dicts with keys: epoch, loss_total, loss_data,
        loss_phys, loss_bc.  Produced by PINNTrainer.
    out_path:
        Optional path to save the figure (PDF or PNG).
    """
    apply_paper_style()
    epochs = [d["epoch"] for d in history]
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    ax.semilogy(epochs, [d["loss_total"] for d in history],
                color="black", lw=1.5, label="total")
    ax.semilogy(epochs, [d["loss_data"] for d in history],
                lw=1.0, ls="--", label="data")
    ax.semilogy(epochs, [d["loss_phys"] for d in history],
                lw=1.0, ls="-.", label="physics")
    ax.semilogy(epochs, [d["loss_bc"] for d in history],
                lw=1.0, ls=":", label="BC")
    # Shade Adam vs L-BFGS phase
    if "phase" in history[0]:
        n_adam = sum(1 for d in history if d.get("phase") == "adam")
        if n_adam < len(history):
            ax.axvline(epochs[n_adam - 1], color="grey", lw=0.7, ls="--",
                       alpha=0.6, label="Adam → L-BFGS")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    if out_path is not None:
        _save(fig, out_path)
    return fig


# ── WSS scatter ───────────────────────────────────────────────────────────────


def plot_wss_scatter(
    wss_pred_pa: np.ndarray,
    wss_ref_pa: np.ndarray,
    out_path: pathlib.Path | str | None = None,
    title: str = "",
) -> plt.Figure:
    """Scatter plot: PINN WSS vs CFD WSS with y = x reference line.

    Parameters
    ----------
    wss_pred_pa, wss_ref_pa:
        (W,) WSS magnitudes [Pa] for PINN and CFD respectively.
    """
    apply_paper_style()
    # Convert to mPa for legibility
    pred_mpa = wss_pred_pa * 1e3
    ref_mpa = wss_ref_pa * 1e3

    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    ax.scatter(ref_mpa, pred_mpa, s=1, alpha=0.3, color="steelblue",
               rasterized=True, linewidths=0)
    lim = max(float(ref_mpa.max()), float(pred_mpa.max())) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="y = x")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("CFD WSS [mPa]")
    ax.set_ylabel("PINN WSS [mPa]")
    if title:
        ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    if out_path is not None:
        _save(fig, out_path)
    return fig


# ── Bland–Altman ──────────────────────────────────────────────────────────────


def plot_bland_altman(
    wss_pred_pa: np.ndarray,
    wss_ref_pa: np.ndarray,
    out_path: pathlib.Path | str | None = None,
    title: str = "",
) -> plt.Figure:
    """Bland–Altman plot: (PINN − CFD) vs ½(PINN + CFD) in mPa.

    Bias line and ±1.96 SD limits of agreement are annotated.
    """
    apply_paper_style()
    means_mpa = (wss_pred_pa + wss_ref_pa) / 2.0 * 1e3
    diffs_mpa = (wss_pred_pa - wss_ref_pa) * 1e3
    bias = float(np.mean(diffs_mpa))
    std_d = float(np.std(diffs_mpa, ddof=1))
    loa_u = bias + 1.96 * std_d
    loa_l = bias - 1.96 * std_d

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.scatter(means_mpa, diffs_mpa, s=1, alpha=0.2, color="steelblue",
               rasterized=True, linewidths=0)
    ax.axhline(bias, color="black", lw=1.2,
               label=f"Bias = {bias:.3f} mPa")
    ax.axhline(loa_u, color="firebrick", lw=0.8, ls="--",
               label=f"+1.96 SD = {loa_u:.3f}")
    ax.axhline(loa_l, color="firebrick", lw=0.8, ls="--",
               label=f"−1.96 SD = {loa_l:.3f}")
    ax.axhline(0.0, color="grey", lw=0.5, ls=":")
    ax.set_xlabel("Mean WSS [mPa]")
    ax.set_ylabel("PINN − CFD [mPa]")
    if title:
        ax.set_title(title)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    if out_path is not None:
        _save(fig, out_path)
    return fig


# ── Sensitivity heatmap ───────────────────────────────────────────────────────


def plot_sensitivity_heatmap(
    voxel_sizes_mm: list[float],
    vnr_values: list[float],
    nrmse_matrix: np.ndarray,
    out_path: pathlib.Path | str | None = None,
    contour_level: float = 0.20,
    title: str = "",
) -> plt.Figure:
    """Figure 6 from doc 04: NRMSE heatmap over (voxel size, VNR) grid.

    Parameters
    ----------
    voxel_sizes_mm:
        List of voxel sizes [mm] (y-axis, ascending).
    vnr_values:
        List of VNR levels (x-axis, ascending — high SNR = right).
    nrmse_matrix:
        (n_voxel, n_vnr) array of WSS NRMSE values.
    contour_level:
        NRMSE iso-contour delineating the reliable operating region.
    """
    apply_paper_style()
    V = np.array(voxel_sizes_mm)
    S = np.array(vnr_values)

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    cf = ax.contourf(S, V, nrmse_matrix, levels=20, cmap="viridis")
    cb = plt.colorbar(cf, ax=ax)
    cb.set_label("WSS NRMSE", fontsize=9)

    # Iso-contour at the "reliable region" boundary
    try:
        cs = ax.contour(S, V, nrmse_matrix, levels=[contour_level],
                        colors=["white"], linewidths=[1.5])
        ax.clabel(cs, fmt=f"{contour_level:.0%}", fontsize=8, colors="white")
    except Exception:
        pass   # skip contour if data range doesn't include the level

    # Annotate cells
    for i, v in enumerate(V):
        for j, s in enumerate(S):
            val = nrmse_matrix[i, j]
            ax.text(s, v, f"{val:.2f}", ha="center", va="center",
                    fontsize=6, color="white")

    ax.set_xlabel("VNR")
    ax.set_ylabel("Voxel size [mm]")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    if out_path is not None:
        _save(fig, out_path)
    return fig


# ── Conservation Q(z) profile ─────────────────────────────────────────────────


def plot_conservation_profile(
    z_cfd: np.ndarray,
    q_cfd: np.ndarray,
    z_pinn: np.ndarray,
    q_pinn: np.ndarray,
    out_path: pathlib.Path | str | None = None,
    title: str = "",
) -> plt.Figure:
    """Plot Q(z) (mean u_z) along z-axis for CFD and PINN (Figure 9)."""
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.plot(z_cfd, q_cfd, color="black", lw=1.2, label="CFD")
    ax.plot(z_pinn, q_pinn, color="steelblue", lw=1.2, ls="--", label="PINN")
    ax.set_xlabel("z (non-dim)")
    ax.set_ylabel("Mean u_z (proxy for Q)")
    if title:
        ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    if out_path is not None:
        _save(fig, out_path)
    return fig


# ── BHPO convergence ──────────────────────────────────────────────────────────


def plot_bhpo_convergence(
    objective_trace: list[float],
    out_path: pathlib.Path | str | None = None,
) -> plt.Figure:
    """Plot GP-minimum convergence trace from the BHPO search."""
    apply_paper_style()
    mins = np.minimum.accumulate(objective_trace)
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.plot(range(1, len(objective_trace) + 1), objective_trace,
            color="steelblue", lw=0.8, alpha=0.6, label="trial NRMSE")
    ax.plot(range(1, len(mins) + 1), mins,
            color="black", lw=1.2, label="running minimum")
    ax.set_xlabel("Trial")
    ax.set_ylabel("WSS NRMSE (val)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    if out_path is not None:
        _save(fig, out_path)
    return fig


# ── Private helper ────────────────────────────────────────────────────────────


def _save(fig: plt.Figure, out_path: pathlib.Path | str) -> None:
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path))

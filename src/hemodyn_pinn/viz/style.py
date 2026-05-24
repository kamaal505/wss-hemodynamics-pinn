"""Matplotlib rcParams and PyVista theme for paper-quality figures.

Usage:
    from hemodyn_pinn.viz.style import apply_paper_style, apply_pyvista_theme
    apply_paper_style()          # call once at script start
    apply_pyvista_theme()        # call before any PyVista render

Colormaps (CLAUDE.md §8):
    Sequential:  viridis, magma, cividis
    Diverging:   RdBu_r  (signed quantities only)
    NEVER:       jet
"""

from __future__ import annotations

import matplotlib as mpl

# ── Matplotlib rcParams ───────────────────────────────────────────────────────

PAPER_RC: dict = {
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "lines.linewidth": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,   # embed TrueType fonts for PDF
    "ps.fonttype": 42,
    "image.cmap": "viridis",
}


def apply_paper_style() -> None:
    """Apply project-wide matplotlib style (call once per script)."""
    mpl.rcParams.update(PAPER_RC)


def reset_style() -> None:
    """Reset matplotlib to its defaults."""
    mpl.rcdefaults()


# ── PyVista theme ─────────────────────────────────────────────────────────────

_PYVISTA_AVAILABLE = False

try:
    import pyvista as pv   # noqa: F401
    _PYVISTA_AVAILABLE = True
except ImportError:
    pass


def apply_pyvista_theme() -> None:
    """Apply the project PyVista theme (white background, black font, viridis).

    Raises ImportError if pyvista is not installed.
    """
    if not _PYVISTA_AVAILABLE:
        raise ImportError(
            "pyvista is required for 3D renders. "
            "Install with: pip install pyvista"
        )
    import pyvista as pv

    theme = pv.themes.DocumentTheme()
    theme.background = "white"
    theme.font.color = "black"
    theme.font.size = 10
    theme.cmap = "viridis"
    pv.global_theme.load_theme(theme)

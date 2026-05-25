"""3D surface WSS renders using PyVista.

All public functions require pyvista. They raise ImportError if absent so
callers can catch it and fall back to a 2D summary plot instead.

Colormaps: viridis (default) — never jet (CLAUDE.md §8).
"""

from __future__ import annotations

import pathlib

import numpy as np


def render_wss_surface(
    wall_points_m: np.ndarray,
    wss_pa: np.ndarray,
    out_path: pathlib.Path | str | None = None,
    cmap: str = "viridis",
    clim: tuple[float, float] | None = None,
    title: str = "WSS [Pa]",
    off_screen: bool = True,
) -> object:
    """Render WSS magnitude as a point cloud on the vessel wall.

    Parameters
    ----------
    wall_points_m:
        (W, 3) wall centroid coordinates [m].
    wss_pa:
        (W,) WSS magnitudes [Pa].
    out_path:
        If given, save screenshot (PNG) to this path.
    cmap:
        Colormap name (never "jet").
    clim:
        Color scale limits [Pa].  Auto-computed from data if None.
    title:
        Scalar bar title.
    off_screen:
        Render off-screen (required for batch / headless runs).

    Returns
    -------
    pv.Plotter
        The plotter; caller can call .show() for interactive use.
    """
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError("pyvista is required for 3D WSS renders.") from exc

    from hemodyn_pinn.viz.style import apply_pyvista_theme
    apply_pyvista_theme()

    cloud = pv.PolyData(wall_points_m.astype(np.float64))
    cloud["WSS [Pa]"] = wss_pa.astype(np.float64)

    pl = pv.Plotter(off_screen=off_screen)
    pl.add_mesh(
        cloud,
        scalars="WSS [Pa]",
        cmap=cmap,
        clim=clim,
        point_size=4,
        render_points_as_spheres=True,
        scalar_bar_args={"title": title, "color": "black"},
    )
    pl.background_color = "white"
    pl.camera_position = "xy"

    if out_path is not None:
        out_path = pathlib.Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pl.screenshot(str(out_path), return_img=False)

    return pl


def render_wss_comparison(
    wall_points_m: np.ndarray,
    wss_ref_pa: np.ndarray,
    wss_pred_pa: np.ndarray,
    out_path: pathlib.Path | str | None = None,
    off_screen: bool = True,
) -> object:
    """Side-by-side WSS render: CFD ground truth (left) vs PINN (right).

    Parameters
    ----------
    wall_points_m:
        (W, 3) wall centroid coordinates [m].
    wss_ref_pa:
        (W,) CFD WSS magnitudes [Pa].
    wss_pred_pa:
        (W,) PINN WSS magnitudes [Pa].
    out_path:
        If given, save screenshot to this path.
    off_screen:
        Render off-screen.

    Returns
    -------
    pv.Plotter
        The plotter object.
    """
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError("pyvista is required for 3D WSS renders.") from exc

    from hemodyn_pinn.viz.style import apply_pyvista_theme
    apply_pyvista_theme()

    clim = (0.0, float(wss_ref_pa.max()))
    pts = wall_points_m.astype(np.float64)

    pl = pv.Plotter(shape=(1, 2), off_screen=off_screen)

    for col, (label, wss) in enumerate([
        ("CFD ground truth", wss_ref_pa),
        ("PINN prediction", wss_pred_pa),
    ]):
        pl.subplot(0, col)
        cloud = pv.PolyData(pts)
        cloud["WSS [Pa]"] = wss.astype(np.float64)
        pl.add_mesh(
            cloud,
            scalars="WSS [Pa]",
            cmap="viridis",
            clim=clim,
            point_size=4,
            render_points_as_spheres=True,
            scalar_bar_args={"title": label, "color": "black"},
        )
        pl.background_color = "white"
        pl.add_title(label, font_size=10, color="black")
        pl.camera_position = "xy"

    if out_path is not None:
        out_path = pathlib.Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pl.screenshot(str(out_path), return_img=False)

    return pl


def render_wss_error(
    wall_points_m: np.ndarray,
    wss_ref_pa: np.ndarray,
    wss_pred_pa: np.ndarray,
    out_path: pathlib.Path | str | None = None,
    off_screen: bool = True,
) -> object:
    """Render pointwise WSS error (pred − ref) using RdBu_r diverging colormap."""
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError("pyvista is required for 3D WSS renders.") from exc

    from hemodyn_pinn.viz.style import apply_pyvista_theme
    apply_pyvista_theme()

    diff = wss_pred_pa - wss_ref_pa
    abs_max = float(np.abs(diff).max())

    cloud = pv.PolyData(wall_points_m.astype(np.float64))
    cloud["Error [Pa]"] = diff.astype(np.float64)

    pl = pv.Plotter(off_screen=off_screen)
    pl.add_mesh(
        cloud,
        scalars="Error [Pa]",
        cmap="RdBu_r",
        clim=(-abs_max, abs_max),
        point_size=4,
        render_points_as_spheres=True,
        scalar_bar_args={"title": "PINN − CFD [Pa]", "color": "black"},
    )
    pl.background_color = "white"
    pl.camera_position = "xy"

    if out_path is not None:
        out_path = pathlib.Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pl.screenshot(str(out_path), return_img=False)

    return pl

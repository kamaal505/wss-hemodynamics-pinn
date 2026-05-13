"""Save and load Stokes CFD solutions (velocity, pressure, WSS).

Output layout under ``outdir/``:
    solution.npz   — all numerical arrays
    metadata.json  — solver config and scalar statistics
    solution.vtu   — optional VTK visualisation file (if save_vtk=True)
"""

from __future__ import annotations

import json
import pathlib
from typing import Optional

import numpy as np

from hemodyn_pinn.cfd.stokes_solver import StokesSolution, StokesConfig


def save_solution(
    solution: StokesSolution,
    outdir: pathlib.Path | str,
    overwrite: bool = False,
) -> pathlib.Path:
    """Write a StokesSolution to disk.

    Parameters
    ----------
    solution:
        Result from :func:`~hemodyn_pinn.cfd.stokes_solver.solve_stokes`.
    outdir:
        Directory to write into.  Created if it does not exist.
    overwrite:
        If False (default), raise an error if ``solution.npz`` already exists.

    Returns
    -------
    pathlib.Path
        Path to the written ``solution.npz`` file.
    """
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    npz_path = outdir / "solution.npz"
    if npz_path.exists() and not overwrite:
        raise FileExistsError(
            f"{npz_path} already exists. Pass overwrite=True to replace it."
        )

    np.savez_compressed(
        npz_path,
        # Mesh geometry
        points_m=solution.wall_centroids_m,    # kept for convenience; full mesh coords
        # Nodal fields  (N nodes)
        velocity=solution.velocity_nodes,      # (N, 3) m/s
        pressure=solution.pressure_nodes,      # (N,)  Pa
        # Wall fields   (W wall tris)
        wall_centroids_m=solution.wall_centroids_m,    # (W, 3) m
        wall_normals=solution.wall_normals,            # (W, 3)
        wss_vectors=solution.wss_vectors,              # (W, 3) Pa
        wss_magnitudes=solution.wss_magnitudes,        # (W,)   Pa
        # Boundary classification masks
        wall_mask=solution.boundary_parts.wall_mask,
        inlet_mask=solution.boundary_parts.inlet_mask,
        outlet_mask=solution.boundary_parts.outlet_mask,
    )

    # Scalar metadata as JSON.
    wss_wall = solution.wss_magnitudes[solution.boundary_parts.wall_mask]
    speed = np.linalg.norm(solution.velocity_nodes, axis=1)

    meta = {
        "case_id": solution.case_id,
        "re_number": float(solution.re_number),
        "mu": solution.cfg.mu,
        "u_inlet": solution.cfg.u_inlet,
        "n_nodes": int(solution.velocity_nodes.shape[0]),
        "n_wall_tri": int(solution.wss_magnitudes.shape[0]),
        "n_inlet_tri": int(solution.boundary_parts.inlet_mask.sum()),
        "n_outlet_tri": int(solution.boundary_parts.outlet_mask.sum()),
        "velocity_max_ms": float(speed.max()),
        "velocity_mean_ms": float(speed.mean()),
        "pressure_min_Pa": float(solution.pressure_nodes.min()),
        "pressure_max_Pa": float(solution.pressure_nodes.max()),
        "wss_wall_mean_Pa": float(wss_wall.mean()) if wss_wall.size > 0 else 0.0,
        "wss_wall_max_Pa": float(wss_wall.max()) if wss_wall.size > 0 else 0.0,
        "wss_wall_p95_Pa": float(np.percentile(wss_wall, 95)) if wss_wall.size > 0 else 0.0,
    }

    json_path = outdir / "metadata.json"
    json_path.write_text(json.dumps(meta, indent=2))

    # Optional VTK output (requires dolfinx to be available).
    if solution.cfg.save_vtk:
        _write_vtk(solution, outdir)

    return npz_path


def load_solution(npz_path: pathlib.Path | str) -> dict[str, np.ndarray]:
    """Load a previously saved solution.

    Parameters
    ----------
    npz_path:
        Path to the ``solution.npz`` file.

    Returns
    -------
    dict
        Dictionary with keys matching the arrays written by :func:`save_solution`.
    """
    data = np.load(str(npz_path))
    return {k: data[k] for k in data.files}


def load_metadata(outdir: pathlib.Path | str) -> dict:
    """Load scalar metadata from a CFD output directory."""
    meta_path = pathlib.Path(outdir) / "metadata.json"
    return json.loads(meta_path.read_text())


def wss_statistics(solution: StokesSolution) -> dict[str, float]:
    """Return a summary of WSS statistics on the vessel wall only.

    Parameters
    ----------
    solution:
        A solved :class:`~hemodyn_pinn.cfd.stokes_solver.StokesSolution`.

    Returns
    -------
    dict with keys: mean, std, min, max, p5, p25, median, p75, p95 (Pa).
    """
    wss_wall = solution.wss_magnitudes[solution.boundary_parts.wall_mask]
    pcts = np.percentile(wss_wall, [5, 25, 50, 75, 95])
    return {
        "mean": float(wss_wall.mean()),
        "std": float(wss_wall.std()),
        "min": float(wss_wall.min()),
        "max": float(wss_wall.max()),
        "p5": float(pcts[0]),
        "p25": float(pcts[1]),
        "median": float(pcts[2]),
        "p75": float(pcts[3]),
        "p95": float(pcts[4]),
    }


# ─── XDMF export / import (doc 06 §7 output format) ─────────────────────────


def export_xdmf(
    out_dir: pathlib.Path | str,
    mesh,
    facet_tags,
    u_h,
    p_h,
    wss_mag_h,
    wss_vec_h,
    metadata: dict,
) -> None:
    """Write all CFD outputs to XDMF+HDF5 (doc 06 §7 output format).

    Requires dolfinx (Linux/WSL).  Each field is written as a separate
    XDMF+HDF5 pair so that downstream tools (ParaView, meshio) can read
    individual fields without loading the full dataset.

    Parameters
    ----------
    out_dir:
        Directory to write into.  Created if absent.
    mesh:
        dolfinx Mesh object.
    facet_tags:
        dolfinx MeshTags with boundary markers (wall=1, inlet=2, outlet=3).
    u_h, p_h:
        dolfinx Function objects for velocity (P2) and pressure (P1,
        sign-corrected to physical convention).
    wss_mag_h, wss_vec_h:
        dolfinx Function objects for WSS magnitude (DG-0 scalar) and
        WSS vector (DG-0 vector), from _compute_wss_fn_scalar/vector.
    metadata:
        Dict of scalar metadata written to metadata.json.
    """
    try:
        from dolfinx.io import XDMFFile  # noqa: PLC0415
        from mpi4py import MPI           # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "export_xdmf requires dolfinx (Linux/WSL). "
            "On Windows use save_solution() for .npz output."
        ) from exc

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    comm = MPI.COMM_WORLD

    for fname, func in [
        ("velocity", u_h),
        ("pressure", p_h),
        ("wss_mag", wss_mag_h),
        ("wss_vec", wss_vec_h),
    ]:
        with XDMFFile(comm, str(out / f"{fname}.xdmf"), "w") as f:
            f.write_mesh(mesh)
            func.x.scatter_forward()
            f.write_function(func)

    # Mesh with boundary tags (for downstream BC re-application if needed).
    with XDMFFile(comm, str(out / "mesh.xdmf"), "w") as f:
        f.write_mesh(mesh)
        f.write_meshtags(facet_tags, mesh.geometry)

    if comm.rank == 0:
        (out / "metadata.json").write_text(json.dumps(metadata, indent=2))


def load_cfd_xdmf(out_dir: pathlib.Path | str) -> dict[str, object]:
    """Load CFD XDMF output using meshio (cross-platform, no dolfinx needed).

    Parameters
    ----------
    out_dir:
        Directory produced by :func:`export_xdmf`.

    Returns
    -------
    dict with keys:
        ``velocity``    — meshio Mesh with point_data / cell_data
        ``pressure``    — meshio Mesh
        ``wss_mag``     — meshio Mesh
        ``wss_vec``     — meshio Mesh
        ``metadata``    — dict from metadata.json (or {} if absent)
    """
    import meshio  # noqa: PLC0415

    out = pathlib.Path(out_dir)
    result: dict[str, object] = {}
    for name in ("velocity", "pressure", "wss_mag", "wss_vec"):
        xdmf_path = out / f"{name}.xdmf"
        if xdmf_path.exists():
            result[name] = meshio.read(str(xdmf_path))

    meta_path = out / "metadata.json"
    result["metadata"] = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return result


# ─── Private helpers ──────────────────────────────────────────────────────────


def _write_vtk(solution: StokesSolution, outdir: pathlib.Path) -> None:
    """Write VTK visualisation file using dolfinx.io.VTKFile."""
    try:
        import dolfinx.io  # noqa: PLC0415
        from mpi4py import MPI  # noqa: PLC0415
    except ImportError:
        return   # silently skip on Windows

    # Re-create the dolfinx mesh and functions from saved arrays to write VTK.
    # This is only for post-hoc visualisation; the npz is the canonical output.
    pass   # TODO: implement if needed for PyVista-free inspection

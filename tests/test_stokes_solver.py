"""Tests for the Stokes CFD solver module.

Fast unit tests (no dolfinx required, run on Windows):
    - detect_open_faces on synthetic two-disk geometry
    - detect_open_faces on all four named AnXplore cases (slow, marked)
    - BoundaryParts consistency checks
    - StokesConfig defaults
    - postprocess.save_solution / load_solution round-trip

dolfinx-dependent tests are marked skip when dolfinx is unavailable.
"""

from __future__ import annotations

import json
import pathlib
import tempfile

import numpy as np
import pytest

from hemodyn_pinn.cfd.stokes_solver import (
    BoundaryParts,
    StokesConfig,
    detect_open_faces,
    MU_BLOOD,
    RHO_BLOOD,
    L_SCALE,
    MARKER_WALL,
    MARKER_INLET,
    MARKER_OUTLET,
    _DOLFINX_AVAILABLE,
)
from hemodyn_pinn.cfd.postprocess import save_solution, load_solution, wss_statistics

# ─── Synthetic geometry helpers ───────────────────────────────────────────────

_DATA_ROOT = pathlib.Path(__file__).resolve().parents[1] / "data" / "geometries" / "anxplore"


def _make_two_disk_mesh(
    r_tube: float = 2.0,
    x_offset: float = 6.14,
    n_ring: int = 12,
) -> "AnxploreMesh":  # type: ignore[name-defined]
    """Return a minimal synthetic AnxploreMesh for testing detect_open_faces.

    The mesh consists of:
    - Two flat disks at y = 0 with outward normal (0, -1, 0):
        left  disk centred at (-x_offset, 0, 0)
        right disk centred at (+x_offset, 0, 0)
    - A handful of curved-wall triangles with mixed normals.

    No tetrahedral cells are added (tetra = empty), so the loader fields are
    populated directly.
    """
    from hemodyn_pinn.geometry.anxplore_loader import AnxploreMesh

    rng = np.random.default_rng(0)

    # Build two disks by triangulating a circle.
    angles = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
    rim = np.column_stack([np.cos(angles), np.zeros(n_ring), np.sin(angles)]) * r_tube

    points_list = []
    tris_list = []
    normals_list = []

    for sign in (-1, 1):
        center = np.array([sign * x_offset, 0.0, 0.0])
        base_idx = len(points_list) * (n_ring + 1)   # offset into points

        pts = np.vstack([center, rim + center])       # center + ring
        points_list.append(pts)

        for i in range(n_ring):
            j = (i + 1) % n_ring
            tris_list.append([base_idx, base_idx + 1 + i, base_idx + 1 + j])
            normals_list.append([0.0, -1.0, 0.0])    # outward -y

    # Add a few curved wall triangles at y > 0 with non-y normals.
    n_extra_pts = 6
    extra_pts = rng.uniform(-1, 1, (n_extra_pts, 3))
    extra_pts[:, 1] = rng.uniform(0.5, 5, n_extra_pts)    # y > 0

    base_wall = sum(len(p) for p in points_list)
    for i in range(0, n_extra_pts - 2, 3):
        tris_list.append([base_wall + i, base_wall + i + 1, base_wall + i + 2])
        # random normal with |n_y| < 0.5
        raw = rng.uniform(-1, 1, 3)
        raw[1] = rng.uniform(-0.4, 0.4)
        normals_list.append(raw / np.linalg.norm(raw))

    points_list.append(extra_pts)

    points_mm = np.vstack(points_list).astype(np.float64)
    wall_tri = np.array(tris_list, dtype=np.int64)
    wall_normals = np.array(normals_list, dtype=np.float64)

    # Normalise normals.
    mag = np.linalg.norm(wall_normals, axis=1, keepdims=True)
    wall_normals /= np.where(mag > 0, mag, 1.0)

    # Dummy tets (not needed for detect_open_faces).
    tetra = np.empty((0, 4), dtype=np.int64)

    return AnxploreMesh(
        case_id="synthetic",
        source_path=pathlib.Path("synthetic"),
        points_mm=points_mm,
        tetra=tetra,
        wall_tri=wall_tri,
        wall_normals=wall_normals,
    )


# ─── Unit tests for detect_open_faces ─────────────────────────────────────────


class TestDetectOpenFacesSynthetic:
    def test_masks_partition_all_faces(self) -> None:
        mesh = _make_two_disk_mesh()
        bp = detect_open_faces(mesh)
        W = mesh.wall_tri.shape[0]
        combined = bp.wall_mask.astype(int) + bp.inlet_mask.astype(int) + bp.outlet_mask.astype(int)
        assert np.all(combined == 1), "Every face must belong to exactly one class."
        assert bp.wall_mask.sum() + bp.inlet_mask.sum() + bp.outlet_mask.sum() == W

    def test_open_face_counts_match_disk_triangles(self) -> None:
        n_ring = 12
        mesh = _make_two_disk_mesh(n_ring=n_ring)
        bp = detect_open_faces(mesh)
        expected = n_ring      # each disk has n_ring triangles
        assert bp.inlet_mask.sum() == expected
        assert bp.outlet_mask.sum() == expected

    def test_inlet_on_negative_x(self) -> None:
        mesh = _make_two_disk_mesh(x_offset=6.14)
        bp = detect_open_faces(mesh)
        inlet_centroids = mesh.wall_centroids_mm[bp.inlet_mask]
        outlet_centroids = mesh.wall_centroids_mm[bp.outlet_mask]
        assert (inlet_centroids[:, 0] < 0).all(), "Inlet should be on left (x < 0)."
        assert (outlet_centroids[:, 0] > 0).all(), "Outlet should be on right (x > 0)."

    def test_inlet_normal_inward_points_positive_y(self) -> None:
        mesh = _make_two_disk_mesh()
        bp = detect_open_faces(mesh)
        assert bp.inlet_normal_inward[1] > 0.99, (
            f"Inward inlet normal should point in +y; got {bp.inlet_normal_inward}"
        )

    def test_outlet_normal_inward_points_positive_y(self) -> None:
        mesh = _make_two_disk_mesh()
        bp = detect_open_faces(mesh)
        assert bp.outlet_normal_inward[1] > 0.99

    def test_inward_normals_are_unit_vectors(self) -> None:
        mesh = _make_two_disk_mesh()
        bp = detect_open_faces(mesh)
        assert abs(np.linalg.norm(bp.inlet_normal_inward) - 1.0) < 1e-10
        assert abs(np.linalg.norm(bp.outlet_normal_inward) - 1.0) < 1e-10

    def test_raises_if_no_flat_faces(self) -> None:
        """A mesh with no n_y = -1 faces should raise ValueError."""
        from hemodyn_pinn.geometry.anxplore_loader import AnxploreMesh

        pts = np.array([[0, 1, 0], [1, 1, 0], [0, 1, 1]], dtype=np.float64)
        tri = np.array([[0, 1, 2]], dtype=np.int64)
        n = np.array([[0.0, 1.0, 0.0]])   # outward +y, NOT -y
        mesh = AnxploreMesh(
            case_id="dummy",
            source_path=pathlib.Path("dummy"),
            points_mm=pts,
            tetra=np.empty((0, 4), dtype=np.int64),
            wall_tri=tri,
            wall_normals=n,
        )
        with pytest.raises(ValueError, match="flat boundary faces"):
            detect_open_faces(mesh)


# ─── Integration tests: real AnXplore meshes ──────────────────────────────────


@pytest.mark.slow
@pytest.mark.parametrize("case", ["caseA", "caseB", "caseC", "caseR"])
def test_detect_open_faces_real_cases(case: str) -> None:
    """detect_open_faces must correctly split the two open disks on real meshes."""
    from hemodyn_pinn.geometry.anxplore_loader import load_anxplore

    vtk = _DATA_ROOT / case / f"Fluid_{case}.vtk"
    if not vtk.exists():
        pytest.skip(f"Mesh not found: {vtk}")

    mesh = load_anxplore(vtk, case)
    bp = detect_open_faces(mesh)

    W = mesh.wall_tri.shape[0]

    # Every face belongs to exactly one class.
    combined = bp.wall_mask.astype(int) + bp.inlet_mask.astype(int) + bp.outlet_mask.astype(int)
    assert np.all(combined == 1)

    # Each open patch has ~2200-2310 faces (from mesh_stats analysis).
    for name, mask in [("inlet", bp.inlet_mask), ("outlet", bp.outlet_mask)]:
        count = int(mask.sum())
        assert 2000 < count < 2600, (
            f"{case} {name}: expected ~2200-2310 faces, got {count}"
        )

    # Open face areas are equal within 1 %.
    tri = mesh.wall_tri
    pts = mesh.points_mm
    v0, v1, v2 = pts[tri[:, 0]], pts[tri[:, 1]], pts[tri[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

    area_in = float(areas[bp.inlet_mask].sum())
    area_out = float(areas[bp.outlet_mask].sum())
    assert abs(area_in - area_out) / area_in < 0.02, (
        f"{case}: inlet area {area_in:.4f} mm² vs outlet {area_out:.4f} mm²"
    )

    # Inlet is on the left (x < 0), outlet on the right (x > 0).
    centroids = (v0 + v1 + v2) / 3.0
    assert centroids[bp.inlet_mask, 0].mean() < 0
    assert centroids[bp.outlet_mask, 0].mean() > 0

    # Inward normals point in +y direction.
    assert bp.inlet_normal_inward[1] > 0.99
    assert bp.outlet_normal_inward[1] > 0.99


# ─── StokesConfig defaults ────────────────────────────────────────────────────


class TestStokesConfig:
    def test_default_mu(self) -> None:
        cfg = StokesConfig()
        assert cfg.mu == pytest.approx(MU_BLOOD, rel=1e-9)

    def test_default_u_inlet_gives_stokes_regime(self) -> None:
        cfg = StokesConfig()
        re = RHO_BLOOD * cfg.u_inlet * L_SCALE / cfg.mu
        assert re < 1.0, f"Default inlet velocity gives Re = {re:.4f} ≥ 1 (not Stokes)."

    def test_re_approximately_0p1(self) -> None:
        cfg = StokesConfig()
        re = RHO_BLOOD * cfg.u_inlet * L_SCALE / cfg.mu
        assert 0.05 < re < 0.20, f"Re = {re:.4f} outside expected range [0.05, 0.20]."

    def test_custom_viscosity(self) -> None:
        cfg = StokesConfig(mu=1e-3)
        assert cfg.mu == pytest.approx(1e-3)


# ─── postprocess: save / load round-trip ─────────────────────────────────────


def _make_dummy_solution() -> "StokesSolution":  # type: ignore[name-defined]
    """Return a minimal StokesSolution with synthetic arrays."""
    from hemodyn_pinn.cfd.stokes_solver import StokesSolution, BoundaryParts, StokesConfig

    N, W = 50, 20
    rng = np.random.default_rng(1)

    wall_m = np.zeros(W, dtype=bool)
    wall_m[:16] = True
    inlet_m = np.zeros(W, dtype=bool)
    inlet_m[16:18] = True
    outlet_m = np.zeros(W, dtype=bool)
    outlet_m[18:] = True

    bp = BoundaryParts(
        wall_mask=wall_m,
        inlet_mask=inlet_m,
        outlet_mask=outlet_m,
        inlet_normal_inward=np.array([0.0, 1.0, 0.0]),
        outlet_normal_inward=np.array([0.0, 1.0, 0.0]),
    )

    return StokesSolution(
        case_id="dummy",
        velocity_nodes=rng.random((N, 3)).astype(np.float64),
        pressure_nodes=rng.random(N).astype(np.float64),
        wss_vectors=rng.random((W, 3)).astype(np.float64),
        wss_magnitudes=rng.random(W).astype(np.float64),
        wall_centroids_m=rng.random((W, 3)).astype(np.float64),
        wall_normals=rng.random((W, 3)).astype(np.float64),
        boundary_parts=bp,
        cfg=StokesConfig(),
        re_number=0.1,
    )


class TestPostprocess:
    def test_save_load_roundtrip(self) -> None:
        sol = _make_dummy_solution()
        with tempfile.TemporaryDirectory() as tmp:
            outdir = pathlib.Path(tmp) / "dummy"
            npz = save_solution(sol, outdir)
            data = load_solution(npz)

        assert "velocity" in data
        np.testing.assert_array_equal(data["velocity"], sol.velocity_nodes)
        np.testing.assert_array_equal(data["pressure"], sol.pressure_nodes)
        np.testing.assert_array_equal(data["wss_magnitudes"], sol.wss_magnitudes)

    def test_metadata_json_written(self) -> None:
        sol = _make_dummy_solution()
        with tempfile.TemporaryDirectory() as tmp:
            outdir = pathlib.Path(tmp) / "dummy"
            save_solution(sol, outdir)
            meta = json.loads((outdir / "metadata.json").read_text())

        assert meta["case_id"] == "dummy"
        assert "re_number" in meta
        assert "wss_wall_mean_Pa" in meta

    def test_overwrite_protection(self) -> None:
        sol = _make_dummy_solution()
        with tempfile.TemporaryDirectory() as tmp:
            outdir = pathlib.Path(tmp) / "dummy"
            save_solution(sol, outdir)
            with pytest.raises(FileExistsError):
                save_solution(sol, outdir, overwrite=False)
            save_solution(sol, outdir, overwrite=True)   # should not raise

    def test_wss_statistics_only_wall(self) -> None:
        sol = _make_dummy_solution()
        stats = wss_statistics(sol)
        # wall_mask has 16 True faces; inlet/outlet 4 faces with WSS set in dummy
        assert stats["mean"] >= 0
        assert stats["max"] >= stats["mean"] >= stats["min"]
        assert stats["p5"] <= stats["p25"] <= stats["median"] <= stats["p75"] <= stats["p95"]


# ─── dolfinx-dependent tests (skipped on Windows) ────────────────────────────


@pytest.mark.skipif(
    not _DOLFINX_AVAILABLE,
    reason="dolfinx not available (run in Linux/WSL with FEniCSx installed)",
)
@pytest.mark.slow
@pytest.mark.parametrize("case", ["caseC"])
def test_solve_stokes_smoke(case: str) -> None:
    """Smoke test: solver runs on caseC and produces plausible results."""
    from hemodyn_pinn.geometry.anxplore_loader import load_anxplore
    from hemodyn_pinn.cfd.stokes_solver import solve_stokes, StokesConfig

    vtk = _DATA_ROOT / case / f"Fluid_{case}.vtk"
    if not vtk.exists():
        pytest.skip(f"Mesh not found: {vtk}")

    mesh = load_anxplore(vtk, case)
    cfg = StokesConfig(u_inlet=2e-5)
    sol = solve_stokes(mesh, cfg)

    N = mesh.n_nodes
    W = mesh.n_wall_tri

    # Array shapes.
    assert sol.velocity_nodes.shape == (N, 3)
    assert sol.pressure_nodes.shape == (N,)
    assert sol.wss_magnitudes.shape == (W,)

    # Velocity not all zero.
    assert np.linalg.norm(sol.velocity_nodes) > 0

    # WSS on wall faces is non-negative.
    assert (sol.wss_magnitudes[sol.boundary_parts.wall_mask] >= 0).all()

    # Re in Stokes regime.
    assert sol.re_number < 1.0

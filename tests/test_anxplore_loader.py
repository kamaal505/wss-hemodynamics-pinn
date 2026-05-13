"""Tests for anxplore_loader.py — Phase 1.2 mandatory test suite.

Fast unit tests use small synthetic meshes constructed in-memory.
Integration tests (marked ``slow``) load real AnXplore VTK files;
run them with ``pytest -m slow`` or omit ``-m 'not slow'`` from CI.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from hemodyn_pinn.geometry.anxplore_loader import (
    AnxploreMesh,
    _compute_outward_normals,
    _compute_outward_normals_from_centroid,
    _extract_boundary_faces,
    load_anxplore,
)

# ---------------------------------------------------------------------------
# Paths to real mesh files (skip integration tests if absent)
# ---------------------------------------------------------------------------

_ANXPLORE = pathlib.Path(__file__).parent.parent / "data" / "geometries" / "anxplore"
_CASE_A = _ANXPLORE / "caseA" / "Fluid_caseA.vtk"
_FLUID_0 = _ANXPLORE / "full_dataset" / "Fluid_0.vtk"


# ---------------------------------------------------------------------------
# Synthetic mesh fixtures
# ---------------------------------------------------------------------------


def _make_two_tet_mesh() -> tuple[np.ndarray, np.ndarray]:
    """Two tets sharing face {1,2,3}; 6 boundary faces expected.

    Vertices:
        0: (0,0,0)  1: (1,0,0)  2: (0,1,0)  3: (0,0,1)  4: (1,1,1)

    Tet 0: [0,1,2,3]   Tet 1: [1,2,3,4]
    Shared face (sorted): (1,2,3)
    Boundary faces: (0,1,2), (0,1,3), (0,2,3), (1,2,4), (1,3,4), (2,3,4)
    """
    points = np.array(
        [[0.0, 0.0, 0.0],
         [1.0, 0.0, 0.0],
         [0.0, 1.0, 0.0],
         [0.0, 0.0, 1.0],
         [1.0, 1.0, 1.0]],
        dtype=np.float64,
    )
    tetra = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64)
    return points, tetra


def _make_single_tet_mesh() -> tuple[np.ndarray, np.ndarray]:
    """A single regular tetrahedron; all 4 faces are boundary faces."""
    s = 1.0 / np.sqrt(2.0)
    points = np.array(
        [[ 1.0,  0.0, -s],
         [-1.0,  0.0, -s],
         [ 0.0,  1.0,  s],
         [ 0.0, -1.0,  s]],
        dtype=np.float64,
    )
    tetra = np.array([[0, 1, 2, 3]], dtype=np.int64)
    return points, tetra


# ---------------------------------------------------------------------------
# Unit tests — boundary face extraction
# ---------------------------------------------------------------------------


class TestExtractBoundaryFaces:
    def test_single_tet_has_four_boundary_faces(self) -> None:
        _, tetra = _make_single_tet_mesh()
        wall_tri, opp_vtx = _extract_boundary_faces(tetra)
        assert wall_tri.shape == (4, 3)
        assert opp_vtx.shape == (4,)

    def test_two_tets_shared_face_gives_six_boundary_faces(self) -> None:
        _, tetra = _make_two_tet_mesh()
        wall_tri, opp_vtx = _extract_boundary_faces(tetra)
        assert wall_tri.shape[0] == 6

    def test_boundary_node_indices_in_range(self) -> None:
        points, tetra = _make_two_tet_mesh()
        wall_tri, _ = _extract_boundary_faces(tetra)
        assert wall_tri.min() >= 0
        assert wall_tri.max() < points.shape[0]

    def test_opp_vtx_indices_in_range(self) -> None:
        points, tetra = _make_two_tet_mesh()
        _, opp_vtx = _extract_boundary_faces(tetra)
        assert opp_vtx.min() >= 0
        assert opp_vtx.max() < points.shape[0]

    def test_all_boundary_faces_are_unique(self) -> None:
        _, tetra = _make_two_tet_mesh()
        wall_tri, _ = _extract_boundary_faces(tetra)
        sorted_rows = np.sort(wall_tri, axis=1)
        # Each sorted face row must be unique
        unique_rows = np.unique(sorted_rows, axis=0)
        assert unique_rows.shape[0] == wall_tri.shape[0]

    def test_shared_face_not_in_boundary(self) -> None:
        """Face {1,2,3} is shared by both tets and must not be a boundary face."""
        _, tetra = _make_two_tet_mesh()
        wall_tri, _ = _extract_boundary_faces(tetra)
        sorted_rows = np.sort(wall_tri, axis=1)
        shared = np.array([[1, 2, 3]])
        matches = np.all(sorted_rows == shared, axis=1)
        assert not matches.any(), "Shared face {1,2,3} must not appear in boundary"

    def test_dtype_is_int64(self) -> None:
        _, tetra = _make_two_tet_mesh()
        wall_tri, opp_vtx = _extract_boundary_faces(tetra)
        assert wall_tri.dtype == np.int64
        assert opp_vtx.dtype == np.int64


# ---------------------------------------------------------------------------
# Unit tests — outward normal computation
# ---------------------------------------------------------------------------


class TestOutwardNormals:
    def test_normals_are_unit_vectors(self) -> None:
        points, tetra = _make_single_tet_mesh()
        wall_tri, opp_vtx = _extract_boundary_faces(tetra)
        normals = _compute_outward_normals(points, wall_tri, opp_vtx)
        magnitudes = np.linalg.norm(normals, axis=1)
        np.testing.assert_allclose(magnitudes, 1.0, atol=1e-12)

    def test_normals_point_outward(self) -> None:
        """Normal must point away from the mesh centroid (outward)."""
        points, tetra = _make_single_tet_mesh()
        wall_tri, opp_vtx = _extract_boundary_faces(tetra)
        normals = _compute_outward_normals(points, wall_tri, opp_vtx)

        face_centroids = (
            points[wall_tri[:, 0]]
            + points[wall_tri[:, 1]]
            + points[wall_tri[:, 2]]
        ) / 3.0
        mesh_centroid = points.mean(axis=0)

        # outward = face_centroids - mesh_centroid should align with normal
        outward_ref = face_centroids - mesh_centroid
        dot = np.einsum("ij,ij->i", normals, outward_ref)
        assert np.all(dot > 0), "All normals must point away from the mesh centroid"

    def test_normals_from_centroid_are_unit_vectors(self) -> None:
        points, tetra = _make_two_tet_mesh()
        wall_tri, _ = _extract_boundary_faces(tetra)
        normals = _compute_outward_normals_from_centroid(points, wall_tri)
        magnitudes = np.linalg.norm(normals, axis=1)
        np.testing.assert_allclose(magnitudes, 1.0, atol=1e-12)

    def test_normals_are_float64(self) -> None:
        points, tetra = _make_single_tet_mesh()
        wall_tri, opp_vtx = _extract_boundary_faces(tetra)
        normals = _compute_outward_normals(points, wall_tri, opp_vtx)
        assert normals.dtype == np.float64

    def test_two_tet_mesh_normals_outward(self) -> None:
        points, tetra = _make_two_tet_mesh()
        wall_tri, opp_vtx = _extract_boundary_faces(tetra)
        normals = _compute_outward_normals(points, wall_tri, opp_vtx)

        face_centroids = (
            points[wall_tri[:, 0]]
            + points[wall_tri[:, 1]]
            + points[wall_tri[:, 2]]
        ) / 3.0
        mesh_centroid = points.mean(axis=0)
        outward_ref = face_centroids - mesh_centroid
        dot = np.einsum("ij,ij->i", normals, outward_ref)
        assert np.all(dot > 0), "All normals must point away from the mesh centroid"


# ---------------------------------------------------------------------------
# Unit tests — AnxploreMesh constructed from synthetic data
# ---------------------------------------------------------------------------


class TestAnxploreMeshProperties:
    @pytest.fixture()
    def synthetic_mesh(self) -> AnxploreMesh:
        points, tetra = _make_two_tet_mesh()
        wall_tri, opp_vtx = _extract_boundary_faces(tetra)
        normals = _compute_outward_normals(points, wall_tri, opp_vtx)
        return AnxploreMesh(
            case_id="synthetic",
            source_path=pathlib.Path("synthetic.vtk"),
            points_mm=points,
            tetra=tetra,
            wall_tri=wall_tri,
            wall_normals=normals,
        )

    def test_n_nodes(self, synthetic_mesh: AnxploreMesh) -> None:
        assert synthetic_mesh.n_nodes == 5

    def test_n_tets(self, synthetic_mesh: AnxploreMesh) -> None:
        assert synthetic_mesh.n_tets == 2

    def test_n_wall_tri(self, synthetic_mesh: AnxploreMesh) -> None:
        assert synthetic_mesh.n_wall_tri == 6

    def test_points_m_scale(self, synthetic_mesh: AnxploreMesh) -> None:
        np.testing.assert_allclose(
            synthetic_mesh.points_m, synthetic_mesh.points_mm * 1e-3
        )

    def test_wall_centroids_shape(self, synthetic_mesh: AnxploreMesh) -> None:
        c = synthetic_mesh.wall_centroids_mm
        assert c.shape == (6, 3)

    def test_mesh_stats_no_inverted_tets(self, synthetic_mesh: AnxploreMesh) -> None:
        stats = synthetic_mesh.mesh_stats()
        assert stats["n_inverted_tets"] == 0

    def test_mesh_stats_vol_positive(self, synthetic_mesh: AnxploreMesh) -> None:
        stats = synthetic_mesh.mesh_stats()
        assert stats["vol_min_mm3"] > 0.0

    def test_mesh_stats_aspect_ratio_finite(self, synthetic_mesh: AnxploreMesh) -> None:
        stats = synthetic_mesh.mesh_stats()
        assert np.isfinite(stats["aspect_ratio_mean"])
        assert stats["aspect_ratio_mean"] >= 1.0


# ---------------------------------------------------------------------------
# Integration tests — real AnXplore files
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestLoadCaseA:
    """Load caseA (tet-only, no explicit surface triangles)."""

    @pytest.fixture(scope="class")
    def mesh(self) -> AnxploreMesh:
        if not _CASE_A.exists():
            pytest.skip(f"Real mesh file not found: {_CASE_A}")
        return load_anxplore(_CASE_A, case_id="caseA")

    def test_node_count(self, mesh: AnxploreMesh) -> None:
        assert mesh.n_nodes == 159_155

    def test_tet_count(self, mesh: AnxploreMesh) -> None:
        assert mesh.n_tets == 912_713

    def test_wall_tri_nonempty(self, mesh: AnxploreMesh) -> None:
        assert mesh.n_wall_tri > 0

    def test_x_span_mm(self, mesh: AnxploreMesh) -> None:
        """Parent artery x-span should be 16.279 mm (±0.002 mm)."""
        span = mesh.points_mm[:, 0].max() - mesh.points_mm[:, 0].min()
        assert abs(span - 16.279) < 0.002

    def test_wall_tri_indices_in_range(self, mesh: AnxploreMesh) -> None:
        assert mesh.wall_tri.min() >= 0
        assert mesh.wall_tri.max() < mesh.n_nodes

    def test_tetra_indices_in_range(self, mesh: AnxploreMesh) -> None:
        assert mesh.tetra.min() >= 0
        assert mesh.tetra.max() < mesh.n_nodes

    def test_wall_normals_unit(self, mesh: AnxploreMesh) -> None:
        mags = np.linalg.norm(mesh.wall_normals, axis=1)
        np.testing.assert_allclose(mags, 1.0, atol=1e-10)

    def test_no_inverted_tets(self, mesh: AnxploreMesh) -> None:
        stats = mesh.mesh_stats()
        assert stats["n_inverted_tets"] == 0, (
            f"Found {stats['n_inverted_tets']} inverted tetrahedra in caseA"
        )

    def test_edge_mean_matches_known(self, mesh: AnxploreMesh) -> None:
        """Mean edge length should be ≈ 0.1563 mm (mesh_stats.txt ground truth)."""
        stats = mesh.mesh_stats()
        assert abs(stats["edge_mean_mm"] - 0.1563) < 0.002

    def test_case_id(self, mesh: AnxploreMesh) -> None:
        assert mesh.case_id == "caseA"

    def test_mesh_uses_boundary_extraction(self, mesh: AnxploreMesh) -> None:
        """Tet-only mesh: wall_tri must have been extracted (not from explicit tris)."""
        import meshio as _meshio
        raw = _meshio.read(str(_CASE_A))
        assert "triangle" not in raw.cells_dict or len(raw.cells_dict["triangle"]) == 0


@pytest.mark.slow
class TestLoadFluid0:
    """Load full_dataset/Fluid_0.vtk (explicit surface triangles present)."""

    @pytest.fixture(scope="class")
    def mesh(self) -> AnxploreMesh:
        if not _FLUID_0.exists():
            pytest.skip(f"Real mesh file not found: {_FLUID_0}")
        return load_anxplore(_FLUID_0, case_id="Fluid_0")

    def test_node_count(self, mesh: AnxploreMesh) -> None:
        assert mesh.n_nodes == 158_840

    def test_tet_count(self, mesh: AnxploreMesh) -> None:
        assert mesh.n_tets == 910_886

    def test_wall_tri_count_matches_explicit(self, mesh: AnxploreMesh) -> None:
        """Explicit triangles in Fluid_0.vtk: 33,116 (from mesh_stats.txt)."""
        assert mesh.n_wall_tri == 33_116

    def test_wall_normals_unit(self, mesh: AnxploreMesh) -> None:
        mags = np.linalg.norm(mesh.wall_normals, axis=1)
        np.testing.assert_allclose(mags, 1.0, atol=1e-10)

    def test_no_inverted_tets(self, mesh: AnxploreMesh) -> None:
        stats = mesh.mesh_stats()
        assert stats["n_inverted_tets"] == 0

    def test_x_span_mm(self, mesh: AnxploreMesh) -> None:
        span = mesh.points_mm[:, 0].max() - mesh.points_mm[:, 0].min()
        assert abs(span - 16.278) < 0.002


@pytest.mark.slow
class TestLoadErrors:
    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_anxplore("/nonexistent/path/mesh.vtk")

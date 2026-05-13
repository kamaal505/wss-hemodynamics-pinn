"""AnXplore fluid mesh loader with boundary extraction and mesh quality checks.

Handles two mesh variants found in the AnXplore dataset:
- Named cases (caseA/B/C/R): volumetric tets only, no explicit surface
  triangles. The wall boundary Γ_w is extracted as the set of tet faces
  shared by exactly one tetrahedron.
- full_dataset (Fluid_0 … Fluid_100): volumetric tets plus 33 k explicit
  surface triangles. These are used directly as Γ_w.

All coordinates are stored in millimetres (as-is from the VTK files).
Convert to SI metres via the ``points_m`` property before any physical
computation.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Optional

import meshio
import numpy as np

from hemodyn_pinn.utils.seeds import MESH_STATS_SEED, make_numpy_rng

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

# For tet [v0, v1, v2, v3], face j uses the three local vertex positions
# listed in row j of this table. Face j is opposite to local vertex j.
_FACE_LOCAL_INDICES = np.array(
    [[1, 2, 3],
     [0, 2, 3],
     [0, 1, 3],
     [0, 1, 2]],
    dtype=np.intp,
)

_EDGE_PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def _extract_boundary_faces(
    tetra: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return boundary triangles extracted from a pure-tet mesh.

    A boundary face is any tet face shared by exactly one tetrahedron.

    Parameters
    ----------
    tetra:
        (T, 4) integer array of global node indices.

    Returns
    -------
    wall_tri:
        (W, 3) int64 — global node indices of each boundary triangle.
    opp_vtx:
        (W,) int64 — global index of the tet vertex opposite to each
        boundary face, used to orient outward normals.
    """
    tetra = tetra.astype(np.int64)
    T = tetra.shape[0]

    # All 4T faces as (4T, 3). Row 4i+j = face j of tet i.
    # tetra[:, _FACE_LOCAL_INDICES] has shape (T, 4, 3).
    all_faces = tetra[:, _FACE_LOCAL_INDICES].reshape(-1, 3)

    # For row 4i+j the opposite vertex is tetra[i, j].
    # tetra.reshape(-1) maps index 4i+j → tetra[i, j]. ✓
    all_opp_flat = tetra.reshape(-1)  # (4T,)

    # Canonical face key: sort vertices, encode as a single int64.
    # Max key = N³ - 1 where N = n_nodes + 1 ≤ ~180k.
    # N³ ≤ (180_001)³ ≈ 5.8e15 < 9.2e18 = int64 max.  ✓
    N = int(tetra.max()) + 1
    sorted_faces = np.sort(all_faces, axis=1)
    keys = (
        sorted_faces[:, 0] * np.int64(N) * np.int64(N)
        + sorted_faces[:, 1] * np.int64(N)
        + sorted_faces[:, 2]
    )

    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)

    boundary_mask = counts[inverse] == 1
    boundary_rows = np.where(boundary_mask)[0]

    return all_faces[boundary_rows], all_opp_flat[boundary_rows]


def _compute_outward_normals(
    points: np.ndarray,
    wall_tri: np.ndarray,
    opp_vtx: np.ndarray,
) -> np.ndarray:
    """Compute outward unit normals for boundary triangles.

    Orientation is determined by the opposite tet vertex: the normal must
    point away from it (i.e. out of the fluid domain).

    Parameters
    ----------
    points:
        (N, 3) node coordinates (any consistent unit).
    wall_tri:
        (W, 3) boundary triangle connectivity.
    opp_vtx:
        (W,) global index of the vertex opposite to each triangle.

    Returns
    -------
    normals:
        (W, 3) outward unit normal vectors.
    """
    v0 = points[wall_tri[:, 0]]
    v1 = points[wall_tri[:, 1]]
    v2 = points[wall_tri[:, 2]]

    normals = np.cross(v1 - v0, v2 - v0)  # (W, 3)

    # inward = vector from face centroid toward the interior (opposite) vertex
    face_centroid = (v0 + v1 + v2) / 3.0
    inward = points[opp_vtx] - face_centroid  # (W, 3)

    # Flip normals that point in the same direction as inward
    dot = np.einsum("ij,ij->i", normals, inward)
    normals[dot > 0] *= -1

    # Normalise (degenerate faces get magnitude 1 to avoid NaN)
    mag = np.linalg.norm(normals, axis=1, keepdims=True)
    mag = np.where(mag > 0.0, mag, 1.0)
    return normals / mag


def _compute_outward_normals_from_centroid(
    points: np.ndarray,
    wall_tri: np.ndarray,
) -> np.ndarray:
    """Compute outward normals using the mesh centroid as interior reference.

    Used when no opposite-vertex information is available (explicit triangle
    case). Valid for compact vascular geometries where the mesh centroid lies
    well inside the fluid domain.
    """
    v0 = points[wall_tri[:, 0]]
    v1 = points[wall_tri[:, 1]]
    v2 = points[wall_tri[:, 2]]

    normals = np.cross(v1 - v0, v2 - v0)  # (W, 3)

    face_centroid = (v0 + v1 + v2) / 3.0
    # inward: from face centroid toward the global mesh centroid
    mesh_centroid = points.mean(axis=0)
    inward = mesh_centroid - face_centroid  # (W, 3)

    dot = np.einsum("ij,ij->i", normals, inward)
    normals[dot > 0] *= -1

    mag = np.linalg.norm(normals, axis=1, keepdims=True)
    mag = np.where(mag > 0.0, mag, 1.0)
    return normals / mag


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass
class AnxploreMesh:
    """Loaded AnXplore fluid mesh with precomputed boundary geometry.

    Attributes
    ----------
    case_id:
        Human-readable case identifier (e.g. ``"caseA"`` or ``"Fluid_0"``).
    source_path:
        Absolute path to the source VTK file.
    points_mm:
        (N, 3) float64 node coordinates in **millimetres** (as stored in
        the AnXplore VTK files).
    tetra:
        (T, 4) int64 tetrahedral element connectivity (global node indices).
    wall_tri:
        (W, 3) int64 boundary (wall) triangle connectivity.  For tet-only
        cases this is extracted algorithmically; for meshes with explicit
        surface triangles the stored triangles are used directly.
    wall_normals:
        (W, 3) float64 outward unit normal vectors at each wall triangle.
        Computed via autograd-compatible cross-product + orientation check;
        NOT finite-differenced.
    """

    case_id: str
    source_path: pathlib.Path
    points_mm: np.ndarray    # (N, 3) float64
    tetra: np.ndarray        # (T, 4) int64
    wall_tri: np.ndarray     # (W, 3) int64
    wall_normals: np.ndarray  # (W, 3) float64

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def n_nodes(self) -> int:
        return int(self.points_mm.shape[0])

    @property
    def n_tets(self) -> int:
        return int(self.tetra.shape[0])

    @property
    def n_wall_tri(self) -> int:
        return int(self.wall_tri.shape[0])

    @property
    def points_m(self) -> np.ndarray:
        """(N, 3) float64 node coordinates in **metres** (SI units)."""
        return self.points_mm * 1e-3

    @property
    def wall_centroids_mm(self) -> np.ndarray:
        """(W, 3) float64 centroid of each wall triangle in mm."""
        pts = self.points_mm
        return (pts[self.wall_tri[:, 0]]
                + pts[self.wall_tri[:, 1]]
                + pts[self.wall_tri[:, 2]]) / 3.0

    # ------------------------------------------------------------------
    # Mesh quality
    # ------------------------------------------------------------------

    def mesh_stats(self) -> dict[str, object]:
        """Return a dict of mesh quality metrics.

        Edge-length statistics are computed from a reproducible random sample
        of ``min(50_000, n_tets)`` tetrahedra (6 edges each → ≤ 300 k edges).
        All tet volumes are evaluated to check for inverted elements.

        Returns
        -------
        dict with keys:
            case_id, n_nodes, n_tets, n_wall_tri,
            bbox_min_mm, bbox_max_mm, bbox_span_mm,
            edge_mean_mm, edge_std_mm, edge_min_mm, edge_max_mm,
            edge_p5_mm, edge_p25_mm, edge_median_mm, edge_p75_mm, edge_p95_mm,
            n_iqr_outlier_edges, n_sample_edges,
            n_inverted_tets, vol_min_mm3, vol_mean_mm3, vol_max_mm3,
            aspect_ratio_mean, aspect_ratio_max.
        """
        pts = self.points_mm
        tet = self.tetra

        # Bounding box
        bbox_min = pts.min(axis=0)
        bbox_max = pts.max(axis=0)

        # Edge-length statistics on a reproducible sample
        rng = make_numpy_rng(MESH_STATS_SEED)
        n_sample = min(50_000, self.n_tets)
        idx = rng.choice(self.n_tets, size=n_sample, replace=False)
        s = tet[idx]  # (n_sample, 4)
        edge_vecs = [
            np.linalg.norm(pts[s[:, a]] - pts[s[:, b]], axis=1)
            for a, b in _EDGE_PAIRS
        ]
        edge_lens = np.concatenate(edge_vecs)  # (6 * n_sample,)

        q25, q75 = np.percentile(edge_lens, [25, 75])
        iqr_fence = 1.5 * (q75 - q25)
        n_iqr_out = int(
            np.sum((edge_lens < q25 - iqr_fence) | (edge_lens > q75 + iqr_fence))
        )

        # Aspect ratio per sampled tet: max_edge / min_edge
        edge_per_tet = np.stack(edge_vecs, axis=1)  # (n_sample, 6)
        ar = edge_per_tet.max(axis=1) / np.where(
            edge_per_tet.min(axis=1) > 0, edge_per_tet.min(axis=1), np.nan
        )

        # Signed tet volumes (all tets — checks for inverted elements)
        v0 = pts[tet[:, 0]]
        e1 = pts[tet[:, 1]] - v0
        e2 = pts[tet[:, 2]] - v0
        e3 = pts[tet[:, 3]] - v0
        signed_vol = np.einsum("ij,ij->i", e1, np.cross(e2, e3)) / 6.0

        return {
            "case_id": self.case_id,
            "n_nodes": self.n_nodes,
            "n_tets": self.n_tets,
            "n_wall_tri": self.n_wall_tri,
            "bbox_min_mm": bbox_min.tolist(),
            "bbox_max_mm": bbox_max.tolist(),
            "bbox_span_mm": (bbox_max - bbox_min).tolist(),
            "edge_mean_mm": float(edge_lens.mean()),
            "edge_std_mm": float(edge_lens.std()),
            "edge_min_mm": float(edge_lens.min()),
            "edge_max_mm": float(edge_lens.max()),
            "edge_p5_mm": float(np.percentile(edge_lens, 5)),
            "edge_p25_mm": float(np.percentile(edge_lens, 25)),
            "edge_median_mm": float(np.median(edge_lens)),
            "edge_p75_mm": float(np.percentile(edge_lens, 75)),
            "edge_p95_mm": float(np.percentile(edge_lens, 95)),
            "n_iqr_outlier_edges": n_iqr_out,
            "n_sample_edges": int(edge_lens.size),
            "n_inverted_tets": int((signed_vol <= 0.0).sum()),
            "vol_min_mm3": float(signed_vol.min()),
            "vol_mean_mm3": float(signed_vol.mean()),
            "vol_max_mm3": float(signed_vol.max()),
            "aspect_ratio_mean": float(np.nanmean(ar)),
            "aspect_ratio_max": float(np.nanmax(ar)),
        }


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------


def load_anxplore(
    path: pathlib.Path | str,
    case_id: Optional[str] = None,
) -> AnxploreMesh:
    """Load a single AnXplore fluid mesh (.vtk) and return an AnxploreMesh.

    Parameters
    ----------
    path:
        Path to the VTK file (e.g. ``data/geometries/anxplore/caseA/Fluid_caseA.vtk``).
    case_id:
        Human-readable identifier.  Defaults to the file stem (e.g. ``"Fluid_caseA"``).

    Returns
    -------
    AnxploreMesh
        Fully populated dataclass with wall triangles and outward normals.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the mesh contains no tetrahedral elements.
    """
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Mesh file not found: {path}")

    if case_id is None:
        case_id = path.stem

    mesh = meshio.read(str(path))
    points_mm = mesh.points.astype(np.float64)  # (N, 3), coordinates in mm

    # Tet connectivity
    cells = mesh.cells_dict
    if "tetra" not in cells:
        raise ValueError(
            f"No tetrahedral cells found in {path}. "
            f"Cell types present: {list(cells.keys())}"
        )
    tetra = cells["tetra"].astype(np.int64)

    # Wall triangles and normals
    if "triangle" in cells and len(cells["triangle"]) > 0:
        # full_dataset variant: explicit surface triangles available
        wall_tri = cells["triangle"].astype(np.int64)
        wall_normals = _compute_outward_normals_from_centroid(points_mm, wall_tri)
    else:
        # Named-case variant (caseA/B/C/R): extract boundary from tets
        wall_tri, opp_vtx = _extract_boundary_faces(tetra)
        wall_normals = _compute_outward_normals(points_mm, wall_tri, opp_vtx)

    return AnxploreMesh(
        case_id=case_id,
        source_path=path.resolve(),
        points_mm=points_mm,
        tetra=tetra,
        wall_tri=wall_tri,
        wall_normals=wall_normals,
    )

"""Steady Stokes flow solver using FEniCSx (dolfinx 0.10) Taylor-Hood P2-P1.

Geometry facts (from mesh inspection of all AnXplore named cases):
- Both inlet and outlet are at y=0 with outward normal (0,-1,0).
- They form two separate circular disks centered at x ≈ ±6.14 mm, separated by a
  gap at |x| < 2 mm (no faces there).
- Left disk (x < 0) is treated as the inlet (Dirichlet BC).
- Right disk (x > 0) is treated as the outlet (natural/do-nothing BC).

FEniCSx is NOT available on Windows natively.  Run in Linux / WSL with:
  sudo add-apt-repository ppa:fenics-packages/fenics && sudo apt install fenicsx
"""

from __future__ import annotations

import dataclasses
import pathlib
import warnings
from typing import Optional

import numpy as np

from hemodyn_pinn.geometry.anxplore_loader import AnxploreMesh

# ─── Optional dolfinx import ──────────────────────────────────────────────────

try:
    import dolfinx
    import dolfinx.fem as dfem
    import dolfinx.mesh as dmesh
    from dolfinx.fem.petsc import LinearProblem
    from mpi4py import MPI
    from petsc4py import PETSc as _PETSc
    import ufl
    from basix.ufl import element as basix_element

    _DOLFINX_AVAILABLE = True
except ImportError:
    _DOLFINX_AVAILABLE = False

# ─── Physical constants ───────────────────────────────────────────────────────

MU_BLOOD: float = 3.5e-3      # Pa·s  Newtonian blood viscosity
RHO_BLOOD: float = 1060.0     # kg/m³
L_SCALE: float = 0.01628      # m     parent-artery diameter (non-dim length)

# Facet marker values used in dolfinx MeshTags
MARKER_WALL: int = 1
MARKER_INLET: int = 2
MARKER_OUTLET: int = 3

# ─── Configuration ────────────────────────────────────────────────────────────


@dataclasses.dataclass
class StokesConfig:
    """Configuration for the steady Stokes solver.

    Parameters
    ----------
    mu:
        Dynamic viscosity [Pa·s].  Default: Newtonian blood estimate.
    u_inlet:
        Mean normal velocity at the inlet [m/s].
        Default 2e-5 m/s gives Re ≈ 0.10 (well in Stokes regime).
    ksp_rtol, ksp_max_it:
        MINRES convergence tolerance and iteration cap (doc 06 §5).
    save_vtk:
        Write VTK output files in addition to the .npz solution.
    """

    mu: float = MU_BLOOD
    u_inlet: float = 2e-5
    ksp_rtol: float = 1e-10
    ksp_max_it: int = 500
    save_vtk: bool = False


# ─── Boundary detection (pure NumPy — runs on Windows) ───────────────────────


@dataclasses.dataclass
class BoundaryParts:
    """Classification of all ``wall_tri`` faces as wall, inlet, or outlet.

    Attributes
    ----------
    wall_mask:
        (W,) bool — True for faces on the vessel wall (no-slip surface).
    inlet_mask:
        (W,) bool — True for the inlet open face (left disk, x < 0).
    outlet_mask:
        (W,) bool — True for the outlet open face (right disk, x > 0).
    inlet_normal_inward:
        (3,) float — unit vector pointing *into* the domain at the inlet.
        For all AnXplore cases this is (0, +1, 0).
    outlet_normal_inward:
        (3,) float — unit vector pointing *into* the domain at the outlet.
        For all AnXplore cases this is (0, +1, 0).
    """

    wall_mask: np.ndarray          # (W,) bool
    inlet_mask: np.ndarray         # (W,) bool
    outlet_mask: np.ndarray        # (W,) bool
    inlet_normal_inward: np.ndarray   # (3,) float64
    outlet_normal_inward: np.ndarray  # (3,) float64


def detect_open_faces(mesh: AnxploreMesh) -> BoundaryParts:
    """Identify inlet/outlet patches in the AnXplore boundary mesh.

    Both open faces lie in the y = 0 plane (outward normal exactly (0,-1,0)).
    They are separated at x = 0: the left disk (x < 0) is labelled *inlet*,
    the right disk (x > 0) *outlet*.

    Parameters
    ----------
    mesh:
        Loaded AnXplore mesh (from :func:`~hemodyn_pinn.geometry.anxplore_loader.load_anxplore`).

    Returns
    -------
    BoundaryParts
        Classification of every face in ``mesh.wall_tri``.

    Raises
    ------
    ValueError
        If fewer than 10 flat faces are found (geometry mismatch).
    """
    n = mesh.wall_normals                                 # (W, 3)
    centroids = mesh.wall_centroids_mm                    # (W, 3)

    # Step 1: find faces with n_y = -1 exactly (planar cut at y = 0).
    # The precision threshold 0.9999 captures all AnXplore cases with zero
    # false positives on the curved wall (validated on caseA/B/C/R and Fluid_0).
    flat_mask = n[:, 1] < -0.9999

    # Ignore the rare 1-face artefact that appears at y ≈ 4.14 in caseR/Fluid_0.
    y_min = centroids[:, 1].min()
    flat_mask &= centroids[:, 1] < y_min + 0.01     # within 0.01 mm of y_min

    n_flat = int(flat_mask.sum())
    if n_flat < 10:
        raise ValueError(
            f"detect_open_faces: only {n_flat} flat boundary faces found "
            f"(expected ~4 400). Check that the mesh is an AnXplore fluid mesh."
        )

    flat_idx = np.where(flat_mask)[0]
    flat_x = centroids[flat_idx, 0]

    # Step 2: split at x = 0 (the two disks are centred at x ≈ ±6.14 mm with
    # a gap of ~12 mm between them, so x = 0 is unambiguous).
    x_split = np.median(flat_x)              # ≈ 0 for symmetric geometry
    left = flat_idx[flat_x < x_split]        # inlet (x < 0)
    right = flat_idx[flat_x >= x_split]      # outlet (x > 0)

    if len(left) == 0 or len(right) == 0:
        raise ValueError(
            "detect_open_faces: could not split flat faces into two groups. "
            f"Median split value = {x_split:.3f} mm, "
            f"flat-face x range = [{flat_x.min():.3f}, {flat_x.max():.3f}] mm."
        )

    W = mesh.wall_tri.shape[0]
    inlet_mask = np.zeros(W, dtype=bool)
    outlet_mask = np.zeros(W, dtype=bool)
    inlet_mask[left] = True
    outlet_mask[right] = True
    wall_mask = ~inlet_mask & ~outlet_mask

    # Inward normal = negative of mean outward normal at each patch.
    def _mean_inward(idx: np.ndarray) -> np.ndarray:
        n_out = mesh.wall_normals[idx].mean(axis=0)
        n_out /= np.linalg.norm(n_out)
        return -n_out

    return BoundaryParts(
        wall_mask=wall_mask,
        inlet_mask=inlet_mask,
        outlet_mask=outlet_mask,
        inlet_normal_inward=_mean_inward(left),
        outlet_normal_inward=_mean_inward(right),
    )


# ─── Results container ────────────────────────────────────────────────────────


@dataclasses.dataclass
class StokesSolution:
    """Results from the FEniCSx Stokes solver.

    All arrays are indexed consistently with the source ``AnxploreMesh``:
    * velocity/pressure arrays have length N (mesh nodes)
    * wss arrays have length W (wall triangles)

    Attributes
    ----------
    case_id:
        Human-readable identifier from the AnxploreMesh.
    velocity_nodes:
        (N, 3) float64 velocity field in m/s at each mesh node.
    pressure_nodes:
        (N,) float64 pressure field in Pa at each mesh node.
    wss_vectors:
        (W, 3) float64 WSS vector [Pa] at each wall triangle centroid.
        Inlet/outlet faces have WSS = 0.
    wss_magnitudes:
        (W,) float64 WSS magnitude [Pa].
    wall_centroids_m:
        (W, 3) float64 wall-triangle centroid coordinates [m].
    wall_normals:
        (W, 3) float64 outward unit normals (copy from AnxploreMesh).
    boundary_parts:
        Face classification (wall / inlet / outlet masks).
    cfg:
        Solver configuration that produced this solution.
    re_number:
        Reynolds number Re = ρ U L / μ for the chosen inlet velocity.
    """

    case_id: str
    velocity_nodes: np.ndarray       # (N, 3)
    pressure_nodes: np.ndarray       # (N,)
    wss_vectors: np.ndarray          # (W, 3)
    wss_magnitudes: np.ndarray       # (W,)
    wall_centroids_m: np.ndarray     # (W, 3)
    wall_normals: np.ndarray         # (W, 3)
    boundary_parts: BoundaryParts
    cfg: StokesConfig
    re_number: float


# ─── Internal FEniCSx helpers ─────────────────────────────────────────────────


def _require_dolfinx() -> None:
    if not _DOLFINX_AVAILABLE:
        raise ImportError(
            "dolfinx (FEniCSx) is required for the Stokes solver but is not "
            "installed.  Run this script in a Linux / WSL environment:\n"
            "  sudo add-apt-repository ppa:fenics-packages/fenics\n"
            "  sudo apt install fenicsx\n"
            "Alternatively use: conda install -c conda-forge fenics-dolfinx mpich"
        )


def _create_dolfinx_mesh(anxplore_mesh: AnxploreMesh):
    """Convert an AnxploreMesh to a dolfinx Mesh (SI units, metres)."""
    _require_dolfinx()

    gdim = 3
    points_m = anxplore_mesh.points_m                  # (N, 3) float64
    cells = anxplore_mesh.tetra.astype(np.int32)       # (T, 4) int32

    coord_el = basix_element("Lagrange", "tetrahedron", 1, shape=(gdim,))
    domain = dmesh.create_mesh(
        MPI.COMM_WORLD,
        cells,
        coord_el,
        points_m,
    )
    return domain


def _mark_boundary_facets(domain, anxplore_mesh: AnxploreMesh, bp: BoundaryParts):
    """Return dolfinx MeshTags with MARKER_WALL / MARKER_INLET / MARKER_OUTLET."""
    _require_dolfinx()
    from scipy.spatial import cKDTree  # noqa: PLC0415

    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_entities(fdim)
    domain.topology.create_connectivity(fdim, tdim)

    boundary_facets = dmesh.exterior_facet_indices(domain.topology)

    # Build a KDTree on our pre-classified face centroids (in metres).
    pts_m = anxplore_mesh.points_m
    tri = anxplore_mesh.wall_tri
    our_centroids = (
        pts_m[tri[:, 0]] + pts_m[tri[:, 1]] + pts_m[tri[:, 2]]
    ) / 3.0                                            # (W, 3)

    our_markers = np.full(len(tri), MARKER_WALL, dtype=np.int32)
    our_markers[bp.inlet_mask] = MARKER_INLET
    our_markers[bp.outlet_mask] = MARKER_OUTLET

    kd = cKDTree(our_centroids)

    # Compute dolfinx boundary-facet centroids.
    fv_conn = domain.topology.connectivity(fdim, 0)
    coords = domain.geometry.x                        # (N, 3) float64

    n_bf = len(boundary_facets)
    dlx_centroids = np.empty((n_bf, 3), dtype=np.float64)
    for i, f in enumerate(boundary_facets):
        v = fv_conn.links(f)
        dlx_centroids[i] = coords[v].mean(axis=0)

    dists, nn_idx = kd.query(dlx_centroids)
    max_dist_mm = dists.max() * 1e3
    if max_dist_mm > 0.5:
        warnings.warn(
            f"Boundary facet matching: max distance {max_dist_mm:.4f} mm. "
            "Mesh reordering may have occurred.",
            stacklevel=2,
        )

    facet_markers = our_markers[nn_idx]
    return dmesh.meshtags(domain, fdim, boundary_facets, facet_markers)


def _extract_nodal_values(
    domain, anxplore_mesh: AnxploreMesh, u_sub, p_sub
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate P2 velocity and P1 pressure to the original mesh nodes.

    Returns
    -------
    velocity_nodes:
        (N, 3) float64 velocity [m/s] at mesh nodes in meshio/AnxploreMesh order.
    pressure_nodes:
        (N,) float64 pressure [Pa] at mesh nodes.
    """
    _require_dolfinx()

    gdim = 3
    P1_vel_el = basix_element("Lagrange", "tetrahedron", 1, shape=(gdim,))
    P1_vel = dfem.functionspace(domain, P1_vel_el)
    u_p1 = dfem.Function(P1_vel)
    u_p1.interpolate(u_sub)

    # dolfinx P1 DOFs match mesh vertices in serial (no MPI reordering).
    velocity_nodes = u_p1.x.array.reshape(-1, gdim).copy()

    # P1 pressure — DOFs already at mesh vertices.
    # dolfinx saddle-point form uses p_dolfinx = -p_physical; correct here.
    pressure_nodes = -p_sub.x.array.copy()

    return velocity_nodes, pressure_nodes


def _compute_wss(
    domain,
    anxplore_mesh: AnxploreMesh,
    u_sub,
    bp: BoundaryParts,
    mu_val: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute WSS at wall triangle centroids via a DG(0) velocity gradient.

    Algorithm
    ---------
    1. Interpolate grad(u_h) into a DG(0) tensor space (constant per tet).
    2. For each wall face centroid, evaluate the gradient in the adjacent cell
       by nudging the point epsilon = 1e-6 m inward along the face normal.
    3. Compute WSS = 2μ [D · n̂]_tangential, D = (∇u + ∇uᵀ)/2.
       (Pressure does not contribute to the tangential traction.)

    Returns
    -------
    wss_vectors:
        (W, 3) float64 WSS vector [Pa]; 0 at inlet/outlet faces.
    wss_magnitudes:
        (W,) float64 WSS magnitude [Pa].
    """
    _require_dolfinx()
    from dolfinx.geometry import (  # noqa: PLC0415
        bb_tree,
        compute_collisions_points,
        compute_colliding_cells,
    )

    # Interpolate ∇u into DG(0) (one 3×3 matrix per tet = constant gradient).
    DG0_el = basix_element("DG", "tetrahedron", 0, shape=(3, 3))
    DG0 = dfem.functionspace(domain, DG0_el)
    grad_expr = dfem.Expression(ufl.grad(u_sub), DG0.element.interpolation_points)
    grad_fn = dfem.Function(DG0)
    grad_fn.interpolate(grad_expr)

    # Wall face centroids and normals (exclude inlet/outlet).
    pts_m = anxplore_mesh.points_m
    tri = anxplore_mesh.wall_tri
    wall_idx = np.where(bp.wall_mask)[0]             # indices into wall_tri
    wall_tri_w = tri[wall_idx]
    wall_normals_w = anxplore_mesh.wall_normals[wall_idx]   # (W_w, 3)
    wall_centroids_w = (
        pts_m[wall_tri_w[:, 0]]
        + pts_m[wall_tri_w[:, 1]]
        + pts_m[wall_tri_w[:, 2]]
    ) / 3.0                                          # (W_w, 3)

    # Nudge 1 µm inward so points lie strictly inside their adjacent tet.
    eps = 1e-6
    eval_pts = wall_centroids_w - eps * wall_normals_w   # (W_w, 3)

    # Find containing cells via bounding-box tree.
    tree = bb_tree(domain, domain.topology.dim)
    cell_cands = compute_collisions_points(tree, eval_pts)
    colliding = compute_colliding_cells(domain, cell_cands, eval_pts)

    cells = np.array(
        [colliding.links(i)[0] if len(colliding.links(i)) > 0 else -1
         for i in range(len(eval_pts))],
        dtype=np.int32,
    )

    missing = int((cells < 0).sum())
    if missing > 0:
        warnings.warn(
            f"_compute_wss: {missing} wall points did not collide with any cell. "
            "Their WSS will be set to zero.",
            stacklevel=2,
        )

    valid = cells >= 0
    grad_vals = np.zeros((len(eval_pts), 9), dtype=np.float64)
    if valid.any():
        grad_vals[valid] = grad_fn.eval(eval_pts[valid], cells[valid])

    # Reshape to (W_w, 3, 3) — row index = row of ∇u (output component).
    grad_mats = grad_vals.reshape(-1, 3, 3)

    # Rate-of-strain D = (∇u + ∇uᵀ)/2
    D = 0.5 * (grad_mats + grad_mats.transpose(0, 2, 1))

    # Traction = 2μ D · n̂
    Dn = np.einsum("wij,wj->wi", D, wall_normals_w)       # (W_w, 3)
    traction = 2.0 * mu_val * Dn

    # WSS = tangential component of traction
    t_norm = np.einsum("wi,wi->w", traction, wall_normals_w)[:, np.newaxis]
    wss_w = traction - t_norm * wall_normals_w              # (W_w, 3)
    wss_mag_w = np.linalg.norm(wss_w, axis=1)

    # Assemble into full (W,) arrays (zeros at inlet/outlet).
    W_total = tri.shape[0]
    wss_vectors = np.zeros((W_total, 3), dtype=np.float64)
    wss_magnitudes = np.zeros(W_total, dtype=np.float64)
    wss_vectors[wall_idx] = wss_w
    wss_magnitudes[wall_idx] = wss_mag_w

    return wss_vectors, wss_magnitudes


# ─── WSS dolfinx-Function helpers (doc 06 §6, used for XDMF export) ─────────


def _compute_wss_fn_scalar(domain, u_h, facet_tags, mu: float):
    """DG-0 L2 projection of WSS magnitude onto wall facets (doc 06 §6).

    Returns a dolfinx Function on the DG-0 scalar space.  Only values at
    wall-adjacent cells are meaningful; interior cells receive zero.
    """
    _require_dolfinx()
    from dolfinx import default_real_type  # noqa: PLC0415

    n = ufl.FacetNormal(domain)
    D = 0.5 * (ufl.grad(u_h) + ufl.grad(u_h).T)
    traction = 2.0 * mu * ufl.dot(D, n)
    wss_vec = traction - ufl.dot(traction, n) * n
    wss_mag = ufl.sqrt(ufl.dot(wss_vec, wss_vec))

    DG0_el = basix_element("DG", domain.basix_cell(), degree=0,
                           dtype=default_real_type)
    S = dfem.functionspace(domain, DG0_el)
    v = ufl.TestFunction(S)
    u_ = ufl.TrialFunction(S)
    ds_wall = ufl.Measure("ds", domain=domain,
                          subdomain_data=facet_tags, subdomain_id=MARKER_WALL)
    prob = LinearProblem(
        ufl.inner(u_, v) * ds_wall,
        ufl.inner(wss_mag, v) * ds_wall,
        petsc_options_prefix="wss_scalar_",
        petsc_options={"ksp_type": "cg", "pc_type": "jacobi", "ksp_rtol": 1e-12},
    )
    return prob.solve()


def _compute_wss_fn_vector(domain, u_h, facet_tags, mu: float):
    """DG-0 L2 projection of WSS vector onto wall facets (doc 06 §6).

    Returns a dolfinx Function on the DG-0 vector space (dim 3).
    """
    _require_dolfinx()
    from dolfinx import default_real_type  # noqa: PLC0415

    n = ufl.FacetNormal(domain)
    D = 0.5 * (ufl.grad(u_h) + ufl.grad(u_h).T)
    traction = 2.0 * mu * ufl.dot(D, n)
    wss_vec = traction - ufl.dot(traction, n) * n

    DG0v_el = basix_element("DG", domain.basix_cell(), degree=0,
                            shape=(domain.geometry.dim,),
                            dtype=default_real_type)
    S3 = dfem.functionspace(domain, DG0v_el)
    v = ufl.TestFunction(S3)
    u_ = ufl.TrialFunction(S3)
    ds_wall = ufl.Measure("ds", domain=domain,
                          subdomain_data=facet_tags, subdomain_id=MARKER_WALL)
    prob = LinearProblem(
        ufl.inner(u_, v) * ds_wall,
        ufl.inner(wss_vec, v) * ds_wall,
        petsc_options_prefix="wss_vector_",
        petsc_options={"ksp_type": "cg", "pc_type": "jacobi", "ksp_rtol": 1e-12},
    )
    return prob.solve()


# ─── Public solver ────────────────────────────────────────────────────────────


def solve_stokes(
    mesh: AnxploreMesh,
    cfg: Optional[StokesConfig] = None,
    out_dir: Optional[pathlib.Path] = None,
) -> StokesSolution:
    """Solve steady Stokes on an AnXplore fluid mesh using FEniCSx P2-P1.

    Governing equations (non-dimensional form trained on; dimensional inputs):

        -∇p + μ Δu = 0
        ∇·u = 0

    Boundary conditions:

        u = Poiseuille(r)      on Γ_inlet  (Dirichlet, parabolic profile)
        u = 0                  on Γ_wall   (Dirichlet, no-slip)
        (μ∇u − pI)·n = 0      on Γ_outlet (natural / do-nothing)

    Parameters
    ----------
    mesh:
        Loaded AnXplore fluid mesh.
    cfg:
        Solver configuration.  Defaults to :class:`StokesConfig`.
    out_dir:
        If provided, write XDMF+HDF5 output files (velocity, pressure,
        wss_mag, wss_vec, mesh) plus metadata.json to this directory.
        Requires dolfinx (Linux/WSL only).  The .npz output written by
        :func:`~hemodyn_pinn.cfd.postprocess.save_solution` is separate.

    Returns
    -------
    StokesSolution
        Velocity, pressure, and WSS fields plus metadata.

    Raises
    ------
    ImportError
        If dolfinx is not available (Windows / missing FEniCSx install).
    """
    _require_dolfinx()

    if cfg is None:
        cfg = StokesConfig()

    bp = detect_open_faces(mesh)

    # ── Build dolfinx mesh ──────────────────────────────────────────────────
    domain = _create_dolfinx_mesh(mesh)

    # ── Mark boundary facets ────────────────────────────────────────────────
    facet_tags = _mark_boundary_facets(domain, mesh, bp)

    fdim = domain.topology.dim - 1

    # ── Taylor-Hood P2-P1 function spaces (separate for nested block system) ─
    gdim = 3
    V_el = basix_element("Lagrange", "tetrahedron", 2, shape=(gdim,))
    Q_el = basix_element("Lagrange", "tetrahedron", 1)
    V = dfem.functionspace(domain, V_el)
    Q = dfem.functionspace(domain, Q_el)

    # ── Variational form — nested block system (doc 06 §5) ──────────────────
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    p, q = ufl.TrialFunction(Q), ufl.TestFunction(Q)

    f = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type((0.0, 0.0, 0.0)))
    zero_q = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(0.0))

    a = [
        [cfg.mu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx, -p * ufl.div(v) * ufl.dx],
        [-q * ufl.div(u) * ufl.dx,                               None                      ],
    ]
    L = [ufl.inner(f, v) * ufl.dx, ufl.inner(zero_q, q) * ufl.dx]

    # Block-diagonal preconditioner: (1/μ) ∫ p q dx for the pressure block
    a_p11 = (1.0 / cfg.mu) * p * q * ufl.dx
    a_p = [[a[0][0], None], [None, a_p11]]

    # ── Dirichlet boundary conditions ────────────────────────────────────────
    # BCs go directly on V — no sub-space collapse needed.
    # Wall: u = 0
    wall_facets = facet_tags.find(MARKER_WALL)
    u_zero = dfem.Function(V)
    wall_dofs = dfem.locate_dofs_topological(V, fdim, wall_facets)
    bc_wall = dfem.dirichletbc(u_zero, wall_dofs)

    # Inlet: parabolic Poiseuille profile (doc 06 §4).
    # u_y(r) = U_max * max(0, 1 - r²/R²),  U_max = 2 * U_mean,
    # velocity direction (0, -1, 0) (inward at the y=0 cut).
    inlet_facets = facet_tags.find(MARKER_INLET)
    u_inlet_fn = dfem.Function(V)
    _R_sq = (2.0e-3) ** 2          # tube radius² in m² (R = 2 mm = 0.002 m)
    _U_max = 2.0 * cfg.u_inlet     # peak centreline velocity

    # Pre-compute inlet disk centroid from the boundary-part masks (metres).
    # dolfinx interpolate() passes ALL DOF coords to the callable, so computing
    # xc/zc from x.mean() inside the closure gives the global mesh centroid
    # (~0), not the inlet disk centre (~-6.14 mm in x).  That makes r² >> R²
    # at every inlet DOF and clamps the Poiseuille profile to zero, causing
    # u ≡ 0 everywhere and hence WSS ≡ 0.
    _inlet_centroids_m = mesh.wall_centroids_mm[bp.inlet_mask] * 1e-3
    _xc = float(_inlet_centroids_m[:, 0].mean())
    _zc = float(_inlet_centroids_m[:, 2].mean())

    def _inlet_profile(x: np.ndarray) -> np.ndarray:
        # x shape: (3, N_dof) — dolfinx passes all DOF coords at once
        r2 = (x[0] - _xc) ** 2 + (x[2] - _zc) ** 2
        speed = _U_max * np.maximum(0.0, 1.0 - r2 / _R_sq)
        vals = np.zeros((3, x.shape[1]), dtype=dolfinx.default_scalar_type)
        vals[1] = -speed   # negative y = into domain
        return vals

    u_inlet_fn.interpolate(_inlet_profile)
    inlet_dofs = dfem.locate_dofs_topological(V, fdim, inlet_facets)
    bc_inlet = dfem.dirichletbc(u_inlet_fn, inlet_dofs)

    bcs = [bc_wall, bc_inlet]

    # ── Solve: MINRES + block-diagonal preconditioner (doc 06 §5) ───────────
    problem = LinearProblem(
        a,
        L,
        kind="nest",
        bcs=bcs,
        P=a_p,
        petsc_options_prefix="stokes_",
        petsc_options={
            "ksp_type":                      "minres",
            "ksp_rtol":                      cfg.ksp_rtol,
            "ksp_max_it":                    cfg.ksp_max_it,
            "ksp_monitor":                   "",
            "ksp_norm_type":                 "unpreconditioned",
            "pc_type":                       "fieldsplit",
            "pc_fieldsplit_type":            "additive",
            "pc_fieldsplit_detect_saddle_point": "",   # auto-split MATNEST blocks
            "fieldsplit_0_ksp_type":         "preonly",
            "fieldsplit_0_pc_type":          "gamg",
            "fieldsplit_1_ksp_type":         "preonly",
            "fieldsplit_1_pc_type":          "jacobi",
        },
    )

    # Pressure nullspace: uniform constant over Q (Stokes pressure defined up to a constant)
    p_null = dfem.Function(Q)
    p_null.x.array[:] = 1.0
    p_null.x.scatter_forward()
    p_null.x.petsc_vec.scale(1.0 / p_null.x.petsc_vec.norm())
    u_null = dfem.Function(V)   # zero velocity component of null vector
    nested_null = _PETSc.Vec().createNest(
        [u_null.x.petsc_vec, p_null.x.petsc_vec], comm=domain.comm
    )
    nsp = _PETSc.NullSpace().create(vectors=[nested_null])
    problem.A.setNullSpace(nsp)

    # Mark SPD blocks so GAMG uses its optimal algebraic-multigrid path
    A00 = problem.A.getNestSubMatrix(0, 0)
    A00.setOption(_PETSc.Mat.Option.SPD, True)
    P00 = problem.P_mat.getNestSubMatrix(0, 0)
    P00.setOption(_PETSc.Mat.Option.SPD, True)
    P11 = problem.P_mat.getNestSubMatrix(1, 1)
    P11.setOption(_PETSc.Mat.Option.SPD, True)

    u_sub, p_sub = problem.solve()

    reason = problem.solver.getConvergedReason()
    if reason <= 0:
        raise RuntimeError(
            f"Stokes MINRES solver did not converge (PETSc reason {reason}). "
            "Increase ksp_max_it or relax ksp_rtol in StokesConfig."
        )

    # ── Nodal values ────────────────────────────────────────────────────────
    velocity_nodes, pressure_nodes = _extract_nodal_values(domain, mesh, u_sub, p_sub)

    # ── WSS ─────────────────────────────────────────────────────────────────
    wss_vectors, wss_magnitudes = _compute_wss(domain, mesh, u_sub, bp, cfg.mu)

    re = RHO_BLOOD * cfg.u_inlet * L_SCALE / cfg.mu

    # ── XDMF export (dolfinx objects still in scope) ────────────────────────
    if out_dir is not None:
        from hemodyn_pinn.cfd.postprocess import export_xdmf  # noqa: PLC0415
        wss_mag_fn = _compute_wss_fn_scalar(domain, u_sub, facet_tags, cfg.mu)
        wss_vec_fn = _compute_wss_fn_vector(domain, u_sub, facet_tags, cfg.mu)

        # dolfinx 0.10: write_function requires degree == mesh geometry degree (1).
        # u_sub is P2 — interpolate down to P1 for XDMF only.
        P1_vel_el = basix_element("Lagrange", "tetrahedron", 1, shape=(gdim,))
        P1_vel = dfem.functionspace(domain, P1_vel_el)
        u_p1 = dfem.Function(P1_vel)
        u_p1.interpolate(u_sub)

        export_xdmf(
            pathlib.Path(out_dir), domain, facet_tags,
            u_p1, p_sub, wss_mag_fn, wss_vec_fn,
            metadata={
                "case_id": mesh.case_id,
                "mu_Pa_s": cfg.mu,
                "rho_kg_m3": RHO_BLOOD,
                "U_mean_m_s": cfg.u_inlet,
                "Re": re,
                "L_char_m": L_SCALE,
                "inlet_bc": "poiseuille_parabolic",
                "pressure_convention": "physical (sign-corrected)",
                "dolfinx_version": "0.10",
            },
        )

    return StokesSolution(
        case_id=mesh.case_id,
        velocity_nodes=velocity_nodes,
        pressure_nodes=pressure_nodes,
        wss_vectors=wss_vectors,
        wss_magnitudes=wss_magnitudes,
        wall_centroids_m=mesh.wall_centroids_mm * 1e-3,
        wall_normals=mesh.wall_normals.copy(),
        boundary_parts=bp,
        cfg=cfg,
        re_number=re,
    )

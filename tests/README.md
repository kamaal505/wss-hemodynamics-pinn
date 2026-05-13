# tests/

Pytest test suite for the `hemodyn_pinn` package. All tests except those marked `slow` run on Windows without FEniCSx.

**Current status:** 116 tests pass (1 dolfinx test auto-skipped on Windows). Zero warnings.

---

## Running the tests

Activate the `.venv` first, then run from the repo root.

### Full suite

**Git Bash (Windows):**
```bash
.venv/Scripts/python -m pytest tests/ -v
```

**macOS / Linux:**
```bash
.venv/bin/python -m pytest tests/ -v
```

### Skip slow integration tests

Tests marked `slow` load real mesh files (~150k nodes) and take several seconds each. Skip them for fast feedback during development.

**Git Bash (Windows):**
```bash
.venv/Scripts/python -m pytest tests/ -v -m "not slow"
```

**macOS / Linux:**
```bash
.venv/bin/python -m pytest tests/ -v -m "not slow"
```

### Run a single test file

**Git Bash (Windows):**
```bash
.venv/Scripts/python -m pytest tests/test_losses.py -v
.venv/Scripts/python -m pytest tests/test_inference.py -v
.venv/Scripts/python -m pytest tests/test_mri_operator.py -v
.venv/Scripts/python -m pytest tests/test_anxplore_loader.py -v
.venv/Scripts/python -m pytest tests/test_stokes_solver.py -v
```

**macOS / Linux:**
```bash
.venv/bin/python -m pytest tests/test_losses.py -v
.venv/bin/python -m pytest tests/test_inference.py -v
.venv/bin/python -m pytest tests/test_mri_operator.py -v
.venv/bin/python -m pytest tests/test_anxplore_loader.py -v
.venv/bin/python -m pytest tests/test_stokes_solver.py -v
```

### Run with coverage

**Git Bash (Windows):**
```bash
.venv/Scripts/python -m pytest tests/ --cov=hemodyn_pinn --cov-report=term-missing
```

**macOS / Linux:**
```bash
.venv/bin/python -m pytest tests/ --cov=hemodyn_pinn --cov-report=term-missing
```

---

## Test files

### `test_anxplore_loader.py`

Covers `src/hemodyn_pinn/geometry/anxplore_loader.py`.

Tests:
- Mesh loading from VTK returns correct node and cell counts
- Wall boundary extraction (tet faces shared by exactly one tetrahedron)
- Inlet/outlet detection and left/right splitting at x=0
- Outward normal directions
- Handling of both named cases (tet-only) and `full_dataset` (explicit surface triangles)

Most tests use a small synthetic tetrahedral mesh. Tests marked `slow` load a real AnXplore VTK file (requires `data/geometries/anxplore/` to be populated by `scripts/fetch_and_inspect.sh`).

### `test_mri_operator.py` *(required by Phase 1.4)*

Covers `src/hemodyn_pinn/mri/` (voxelise, noise, operator).

26 tests including:
- `VoxelGrid` bounding box and cell index arithmetic
- `assign_nodes_to_voxels` correctness on synthetic grids
- Volume-averaging reduces to the exact mean for a uniform field
- Noise draws have correct mean (≈0) and std (≈σ)
- `sigma_from_vnr` formula: σ = VENC/VNR
- `SyntheticMRIOperator` round-trip on a constant velocity field
- Reproducibility: same seed → identical noise realisations

### `test_losses.py` *(required by Phase 1.5)*

Covers `src/hemodyn_pinn/pinn/losses.py`.

22 tests including:
- `data_loss` is zero when network perfectly predicts observations
- `stokes_residual_loss` is zero for an exact Stokes solution (verified analytically)
- `bc_loss` is zero when network outputs zero at wall points
- `pressure_anchor_loss` is zero when predicted pressure matches the anchor
- `total_loss` is a correctly weighted sum of components
- Gradient flow: all loss terms produce non-None `.grad` on network parameters

### `test_inference.py` *(required by Phase 1.5)*

Covers `src/hemodyn_pinn/pinn/inference.py`.

11 tests including:
- `compute_velocity_jacobian` returns a tensor of shape `(N, 3, 3)`
- Jacobian values match finite-difference estimates on a simple network
- `compute_wss` Couette flow check: WSS magnitude matches analytic value μ·du/dz
- Tangential projection removes the normal component (τ_w · n̂ ≈ 0)
- Dimensional scaling: non-dim network output correctly converted to Pa
- `predict_velocity_field` shape and dtype

### `test_stokes_solver.py`

Covers `src/hemodyn_pinn/cfd/stokes_solver.py`.

Contains fast unit tests (BCs, mesh assembly, output format) that run without FEniCSx by mocking `dolfinx`, plus one integration test marked `slow` and `dolfinx` that actually solves the Stokes system. The integration test is automatically skipped if `dolfinx` is not importable (e.g., on Windows).

---

## Adding new tests

- Place test files in `tests/` with the `test_` prefix.
- Mark tests that load real mesh files with `@pytest.mark.slow`.
- Mark tests that require `dolfinx` with `@pytest.mark.skipif(not HAS_DOLFINX, reason="dolfinx not installed")`.
- Use `src/hemodyn_pinn/utils/seeds.py` for any RNG — never hard-code seeds.
- Mandatory test files for each phase are listed in `CLAUDE.md §9`.

# tests/

Pytest test suite for the `hemodyn_pinn` package. All tests except those marked `slow` run on Windows without FEniCSx.

**Current status:** 262 tests collected; all pass on Windows (1 dolfinx integration test auto-skipped without FEniCSx).

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

Includes:
- `data_loss` / `inlet_loss` zero when prediction is exact; relative form ≈ 1 for a zero prediction (the collapse signature)
- `aux_data_loss` (dense interpolant supervision) zero when exact, ≈ 1 for zero prediction
- `magnitude_floor_loss` one-sided: zero when RMS meets the floor, positive below
- `stokes_residual_loss` zero for an exact Stokes solution (verified analytically); `skip_div_loss` variant
- `bc_loss` / `pressure_anchor_loss` zero in the respective trivial cases
- `total_loss` is a correctly weighted sum; inlet/aux/magnitude-floor terms add positive contributions when weighted
- `SelfAdaptiveLoss` (SA-PINN) weights and gradient reversal
- Gradient flow: all loss terms produce non-None `.grad` on network parameters

### `test_interpolant.py`

Covers `src/hemodyn_pinn/pinn/interpolant.py` (dense MRI interpolant prior).

- `build_velocity_interpolant` (rbf / linear) reproduces the sample voxels and returns finite values at arbitrary queries
- `dense_aux_targets` injects zero velocity at the wall and fills out-of-hull queries (no NaNs)
- Unknown method raises `ValueError`

### `test_trainer.py`

Covers `src/hemodyn_pinn/pinn/trainer.py` init and curriculum.

- Architecture flags: wall-bias subset, hard-SDF precompute, vec-potential `skip_div`, SA-PINN init
- `TestCurriculumAux`: `target_rms` precompute, aux flags, decay schedule (`lambda_aux` → 0), warm-up zeroes physics, short `run_adam` completes

### `test_bhpo.py`

Covers `src/hemodyn_pinn/bhpo/` (space, objective, search) — 11-dim search decode/suggest, val-split, a short trial smoke, and **trial-failure robustness**: OOM is retried with smaller chunks (`status=ok`), persistent OOM / genuine errors are classified and recorded (`status=oom`/`error`) rather than silently abandoned, and the trial log writes a `status` column.

### `test_inference.py` (batched WSS)

Beyond the WSS-autograd checks: `compute_wss_batched` is bit-equivalent to the unbatched path, processes all wall faces, halves the batch and recovers on a simulated OOM, and re-raises only below the batch floor; `_is_oom_error` classifies CUDA/MPS OOM vs other errors.

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

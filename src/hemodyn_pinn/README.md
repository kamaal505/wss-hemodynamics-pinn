# src/hemodyn_pinn/

The `hemodyn_pinn` Python package. Install with `uv pip install -e .` from the repo root. All pipeline scripts import from this package.

---

## Package structure

```
hemodyn_pinn/
├── geometry/        # AnXplore mesh loading and boundary extraction
├── cfd/             # Stokes/NS solver and postprocessing (FEniCSx)
├── mri/             # Synthetic MRI forward operator (voxelisation + noise)
├── pinn/            # PINN architecture, losses, training, and inference
├── baselines/       # Comparison methods (trilinear interp + FD, supervised MLP)
├── eval/            # Evaluation metrics (NRMSE, R², Bland–Altman)
├── viz/             # Figure generation (one script per paper figure)
├── utils/           # Shared utilities (RNG seeds, I/O, logging)
└── bhpo/            # Bayesian HPO module — Phase 3, not yet active
```

---

## geometry/

Loading and preprocessing of AnXplore VTK mesh files.

### `anxplore_loader.py`

Loads a single AnXplore case from a `.vtk` file and extracts the wall boundary, inlet face, and outlet face.

Key classes and functions:

| Name | Description |
|------|-------------|
| `AnXploreCase` | Dataclass holding mesh nodes, tets, wall/inlet/outlet face indices, and outward normals |
| `load_case(vtk_path)` | Load a VTK file and return an `AnXploreCase` |
| `extract_boundary_faces(cells)` | Extract wall surface as tet faces shared by exactly one tetrahedron |
| `detect_inlet_outlet(faces, normals)` | Split the open faces at y-min into inlet (x < 0) and outlet (x > 0) |

**Usage:**
```python
from hemodyn_pinn.geometry.anxplore_loader import load_case
case = load_case("data/geometries/anxplore/caseC/Fluid_caseC.vtk")
# case.wall_nodes     — shape (N_w, 3) in mm
# case.inlet_nodes    — shape (N_in, 3) in mm
# case.outlet_nodes   — shape (N_out, 3) in mm
```

> **Important:** Mesh coordinates are in **millimetres**. Convert to metres (`× 1e-3`) before any physical computation.

---

## cfd/

Steady Stokes CFD solver using FEniCSx (dolfinx). **FEniCSx must be installed separately — it cannot be pip-installed. See the root [README](../../README.md#installation).**

### `stokes_solver.py`

| Name | Description |
|------|-------------|
| `solve_stokes(mesh, cfg, out_dir)` | Assemble and solve the steady Stokes system; returns nodal velocity, pressure, and WSS arrays |
| `_build_inlet_bc(V, inlet_dofs, u_inlet)` | Poiseuille parabolic inlet profile via collapsed-subspace interpolation |
| `_compute_wss_fn_scalar/vector` | DG-0 L² projection of WSS magnitude and vector onto the wall surface |

Physical parameters (from `configs/cfd/stokes_default.yaml`):
- `mu = 3.5e-3` Pa·s (blood viscosity)
- `u_inlet = 2e-5` m/s → Re ≈ 0.10 (Stokes regime confirmed)

Pressure sign convention: `dolfinx` returns `-p`; `_extract_nodal_values` negates it to give physical pressure.

### `postprocess.py`

| Name | Description |
|------|-------------|
| `export_xdmf(out_dir, u_fn, p_fn, ...)` | Write velocity, pressure, WSS to XDMF+HDF5 for ParaView |
| `load_cfd_xdmf(out_dir)` | Read back a previously saved XDMF solution |
| `compute_wss_pointwise(u_grad, normals, mu)` | Compute WSS from velocity Jacobian and wall normals |

---

## mri/

Synthetic 4D-flow MRI forward operator. No FEniCSx dependency — runs on all platforms.

The forward model is:

```
u_MRI(x_v) = (1/|V_v|) ∫_{V_v} χ_Ω(x) u_CFD(x) dV  +  η,    η ~ N(0, σ² I)
```

### `voxelise.py`

| Name | Description |
|------|-------------|
| `VoxelGrid` | Defines the voxel lattice over the mesh bounding box |
| `assign_nodes_to_voxels(points, grid)` | Map each mesh node to its voxel index |
| `average_field_over_voxels(field, assignments, grid)` | Volume-average a nodal field to produce voxelised observations |

### `noise.py`

| Name | Description |
|------|-------------|
| `add_gaussian_noise(signal, sigma, rng)` | Add i.i.d. Gaussian noise to voxelised velocity |
| `sigma_from_vnr(venc, vnr)` | Compute noise std from velocity encoding and velocity-to-noise ratio: σ = VENC/VNR |

### `operator.py`

| Name | Description |
|------|-------------|
| `SyntheticMRIOperator` | Full forward operator: voxelise + noise. Wraps `VoxelGrid`, `average_field_over_voxels`, and `add_gaussian_noise` |
| `MRIObservation` | Dataclass: voxel centres, velocity components, sigma, config |
| `MRIConfig` | Config dataclass matching `configs/mri/*.yaml` |

**Usage:**
```python
from hemodyn_pinn.mri.operator import SyntheticMRIOperator, MRIConfig
cfg = MRIConfig(voxel_size_mm=1.0, sigma=0.0, rng_seed=1701)
op = SyntheticMRIOperator(cfg)
obs = op.apply(mesh_nodes_mm, velocity_field)
# obs.centres    — voxel centres (m)
# obs.u, obs.v, obs.w — voxelised velocity components (m/s)
```

---

## pinn/

Physics-informed neural network: architecture, loss functions, training loop, and WSS inference. PyTorch 2.x throughout. No FEniCSx dependency.

### `networks.py`

| Name | Description |
|------|-------------|
| `MLP` | Coordinate-based MLP. Input: x̂=(x,y,z)/L ∈ ℝ³. Output: (û,v̂,ŵ,p̂). Tanh activations throughout (never ReLU — breaks second-order autograd). |
| `RFFEncoder` | Random Fourier Features encoder: φ(x) = [sin(2πBx), cos(2πBx)], B_ij ~ N(0, σ²). |
| `PINNNetwork` | Wraps MLP (+ optional RFFEncoder). Handles non-dimensionalisation. |

### `losses.py`

| Name | Description |
|------|-------------|
| `data_loss(network, obs_pts, obs_vel)` | MSE between network velocity prediction and MRI observations |
| `stokes_residual_loss(network, colloc_pts, mu, L, U)` | Stokes momentum residual via 2nd-order autograd Laplacian |
| `bc_loss(network, wall_pts)` | No-slip: ‖u_θ(x_w)‖² at wall collocation points |
| `pressure_anchor_loss(network, anchor_pt)` | Fix p=0 at one outlet point (prevents pressure drift) |
| `total_loss(network, ..., lambdas)` | Weighted sum: λ_data·L_data + λ_phys·L_phys + λ_bc·L_bc + λ_anchor·L_anchor |

### `sampling.py`

| Name | Description |
|------|-------------|
| `sample_collocation(interior_pts, n, rng)` | Sample n collocation points from the mesh interior each epoch |
| `sample_wall_bc(wall_pts, n, rng)` | Sample n wall boundary points for the no-slip loss |
| `make_pressure_anchor(outlet_pts)` | Select one outlet point as the pressure anchor |
| `epoch_rng(base_seed, epoch)` | Deterministic per-epoch RNG for reproducible collocation resampling |

### `trainer.py`

| Name | Description |
|------|-------------|
| `PINNConfig` | Dataclass matching `configs/model/pinn_base.yaml` training section |
| `PINNTrainer` | Full training loop: Adam (cosine LR 1e-3→1e-6) → L-BFGS (strong Wolfe). Checkpointing every N epochs; best-state restoration. |

**Usage:**
```python
from hemodyn_pinn.pinn.networks import PINNNetwork
from hemodyn_pinn.pinn.trainer import PINNTrainer, PINNConfig

net = PINNNetwork(n_hidden=128, n_layers=4, use_rff=False)
cfg = PINNConfig(n_adam=50000, n_lbfgs=5000, lr_adam=1e-3)
trainer = PINNTrainer(net, cfg, case, obs, out_dir)
trainer.train()
```

### `inference.py`

WSS recovery from a trained network via automatic differentiation.

| Name | Description |
|------|-------------|
| `compute_velocity_jacobian(network, pts)` | ∂(u,v,w)/∂(x,y,z) at a batch of points via `torch.autograd.grad` |
| `compute_wss(network, wall_pts, normals, mu, L, U)` | Full WSS pipeline: Jacobian → strain-rate tensor → tangential traction |
| `predict_velocity_field(network, pts, L, U)` | Evaluate network at arbitrary points and return dimensional velocity |

WSS formula (no finite differences):
```
D_θ(x_w) = ½(∇u_θ + (∇u_θ)ᵀ)|_{x_w}
τ_w = 2μ [D_θ · n̂]_tangential
```

---

## utils/

### `seeds.py`

Centralised RNG seed registry. **Never hard-code seeds elsewhere in the codebase — always import from here.**

| Constant | Value | Used for |
|----------|-------|----------|
| `MRI_NOISE_SEED` | 1701 | MRI noise draws |
| `RFF_SEED` | 2718 | Random Fourier Feature B matrix |
| `PINN_COLLOC_SEED` | 3141 | Base seed for collocation resampling |
| `PINN_TRAIN_SEED` | 9999 | Network weight initialisation |

---

## baselines/ *(stub — Phase 2)*

Comparison methods against which the PINN is benchmarked:

| Module | Method |
|--------|--------|
| `trilinear_fd.py` | Trilinear interpolation of MRI voxels to wall nodes + finite-difference WSS |
| `supervised_mlp.py` | Data-only MLP trained on MRI observations (no physics loss) |

---

## eval/ *(stub — Phase 2)*

Evaluation metrics for comparing PINN WSS predictions against CFD ground truth:

| Module | Metrics |
|--------|---------|
| `field_metrics.py` | NRMSE, R², MAE over the full velocity/WSS field |
| `wss_metrics.py` | Pointwise relative error, peak WSS error, spatial correlation |
| `conservation.py` | Volumetric flow-rate Q(z) constancy, bifurcation balance |
| `bland_altman.py` | Bland–Altman plot data and 95% limits of agreement |

---

## viz/ *(stub — Phase 2)*

Figure generation. **One Python script per paper figure.** See `docs/project_knowledge/04_research_artefacts_and_figures.tex` for the full figure plan.

- No jet colormaps. Use `viridis`/`magma`/`cividis` (sequential) or `RdBu_r` (diverging).
- Output format: vector PDF or 600 dpi PNG with labelled scalar bars.

| Module | Purpose |
|--------|---------|
| `style.py` | Global matplotlib rcParams and PyVista theme |
| `render_3d.py` | PyVista 3D surface renders of WSS fields |
| `plots_2d.py` | 2D scatter, error-map, and Bland–Altman helpers |
| `make_figure_N.py` | One script per paper figure (N = 1…12) |

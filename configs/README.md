# configs/

Hydra configuration files for the hemodyn-pinn pipeline. Every pipeline script (`scripts/02_*.py` through `scripts/05_*.py`) is a Hydra app that composes its configuration from these files.

The top-level entry point is [`config.yaml`](config.yaml). Sub-group directories correspond to Hydra config groups; a script selects one config per group.

---

## Directory structure

```
configs/
├── config.yaml          # top-level defaults — references one file from each group
├── geometry/            # one file per AnXplore case
│   ├── caseA.yaml
│   ├── caseB.yaml
│   ├── caseC.yaml       # simplest geometry — recommended starting point
│   └── caseR.yaml
├── cfd/                 # Stokes / NS solver parameters
│   └── stokes_default.yaml
├── mri/                 # synthetic MRI voxel-size configs
│   ├── voxel_0p5mm.yaml
│   ├── voxel_1p0mm.yaml  # default
│   ├── voxel_1p5mm.yaml
│   ├── voxel_2p0mm.yaml
│   └── voxel_2p5mm.yaml
└── model/               # PINN architecture + training hyperparameters
    ├── pinn_base.yaml   # plain MLP, no RFF
    └── pinn_rff.yaml    # MLP with Random Fourier Features
```

---

## config.yaml — top-level defaults

```yaml
defaults:
  - geometry: caseA
  - cfd: stokes_default
  - mri: voxel_1p0mm
  - model: pinn_base
```

These defaults are overridden per-run on the command line (see below).

---

## geometry/

One file per AnXplore case. Specifies the path to the VTK mesh file.

| File | `case_id` | z-span (mm) | Description |
|------|-----------|-------------|-------------|
| `caseC.yaml` | caseC | 4.05 | Shallowest aneurysm — simplest geometry, start here |
| `caseA.yaml` | caseA | 4.61 | Moderate complexity |
| `caseR.yaml` | caseR | 6.22 | More complex |
| `caseB.yaml` | caseB | 6.39 | Deepest aneurysm — most complex geometry |

**Key fields:**
```yaml
case_id: caseC
vtk_path: data/geometries/anxplore/caseC/Fluid_caseC.vtk
```

---

## cfd/stokes_default.yaml

Parameters for the steady Stokes solver (`scripts/02_run_cfd.py`).

| Key | Value | Description |
|-----|-------|-------------|
| `mu` | 3.5e-3 | Blood viscosity (Pa·s) |
| `u_inlet` | 2.0e-5 | Inlet normal velocity (m/s) → Re ≈ 0.10 |
| `petsc_ksp_type` | preonly | PETSc Krylov solver type |
| `petsc_pc_type` | lu | Preconditioner (direct) |
| `petsc_pc_solver` | mumps | MUMPS sparse direct solver |
| `save_vtk` | false | Also write a `.vtu` file alongside XDMF |

Override on the command line:
```bash
# Git Bash / macOS / Linux
python scripts/02_run_cfd.py geometry=caseC cfd.u_inlet=1e-5 cfd.save_vtk=true
```

---

## mri/

Each file configures one voxel-size operating point for `scripts/03_generate_synthetic_mri.py`.

| File | `voxel_size_mm` | Notes |
|------|-----------------|-------|
| `voxel_0p5mm.yaml` | 0.5 | Finest — closest to real 4D-flow MRI |
| `voxel_1p0mm.yaml` | 1.0 | Default |
| `voxel_1p5mm.yaml` | 1.5 | |
| `voxel_2p0mm.yaml` | 2.0 | |
| `voxel_2p5mm.yaml` | 2.5 | Coarsest — most information loss |

**Key fields (all files):**
```yaml
voxel_size_mm: 1.0
sigma: 0.0           # Gaussian noise std (m/s); set to VENC/VNR for noisy runs
rng_seed: 1701       # reproducibility seed for noise draws
min_nodes_per_voxel: 1
```

Override noise level on the command line:
```bash
# Git Bash / macOS / Linux
python scripts/03_generate_synthetic_mri.py geometry=caseC mri=voxel_1p0mm mri.sigma=2e-6
```

---

## model/

Configures the PINN architecture and all training hyperparameters for `scripts/04_train_pinn.py`.

### pinn_base.yaml — plain MLP

| Key | Default | Description |
|-----|---------|-------------|
| `model.n_hidden` | 128 | Hidden units per layer |
| `model.n_layers` | 4 | Number of hidden layers |
| `model.use_rff` | false | No Random Fourier Features |
| `training.n_colloc` | 10000 | Collocation points per epoch |
| `training.n_wall_bc` | 2000 | Wall BC points per epoch |
| `training.n_adam` | 50000 | Adam iterations |
| `training.n_lbfgs` | 5000 | L-BFGS max iterations |
| `training.lr_adam` | 1e-3 | Adam learning rate |
| `training.lambda_data` | 1.0 | MRI data loss weight |
| `training.lambda_phys` | 1.0 | Stokes residual loss weight |
| `training.lambda_bc` | 10.0 | No-slip BC loss weight |
| `training.lambda_anchor` | 10.0 | Pressure anchor loss weight |
| `training.checkpoint_every` | 1000 | Save checkpoint every N Adam steps |
| `training.device` | `"cpu"` | Set to `"cuda"` for GPU |

### pinn_rff.yaml — MLP with Random Fourier Features

Extends `pinn_base.yaml` with:

| Key | Default | Description |
|-----|---------|-------------|
| `model.use_rff` | true | Prepend RFF encoder |
| `model.rff_features` | 128 | RFF output dimension |
| `model.rff_sigma` | 1.0 | Gaussian bandwidth for B matrix |

---

## Overriding config values

Any config key can be overridden from the command line using dot notation:

```bash
# Git Bash (Windows)
python scripts/04_train_pinn.py geometry=caseC model=pinn_rff \
    training.n_adam=20000 training.device=cuda model.rff_sigma=2.0

# macOS / Linux
python scripts/04_train_pinn.py geometry=caseC model=pinn_rff \
    training.n_adam=20000 training.device=cuda model.rff_sigma=2.0
```

## Multirun sweeps

Use `--multirun` to launch a grid over multiple config values:

```bash
# Git Bash / macOS / Linux
python scripts/03_generate_synthetic_mri.py --multirun \
    geometry=caseA,caseB,caseC,caseR \
    mri=voxel_0p5mm,voxel_1p0mm,voxel_1p5mm,voxel_2p0mm,voxel_2p5mm
```

Hydra writes per-run outputs to `outputs/` (local, not committed).

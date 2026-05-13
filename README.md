# hemodyn-pinn

**PINN-based Wall Shear Stress Recovery from Sparse, Noisy 4D-flow MRI**

Given sparse, noisy velocity samples from synthetic 4D-flow MRI (downsampled from CFD), this project studies how a physics-informed neural network (PINN) recovers the wall shear stress (WSS) field, and how recovery quality depends on (i) MRI voxel size, (ii) noise level, and (iii) aneurysm geometry.

**Geometry source:** [AnXplore](https://github.com/aurelegoetz/AnXplore) — 101 patient-derived intracranial aneurysm geometries (Goetz et al. 2024, MIT licensed).  
**Flow regime:** Steady incompressible Stokes (Re ≈ 0.10).  
**CFD solver:** FEniCSx (dolfinx 0.10), Stokes with Poiseuille inlet.  
**PINN:** PyTorch 2.x, tanh MLP with optional Random Fourier Features.

---

## Repository layout

```
hemodyn-pinn/
├── configs/              # Hydra configuration files (geometry, cfd, mri, model)
├── data/                 # local only — not committed (see §Data)
├── docs/project_knowledge/   # authoritative LaTeX reference documents
├── notebooks/            # exploratory Jupyter notebooks
├── paper/                # manuscript source (LaTeX)
├── scripts/              # numbered pipeline entry points — run in order
├── src/hemodyn_pinn/     # importable Python package
└── tests/                # pytest test suite (116 tests, all non-FEniCSx run on Windows)
```

See [`configs/README.md`](configs/README.md), [`scripts/README.md`](scripts/README.md), [`src/hemodyn_pinn/README.md`](src/hemodyn_pinn/README.md), and [`tests/README.md`](tests/README.md) for folder-level details.

---

## Installation

### Requirements

- Python ≥ 3.11
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- FEniCSx (CFD steps only; Linux / WSL required — see below)

### 1. Clone and create the virtual environment

**Git Bash (Windows):**
```bash
git clone <repo-url> wss-hemodynamics-pinn
cd wss-hemodynamics-pinn
python -m venv .venv
source .venv/Scripts/activate
pip install uv
uv pip install -r requirements.txt
uv pip install -e .
```

**macOS / Linux:**
```bash
git clone <repo-url> wss-hemodynamics-pinn
cd wss-hemodynamics-pinn
python3 -m venv .venv
source .venv/bin/activate
pip install uv
uv pip install -r requirements.txt
uv pip install -e .
```

### 2. FEniCSx — CFD solver (Linux / WSL only)

`dolfinx` has compiled C++ dependencies (PETSc, MPI) that cannot be installed via pip. Install it **before** running CFD scripts.

**Ubuntu / Debian (or WSL on Windows):**
```bash
sudo add-apt-repository ppa:fenics-packages/fenics
sudo apt update && sudo apt install -y fenicsx python3-pip python3-scipy
pip3 install meshio hydra-core omegaconf
```

**macOS (conda-forge):**
```bash
conda install -c conda-forge fenics-dolfinx mpich
pip install meshio hydra-core omegaconf
```

> **Windows:** Scripts 01 (`fetch_and_inspect.sh`) and 02 (`02_run_cfd.py`) must run under WSL. All other scripts (03, 04, 05) run natively on Windows. WSL mounts your Windows drive at `/mnt/<letter>/`, so the repo at `D:\Math\CFD\wss-hemodynamics-pinn` is accessible as `/mnt/d/Math/CFD/wss-hemodynamics-pinn` inside WSL.

---

## Data

All data is generated or fetched locally and is **not committed** to the repository. Directories are created automatically by the pipeline scripts.

| Path | Contents | Produced by |
|------|----------|-------------|
| `data/geometries/anxplore/` | AnXplore VTK meshes (cloned) | Step 0 |
| `data/cfd/<case_id>/` | `solution.npz`, `metadata.json`, XDMF fields | Step 2 |
| `data/synthetic_mri/<case_id>/<voxel>/` | `mri_obs.npz` | Step 3 |
| `data/checkpoints/<case_id>/<run_id>/` | `best_model.pt`, `loss_history.npy` | Step 4 |
| `data/results/` | Evaluation outputs and figures | Step 5 |

---

## Pipeline

Run scripts in the numbered order. Steps 0–2 require Linux/WSL for FEniCSx; steps 3–5 run on any OS.

### Step 0 — Fetch AnXplore geometries

**Git Bash / macOS / Linux:**
```bash
bash scripts/fetch_and_inspect.sh
```

Clones the AnXplore repository into `data/geometries/anxplore/` and writes mesh statistics to `data/geometries/anxplore/mesh_stats.txt` and `mesh_stats_per_case.csv`.

Quick sanity check on the first 5 cases only:
```bash
MAX_CASES=5 bash scripts/fetch_and_inspect.sh
```

### Step 2 — Run Stokes CFD *(Linux / WSL only)*

Activate the `.venv` first (or use a WSL Python that has `dolfinx` installed).

```bash
# Single case (caseC is the simplest geometry — start here)
python scripts/02_run_cfd.py geometry=caseC

# Override inlet velocity
python scripts/02_run_cfd.py geometry=caseC cfd.u_inlet=1e-5

# All four named cases sequentially
for case in caseA caseB caseC caseR; do
    python scripts/02_run_cfd.py geometry=$case
done

# Hydra multirun (runs all cases in sequence)
python scripts/02_run_cfd.py --multirun geometry=caseA,caseB,caseC,caseR
```

**From WSL on Windows:**
```bash
cd /mnt/d/Math/CFD/wss-hemodynamics-pinn
python scripts/02_run_cfd.py geometry=caseC
```

Output per case: `data/cfd/<case_id>/solution.npz` and `data/cfd/<case_id>/metadata.json`.

### Step 3 — Generate synthetic MRI *(any OS)*

Requires `data/cfd/<case_id>/solution.npz` from Step 2.

**Git Bash (Windows) / macOS / Linux:**
```bash
# Default: 1.0 mm voxels, noiseless, caseA
python scripts/03_generate_synthetic_mri.py geometry=caseA

# 0.5 mm voxels with Gaussian noise (sigma = VENC/VNR)
python scripts/03_generate_synthetic_mri.py \
    geometry=caseC mri=voxel_0p5mm mri.sigma=4e-6

# All voxel sizes for one case
python scripts/03_generate_synthetic_mri.py --multirun \
    geometry=caseC \
    mri=voxel_0p5mm,voxel_1p0mm,voxel_1p5mm,voxel_2p0mm,voxel_2p5mm

# Full sweep: all cases × all voxel sizes
python scripts/03_generate_synthetic_mri.py --multirun \
    geometry=caseA,caseB,caseC,caseR \
    mri=voxel_0p5mm,voxel_1p0mm,voxel_1p5mm,voxel_2p0mm,voxel_2p5mm
```

Output: `data/synthetic_mri/<case_id>/<voxel_config>/mri_obs.npz`.

### Step 4 — Train PINN *(any OS)*

Requires `data/cfd/<case_id>/solution.npz` and `data/synthetic_mri/...` from Steps 2–3.

**Git Bash (Windows) / macOS / Linux:**
```bash
# Plain MLP on caseC (recommended first run)
python scripts/04_train_pinn.py geometry=caseC model=pinn_base

# Random Fourier Features variant
python scripts/04_train_pinn.py geometry=caseC model=pinn_rff

# GPU + fewer Adam iterations (quick test)
python scripts/04_train_pinn.py geometry=caseC \
    training.device=cuda training.n_adam=10000

# Override loss weights
python scripts/04_train_pinn.py geometry=caseC \
    training.lambda_phys=10.0 training.lambda_bc=100.0
```

Output: `data/checkpoints/<case_id>/<run_id>/best_model.pt` and `loss_history.npy`.

---

## Testing

Tests cover the geometry loader, MRI operator, PINN losses, and inference. All 116 tests run on Windows without FEniCSx (1 dolfinx test is auto-skipped).

**Git Bash (Windows):**
```bash
# Full suite
.venv/Scripts/python -m pytest tests/ -v

# Skip slow integration tests (those that load real mesh files)
.venv/Scripts/python -m pytest tests/ -v -m "not slow"

# Single test file
.venv/Scripts/python -m pytest tests/test_losses.py -v
```

**macOS / Linux:**
```bash
# Full suite
.venv/bin/python -m pytest tests/ -v

# Skip slow integration tests
.venv/bin/python -m pytest tests/ -v -m "not slow"

# Single test file
.venv/bin/python -m pytest tests/test_losses.py -v
```

---

## Physical parameters

| Quantity | Value | Notes |
|----------|-------|-------|
| Blood density ρ | 1060 kg/m³ | Newtonian estimate |
| Blood viscosity μ | 3.5×10⁻³ Pa·s | Newtonian estimate |
| Inlet velocity U | 2×10⁻⁵ m/s | Re ≈ 0.10 (Stokes regime) |
| Length scale L | 16.28 mm | Parent artery diameter (x-span) |
| MRI voxel sweep | 0.5–2.5 mm | 5 configs |
| VNR sweep | 5–30 | σ = VENC/VNR |

Mesh coordinates are in **millimetres**. Convert to metres before any physical computation: `x_SI = x_mm × 1e-3`.

---

## Key references

- Goetz et al. (2024). AnXplore: a comprehensive FSI study of 101 intracranial aneurysms. *Front. Bioeng. Biotechnol.* https://doi.org/10.3389/fbioe.2024.1433811
- Tancik et al. (2020). Fourier features let networks learn high-frequency functions in low-dimensional domains. *NeurIPS.* https://arxiv.org/abs/2006.10739
- Wang, Wang, Perdikaris (2021). On the eigenvector bias of Fourier feature networks. *CMAME.* https://arxiv.org/abs/2012.10047
- Karniadakis et al. (2021). Physics-informed machine learning. *Nat. Rev. Phys.* https://doi.org/10.1038/s42254-021-00314-5
- Yang, Meng, Karniadakis (2021). B-PINNs. *JCP.* https://arxiv.org/abs/2003.06097

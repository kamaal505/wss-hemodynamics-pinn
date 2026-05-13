# scripts/

Numbered pipeline entry points. Run them in order. Each script is a self-contained Hydra application; all configuration comes from [`configs/`](../configs/).

**OS requirements at a glance:**

| Script | Windows (native) | macOS / Linux | WSL |
|--------|:---:|:---:|:---:|
| `fetch_and_inspect.sh` | Git Bash ✓ | ✓ | ✓ |
| `02_run_cfd.py` | ✗ (needs FEniCSx) | ✓ | ✓ |
| `03_generate_synthetic_mri.py` | ✓ | ✓ | ✓ |
| `04_train_pinn.py` | ✓ | ✓ | ✓ |

---

## fetch_and_inspect.sh

**What it does:** Clones the AnXplore dataset from GitHub into `data/geometries/anxplore/`, then runs a mesh inspection script that computes per-case statistics (node count, bounding box, edge-length distribution) and writes:

- `data/geometries/anxplore/inventory.tsv` — one row per mesh file
- `data/geometries/anxplore/mesh_stats.txt` — human-readable cohort report
- `data/geometries/anxplore/mesh_stats_per_case.csv` — one row per case (load with pandas)

**Requirements:** `git`, internet access, Python ≥ 3.11 with `meshio` and `numpy` installed in `.venv`. Approx. 200 MB disk space for the full clone.

```bash
# Git Bash (Windows) — from repo root with .venv activated
bash scripts/fetch_and_inspect.sh

# macOS / Linux
bash scripts/fetch_and_inspect.sh

# Quick sanity check: inspect first 5 cases only
MAX_CASES=5 bash scripts/fetch_and_inspect.sh
```

---

## 02_run_cfd.py

**What it does:** Runs the steady Stokes CFD solver (FEniCSx / dolfinx) on one AnXplore case. Applies a Poiseuille parabolic inlet profile and no-slip walls. Outputs velocity, pressure, and WSS fields.

**Requires:** FEniCSx installed in the active Python environment (Linux / WSL only). See [Installation](../README.md#installation) in the root README.

**Output per case:**
- `data/cfd/<case_id>/solution.npz` — velocity (u, v, w), pressure, WSS magnitude and vector at all mesh nodes
- `data/cfd/<case_id>/metadata.json` — solver configuration and scalar summary statistics
- `data/cfd/<case_id>/solution.xdmf` + `.h5` — ParaView-compatible field export (optional)

```bash
# macOS / Linux (with FEniCSx installed)

# Single case — caseC is the simplest geometry; start here
python scripts/02_run_cfd.py geometry=caseC

# Override inlet velocity
python scripts/02_run_cfd.py geometry=caseC cfd.u_inlet=1e-5

# Also write a .vtu file for visualisation in ParaView
python scripts/02_run_cfd.py geometry=caseC cfd.save_vtk=true

# All four named cases sequentially
for case in caseA caseB caseC caseR; do
    python scripts/02_run_cfd.py geometry=$case
done

# Hydra multirun (runs all cases in sequence under outputs/)
python scripts/02_run_cfd.py --multirun geometry=caseA,caseB,caseC,caseR
```

```bash
# Git Bash (Windows) — must run inside WSL
# Open a WSL terminal, then:
cd /mnt/d/Math/CFD/wss-hemodynamics-pinn
python scripts/02_run_cfd.py geometry=caseC
```

---

## 03_generate_synthetic_mri.py

**What it does:** Reads a CFD solution and simulates 4D-flow MRI acquisition: averages the velocity field over each voxel (downsampling), then adds optional Gaussian noise (σ = VENC/VNR). Produces the MRI observation arrays used as PINN training data.

**Requires:** `data/cfd/<case_id>/solution.npz` from `02_run_cfd.py`. No FEniCSx required — runs on all platforms.

**Output:** `data/synthetic_mri/<case_id>/<voxel_config>/mri_obs.npz` — voxel centre coordinates, voxelised velocity components, noise standard deviation, and config metadata.

```bash
# Git Bash (Windows) — from repo root with .venv activated

# Default: 1.0 mm voxels, noiseless, caseA
python scripts/03_generate_synthetic_mri.py geometry=caseA

# 0.5 mm voxels with Gaussian noise (sigma in m/s)
python scripts/03_generate_synthetic_mri.py \
    geometry=caseC mri=voxel_0p5mm mri.sigma=4e-6

# All voxel sizes for caseC
python scripts/03_generate_synthetic_mri.py --multirun \
    geometry=caseC \
    mri=voxel_0p5mm,voxel_1p0mm,voxel_1p5mm,voxel_2p0mm,voxel_2p5mm

# Full sweep: all cases x all voxel sizes
python scripts/03_generate_synthetic_mri.py --multirun \
    geometry=caseA,caseB,caseC,caseR \
    mri=voxel_0p5mm,voxel_1p0mm,voxel_1p5mm,voxel_2p0mm,voxel_2p5mm
```

```bash
# macOS / Linux

python scripts/03_generate_synthetic_mri.py geometry=caseA

python scripts/03_generate_synthetic_mri.py \
    geometry=caseC mri=voxel_0p5mm mri.sigma=4e-6

python scripts/03_generate_synthetic_mri.py --multirun \
    geometry=caseA,caseB,caseC,caseR \
    mri=voxel_0p5mm,voxel_1p0mm,voxel_1p5mm,voxel_2p0mm,voxel_2p5mm
```

---

## 04_train_pinn.py

**What it does:** Trains the physics-informed neural network on one case. Loads the CFD solution (for WSS ground-truth validation only) and the synthetic MRI observations (as training signal). Runs Adam optimisation followed by L-BFGS fine-tuning. Saves checkpoints and the best model.

**Requires:**
- `data/cfd/<case_id>/solution.npz` (from `02_run_cfd.py`)
- `data/synthetic_mri/<case_id>/<voxel_config>/mri_obs.npz` (from `03_generate_synthetic_mri.py`)

No FEniCSx required — runs on all platforms. GPU supported via `training.device=cuda`.

**Output:** `data/checkpoints/<case_id>/<run_id>/`
- `best_model.pt` — lowest validation-loss checkpoint
- `checkpoint_adam_<step>.pt` — periodic Adam checkpoints
- `loss_history.npy` — full loss curve (Adam + L-BFGS)

```bash
# Git Bash (Windows) — from repo root with .venv activated

# Plain MLP on caseC (recommended first run)
python scripts/04_train_pinn.py geometry=caseC model=pinn_base

# Random Fourier Features variant
python scripts/04_train_pinn.py geometry=caseC model=pinn_rff

# GPU training
python scripts/04_train_pinn.py geometry=caseC training.device=cuda

# Short test run (reduced Adam iterations)
python scripts/04_train_pinn.py geometry=caseC training.n_adam=5000 training.n_lbfgs=500

# Override loss weights
python scripts/04_train_pinn.py geometry=caseC \
    training.lambda_phys=10.0 training.lambda_bc=100.0

# Multiple cases (sequential multirun)
python scripts/04_train_pinn.py --multirun geometry=caseA,caseB,caseC,caseR
```

```bash
# macOS / Linux

python scripts/04_train_pinn.py geometry=caseC model=pinn_base

python scripts/04_train_pinn.py geometry=caseC model=pinn_rff \
    training.device=cuda training.n_adam=20000
```

---

## Planned scripts (not yet implemented)

These scripts are listed in the project plan and will be added in later phases:

| Script | Phase | Purpose |
|--------|-------|---------|
| `05_evaluate.py` | 2 | Compute WSS metrics (NRMSE, R², Bland–Altman) vs CFD ground truth |
| `06_run_sweep.py` | 2 | Automated parameter sweep over voxel size × noise level |
| `07_make_all_figures.py` | 4 | Batch figure generation for all paper figures |
| `04b_bhpo_search.py` | 3 | Bayesian hyperparameter optimisation search |
| `04c_train_pinn_with_bhpo.py` | 3 | Retrain PINN at BHPO-optimal hyperparameters |

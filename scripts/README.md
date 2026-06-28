# scripts/

Numbered pipeline entry points. Run them in order. Each script is a self-contained Hydra application; all configuration comes from [`configs/`](../configs/).

**OS requirements at a glance:**

| Script | Windows (native) | macOS / Linux | WSL |
|--------|:---:|:---:|:---:|
| `fetch_and_inspect.sh` | Git Bash ✓ | ✓ | ✓ |
| `02_run_cfd.py` | ✗ (needs FEniCSx) | ✓ | ✓ |
| `03_generate_synthetic_mri.py` | ✓ | ✓ | ✓ |
| `04_train_pinn.py` | ✓ | ✓ | ✓ |
| `04b_bhpo_search.py` | ✓ | ✓ | ✓ |
| `04c_train_pinn_with_bhpo.py` | ✓ | ✓ | ✓ |
| `05_evaluate.py` | ✓ | ✓ | ✓ |
| `06_run_sweep.py` | ✓ | ✓ | ✓ |
| `07_make_all_figures.py` | ✓ | ✓ | ✓ |
| `00_diagnose_collapse.py` | ✓ | ✓ | ✓ |

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

**What it does:** Trains the physics-informed neural network on one case with fixed hyperparameters. Loads the CFD solution (for WSS ground-truth validation only) and the synthetic MRI observations (training signal). Builds a dense interpolant prior from the MRI voxels, then runs a curriculum warm-up → Adam → L-BFGS schedule with the anti-collapse loss recipe (relative data loss, inflow constraint, decaying interpolant prior, magnitude floor; hard-SDF no-slip). Saves checkpoints and the best model (selected by observation misfit).

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

# Ablate the anti-collapse recipe (interpolant prior / warm-up / magnitude floor off)
python scripts/04_train_pinn.py geometry=caseC \
    training.lambda_aux=0.0 training.n_warmup=0 training.lambda_mag_floor=0.0

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

## 04b_bhpo_search.py

**What it does:** Bayesian hyperparameter optimisation (Optuna TPE) on **caseC** (the reference geometry). Trains a short PINN per trial with the anti-collapse recipe and minimises WSS NRMSE on 20% held-out wall faces.

**Requires:** `data/cfd/caseC/solution.npz`, `data/synthetic_mri/caseC/voxel_*/mri_obs.npz`.

**Output:** `data/bhpo_runs/caseC/` — `best_params.json`, `trial_log.csv`, `optuna_study.db`.

```bash
python scripts/04b_bhpo_search.py geometry=caseC
python scripts/04b_bhpo_search.py geometry=caseC bhpo.n_calls=30 bhpo.device=mps
```

## 04c_train_pinn_with_bhpo.py

**What it does:** Retrains the PINN at full budget using the BHPO-winning hyperparameters (read from `data/bhpo_runs/<bhpo.source_case_id>/`). Can be applied to any geometry.

**Output:** `data/checkpoints/<case_id>_<voxel_tag>_bhpo/` — checkpoints, `best_model.pt`, `bhpo_params.json` (provenance for `05_evaluate.py`).

```bash
python scripts/04c_train_pinn_with_bhpo.py geometry=caseC
python scripts/04c_train_pinn_with_bhpo.py geometry=caseA   # reuses caseC's BHPO result
```

## 00_diagnose_collapse.py

**What it does:** Loads a trained checkpoint and reports the predicted/CFD velocity-magnitude ratio, WSS-magnitude ratio, WSS NRMSE, and every per-term loss, ending with a `HEALTHY`/`COLLAPSED` verdict. Run it before and after any training change to confirm the field has not collapsed to the trivial Stokes solution.

```bash
python scripts/00_diagnose_collapse.py geometry=caseC eval.run_suffix=adam50000 model=pinn_base
python scripts/00_diagnose_collapse.py geometry=caseC        # BHPO-retrained run (default suffix)
```

## 05_evaluate.py · 06_run_sweep.py · 07_make_all_figures.py

| Script | Purpose |
|--------|---------|
| `05_evaluate.py` | WSS metrics (NRMSE, R², Bland–Altman, conservation) vs CFD ground truth → `data/results/` |
| `06_run_sweep.py` | Discover all checkpoints → `sweep_results.csv` |
| `07_make_all_figures.py` | Batch-generate all paper figures → `paper/figures/` |

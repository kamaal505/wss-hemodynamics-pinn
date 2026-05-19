---
output:
  pdf_document: default
  html_document: default
editor_options: 
  markdown: 
    wrap: sentence
---

# CLAUDE.md — hemodyn-pinn project

# Physics-Informed Neural Network for WSS Recovery from Sparse 4D-flow MRI

# 

# This file is the navigation map for Claude Code.

# It records decisions, calibrated values, and pointers — not subject knowledge.

# For physics, ML theory, and algorithms, go to docs/project_knowledge/.

# 

# Last updated: 2026-05-18

------------------------------------------------------------------------

## 0. Session startup — READ THIS FIRST

This file is the sole persistent memory for Claude Code across sessions.
The claude.ai web-app chats are NOT accessible from Claude Code.

### Mandatory first steps

1.  Read this entire file.
2.  Read the docs listed in §11 that are relevant to the task. They win over this file.
3.  Check §4 for current phase and next deliverable.
4.  Check §9 for open questions before making assumptions.

### Where things live

| What                   | Where                                    |
|------------------------|------------------------------------------|
| Project knowledge docs | docs/project_knowledge/ (six .tex files) |
| Mesh statistics        | data/geometries/anxplore/mesh_stats.txt  |
| AnXplore geometries    | data/geometries/anxplore/                |
| Source modules         | src/hemodyn_pinn/                        |
| Pipeline scripts       | scripts/ (numbered, run in order)        |
| Figure generation      | src/hemodyn_pinn/viz/                    |
| Hydra configs          | configs/                                 |
| Tests                  | tests/                                   |

### After a gap

Ask "which files changed since last session?" then read those files.
Do not assume the codebase matches this file — it may be ahead or behind.

### Model configuration

Default: sonnet (claude-sonnet-4-6).
Background/fast: haiku (claude-haiku-4-5-20251001).
Use `/model haiku` for mechanical tasks (boilerplate, config stubs).
Do not use Opus unless specified.

------------------------------------------------------------------------

## 1. Project identity

**Title:** PINN-based Wall Shear Stress Recovery from Sparse, Noisy 4D-flow MRI\
**Goal:** Recover wall shear fields from synthetic 4D-flow MRI (downsampled CFD); characterise dependence on sample density, noise level, and geometry irregularity.\
**Output:** Methods + benchmark paper.
Primary artefact: open benchmark (CFD fields + synthetic MRI inputs + checkpoints).
Not a clinical validation paper.\
**Target venues:** CMAME, Computers in Biology and Medicine, Medical Image Analysis, JCP.

------------------------------------------------------------------------

## 2. Architectural decisions

### 2.1 Data sources

| Role | Source | Notes |
|---------------------|---------------------------|------------------------|
| Primary | AnXplore (Goetz et al. 2024, MIT). 101 patient-derived aneurysm cases. <https://github.com/aurelegoetz/AnXplore> | In use |
| Secondary | Vascular Model Repository (VMR), 2–3 showcase cases. <https://www.vascularmodel.com> | Phase 2; site was down 2026-05-13, re-check before use |
| Tertiary | AneuX (CC BY-NC — do NOT redistribute as benchmark); AneuriskWeb (CC BY-NC — comparison figures only) | Only if reviewers request cohort breadth |

We generate our own CFD ground truth.
We do NOT use Goetz et al.'s FSI outputs (pulsatile FSI vs our steady Stokes).

### 2.2 Physics (see doc 02 for equations)

-   Flow model: steady incompressible Newtonian Stokes (Re \<\< 1). Pulsatility, FSI, non-Newtonian: deferred to follow-up paper.
-   Key parameters: ρ = 1060 kg/m³, μ = 3.5×10⁻³ Pa·s, U_inlet = 2×10⁻⁵ m/s → Re ≈ 0.10.

### 2.3 PINN architecture (see doc 03 for full spec)

-   Coordinate-based MLP: input (x,y,z)/L, output (û,v̂,ŵ,p̂). Tanh activations **only** — never ReLU (breaks second-order autograd).
-   Optional: Random Fourier Features encoder; hard no-slip via SDF mask.
-   Framework: PyTorch 2.x, from-scratch. DeepXDE for prototype only.

### 2.4 Loss, WSS computation, training recipe

-   Three loss terms: MRI data fit, Stokes PDE residual + continuity, no-slip BC. See doc 03 for formulas and weight defaults.
-   WSS via autograd on the trained network — no finite differences. See doc 03.
-   Optimiser: Adam → L-BFGS. Cosine LR 1e-3→1e-6. Collocation points resampled each epoch. Pressure anchor at one outlet DOF.

### 2.5 BHPO — Phase 3 only

-   Modules: `src/hemodyn_pinn/bhpo/`, scripts `04b_bhpo_search.py`, `04c_train_pinn_with_bhpo.py`.
-   Do NOT activate until the fixed-HP pipeline runs end-to-end.
-   hydra-optuna-sweeper DEFERRED (version conflict with Optuna 4.7); add at Phase 3.

### 2.6 Toolstack

| Purpose | Tool |
|-------------------------------------------|-----------------------------|
| Deep learning | PyTorch 2.x |
| CFD solver | FEniCSx (dolfinx 0.10) — Linux/WSL only, not pip-installable |
| Meshing | gmsh 4.13+ |
| 3D visualisation | PyVista 0.44+ |
| 2D figures | matplotlib 3.9+, seaborn 0.13+ |
| Mesh I/O | meshio 5.3+ |
| SDF | libigl 2.5+ |
| Config management | Hydra 1.3+ |
| Experiment tracking | Weights & Biases 0.19+ |
| HPO (Phase 3) | scikit-optimize 0.10, Optuna 4.7+ |
| Statistics | pingouin 0.5+ (Bland–Altman, ICC) |
| Testing | pytest 8.2+ |
| Linting / types | ruff 0.4+, mypy 1.10 (\<2.0) |

------------------------------------------------------------------------

## 3. Project tree

```         
hemodyn-pinn/
├── CLAUDE.md
├── configs/
│   ├── config.yaml                  ← Hydra root config
│   ├── geometry/                    ← one .yaml per AnXplore case
│   ├── cfd/                         ← stokes_default.yaml, ns_steady.yaml
│   ├── mri/                         ← voxel_0p5mm.yaml … voxel_2p5mm.yaml
│   ├── model/                       ← pinn_base.yaml, pinn_rff.yaml
│   ├── bhpo/                        ← search_space.yaml, gp_default.yaml
│   └── sweep/
├── src/hemodyn_pinn/
│   ├── geometry/
│   │   ├── anxplore_loader.py       ← load AnXplore meshes; boundary extraction
│   │   ├── vmr_loader.py            ← VMR mesh loader (Phase 2)
│   │   ├── bulge_scaler.py          ← controlled-severity geometry axis
│   │   ├── parametric.py            ← idealised stenosis-bifurcation mesh
│   │   ├── sdf.py                   ← signed distance field for hard no-slip
│   │   └── meshing.py
│   ├── cfd/
│   │   ├── stokes_solver.py         ← FEniCSx P2-P1 steady Stokes (MUMPS)
│   │   ├── ns_solver.py             ← Navier-Stokes (Phase 2+)
│   │   └── postprocess.py           ← save/load .npz + XDMF; WSS statistics
│   ├── mri/
│   │   ├── operator.py              ← SyntheticMRIOperator: voxel-average + noise
│   │   ├── voxelise.py              ← VoxelGrid; node-to-voxel assignment
│   │   └── noise.py                 ← Gaussian noise; sigma_from_vnr
│   ├── pinn/
│   │   ├── networks.py              ← MLP, RFFEncoder, PINNNetwork
│   │   ├── losses.py                ← data / physics / BC / anchor losses
│   │   ├── sampling.py              ← collocation and wall-BC point sampling
│   │   ├── trainer.py               ← PINNTrainer + PINNConfig (Adam → L-BFGS)
│   │   └── inference.py             ← WSS via autograd; velocity prediction
│   ├── bhpo/                        ← Phase 3; do not activate yet
│   ├── bayesian/                    ← B-PINN (Phase 4)
│   ├── baselines/
│   │   ├── trilinear_fd.py          ← trilinear interpolation + FD WSS baseline
│   │   └── supervised_mlp.py        ← data-only MLP baseline
│   ├── eval/
│   │   ├── field_metrics.py         ← NRMSE, R², MAE
│   │   ├── wss_metrics.py           ← pointwise relative error, peak WSS error
│   │   ├── conservation.py          ← flow-rate constancy, bifurcation balance
│   │   └── bland_altman.py          ← Bland–Altman + ICC
│   ├── viz/
│   │   ├── style.py                 ← rcParams, PyVista theme
│   │   ├── render_3d.py
│   │   ├── plots_2d.py
│   │   └── make_figure_N.py         ← one script per paper figure (see doc 04)
│   └── utils/
│       ├── io.py
│       ├── seeds.py                 ← all RNG seeds; never hard-code elsewhere
│       └── logging.py
├── scripts/                         ← numbered pipeline; run in order
│   ├── 01_fetch_geometries.py
│   ├── 02_run_cfd.py
│   ├── 03_generate_synthetic_mri.py
│   ├── 04_train_pinn.py
│   ├── 04b_bhpo_search.py           ← Phase 3
│   ├── 04c_train_pinn_with_bhpo.py  ← Phase 3
│   ├── 05_evaluate.py
│   ├── 06_run_sweep.py
│   └── 07_make_all_figures.py
├── data/                            ← NOT committed to repo
│   ├── geometries/anxplore/         ← git-cloned AnXplore VTK files
│   ├── geometries/vmr/              ← manual download
│   ├── geometries/idealised/        ← generated
│   ├── cfd/                         ← solver outputs (.npz, XDMF+HDF5)
│   ├── synthetic_mri/
│   ├── checkpoints/
│   ├── bhpo_runs/
│   └── results/
├── tests/
├── notebooks/
│   ├── 00_sanity_check.ipynb
│   └── 01_anxplore_exploration.ipynb
└── paper/
    ├── main.tex
    ├── refs.bib
    └── figures/
```

------------------------------------------------------------------------

## 4. Phase plan and current status

| Phase | Scope | Status |
|---------------|--------------------------------|--------------------------|
| 0 | Project setup, docs, environment | COMPLETE |
| 1.1 | Environment + AnXplore data fetch | COMPLETE |
| 1.2 | anxplore_loader.py + mesh checks | COMPLETE |
| 1.3 | Stokes CFD on 3–5 cases (FEniCSx) | CODE COMPLETE |
| 1.4 | Synthetic MRI forward operator | COMPLETE |
| 1.5 | Prototype PINN, end-to-end on one case | CODE COMPLETE (awaits CFD data) |
| 2 | Core results figures at fixed HPs | not started |
| 3 | BHPO integration + refined results | not started |
| 4 | Ablation, supplementary, paper draft | not started |

**Next deliverable:** Run end-to-end on caseC (Linux/WSL required for FEniCSx): 1.
`python scripts/02_run_cfd.py geometry=caseC` 2.
`python scripts/03_generate_synthetic_mri.py geometry=caseC` 3.
`python scripts/04_train_pinn.py geometry=caseC`

### Phase completion records

**Phase 1.3 (2026-05-13):** Stokes solver aligned to doc 06.
- Pressure sign corrected: `_extract_nodal_values` negates p_sub (physical = −dolfinx convention).
- Inlet BC: Poiseuille parabolic profile via collapsed-subspace `Function.interpolate()`.
- XDMF+HDF5 export: `export_xdmf()` and `load_cfd_xdmf()` added to postprocess.py.
- 84 fast tests pass.

**Phase 1.4 (2026-05-13):** Synthetic MRI operator.
- `mri/voxelise.py`, `mri/noise.py`, `mri/operator.py`.
Voxel sizes: 0.5–2.5 mm (5 configs).
VNR sweep: {5,10,15,20,30}.
- 26/26 tests pass (Windows).

**Phase 1.5 (2026-05-14):** Prototype PINN.
- `pinn/networks.py`, `losses.py`, `sampling.py`, `trainer.py`, `inference.py`.
- Seeds added: RFF_SEED=2718, PINN_COLLOC_SEED=3141, PINN_TRAIN_SEED=9999.
- 116 tests pass (1 dolfinx-skipped, 0 warnings).

------------------------------------------------------------------------

## 5. Calibrated parameters

### Physical constants and scales

| Quantity | Value | Notes |
|------------------------------|---------------------|---------------------|
| μ (viscosity) | 3.5×10⁻³ Pa·s | Newtonian blood; in `stokes_solver.MU_BLOOD` |
| ρ (density) | 1060 kg/m³ | in `stokes_solver.RHO_BLOOD` |
| L (length scale) | 0.01628 m | x-span (parent artery diameter); in `stokes_solver.L_SCALE` |
| U_inlet | 2×10⁻⁵ m/s | → Re ≈ 0.10; confirmed Stokes regime |
| VTK coordinates | mm | Multiply × 1e-3 for SI before any physical computation |

### AnXplore mesh summary

| Case         | Nodes   | Tets      | z-span (mm) | Notes                 |
|--------------|---------|-----------|-------------|-----------------------|
| caseC        | 147,432 | 844,320   | 4.050       | Simplest — start here |
| caseA        | 159,155 | 912,713   | 4.614       |                       |
| caseR        | 178,675 | 1,028,394 | 6.223       |                       |
| caseB        | 170,087 | 975,723   | 6.385       | Most complex          |
| full_dataset | 158,840 | 910,886   | 5.046       | Has 33k explicit tris |

**Structural difference:** caseA/B/C/R have no explicit surface triangles — boundary faces extracted from tets in `anxplore_loader.py`.
`full_dataset` has 33k explicit surface triangles used directly as Γ_w.

**Inlet/outlet topology:** Both open faces at y=y_min, outward normal (0,−1,0).
Left disk (x\<0) = inlet; right disk (x\>0) = outlet.
Detection: n_y \< −0.9999 at y_min, split at x=0.

------------------------------------------------------------------------

## 6. Environment and setup

-   Virtual environment: `.venv/` at repo root (uv pip).

-   Python: `.venv/Scripts/python` (Windows) or `.venv/bin/python` (Linux/macOS).

-   **FEniCSx:** NOT pip/uv installable.
    Linux/WSL only:

    ```         
    sudo add-apt-repository ppa:fenics-packages/fenics && sudo apt install fenicsx
    ```

-   mypy capped at \<2.0 (large 2.x wheels cause uv timeout on slow connections).

-   hydra-optuna-sweeper: commented out in requirements.txt; re-enable at Phase 3.

-   Shell scripts: Git Bash compatible (no mapfile/stat -c/process substitution).

-   **dolfinx 0.10 API:** `create_mesh(comm, cells, e, x)` — element is the 3rd argument, coordinates are 4th.
    (Swapped vs ≤ 0.8.)

------------------------------------------------------------------------

## 7. Mandatory tests

| Test file | Covers | Required by |
|-------------------------|------------------|-----------------------------|
| test_anxplore_loader.py | Mesh loading, boundary extraction | Phase 1.2 |
| test_mri_operator.py | Forward operator correctness + noise stats | Phase 1.4 |
| test_losses.py | Each loss term in isolation | Phase 1.5 |
| test_inference.py | WSS-from-network path (autograd → τ_w) | Phase 1.5 |
| test_bulge_scaler.py | Controlled severity axis | Phase 2 |
| test_bhpo.py | BHPO smoke test | Phase 3 |

Do not mark a phase complete without its tests passing.

------------------------------------------------------------------------

## 8. Code conventions

-   Python 3.11+. Type hints throughout. NumPy-style docstrings.
-   **Never** ReLU in PINN backbone. **Never** jet colormaps.
-   All RNG seeds in `utils/seeds.py` — never hard-code.
-   Prefer functional/pure code; classes for stateful components only.
-   Figure output: vector PDF or 600 dpi PNG. One script per paper figure (see doc 04).
-   Colormaps: viridis/magma/cividis (sequential); RdBu_r (diverging, signed quantities only).

------------------------------------------------------------------------

## 9. Open questions

| \# | Item | Status |
|-----------------|------------------------|--------------------------------|
| 1 | Re target for Phase 1 CFD | CLOSED: U=2×10⁻⁵ m/s → Re≈0.10 |
| 2 | full_dataset contents | CLOSED: 101 cases (Fluid_0 … Fluid_100) |
| 3 | Boundary extraction for caseA/B/C/R | CLOSED: implemented in anxplore_loader.py |
| 4 | AnXplore inlet/outlet topology | CLOSED: see §5 |
| 5 | hydra-optuna-sweeper Phase 3 version | DEFERRED: stable 1.2.0 caps Optuna \<3.0; check for 1.3.x at Phase 3 |

------------------------------------------------------------------------

## 10. Key references

-   Goetz et al. (2024). AnXplore. *Front. Bioeng. Biotechnol.* DOI: 10.3389/fbioe.2024.1433811
-   Tancik et al. (2020). Fourier features / spectral bias. *NeurIPS.* arXiv:2006.10739
-   Wang, Wang, Perdikaris (2021). Adaptive loss weights for PINNs. *CMAME.* arXiv:2012.10047
-   Karniadakis et al. (2021). Physics-informed machine learning. *Nat. Rev. Phys.*
-   Yang, Meng, Karniadakis (2021). B-PINNs. *JCP.* arXiv:2003.06097
-   Snoek, Larochelle, Adams (2012). Bayesian optimisation of ML. *NeurIPS.*

------------------------------------------------------------------------

## 11. Project knowledge documents

Go to these for all subject-matter detail.
Do NOT copy their content into this file.

| Doc | File | Read when... |
|-----------------|-----------------|--------------------------------------|
| 01 | 01_hemodynamics_glossary.tex | Unfamiliar hemodynamics/imaging term (WSS, TAWSS, OSI, VNR, …) |
| 02 | 02_fluid_dynamics_primer.tex | NS equations, Stokes limit, WSS formula, non-dimensionalisation, MRI forward operator |
| 03 | 03_ml_concepts.tex | Network architecture, RFF, loss terms, training pitfalls, optimiser recipe, B-PINNs, metrics |
| 04 | 04_research_artefacts_and_figures.tex | Any paper figure or table (exact spec for all 12 figures + 3 tables) |
| 05 | 05_results_interpretation.tex | Interpreting metrics, writing Discussion, clinical relevance claims |
| 06 | 06_fenicsx_cfd_implementation.tex | Any FEniCSx/dolfinx implementation question |

These files are READ-ONLY reference material.
Do not modify them.

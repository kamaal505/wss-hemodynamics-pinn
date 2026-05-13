# CLAUDE.md — hemodyn-pinn project
# Physics-Informed Neural Network for WSS Recovery from Sparse 4D-flow MRI
#
# This file is the durable project memory for Claude Code.
# Update it whenever a decision is made, a parameter is calibrated,
# or a phase is completed. Do not rely on chat history for any of this.
#
# Last updated: 2026-05-14  (Phase 1.5 complete — prototype PINN implemented)

---

## 0. Claude Code session startup — READ THIS FIRST

This file is the sole persistent memory for Claude Code across sessions.
The claude.ai web-app chats are NOT accessible from Claude Code.

### Every session: mandatory first steps
1. Read this entire file before touching any code or files.
2. Read the project knowledge documents listed in section 14 that are
   relevant to the task at hand. They are the authoritative specification;
   this file is a summary. When they conflict, the tex files win.
3. Check section 5 (phase table) to confirm current phase and next deliverable.
4. Check section 12 (open questions) before making any assumption that
   could be answered there.

### Orientation: where things live
| What                        | Where                                        |
|-----------------------------|----------------------------------------------|
| This file                   | CLAUDE.md (repo root)                        |
| Project knowledge docs      | docs/project_knowledge/ (six tex files)           |
| Mesh statistics             | data/geometries/anxplore/mesh_stats.txt      |
|                             | data/geometries/anxplore/mesh_stats_per_case.csv |
| AnXplore geometries         | data/geometries/anxplore/                    |
| All source modules          | src/hemodyn_pinn/                            |
| Pipeline scripts            | scripts/ (numbered, run in order)            |
| Figure generation           | src/hemodyn_pinn/viz/                        |
| Configs                     | configs/ (Hydra)                             |
| Tests                       | tests/                                       |

### How to continue after a gap
If resuming mid-phase, ask the user: "Which files have changed since the
last session?" then read those files before proceeding. Do not assume the
codebase matches what CLAUDE.md describes — it may be ahead or behind.

### Model configuration for this project
Default model: sonnet (claude-sonnet-4-6)
Background/fast model: haiku (claude-haiku-4-5-20251001)
Opus: do not use unless specified.
Mid-session switching: use /model haiku for mechanical tasks
(boilerplate, docstrings, config stubs), /model sonnet for everything else.

---

## 1. Project identity

**Title:** PINN-based Wall Shear Stress Recovery from Sparse, Noisy 4D-flow MRI

**Central question:**
Given sparse, noisy velocity samples from synthetic 4D-flow MRI (downsampled CFD),
how does a physics-informed reconstruction recover the wall shear field, and what is
the dependence on (i) sample density, (ii) noise level, (iii) geometry irregularity?

**Target venues:** CMAME, Computers in Biology and Medicine, Medical Image Analysis,
Journal of Computational Physics.

**Paper framing:** methods + benchmark paper. The open benchmark (CFD fields +
synthetic MRI inputs at controlled operating points + pre-trained checkpoints) is a
primary release artefact. Do NOT reframe as a clinical validation paper.

---

## 2. Team context

Small applied-mathematics team. Comfortable with PDE theory, optimisation, linear
algebra. Learning hemodynamics, fluid mechanics modelling, and ML practice in
parallel. Python throughout. No clinical data collection — synthetic-on-real-geometry
path only.

---

## 3. Fixed architectural decisions — do not revisit without explicit instruction

### 3.1 Data path
- **Primary:** AnXplore dataset (Goetz et al. 2024, MIT licensed).
  Patient-derived intracranial aneurysm geometries on a standardised toroidal
  parent vessel. 101 cases indexed 0–100. Working cohort: 20–30 cases.
  URL: https://github.com/aurelegoetz/AnXplore
  DOI: https://doi.org/10.3389/fbioe.2024.1433811

- **Secondary:** Vascular Model Repository (VMR), 2–3 cases as patient-realism
  showcase figures. Manual registration required. NOT needed until Phase 2.
  URL: https://www.vascularmodel.com
  Status (checked 2026-05-13): dataset gallery page rendering placeholder text;
  may be a temporary site issue. Re-check before Phase 2 download.
  License: research use; derivative results redistributable; do NOT redistribute
  geometries.

- **Optional anchor:** One parametric idealised stenosis-bifurcation case
  (FEniCSx + gmsh), swept over ~10 severity values, for a clean phase-diagram
  anchor. Can be replaced by scaling an AnXplore bulge along a controlled axis.

- **Tertiary (only if reviewers request cohort breadth):**
  - AneuX: https://zenodo.org/records/6678442  (750 geometries, no CFD)
    LICENSE WARNING: CC BY-NC 4.0. The NC clause encumbers any derivative
    dataset released alongside the paper. Do NOT use as a geometry source
    for the benchmark. Only viable for cohort-breadth figures if reviewers
    insist and you do not redistribute the geometries themselves.
    Download: data-v1.0.zip (13 MB metadata) + models-v1.0.zip (6.3 GB meshes).
    Only download models if absolutely required.
  - AneuriskWeb: http://ecm2.mathcs.emory.edu/aneuriskweb/ (65 ICA cases,
    pre-computed CFD-WSS). License: CC BY-NC 3.0. Use for external WSS
    baseline comparison figures only (not redistributed). Available without
    registration. Download selectively at Phase 2.

- We generate our own CFD ground truth on AnXplore geometries. We do NOT use
  Goetz et al.'s FSI simulation outputs (they used pulsatile FSI; we need steady
  Stokes/NS with our own BC choices).

### 3.2 Flow regime
- **Phase 1–2:** Steady, incompressible, Newtonian, Stokes regime (Re << 1).
  Governing equations:  -∇p + μΔu = 0,  ∇·u = 0.
- Pulsatility (Womersley), FSI, non-Newtonian rheology: explicitly deferred to
  follow-up paper. State as limitations.
- Blood parameters: ρ = 1060 kg/m³, μ = 3.5×10⁻³ Pa·s (Newtonian estimate).

### 3.3 PINN architecture
- Coordinate-based MLP. Input: x ∈ ℝ³. Output: (u, v, w, p).
- Activations: tanh throughout. NEVER ReLU (breaks second-order autograd).
- Optional input encoding: Random Fourier Features (RFF),
    φ(x) = [sin(2πBx), cos(2πBx)],  B_ij ~ N(0, σ²).
- Optional hard no-slip: u_θ(x) = ψ(x)·N_θ(x),  ψ = SDF (vanishes on wall).
- Framework: PyTorch 2.x, from-scratch implementation.
  DeepXDE acceptable for first prototype only; not for final submission.

### 3.4 Loss function
  L(θ) = λ_data·L_data + λ_phys·L_phys + λ_bc·L_bc

  L_data = (1/N_d) Σ ‖u_θ(x_i) - u_obs_i‖²        (MRI observation fit)
  L_phys = (1/N_r) Σ ‖R_NS(u_θ, p_θ)(x_j)‖²        (PDE residual)
         + (1/N_r) Σ |∇·u_θ(x_j)|²                  (continuity)
  L_bc   = (1/N_b) Σ ‖u_θ(x_k^w)‖²                  (no-slip)

  Initial weights: literature defaults (to be recorded when calibrated).
  Adaptive balancing (Wang–Teng–Perdikaris) reserved for ablation study.

### 3.5 WSS computation
  At wall point x_w with outward normal n̂, via autograd on the trained network:
    D_θ(x_w) = ½(∇u_θ + (∇u_θ)ᵀ)|_{x_w}
    τ_w = 2μ [D_θ·n̂]_tangential
  No finite differences. The network is differentiable everywhere by construction.

### 3.6 Training recipe (Phase 1 defaults)
- Optimiser: Adam → L-BFGS (switch at 50k–200k iterations; exact epoch TBD).
- LR schedule: cosine annealing, 1e-3 → 1e-6.
- Collocation points: 10⁴–10⁵ per epoch, resampled each epoch.
- Pressure anchor: fix p = 0 at one outlet point (prevents pressure drift).
- Checkpointing: save every N epochs; keep lowest validation (physics + data) loss.

### 3.7 Bayesian HPO (BHPO) — reserved for Phase 3
- Module: src/hemodyn_pinn/bhpo/
- Script slots: scripts/04b_bhpo_search.py, scripts/04c_train_pinn_with_bhpo.py
- DO NOT activate until end-to-end pipeline runs at fixed hyperparameters.
- BHPO objective: validation MSE on held-out MRI samples + small physics residual
  penalty. DO NOT optimise on WSS error (methodological leak).
- Algorithm: GP-EI via scikit-optimize (primary); Optuna TPE (fallback).
- Budget: ~150 trials total (50 per representative geometry × 3 geometries).
- Multi-fidelity trick: run BHPO at 30k iterations, retrain at optimum to full
  length. Saves ~4–5× compute.
- hydra-optuna-sweeper: DEFERRED. The stable 1.2.0 caps Optuna < 3.0 (conflicts
  with optuna >= 4.7). Add at Phase 3 with --prerelease=allow, or check for a
  stable 1.3.x release by then.

### 3.8 Toolstack
| Purpose              | Tool                          |
|----------------------|-------------------------------|
| Deep learning        | PyTorch 2.x                   |
| CFD ground truth     | FEniCSx (dolfinx 0.10)        |
| Meshing              | gmsh 4.13+                    |
| 3D visualisation     | PyVista 0.44+                 |
| 2D figures           | matplotlib 3.9+, seaborn 0.13+|
| Mesh I/O             | meshio 5.3+                   |
| SDF computation      | libigl 2.5+                   |
| Config management    | Hydra 1.3+                    |
| Experiment tracking  | Weights & Biases 0.19+        |
| HPO (Phase 3)        | scikit-optimize 0.10, Optuna 4.7+ |
| Statistics           | pingouin 0.5+ (Bland–Altman, ICC) |
| Testing              | pytest 8.2+                   |
| Linting / types      | ruff 0.4+, mypy 1.10 (<2.0)   |

**FEniCSx install note:** dolfinx is NOT pip/uv installable. Install via:
  Ubuntu/Debian: sudo add-apt-repository ppa:fenics-packages/fenics && sudo apt install fenicsx
  macOS/other:  conda install -c conda-forge fenics-dolfinx mpich

---

## 4. Project tree (canonical)

hemodyn-pinn/
├── CLAUDE.md                        ← this file
├── README.md
├── pyproject.toml
├── requirements.txt
├── environment.yml
├── LICENSE
├── configs/
│   ├── config.yaml
│   ├── geometry/                    # one .yaml per AnXplore case used
│   ├── cfd/                         # stokes_default.yaml, ns_steady.yaml
│   ├── mri/                         # voxel_0p5mm.yaml … voxel_2p5mm.yaml
│   ├── model/                       # pinn_base.yaml, pinn_rff.yaml, etc.
│   ├── bhpo/                        # search_space.yaml, gp_default.yaml
│   └── sweep/
├── src/hemodyn_pinn/
│   ├── geometry/
│   │   ├── anxplore_loader.py       ← load AnXplore meshes
│   │   ├── vmr_loader.py
│   │   ├── bulge_scaler.py          ← controlled-severity axis
│   │   ├── parametric.py            ← idealised stenosis-bifurcation
│   │   ├── sdf.py                   ← SDF for hard no-slip
│   │   └── meshing.py
│   ├── cfd/
│   │   ├── stokes_solver.py
│   │   ├── ns_solver.py
│   │   └── postprocess.py
│   ├── mri/
│   │   ├── operator.py              ← voxel-averaging + noise forward op
│   │   ├── voxelise.py
│   │   └── noise.py
│   ├── pinn/
│   │   ├── networks.py              ← MLP + RFF + hard no-slip wrapper
│   │   ├── losses.py
│   │   ├── sampling.py
│   │   ├── trainer.py
│   │   └── inference.py
│   ├── bhpo/                        ← Phase 3; do not activate yet
│   ├── bayesian/                    ← B-PINN (separate from BHPO)
│   ├── baselines/
│   │   ├── trilinear_fd.py          ← trilinear interp + FD WSS
│   │   └── supervised_mlp.py
│   ├── eval/
│   │   ├── field_metrics.py         ← NRMSE, R², MAE
│   │   ├── wss_metrics.py           ← pointwise rel. error, peak WSS error
│   │   ├── conservation.py          ← Q(z) constancy, bifurcation balance
│   │   └── bland_altman.py
│   ├── viz/
│   │   ├── style.py                 ← rcParams, PyVista theme
│   │   ├── render_3d.py
│   │   ├── plots_2d.py
│   │   └── make_figure_N.py         ← one script per paper figure
│   └── utils/
│       ├── io.py
│       ├── seeds.py                 ← centralised RNG seed management
│       └── logging.py
├── scripts/
│   ├── 01_fetch_geometries.py       ← fetch_and_inspect.sh feeds into this
│   ├── 02_run_cfd.py
│   ├── 03_generate_synthetic_mri.py
│   ├── 04_train_pinn.py
│   ├── 04b_bhpo_search.py           ← Phase 3
│   ├── 04c_train_pinn_with_bhpo.py  ← Phase 3
│   ├── 05_evaluate.py
│   ├── 06_run_sweep.py
│   └── 07_make_all_figures.py
├── data/
│   ├── geometries/
│   │   ├── anxplore/                ← git-cloned; DO NOT commit to repo
│   │   ├── vmr/                     ← manual download; DO NOT commit
│   │   └── idealised/               ← generated
│   ├── cfd/
│   ├── synthetic_mri/
│   ├── checkpoints/
│   ├── bhpo_runs/
│   └── results/
├── tests/
│   ├── test_anxplore_loader.py
│   ├── test_bulge_scaler.py
│   ├── test_mri_operator.py         ← REQUIRED (in architecture doc)
│   ├── test_losses.py               ← REQUIRED
│   ├── test_inference.py            ← REQUIRED
│   └── test_bhpo.py                 ← Phase 3
├── notebooks/
│   ├── 00_sanity_check.ipynb
│   └── 01_anxplore_exploration.ipynb
└── paper/
    ├── main.tex
    ├── refs.bib
    └── figures/

---

## 5. Phase plan and current status

| Phase | Scope                                      | Status        |
|-------|--------------------------------------------|---------------|
| 0     | Project setup, docs, environment           | COMPLETE      |
| 1.1   | Environment + AnXplore data fetch          | COMPLETE      |
| 1.2   | anxplore_loader.py + mesh quality checks   | COMPLETE      |
| 1.3   | Stokes CFD on 3–5 cases (FEniCSx)          | CODE COMPLETE (doc 06 aligned) |
| 1.4   | Synthetic MRI forward operator             | COMPLETE      |
| 1.5   | Prototype PINN, end-to-end on one case     | CODE COMPLETE (awaits CFD data) |
| 2     | Core results figures at fixed HPs          | not started   |
| 3     | BHPO integration + refined results         | not started   |
| 4     | Ablation, supplementary, paper draft       | not started   |

**Last completed deliverable:** Phase 1.5 — Prototype PINN (code complete).
  See full record below.

**Phase 1.4 — Synthetic MRI forward operator (2026-05-13):**
  - src/hemodyn_pinn/mri/voxelise.py: VoxelGrid + assign_nodes_to_voxels
  - src/hemodyn_pinn/mri/noise.py: add_gaussian_noise, sigma_from_vnr
  - src/hemodyn_pinn/mri/operator.py: SyntheticMRIOperator, MRIObservation, MRIConfig
  - utils/seeds.py: added MRI_NOISE_SEED = 1701
  - configs/mri/voxel_{0p5,1p0,1p5,2p0,2p5}mm.yaml: 5 sweep configs
  - configs/config.yaml: added mri default + output.mri_base_dir
  - scripts/03_generate_synthetic_mri.py: Hydra pipeline script
  - tests/test_mri_operator.py: 26/26 tests pass on Windows
  All 64 fast tests (38 loader + 26 MRI) pass on Windows.
  FEniCSx solver code (Phase 1.3) is code-complete but awaits WSL execution.

**Phase 1.3 — all doc 06 deviations corrected (2026-05-13):**
  - Pressure sign: _extract_nodal_values now negates p_sub (physical = -dolfinx).
  - Inlet BC: replaced plug flow with Poiseuille parabolic profile via
    collapsed-subspace Function.interpolate() (doc 06 §4).
  - Output format: XDMF+HDF5 (velocity, pressure, wss_mag, wss_vec, mesh,
    metadata.json) written inside solve_stokes() when out_dir is provided.
    Added: _compute_wss_fn_scalar/vector (doc 06 §6 DG-0 L2 projection),
    export_xdmf() and load_cfd_xdmf() in postprocess.py.
    The .npz is still written alongside XDMF for the cross-platform pipeline.
  All 84 fast tests pass (64 mri+loader, 19 stokes-fast + 1 dolfinx-skipped).

**Phase 1.5 — Prototype PINN (2026-05-14, code complete, awaits CFD data):**
  - src/hemodyn_pinn/pinn/networks.py: MLP, RFFEncoder, PINNNetwork
      Input: x̂=(x,y,z)/L  Output: (û,v̂,ŵ,p̂)  Tanh activations throughout.
  - src/hemodyn_pinn/pinn/losses.py: data_loss, stokes_residual_loss (2nd-order
      autograd Laplacian), bc_loss, pressure_anchor_loss, total_loss.
  - src/hemodyn_pinn/pinn/sampling.py: sample_collocation, sample_wall_bc,
      make_pressure_anchor, epoch_rng (per-epoch collocation seeds).
  - src/hemodyn_pinn/pinn/trainer.py: PINNTrainer + PINNConfig.
      Adam (cosine LR 1e-3→1e-6) → L-BFGS (strong Wolfe line search).
      Checkpointing every N epochs; best-state restoration.
  - src/hemodyn_pinn/pinn/inference.py: compute_velocity_jacobian,
      compute_wss (autograd, tangential projection), predict_velocity_field.
  - utils/seeds.py: added RFF_SEED=2718, PINN_COLLOC_SEED=3141, PINN_TRAIN_SEED=9999.
  - configs/model/pinn_base.yaml + pinn_rff.yaml: Hydra model configs.
  - configs/config.yaml: added model default + output.pinn_base_dir.
  - scripts/04_train_pinn.py: Hydra pipeline script (loads mesh + CFD + MRI obs).
  - tests/test_losses.py: 22 tests — data loss, Stokes residual, BC, anchor,
      total; exact-solution zero-residual check verified.
  - tests/test_inference.py: 11 tests — Jacobian shape/values, Couette WSS
      magnitude check, tangential projection, dimensional scaling.
  All 116 tests pass (1 dolfinx-skipped). Zero warnings.

**Next deliverable:** Phase 1.5 complete → run actual end-to-end training.
  Step 1: Run WSL FEniCSx CFD to produce data/cfd/caseC/solution.npz.
  Step 2: Run scripts/03_generate_synthetic_mri.py for caseC.
  Step 3: Run scripts/04_train_pinn.py geometry=caseC.
  See §13 for WSL setup.

**WSL setup (one-time, run in Windows Terminal):**
  wsl --install -d Ubuntu         # restart after install
  # In WSL Ubuntu:
  sudo add-apt-repository ppa:fenics-packages/fenics
  sudo apt update && sudo apt install -y fenicsx python3-pip python3-scipy
  pip3 install meshio hydra-core omegaconf
  # Run from Windows path (WSL mounts Windows at /mnt/d/...):
  cd /mnt/d/Math/CFD/wss-hemodynamics-pinn
  python scripts/02_run_cfd.py geometry=caseC

---

## 6. Calibrated physical parameters and geometry facts

### Coordinate system and units
- All AnXplore mesh coordinates are in **millimetres**.
- Convert to metres before any physical computation: x_SI = x_mm × 1e-3.

### Non-dimensionalisation (to be applied in CFD and PINN)
- Length scale:  L = 16.28 mm = 0.01628 m  (x-span = parent artery diameter,
  consistent across all 5 cases to within 0.001 mm — use 16.28 mm as the
  canonical value).
- Velocity scale: U — set when CFD BCs are chosen (target Re << 1 for Phase 1).
  For strict Stokes: U ≲ 0.2 mm/s = 2×10⁻⁴ m/s.
- Pressure scale: μU/L (Stokes regime; gives non-dim Stokes: -∇̂p̂ + Δ̂û = 0).
  Note: ρU² is the NS scale; for Stokes (Re<<1) μU/L is the correct choice.
- Time scale: L/U (not needed for steady Stokes).
- All PINN inputs and outputs must be non-dimensional. Train in non-dimensional
  coordinates; convert back for metric evaluation.

### Mesh geometry (from mesh_stats files, 5 named cases)
| Case        | Nodes   | Tets      | Surf. tris | z-span (mm) | BBox vol (mm³) |
|-------------|---------|-----------|------------|-------------|----------------|
| caseA       | 159,155 | 912,713   | 0          | 4.614       | 974.1          |
| caseB       | 170,087 | 975,723   | 0          | 6.385       | 1381.3         |
| caseC       | 147,432 | 844,320   | 0          | 4.050       | 791.8          |
| caseR       | 178,675 | 1,028,394 | 0          | 6.223       | 1327.9         |
| full_dataset| 158,840 | 910,886   | 33,116     | 5.046       | 1084.9         |

- x-span = 16.279 mm across all cases (±0.001 mm) — parent artery diameter.
- y-span = 12.0–13.3 mm — arc length along torus.
- z-span = 4.1–6.4 mm — aneurysm bulge depth; the geometry-irregularity axis.
  caseC (4.1 mm) is shallowest/simplest; caseB (6.4 mm) is deepest/most complex.

### Mesh resolution
- Mean edge length: 0.155–0.158 mm across all cases (very consistent).
- p5 edge length: ~0.035 mm (finest near-wall elements).
- Max edge length: 0.357–0.371 mm (coarsest bulk elements).
- IQR outlier fraction ~14%: expected; reflects intentional mesh grading (finer
  near wall), NOT pathological elements.
- The mean edge length (0.157 mm) is 3–16× finer than planned MRI voxels
  (0.5–2.5 mm). Each voxel contains O(10³) mesh nodes. This is the information
  compression the PINN must invert.

### Critical structural difference between named cases and full_dataset
- caseA, caseB, caseC, caseR: volumetric tets only. NO explicit surface
  triangles. Wall boundary Γ_w must be extracted as the set of tet faces shared
  by exactly one tetrahedron. This is needed for: (a) SDF construction,
  (b) WSS evaluation, (c) no-slip BC collocation points.
- full_dataset (Fluid_0.vtk): has 33,116 explicit surface triangles. These ARE
  Γ_w and can be used directly.
- anxplore_loader.py must handle both cases transparently.

---

## 7. Synthetic MRI forward operator (to be implemented in mri/operator.py)

  u_MRI(x_v) = (1/|V_v|) ∫_{V_v} χ_Ω(x) u_CFD(x) dV  +  η,
  η ~ N(0, σ_v² I)

- Voxel sizes in sweep: Δx ∈ {0.5, 1.0, 1.5, 2.0, 2.5} mm (configs in configs/mri/).
- Noise: σ_v = VENC / VNR;  VNR sweep covers ~ {5, 10, 15, 20, 30}.
- Phase 1: spatial averaging + Gaussian noise only.
- Deferred (Phase 3+): aliasing (VENC wrapping), displacement artefact.
- Tests: test_mri_operator.py is REQUIRED before Phase 2.

---

## 8. Headline claims and supporting figures

| Claim | Description                                              | Key figure  |
|-------|----------------------------------------------------------|-------------|
| C1    | PINN WSS recovery feasible across broad operating range  | Fig 5, 7, 8 |
| C2    | Reliable/unreliable boundary exists in (Δx, SNR) space  | Fig 6       |
| C3    | Methodological tweak expands reliable region             | Fig 10      |
| C4    | B-PINN uncertainty is calibrated                         | Fig 11      |

Full figure plan: docs/project_knowledge/04_research_artefacts_and_figures.tex
Figure code lives in src/hemodyn_pinn/viz/. One script per paper figure.
NO jet colormaps anywhere. Use viridis/magma/cividis (sequential) or RdBu_r
(diverging, signed quantities only).

---

## 9. Mandatory tests (from architecture document)

These tests are non-negotiable. Do not mark a phase complete without them passing.

| Test file                  | What it covers                              | Required by |
|----------------------------|---------------------------------------------|-------------|
| test_mri_operator.py       | Forward operator correctness + noise stats  | Phase 1.4   |
| test_losses.py             | Each loss term (data, phys, BC) in isolation| Phase 1.5   |
| test_inference.py          | WSS-from-network path (autograd → τ_w)      | Phase 1.5   |
| test_anxplore_loader.py    | Mesh loading, boundary extraction           | Phase 1.2   |
| test_bulge_scaler.py       | Controlled severity axis                    | Phase 2     |
| test_bhpo.py               | Smoke test for BHPO module                  | Phase 3     |

---

## 10. Code conventions

- Python 3.11+. Type hints throughout. NumPy-style docstrings.
- No ReLU activations in the PINN backbone (breaks second-order autograd).
- No jet colormaps (ever).
- No hard-coded random seeds. Use src/hemodyn_pinn/utils/seeds.py for all RNG.
- No localStorage (irrelevant for this project, but noted for any web artefacts).
- Prefer functional/pure code; classes for stateful components (network, trainer,
  optimiser wrapper).
- All figure output: vector PDF or 600 dpi PNG. Scalar bars labelled with units.
- File locations:
    src/hemodyn_pinn/   ← all importable modules
    scripts/            ← pipeline entry points (numbered, sequential)
    src/hemodyn_pinn/viz/ ← figure-generation code (one script per figure)

---

## 11. Key references

Goetz et al. (2024). AnXplore: a comprehensive FSI study of 101 intracranial
  aneurysms. Front. Bioeng. Biotechnol. DOI: 10.3389/fbioe.2024.1433811

Tancik et al. (2020). Fourier features let networks learn high-frequency
  functions in low-dimensional domains. NeurIPS.
  https://arxiv.org/abs/2006.10739

Wang, Wang, Perdikaris (2021). On the eigenvector bias of Fourier feature
  networks: from regression to solving multi-scale PDEs with PINNs. CMAME.
  https://arxiv.org/abs/2012.10047

Karniadakis et al. (2021). Physics-informed machine learning. Nat. Rev. Phys.
  https://doi.org/10.1038/s42254-021-00314-5

Yang, Meng, Karniadakis (2021). B-PINNs: Bayesian physics-informed neural
  networks for forward and inverse PDE problems with noisy data. JCP.
  https://arxiv.org/abs/2003.06097

Snoek, Larochelle, Adams (2012). Practical Bayesian optimization of machine
  learning algorithms. NeurIPS.  [for BHPO section of paper]

---

## 12. Open questions and active risks

| # | Item                                                   | Status   |
|---|--------------------------------------------------------|----------|
| 1 | Exact Re target for Phase 1 CFD (set U to give Re<<1) | CLOSED   |
|   |   U = 2×10⁻⁵ m/s → Re = ρUL/μ = 1060×2e-5×0.01628/3.5e-3 ≈ 0.10 |          |
|   |   Confirmed Stokes regime (Re << 1). Use this U in all Phase 1–2 runs.  |    |
| 2 | Whether full_dataset/ contains all 101 cases or just  | CLOSED (101 cases) |
|   | Fluid_0.vtk — inspect directory before building loader |          |
| 3 | Boundary face extraction for cases A/B/C/R (no surface | CLOSED   |
|   | tris) — implemented and tested in anxplore_loader.py   |          |
| 4 | AnXplore inlet/outlet topology                        | CLOSED   |
|   |   Both open faces at y=0, outward normal (0,-1,0).    |          |
|   |   Two separate circular disks: left (x<0) = inlet,   |          |
|   |   right (x>0) = outlet.  Tube radius ≈ 2 mm.         |          |
|   |   Detected by: n_y < -0.9999 at y_min, split at x=0. |          |
| 5 | hydra-optuna-sweeper Phase 3 version                  | DEFERRED |

---

## 13. Environment notes

- Virtual environment: .venv at repo root (uv pip).
- Python location (Git Bash / Windows): .venv/Scripts/python
- Python location (Linux / macOS):      .venv/bin/python
- FEniCSx: installed out-of-band via apt or conda-forge (NOT in requirements.txt).
- mypy capped at <2.0 (2.x wheels are large and cause uv timeout on slow connections;
  raise UV_HTTP_TIMEOUT or switch to 2.x when bandwidth allows).
- hydra-optuna-sweeper: commented out in requirements.txt; add at Phase 3.
- Shell scripts: written for Git Bash compatibility (no mapfile, no stat -c,
  no process substitution). All heavy logic delegated to Python.

---

## 14. Project knowledge documents

Six tex reference documents live in docs/project_knowledge/.
Claude Code CAN read these directly — do so whenever
a task touches the topics they cover. Do not rely on CLAUDE.md summaries alone for implementation detail; go to the source document.

| # | Filename                                    | Read when...                                          |
|---|---------------------------------------------|-------------------------------------------------------|
| 01| 01_hemodynamics_glossary.tex                | Encountering an unfamiliar hemodynamics / imaging /   |
|   |                                             | clinical term. Authoritative definitions for the      |
|   |                                             | project (WSS, TAWSS, OSI, VNR, partial-volume, etc.)  |
| 02| 02_fluid_dynamics_primer.tex                | Writing or reviewing: NS equations, Stokes limit,     |
|   |                                             | WSS formula, boundary layers, non-dimensionalisation, |
|   |                                             | the synthetic MRI forward operator, or any scaling    |
|   |                                             | argument. Also: Womersley analytical reference soln.  |
| 03| 03_ml_concepts.tex                          | Writing or reviewing: network architecture, RFF,      |
|   |                                             | spectral bias, PINN loss terms, training pitfalls,    |
|   |                                             | optimiser recipe, WSS-via-autograd, B-PINNs, or any   |
|   |                                             | evaluation metric. The do/don't lists are binding.    |
| 04| 04_research_artefacts_and_figures.tex       | Producing ANY figure or table for the paper. Contains |
|   |                                             | exact content, purpose, and reader-takeaway for all   |
|   |                                             | 12 figures and 3 tables. One script per figure rule   |
|   |                                             | is defined here. Check before writing viz code.       |
| 05| 05_results_interpretation.tex               | Interpreting metric values, writing the Discussion    |
|   |                                             | section, or making any claim about clinical relevance.|
|   |                                             | Contains the five-layer interpretation framework and  |
|   |                                             | the per-metric physical translation tables.           |
| 06| 06_fenicsx_cfd_implementation.tex      | Any question about FEniCSx implementation       |

### Folder structure for docs
docs/
└── project_knowledge/
    ├── 01_hemodynamics_glossary.tex
    ├── 02_fluid_dynamics_primer.tex
    ├── 03_ml_concepts.tex
    ├── 04_research_artefacts_and_figures.tex
    ├── 05_results_interpretation.tex
    └── 06_fenicsx_cfd_implementation.tex

These files are READ-ONLY reference material. Do not modify them.
Do not summarise them back into CLAUDE.md — that creates a stale copy.
Instead, record only decisions and calibrated values derived from them.
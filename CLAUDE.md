---
output:
  pdf_document: default
  html_document: default
editor_options:
  markdown:
    wrap: sentence
---

# CLAUDE.md — hemodyn-pinn

**PINN-based Wall Shear Stress Recovery from Sparse 4D-flow MRI**

Navigation map for Claude Code. Last updated: 2026-05-25.

---

## 0. Session startup — READ THIS FIRST

1. Read this file.
2. Read the §10 docs relevant to the task — they override this file.
3. Check §4 for current phase and the immediate next step.

### Where things live

| What | Where |
|---|---|
| Knowledge docs | `docs/project_knowledge/` (six .tex files) |
| Source modules | `src/hemodyn_pinn/` |
| Pipeline scripts | `scripts/` (numbered; run in order) |
| Hydra configs | `configs/` |
| Tests | `tests/` |
| Interim result notes | `notes/results/` |

### After a gap
Run `git log --oneline -10` and read changed files before assuming the codebase matches this file.

### Model config
Default: `claude-sonnet-4-6`. Background: `claude-haiku-4-5-20251001`. No Opus unless requested.

---

## 1. Project identity

**Goal:** Recover wall shear stress (WSS) from sparse, noisy synthetic 4D-flow MRI (downsampled CFD). Characterise dependence on voxel size, noise level, and geometry.  
**Output:** Methods + benchmark paper. Primary artefact: open benchmark (CFD fields + synthetic MRI inputs + checkpoints).  
**Target venues:** CMAME, Computers in Biology and Medicine, Medical Image Analysis, JCP.

---

## 2. Architectural decisions

### 2.1 Data sources

| Role | Source | Status |
|---|---|---|
| Primary | AnXplore (Goetz et al. 2024, MIT). 101 patient-derived aneurysms. | In use |
| Secondary | Vascular Model Repository (VMR). | Phase 2 — re-check availability |
| Tertiary | AneuX / AneuriskWeb (CC BY-NC). | Comparison figures only; do NOT redistribute |

We generate our own CFD ground truth. Do NOT use Goetz et al.'s FSI outputs.

### 2.2 Physics
Steady incompressible Newtonian Stokes (Re ≪ 1). Pulsatility, FSI, non-Newtonian: follow-up paper.  
Equations, scaling, and WSS formula: **doc 02**.

### 2.3 PINN architecture
Coordinate-based MLP: `(x,y,z)/L → (û,v̂,ŵ,p̂)`. Activations: `tanh` (default), `swish`, `gelu`. **Never ReLU** (breaks second-order autograd). Optional RFF encoder. Framework: PyTorch 2.x, from-scratch.  
Full spec: **doc 03**. Activation and RFF are BHPO search dimensions.

### 2.4 Training recipe
Adam → L-BFGS. Cosine LR. Collocation points resampled each epoch; optional wall-biased sampling. Pressure anchor at one outlet face. WSS computed via autograd only — no finite differences.  
Loss weights and HP defaults: `configs/model/*.yaml`. Details: **doc 03**.

### 2.5 BHPO
GP surrogate (scikit-optimize `gp_minimize`), 10-dim search space. Objective: WSS NRMSE on 20 % held-out wall faces, **caseC only**. Winner retrained at full budget via `04c`.  
Config: `configs/bhpo/default.yaml`.

### 2.6 Toolstack

| Purpose | Tool |
|---|---|
| Deep learning | PyTorch 2.x |
| CFD solver | FEniCSx (dolfinx 0.10) — **Linux/WSL only** |
| 3D viz | PyVista 0.44+ |
| 2D figures | matplotlib 3.9+, seaborn 0.13+ |
| Config | Hydra 1.3+ |
| Experiment tracking | W&B 0.19+ |
| HPO | scikit-optimize 0.10, Optuna 4.7+ |
| Stats | pingouin 0.5+ |
| Testing | pytest 8.2+ |
| Lint / types | ruff 0.4+, mypy 1.10 (<2.0) |

---

## 3. Project tree

```
hemodyn-pinn/
├── configs/
│   ├── config.yaml              ← Hydra root; output dirs, eval defaults
│   ├── geometry/                ← caseA/B/C/R.yaml
│   ├── cfd/                     ← stokes_default.yaml
│   ├── mri/                     ← voxel_0p5mm … voxel_2p5mm.yaml
│   ├── model/                   ← pinn_base.yaml, pinn_rff.yaml
│   └── bhpo/                    ← default.yaml (search + retrain budgets)
├── src/hemodyn_pinn/
│   ├── geometry/
│   │   ├── anxplore_loader.py   ← VTK load; boundary faces from tets
│   │   ├── vmr_loader.py        ← Phase 2
│   │   ├── bulge_scaler.py      ← controlled-severity geometry axis
│   │   ├── parametric.py        ← idealised stenosis-bifurcation mesh
│   │   └── sdf.py               ← SDF for hard no-slip
│   ├── cfd/
│   │   ├── stokes_solver.py     ← FEniCSx P1-P1 BP Stokes (MINRES/GAMG)
│   │   ├── ns_solver.py         ← Phase 2+
│   │   └── postprocess.py       ← save/load .npz + XDMF; WSS statistics
│   ├── mri/
│   │   ├── operator.py          ← SyntheticMRIOperator: voxel-average + noise
│   │   ├── voxelise.py
│   │   └── noise.py
│   ├── pinn/
│   │   ├── networks.py          ← MLP, RFFEncoder, PINNNetwork; L_SCALE, U_SCALE, MU
│   │   ├── losses.py            ← data / physics / BC / anchor losses
│   │   ├── sampling.py          ← collocation + wall-BC sampling; wall-biased variant
│   │   ├── trainer.py           ← PINNTrainer + PINNConfig (Adam → L-BFGS)
│   │   └── inference.py         ← WSS via autograd; velocity prediction
│   ├── bhpo/                    ← space.py, objective.py, search.py
│   ├── bayesian/                ← B-PINN (Phase 4)
│   ├── baselines/
│   │   ├── trilinear_fd.py      ← trilinear interp + FD WSS baseline
│   │   └── supervised_mlp.py    ← data-only MLP baseline
│   ├── eval/
│   │   ├── field_metrics.py     ← NRMSE, R², MAE (RMS-normalised)
│   │   ├── wss_metrics.py       ← peak WSS error, mean relative error
│   │   ├── bland_altman.py      ← Bland–Altman stats + ICC(1,1)
│   │   └── conservation.py      ← flow-rate proxy; z-profile
│   ├── viz/
│   │   ├── style.py             ← matplotlib PAPER_RC; PyVista white/viridis theme
│   │   ├── render_3d.py         ← PyVista point-cloud WSS renders (single / comparison / error)
│   │   └── plots_2d.py          ← loss curves, WSS scatter, BA, heatmap, conservation, BHPO trace
│   └── utils/
│       ├── io.py
│       ├── seeds.py             ← all RNG seeds — never hard-code elsewhere
│       └── logging.py
├── scripts/
│   ├── 01_fetch_geometries.py
│   ├── 02_run_cfd.py            ← requires WSL
│   ├── 03_generate_synthetic_mri.py
│   ├── 04_train_pinn.py         ← fixed-HP baseline training
│   ├── 04b_bhpo_search.py       ← BHPO search (geometry=caseC only)
│   ├── 04c_train_pinn_with_bhpo.py ← retrain with winning HPs
│   ├── 05_evaluate.py           ← WSS metrics + BA + conservation; writes data/results/
│   ├── 06_run_sweep.py          ← discover all checkpoints → sweep_results.csv
│   └── 07_make_all_figures.py   ← figs 3,5,6,8,9,B,L → paper/figures/
├── notes/
│   └── results/
│       └── baseline_caseC_1mm.md  ← Phase 1.5 baseline (NRMSE=1.661, root cause, fix)
├── data/                        ← NOT committed
│   ├── geometries/anxplore/
│   ├── cfd/                     ← solution.npz per case
│   ├── synthetic_mri/           ← mri_obs.npz per case/voxel
│   ├── checkpoints/             ← <run_id>/best_model.pt + bhpo_params.json
│   ├── bhpo_runs/               ← best_params.json, trial_log.csv
│   └── results/                 ← eval_metrics.json, wss_comparison.npz, sweep_results.csv
├── tests/
├── notebooks/
└── paper/
    ├── main.tex
    ├── refs.bib
    └── figures/                 ← written by 07_make_all_figures.py
```

---

## 4. Phase plan

| Phase | Scope | Status |
|---|---|---|
| 0–1.4 | Setup, geometry, CFD, synthetic MRI | COMPLETE |
| 1.5 | Prototype PINN end-to-end | COMPLETE |
| 3-prep | PINN tweaks + BHPO module | COMPLETE |
| 3 | BHPO search + retrain + evaluation pipeline | **IN PROGRESS** |
| 2 | Core results figures at fixed HPs | not started |
| 4 | Ablation, supplementary, paper draft | not started |

**Next deliverable (M3 Pro / Linux/WSL):**
```
# Standard (baseline with Fixes E + D):
python scripts/04b_bhpo_search.py geometry=caseC
python scripts/04c_train_pinn_with_bhpo.py geometry=caseC

# With hard SDF + vector potential (Fixes A + B, recommended):
python scripts/04b_bhpo_search.py geometry=caseC bhpo.use_hard_sdf=true bhpo.use_vec_potential=true
python scripts/04c_train_pinn_with_bhpo.py geometry=caseC bhpo.use_hard_sdf=true bhpo.use_vec_potential=true
python scripts/04c_train_pinn_with_bhpo.py geometry=caseA bhpo.use_hard_sdf=true bhpo.use_vec_potential=true

python scripts/05_evaluate.py geometry=caseC
python scripts/05_evaluate.py geometry=caseA
python scripts/06_run_sweep.py
python scripts/07_make_all_figures.py
```

**Latest phase record (Fixes E,A,B,D + Optuna BHPO, 2026-06-03):**  
Implemented four physics/architecture improvements:
- **Fix E**: `wall_bias_frac=0.4` — near-wall collocation sampling (free, highest single-knob impact)  
- **Fix A**: Hard SDF no-slip (`geometry/sdf.py` + `WallSDF`; `networks.py` `use_hard_sdf`; analytical WSS in `inference.py`)  
- **Fix B**: Vector potential div-free network (`networks.py` `use_vec_potential`, `_curl`, `forward_raw`; `losses.py` `skip_div_loss`)  
- **Fix D**: Self-adaptive loss weights (`losses.py` `SelfAdaptiveLoss`; gradient reversal in `trainer.py`)  
BHPO stack rewritten: Optuna TPE (`bhpo/search.py`), Optuna-native suggest (`bhpo/space.py`), MPS auto-detect + arch flags (`bhpo/objective.py`). Config tuned for M3 Pro: `n_adam_trial=10000`, `device="auto"`, `backend="optuna"`.  
Full derivation and caveats: `notes/implementation_fixes_E_A_B_D.md`.  
Earlier phase records: `git log --oneline`.

---

## 5. Calibrated parameters

### Physical constants

| Quantity | Value | Source in code |
|---|---|---|
| μ | 3.5×10⁻³ Pa·s | `stokes_solver.MU_BLOOD` |
| ρ | 1060 kg/m³ | `stokes_solver.RHO_BLOOD` |
| L_SCALE | 0.01628 m | `networks.L_SCALE` |
| U_SCALE | 2×10⁻⁵ m/s (Re ≈ 0.10) | `networks.U_SCALE` |
| VTK coords | mm | multiply × 1e-3 before SI computation |

### RNG seeds (all in `utils/seeds.py`)

| Constant | Value | Purpose |
|---|---|---|
| RFF_SEED | 2718 | RFFEncoder bandwidth matrix |
| PINN_COLLOC_SEED | 3141 | Per-epoch collocation sampling |
| PINN_TRAIN_SEED | 9999 | Network weight init |
| BHPO_SEED | 7777 | scikit-optimize GP random state |
| BHPO_VAL_SEED | 4321 | Train/val wall-face split |

### AnXplore mesh summary

| Case | Nodes | Tets | Notes |
|---|---|---|---|
| caseC | 147,432 | 844,320 | Simplest — BHPO reference case |
| caseA | 159,155 | 912,713 | |
| caseR | 178,675 | 1,028,394 | |
| caseB | 170,087 | 975,723 | Most complex |
| full_dataset | 158,840 | 910,886 | 33k explicit surface tris |

caseA/B/C/R: no explicit surface triangles — boundary faces extracted from tets in `anxplore_loader.py`.  
Inlet/outlet: both at y = y_min, outward normal (0,−1,0); left (x < 0) = inlet, right (x > 0) = outlet.

---

## 6. Environment

- Virtual env: `.venv/` (uv pip). Windows: `.venv/Scripts/python` | Linux: `.venv/bin/python`.
- FEniCSx: Linux/WSL only: `sudo add-apt-repository ppa:fenics-packages/fenics && sudo apt install fenicsx`
- dolfinx 0.10 API: `create_mesh(comm, cells, element, coords)` — element is 3rd argument.
- Solver: P1-P1 Brezzi-Pitkäranta (MINRES/GAMG). **Do not revert to P2-P1** — OOM on 8 GB. See `memory/project_p1p1_bp_element.md`.
- mypy capped at `<2.0`. `hydra-optuna-sweeper` commented out (Optuna version conflict — revisit Phase 4).

---

## 7. Mandatory tests

| File | Covers | Required by |
|---|---|---|
| test_anxplore_loader.py | Mesh load, boundary extraction | Phase 1.2 |
| test_mri_operator.py | Forward operator + noise stats | Phase 1.4 |
| test_losses.py | Each loss term | Phase 1.5 |
| test_inference.py | WSS autograd path | Phase 1.5 |
| test_bulge_scaler.py | Severity axis | Phase 2 |
| test_bhpo.py | BHPO smoke test | Phase 3 |

Do not mark a phase complete without its tests passing.

---

## 8. Code conventions

- Python 3.11+. Type hints throughout. NumPy-style docstrings.
- **Never ReLU** in PINN backbone. **Never jet** colormaps.
- All RNG seeds in `utils/seeds.py` — never hard-code elsewhere.
- Figures: vector PDF or 600 dpi PNG. Colormaps: `viridis`/`magma`/`cividis` (sequential); `RdBu_r` (diverging, signed quantities only).
- No comments unless the WHY is non-obvious. No multi-line docstrings.

---

## 9. Key references

- Goetz et al. (2024). AnXplore. *Front. Bioeng. Biotechnol.* 10.3389/fbioe.2024.1433811
- Tancik et al. (2020). Fourier features. *NeurIPS.* arXiv:2006.10739
- Wang, Wang, Perdikaris (2021). Adaptive PINN loss weights. *CMAME.* arXiv:2012.10047
- Karniadakis et al. (2021). Physics-informed ML. *Nat. Rev. Phys.*
- Yang, Meng, Karniadakis (2021). B-PINNs. *JCP.* arXiv:2003.06097

---

## 10. Knowledge documents (READ-ONLY — do not modify)

| Doc | File | Read when |
|---|---|---|
| 01 | 01_hemodynamics_glossary.tex | Unfamiliar term (WSS, TAWSS, OSI, VNR, …) |
| 02 | 02_fluid_dynamics_primer.tex | NS equations, Stokes limit, WSS formula, MRI forward operator |
| 03 | 03_ml_concepts.tex | Network arch, RFF, loss terms, optimiser recipe, B-PINNs, metrics |
| 04 | 04_research_artefacts_and_figures.tex | All paper figures (12 figs + 3 tables) |
| 05 | 05_results_interpretation.tex | Metric interpretation, Discussion writing |
| 06 | 06_fenicsx_cfd_implementation.tex | Any FEniCSx/dolfinx question |
| 07 | 07_research_question_and_motivation.tex | Project overview, clinical problem, research questions C1–C4, prior-work positioning |

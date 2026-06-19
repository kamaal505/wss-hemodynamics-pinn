# Graph Report - .  (2026-06-20)

## Corpus Check
- 20 files · ~46,760 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 941 nodes · 1975 edges · 55 communities (40 shown, 15 thin omitted)
- Extraction: 77% EXTRACTED · 23% INFERRED · 0% AMBIGUOUS · INFERRED: 456 edges (avg confidence: 0.68)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]
- [[_COMMUNITY_Community 48|Community 48]]
- [[_COMMUNITY_Community 49|Community 49]]
- [[_COMMUNITY_Community 50|Community 50]]
- [[_COMMUNITY_Community 51|Community 51]]
- [[_COMMUNITY_Community 52|Community 52]]
- [[_COMMUNITY_Community 53|Community 53]]

## God Nodes (most connected - your core abstractions)
1. `PINNNetwork` - 64 edges
2. `PINNTrainer` - 49 edges
3. `SelfAdaptiveLoss` - 41 edges
4. `PINNNetwork` - 32 edges
5. `compute_wss()` - 31 edges
6. `AnxploreMesh` - 28 edges
7. `SyntheticMRIOperator` - 28 edges
8. `WallSDF` - 27 edges
9. `BHPOObjective` - 26 edges
10. `main()` - 25 edges

## Surprising Connections (you probably didn't know these)
- `Poiseuille flow analytical solution u_x = 4y(1-y), p(x)=8(1-x)` --semantically_similar_to--> `Analytical WSS formula under hard-SDF constraint`  [INFERRED] [semantically similar]
  docs/project_knowledge/Test problem 1 - Channel flow (Poiseuille flow) — FEniCSx tutorial.pdf → notes/implementation_fixes.md
- `main()` --references--> `pinn_base config`  [INFERRED]
  scripts/04_train_pinn.py → configs/model/pinn_base.yaml
- `main()` --references--> `bhpo default config`  [INFERRED]
  scripts/04b_bhpo_search.py → configs/bhpo/default.yaml
- `eval block — run_suffix=bhpo, device, wss_batch_size=5000` --configures--> `main()`  [EXTRACTED]
  configs/config.yaml → scripts/05_evaluate.py
- `RFF params — rff_features=128, rff_sigma=2.0 (tune in {1,2,5,10})` --configures--> `RFFEncoder`  [EXTRACTED]
  configs/model/pinn_rff.yaml → src/hemodyn_pinn/pinn/networks.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Magnitude collapse fix (relative loss + inlet term + checkpoint selection + lambda_data search)** — concept_relative_data_loss, concept_inflow_magnitude_constraint, concept_observation_misfit_checkpoint_selection, concept_lambda_data_bhpo_search_dimension, concept_magnitude_collapse [EXTRACTED 0.90]
- **PINN BHPO→retrain→evaluate→sweep pipeline** — scripts_04b_bhpo_search_main, scripts_04c_train_pinn_with_bhpo_main, scripts_05_evaluate_main, scripts_06_run_sweep_main [INFERRED 0.85]
- **Physics/architecture enforcement fixes (A,B,D,E)** — concept_hard_sdf_noslip, concept_vector_potential_divfree, concept_self_adaptive_loss_weights, concept_wall_biased_sampling [INFERRED 0.85]

## Communities (55 total, 15 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.05
Nodes (43): Bayesian Hyperparameter Optimisation for the PINN.  Public API ---------- BHPOOb, BHPOObjective, BHPOObjective.optuna_objective, BHPO objective function: WSS NRMSE on held-out wall faces.  The ``BHPOObjective`, Optuna objective: sample HPs, train PINN, return WSS NRMSE., skopt interface: accepts a raw parameter list., Callable objective for BHPO search (Optuna and skopt compatible).      Parameter, auto_device() (+35 more)

### Community 1 - "Community 1"
Cohesion: 0.06
Nodes (45): Compute WSS at wall triangle centroids via a DG(0) velocity gradient.      Alg, compute_velocity_jacobian(), compute_wss(), compute_wss_hard_sdf(), predict_velocity_field(), WSS inference from a trained PINNNetwork via autograd.  Standard path (use_hard_, Compute WSS using the analytical hard-SDF formula (Fix A).      Requires net.use, Dispatch to the appropriate WSS computation based on network flags.      Uses th (+37 more)

### Community 2 - "Community 2"
Cohesion: 0.05
Nodes (34): ActivationName, dim_velocity(), dim_wss(), MLP, nondim_coords(), nondim_velocity(), PINN network architectures for Stokes flow reconstruction.  All modules operate, Fully-connected network with configurable smooth activations. (+26 more)

### Community 3 - "Community 3"
Cohesion: 0.07
Nodes (52): Figure, fig3_cfd_ground_truth(), fig5_wss_comparison(), fig6_sensitivity_heatmap(), fig8_bland_altman(), fig9_conservation(), figB_bhpo_convergence(), figL_loss_curves() (+44 more)

### Community 4 - "Community 4"
Cohesion: 0.06
Nodes (45): Architecture flags — use_hard_sdf/use_vec_potential/use_adaptive_weights (fixed per run), configs/bhpo/default.yaml — BHPO search & retrain config (M3 Pro tuned), Retrain settings — source_case_id=caseC, n_adam_full=50000, n_lbfgs_full=5000, Search settings — n_calls=60, n_adam_trial=10000, backend=optuna, device=auto, CLAUDE.md §4 — Phase plan (Phase 3 IN PROGRESS), eval block — run_suffix=bhpo, device, wss_batch_size=5000, config.yaml — Hydra root defaults & output dirs, configs/README.md — Hydra config group documentation (+37 more)

### Community 5 - "Community 5"
Cohesion: 0.10
Nodes (28): Observation-misfit checkpoint selection, epoch_rng(), Return a deterministic RNG for a given epoch.      Using epoch-dependent seeds e, Draw a random subset of interior points as collocation points.      Parameters, Draw a random subset of wall centroid points.      Parameters     ----------, Draw collocation points with a fraction concentrated near the wall.      ``wall_, sample_collocation(), sample_collocation_biased() (+20 more)

### Community 6 - "Community 6"
Cohesion: 0.09
Nodes (24): Unsigned wall-distance function for the hard no-slip SDF constraint.  Computes t, Approximate unsigned distance from arbitrary lumen points to the wall.      Para, Return distances (m) from each query point to the nearest wall face.          Pa, Return non-dimensional distances for non-dimensional query coords.          Para, WallSDF, _find_indices(), Return integer indices of `sampled` rows within `pool`.      Uses a hash of (x,, ndarray (+16 more)

### Community 7 - "Community 7"
Cohesion: 0.12
Nodes (29): load_solution(), Load a previously saved solution.      Parameters     ----------     npz_path:, CLAUDE.md §5 — Calibrated physical constants & RNG seeds, CLAUDE.md — project navigation map for Claude Code, Batched WSS autograd inference to avoid OOM, BHPO search → retrain → evaluate pipeline, run_id naming convention (case_voxel_suffix), voxel_tag string format helper (voxel_XpYmm) (+21 more)

### Community 8 - "Community 8"
Cohesion: 0.11
Nodes (10): TestAnxploreMeshProperties, AnxploreMesh, Load caseA (tet-only, no explicit surface triangles)., Parent artery x-span should be 16.279 mm (±0.002 mm)., Mean edge length should be ≈ 0.1563 mm (mesh_stats.txt ground truth)., Tet-only mesh: wall_tri must have been extracted (not from explicit tris)., Load full_dataset/Fluid_0.vtk (explicit surface triangles present)., Explicit triangles in Fluid_0.vtk: 33,116 (from mesh_stats.txt). (+2 more)

### Community 9 - "Community 9"
Cohesion: 0.13
Nodes (16): PINNConfig, _make_data(), _make_trainer(), Tests for PINNTrainer.__init__ with the new architecture flags (Fixes A, B, D, E, use_hard_sdf=True without wall_pts_m must raise ValueError., When use_hard_sdf=True, the trainer must zero out the BC weight., _skip_div must be True whenever use_vec_potential=True., _sa_loss must not be None when use_adaptive_weights=True. (+8 more)

### Community 10 - "Community 10"
Cohesion: 0.15
Nodes (25): Magnitude collapse (trivial Stokes solution), Magnitude-collapse fix notes (Fixes F,G,H,I), bc_loss(), data_loss(), _grad(), inlet_loss(), pressure_anchor_loss(), PINN loss terms for steady Stokes flow reconstruction.  Non-dimensional Stokes e (+17 more)

### Community 11 - "Community 11"
Cohesion: 0.12
Nodes (14): Apply voxel averaging and noise to CFD nodal data.          Parameters         -, assign_nodes_to_voxels(), Cartesian voxel grid definition and mesh-node-to-voxel assignment.  All coordina, Axis-aligned Cartesian voxel grid in millimetre coordinates.      Attributes, Build a grid that covers the bounding box of *points_mm*.          Parameters, Total number of voxels., Return (nx, ny, nz, 3) array of voxel centre coordinates in mm., Convert (ix, iy, iz) voxel indices to flat C-order indices. (+6 more)

### Community 12 - "Community 12"
Cohesion: 0.11
Nodes (18): TestBCLoss, TestPressureAnchorLoss, ExactStokesNet, Tensor, Tests for hemodyn_pinn.pinn.losses.  Each loss term is tested in isolation with, Backprop through data_loss reaches network parameters., Returns u=y², v=0, w=0, p=2*x — an exact non-dim Stokes solution.      Verificat, Residual must vanish for the analytical Stokes solution. (+10 more)

### Community 13 - "Community 13"
Cohesion: 0.12
Nodes (21): RMS-normalised NRMSE (consistent with BHPO objective), WSS NRMSE on held-out wall faces as BHPO objective, compute_field_metrics(), mae(), nrmse(), r2(), Scalar accuracy metrics for field comparisons.  All functions accept flat or mul, Root-mean-square error normalised by RMS of the reference field.      Parameters (+13 more)

### Community 14 - "Community 14"
Cohesion: 0.15
Nodes (22): BoundaryParts, _compute_wss_fn_scalar(), _compute_wss_fn_vector(), _create_dolfinx_mesh(), _extract_nodal_values(), _mark_boundary_facets(), Steady Stokes flow solver using FEniCSx (dolfinx 0.10) Taylor-Hood P2-P1.  Geo, Convert an AnxploreMesh to a dolfinx Mesh (SI units, metres). (+14 more)

### Community 15 - "Community 15"
Cohesion: 0.12
Nodes (15): AnxploreMesh, _compute_outward_normals_from_centroid(), load_anxplore(), AnXplore fluid mesh loader with boundary extraction and mesh quality checks.  Ha, Compute outward normals using the mesh centroid as interior reference.      Used, Loaded AnXplore fluid mesh with precomputed boundary geometry.      Attributes, (N, 3) float64 node coordinates in **metres** (SI units)., (W, 3) float64 centroid of each wall triangle in mm. (+7 more)

### Community 16 - "Community 16"
Cohesion: 0.17
Nodes (13): MRIConfig, Apply the synthetic 4D-flow MRI forward operator to CFD nodal data.      Paramet, Configuration for the synthetic MRI forward operator.      Parameters     ------, SyntheticMRIOperator, main(), DictConfig, Generate synthetic 4D-flow MRI observations from CFD solution data.  Usage (Wind, Voxel average of a constant field must equal that constant in every voxel. (+5 more)

### Community 17 - "Community 17"
Cohesion: 0.14
Nodes (12): Return a dict of mesh quality metrics.          Edge-length statistics are compu, add_gaussian_noise(), Additive Gaussian noise model for synthetic 4D-flow MRI.  The noise model is:, Compute noise standard deviation from VENC and VNR.      Parameters     --------, Add independent Gaussian noise to each velocity component.      Parameters     -, sigma_from_vnr(), ndarray, With many samples, noise should be zero-mean to within ~3σ/√N. (+4 more)

### Community 18 - "Community 18"
Cohesion: 0.16
Nodes (6): PINNNetwork, A non-zero lambda_inlet must add a positive contribution to total., Full loss (momentum + div) ≥ momentum-only loss (non-negative div term)., TestLossesWithSDFVals, TestStokesResidualSkipDiv, TestTotalLoss

### Community 19 - "Community 19"
Cohesion: 0.15
Nodes (9): Self-adaptive loss weights (Fix D), Optimizer, Learnable loss-weight module (McClenny & Braga-Neto, 2023, JCP).      Each weigh, Compute weighted sum Σ_k exp(log_λ_k) · L_k.          Parameters         -------, Current weight values (detached, for logging)., SelfAdaptiveLoss, Reversing the gradient and taking an SGD step must increase λ when loss is large, lambda=0 is clamped to 1e-8 before log; must not raise. (+1 more)

### Community 20 - "Community 20"
Cohesion: 0.18
Nodes (11): Manages Adam → L-BFGS training of a PINNNetwork.      Parameters     ----------, Run the Adam optimisation phase with cosine-annealing LR., Metric used to pick the best checkpoint.          Selecting by total loss reward, Run L-BFGS refinement after Adam., Run the full Adam → L-BFGS training pipeline., PINNTrainer, TestAdaptiveWeightsInit (Fix D — _sa_loss), TestHardSDFInit (Fix A — _sdf_interior/_sdf_data, _effective_lambda_bc) (+3 more)

### Community 21 - "Community 21"
Cohesion: 0.19
Nodes (16): export_xdmf(), load_cfd_xdmf(), load_metadata(), Save and load Stokes CFD solutions (velocity, pressure, WSS).  Output layout und, Load scalar metadata from a CFD output directory., Return a summary of WSS statistics on the vessel wall only.      Parameters, Write all CFD outputs to XDMF+HDF5 (doc 06 §7 output format).      Requires dolf, Load CFD XDMF output using meshio (cross-platform, no dolfinx needed).      Para (+8 more)

### Community 22 - "Community 22"
Cohesion: 0.21
Nodes (11): detect_open_faces(), Identify inlet/outlet patches in the AnXplore boundary mesh.      Both open fa, TestStokesConfig (Re ≈ 0.1 regime check), test_detect_open_faces_real_cases (slow, parametrized caseA/B/C/R), TestDetectOpenFacesSynthetic, _make_two_disk_mesh (synthetic open-face geometry helper), test_stokes_solver.py — Stokes CFD solver tests, test_solve_stokes_smoke (dolfinx-only, skipped on Windows) (+3 more)

### Community 23 - "Community 23"
Cohesion: 0.17
Nodes (13): BHPOObjective._run_trial, src/hemodyn_pinn/README.md (package docs), PINNNetwork, Coordinate-based PINN for steady Stokes flow.      Input:  x̂ = (x̂, ŷ, ẑ) — non, PINNConfig, PINN training loop: Adam → L-BFGS with checkpointing.  Training recipe (CLAUDE.m, Hyperparameters for the PINN training loop.      Parameters     ----------     n, DictConfig (+5 more)

### Community 24 - "Community 24"
Cohesion: 0.16
Nodes (13): bland_altman_stats(), icc_one_way(), Bland–Altman agreement statistics and intraclass correlation coefficient.  Refer, Compute Bland–Altman agreement statistics.      Parameters     ----------     pr, Intraclass correlation coefficient ICC(1,1) — one-way random effects.      Treat, 06_run_sweep _run_evaluation, main(), DictConfig (+5 more)

### Community 25 - "Community 25"
Cohesion: 0.26
Nodes (6): _extract_boundary_faces(), Return boundary triangles extracted from a pure-tet mesh.      A boundary face i, TestExtractBoundaryFaces, _make_two_tet_mesh(), Face {1,2,3} is shared by both tets and must not be a boundary face., Two tets sharing face {1,2,3}; 6 boundary faces expected.      Vertices:

### Community 26 - "Community 26"
Cohesion: 0.18
Nodes (7): TestDataLoss, Relative data loss ≈ 1.0 when the network predicts zero velocity.          This, relative == absolute / mean(u_obs²)., Tiny PINN network for fast testing (2 hidden layers, 16 neurons)., Loss is zero when predicted velocity matches observations exactly., small_net(), TestInletLoss

### Community 27 - "Community 27"
Cohesion: 0.19
Nodes (7): ndarray, Tests for the synthetic MRI forward operator (Phase 1.4 REQUIRED).  Covers:   -, Interior voxels should accumulate more nodes than edge voxels., Return (points_mm, velocity_ms) on a regular Cartesian lattice., TestMRIObservationIO, TestPartialVolume, _uniform_box_nodes()

### Community 28 - "Community 28"
Cohesion: 0.24
Nodes (8): _compute_outward_normals(), Compute outward unit normals for boundary triangles.      Orientation is determi, TestOutwardNormals, _make_single_tet_mesh(), ndarray, Tests for anxplore_loader.py — Phase 1.2 mandatory test suite.  Fast unit tests, Normal must point away from the mesh centroid (outward)., A single regular tetrahedron; all 4 faces are boundary faces.

### Community 29 - "Community 29"
Cohesion: 0.21
Nodes (8): _make_dummy_solution(), Tests for the Stokes CFD solver module.  Fast unit tests (no dolfinx required,, detect_open_faces must correctly split the two open disks on real meshes., Return a minimal StokesSolution with synthetic arrays., Smoke test: solver runs on caseC and produces plausible results., test_detect_open_faces_real_cases(), test_solve_stokes_smoke(), TestPostprocess

### Community 30 - "Community 30"
Cohesion: 0.31
Nodes (6): Results from the FEniCSx Stokes solver.      All arrays are indexed consistent, Configuration for the steady Stokes solver.      Parameters     ----------, StokesConfig, StokesSolution, ndarray, TestStokesConfig

### Community 31 - "Community 31"
Cohesion: 0.22
Nodes (7): MRIObservation, Synthetic 4D-flow MRI forward operator.  Implements the forward model:      u_MR, Load a previously saved MRIObservation from an .npz file., Output of the synthetic MRI forward operator.      Attributes     ----------, Save to a compressed .npz file.          Parameters         ----------         p, Path, TestMRIObservationIO (save/load round-trip)

### Community 32 - "Community 32"
Cohesion: 0.36
Nodes (7): Count-weighted normal-velocity proxy for flow-rate conservation (face areas unavailable), flow_rate_conservation_error(), Flow-rate conservation diagnostics for PINN velocity fields.  Checks that the PI, Estimate mass-conservation error from face-centroid velocities.      Uses count-, Estimate flow rate Q(z) along z-axis cross-sections.      Samples a uniform grid, z_profile_flow_rate(), ndarray

### Community 33 - "Community 33"
Cohesion: 0.29
Nodes (8): Hard SDF no-slip (Fix A), Inflow magnitude constraint, Relative (scale-invariant) data loss, Vector-potential div-free network (Fix B), Wall-biased collocation sampling (Fix E), bhpo default config, pinn_base config, pinn_rff config

### Community 34 - "Community 34"
Cohesion: 0.29
Nodes (6): AnXplore dataset (Goetz et al. 2024, MIT license), fetch_and_inspect.sh (AnXplore acquisition), Goetz et al. (2024) — AnXplore paper, main(), DictConfig, Run steady Stokes CFD on one or more AnXplore cases.  Usage (Linux / WSL with

### Community 35 - "Community 35"
Cohesion: 0.29
Nodes (7): ExactStokesNet (analytic Stokes fixture), test_losses.py — per-term loss tests (Phase 1.5), TestSelfAdaptiveLoss (Fix D, gradient reversal), TestStokesResidualSkipDiv (Fix B, skip_div_loss), TestStokesResidualLoss (exact Stokes solution check), TestTotalLoss (weighted-sum consistency), TestLossesWithSDFVals (Fix A, sdf_vals kwargs)

### Community 36 - "Community 36"
Cohesion: 0.40
Nodes (4): _compute_wss_batched(), ndarray, Evaluate a trained PINN checkpoint against CFD ground truth.  Loads the best che, Compute WSS magnitudes in batches to avoid autograd OOM on large walls.      Ret

### Community 37 - "Community 37"
Cohesion: 0.40
Nodes (5): test_mri_operator.py — synthetic MRI forward operator tests (Phase 1.4 REQUIRED), TestNoiseModel (sigma_from_vnr, add_gaussian_noise), TestPartialVolume (boundary-voxel diagnostic), TestSyntheticMRIOperator (correctness, reproducibility), tests/README.md — suite overview (116 tests)

## Ambiguous Edges - Review These
- `__init__.py` → `compute_field_metrics()`  [AMBIGUOUS]
  src/hemodyn_pinn/eval/__init__.py · relation: references

## Knowledge Gaps
- **35 isolated node(s):** `DictConfig`, `Path`, `ndarray`, `ActivationName`, `Goetz et al. (2024) — AnXplore paper` (+30 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **15 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `__init__.py` and `compute_field_metrics()`?**
  _Edge tagged AMBIGUOUS (relation: references) - confidence is low._
- **Why does `PINNNetwork` connect `Community 23` to `Community 0`, `Community 1`, `Community 2`, `Community 4`, `Community 36`, `Community 7`, `Community 9`, `Community 10`, `Community 12`, `Community 18`, `Community 19`, `Community 20`, `Community 24`, `Community 26`?**
  _High betweenness centrality (0.250) - this node is a cross-community bridge._
- **Why does `PINNTrainer` connect `Community 20` to `Community 0`, `Community 33`, `Community 5`, `Community 6`, `Community 7`, `Community 9`, `Community 10`, `Community 19`, `Community 23`, `Community 31`?**
  _High betweenness centrality (0.185) - this node is a cross-community bridge._
- **Why does `load_anxplore()` connect `Community 15` to `Community 34`, `Community 7`, `Community 8`, `Community 16`, `Community 22`, `Community 23`, `Community 25`, `Community 28`, `Community 29`?**
  _High betweenness centrality (0.119) - this node is a cross-community bridge._
- **Are the 43 inferred relationships involving `PINNNetwork` (e.g. with `BHPOObjective` and `PINNConfig`) actually correct?**
  _`PINNNetwork` has 43 INFERRED edges - model-reasoned connections that need verification._
- **Are the 17 inferred relationships involving `PINNTrainer` (e.g. with `BHPOObjective` and `._run_trial()`) actually correct?**
  _`PINNTrainer` has 17 INFERRED edges - model-reasoned connections that need verification._
- **Are the 30 inferred relationships involving `SelfAdaptiveLoss` (e.g. with `Optimizer` and `PINNConfig`) actually correct?**
  _`SelfAdaptiveLoss` has 30 INFERRED edges - model-reasoned connections that need verification._
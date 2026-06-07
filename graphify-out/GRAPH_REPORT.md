# Graph Report - .  (2026-06-08)

## Corpus Check
- Corpus is ~42,753 words - fits in a single context window. You may not need a graph.

## Summary
- 966 nodes · 1942 edges · 68 communities (60 shown, 8 thin omitted)
- Extraction: 78% EXTRACTED · 22% INFERRED · 0% AMBIGUOUS · INFERRED: 426 edges (avg confidence: 0.67)
- Token cost: 444,394 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Hydra Configs & Geometry Cases|Hydra Configs & Geometry Cases]]
- [[_COMMUNITY_Collocation Sampling & Loss Refs|Collocation Sampling & Loss Refs]]
- [[_COMMUNITY_Hard-SDF Wall Distance|Hard-SDF Wall Distance]]
- [[_COMMUNITY_Synthetic MRI Forward Operator|Synthetic MRI Forward Operator]]
- [[_COMMUNITY_PINN Trainer Architecture Tests|PINN Trainer Architecture Tests]]
- [[_COMMUNITY_Voxelisation & MRI Application|Voxelisation & MRI Application]]
- [[_COMMUNITY_AnXplore Mesh Property Tests|AnXplore Mesh Property Tests]]
- [[_COMMUNITY_2D Figure Plotting|2D Figure Plotting]]
- [[_COMMUNITY_Stokes CFD Solver|Stokes CFD Solver]]
- [[_COMMUNITY_Field Accuracy Metrics|Field Accuracy Metrics]]
- [[_COMMUNITY_Project Constants & Pipeline|Project Constants & Pipeline]]
- [[_COMMUNITY_WSS Evaluation Pipeline|WSS Evaluation Pipeline]]
- [[_COMMUNITY_CFD Postprocessing IO|CFD Postprocessing I/O]]
- [[_COMMUNITY_MRI Noise Model|MRI Noise Model]]
- [[_COMMUNITY_Paper Figure Generation|Paper Figure Generation]]
- [[_COMMUNITY_PINN Adam-LBFGS Training Loop|PINN Adam-LBFGS Training Loop]]
- [[_COMMUNITY_WSS Inference Tests|WSS Inference Tests]]
- [[_COMMUNITY_BHPO Search Loop|BHPO Search Loop]]
- [[_COMMUNITY_PINN Network Architecture|PINN Network Architecture]]
- [[_COMMUNITY_BHPO Objective & Tests|BHPO Objective & Tests]]
- [[_COMMUNITY_Stokes Residual Loss Tests|Stokes Residual Loss Tests]]
- [[_COMMUNITY_AnXplore Mesh Loader|AnXplore Mesh Loader]]
- [[_COMMUNITY_Boundary Face Extraction Tests|Boundary Face Extraction Tests]]
- [[_COMMUNITY_Hard-SDF WSS Computation|Hard-SDF WSS Computation]]
- [[_COMMUNITY_Self-Adaptive Loss Weights|Self-Adaptive Loss Weights]]
- [[_COMMUNITY_Stokes Config & AnXplore Source|Stokes Config & AnXplore Source]]
- [[_COMMUNITY_Outward Normal Tests|Outward Normal Tests]]
- [[_COMMUNITY_Standard WSS Computation Tests|Standard WSS Computation Tests]]
- [[_COMMUNITY_Total Loss Computation|Total Loss Computation]]
- [[_COMMUNITY_Optuna HP Sampling Tests|Optuna HP Sampling Tests]]
- [[_COMMUNITY_InletOutlet Detection Tests|Inlet/Outlet Detection Tests]]
- [[_COMMUNITY_PINN Training Script|PINN Training Script]]
- [[_COMMUNITY_BHPO Device & Param Tests|BHPO Device & Param Tests]]
- [[_COMMUNITY_AnXplore Loading Integration Tests|AnXplore Loading Integration Tests]]
- [[_COMMUNITY_3D WSS Rendering|3D WSS Rendering]]
- [[_COMMUNITY_Loss Function Unit Tests|Loss Function Unit Tests]]
- [[_COMMUNITY_Stokes Solver & Postproc Tests|Stokes Solver & Postproc Tests]]
- [[_COMMUNITY_BHPO Search Space|BHPO Search Space]]
- [[_COMMUNITY_Data Loss Function|Data Loss Function]]
- [[_COMMUNITY_Pressure Anchor Loss|Pressure Anchor Loss]]
- [[_COMMUNITY_MLP Backbone|MLP Backbone]]
- [[_COMMUNITY_Velocity Jacobian Computation|Velocity Jacobian Computation]]
- [[_COMMUNITY_Velocity Field Prediction|Velocity Field Prediction]]
- [[_COMMUNITY_Boundary Condition Loss|Boundary Condition Loss]]
- [[_COMMUNITY_Loss Term Test Suite|Loss Term Test Suite]]
- [[_COMMUNITY_Fluid_0 Mesh Loading Tests|Fluid_0 Mesh Loading Tests]]
- [[_COMMUNITY_Hard-SDF Network Tests|Hard-SDF Network Tests]]
- [[_COMMUNITY_Flow Conservation Diagnostics|Flow Conservation Diagnostics]]
- [[_COMMUNITY_Random Fourier Features Encoder|Random Fourier Features Encoder]]
- [[_COMMUNITY_WSS Pipeline Test Overview|WSS Pipeline Test Overview]]
- [[_COMMUNITY_Bland-Altman Agreement Stats|Bland-Altman Agreement Stats]]
- [[_COMMUNITY_WSS Auto-Dispatch|WSS Auto-Dispatch]]
- [[_COMMUNITY_Loss Term Implementation|Loss Term Implementation]]
- [[_COMMUNITY_MRI Operator Test Suite|MRI Operator Test Suite]]
- [[_COMMUNITY_PINN Network Tests|PINN Network Tests]]
- [[_COMMUNITY_Vector Potential Tests|Vector Potential Tests]]
- [[_COMMUNITY_Stokes Solver Test Suite|Stokes Solver Test Suite]]
- [[_COMMUNITY_BHPO Trial Execution|BHPO Trial Execution]]
- [[_COMMUNITY_MRI Observation IO|MRI Observation I/O]]
- [[_COMMUNITY_Combined SDF+VecPotential Tests|Combined SDF+VecPotential Tests]]
- [[_COMMUNITY_AnXplore Fetch Script|AnXplore Fetch Script]]
- [[_COMMUNITY_Synthetic MRI Generation Script|Synthetic MRI Generation Script]]
- [[_COMMUNITY_CFD Package Init|CFD Package Init]]
- [[_COMMUNITY_Top-Level Package Init|Top-Level Package Init]]
- [[_COMMUNITY_MRI Package Init|MRI Package Init]]
- [[_COMMUNITY_PINN Package Init|PINN Package Init]]
- [[_COMMUNITY_Viz Package Init|Viz Package Init]]

## God Nodes (most connected - your core abstractions)
1. `PINNNetwork` - 67 edges
2. `PINNTrainer` - 37 edges
3. `SelfAdaptiveLoss` - 36 edges
4. `AnxploreMesh` - 29 edges
5. `PINNNetwork` - 29 edges
6. `AnxploreMesh` - 28 edges
7. `SyntheticMRIOperator` - 28 edges
8. `WallSDF` - 27 edges
9. `VoxelGrid` - 26 edges
10. `BHPOObjective` - 25 edges

## Surprising Connections (you probably didn't know these)
- `Poiseuille flow analytical solution u_x = 4y(1-y), p(x)=8(1-x)` --semantically_similar_to--> `Analytical WSS formula under hard-SDF constraint`  [INFERRED] [semantically similar]
  docs/project_knowledge/Test problem 1 - Channel flow (Poiseuille flow) — FEniCSx tutorial.pdf → notes/implementation_fixes.md
- `eval block — run_suffix=bhpo, device, wss_batch_size=5000` --configures--> `main()`  [EXTRACTED]
  configs/config.yaml → scripts/05_evaluate.py
- `RFF params — rff_features=128, rff_sigma=2.0 (tune in {1,2,5,10})` --configures--> `RFFEncoder`  [EXTRACTED]
  configs/model/pinn_rff.yaml → src/hemodyn_pinn/pinn/networks.py
- `README.md — project overview & pipeline usage` --documents--> `main()`  [EXTRACTED]
  README.md → scripts/02_run_cfd.py
- `Search settings — n_calls=60, n_adam_trial=10000, backend=optuna, device=auto` --configures--> `main()`  [EXTRACTED]
  configs/bhpo/default.yaml → scripts/04b_bhpo_search.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **BHPO search → retrain → evaluate → sweep → figures pipeline** — scripts_04b_bhpo_search_main, scripts_04c_train_pinn_with_bhpo_main, scripts_05_evaluate_main, scripts_06_run_sweep_main, scripts_07_make_all_figures_main [EXTRACTED 0.95]
- **Evaluation metric suite consumed by 05_evaluate/06_run_sweep (NRMSE, R2, MAE, BA, ICC, conservation)** — eval_field_metrics_compute_field_metrics, eval_wss_metrics_compute_wss_metrics, eval_bland_altman_bland_altman_stats, eval_bland_altman_icc_one_way, eval_conservation_flow_rate_conservation_error [EXTRACTED 0.90]
- **solve_stokes internal helper chain (mesh build → BC marking → WSS extraction → export)** — cfd_stokes_solver_solve_stokes, cfd_stokes_solver_create_dolfinx_mesh, cfd_stokes_solver_mark_boundary_facets, cfd_stokes_solver_extract_nodal_values, cfd_stokes_solver_compute_wss, cfd_postprocess_export_xdmf [EXTRACTED 0.90]
- **Physics-enforcement fixes E (wall-biased sampling), A (hard SDF no-slip), B (vector potential), D (self-adaptive loss weights) wired through trainer/networks/losses/inference** — pinn_sampling_sample_collocation_biased, geometry_sdf_wallsdf, pinn_networks_pinnnetwork_curl, pinn_losses_selfadaptiveloss, pinn_trainer_pinntrainer, pinn_inference_compute_wss_hard_sdf [EXTRACTED 0.90]
- **Synthetic MRI forward-operator chain: voxel grid construction -> node assignment -> averaging -> Gaussian noise -> MRIObservation** — mri_voxelise_voxelgrid, mri_voxelise_assign_nodes_to_voxels, mri_operator_syntheticmrioperator_apply, mri_noise_add_gaussian_noise, mri_noise_sigma_from_vnr, mri_operator_mriobservation [EXTRACTED 0.90]
- **AnXplore mesh load -> boundary-face extraction -> outward-normal orientation -> mesh quality stats** — geometry_anxplore_loader_load_anxplore, geometry_anxplore_loader_extract_boundary_faces, geometry_anxplore_loader_compute_outward_normals, geometry_anxplore_loader_compute_outward_normals_from_centroid, geometry_anxplore_loader_anxploremesh_mesh_stats [EXTRACTED 0.90]
- **Phase 3 mandatory test suite (anxplore_loader, mri_operator, losses, inference, bhpo) gating phase completion per CLAUDE.md §7** — test_anxplore_loader_module, test_mri_operator_module, test_losses_module, test_inference_module, test_bhpo_module, claude_md_phase_plan [EXTRACTED 0.90]
- **Test files added/extended specifically to cover Fixes A/B/D/E (hard SDF, vector potential, self-adaptive loss, wall-biased sampling)** — test_networks_module, test_sdf_module, test_sampling_module, test_trainer_module, test_losses_module, test_inference_module, implementation_fixes_fix_a_hard_sdf, implementation_fixes_fix_b_vector_potential, implementation_fixes_fix_d_self_adaptive_loss, implementation_fixes_fix_e_wall_bias [EXTRACTED 0.90]
- **Hydra config groups composed by config.yaml — geometry, cfd, mri, model, bhpo — selected per pipeline script invocation** — config_yaml_root, geometry_caseA_yaml_config, geometry_caseB_yaml_config, geometry_caseC_yaml_config, geometry_caseR_yaml_config, stokes_default_yaml_config, pinn_base_yaml_config, pinn_rff_yaml_config, bhpo_default_yaml_config, configs_readme [EXTRACTED 0.90]
- **Top-level project documentation set: README, CLAUDE.md, configs/README, tests/README, requirements.txt** — readme_root, claude_md_root, configs_readme, tests_readme, requirements_txt [EXTRACTED 0.85]
- **MRI voxel-size resolution sweep (0.5/1.0/1.5/2.0/2.5 mm configs)** — mri_voxel_0p5mm_yaml_config, mri_voxel_1p0mm_yaml_config, mri_voxel_1p5mm_yaml_config, mri_voxel_2p0mm_yaml_config, mri_voxel_2p5mm_yaml_config, mri_voxel_resolution_sweep [INFERRED 0.85]
- **Fixes E, A, B, D — combined physics-structural NRMSE improvement programme** — implementation_fixes_fix_e_wall_bias, implementation_fixes_fix_a_hard_sdf, implementation_fixes_fix_b_vector_potential, implementation_fixes_fix_d_self_adaptive_loss, results_baseline_casec_1mm_wss_nrmse_1_661 [EXTRACTED 1.00]
- **BHPO acceleration chain: Optuna TPE + MPS auto-detect + adjusted trial budget** — implementation_fixes_bhpo_optuna_acceleration, implementation_fixes_mps_auto_detection, implementation_fixes_trial_budget_adjustment [EXTRACTED 1.00]

## Communities (68 total, 8 thin omitted)

### Community 0 - "Hydra Configs & Geometry Cases"
Cohesion: 0.05
Nodes (56): Architecture flags — use_hard_sdf/use_vec_potential/use_adaptive_weights (fixed per run), configs/bhpo/default.yaml — BHPO search & retrain config (M3 Pro tuned), Retrain settings — source_case_id=caseC, n_adam_full=50000, n_lbfgs_full=5000, Search settings — n_calls=60, n_adam_trial=10000, backend=optuna, device=auto, eval block — run_suffix=bhpo, device, wss_batch_size=5000, config.yaml — Hydra root defaults & output dirs, configs/README.md — Hydra config group documentation, configs/geometry/caseA.yaml — case_id=caseA, z-span 4.614 mm (+48 more)

### Community 1 - "Collocation Sampling & Loss Refs"
Cohesion: 0.07
Nodes (35): CLAUDE.md §4 — Phase plan (Phase 3 IN PROGRESS), Arzani et al. 2021, Physics of Fluids — wall-biased sampling NRMSE reduction, Fix D: Self-adaptive loss weights (SA-PINN), Fix E: wall-biased collocation fraction = 0.4, McClenny & Braga-Neto 2023, JCP — self-adaptive PINNs, Wang et al. 2022, CMAME — SA-PINN, epoch_rng(), Collocation and boundary point sampling utilities.  All functions operate in non (+27 more)

### Community 2 - "Hard-SDF Wall Distance"
Cohesion: 0.09
Nodes (24): Unsigned wall-distance function for the hard no-slip SDF constraint.  Computes t, Approximate unsigned distance from arbitrary lumen points to the wall.      Para, Return distances (m) from each query point to the nearest wall face.          Pa, Return non-dimensional distances for non-dimensional query coords.          Para, WallSDF, _find_indices(), Return integer indices of `sampled` rows within `pool`.      Uses a hash of (x,, ndarray (+16 more)

### Community 3 - "Synthetic MRI Forward Operator"
Cohesion: 0.13
Nodes (18): MRIConfig, Synthetic 4D-flow MRI forward operator.  Implements the forward model:      u_MR, Apply the synthetic 4D-flow MRI forward operator to CFD nodal data.      Paramet, Configuration for the synthetic MRI forward operator.      Parameters     ------, SyntheticMRIOperator, ndarray, Tests for the synthetic MRI forward operator (Phase 1.4 REQUIRED).  Covers:   -, Voxel average of a constant field must equal that constant in every voxel. (+10 more)

### Community 4 - "PINN Trainer Architecture Tests"
Cohesion: 0.14
Nodes (17): PINNConfig, PINNTrainer, _make_data(), _make_trainer(), Tests for PINNTrainer.__init__ with the new architecture flags (Fixes A, B, D, E, use_hard_sdf=True without wall_pts_m must raise ValueError., When use_hard_sdf=True, the trainer must zero out the BC weight., _skip_div must be True whenever use_vec_potential=True. (+9 more)

### Community 5 - "Voxelisation & MRI Application"
Cohesion: 0.12
Nodes (14): Apply voxel averaging and noise to CFD nodal data.          Parameters         -, assign_nodes_to_voxels(), Cartesian voxel grid definition and mesh-node-to-voxel assignment.  All coordina, Axis-aligned Cartesian voxel grid in millimetre coordinates.      Attributes, Build a grid that covers the bounding box of *points_mm*.          Parameters, Total number of voxels., Return (nx, ny, nz, 3) array of voxel centre coordinates in mm., Convert (ix, iy, iz) voxel indices to flat C-order indices. (+6 more)

### Community 6 - "AnXplore Mesh Property Tests"
Cohesion: 0.13
Nodes (7): AnxploreMesh, Load caseA (tet-only, no explicit surface triangles)., Parent artery x-span should be 16.279 mm (±0.002 mm)., Mean edge length should be ≈ 0.1563 mm (mesh_stats.txt ground truth)., Tet-only mesh: wall_tri must have been extracted (not from explicit tris)., TestAnxploreMeshProperties, TestLoadCaseA

### Community 7 - "2D Figure Plotting"
Cohesion: 0.16
Nodes (24): Figure, fig8_bland_altman(), Scatter + Bland–Altman for WSS agreement — Fig 8 from doc 04., ndarray, Path, plot_bhpo_convergence(), plot_bland_altman(), plot_conservation_profile() (+16 more)

### Community 8 - "Stokes CFD Solver"
Cohesion: 0.15
Nodes (24): BoundaryParts, _compute_wss(), _compute_wss_fn_scalar(), _compute_wss_fn_vector(), _create_dolfinx_mesh(), _extract_nodal_values(), _mark_boundary_facets(), Steady Stokes flow solver using FEniCSx (dolfinx 0.10) Taylor-Hood P2-P1.  Geo (+16 more)

### Community 9 - "Field Accuracy Metrics"
Cohesion: 0.12
Nodes (21): RMS-normalised NRMSE (consistent with BHPO objective), WSS NRMSE on held-out wall faces as BHPO objective, compute_field_metrics(), mae(), nrmse(), r2(), Scalar accuracy metrics for field comparisons.  All functions accept flat or mul, Root-mean-square error normalised by RMS of the reference field.      Parameters (+13 more)

### Community 10 - "Project Constants & Pipeline"
Cohesion: 0.13
Nodes (20): CLAUDE.md §5 — Calibrated physical constants & RNG seeds, CLAUDE.md — project navigation map for Claude Code, Architecture flags (use_hard_sdf/use_vec_potential/use_adaptive_weights) must match between BHPO search and retrain, BHPO search → retrain → evaluate pipeline, MRIObservation, Output of the synthetic MRI forward operator.      Attributes     ----------, L_SCALE / U_SCALE / MU (physical scale constants), make_pressure_anchor() (+12 more)

### Community 11 - "WSS Evaluation Pipeline"
Cohesion: 0.13
Nodes (20): Batched WSS autograd inference to avoid OOM, run_id naming convention (case_voxel_suffix), voxel_tag string format helper (voxel_XpYmm), src/hemodyn_pinn/README.md (package docs), PINNNetwork, Coordinate-based PINN for steady Stokes flow.      Input:  x̂ = (x̂, ŷ, ẑ) — non, _compute_wss_batched(), main() (+12 more)

### Community 12 - "CFD Postprocessing I/O"
Cohesion: 0.15
Nodes (19): export_xdmf(), load_cfd_xdmf(), load_metadata(), Save and load Stokes CFD solutions (velocity, pressure, WSS).  Output layout und, Load scalar metadata from a CFD output directory., Return a summary of WSS statistics on the vessel wall only.      Parameters, Write all CFD outputs to XDMF+HDF5 (doc 06 §7 output format).      Requires dolf, Load CFD XDMF output using meshio (cross-platform, no dolfinx needed).      Para (+11 more)

### Community 13 - "MRI Noise Model"
Cohesion: 0.14
Nodes (12): Return a dict of mesh quality metrics.          Edge-length statistics are compu, add_gaussian_noise(), Additive Gaussian noise model for synthetic 4D-flow MRI.  The noise model is:, Compute noise standard deviation from VENC and VNR.      Parameters     --------, Add independent Gaussian noise to each velocity component.      Parameters     -, sigma_from_vnr(), ndarray, With many samples, noise should be zero-mean to within ~3σ/√N. (+4 more)

### Community 14 - "Paper Figure Generation"
Cohesion: 0.17
Nodes (19): load_solution(), Load a previously saved solution.      Parameters     ----------     npz_path:, fig3_cfd_ground_truth(), fig5_wss_comparison(), fig6_sensitivity_heatmap(), fig9_conservation(), figB_bhpo_convergence(), figL_loss_curves() (+11 more)

### Community 15 - "PINN Adam-LBFGS Training Loop"
Cohesion: 0.18
Nodes (11): Optimizer, PINNTrainer, Manages Adam → L-BFGS training of a PINNNetwork.      Parameters     ----------, Run the Adam optimisation phase with cosine-annealing LR., Run L-BFGS refinement after Adam., Run the full Adam → L-BFGS training pipeline., TestAdaptiveWeightsInit (Fix D — _sa_loss), TestHardSDFInit (Fix A — _sdf_interior/_sdf_data, _effective_lambda_bc) (+3 more)

### Community 16 - "WSS Inference Tests"
Cohesion: 0.15
Nodes (11): _ConstantTangentialNet, LinearVelocityNet, _NormalAlignedNet, Tensor, Tests for hemodyn_pinn.pinn.inference.  Verifies the WSS-from-autograd pipeline, Raw velocity = (1, 0, 0) — purely tangential to n = (0, 1, 0).      Expected har, Raw velocity = (0, 1, 0) — parallel to n = (0, 1, 0).      Expected hard-SDF WSS, u = (A x, 0, 0, 0) — linear shear in x-direction.      ∇u_0 = (A, 0, 0);  ∇u_1 = (+3 more)

### Community 17 - "BHPO Search Loop"
Cohesion: 0.21
Nodes (15): Bayesian Hyperparameter Optimisation for the PINN.  Public API ---------- BHPOOb, load_best_params(), BHPO search loop — Optuna TPE (primary) with skopt GP fallback.  The primary bac, Run the legacy GP-based BHPO search with scikit-optimize., Run BHPO search, auto-selecting Optuna or skopt.      Parameters     ----------, Load the winning hyperparameter dict from a completed BHPO run., Run the BHPO search with Optuna TPE.      Parameters     ----------     objectiv, run_bhpo_search() (+7 more)

### Community 18 - "PINN Network Architecture"
Cohesion: 0.15
Nodes (11): dim_velocity(), dim_wss(), nondim_coords(), nondim_velocity(), PINN network architectures for Stokes flow reconstruction.  All modules operate, Encode spatial coordinates.          Parameters         ----------         x:, Compute curl(A) via autograd.          Parameters         ----------         A:, Run encoder + MLP, returning raw (N, 4) output. (+3 more)

### Community 19 - "BHPO Objective & Tests"
Cohesion: 0.22
Nodes (9): BHPOObjective, BHPO objective function: WSS NRMSE on held-out wall faces.  The ``BHPOObjective`, Callable objective for BHPO search (Optuna and skopt compatible).      Parameter, _minimal_bhpo_data(), BHPO module tests (Phase 3 gate — required by CLAUDE.md §7).  Covers: - space.py, Full-stack trial: verify a trial completes and returns a finite float., Small synthetic arrays for BHPOObjective construction., TestBHPOObjectiveInit (+1 more)

### Community 20 - "Stokes Residual Loss Tests"
Cohesion: 0.18
Nodes (7): Stokes momentum + continuity residual loss at collocation points.      Parameter, stokes_residual_loss(), Residual must vanish for the analytical Stokes solution., Backprop through physics residual (second-order autograd) works., Full loss (momentum + div) ≥ momentum-only loss (non-negative div term)., TestStokesResidualLoss, TestStokesResidualSkipDiv

### Community 21 - "AnXplore Mesh Loader"
Cohesion: 0.16
Nodes (8): AnxploreMesh, _compute_outward_normals_from_centroid(), AnXplore fluid mesh loader with boundary extraction and mesh quality checks.  Ha, Compute outward normals using the mesh centroid as interior reference.      Used, Loaded AnXplore fluid mesh with precomputed boundary geometry.      Attributes, (N, 3) float64 node coordinates in **metres** (SI units)., (W, 3) float64 centroid of each wall triangle in mm., ndarray

### Community 22 - "Boundary Face Extraction Tests"
Cohesion: 0.26
Nodes (6): _extract_boundary_faces(), Return boundary triangles extracted from a pure-tet mesh.      A boundary face i, _make_two_tet_mesh(), Face {1,2,3} is shared by both tets and must not be a boundary face., Two tets sharing face {1,2,3}; 6 boundary faces expected.      Vertices:, TestExtractBoundaryFaces

### Community 23 - "Hard-SDF WSS Computation"
Cohesion: 0.22
Nodes (8): compute_wss_hard_sdf(), Compute WSS using the analytical hard-SDF formula (Fix A).      Requires net.use, PINNNetwork, u_raw = (1,0,0), n = (0,1,0) → |WSS| = μ(U/L) exactly., u_raw parallel to n → tangential component = 0 → WSS = 0., The WSS vector must have zero normal component., With use_hard_sdf=True, auto must agree with compute_wss_hard_sdf., TestComputeWSSHardSDF

### Community 24 - "Self-Adaptive Loss Weights"
Cohesion: 0.20
Nodes (6): Learnable loss-weight module (McClenny & Braga-Neto, 2023, JCP).      Each weigh, Current weight values (detached, for logging)., SelfAdaptiveLoss, Reversing the gradient and taking an SGD step must increase λ when loss is large, lambda=0 is clamped to 1e-8 before log; must not raise., TestSelfAdaptiveLoss

### Community 25 - "Stokes Config & AnXplore Source"
Cohesion: 0.20
Nodes (9): Configuration for the steady Stokes solver.      Parameters     ----------, StokesConfig, AnXplore dataset (Goetz et al. 2024, MIT license), fetch_and_inspect.sh (AnXplore acquisition), Goetz et al. (2024) — AnXplore paper, main(), DictConfig, Run steady Stokes CFD on one or more AnXplore cases.  Usage (Linux / WSL with (+1 more)

### Community 26 - "Outward Normal Tests"
Cohesion: 0.20
Nodes (9): _compute_outward_normals(), Compute outward unit normals for boundary triangles.      Orientation is determi, _make_single_tet_mesh(), ndarray, Tests for anxplore_loader.py — Phase 1.2 mandatory test suite.  Fast unit tests, Normal must point away from the mesh centroid (outward)., A single regular tetrahedron; all 4 faces are boundary faces., TestLoadErrors (+1 more)

### Community 27 - "Standard WSS Computation Tests"
Cohesion: 0.22
Nodes (8): compute_wss(), Compute WSS using the standard autograd path.      Use this when use_hard_sdf=Fa, WSS magnitude for Couette flow (u=y) with n=(0,1,0) is μU/L., WSS must be purely tangential — no component along n̂., Order-of-magnitude check: WSS for Couette flow is ~μU/L (Pa)., Couette-like flow: u = y, v = 0, w = 0, p = 0.      ∂u/∂y = 1 — the only non-zer, ShearFlowNet, TestComputeWSS

### Community 28 - "Total Loss Computation"
Cohesion: 0.29
Nodes (5): Compute the full PINN loss and return a breakdown dict.      Parameters     ----, total_loss(), PINNNetwork, TestLossesWithSDFVals, TestTotalLoss

### Community 29 - "Optuna HP Sampling Tests"
Cohesion: 0.26
Nodes (5): optuna_suggest(), Sample one hyperparameter configuration from an Optuna Trial.      The TPE sampl, _MockTrial, Minimal stand-in for optuna.Trial, used to test optuna_suggest., TestOptunaWatchdog

### Community 30 - "Inlet/Outlet Detection Tests"
Cohesion: 0.29
Nodes (6): detect_open_faces(), Identify inlet/outlet patches in the AnXplore boundary mesh.      Both open fa, _make_two_disk_mesh(), A mesh with no n_y = -1 faces should raise ValueError., Return a minimal synthetic AnxploreMesh for testing detect_open_faces.      Th, TestDetectOpenFacesSynthetic

### Community 31 - "PINN Training Script"
Cohesion: 0.20
Nodes (9): PINNConfig, PINN training loop: Adam → L-BFGS with checkpointing.  Training recipe (CLAUDE.m, Hyperparameters for the PINN training loop.      Parameters     ----------     n, README.md — project overview & pipeline usage, main(), DictConfig, Train the PINN on one AnXplore case.  Requires:   data/cfd/<case_id>/solution, ndarray (+1 more)

### Community 32 - "BHPO Device & Param Tests"
Cohesion: 0.18
Nodes (10): auto_device(), Return "mps", "cuda", or "cpu" depending on hardware availability., TestAutoDevice, TestDecodeParams, _MockTrial (optuna.Trial stand-in), test_bhpo.py — Phase 3 BHPO smoke test (CLAUDE.md §7), TestBHPOObjectiveInit, TestBHPOObjectiveSmoke (slow, _run_trial) (+2 more)

### Community 33 - "AnXplore Loading Integration Tests"
Cohesion: 0.18
Nodes (10): load_anxplore(), Load a single AnXplore fluid mesh (.vtk) and return an AnxploreMesh.      Parame, Path, TestExtractBoundaryFaces, TestLoadCaseA (slow, integration), TestLoadErrors, TestLoadFluid0 (slow, integration), TestAnxploreMeshProperties (+2 more)

### Community 34 - "3D WSS Rendering"
Cohesion: 0.26
Nodes (11): ndarray, Path, 3D surface WSS renders using PyVista.  All public functions require pyvista. The, Render pointwise WSS error (pred − ref) using RdBu_r diverging colormap., Render WSS magnitude as a point cloud on the vessel wall.      Parameters     --, Side-by-side WSS render: CFD ground truth (left) vs PINN (right).      Parameter, render_wss_comparison(), render_wss_error() (+3 more)

### Community 35 - "Loss Function Unit Tests"
Cohesion: 0.20
Nodes (8): ExactStokesNet, Tensor, Tests for hemodyn_pinn.pinn.losses.  Each loss term is tested in isolation with, Returns u=y², v=0, w=0, p=2*x — an exact non-dim Stokes solution.      Verificat, Small sets of synthetic coordinate tensors., Always outputs (0, 0, 0, p) where p varies, for BC loss testing., rng_pts(), ZeroVelocityNet

### Community 36 - "Stokes Solver & Postproc Tests"
Cohesion: 0.21
Nodes (8): _make_dummy_solution(), Tests for the Stokes CFD solver module.  Fast unit tests (no dolfinx required,, detect_open_faces must correctly split the two open disks on real meshes., Return a minimal StokesSolution with synthetic arrays., Smoke test: solver runs on caseC and produces plausible results., test_detect_open_faces_real_cases(), test_solve_stokes_smoke(), TestPostprocess

### Community 37 - "BHPO Search Space"
Cohesion: 0.27
Nodes (4): decode_params(), BHPO search space definition and parameter decoding.  Supports both the legacy s, Convert a raw skopt parameter list to a human-readable HP dict., TestDecodeParams

### Community 38 - "Data Loss Function"
Cohesion: 0.24
Nodes (7): data_loss(), MSE between network velocity and MRI observations.      Parameters     ---------, Backprop through data_loss reaches network parameters., Tiny PINN network for fast testing (2 hidden layers, 16 neurons)., Loss is zero when predicted velocity matches observations exactly., small_net(), TestDataLoss

### Community 39 - "Pressure Anchor Loss"
Cohesion: 0.24
Nodes (7): pressure_anchor_loss(), Pin pressure to zero at one outlet centroid (gauge fixing)., Module, TestPressureAnchorLoss, Returns p=0 always (for pressure anchor = 0 test)., TestPressureAnchorLoss, ZeroPressureAtPointNet

### Community 40 - "MLP Backbone"
Cohesion: 0.33
Nodes (4): ActivationName, MLP, Fully-connected network with configurable smooth activations., TestMLP

### Community 41 - "Velocity Jacobian Computation"
Cohesion: 0.28
Nodes (5): compute_velocity_jacobian(), WSS inference from a trained PINNNetwork via autograd.  Standard path (use_hard_, Compute ∇û at wall points via autograd (standard path only).      Parameters, Jacobian of u=(Ax, 0, 0) should be diag(A, 0, 0) in first row., TestComputeVelocityJacobian

### Community 42 - "Velocity Field Prediction"
Cohesion: 0.28
Nodes (6): predict_velocity_field(), Return dimensional velocity and pressure at arbitrary points.      Parameters, Tensor, predict_velocity_field runs under no_grad., Output should scale with U_SCALE and pressure scale., TestPredictVelocityField

### Community 43 - "Boundary Condition Loss"
Cohesion: 0.31
Nodes (4): bc_loss(), No-slip BC loss on the vessel wall (soft constraint).      When use_hard_sdf=Tru, Forward pass returning (û, v̂, ŵ, p̂).          Parameters         ----------, TestBCLoss

### Community 44 - "Loss Term Test Suite"
Cohesion: 0.22
Nodes (9): TestBCLoss, TestDataLoss, ExactStokesNet (analytic Stokes fixture), test_losses.py — per-term loss tests (Phase 1.5), TestSelfAdaptiveLoss (Fix D, gradient reversal), TestStokesResidualSkipDiv (Fix B, skip_div_loss), TestStokesResidualLoss (exact Stokes solution check), TestTotalLoss (weighted-sum consistency) (+1 more)

### Community 45 - "Fluid_0 Mesh Loading Tests"
Cohesion: 0.22
Nodes (3): Load full_dataset/Fluid_0.vtk (explicit surface triangles present)., Explicit triangles in Fluid_0.vtk: 33,116 (from mesh_stats.txt)., TestLoadFluid0

### Community 46 - "Hard-SDF Network Tests"
Cohesion: 0.22
Nodes (5): The pressure column (index 3) must be identical for sdf=0 and sdf=1., u_forward(x, sdf=d) must equal d * u_raw(x) componentwise., forward_raw must match the plain forward of an identical network without SDF., Velocity output must be exactly 0 when sdf_vals = 0., TestHardSDF

### Community 47 - "Flow Conservation Diagnostics"
Cohesion: 0.36
Nodes (7): Count-weighted normal-velocity proxy for flow-rate conservation (face areas unavailable), flow_rate_conservation_error(), Flow-rate conservation diagnostics for PINN velocity fields.  Checks that the PI, Estimate mass-conservation error from face-centroid velocities.      Uses count-, Estimate flow rate Q(z) along z-axis cross-sections.      Samples a uniform grid, z_profile_flow_rate(), ndarray

### Community 48 - "Random Fourier Features Encoder"
Cohesion: 0.39
Nodes (3): Random Fourier Features positional encoder.      Replaces raw (x, y, z) with, RFFEncoder, TestRFFEncoder

### Community 49 - "WSS Pipeline Test Overview"
Cohesion: 0.25
Nodes (8): TestComputeWSS (Couette flow checks), TestComputeWSSAuto (Fix A dispatch), TestComputeWSSHardSDF (Fix A), test_inference.py — WSS autograd pipeline tests (Phase 1.5), TestPredictVelocityField, ShearFlowNet (Couette analytic fixture), TestComputeVelocityJacobian, tests/README.md — suite overview (116 tests)

### Community 50 - "Bland-Altman Agreement Stats"
Cohesion: 0.33
Nodes (6): bland_altman_stats(), icc_one_way(), Bland–Altman agreement statistics and intraclass correlation coefficient.  Refer, Compute Bland–Altman agreement statistics.      Parameters     ----------     pr, Intraclass correlation coefficient ICC(1,1) — one-way random effects.      Treat, ndarray

### Community 51 - "WSS Auto-Dispatch"
Cohesion: 0.33
Nodes (5): compute_wss_auto(), Dispatch to the appropriate WSS computation based on network flags.      Uses th, Module, Without use_hard_sdf, auto must agree with the standard autograd path., TestComputeWSSAuto

### Community 52 - "Loss Term Implementation"
Cohesion: 0.29
Nodes (5): _grad(), PINN loss terms for steady Stokes flow reconstruction.  Non-dimensional Stokes e, Compute weighted sum Σ_k exp(log_λ_k) · L_k.          Parameters         -------, Gradient of a scalar field w.r.t. x via autograd., Tensor

### Community 53 - "MRI Operator Test Suite"
Cohesion: 0.29
Nodes (7): TestAssignNodesToVoxels, test_mri_operator.py — synthetic MRI forward operator tests (Phase 1.4 REQUIRED), TestNoiseModel (sigma_from_vnr, add_gaussian_noise), TestMRIObservationIO (save/load round-trip), TestPartialVolume (boundary-voxel diagnostic), TestSyntheticMRIOperator (correctness, reproducibility), TestVoxelGrid

### Community 55 - "Vector Potential Tests"
Cohesion: 0.29
Nodes (3): div(u) = ∂u/∂x + ∂v/∂y + ∂w/∂z must be ≈ 0 for a vec-potential network., _curl must return (N, 3)., TestVecPotential

### Community 56 - "Stokes Solver Test Suite"
Cohesion: 0.40
Nodes (6): TestStokesConfig (Re ≈ 0.1 regime check), test_detect_open_faces_real_cases (slow, parametrized caseA/B/C/R), TestDetectOpenFacesSynthetic, _make_two_disk_mesh (synthetic open-face geometry helper), test_stokes_solver.py — Stokes CFD solver tests, test_solve_stokes_smoke (dolfinx-only, skipped on Windows)

### Community 58 - "MRI Observation I/O"
Cohesion: 0.40
Nodes (3): Load a previously saved MRIObservation from an .npz file., Save to a compressed .npz file.          Parameters         ----------         p, Path

### Community 60 - "AnXplore Fetch Script"
Cohesion: 0.83
Nodes (3): fetch_and_inspect.sh script, hr(), log()

### Community 61 - "Synthetic MRI Generation Script"
Cohesion: 0.50
Nodes (3): main(), DictConfig, Generate synthetic 4D-flow MRI observations from CFD solution data.  Usage (Wind

## Ambiguous Edges - Review These
- `__init__.py` → `compute_field_metrics()`  [AMBIGUOUS]
  src/hemodyn_pinn/eval/__init__.py · relation: references

## Knowledge Gaps
- **26 isolated node(s):** `DictConfig`, `Path`, `ndarray`, `Goetz et al. (2024) — AnXplore paper`, `TestExtractBoundaryFaces` (+21 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **8 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `__init__.py` and `compute_field_metrics()`?**
  _Edge tagged AMBIGUOUS (relation: references) - confidence is low._
- **Why does `PINNNetwork` connect `WSS Evaluation Pipeline` to `Hydra Configs & Geometry Cases`, `PINN Trainer Architecture Tests`, `Project Constants & Pipeline`, `PINN Adam-LBFGS Training Loop`, `WSS Inference Tests`, `PINN Network Architecture`, `BHPO Objective & Tests`, `Stokes Residual Loss Tests`, `Hard-SDF WSS Computation`, `Self-Adaptive Loss Weights`, `Standard WSS Computation Tests`, `Total Loss Computation`, `PINN Training Script`, `Loss Function Unit Tests`, `Data Loss Function`, `Pressure Anchor Loss`, `MLP Backbone`, `Velocity Jacobian Computation`, `Velocity Field Prediction`, `Boundary Condition Loss`, `Hard-SDF Network Tests`, `Random Fourier Features Encoder`, `WSS Pipeline Test Overview`, `WSS Auto-Dispatch`, `PINN Network Tests`, `Vector Potential Tests`, `Combined SDF+VecPotential Tests`?**
  _High betweenness centrality (0.387) - this node is a cross-community bridge._
- **Why does `PINNTrainer` connect `PINN Adam-LBFGS Training Loop` to `Collocation Sampling & Loss Refs`, `Hard-SDF Wall Distance`, `PINN Trainer Architecture Tests`, `Project Constants & Pipeline`, `WSS Evaluation Pipeline`, `BHPO Objective & Tests`, `Self-Adaptive Loss Weights`, `PINN Training Script`?**
  _High betweenness centrality (0.172) - this node is a cross-community bridge._
- **Why does `load_anxplore()` connect `AnXplore Loading Integration Tests` to `Stokes Solver & Postproc Tests`, `Project Constants & Pipeline`, `WSS Evaluation Pipeline`, `AnXplore Mesh Loader`, `Boundary Face Extraction Tests`, `Stokes Solver Test Suite`, `Stokes Config & AnXplore Source`, `Outward Normal Tests`, `Synthetic MRI Generation Script`, `PINN Training Script`?**
  _High betweenness centrality (0.150) - this node is a cross-community bridge._
- **Are the 45 inferred relationships involving `PINNNetwork` (e.g. with `BHPOObjective` and `PINNConfig`) actually correct?**
  _`PINNNetwork` has 45 INFERRED edges - model-reasoned connections that need verification._
- **Are the 15 inferred relationships involving `PINNTrainer` (e.g. with `BHPOObjective` and `.nondim()`) actually correct?**
  _`PINNTrainer` has 15 INFERRED edges - model-reasoned connections that need verification._
- **Are the 28 inferred relationships involving `SelfAdaptiveLoss` (e.g. with `Optimizer` and `PINNConfig`) actually correct?**
  _`SelfAdaptiveLoss` has 28 INFERRED edges - model-reasoned connections that need verification._
# Baseline result: caseC, 1 mm voxel, fixed HPs

**Date:** 2026-05-14 (Phase 1.5 prototype run, student's Mac)  
**Script:** `scripts/04_train_pinn.py geometry=caseC model=pinn_base`  
**Config:** pinn_base defaults — λ_bc=10, λ_anchor=10, tanh, 4×128, Adam 50k

## Metrics

| Metric | Value |
|--------|-------|
| WSS NRMSE | 1.661 |
| R² | −0.52 |
| WSS mean (PINN) | ~40× underestimate vs CFD |

## Root cause

λ_bc = 10 with only 445 MRI voxels created a BC attractor: the model converged to
near-zero velocity everywhere rather than fitting the sparse MRI signal.
λ_anchor = 10 had the same over-weighting effect.

## Fix applied in 3-prep

- λ_anchor lowered to 1.0 (in pinn_base.yaml and pinn_rff.yaml).
- λ_phys and λ_bc made BHPO search dimensions (not fixed).
- BHPO objective: WSS NRMSE on 20 % held-out wall faces (seed 4321).

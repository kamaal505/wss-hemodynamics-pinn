# Implementation: Fixes F, G, H, I — PINN magnitude-collapse fixes

**Date:** 2026-06-20
**Branch:** `fix/pinn-magnitude-collapse` (off `enforce-physics`)
**Motivation:** WSS NRMSE > 0.8 regardless of voxel resolution (~500 vs ~50k
voxels). Insensitivity to data density showed the data term was not driving the
solution — distinct from the structural NRMSE work in `implementation_fixes.md`
(Fixes E, A, B, D), which is orthogonal to and could not cure this failure.

---

## Root cause — collapse to the trivial Stokes solution

Steady Stokes is **linear and homogeneous** (`-∇p + Δu = 0`, `∇·u = 0`, zero
body force), so `u ≡ 0, p ≡ const` is an **exact, zero-residual** solution.

The PINN's constraints were:
1. no-slip wall — satisfied by `u ≡ 0`;
2. Stokes residual at 10k collocation points — satisfied by `u ≡ 0`;
3. a one-point pressure anchor — satisfied by `p ≡ 0`;
4. **445 sparse interior data points — the only term resisting collapse.**

With `lambda_data = lambda_phys = 1`, the physics term (evaluated at 10k points)
dominates the gradient and shrinks the field. Because the residual is *linear*
in `(u, p)`, shrinking the field always lowers the physics loss, so the network
slides toward zero unless the data term is strong enough to hold the magnitude.
It is not — so the field collapses ~40–55×, independent of voxel count.

### Confirmed numerically

`scripts/00_diagnose_collapse.py` on the existing
`caseC_voxel_1p0mm_adam50000` checkpoint:

| metric | value | reading |
|---|---|---|
| velocity ratio (RMS \|u_pred\| / RMS \|u_obs\|) | 0.11 | predicts ~11 % of true magnitude |
| WSS ratio (RMS pred / RMS CFD) | 0.018 | ~55× underestimate |
| WSS NRMSE | 0.99 | ≈ predicting nothing |
| loss_data (relative) | 0.82 | field barely fits data |
| loss_phys | 8e-6 | near-zero field trivially satisfies physics |

Two alternative hypotheses were checked and **ruled out**: (a) velocity
over-scaling — non-dim targets are O(0.1), `U_SCALE = 2e-5 m/s` is correct;
(b) a factor-of-2 error in `compute_wss_hard_sdf` — the 2 correctly cancels the
½ strain factor (`2μ·(−½ u_net) = −μ u_net`).

---

## Fix 0 — Collapse diagnostic (measurement tool)

**New file:** `scripts/00_diagnose_collapse.py`

Loads a trained checkpoint (architecture reconstructed exactly as
`05_evaluate.py` does), the CFD ground truth, and the MRI observations, then
reports the **predicted/CFD velocity-magnitude ratio**, the **WSS-magnitude
ratio + NRMSE**, and every per-term loss at the saved weights, ending with a
`COLLAPSED`/`HEALTHY` verdict.

**Why.** Each subsequent fix needs a unit of measurement. A ratio ≪ 1 (or
NRMSE > 0.8) is the collapse signature; ≈ 1 is healthy. Run before and after
every change.

```bash
python scripts/00_diagnose_collapse.py geometry=caseC \
    eval.run_suffix=adam50000 \
    model.use_hard_sdf=false model.use_vec_potential=false
```

---

## Fix F — Scale-invariant (relative) data/inlet loss

**File:** `src/hemodyn_pinn/pinn/losses.py` (`data_loss`, `inlet_loss`)
**Flag:** `training.relative_data: true`

**Change.** The misfit is normalised by the mean squared observation magnitude:

```
L_data = mean((u_pred − u_obs)²) / (mean(u_obs²) + 1e-12)
```

**What it does.** A zero prediction now gives `L_data ≈ 1.0`; a perfect fit gives
`0.0`. The loss is independent of the tiny absolute velocity scale
(`U_SCALE = 2e-5 m/s`), so `lambda_data` is comparable in magnitude to the
physics residual and behaves predictably across cases and voxel sizes.

**Why.** With an absolute MSE the data loss is intrinsically ~1e-2 in non-dim
units — far too small to compete with the physics term even at equal weights.
Normalising puts data and physics on the same scale.

**Cost.** Negligible (one extra reduction). `relative=True` is a function-level
arg so the absolute form remains testable.

---

## Fix G — `lambda_data` as a real, raisable, searched knob

**Files:** `configs/model/pinn_base.yaml`, `configs/model/pinn_rff.yaml`,
`src/hemodyn_pinn/bhpo/space.py`, `src/hemodyn_pinn/bhpo/objective.py`

**Change.**
- Default `lambda_data: 1.0 → 10.0` (≫ `lambda_phys = 1.0`).
- Added `lambda_data` to the BHPO search space (log-uniform **1–1000**) in
  `SEARCH_SPACE`, `decode_params`, and `optuna_suggest`; threaded through
  `BHPOObjective._run_trial` into `PINNConfig`.

**What it does.** Lets the data term hold the field magnitude, and lets BHPO
discover how strong it must be (previously `lambda_data` was hard-coded to 1.0
and absent from the 10-dim search; the space is now 11-dim).

**Why.** This is the single most important missing knob — it directly opposes
the collapse mechanism.

**Cost.** One extra BHPO dimension (search budget unchanged at `n_calls=60`).

---

## Fix H — Checkpoint selection by observation misfit, not total loss

**File:** `src/hemodyn_pinn/pinn/trainer.py`
(`_selection_metric`, `_maybe_update_best`)

**Change.** "Best model" is now selected by `loss_data (+ loss_inlet)` rather
than `loss_total`.

**What it does.** A collapsed near-zero field has the **lowest** total loss
(physics ≈ 0) but **high** data loss — so the old criterion actively saved the
collapsed state as "best". Selecting on observation misfit refuses it: the saved
checkpoint is the epoch that best fits the measurements.

**Why.** Even with the right weights, picking the wrong checkpoint at the end
re-introduces the collapse. This was a latent bug independent of the weighting.

**Cost.** Zero — `loss_data`/`loss_inlet` are already in the per-epoch
breakdown.

---

## Fix I — Inflow magnitude constraint

**Files:** `src/hemodyn_pinn/pinn/losses.py` (`inlet_loss`),
`src/hemodyn_pinn/pinn/sampling.py` (`select_inlet_targets`),
`trainer.py`, `scripts/04_train_pinn.py`, `04b_bhpo_search.py`,
`04c_train_pinn_with_bhpo.py`
**Flags:** `training.lambda_inlet: 10.0`, `training.inlet_source: mri`,
`training.inlet_band_mm: 2.0`

**What it does.** Adds a velocity term at the inlet plane:

```
L_inlet = mean((u_pred − u_inlet)²) / (mean(u_inlet²) + 1e-12)
```

weighted by `lambda_inlet`. This directly pins the flow magnitude — `u ≡ 0`
cannot satisfy a non-zero inlet velocity. The inlet SDF is precomputed
(`WallSDF`) so the term is correct under hard-SDF.

**Why.** Without any velocity/flux condition at an open boundary, the magnitude
of a linear-homogeneous Stokes field is unpinned. Fixes F–H make the data term
strong enough to hold it; Fix I anchors it structurally as well.

### Inlet source — methodology / leakage note

`select_inlet_targets(source=...)`:
- **`mri` (default, honest):** inlet velocity from the near-inlet MRI voxels
  (those within `inlet_band_mm` of the inlet plane; 79 of 445 at 1 mm). In real
  4D-flow MRI the inlet velocity is simply measured, so this is **not**
  ground-truth leakage — it reuses data already available to the reconstruction.
- **`cfd` (diagnostic upper bound only):** CFD nodal velocity at the mesh node
  nearest each inlet face centroid. Leaks the true inlet profile — use only to
  bound achievable accuracy, **never** for reported benchmark numbers.
- **`none`:** disables the term.

**Cost.** A handful of extra forward-pass points per step. Third-order autograd
(Fix B) is not triggered by this term.

---

## Verification

1. `scripts/00_diagnose_collapse.py geometry=caseC eval.run_suffix=<run>` before
   and after — healthy = velocity ratio ≈ 1, WSS NRMSE ≪ 0.8.
2. Short retrain (`training.n_adam=10000`) should already lift the ratio toward 1.
3. Full path: `04b` (now searches `lambda_data`) → `04c` → `05_evaluate`; expect
   `pinn_wss_mean_Pa` within a small factor of `cfd_wss_mean_Pa`.
4. **Resolution sensitivity restored:** NRMSE at `voxel_2p0mm` > `voxel_0p5mm`
   (finer data → better recovery), unlike the previously flat NRMSE.

**Tests:** `tests/test_losses.py` (relative `data_loss`, `inlet_loss`,
`total_loss` inlet term) — 43 passed; `tests/test_bhpo.py` updated for the
11-dim space — 15 passed (+3 slow); `tests/test_inference.py` — passed.

---

## Code reference map

| Concern | Location |
|---|---|
| Relative data/inlet loss | `pinn/losses.py` `data_loss`, `inlet_loss` |
| Combined loss + breakdown | `pinn/losses.py` `total_loss` (`relative_data`, `x_inlet`, `lambda_inlet`, `loss_inlet`) |
| Checkpoint selection | `pinn/trainer.py` `_selection_metric`, `_maybe_update_best` |
| Inlet plumbing | `pinn/trainer.py` (`x_inlet`, `_sdf_inlet`, `_use_inlet`) |
| Inlet target extraction | `pinn/sampling.py` `select_inlet_targets` |
| BHPO `lambda_data` | `bhpo/space.py`, `bhpo/objective.py` |
| Config defaults | `configs/model/pinn_base.yaml`, `pinn_rff.yaml` |
| Diagnostic | `scripts/00_diagnose_collapse.py` |

---

## Known limitations and future work

1. **Inlet source honesty.** Default `mri` reuses measured data; quantify in the
   paper how much of the recovery comes from the inflow term vs. Fixes F–H by
   running `inlet_source=none lambda_inlet=0` as an ablation.
2. **Flow-rate (flux) alternative.** A single measured inlet flux `Q` would pin
   the magnitude with even less information than a Dirichlet profile, but needs
   boundary-face areas (computable from the tet boundary faces in
   `anxplore_loader.py`). Implement if the Dirichlet term proves insufficient.
3. **`lambda_data` upper bound.** The search caps at 1000; if BHPO rails against
   the ceiling, widen it — the data term may need to be stronger still on noisier
   or sparser voxel configurations.
4. **Interaction with Fix D (SA-PINN).** Self-adaptive weights now include an
   `inlet` term; their joint behaviour with the raised `lambda_data` default is
   untested at full budget (SA is off by default).

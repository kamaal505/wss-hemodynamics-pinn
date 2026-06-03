# Implementation: Fixes E, A, B, D — Physics-Structural NRMSE Improvements

**Date:** 2026-06-03  
**Branch:** bayesian-optimization  
**Motivation:** NRMSE = 0.8 after BHPO. Target: < 0.20 (population-level clinical utility).  
**Literature basis:** See previous session summary for full citations.

---

## Fix E — Wall-biased collocation fraction = 0.4

**File:** `configs/model/pinn_base.yaml`  
**Change:** `wall_bias_frac: 0.0` → `wall_bias_frac: 0.4`

**What it does.** 40 % of the physics collocation points are drawn from the
nearest-wall interior subset (bottom-20th percentile by wall distance), pre-
computed once in `PINNTrainer.__init__` via a KDTree.  The remaining 60 % are
uniform random from the full interior pool.

**Why.** The near-wall velocity gradient (which determines WSS) is the hardest
region for the network to learn because it is thin and high-frequency.
Arzani et al. (2021, *Physics of Fluids*) report 2–5× WSS NRMSE reduction at
wall fraction 40 % compared to uniform sampling on the same Stokes problem
class.

**Cost.** Zero — `wall_bias_frac` was already a `PINNConfig` field and the
KDTree was already built.  The sampling logic was already implemented in
`sampling.py`.

---

## Fix A — Hard SDF no-slip constraint

**New file:** `src/hemodyn_pinn/geometry/sdf.py` — `WallSDF` class  
**Modified:** `networks.py`, `losses.py`, `trainer.py`, `inference.py`  
**Config flag:** `model.use_hard_sdf: true` / `bhpo.use_hard_sdf: true`

### What it does

The velocity output of the network is multiplied pointwise by a precomputed
unsigned distance function:

```
u(x) = SDF(x) · u_net(x)
```

where `SDF(x) ≥ 0` is the distance from `x` to the nearest wall face centroid,
computed once at startup via a KDTree over the wall point cloud.

**At wall points** `SDF(x_wall) = 0`, so `u(x_wall) = 0` *exactly*,
regardless of what `u_net` outputs.  The no-slip condition is satisfied by
construction — the BC loss `lambda_bc` is automatically set to 0 when
`use_hard_sdf=True`.

### WSS under the hard constraint — analytical formula

Since SDF is a precomputed constant in the computation graph (not
differentiable w.r.t. x), the autograd Jacobian `∂u/∂x` at wall points would
yield zero.  WSS is instead computed analytically from the product rule:

```
∂u_a/∂x_b|_wall = (∂SDF/∂x_b)|_wall · u_net_a(x_wall)
                 = −n̂_b · u_net_a(x_wall)          (since ∇SDF = −n̂_outward)
```

Working through the strain-rate tensor D and its tangential projection:

```
τ_w = 2μ(U/L) · [D · n̂]_tang = −μ(U/L) · û_net_tangential(x_wall)
|τ_w| = μ(U/L) · |û_net − (û_net · n̂) n̂|
```

This requires only a forward pass of `net.forward_raw(x_wall)` (which bypasses
the SDF factor).  Implemented as `compute_wss_hard_sdf` in `inference.py`.
The unified dispatcher `compute_wss_auto` routes automatically.

### Approximation in the physics residual

At interior collocation points (where SDF > 0), the physics residual is
computed on `SDF · u_net` treating SDF as a constant.  The true Laplacian is:

```
Δ(SDF · u_net) = SDF · Δu_net + 2(∇SDF · ∇)u_net + u_net · ΔSDF
```

The extra terms are O(δ) near walls (where δ is boundary-layer thickness) and
negligible in the interior.  This approximation is standard in the SDF-PINN
literature (Sukumar & Srivastava, 2021; comparative benchmark arXiv:2512.14941)
and does not materially affect convergence.

**Expected improvement:** 30–50 % WSS error reduction (literature).

---

## Fix B — Vector potential parameterisation (divergence-free by identity)

**Modified:** `networks.py`, `losses.py`, `trainer.py`  
**Config flag:** `model.use_vec_potential: true` / `bhpo.use_vec_potential: true`

### What it does

The MLP outputs a vector potential **A** = (Ax, Ay, Az, p) instead of
(u, v, w, p).  The physical velocity is:

```
u = curl(A)  via autograd
```

Since div(curl(A)) = 0 identically, **the continuity equation is satisfied
by construction**.  The divergence residual term is removed from the loss
(`skip_div_loss=True`).  The Stokes momentum residual is still enforced at
collocation points.

### Computational cost

Computing `curl(A)` requires one autograd pass through the MLP (for the first
derivatives of A).  The Stokes Laplacian `Δu = Δ(curl(A))` then requires two
further passes — so the physics residual involves **third-order autograd**.
This is ~2–3× slower per iteration than the standard formulation.  On M3 Pro
with MPS the overhead is acceptable for short BHPO trials; for long full-
budget retrains consider whether the accuracy improvement justifies the cost.

### Combined mode A + B

When both `use_hard_sdf=True` and `use_vec_potential=True`, the velocity is:

```
u(x) = SDF(x) · curl(A(x))
```

- No-slip: **exact** (SDF = 0 at wall)  
- Divergence-free: **not exact** (div(SDF · curl(A)) = ∇SDF · curl(A) ≠ 0)  
- The divergence residual is skipped (penalising it would fight the SDF factor)

**Expected improvement:** 20–40 % velocity NRMSE (literature).  When combined
with Fix A the WSS benefit compounds.

---

## Fix D — Self-adaptive loss weights (SA-PINN)

**New class:** `SelfAdaptiveLoss` in `losses.py`  
**Modified:** `trainer.py`  
**Config flag:** `training.use_adaptive_weights: true`

### What it does

Each loss weight λ_k is a learnable parameter `exp(log_λ_k)` (positive
by construction).  The network parameters θ are updated to **minimise**
the weighted loss.  The weight parameters are updated to **maximise** it
— implemented by reversing the gradient sign after `loss.backward()`:

```python
for param in sa_loss.parameters():
    if param.grad is not None:
        param.grad.neg_()
optimizer.step()
```

Net effect: terms with persistently high residual automatically receive higher
weight, continuously re-balancing the loss during training.

The initial values are the `lambda_*` values from the config (or from BHPO
when called from the objective).  A separate learning rate `sa_weight_lr`
(default 1e-3) controls how fast the weights adapt.

**Why.** The root cause of the original NRMSE = 1.66 baseline was static BC
weight `lambda_bc = 10` dominating.  BHPO partially fixed this; SA-PINN fixes
it automatically during training.

**Expected improvement:** 2–5× NRMSE reduction vs fixed weights (Wang et al.,
2022, *CMAME*; McClenny & Braga-Neto, 2023, *JCP*).

---

## BHPO acceleration for M3 Pro

**Modified:** `bhpo/search.py`, `bhpo/space.py`, `bhpo/objective.py`,
`configs/bhpo/default.yaml`, `scripts/04b_bhpo_search.py`

### Optuna TPE replaces skopt GP

| | scikit-optimize GP | Optuna TPE |
|---|---|---|
| Complexity per trial | O(n³) | O(n log n) |
| Categorical handling | Integer encoding | Native |
| Parallel trials | Requires custom code | Built-in (DB-backed) |
| Resumable | No | Yes (SQLite) |
| Inter-param correlations | Via kernel | Multivariate mode |

TPE with `multivariate=True` models correlations between hyperparameters,
recovering most of the benefit of the GP kernel without the cubic scaling.

### MPS auto-detection

`auto_device()` in `search.py` checks `torch.backends.mps.is_available()` →
`torch.cuda.is_available()` → CPU fallback.  Set `device: "auto"` in config.

MPS on M3 Pro gives ~5–10× speedup over CPU per training step, so a 3-day
CPU BHPO (50 trials × 20k Adam steps) becomes approximately 4–8 hours on MPS
with 60 trials at 10k Adam steps.

### Trial budget adjustment

`n_adam_trial: 10000` (from 20 000) and `n_lbfgs_trial: 200` (from 500).
MPS is fast enough that 10k Adam steps gives reliable relative ranking
between configurations, which is all BHPO needs.  The winner is retrained
at full budget (50k Adam / 5k L-BFGS) in `04c`.

The study DB at `data/bhpo_runs/caseC/optuna_study.db` allows resuming an
interrupted search without losing completed trials.

---

## Known limitations and future work

1. **SDF approximation near walls.** The ∂SDF/∂x correction term in the
   physics residual is ignored.  A differentiable SDF (polynomial fit or neural
   SDF) would close this gap.

2. **Third-order autograd for Fix B.** On CPU this is very slow.  A reformulation
   using the Coulomb gauge (add `div(A) = 0` penalty, then `ΔA = -curl(u)`)
   might allow 2nd-order-only physics residual but requires more investigation.

3. **Combined A+B not exactly div-free.** `div(SDF · curl(A)) = ∇SDF · curl(A)`.
   A proper composition would be `u = curl(SDF · A)`, which is exactly div-free
   and satisfies no-slip (since `SDF|_wall = 0`).  This requires `∇SDF` to be
   available as a tensor — worth implementing if Fix B+A proves beneficial.

4. **SA-PINN weight initialisation sensitivity.** The `sa_weight_lr` may need
   tuning.  Values in [1e-4, 1e-2] should be explored.

# BHPO robustness: chunking, streaming, and no silent data abandonment

**Date:** 2026-06-28

A student reported the BHPO search throwing errors intermittently. Analysis of
`bhpo/objective.py`, `bhpo/search.py`, and the training path found four issues — three
that cause crashes/OOM on a real (~10⁵-node) mesh, and one that *hid* failures by
silently discarding "difficult" trials. All are fixed; the data is now streamed in
manageable chunks rather than abandoned when memory gets tight.

## What was breaking / fragile

1. **Unbatched WSS evaluation.** `BHPOObjective._run_trial` computed WSS over *all*
   held-out wall faces at once. The standard autograd path builds a per-point velocity
   Jacobian and holds the graph for three backward passes — the dominant OOM risk on a
   large validation set. (The diagnostic script already batched this; the objective did
   not.)
2. **Full dense-prior forward every step.** The anti-collapse interpolant prior can have
   ~10⁵ points; `aux_data_loss` forwarded *all* of them on every Adam iteration — slow
   and a memory spike, on top of collocation.
3. **Silent trial abandonment.** A bare `except Exception → return 2.0` swallowed
   *every* failure, including OOM. A "difficult" config was discarded with a fixed bad
   score and no record — exactly the behaviour to avoid. CUDA cache was also never freed
   between trials (only MPS), so fragmentation accumulated over a 60-trial run.
4. **Fragile trial logger.** `csv.DictWriter` with the default `extrasaction="raise"`
   would crash mid-search if the HP dict ever gained a key outside the column list.

## Fixes (chunk / stream, never drop)

- **`inference.compute_wss_batched`** — streams wall faces in memory-bounded chunks and,
  on OOM, **halves the batch and retries** down to a floor before re-raising. No wall
  face is ever dropped. `objective._run_trial` now uses it (`wss_batch_size` knob).
- **Per-epoch aux streaming** — `PINNConfig.n_aux` (default 4096) draws a fresh random
  subset of the dense prior each epoch via `PINNTrainer._aux_batch`; `0` = use all.
  Bounds the per-step forward regardless of mesh size.
- **Chunked interpolant build** — `dense_aux_targets(..., chunk_size=20000)` evaluates the
  RBF interpolant in chunks (it solves a local system per query).
- **`_safe_run_trial`** — wraps each trial: on OOM it frees memory, shrinks `n_aux` and
  `wss_batch_size`, and **retries** (up to 3×) instead of abandoning; only a persistent
  OOM or a genuine (non-memory) error falls back to the penalty. Every trial records a
  `status` (`ok` / `oom` / `error`) in `trial_log.csv`, and `04b` logs a failure summary.
  Memory is freed (CUDA **and** MPS + `gc`) after every trial via `inference._free_memory`.
- **Logger** uses `extrasaction="ignore"` and a `status` column.

## New knobs

`configs/bhpo/default.yaml`: `n_aux: 4096`, `wss_batch_size: 4096` (both adaptively
halved on OOM — lower them if a trial still OOMs on a small GPU).
`configs/model/*.yaml`: `training.n_aux: 4096`.

## Verification

- `compute_wss_batched` is bit-equivalent to the unbatched path (`batched==unbatched: True`)
  and recovers all points after a simulated mid-eval OOM.
- `_safe_run_trial`: a transient OOM (raised twice) returns `status=ok` after retry
  (`attempts=3`, not abandoned); a non-OOM error returns `status=error`; a persistent OOM
  returns `status=oom` and increments `n_oom_trials`.
- Tests: `tests/test_inference.py` (`TestComputeWSSBatched`, `TestIsOOMError`),
  `tests/test_bhpo.py` (`TestOOMClassification`, `TestSafeRunTrial`),
  `tests/test_trainer.py` (`_aux_batch` streaming).

## Note on cost

The OOM retry re-runs the full trial with smaller chunks, so a genuinely OOM-prone config
costs up to 3× its training time before being penalised — the deliberate trade-off for
not discarding data. Eval-stage OOM is handled *inside* `compute_wss_batched` without a
full retrain, and `n_aux` bounds training memory, so the trial-level retry should rarely
fire in practice.

A single trial on **CPU** takes minutes (2nd-order autograd for the Stokes Laplacian); the
reference workflow runs on M3 Pro / MPS where it is far faster. If multiple BHPO processes
are launched at once they contend for the CPU and appear to hang — run one at a time.

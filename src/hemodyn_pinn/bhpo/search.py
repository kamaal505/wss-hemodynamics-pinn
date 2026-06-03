"""BHPO search loop — Optuna TPE (primary) with skopt GP fallback.

The primary backend is Optuna 4.7+ with the TPE sampler.  Advantages over
the legacy GP (scikit-optimize) for this problem:
    - TPE scales O(n log n) vs O(n³) for GP — critical for 50+ trials
    - Native handling of categorical dimensions (activation, n_hidden, …)
    - Pruning support for early termination of bad trials
    - Single-process sequential execution compatible with MPS (Apple Silicon
      MPS is not fork-safe; parallel processes require a separate DB setup)

The legacy ``skopt`` path is preserved as a fallback for environments where
Optuna is unavailable.

Output files written to out_dir:
    best_params.json   — winning HP dict (read by 04c)
    convergence.json   — per-trial objective trace (same schema as before)
    trial_log.csv      — one row per trial (from BHPOObjective)
    optuna_study.db    — SQLite study (Optuna path only; enables resuming)
"""

from __future__ import annotations

import json
import logging
import pathlib
import pickle
from typing import Callable, Optional

from hemodyn_pinn.utils.seeds import BHPO_SEED

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device auto-detection
# ---------------------------------------------------------------------------


def auto_device() -> str:
    """Return "mps", "cuda", or "cpu" depending on hardware availability."""
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------------------
# Optuna search (primary)
# ---------------------------------------------------------------------------


def run_bhpo_search_optuna(
    objective_fn: "BHPOObjective",  # type: ignore[name-defined]  # noqa: F821
    n_calls: int = 50,
    n_initial_points: int = 10,
    out_dir: Optional[pathlib.Path | str] = None,
) -> tuple[object, dict]:
    """Run the BHPO search with Optuna TPE.

    Parameters
    ----------
    objective_fn:
        A ``BHPOObjective`` instance.  Its ``optuna_objective`` method is
        used as the Optuna objective.
    n_calls:
        Total number of trials (random warm-up + TPE).
    n_initial_points:
        Number of random trials before TPE takes over.
    out_dir:
        If given, writes outputs including an SQLite study DB for resuming.

    Returns
    -------
    study : optuna.Study
    best_hp : dict
    """
    import optuna
    from optuna.samplers import TPESampler

    out_path = pathlib.Path(out_dir) if out_dir is not None else None
    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)

    storage: Optional[str] = None
    if out_path is not None:
        db = out_path / "optuna_study.db"
        storage = f"sqlite:///{db}"

    sampler = TPESampler(
        seed=BHPO_SEED,
        n_startup_trials=n_initial_points,
        multivariate=True,   # model inter-parameter correlations
    )
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        storage=storage,
        study_name="bhpo",
        load_if_exists=True,   # resume if DB already exists
    )

    # Silence Optuna's per-trial INFO logs — our objective logs its own.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    log.info(
        "Starting Optuna BHPO: %d trials (%d random warm-up) seed=%d",
        n_calls, n_initial_points, BHPO_SEED,
    )
    study.optimize(objective_fn.optuna_objective, n_trials=n_calls)

    best_hp = study.best_params
    log.info(
        "BHPO complete. Best WSS-NRMSE=%.4f\n  %s",
        study.best_value,
        "  ".join(f"{k}={v}" for k, v in best_hp.items()),
    )

    if out_path is not None:
        _save_results_optuna(study, out_path)

    return study, best_hp


def _save_results_optuna(study: object, out_dir: pathlib.Path) -> None:
    import optuna
    assert isinstance(study, optuna.Study)

    with open(out_dir / "best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)

    func_vals = [t.value for t in study.trials if t.value is not None]
    best_so_far: list[float] = []
    current_best = float("inf")
    for v in func_vals:
        current_best = min(current_best, v)
        best_so_far.append(current_best)

    convergence = {
        "func_vals":   func_vals,
        "best_so_far": best_so_far,
        "best_nrmse":  study.best_value,
        "best_params": study.best_params,
    }
    with open(out_dir / "convergence.json", "w") as f:
        json.dump(convergence, f, indent=2)

    log.info("BHPO results saved to %s", out_dir)


# ---------------------------------------------------------------------------
# Legacy skopt search (fallback)
# ---------------------------------------------------------------------------


def run_bhpo_search_skopt(
    objective_fn: Callable[[list], float],
    n_calls: int = 50,
    n_initial_points: int = 10,
    out_dir: Optional[pathlib.Path | str] = None,
) -> tuple[object, dict]:
    """Run the legacy GP-based BHPO search with scikit-optimize."""
    from skopt import gp_minimize
    from hemodyn_pinn.bhpo.space import SEARCH_SPACE, decode_params

    log.info(
        "Starting skopt BHPO: %d calls (%d initial random) seed=%d",
        n_calls, n_initial_points, BHPO_SEED,
    )
    result = gp_minimize(
        func=objective_fn,
        dimensions=SEARCH_SPACE,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        random_state=BHPO_SEED,
        noise=1e-6,
        verbose=True,
    )
    best_hp = decode_params(result.x)
    log.info(
        "BHPO complete. Best WSS-NRMSE=%.4f\n  %s",
        float(result.fun),
        "  ".join(f"{k}={v}" for k, v in best_hp.items()),
    )
    if out_dir is not None:
        _save_results_skopt(result, best_hp, pathlib.Path(out_dir))
    return result, best_hp


def _save_results_skopt(result: object, best_hp: dict, out_dir: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "result.pkl", "wb") as f:
        pickle.dump(result, f)
    with open(out_dir / "best_params.json", "w") as f:
        json.dump(best_hp, f, indent=2)
    func_vals = [float(v) for v in result.func_vals]
    best_so_far: list[float] = []
    current_best = float("inf")
    for v in func_vals:
        current_best = min(current_best, v)
        best_so_far.append(current_best)
    convergence = {
        "func_vals": func_vals,
        "best_so_far": best_so_far,
        "best_nrmse": float(result.fun),
        "best_params": best_hp,
    }
    with open(out_dir / "convergence.json", "w") as f:
        json.dump(convergence, f, indent=2)
    log.info("BHPO results saved to %s", out_dir)


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------


def run_bhpo_search(
    objective_fn,
    n_calls: int = 50,
    n_initial_points: int = 10,
    out_dir: Optional[pathlib.Path | str] = None,
    backend: str = "auto",
) -> tuple[object, dict]:
    """Run BHPO search, auto-selecting Optuna or skopt.

    Parameters
    ----------
    backend:
        "optuna" | "skopt" | "auto" (prefers Optuna if installed).
    """
    use_optuna = False
    if backend == "optuna":
        use_optuna = True
    elif backend == "skopt":
        use_optuna = False
    else:
        try:
            import optuna  # noqa: F401
            use_optuna = True
        except ImportError:
            log.warning("Optuna not found, falling back to skopt.")

    if use_optuna:
        return run_bhpo_search_optuna(
            objective_fn, n_calls=n_calls,
            n_initial_points=n_initial_points, out_dir=out_dir,
        )
    return run_bhpo_search_skopt(
        objective_fn, n_calls=n_calls,
        n_initial_points=n_initial_points, out_dir=out_dir,
    )


# ---------------------------------------------------------------------------
# Result loader (unchanged public API)
# ---------------------------------------------------------------------------


def load_best_params(bhpo_out_dir: pathlib.Path | str) -> dict:
    """Load the winning hyperparameter dict from a completed BHPO run."""
    path = pathlib.Path(bhpo_out_dir) / "best_params.json"
    if not path.exists():
        raise FileNotFoundError(
            f"BHPO result not found: {path}\n"
            "Run scripts/04b_bhpo_search.py first."
        )
    with open(path) as f:
        return json.load(f)

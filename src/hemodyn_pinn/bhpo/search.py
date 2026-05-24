"""GP-based BHPO search loop using scikit-optimize.

Usage
-----
Instantiate a ``BHPOObjective``, then call ``run_bhpo_search``::

    result, best_hp = run_bhpo_search(
        objective_fn=bhpo_obj,
        n_calls=50,
        n_initial_points=10,
        out_dir=pathlib.Path("data/bhpo_runs/caseC"),
    )

The ``objective_fn`` must accept a raw skopt params list and return a float
(i.e. ``BHPOObjective.__call__`` satisfies this directly).
"""

from __future__ import annotations

import json
import logging
import pathlib
import pickle
from typing import Callable, Optional

from skopt import gp_minimize

from hemodyn_pinn.bhpo.space import SEARCH_SPACE, decode_params
from hemodyn_pinn.utils.seeds import BHPO_SEED

log = logging.getLogger(__name__)


def run_bhpo_search(
    objective_fn: Callable[[list], float],
    n_calls: int = 50,
    n_initial_points: int = 10,
    out_dir: Optional[pathlib.Path | str] = None,
) -> tuple[object, dict]:
    """Run the BHPO search with a GP surrogate (Matérn kernel).

    Parameters
    ----------
    objective_fn:
        Callable that accepts a raw skopt params list and returns a float
        (WSS NRMSE).  ``BHPOObjective.__call__`` satisfies this interface.
    n_calls:
        Total number of objective evaluations (initial random + GP-guided).
    n_initial_points:
        Number of random evaluations before the GP starts fitting.  Should
        be at least ``len(SEARCH_SPACE)`` for reasonable initial coverage.
    out_dir:
        If given, writes ``result.pkl``, ``best_params.json``, and
        ``convergence.json`` into this directory after the search completes.

    Returns
    -------
    result : skopt.OptimizeResult
        Full result object (includes all evaluated points and function values).
    best_hp : dict
        Decoded hyperparameter dict at the best-found point.
    """
    log.info(
        "Starting BHPO: %d calls (%d initial random) seed=%d",
        n_calls, n_initial_points, BHPO_SEED,
    )

    result = gp_minimize(
        func=objective_fn,
        dimensions=SEARCH_SPACE,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        random_state=BHPO_SEED,
        # Small noise floor prevents the GP from over-fitting perfect noiseless
        # evaluations and keeps the Cholesky factorisation stable.
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
        _save_results(result, best_hp, pathlib.Path(out_dir))

    return result, best_hp


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _save_results(
    result: object,
    best_hp: dict,
    out_dir: pathlib.Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "result.pkl", "wb") as f:
        pickle.dump(result, f)

    with open(out_dir / "best_params.json", "w") as f:
        json.dump(best_hp, f, indent=2)

    # Convergence trace: (trial_index, best_so_far) pairs — useful for plots.
    func_vals = [float(v) for v in result.func_vals]
    best_so_far = []
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


def load_best_params(bhpo_out_dir: pathlib.Path | str) -> dict:
    """Load the winning hyperparameter dict from a completed BHPO run.

    Parameters
    ----------
    bhpo_out_dir:
        Directory produced by ``run_bhpo_search`` (contains ``best_params.json``).
    """
    path = pathlib.Path(bhpo_out_dir) / "best_params.json"
    if not path.exists():
        raise FileNotFoundError(
            f"BHPO result not found: {path}\n"
            "Run scripts/04b_bhpo_search.py first."
        )
    with open(path) as f:
        return json.load(f)

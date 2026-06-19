"""BHPO search space definition and parameter decoding.

Supports both the legacy skopt interface (decode_params from a list) and the
Optuna interface (optuna_suggest from a Trial object).  The Optuna TPE sampler
handles categorical dimensions natively, so integer-encoding of categoricals
is no longer needed in the Optuna path.

The search space covers 11 dimensions; lambda_anchor is fixed.  lambda_data is
searched (log-uniform, 1–1000): under-weighting it collapses the Stokes field to
the trivial u≡0 solution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import optuna

# ---------------------------------------------------------------------------
# Discrete choice mappings
# ---------------------------------------------------------------------------

N_HIDDEN_MAP: list[int] = [64, 128, 256]
ACTIVATION_MAP: list[str] = ["tanh", "swish", "gelu"]
N_COLLOC_MAP: list[int] = [5_000, 10_000, 20_000]

# ---------------------------------------------------------------------------
# Legacy skopt search space (kept for backward compatibility)
# ---------------------------------------------------------------------------

try:
    from skopt.space import Integer, Real
    SEARCH_SPACE = [
        Real(1.0, 1000.0, prior="log-uniform", name="lambda_data"),
        Real(0.01, 10.0, prior="log-uniform", name="lambda_phys"),
        Real(0.10, 10.0, prior="log-uniform", name="lambda_bc"),
        Integer(3, 6,    name="n_layers"),
        Integer(0, 2,    name="n_hidden_idx"),
        Integer(0, 2,    name="activation_idx"),
        Integer(0, 1,    name="use_rff_int"),
        Real(0.5, 10.0, prior="log-uniform", name="rff_sigma"),
        Real(5e-4, 5e-3, prior="log-uniform", name="lr_adam"),
        Integer(0, 2,    name="n_colloc_idx"),
        Real(0.0, 0.5,   name="wall_bias_frac"),
    ]
    HP_NAMES: list[str] = [dim.name for dim in SEARCH_SPACE]
except ImportError:
    SEARCH_SPACE = []   # type: ignore[assignment]
    HP_NAMES = []


def decode_params(params: list) -> dict:
    """Convert a raw skopt parameter list to a human-readable HP dict."""
    (
        lambda_data, lambda_phys, lambda_bc,
        n_layers, n_hidden_idx, activation_idx, use_rff_int, rff_sigma,
        lr_adam, n_colloc_idx, wall_bias_frac,
    ) = params
    return {
        "lambda_data":    float(lambda_data),
        "lambda_phys":    float(lambda_phys),
        "lambda_bc":      float(lambda_bc),
        "n_layers":       int(n_layers),
        "n_hidden":       N_HIDDEN_MAP[int(n_hidden_idx)],
        "activation":     ACTIVATION_MAP[int(activation_idx)],
        "use_rff":        bool(int(use_rff_int)),
        "rff_sigma":      float(rff_sigma),
        "lr_adam":        float(lr_adam),
        "n_colloc":       N_COLLOC_MAP[int(n_colloc_idx)],
        "wall_bias_frac": float(wall_bias_frac),
    }


# ---------------------------------------------------------------------------
# Optuna suggest interface
# ---------------------------------------------------------------------------


def optuna_suggest(trial: "optuna.Trial") -> dict:
    """Sample one hyperparameter configuration from an Optuna Trial.

    The TPE sampler handles all dimension types natively — no integer-encoding
    of categoricals is needed here.  This is the recommended path on M3 Pro.

    Parameters
    ----------
    trial:
        An ``optuna.Trial`` object provided by the study's objective function.

    Returns
    -------
    dict
        Human-readable HP dict with the same keys as ``decode_params``.
    """
    lambda_data    = trial.suggest_float("lambda_data", 1.0, 1000.0, log=True)
    lambda_phys    = trial.suggest_float("lambda_phys", 0.01, 10.0, log=True)
    lambda_bc      = trial.suggest_float("lambda_bc",   0.10, 10.0, log=True)
    n_layers       = trial.suggest_int("n_layers",      3, 6)
    n_hidden       = trial.suggest_categorical("n_hidden",    N_HIDDEN_MAP)
    activation     = trial.suggest_categorical("activation",  ACTIVATION_MAP)
    use_rff        = trial.suggest_categorical("use_rff",     [False, True])
    rff_sigma      = trial.suggest_float("rff_sigma",   0.5, 10.0, log=True)
    lr_adam        = trial.suggest_float("lr_adam",     5e-4, 5e-3, log=True)
    n_colloc       = trial.suggest_categorical("n_colloc",    N_COLLOC_MAP)
    wall_bias_frac = trial.suggest_float("wall_bias_frac", 0.0, 0.5)

    return {
        "lambda_data":    lambda_data,
        "lambda_phys":    lambda_phys,
        "lambda_bc":      lambda_bc,
        "n_layers":       n_layers,
        "n_hidden":       n_hidden,
        "activation":     activation,
        "use_rff":        use_rff,
        "rff_sigma":      rff_sigma,
        "lr_adam":        lr_adam,
        "n_colloc":       n_colloc,
        "wall_bias_frac": wall_bias_frac,
    }

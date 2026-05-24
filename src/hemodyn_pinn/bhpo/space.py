"""BHPO search space definition and parameter decoding.

Categorical dimensions (activation, n_hidden, use_rff, n_colloc) are encoded
as Integer indices so the GP kernel stays numerically well-behaved.  Use
``decode_params`` to convert a raw skopt parameter list to a human-readable
hyperparameter dict before passing it to the PINN trainer.
"""

from __future__ import annotations

from skopt.space import Integer, Real

# ---------------------------------------------------------------------------
# Discrete choice mappings (index → value)
# ---------------------------------------------------------------------------

N_HIDDEN_MAP: list[int] = [64, 128, 256]
ACTIVATION_MAP: list[str] = ["tanh", "swish", "gelu"]
N_COLLOC_MAP: list[int] = [5_000, 10_000, 20_000]

# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------
# Order here is the canonical order used by skopt and decode_params.
# Do NOT reorder without updating decode_params.

SEARCH_SPACE = [
    # Loss weights (λ_data fixed at 1.0; λ_anchor fixed at 1.0)
    Real(0.01, 10.0, prior="log-uniform", name="lambda_phys"),
    Real(0.10, 10.0, prior="log-uniform", name="lambda_bc"),
    # Architecture
    Integer(3, 6,    name="n_layers"),
    Integer(0, 2,    name="n_hidden_idx"),    # 0→64, 1→128, 2→256
    Integer(0, 2,    name="activation_idx"),  # 0→tanh, 1→swish, 2→gelu
    Integer(0, 1,    name="use_rff_int"),     # 0→False, 1→True
    Real(0.5, 10.0, prior="log-uniform", name="rff_sigma"),
    # Training dynamics
    Real(5e-4, 5e-3, prior="log-uniform", name="lr_adam"),
    Integer(0, 2,    name="n_colloc_idx"),    # 0→5000, 1→10000, 2→20000
    Real(0.0, 0.5,   name="wall_bias_frac"),
]

HP_NAMES: list[str] = [dim.name for dim in SEARCH_SPACE]


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

def decode_params(params: list) -> dict:
    """Convert a raw skopt parameter list to a human-readable HP dict.

    Parameters
    ----------
    params:
        List of values in the same order as ``SEARCH_SPACE``.

    Returns
    -------
    dict
        Keys: lambda_phys, lambda_bc, n_layers, n_hidden, activation,
              use_rff, rff_sigma, lr_adam, n_colloc, wall_bias_frac.
    """
    (
        lambda_phys, lambda_bc,
        n_layers, n_hidden_idx, activation_idx, use_rff_int, rff_sigma,
        lr_adam, n_colloc_idx, wall_bias_frac,
    ) = params

    return {
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

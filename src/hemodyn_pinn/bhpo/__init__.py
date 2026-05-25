"""Bayesian Hyperparameter Optimisation for the PINN.

Public API
----------
BHPOObjective  — callable objective (WSS NRMSE on held-out wall faces)
run_bhpo_search — GP-based search loop via scikit-optimize
load_best_params — load winning HPs from a completed BHPO run
decode_params   — decode a raw skopt params list to a HP dict
SEARCH_SPACE    — skopt dimension list (canonical ordering)
"""

from hemodyn_pinn.bhpo.objective import BHPOObjective
from hemodyn_pinn.bhpo.search import load_best_params, run_bhpo_search
from hemodyn_pinn.bhpo.space import SEARCH_SPACE, decode_params

__all__ = [
    "BHPOObjective",
    "run_bhpo_search",
    "load_best_params",
    "decode_params",
    "SEARCH_SPACE",
]

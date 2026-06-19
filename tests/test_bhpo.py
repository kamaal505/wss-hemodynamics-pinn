"""BHPO module tests (Phase 3 gate — required by CLAUDE.md §7).

Covers:
- space.py: decode_params key/value correctness; optuna_suggest key coverage.
- search.py: auto_device returns a valid string.
- objective.py: BHPOObjective val-split sizes; missing wall_pts_m raises; one-trial smoke.

All tests run on CPU.  No Optuna study DB is created (out_dir=None).
The smoke test uses n_adam_trial=5 and a tiny network (n_hidden=16, n_layers=2)
to complete in < 30 s.
"""

from __future__ import annotations

import numpy as np
import pytest

from hemodyn_pinn.bhpo.space import (
    ACTIVATION_MAP,
    N_COLLOC_MAP,
    N_HIDDEN_MAP,
    decode_params,
    optuna_suggest,
)
from hemodyn_pinn.bhpo.search import auto_device
from hemodyn_pinn.bhpo.objective import BHPOObjective


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


_ALL_HP_KEYS = frozenset(
    ["lambda_data", "lambda_phys", "lambda_bc", "n_layers", "n_hidden",
     "activation", "use_rff", "rff_sigma", "lr_adam",
     "n_colloc", "wall_bias_frac"]
)

_BASE_PARAMS = [
    10.0,  # lambda_data
    1.0,   # lambda_phys
    5.0,   # lambda_bc
    4,     # n_layers
    1,     # n_hidden_idx  → N_HIDDEN_MAP[1] = 128
    0,     # activation_idx → ACTIVATION_MAP[0] = "tanh"
    0,     # use_rff_int   → False
    1.0,   # rff_sigma
    1e-3,  # lr_adam
    1,     # n_colloc_idx  → N_COLLOC_MAP[1] = 10_000
    0.3,   # wall_bias_frac
]


class _MockTrial:
    """Minimal stand-in for optuna.Trial, used to test optuna_suggest."""

    def suggest_float(self, name: str, low: float, high: float, **kwargs) -> float:
        return (low + high) / 2.0

    def suggest_int(self, name: str, low: int, high: int, **kwargs) -> int:
        return low

    def suggest_categorical(self, name: str, choices: list):
        return choices[0]


def _minimal_bhpo_data():
    """Small synthetic arrays for BHPOObjective construction."""
    rng = np.random.default_rng(42)
    n_int, n_wall, n_data = 100, 30, 15

    interior = rng.uniform(0.1, 0.9, (n_int, 3)).astype(np.float32)
    wall     = rng.uniform(0.0, 0.05, (n_wall, 3)).astype(np.float32)
    normals  = rng.normal(0, 1, (n_wall, 3)).astype(np.float32)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    cfd_wss  = rng.uniform(1e-6, 1e-5, (n_wall,)).astype(np.float32)
    anchor   = rng.uniform(0, 1, (1, 3)).astype(np.float32)
    x_data   = rng.uniform(0.1, 0.9, (n_data, 3)).astype(np.float32)
    u_obs    = rng.normal(0, 1e-4, (n_data, 3)).astype(np.float32)
    wall_m   = wall * 0.01628   # SI metres (L_SCALE)
    return interior, wall, normals, cfd_wss, anchor, x_data, u_obs, wall_m


# ---------------------------------------------------------------------------
# decode_params
# ---------------------------------------------------------------------------


class TestDecodeParams:
    def test_returns_all_hp_keys(self) -> None:
        hp = decode_params(_BASE_PARAMS)
        assert set(hp.keys()) == _ALL_HP_KEYS

    def test_n_hidden_index_mapping(self) -> None:
        for idx, expected in enumerate(N_HIDDEN_MAP):
            params = list(_BASE_PARAMS); params[4] = idx
            assert decode_params(params)["n_hidden"] == expected

    def test_activation_index_mapping(self) -> None:
        for idx, expected in enumerate(ACTIVATION_MAP):
            params = list(_BASE_PARAMS); params[5] = idx
            assert decode_params(params)["activation"] == expected

    def test_n_colloc_index_mapping(self) -> None:
        for idx, expected in enumerate(N_COLLOC_MAP):
            params = list(_BASE_PARAMS); params[9] = idx
            assert decode_params(params)["n_colloc"] == expected

    def test_use_rff_bool_conversion(self) -> None:
        p_false = list(_BASE_PARAMS); p_false[6] = 0
        p_true  = list(_BASE_PARAMS); p_true[6] = 1
        assert decode_params(p_false)["use_rff"] is False
        assert decode_params(p_true)["use_rff"] is True

    def test_float_fields_are_float(self) -> None:
        hp = decode_params(_BASE_PARAMS)
        for key in ("lambda_data", "lambda_phys", "lambda_bc", "rff_sigma",
                    "lr_adam", "wall_bias_frac"):
            assert isinstance(hp[key], float), f"{key} should be float"


# ---------------------------------------------------------------------------
# optuna_suggest
# ---------------------------------------------------------------------------


class TestOptunaWatchdog:
    def test_returns_all_hp_keys(self) -> None:
        hp = optuna_suggest(_MockTrial())  # type: ignore[arg-type]
        assert set(hp.keys()) == _ALL_HP_KEYS

    def test_n_hidden_in_map(self) -> None:
        hp = optuna_suggest(_MockTrial())  # type: ignore[arg-type]
        assert hp["n_hidden"] in N_HIDDEN_MAP

    def test_activation_in_map(self) -> None:
        hp = optuna_suggest(_MockTrial())  # type: ignore[arg-type]
        assert hp["activation"] in ACTIVATION_MAP

    def test_n_colloc_in_map(self) -> None:
        hp = optuna_suggest(_MockTrial())  # type: ignore[arg-type]
        assert hp["n_colloc"] in N_COLLOC_MAP

    def test_n_layers_in_range(self) -> None:
        hp = optuna_suggest(_MockTrial())  # type: ignore[arg-type]
        assert 3 <= hp["n_layers"] <= 6


# ---------------------------------------------------------------------------
# auto_device
# ---------------------------------------------------------------------------


class TestAutoDevice:
    def test_returns_valid_device_string(self) -> None:
        assert auto_device() in {"mps", "cuda", "cpu"}

    def test_cpu_when_no_accelerator_present(self) -> None:
        import torch
        if not torch.backends.mps.is_available() and not torch.cuda.is_available():
            assert auto_device() == "cpu"


# ---------------------------------------------------------------------------
# BHPOObjective — initialisation
# ---------------------------------------------------------------------------


class TestBHPOObjectiveInit:
    def test_val_split_sizes(self) -> None:
        interior, wall, normals, cfd_wss, anchor, x_data, u_obs, _ = _minimal_bhpo_data()
        n_wall = wall.shape[0]
        obj = BHPOObjective(
            interior_pts_nondim=interior, wall_pts_nondim=wall,
            wall_normals=normals, cfd_wss_pa=cfd_wss,
            anchor_pt_nondim=anchor, x_data=x_data,
            u_obs_nondim=u_obs, val_frac=0.2,
            n_adam_trial=2, n_lbfgs_trial=1, device="cpu",
        )
        n_val_expected = max(1, int(n_wall * 0.2))
        assert obj._val_wall_pts.shape[0] == n_val_expected
        assert obj._train_wall_pts.shape[0] == n_wall - n_val_expected

    def test_raises_hard_sdf_without_wall_m(self) -> None:
        interior, wall, normals, cfd_wss, anchor, x_data, u_obs, _ = _minimal_bhpo_data()
        with pytest.raises(ValueError, match="wall_pts_m"):
            BHPOObjective(
                interior_pts_nondim=interior, wall_pts_nondim=wall,
                wall_normals=normals, cfd_wss_pa=cfd_wss,
                anchor_pt_nondim=anchor, x_data=x_data,
                u_obs_nondim=u_obs, wall_pts_m=None,
                use_hard_sdf=True, n_adam_trial=2, device="cpu",
            )


# ---------------------------------------------------------------------------
# BHPOObjective — smoke test (_run_trial)
# ---------------------------------------------------------------------------


_MINIMAL_HP = {
    "lambda_phys": 1.0, "lambda_bc": 1.0, "n_layers": 2,
    "n_hidden": 16, "activation": "tanh", "use_rff": False,
    "rff_sigma": 1.0, "lr_adam": 1e-3, "n_colloc": 50,
    "wall_bias_frac": 0.0,
}


@pytest.mark.slow
class TestBHPOObjectiveSmoke:
    """Full-stack trial: verify a trial completes and returns a finite float."""

    def test_one_trial_standard(self) -> None:
        interior, wall, normals, cfd_wss, anchor, x_data, u_obs, _ = _minimal_bhpo_data()
        obj = BHPOObjective(
            interior_pts_nondim=interior, wall_pts_nondim=wall,
            wall_normals=normals, cfd_wss_pa=cfd_wss,
            anchor_pt_nondim=anchor, x_data=x_data,
            u_obs_nondim=u_obs, val_frac=0.2,
            n_adam_trial=5, n_lbfgs_trial=1, device="cpu",
        )
        nrmse = obj._run_trial(_MINIMAL_HP)
        assert isinstance(nrmse, float)
        assert 0.0 <= nrmse <= 2.0 + 1e-6   # [0, FAILURE_PENALTY]

    def test_one_trial_hard_sdf(self) -> None:
        interior, wall, normals, cfd_wss, anchor, x_data, u_obs, wall_m = _minimal_bhpo_data()
        obj = BHPOObjective(
            interior_pts_nondim=interior, wall_pts_nondim=wall,
            wall_normals=normals, cfd_wss_pa=cfd_wss,
            anchor_pt_nondim=anchor, x_data=x_data,
            u_obs_nondim=u_obs, wall_pts_m=wall_m, val_frac=0.2,
            n_adam_trial=5, n_lbfgs_trial=1, device="cpu",
            use_hard_sdf=True,
        )
        hp = dict(_MINIMAL_HP, lambda_bc=0.0)
        nrmse = obj._run_trial(hp)
        assert isinstance(nrmse, float)
        assert 0.0 <= nrmse <= 2.0 + 1e-6

    def test_one_trial_adaptive_weights(self) -> None:
        interior, wall, normals, cfd_wss, anchor, x_data, u_obs, _ = _minimal_bhpo_data()
        obj = BHPOObjective(
            interior_pts_nondim=interior, wall_pts_nondim=wall,
            wall_normals=normals, cfd_wss_pa=cfd_wss,
            anchor_pt_nondim=anchor, x_data=x_data,
            u_obs_nondim=u_obs, val_frac=0.2,
            n_adam_trial=5, n_lbfgs_trial=1, device="cpu",
            use_adaptive_weights=True,
        )
        nrmse = obj._run_trial(_MINIMAL_HP)
        assert isinstance(nrmse, float)
        assert 0.0 <= nrmse <= 2.0 + 1e-6

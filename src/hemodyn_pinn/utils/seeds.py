"""Centralised RNG seed management.

All random number generation in this project must go through this module.
Hard-coded seeds are forbidden in application code; reference the named
constants here instead so that seeds can be changed in one place.
"""

from __future__ import annotations

import numpy as np

# Fixed seed used by AnxploreMesh.mesh_stats() when sampling edges/tets.
# Changing this value will alter reported statistics but not correctness.
MESH_STATS_SEED: int = 42

# Seed for Gaussian noise in the synthetic MRI forward operator.
MRI_NOISE_SEED: int = 1701


def make_numpy_rng(seed: int) -> np.random.Generator:
    """Return a fresh NumPy ``default_rng`` seeded with *seed*."""
    return np.random.default_rng(seed)


# Seed for Random Fourier Features bandwidth matrix initialisation.
RFF_SEED: int = 2718

# Seed for PINN collocation point sampling each epoch.
PINN_COLLOC_SEED: int = 3141

# Seed for PyTorch network weight initialisation.
PINN_TRAIN_SEED: int = 9999

# Seed for the scikit-optimize GP surrogate random state.
BHPO_SEED: int = 7777

# Seed for the train/validation wall-face split in BHPOObjective.
BHPO_VAL_SEED: int = 4321

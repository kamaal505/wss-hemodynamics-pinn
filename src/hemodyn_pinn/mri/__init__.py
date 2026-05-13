"""Synthetic 4D-flow MRI forward operator.

Transforms a CFD nodal velocity field into synthetic MRI observations by
applying spatial voxel averaging and additive Gaussian noise.
"""

from hemodyn_pinn.mri.operator import MRIConfig, MRIObservation, SyntheticMRIOperator
from hemodyn_pinn.mri.voxelise import VoxelGrid, assign_nodes_to_voxels
from hemodyn_pinn.mri.noise import add_gaussian_noise, sigma_from_vnr

__all__ = [
    "MRIConfig",
    "MRIObservation",
    "SyntheticMRIOperator",
    "VoxelGrid",
    "assign_nodes_to_voxels",
    "add_gaussian_noise",
    "sigma_from_vnr",
]

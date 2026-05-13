"""Synthetic 4D-flow MRI forward operator.

Implements the forward model:

    u_MRI(x_v) = (1/|V_v|) * integral_{V_v} chi_Omega(x) u_CFD(x) dV  +  eta
    eta ~ N(0, sigma_v^2 I)

The integral is approximated by averaging CFD nodal values that fall inside
each voxel.  Since the AnXplore mesh is very fine relative to MRI voxels
(~0.157 mm mean edge vs 0.5--2.5 mm voxels), each non-empty voxel contains
O(10^2)--O(10^3) nodes, making the Monte-Carlo estimate accurate.

All coordinates are in millimetres; velocity arrays are in m/s.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Optional

import numpy as np

from hemodyn_pinn.mri.noise import add_gaussian_noise
from hemodyn_pinn.mri.voxelise import VoxelGrid, assign_nodes_to_voxels
from hemodyn_pinn.utils.seeds import MRI_NOISE_SEED, make_numpy_rng


@dataclasses.dataclass
class MRIConfig:
    """Configuration for the synthetic MRI forward operator.

    Parameters
    ----------
    voxel_size_mm:
        Isotropic voxel edge length in mm.  Sweep: {0.5, 1.0, 1.5, 2.0, 2.5}.
    sigma:
        Noise standard deviation in m/s.  0.0 = noiseless.
        Compute from VENC and VNR via :func:`~hemodyn_pinn.mri.noise.sigma_from_vnr`.
    rng_seed:
        Random seed for noise generation (default: centralised constant).
    min_nodes_per_voxel:
        Voxels with fewer nodes are treated as outside the domain and dropped.
        Default 1 keeps all non-empty voxels.
    """

    voxel_size_mm: float = 1.0
    sigma: float = 0.0
    rng_seed: int = MRI_NOISE_SEED
    min_nodes_per_voxel: int = 1


@dataclasses.dataclass
class MRIObservation:
    """Output of the synthetic MRI forward operator.

    Attributes
    ----------
    voxel_centers_mm:
        (V, 3) coordinates of non-empty voxel centres in mm.
    velocity_ms:
        (V, 3) voxel-averaged (and possibly noisy) velocity in m/s.
    n_nodes_per_voxel:
        (V,) number of mesh nodes that contributed to each voxel.
    grid:
        The :class:`~hemodyn_pinn.mri.voxelise.VoxelGrid` used.
    cfg:
        The :class:`MRIConfig` that generated this observation.
    """

    voxel_centers_mm: np.ndarray     # (V, 3)
    velocity_ms: np.ndarray          # (V, 3)
    n_nodes_per_voxel: np.ndarray    # (V,) int32
    grid: VoxelGrid
    cfg: MRIConfig

    def save(self, path: pathlib.Path | str, overwrite: bool = False) -> pathlib.Path:
        """Save to a compressed .npz file.

        Parameters
        ----------
        path:
            Output file path (`.npz` extension added if absent).
        overwrite:
            If False, raise FileExistsError when the file already exists.
        """
        path = pathlib.Path(path)
        if not path.suffix:
            path = path.with_suffix(".npz")
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"{path} already exists. Pass overwrite=True to replace."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            voxel_centers_mm=self.voxel_centers_mm,
            velocity_ms=self.velocity_ms,
            n_nodes_per_voxel=self.n_nodes_per_voxel,
            voxel_size_mm=np.array(self.cfg.voxel_size_mm),
            sigma=np.array(self.cfg.sigma),
            grid_origin=self.grid.origin,
            grid_shape=np.array(self.grid.shape, dtype=np.int32),
        )
        return path

    @classmethod
    def load(cls, path: pathlib.Path | str) -> "MRIObservation":
        """Load a previously saved MRIObservation from an .npz file."""
        data = np.load(str(path))
        voxel_size_mm = float(data["voxel_size_mm"])
        cfg = MRIConfig(
            voxel_size_mm=voxel_size_mm,
            sigma=float(data["sigma"]),
        )
        grid = VoxelGrid(
            origin=data["grid_origin"],
            voxel_size_mm=voxel_size_mm,
            shape=tuple(data["grid_shape"].tolist()),
        )
        return cls(
            voxel_centers_mm=data["voxel_centers_mm"],
            velocity_ms=data["velocity_ms"],
            n_nodes_per_voxel=data["n_nodes_per_voxel"],
            grid=grid,
            cfg=cfg,
        )


class SyntheticMRIOperator:
    """Apply the synthetic 4D-flow MRI forward operator to CFD nodal data.

    Parameters
    ----------
    cfg:
        Operator configuration.
    """

    def __init__(self, cfg: Optional[MRIConfig] = None) -> None:
        self.cfg = cfg if cfg is not None else MRIConfig()

    def apply(
        self,
        points_mm: np.ndarray,
        velocity_ms: np.ndarray,
    ) -> MRIObservation:
        """Apply voxel averaging and noise to CFD nodal data.

        Parameters
        ----------
        points_mm:
            (N, 3) mesh node coordinates in mm.
        velocity_ms:
            (N, 3) CFD nodal velocity in m/s.

        Returns
        -------
        MRIObservation
            Voxel-averaged, possibly noisy observations.

        Raises
        ------
        ValueError
            If no nodes fall inside any voxel (geometry/units mismatch).
        """
        if points_mm.shape[0] != velocity_ms.shape[0]:
            raise ValueError(
                f"points_mm has {points_mm.shape[0]} rows but "
                f"velocity_ms has {velocity_ms.shape[0]} rows."
            )

        cfg = self.cfg
        grid = VoxelGrid.from_bbox(points_mm, cfg.voxel_size_mm, padding_mm=0.0)

        flat_indices = assign_nodes_to_voxels(points_mm, grid)

        # Accumulate velocity sums and node counts per voxel
        n_voxels = grid.n_voxels
        vel_sum = np.zeros((n_voxels, 3), dtype=np.float64)
        counts = np.zeros(n_voxels, dtype=np.int32)

        valid = flat_indices >= 0
        valid_idx = flat_indices[valid]
        valid_vel = velocity_ms[valid].astype(np.float64)

        # Use np.add.at for correct accumulation (handles repeated indices)
        np.add.at(vel_sum, valid_idx, valid_vel)
        np.add.at(counts, valid_idx, 1)

        # Select non-empty voxels
        occupied = counts >= cfg.min_nodes_per_voxel
        n_occ = int(occupied.sum())
        if n_occ == 0:
            raise ValueError(
                "No voxels contain mesh nodes. Check that points_mm coordinates "
                "are in millimetres and that voxel_size_mm is appropriate."
            )

        # Mean velocity per occupied voxel
        vel_mean = vel_sum[occupied] / counts[occupied, np.newaxis]

        # Compute voxel centres for occupied voxels
        all_centers_flat = grid.centers().reshape(-1, 3)  # (n_voxels, 3)
        occ_centers = all_centers_flat[occupied]

        # Add noise
        rng = make_numpy_rng(cfg.rng_seed)
        vel_obs = add_gaussian_noise(vel_mean, cfg.sigma, rng)

        return MRIObservation(
            voxel_centers_mm=occ_centers.astype(np.float64),
            velocity_ms=vel_obs.astype(np.float64),
            n_nodes_per_voxel=counts[occupied],
            grid=grid,
            cfg=cfg,
        )

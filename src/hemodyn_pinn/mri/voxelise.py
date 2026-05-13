"""Cartesian voxel grid definition and mesh-node-to-voxel assignment.

All coordinates are in millimetres throughout this module.
"""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass
class VoxelGrid:
    """Axis-aligned Cartesian voxel grid in millimetre coordinates.

    Attributes
    ----------
    origin:
        (3,) lower-left-back corner of the grid in mm.
    voxel_size_mm:
        Isotropic voxel edge length in mm.
    shape:
        (nx, ny, nz) number of voxels along each axis.
    """

    origin: np.ndarray        # (3,) float64, mm
    voxel_size_mm: float
    shape: tuple[int, int, int]

    @classmethod
    def from_bbox(
        cls,
        points_mm: np.ndarray,
        voxel_size_mm: float,
        padding_mm: float = 0.0,
    ) -> "VoxelGrid":
        """Build a grid that covers the bounding box of *points_mm*.

        Parameters
        ----------
        points_mm:
            (N, 3) array of coordinates in mm.
        voxel_size_mm:
            Isotropic voxel edge length in mm.
        padding_mm:
            Extra margin added on all six sides before snapping to grid.
        """
        lo = points_mm.min(axis=0) - padding_mm
        hi = points_mm.max(axis=0) + padding_mm
        origin = np.floor(lo / voxel_size_mm) * voxel_size_mm
        upper = np.ceil(hi / voxel_size_mm) * voxel_size_mm
        extent = upper - origin
        nx, ny, nz = np.ceil(extent / voxel_size_mm).astype(int).tolist()
        return cls(origin=origin, voxel_size_mm=voxel_size_mm, shape=(nx, ny, nz))

    @property
    def n_voxels(self) -> int:
        """Total number of voxels."""
        nx, ny, nz = self.shape
        return nx * ny * nz

    def centers(self) -> np.ndarray:
        """Return (nx, ny, nz, 3) array of voxel centre coordinates in mm."""
        nx, ny, nz = self.shape
        h = self.voxel_size_mm
        xs = self.origin[0] + (np.arange(nx) + 0.5) * h
        ys = self.origin[1] + (np.arange(ny) + 0.5) * h
        zs = self.origin[2] + (np.arange(nz) + 0.5) * h
        # meshgrid with indexing 'ij' so axis 0 = x, 1 = y, 2 = z
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        return np.stack([gx, gy, gz], axis=-1)  # (nx, ny, nz, 3)

    def flat_index(self, ix: np.ndarray, iy: np.ndarray, iz: np.ndarray) -> np.ndarray:
        """Convert (ix, iy, iz) voxel indices to flat C-order indices."""
        nx, ny, nz = self.shape
        return ix * (ny * nz) + iy * nz + iz


def assign_nodes_to_voxels(
    points_mm: np.ndarray,
    grid: VoxelGrid,
) -> np.ndarray:
    """Assign each mesh node to its containing voxel.

    Parameters
    ----------
    points_mm:
        (N, 3) mesh node coordinates in mm.
    grid:
        Target voxel grid.

    Returns
    -------
    flat_indices : (N,) int32
        Flat C-order voxel index for each node.  Nodes outside the grid
        receive index -1.
    """
    h = grid.voxel_size_mm
    nx, ny, nz = grid.shape

    # Fractional voxel coordinates
    frac = (points_mm - grid.origin) / h   # (N, 3)

    ix = np.floor(frac[:, 0]).astype(np.int32)
    iy = np.floor(frac[:, 1]).astype(np.int32)
    iz = np.floor(frac[:, 2]).astype(np.int32)

    inside = (
        (ix >= 0) & (ix < nx)
        & (iy >= 0) & (iy < ny)
        & (iz >= 0) & (iz < nz)
    )

    flat = np.full(len(points_mm), -1, dtype=np.int32)
    flat[inside] = grid.flat_index(ix[inside], iy[inside], iz[inside])
    return flat

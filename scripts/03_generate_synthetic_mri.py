"""Generate synthetic 4D-flow MRI observations from CFD solution data.

Usage (Windows .venv or Linux):

    # Default: 1.0 mm voxels, noiseless, caseA
    python scripts/03_generate_synthetic_mri.py geometry=caseA

    # 0.5 mm voxels with noise (sigma from VENC=4e-5 m/s, VNR=10)
    python scripts/03_generate_synthetic_mri.py \\
        geometry=caseC mri=voxel_0p5mm mri.sigma=4e-6

    # Sweep all voxel sizes for one case
    python scripts/03_generate_synthetic_mri.py --multirun \\
        geometry=caseC \\
        mri=voxel_0p5mm,voxel_1p0mm,voxel_1p5mm,voxel_2p0mm,voxel_2p5mm

    # All cases x all voxel sizes
    python scripts/03_generate_synthetic_mri.py --multirun \\
        geometry=caseA,caseB,caseC,caseR \\
        mri=voxel_0p5mm,voxel_1p0mm,voxel_1p5mm,voxel_2p0mm,voxel_2p5mm

Requires:
    CFD solution in data/cfd/<case_id>/solution.npz (produced by 02_run_cfd.py).
    No FEniCSx required — runs on Windows.

Output:
    data/synthetic_mri/<case_id>/voxel_<X>mm/mri_obs.npz
"""

from __future__ import annotations

import logging
import pathlib
import sys
import time

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root / "src"))


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    from hemodyn_pinn.cfd.postprocess import load_solution
    from hemodyn_pinn.mri.noise import sigma_from_vnr
    from hemodyn_pinn.mri.operator import MRIConfig, SyntheticMRIOperator

    # ── Load CFD solution ───────────────────────────────────────────────────
    cfd_dir = _root / cfg.output.base_dir / cfg.geometry.case_id
    npz_path = cfd_dir / "solution.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"CFD solution not found: {npz_path}\n"
            f"Run scripts/02_run_cfd.py geometry={cfg.geometry.case_id} first."
        )

    log.info("Loading CFD solution: %s", npz_path)
    t0 = time.perf_counter()
    sol = load_solution(npz_path)

    # solution.npz stores nodal coordinates via 'points_m' (actually wall_centroids_m,
    # legacy key) but for MRI we need the full mesh node positions in mm.
    # The CFD postprocess saves velocity at mesh nodes (N,3) alongside wall arrays.
    # We load geometry from the original VTK to get all node coordinates.
    # Fallback: use the anxplore_loader to get points_mm.
    from hemodyn_pinn.geometry.anxplore_loader import load_anxplore

    vtk_path = _root / cfg.geometry.vtk_path
    log.info("Loading mesh for node coordinates: %s", vtk_path)
    mesh = load_anxplore(vtk_path, cfg.geometry.case_id)
    points_mm = mesh.points_mm                    # (N, 3) mm
    velocity_ms = sol["velocity"]                 # (N, 3) m/s

    load_time = time.perf_counter() - t0
    bbox_vol = float(
        (points_mm.max(axis=0) - points_mm.min(axis=0)).prod()
    )
    approx_voxels = int(bbox_vol / cfg.mri.voxel_size_mm ** 3)
    log.info(
        "  N_nodes=%d  ~%d voxels in bbox  (%.1f s)",
        points_mm.shape[0],
        approx_voxels,
        load_time,
    )

    # ── Build operator ──────────────────────────────────────────────────────
    mri_cfg = MRIConfig(
        voxel_size_mm=float(cfg.mri.voxel_size_mm),
        sigma=float(cfg.mri.sigma),
        rng_seed=int(cfg.mri.rng_seed),
        min_nodes_per_voxel=int(cfg.mri.min_nodes_per_voxel),
    )
    log.info(
        "MRI operator: voxel=%.1f mm  sigma=%.2e m/s  seed=%d",
        mri_cfg.voxel_size_mm, mri_cfg.sigma, mri_cfg.rng_seed,
    )

    operator = SyntheticMRIOperator(mri_cfg)

    # ── Apply ───────────────────────────────────────────────────────────────
    t1 = time.perf_counter()
    obs = operator.apply(points_mm, velocity_ms)
    apply_time = time.perf_counter() - t1

    log.info(
        "  non-empty voxels=%d  nodes/voxel mean=%.1f  (%.1f s)",
        obs.voxel_centers_mm.shape[0],
        obs.n_nodes_per_voxel.mean(),
        apply_time,
    )
    log.info(
        "  |u| range [%.3e, %.3e] m/s",
        float(abs(obs.velocity_ms).min()),
        float(abs(obs.velocity_ms).max()),
    )

    # ── Save ────────────────────────────────────────────────────────────────
    # e.g. voxel_size_mm=1.0 -> "voxel_1p0mm"
    voxel_tag = f"voxel_{cfg.mri.voxel_size_mm:.1f}mm".replace(".", "p")
    out_dir = _root / cfg.output.mri_base_dir / cfg.geometry.case_id / voxel_tag
    out_path = obs.save(out_dir / "mri_obs", overwrite=True)
    log.info("Saved MRI observation to %s", out_path)


if __name__ == "__main__":
    main()

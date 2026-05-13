"""Run steady Stokes CFD on one or more AnXplore cases.

Usage (Linux / WSL with FEniCSx installed):

    # Single case with defaults:
    python scripts/02_run_cfd.py geometry=caseC

    # All four named cases:
    for case in caseA caseB caseC caseR; do
        python scripts/02_run_cfd.py geometry=$case
    done

    # Override inlet velocity:
    python scripts/02_run_cfd.py geometry=caseC cfd.u_inlet=1e-5

    # Multirun sweep:
    python scripts/02_run_cfd.py --multirun geometry=caseA,caseB,caseC,caseR

Environment:
    Must be run in a Linux / WSL environment with dolfinx installed.
    On Windows you will get an ImportError with installation instructions.

Output per case:
    data/cfd/<case_id>/solution.npz   — velocity, pressure, WSS arrays
    data/cfd/<case_id>/metadata.json  — scalar statistics and solver config
"""

from __future__ import annotations

import logging
import pathlib
import sys
import time

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

# Add project src to path so the package is importable without pip install -e.
_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root / "src"))


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    # ── Imports (deferred so Hydra loads before dolfinx) ───────────────────
    from hemodyn_pinn.geometry.anxplore_loader import load_anxplore
    from hemodyn_pinn.cfd.stokes_solver import StokesConfig, solve_stokes
    from hemodyn_pinn.cfd.postprocess import save_solution, wss_statistics

    # ── Load mesh ───────────────────────────────────────────────────────────
    vtk_path = _root / cfg.geometry.vtk_path
    if not vtk_path.exists():
        raise FileNotFoundError(f"VTK mesh not found: {vtk_path}")

    log.info("Loading mesh: %s", vtk_path)
    t0 = time.perf_counter()
    mesh = load_anxplore(vtk_path, cfg.geometry.case_id)
    log.info(
        "  nodes=%d  tets=%d  wall_tris=%d  (%.1f s)",
        mesh.n_nodes, mesh.n_tets, mesh.n_wall_tri,
        time.perf_counter() - t0,
    )

    # ── Stokes solver config ─────────────────────────────────────────────────
    solver_cfg = StokesConfig(
        mu=float(cfg.cfd.mu),
        u_inlet=float(cfg.cfd.u_inlet),
        petsc_ksp_type=cfg.cfd.petsc_ksp_type,
        petsc_pc_type=cfg.cfd.petsc_pc_type,
        petsc_pc_solver=cfg.cfd.petsc_pc_solver,
        save_vtk=bool(cfg.cfd.save_vtk),
    )
    log.info(
        "Solver: mu=%.2e Pa·s  u_inlet=%.2e m/s  Re≈%.3f",
        solver_cfg.mu, solver_cfg.u_inlet, solver_cfg.re_number if hasattr(solver_cfg, "re_number") else
        1060 * solver_cfg.u_inlet * 0.01628 / solver_cfg.mu,
    )

    # ── Solve ────────────────────────────────────────────────────────────────
    outdir = _root / cfg.output.base_dir / cfg.geometry.case_id
    log.info("Solving Stokes (MUMPS direct solver) …")
    t1 = time.perf_counter()
    # out_dir triggers XDMF+HDF5 export (doc 06 §7) alongside the .npz below.
    solution = solve_stokes(mesh, solver_cfg, out_dir=outdir)
    elapsed = time.perf_counter() - t1
    log.info("Solve complete in %.1f s  Re=%.4f", elapsed, solution.re_number)

    # ── Post-process statistics ──────────────────────────────────────────────
    stats = wss_statistics(solution)
    log.info(
        "WSS (wall only): mean=%.4f  max=%.4f  p95=%.4f  [Pa]",
        stats["mean"], stats["max"], stats["p95"],
    )

    # ── Save .npz (cross-platform; XDMF already written inside solve_stokes) ──
    npz_path = save_solution(solution, outdir, overwrite=True)
    log.info("Saved to %s", npz_path)


if __name__ == "__main__":
    main()

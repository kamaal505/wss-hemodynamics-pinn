#!/usr/bin/env bash
# =============================================================================
# fetch_and_inspect.sh
#
# Phase 1.1 — AnXplore data acquisition and preliminary mesh statistics.
#
# Compatible with: Git Bash on Windows, bash on Linux/macOS.
#
# What this does
# --------------
# 1. Clones the AnXplore dataset into data/geometries/anxplore/.
# 2. Passes the directory to a Python script that:
#      - Inventories every mesh file (extension, size, case ID)
#      - Per mesh: node count, cell counts, bounding box, edge-length
#        distribution (mean, std, min, max, percentiles, IQR outliers),
#        and any embedded field names (velocity, pressure, WSS, etc.)
#      - Cohort summary: mean+/-std, median, p5/p25/p75/p95 across cases
#    Outputs:
#      data/geometries/anxplore/inventory.tsv
#      data/geometries/anxplore/mesh_stats.txt
#      data/geometries/anxplore/mesh_stats_per_case.csv
# 3. Prints URLs for datasets that require manual download (VMR).
#
# Usage (from repo root, with .venv activated)
# --------------------------------------------
#   bash scripts/fetch_and_inspect.sh
#
#   To limit inspection to N cases for a quick sanity check:
#     MAX_CASES=5 bash scripts/fetch_and_inspect.sh
#
# Requirements
# ------------
#   git, Python >= 3.11 with meshio and numpy installed in .venv
#   Internet access to github.com
#   ~200 MB disk space for the full AnXplore clone
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve repo root
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${REPO_ROOT}/data/geometries/anxplore"
ANXPLORE_URL="https://github.com/aurelegoetz/AnXplore.git"
MAX_CASES="${MAX_CASES:-999}"

# ---------------------------------------------------------------------------
# Locate Python inside .venv — Git Bash uses Scripts/, Linux uses bin/
# ---------------------------------------------------------------------------
if [ -f "${REPO_ROOT}/.venv/Scripts/python" ]; then
    PYTHON="${REPO_ROOT}/.venv/Scripts/python"
elif [ -f "${REPO_ROOT}/.venv/bin/python" ]; then
    PYTHON="${REPO_ROOT}/.venv/bin/python"
else
    PYTHON="python"
fi

log() { echo "[$(date '+%H:%M:%S')] $*"; }
hr()  { echo "----------------------------------------------------------------------"; }

# ---------------------------------------------------------------------------
# 1. Clone / update AnXplore
# ---------------------------------------------------------------------------
hr
log "DATA SOURCE: AnXplore"
log "  URL   : ${ANXPLORE_URL}"
log "  Paper : https://doi.org/10.3389/fbioe.2024.1433811"
log "  Goetz et al. (2024), Front. Bioeng. Biotechnol."
log "  License: MIT"
hr

mkdir -p "${DATA_DIR}"

if [ -d "${DATA_DIR}/.git" ]; then
    log "Repo exists at ${DATA_DIR}. Pulling latest..."
    git -C "${DATA_DIR}" pull --ff-only
else
    log "Cloning AnXplore into ${DATA_DIR} ..."
    git clone --depth 1 "${ANXPLORE_URL}" "${DATA_DIR}"
fi
log "Clone/update complete."

# ---------------------------------------------------------------------------
# 2. VMR / secondary source reminders
# ---------------------------------------------------------------------------
hr
log "DATA SOURCE: Vascular Model Repository (VMR) -- manual download required"
log "  URL    : https://www.vascularmodel.com"
log "  Action : register, download 2-3 cases into data/geometries/vmr/"
log "  License: research use; derivative results redistributable"
hr
log "DATA SOURCE: AneuX (tertiary, geometry only)"
log "  URL    : https://zenodo.org/records/6678442"
log "DATA SOURCE: AneuriskWeb (tertiary, pre-computed CFD-WSS)"
log "  URL    : http://ecm2.mathcs.emory.edu/aneuriskweb/"
hr

# ---------------------------------------------------------------------------
# 3. Write the Python inspector to a temp file and run it
#    (avoids heredoc + variable-expansion conflicts in Git Bash)
# ---------------------------------------------------------------------------
INSPECTOR="${REPO_ROOT}/data/geometries/_inspector_tmp.py"

cat > "${INSPECTOR}" << 'PYEOF'
import sys
import os
import csv
import pathlib
import traceback
from datetime import datetime

data_dir  = pathlib.Path(sys.argv[1])
max_cases = int(sys.argv[2])

try:
    import meshio
    import numpy as np
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}")
    print("        Run: uv pip install meshio numpy")
    sys.exit(1)

np.random.seed(0)

inv_tsv   = data_dir / "inventory.tsv"
stats_txt = data_dir / "mesh_stats.txt"
csv_path  = data_dir / "mesh_stats_per_case.csv"

# ── helpers ──────────────────────────────────────────────────────────────────
def fmt_bytes(n):
    n = int(n)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n //= 1024
    return f"{n:.1f} TB"

def iqr_outliers(arr):
    if arr.size < 4:
        return 0
    q1, q3 = np.percentile(arr, [25, 75])
    fence = 1.5 * (q3 - q1)
    return int(np.sum((arr < q1 - fence) | (arr > q3 + fence)))

def pct_str(arr):
    p = np.percentile(arr, [5, 25, 50, 75, 95])
    return (f"p5={p[0]:.3g}  p25={p[1]:.3g}  median={p[2]:.3g}  "
            f"p75={p[3]:.3g}  p95={p[4]:.3g}")

# ── inventory ────────────────────────────────────────────────────────────────
MESH_EXTS = {".xdmf", ".msh", ".vtu", ".vtk", ".stl", ".h5", ".hdf5"}
SKIP_EXTS = {".h5", ".hdf5"}

all_files = []
for root, dirs, files in os.walk(data_dir):
    dirs[:] = [d for d in sorted(dirs) if not d.startswith(".")]
    for fname in sorted(files):
        ext = pathlib.Path(fname).suffix.lower()
        if ext in MESH_EXTS:
            fpath = pathlib.Path(root) / fname
            size  = fpath.stat().st_size
            rel_parts = fpath.relative_to(data_dir).parts
            case_id = rel_parts[0] if rel_parts else "unknown"
            all_files.append({
                "path":       str(fpath),
                "extension":  ext.lstrip("."),
                "size_bytes": size,
                "case_id":    case_id,
            })

print(f"  Found {len(all_files)} mesh-related file(s).")

with open(inv_tsv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["path","extension","size_bytes","case_id"],
                       delimiter="\t")
    w.writeheader()
    w.writerows(all_files)
print(f"  Inventory written -> {inv_tsv}")

# ── select primary file per case ─────────────────────────────────────────────
PRIORITY = {".xdmf": 0, ".msh": 1, ".vtu": 2, ".vtk": 3, ".stl": 4}
case_primary = {}
for row in all_files:
    ext = "." + row["extension"].lower()
    if ext in SKIP_EXTS:
        continue
    cid = row["case_id"]
    pri = PRIORITY.get(ext, 99)
    if cid not in case_primary or pri < case_primary[cid][0]:
        case_primary[cid] = (pri, row)

to_inspect = [v for _, v in sorted(case_primary.values(),
                                   key=lambda x: x[1]["case_id"])]
to_inspect  = to_inspect[:max_cases]
print(f"  Inspecting {len(to_inspect)} case(s) (MAX_CASES={max_cases})")

# ── per-case loop ────────────────────────────────────────────────────────────
lines   = []
records = []
agg     = {"n_pts":[], "n_tets":[], "n_tris":[], "bbox_vol":[],
           "edge_mean":[], "edge_std":[], "edge_min":[], "edge_max":[]}
all_fields = set()

SEP = "=" * 70

lines.append(SEP)
lines.append("  AnXplore Mesh Statistics")
lines.append(f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
lines.append(f"  Source    : {data_dir}")
lines.append(f"  Cases     : {len(to_inspect)} inspected / {len(case_primary)} total")
lines.append(SEP)

for i, row in enumerate(to_inspect):
    fpath = row["path"]
    case  = row["case_id"]
    size  = row["size_bytes"]
    rel   = os.path.relpath(fpath, data_dir)

    lines.append(f"\n[{i+1:03d}/{len(to_inspect)}]  {case}")
    lines.append(f"  File : {rel}  ({fmt_bytes(size)})")

    try:
        mesh = meshio.read(fpath)
    except Exception:
        msg = traceback.format_exc().strip().splitlines()[-1]
        lines.append(f"  [SKIP] Read error: {msg}")
        continue

    pts = mesh.points
    n_pts = len(pts)
    cell_counts = {cb.type: len(cb.data) for cb in mesh.cells}
    n_tets = cell_counts.get("tetra", 0) + cell_counts.get("tetra10", 0)
    n_tris = (cell_counts.get("triangle", 0)
              + cell_counts.get("triangle6", 0))

    if n_pts > 0:
        xyz_min = pts.min(axis=0)
        xyz_max = pts.max(axis=0)
        dims    = xyz_max - xyz_min
        bbox_vol = float(np.prod(np.where(dims > 0, dims, 1.0)))
    else:
        xyz_min = xyz_max = dims = np.zeros(3)
        bbox_vol = 0.0

    # Edge-length distribution
    edge_lens = np.array([])
    tet_data  = next((cb.data for cb in mesh.cells
                      if cb.type in ("tetra", "tetra10")), None)
    if tet_data is not None and len(tet_data) > 0:
        idx    = np.random.choice(len(tet_data),
                                  min(50_000, len(tet_data)), replace=False)
        sample = tet_data[idx]
        pairs  = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
        edge_lens = np.concatenate([
            np.linalg.norm(pts[sample[:, a]] - pts[sample[:, b]], axis=1)
            for a, b in pairs
        ])

    pd_keys = list(mesh.point_data.keys())
    cd_keys = list(mesh.cell_data.keys())
    all_fields.update(pd_keys + cd_keys)

    lines.append(f"  Points      : {n_pts:>10,}")
    lines.append(f"  Tetrahedra  : {n_tets:>10,}")
    lines.append(f"  Triangles   : {n_tris:>10,}  (surface elements)")
    lines.append(f"  Cell types  : {list(cell_counts.keys())}")
    lines.append(f"  BBox x [mm] : [{xyz_min[0]:>9.3f},  {xyz_max[0]:>9.3f}]"
                 f"  span={dims[0]:.3f}")
    lines.append(f"  BBox y [mm] : [{xyz_min[1]:>9.3f},  {xyz_max[1]:>9.3f}]"
                 f"  span={dims[1]:.3f}")
    lines.append(f"  BBox z [mm] : [{xyz_min[2]:>9.3f},  {xyz_max[2]:>9.3f}]"
                 f"  span={dims[2]:.3f}")

    if edge_lens.size > 0:
        n_out = iqr_outliers(edge_lens)
        lines.append(f"  Edge length [mm]:")
        lines.append(f"    mean +/- std : {edge_lens.mean():.4f} +/- {edge_lens.std():.4f}")
        lines.append(f"    min / max    : {edge_lens.min():.4f} / {edge_lens.max():.4f}")
        lines.append(f"    {pct_str(edge_lens)}")
        lines.append(f"    IQR outliers : {n_out:,} / {edge_lens.size:,} "
                     f"({100*n_out/max(edge_lens.size,1):.2f}%)")
        agg["edge_mean"].append(float(edge_lens.mean()))
        agg["edge_std"].append(float(edge_lens.std()))
        agg["edge_min"].append(float(edge_lens.min()))
        agg["edge_max"].append(float(edge_lens.max()))

    if pd_keys:
        lines.append(f"  Point fields: {pd_keys}")
    if cd_keys:
        lines.append(f"  Cell fields : {cd_keys}")
    if not pd_keys and not cd_keys:
        lines.append(f"  Fields      : none (geometry-only; CFD not yet run)")

    agg["n_pts"].append(n_pts)
    agg["n_tets"].append(n_tets)
    agg["n_tris"].append(n_tris)
    agg["bbox_vol"].append(bbox_vol)
    records.append({
        "case":           case,
        "n_pts":          n_pts,
        "n_tets":         n_tets,
        "n_tris":         n_tris,
        "bbox_x_span_mm": round(float(dims[0]), 4),
        "bbox_y_span_mm": round(float(dims[1]), 4),
        "bbox_z_span_mm": round(float(dims[2]), 4),
        "bbox_vol_mm3":   round(bbox_vol, 2),
        "edge_mean_mm":   round(float(edge_lens.mean()), 5) if edge_lens.size else "",
        "edge_std_mm":    round(float(edge_lens.std()),  5) if edge_lens.size else "",
        "edge_min_mm":    round(float(edge_lens.min()),  5) if edge_lens.size else "",
        "edge_max_mm":    round(float(edge_lens.max()),  5) if edge_lens.size else "",
    })

# ── cohort summary ────────────────────────────────────────────────────────────
lines.append("\n" + SEP)
lines.append("  COHORT SUMMARY")
lines.append(SEP)

def cohort_block(values, label, unit):
    if not values:
        return
    a = np.array(values, dtype=float)
    lines.append(f"\n  {label}  [{unit}]  (n={len(a)})")
    lines.append(f"    mean +/- std : {a.mean():.3g} +/- {a.std():.3g}")
    lines.append(f"    median       : {float(np.median(a)):.3g}")
    lines.append(f"    min / max    : {a.min():.3g} / {a.max():.3g}")
    lines.append(f"    {pct_str(a)}")
    n_out = iqr_outliers(a)
    lines.append(f"    IQR outliers : {n_out} / {len(a)} "
                 f"({100*n_out/max(len(a),1):.1f}%)")

cohort_block(agg["n_pts"],     "Node count",                "nodes")
cohort_block(agg["n_tets"],    "Tetrahedra count",          "cells")
cohort_block(agg["n_tris"],    "Surface triangles",         "cells")
cohort_block(agg["bbox_vol"],  "Bounding-box volume",       "mm3")
cohort_block(agg["edge_mean"], "Mean edge length per case", "mm")
cohort_block(agg["edge_max"],  "Max edge length per case",  "mm")

if all_fields:
    lines.append(f"\n  Field names encountered:")
    for fn in sorted(all_fields):
        lines.append(f"    - {fn}")
else:
    lines.append("\n  No embedded field data (geometry-only meshes).")
    lines.append("  -> Next step: scripts/02_run_cfd.py")

lines.append("\n" + SEP)
lines.append("  DATA SOURCE REFERENCE")
lines.append(SEP)
lines.append("  AnXplore")
lines.append("    URL     : https://github.com/aurelegoetz/AnXplore")
lines.append("    DOI     : https://doi.org/10.3389/fbioe.2024.1433811")
lines.append("    Cite    : Goetz et al. (2024), Front. Bioeng. Biotechnol.")
lines.append("    License : MIT")
lines.append("")
lines.append("  Vascular Model Repository (VMR)  -- manual download required")
lines.append("    URL     : https://www.vascularmodel.com")
lines.append("    License : research use; derivative results redistributable")
lines.append("")
lines.append("  AneuX morphology database  (tertiary, geometry only)")
lines.append("    URL     : https://zenodo.org/records/6678442")
lines.append("")
lines.append("  AneuriskWeb  (tertiary, pre-computed CFD-WSS)")
lines.append("    URL     : http://ecm2.mathcs.emory.edu/aneuriskweb/")
lines.append(SEP)

report = "\n".join(lines)
print(report)
stats_txt.write_text(report, encoding="utf-8")
print(f"\n  Report written  -> {stats_txt}")

if records:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)
    print(f"  Per-case CSV    -> {csv_path}")

PYEOF

log "Running inspector..."
"${PYTHON}" "${INSPECTOR}" "${DATA_DIR}" "${MAX_CASES}"

# Clean up temp file
rm -f "${INSPECTOR}"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
hr
log "Complete. Outputs in data/geometries/anxplore/:"
log "  inventory.tsv           -- one row per mesh file"
log "  mesh_stats.txt          -- full human-readable report"
log "  mesh_stats_per_case.csv -- one row per case (load with pandas)"
hr
log "Next: review mesh_stats.txt, then run scripts/02_run_cfd.py"
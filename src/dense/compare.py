# ╔══════════════════════════════════════════════════════════════════════════════════════╗
# ║  ⠀⠀⠀⠀⣠⠶⡒⠒⢬⡲⣮⠂⣆⣀⠀⠀⠀⠀⠀⠀⢀⣤⣴⣦⣤⡀⠀⠀⠀⠀   MATEUS OLIVEIRA                        ║
# ║  ⠀⠀⠀⣀⣥⠠⣿⠆⠐⣻⣾⣿⣿⢷⡄⠀⠀⠀⠀⢠⡿⠋⠉⠉⠙⢿⡄⠀⠀⠀   m203656@dac.unicamp.edu.br             ║
# ║  ⠀⠀⢘⡵⢋⠄⡙⠒⣤⣄⣉⠙⣿⣗⠑⡄⠀⠀⠀⠘⡇⠀⠀⠀⠀⠈⡇⠀⠀⠀   UNICAMP - Universidade Estadual de     ║
# ║  ⠀⣴⢿⡜⢡⡞⢀⢼⣿⣿⣿⣿⣿⣿⠟⣂⠀⠀⢀⣀⠱⡀⠀⠀⠀⢰⠁⠀⠀⠀               Campinas                     ║
# ║  ⠰⢫⢟⡇⢸⡇⢸⢾⣿⣿⣿⣿⣿⣿⡷⠰⠀⢰⡏⠀⠀⢡⠀⠀⢠⠃⠀⠀⠀⠀   IC - Institute of Computing            ║
# ║  ⢰⠁⣿⢣⣿⠇⢀⣿⣿⡿⠿⠤⣭⣥⣶⡆⠀⠸⣷⣤⣠⡾⠀⢀⡇⠀⠀⠀⠀⠀   Computer Science Department              ║
# ║  ⡞⣰⣧⠟⡝⢸⢸⣿⣥⠖⣴⡆⣤⣬⠉⠀⠀⠀⠈⠉⠉⠀⠀⢸⣇⠀⠀⠀⠀⠀   github.com/oliveiraMats2              ║
# ║  ⠀⡿⡟⢸⡇⠸⡄⢹⣿⢸⣿⣇⡏⠟⣰⣄⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠁⠀⠀⠀   linkedin.com/in/mateus-eng            ║
# ║  ⠀⠇⣧⠘⡇⠦⣹⣸⣿⡇⡿⡿⣡⣼⣿⣿⣷⣦⣄⡀⠀⠀⣸⣿⣿⠄⠻⢷⣦⠀                                            ║
# ║  ⠀⢀⠘⣇⢹⡸⣿⣿⣿⢹⢃⣠⣿⣿⣿⣿⣿⣿⣿⣿⣆⠀⠑⠋⠉⠀⠀⠈⣿⣧   UNICAMP · IC · 2026                    ║
# ║  ⠀⢸⣿⡌⠘⢷⣿⣿⡏⢀⣾⣿⣿⣿⣿⣿⣿⢻⣿⣿⣿⡆⠀⠀⠀⠀⠀⠀⣿⡿                                            ║
# ║  ⠀⠈⣿⣿⣦⡌⢿⠏⣰⣿⣿⣿⣿⣿⣿⡿⡏⣼⣿⣿⣿⡇⣄⠀⠀⠀⢀⣼⣿⠇                                            ║
# ║  ⠀⠀⠹⣿⣿⢻⡀⣼⣿⣿⢻⣿⣿⣿⣿⡇⡇⢻⣿⣿⣿⡇⣿⣿⣶⣿⣿⠟⠁⠀                                            ║
# ║  ⠀⠀⠀⢻⣿⣦⡓⢿⣿⣿⡆⣿⣿⣿⣿⢃⣶⡸⣿⣿⣿⡇⠀⠉⠉⠁⠀⠀⠀⠀                                            ║
# ║  ⠀⠀⠀⠈⣿⣿⣿⡆⠀⠀⠀⣿⣿⣿⡟⣼⡿⠁⢹⣿⣿⣷⠀⠀⠀⠀⠀⠀⠀⠀                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════════════╝

# PROMPT COMPLEMENT
# Stage: dense_compare (after dense + cleanup of every dense source + planes + floorplan).
# Goal: compare the dense clouds (patch_match, openmvs) against each other and the sparse reference.
# In:  <run_dir>/clean/{sparse,<method>}/points.npy (working units), <run_dir>/planes/T_floor.npy,
#      <artifacts_dir>/floorplan.csv (area_m2), <artifacts_dir>/dense_metrics.csv (method, t_s).
# Out: <artifacts_dir>/dense_compare.csv, key `method`, columns
#      method, n_dense, density_pts_m2, completeness_frac, chamfer_to_sparse_cm, t_s (one row per method + `sparse`).
# YAML: dense_compare.enabled; dense.methods; floorplan.cell_cm (cell size reused); scale.provisional_extent_cm.
# Rules: reads only. completeness = fraction of floor XY cells (occupied by the sparse cloud after T_floor)
#        hit by >= 1 dense point. chamfer = mean of the two directed NN distances (open3d, no scipy), in cm;
#        empty when unit=model. Missing method -> row with n_dense=0 + WARNING. Missing area -> density empty.

# >>> BEGIN IMPLEMENTATION >>>
"""Dense cloud comparison: count, density, floor completeness and chamfer distance to the sparse cloud."""
import logging
from pathlib import Path

import numpy as np
import open3d as o3d

from src import stage_io

log = logging.getLogger(__name__)

COLUMNS = ["method", "n_dense", "density_pts_m2", "completeness_frac", "chamfer_to_sparse_cm", "t_s"]

def to_floor(points: np.ndarray, T: np.ndarray) -> np.ndarray:

    return points @ T[:3, :3].T + T[:3, 3]

def cell_ids(xy: np.ndarray, origin: np.ndarray, cell: float) -> set[tuple[int, int]]:

    ij = np.floor((xy - origin) / cell).astype(np.int64)

    return set(map(tuple, np.unique(ij, axis=0)))

def chamfer(a: np.ndarray, b: np.ndarray) -> float:
    """Mean of the two directed nearest-neighbour mean distances (same units as the inputs)."""

    pa, pb = (o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p.astype(np.float64))) for p in (a, b))
    d_ab = np.asarray(pa.compute_point_cloud_distance(pb))
    d_ba = np.asarray(pb.compute_point_cloud_distance(pa))

    return float(0.5 * (d_ab.mean() + d_ba.mean()))

def first_value(path: Path, column: str, key: str | None = None, value: str | None = None) -> str:

    for r in stage_io.read_csv(path):
        if key is None or r.get(key) == value:
            return r.get(column, "") or ""

    return ""

def run(cfg: dict, run_dir: str, artifacts_dir: str) -> Path | None:

    if not cfg["dense_compare"]["enabled"]:
        log.info("dense_compare disabled")
        return None

    run_dir, artifacts_dir = Path(run_dir), Path(artifacts_dir)
    sparse_path = run_dir / "clean" / "sparse" / "points.npy"
    if not sparse_path.exists():
        log.warning("dense_compare: %s missing, nothing to compare", sparse_path)
        return None

    sparse = np.load(sparse_path).reshape(-1, 3)
    t_path = run_dir / "planes" / "T_floor.npy"
    if t_path.exists():
        T = np.load(t_path)
    else:
        log.warning("dense_compare: %s missing, using identity (floor footprint may be wrong)", t_path)
        T = np.eye(4)

    _, unit = stage_io.scale_info(run_dir)
    upc = stage_io.units_per_cm(run_dir, sparse, room_extent_cm=cfg["scale"]["provisional_extent_cm"])
    cell = cfg["floorplan"]["cell_cm"] * upc

    area = first_value(artifacts_dir / "floorplan.csv", "area_m2")
    area_m2 = float(area) if area not in ("", "nan") else float("nan")
    if not np.isfinite(area_m2) or area_m2 <= 0:
        log.warning("dense_compare: area_m2 unavailable in floorplan.csv, density_pts_m2 left empty")
        area_m2 = float("nan")

    sparse_xy = to_floor(sparse, T)[:, :2]
    origin = sparse_xy.min(axis=0)
    floor_cells = cell_ids(sparse_xy, origin, cell)
    csv_path = artifacts_dir / "dense_compare.csv"

    def write(method: str, points: np.ndarray | None, t_s: str) -> dict:

        n = 0 if points is None else len(points)
        row = dict.fromkeys(COLUMNS, "")
        row.update(method=method, n_dense=n, t_s=t_s)
        if np.isfinite(area_m2):
            row["density_pts_m2"] = n / area_m2
        if points is None or n == 0:
            row["completeness_frac"] = 0.0
        else:
            hit = cell_ids(to_floor(points, T)[:, :2], origin, cell) & floor_cells
            row["completeness_frac"] = len(hit) / max(len(floor_cells), 1)
            if unit == "cm":
                row["chamfer_to_sparse_cm"] = chamfer(points, sparse) / upc
        stage_io.upsert_csv(csv_path, row, key="method")
        stage_io.log_metrics(log, {f"{method}.{k}": v for k, v in row.items() if k != "method"})

        return row

    if unit != "cm":
        log.warning("dense_compare: scale unknown, chamfer_to_sparse_cm left empty")

    write("sparse", sparse, first_value(artifacts_dir / "sfm_metrics.csv", "t_s"))
    for method in cfg["dense"]["methods"]:
        path = run_dir / "clean" / method / "points.npy"
        if not path.exists():
            log.warning("dense_compare: %s missing, row with n_dense=0", path)
            points = None
        else:
            points = np.load(path).reshape(-1, 3)
        write(method, points, first_value(artifacts_dir / "dense_metrics.csv", "t_s", "method", method))

    return csv_path

if __name__ == "__main__":
    import tempfile
    logging.basicConfig(level=logging.INFO)
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        run_dir, art = Path(tmp) / "run", Path(tmp) / "art"
        (run_dir / "scale").mkdir(parents=True)
        np.save(run_dir / "scale" / "scale.npy", np.array([1.0]))
        floor = np.c_[rng.uniform(0, 400, (20000, 2)), np.zeros(20000)]  # 4 m x 4 m floor in cm
        (run_dir / "clean" / "sparse").mkdir(parents=True)
        np.save(run_dir / "clean" / "sparse" / "points.npy", floor)
        half = floor[floor[:, 0] < 200] + rng.normal(0, 0.5, (int((floor[:, 0] < 200).sum()), 3))
        (run_dir / "clean" / "patch_match").mkdir(parents=True)
        np.save(run_dir / "clean" / "patch_match" / "points.npy", half)
        (run_dir / "clean" / "full").mkdir(parents=True)
        np.save(run_dir / "clean" / "full" / "points.npy", floor + rng.normal(0, 0.5, floor.shape))
        (run_dir / "planes").mkdir()
        np.save(run_dir / "planes" / "T_floor.npy", np.eye(4))
        stage_io.upsert_csv(art / "floorplan.csv", {"area_m2": 16.0})
        stage_io.upsert_csv(art / "dense_metrics.csv", {"method": "patch_match", "t_s": 12.5}, "method")
        cfg = {"dense_compare": {"enabled": True}, "dense": {"methods": ["patch_match", "openmvs", "full"]},
               "floorplan": {"cell_cm": 10.0}, "scale": {"provisional_extent_cm": 500}}
        out = run(cfg, str(run_dir), str(art))
        rows = {r["method"]: r for r in stage_io.read_csv(out)}
        assert list(rows["sparse"]) == COLUMNS
        pm, om = rows["patch_match"], rows["openmvs"]
        assert abs(float(pm["completeness_frac"]) - 0.5) < 0.03, pm
        assert float(pm["chamfer_to_sparse_cm"]) < 60, pm  # half the sparse has no dense neighbour nearby
        assert float(rows["full"]["chamfer_to_sparse_cm"]) < 2 and float(rows["full"]["completeness_frac"]) > 0.99
        assert pm["t_s"] == "12.5" and abs(float(pm["density_pts_m2"]) - len(half) / 16) < 1e-6
        assert om["n_dense"] == "0" and float(om["completeness_frac"]) == 0.0 and om["chamfer_to_sparse_cm"] == ""
        assert float(rows["sparse"]["completeness_frac"]) == 1.0 and float(rows["sparse"]["chamfer_to_sparse_cm"]) == 0.0
        (run_dir / "scale" / "scale.npy").unlink()
        rows = {r["method"]: r for r in stage_io.read_csv(run(cfg, str(run_dir), str(art)))}
        assert rows["patch_match"]["chamfer_to_sparse_cm"] == ""
    print("OK")
# <<< END IMPLEMENTATION <<<

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
# Stage: dense_compare (after dense + cleanup of every dense source + planes + floorplan + room_fill).
# Goal: THE comparison table of the densifiers (X3): every dense method against the sparse reference.
# In:  <run_dir>/clean/{sparse,<method>}/points.npy (working units), <run_dir>/dense/<method>/points.npy (raw,
#      model units; every method present: patch_match, plane_sweep, sgm, casmvsnet, mvsnet, patchmatchnet,
#      openmvs, vggt_omega, mapanything), <run_dir>/planes/T_floor.npy, <run_dir>/scale/scale.npy,
#      <artifacts_dir>/floorplan.csv (area_m2), <artifacts_dir>/room_fill.csv (polygon_area_m2, floor_area_m2,
#      cm_per_unit), <artifacts_dir>/dense_metrics.csv (method, t_s, vram_peak_gb).
# Out: <artifacts_dir>/dense_compare.csv, key `method`, columns
#      method, n_dense, density_pts_m2, completeness_frac, chamfer_to_sparse_cm, t_s (one row per method + `sparse`)
#      + n_dense_raw, n_dense_clean, density_raw_pts_m2, area_m2, area_source, points_from, cm_per_unit,
#      scale_source, vram_peak_gb, nn_device.
# YAML: dense_compare.{enabled, device: auto|CPU:0|CUDA:0}; dense.methods; floorplan.cell_cm (cell size reused);
#       scale.provisional_extent_cm.
# Rules: reads only. Methods = dense.methods + every dense/*/points.npy + clean/*/points.npy. Points = clean cloud,
#        raw dense (scaled to working units) when the method was not cleaned. n_dense = points used.
#        Area: floorplan area_m2 > room_fill polygon_area_m2 > floor_area_m2 (reprojection.floor_area).
#        completeness = fraction of floor XY cells (occupied by the sparse cloud after T_floor) hit by >= 1 dense
#        point. chamfer = mean of the two directed NN distances (open3d.core.nns on CUDA, CPU fallback), in cm:
#        scale.npy when unit=cm, else room_fill cm_per_unit (scale_source says which), empty when neither.
#        Missing method -> row with n_dense=0 + WARNING. Missing area -> density empty.

# >>> BEGIN IMPLEMENTATION >>>
"""Dense cloud comparison: count, density, floor completeness, chamfer to the sparse cloud, time and VRAM."""
import logging
from pathlib import Path

import numpy as np
import open3d as o3d

from src import stage_io
from src.evaluation.reprojection import floor_area

log = logging.getLogger(__name__)

COLUMNS = ["method", "n_dense", "density_pts_m2", "completeness_frac", "chamfer_to_sparse_cm", "t_s"]
EXTRA = ["n_dense_raw", "n_dense_clean", "density_raw_pts_m2", "area_m2", "area_source", "points_from",
         "cm_per_unit", "scale_source", "vram_peak_gb", "nn_device"]
DEFAULTS = {"device": "auto"}
CHUNK = 1_000_000  # query points per CUDA knn call

def to_floor(points: np.ndarray, T: np.ndarray) -> np.ndarray:

    return points @ T[:3, :3].T + T[:3, 3]

def cell_ids(xy: np.ndarray, origin: np.ndarray, cell: float) -> set[tuple[int, int]]:

    ij = np.floor((xy - origin) / cell).astype(np.int64)

    return set(map(tuple, np.unique(ij, axis=0)))

def nn_distances(query: np.ndarray, target: np.ndarray, device: str = "auto") -> tuple[np.ndarray, str]:
    """Distance from every query point to its nearest target point and the device used.
    open3d.core.nns knn on CUDA (float32, centred on the target mean); CPU KD-tree fallback."""

    if device != "CPU:0" and o3d.core.cuda.is_available():
        try:
            dev = o3d.core.Device("CUDA:0" if device == "auto" else device)
            c = target.mean(axis=0)
            nns = o3d.core.nns.NearestNeighborSearch(o3d.core.Tensor((target - c).astype(np.float32), device=dev))
            nns.knn_index()
            out = []
            for i in range(0, len(query), CHUNK):
                q = o3d.core.Tensor((query[i:i + CHUNK] - c).astype(np.float32), device=dev)
                out.append(nns.knn_search(q, 1)[1].cpu().numpy().ravel())
            return np.sqrt(np.concatenate(out).astype(np.float64)), str(dev)
        except Exception as e:  # out of memory, unsupported build: fall back to the CPU
            log.warning("dense_compare: CUDA nearest neighbours failed (%s), using CPU", e)

    pq, pt = (o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p.astype(np.float64))) for p in (query, target))

    return np.asarray(pq.compute_point_cloud_distance(pt)), "CPU:0"

def chamfer(a: np.ndarray, b: np.ndarray, device: str = "auto") -> tuple[float, str]:
    """(mean of the two directed nearest-neighbour mean distances, device); same units as the inputs."""

    d_ab, dev = nn_distances(a, b, device)
    d_ba, _ = nn_distances(b, a, device)

    return float(0.5 * (d_ab.mean() + d_ba.mean())), dev

def first_value(path: Path, column: str, key: str | None = None, value: str | None = None) -> str:

    for r in stage_io.read_csv(path):
        if key is None or r.get(key) == value:
            return r.get(column, "") or ""

    return ""

def cm_scale(run_dir: Path, artifacts_dir: Path) -> tuple[float, float, str]:
    """(cm per model unit, cm per working unit, source): scale.npy (working units are cm) > room_fill
    cm_per_unit (working units are model units) > (nan, nan, "")."""

    s, unit = stage_io.scale_info(run_dir)
    if unit == "cm":
        return s, 1.0, "scale.npy"

    for r in stage_io.read_csv(artifacts_dir / "room_fill.csv"):
        for col in ("cm_per_unit", "cm_per_model_unit"):
            try:
                v = float(r.get(col) or "nan")
            except ValueError:
                continue
            if np.isfinite(v) and v > 0:
                return v, v, "room_fill"

    return float("nan"), float("nan"), ""

def count(path: Path) -> int:

    return len(np.load(path, mmap_mode="r")) if path.exists() else 0

def run(cfg: dict, run_dir: str, artifacts_dir: str) -> Path | None:

    if not cfg["dense_compare"]["enabled"]:
        log.info("dense_compare disabled")
        return None

    c = {**DEFAULTS, **{k: v for k, v in cfg["dense_compare"].items() if v is not None}}
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

    s_model, cm_per_work, scale_source = cm_scale(run_dir, artifacts_dir)
    _, unit = stage_io.scale_info(run_dir)
    if scale_source:
        upc = 1.0 / cm_per_work
        log.info("dense_compare: %.4g cm per working unit from %s", cm_per_work, scale_source)
    else:
        upc = stage_io.units_per_cm(run_dir, sparse, room_extent_cm=cfg["scale"]["provisional_extent_cm"])
        log.warning("dense_compare: scale unknown (no scale.npy, no room_fill cm_per_unit), "
                    "chamfer_to_sparse_cm left empty")
    cell = cfg["floorplan"]["cell_cm"] * upc

    area_m2, area_source = floor_area(artifacts_dir)
    if not area_source:
        log.warning("dense_compare: no floor area (floorplan area_m2, room_fill areas), density_pts_m2 left empty")

    sparse_xy = to_floor(sparse, T)[:, :2]
    origin = sparse_xy.min(axis=0)
    floor_cells = cell_ids(sparse_xy, origin, cell)
    csv_path = artifacts_dir / "dense_compare.csv"
    metrics = artifacts_dir / "dense_metrics.csv"

    def write(method: str, points: np.ndarray | None, extra: dict) -> dict:

        n = 0 if points is None else len(points)
        row = dict.fromkeys(COLUMNS + EXTRA, "")
        row.update(method=method, n_dense=n, area_source=area_source, scale_source=scale_source, **extra)
        if area_source:
            row["area_m2"] = area_m2
            row["density_pts_m2"] = n / area_m2
            row["density_raw_pts_m2"] = int(row["n_dense_raw"] or 0) / area_m2
        if scale_source:
            row["cm_per_unit"] = s_model
        if points is None or n == 0:
            row["completeness_frac"] = 0.0
        else:
            hit = cell_ids(to_floor(points, T)[:, :2], origin, cell) & floor_cells
            row["completeness_frac"] = len(hit) / max(len(floor_cells), 1)
            if scale_source:
                d, row["nn_device"] = chamfer(points, sparse, c["device"])
                row["chamfer_to_sparse_cm"] = d * cm_per_work
        stage_io.upsert_csv(csv_path, row, key="method")
        stage_io.log_metrics(log, {f"{method}.{k}": v for k, v in row.items() if k != "method"})

        return row

    sfm_csv = artifacts_dir / "sfm_metrics.csv"
    t_sfm = first_value(sfm_csv, "t_s") or first_value(sfm_csv, "t_total_s")
    write("sparse", sparse, {"t_s": t_sfm, "points_from": "clean",
                             "n_dense_raw": count(run_dir / "sfm" / "points3d.npy"), "n_dense_clean": len(sparse)})
    found = {p.parent.name for d in ("dense", "clean") for p in (run_dir / d).glob("*/points.npy")} - {"sparse"}
    methods = list(dict.fromkeys(list(cfg["dense"]["methods"]) + sorted(found)))
    for method in methods:
        clean_path, raw_path = run_dir / "clean" / method / "points.npy", run_dir / "dense" / method / "points.npy"
        extra = {k: first_value(metrics, k, "method", method) for k in ("t_s", "vram_peak_gb")}
        extra.update(n_dense_raw=count(raw_path), n_dense_clean=count(clean_path))
        if clean_path.exists():
            points, extra["points_from"] = np.load(clean_path).reshape(-1, 3), "clean"
        elif raw_path.exists():
            log.warning("dense_compare: %s not cleaned, comparing the raw cloud %s", method, raw_path)
            points, extra["points_from"] = np.load(raw_path).reshape(-1, 3), "raw"
            if unit == "cm":
                points = points * s_model  # raw dense is in model units, clean clouds in cm
        else:
            log.warning("dense_compare: %s missing, row with n_dense=0", clean_path)
            points = None
        write(method, points, extra)

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
        stage_io.upsert_csv(art / "dense_metrics.csv", {"method": "patch_match", "t_s": 12.5, "vram_peak_gb": 3.2},
                            "method")
        (run_dir / "dense" / "sgm").mkdir(parents=True)  # raw only (not cleaned), model units (s = 1)
        np.save(run_dir / "dense" / "sgm" / "points.npy", floor[::2] + rng.normal(0, 0.5, floor[::2].shape))
        cfg = {"dense_compare": {"enabled": True}, "dense": {"methods": ["patch_match", "openmvs", "full"]},
               "floorplan": {"cell_cm": 10.0}, "scale": {"provisional_extent_cm": 500}}
        out = run(cfg, str(run_dir), str(art))
        rows = {r["method"]: r for r in stage_io.read_csv(out)}
        assert list(rows["sparse"]) == COLUMNS + EXTRA
        pm, om = rows["patch_match"], rows["openmvs"]
        assert abs(float(pm["completeness_frac"]) - 0.5) < 0.03, pm
        assert float(pm["chamfer_to_sparse_cm"]) < 60, pm  # half the sparse has no dense neighbour nearby
        assert float(rows["full"]["chamfer_to_sparse_cm"]) < 2 and float(rows["full"]["completeness_frac"]) > 0.99
        assert pm["t_s"] == "12.5" and abs(float(pm["density_pts_m2"]) - len(half) / 16) < 1e-6
        assert pm["vram_peak_gb"] == "3.2" and pm["scale_source"] == "scale.npy" and pm["points_from"] == "clean"
        sg = rows["sgm"]
        assert sg["points_from"] == "raw" and sg["n_dense_raw"] == "10000" and sg["n_dense_clean"] == "0"
        assert float(sg["chamfer_to_sparse_cm"]) < 4 and float(sg["completeness_frac"]) > 0.95
        a, b = rng.uniform(0, 100, (5000, 3)), rng.uniform(0, 100, (7000, 3))
        (d_gpu, dev), (d_cpu, _) = chamfer(a, b), chamfer(a, b, "CPU:0")
        assert abs(d_gpu - d_cpu) < 1e-3, (d_gpu, d_cpu)
        print(f"chamfer {dev} {d_gpu:.5f} vs CPU {d_cpu:.5f}")
        assert om["n_dense"] == "0" and float(om["completeness_frac"]) == 0.0 and om["chamfer_to_sparse_cm"] == ""
        assert float(rows["sparse"]["completeness_frac"]) == 1.0 and float(rows["sparse"]["chamfer_to_sparse_cm"]) == 0.0
        (run_dir / "scale" / "scale.npy").unlink()
        rows = {r["method"]: r for r in stage_io.read_csv(run(cfg, str(run_dir), str(art)))}
        assert rows["patch_match"]["chamfer_to_sparse_cm"] == "" and rows["patch_match"]["scale_source"] == ""
        # room_fill fallback: clean clouds in model units, 2 cm per unit, area from the room_fill polygon
        stage_io.upsert_csv(art / "floorplan.csv", {"area_m2": ""})
        stage_io.upsert_csv(art / "room_fill.csv", {"source": "patch_match", "cm_per_unit": 2.0,
                                                    "polygon_area_m2": 8.0, "floor_area_m2": 7.0})
        rows = {r["method"]: r for r in stage_io.read_csv(run(cfg, str(run_dir), str(art)))}
        full = rows["full"]
        assert full["scale_source"] == "room_fill" and full["area_source"] == "room_fill.polygon_area_m2"
        assert 0.5 < float(full["chamfer_to_sparse_cm"]) < 4 and float(full["density_pts_m2"]) == 20000 / 8
    print("OK")
# <<< END IMPLEMENTATION <<<

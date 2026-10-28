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
# Stage: poisson (after cleanup/dense_compare, before close_holes and mesh).
# Goal: Poisson surface of the point cloud, once per variant: pre = cleaned cloud, raw = same source without
#       the cleanup stage (compares "with" vs "without pre-processing").
# In:  pre -> clean/<src>/{points,colors}.npy; raw -> dense/<src>/{points,colors}.npy * s (unit=cm) or, for
#      sparse, scale/points3d_cm.npy + sfm/colors.npy (src.cloud.clean.load_source); scale/centers_cm.npy.
# Out: mesh/poisson/<variant>/{mesh.ply,vertices,faces,colors}.npy, exports/meshlab/poisson_<variant>.ply;
#      artifacts_dir/poisson.csv keyed by variant (variant, source, n_points_in, n_vertices, n_faces, depth,
#      density_quantile, normals_device, t_normals_s, t_poisson_s, t_s). With poisson.method set:
#      mesh/poisson_<method>/<variant>/..., exports/meshlab/poisson_<method>_<variant>.ply, CSV variant
#      "<method>/<variant>" (nothing of the default output is overwritten).
# YAML: poisson: {enabled, source: dense|sparse, variants: [pre, raw], depth, density_quantile, normal_knn, threads,
#       method: null, device: auto}
# Rules: source dense = first cfg["dense"]["methods"] entry with output for the variant, else sparse (WARNING).
#        poisson.method null -> that default; a method name (or a list of names) -> the cloud of each named method
#        (dense/<m> or sparse), missing -> WARNING and that method skipped, so every densifier can be meshed.
#        The nearest-camera search of the orientation runs on poisson.device (auto -> CUDA:0 when Open3D sees a
#        GPU: open3d.core.nns 1-NN in float64, ~100x faster than the numpy search; any failure -> CPU, WARNING);
#        the Open3D CUDA cache is released after it. Normal estimation stays on the legacy CPU KD-tree (faster
#        than open3d.t on CUDA: FP64 is slow on GeForce cards, FP32 still ~7x slower) and Poisson on the CPU.
#        Normals (knn) are oriented per point towards its NEAREST camera center (more robust in a room than
#        one mean center, which flips the walls behind it); no centers -> tangent-plane consistency (WARNING).
#        Vertices with density below the density_quantile are dropped. threads -1 (all cores) oversubscribes
#        on many-core machines (~30x slower with colors); 8 is fast. Large raw clouds run as is (size and
#        time are logged).
# >>> BEGIN IMPLEMENTATION >>>
"""Poisson surface reconstruction of the point cloud, with (pre) and without (raw) the cleanup stage."""
import logging
import shutil
import time
from pathlib import Path

import numpy as np
import open3d as o3d

from src import stage_io
from src.cloud import clean
from src.cloud.clean import release_cuda, resolve_device

log = logging.getLogger(__name__)

VARIANTS = ("pre", "raw")

def cloud_path(run_dir: Path, source: str, variant: str) -> Path:

    if variant == "pre":
        return run_dir / "clean" / source / "points.npy"

    return run_dir / "scale" / "points3d_cm.npy" if source == "sparse" else run_dir / "dense" / source / "points.npy"

def pick_source(cfg: dict, run_dir: Path, variant: str = "pre", method: str | None = None) -> str | None:
    """poisson.source dense -> first dense.methods entry with a cloud for the variant, else sparse (WARNING).
    method given -> that source only (None + WARNING when its cloud is missing)."""

    run_dir = Path(run_dir)
    if method is not None:
        if cloud_path(run_dir, method, variant).exists():
            return method
        log.warning("poisson.method=%s: no %s cloud (%s), skipped", method, variant, cloud_path(run_dir, method, variant))
        return None
    if cfg["poisson"]["source"] == "dense":
        for method in cfg["dense"]["methods"]:
            if cloud_path(run_dir, method, variant).exists():
                return method
        log.warning("poisson.source=dense but no %s cloud for %s: falling back to sparse",
                    variant, cfg["dense"]["methods"])

    return "sparse" if cloud_path(run_dir, "sparse", variant).exists() else None

def load_cloud(run_dir: Path, source: str, variant: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Points in working units and RGB uint8 colors (None when missing) of a source/variant."""

    run_dir = Path(run_dir)
    if variant == "raw":
        return clean.load_source(run_dir, source)

    points = np.load(cloud_path(run_dir, source, variant)).astype(np.float64).reshape(-1, 3)
    c_path = run_dir / "clean" / source / "colors.npy"
    colors = np.load(c_path).astype(np.uint8).reshape(-1, 3) if c_path.exists() else None
    if colors is not None and len(colors) != len(points):
        log.warning("poisson: clean/%s colors misaligned, ignored", source)
        colors = None

    return points, colors

def load_centers(run_dir: Path) -> np.ndarray | None:

    run_dir = Path(run_dir)
    path = run_dir / "scale" / "centers_cm.npy"
    if path.exists():
        return np.load(path).reshape(-1, 3)

    path = run_dir / "sfm" / "centers.npy"
    if not path.exists():
        return None

    s, unit = stage_io.scale_info(run_dir)
    log.warning("scale/centers_cm.npy missing: using sfm/centers.npy%s", " * s" if unit == "cm" else " (model units)")

    return np.load(path).reshape(-1, 3) * (s if unit == "cm" else 1.0)

def orient_to_nearest_camera(points: np.ndarray, normals: np.ndarray, centers: np.ndarray,
                             budget: int = 20_000_000) -> np.ndarray:
    """Flip each normal so it points to its nearest camera center (chunked: chunk * n_centers <= budget)."""

    out = normals.copy()
    c2 = (centers ** 2).sum(1)
    chunk = max(1, budget // max(1, len(centers)))
    for i in range(0, len(points), chunk):
        p = points[i:i + chunk]
        nearest = (c2[None, :] - 2.0 * p @ centers.T).argmin(1)  # |p|^2 is constant per row
        flip = (out[i:i + chunk] * (centers[nearest] - p)).sum(1) < 0
        out[i:i + chunk][flip] *= -1

    return out

def nearest_center_tensor(points: np.ndarray, centers: np.ndarray, device: str) -> np.ndarray:
    """Index of the nearest camera center of each point (open3d.core.nns 1-NN, float64) on an Open3D device."""

    dev = o3d.core.Device(device)
    nns = o3d.core.nns.NearestNeighborSearch(o3d.core.Tensor(np.ascontiguousarray(centers, dtype=np.float64), device=dev))
    nns.knn_index()

    return nns.knn_search(o3d.core.Tensor(np.ascontiguousarray(points, dtype=np.float64), device=dev), 1)[0].cpu().numpy().reshape(-1)

def estimate_normals(pcd: o3d.geometry.PointCloud, points: np.ndarray, centers: np.ndarray | None,
                     pcfg: dict) -> str:
    """Fill pcd.normals: knn normals (legacy CPU KD-tree), flipped towards the nearest camera center searched on
    poisson.device (CUDA: open3d.core.nns; CPU or failure: chunked numpy), else tangent-plane consistency.
    Returns the device of the nearest-camera search."""

    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=pcfg["normal_knn"]))
    if centers is None or len(centers) == 0:
        log.warning("no camera centers: normals oriented by tangent-plane consistency only")
        pcd.orient_normals_consistent_tangent_plane(pcfg["normal_knn"])
        return "CPU:0"

    normals = np.asarray(pcd.normals).copy()
    device = resolve_device(pcfg.get("device", "auto"), "poisson")
    if device.startswith("CUDA"):
        try:
            nearest = nearest_center_tensor(points, centers, device)
            normals[(normals * (centers[nearest] - points)).sum(1) < 0] *= -1
            pcd.normals = o3d.utility.Vector3dVector(normals)
            return device
        except Exception as exc:  # noqa: BLE001 - any device failure falls back to the CPU search
            log.warning("poisson: nearest-camera search on %s failed (%s), using the CPU", device, exc)
        finally:
            release_cuda()
    pcd.normals = o3d.utility.Vector3dVector(orient_to_nearest_camera(points, normals, centers))

    return "CPU:0"

def reconstruct(points: np.ndarray, colors: np.ndarray | None, centers: np.ndarray | None,
                pcfg: dict, info: dict | None = None) -> o3d.geometry.TriangleMesh:
    """Normals (knn, oriented to the nearest camera searched on poisson.device) -> Poisson(depth, threads, CPU)
    -> density-quantile trim. info (optional) receives normals_device, t_normals_s and t_poisson_s."""

    t0 = time.perf_counter()
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.astype(np.float64)))
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    device = estimate_normals(pcd, points, centers, pcfg)
    t1 = time.perf_counter()

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=pcfg["depth"],
                                                                             n_threads=pcfg["threads"])
    densities = np.asarray(densities)
    if len(densities):
        mesh.remove_vertices_by_mask(densities < np.quantile(densities, pcfg["density_quantile"]))
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    mesh.triangle_normals = o3d.utility.Vector3dVector()
    if info is not None:
        info.update(normals_device=device, t_normals_s=round(t1 - t0, 3), t_poisson_s=round(time.perf_counter() - t1, 3))

    return mesh

def save_mesh(mesh: o3d.geometry.TriangleMesh, out_dir: Path) -> Path:
    """mesh.ply + vertices/faces/colors.npy in out_dir."""

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "vertices.npy", np.asarray(mesh.vertices))
    np.save(out_dir / "faces.npy", np.asarray(mesh.triangles))
    if mesh.has_vertex_colors():
        np.save(out_dir / "colors.npy", (np.asarray(mesh.vertex_colors) * 255).round().astype(np.uint8))
    path = out_dir / "mesh.ply"
    o3d.io.write_triangle_mesh(str(path), mesh, write_vertex_colors=mesh.has_vertex_colors())

    return path

def run(cfg: dict, run_dir: str, artifacts_dir: str) -> dict | None:

    pcfg = cfg["poisson"]
    if not pcfg["enabled"]:
        log.info("poisson disabled")
        return None

    run_dir = Path(run_dir)
    centers = load_centers(run_dir)
    meshlab = run_dir / "exports" / "meshlab"
    meshlab.mkdir(parents=True, exist_ok=True)
    methods = pcfg.get("method")
    methods = [None] if methods is None else [methods] if isinstance(methods, str) else list(methods)
    rows = {}
    for method in methods:
        for variant in pcfg["variants"]:
            t0 = time.perf_counter()
            key = variant if method is None else f"{method}/{variant}"
            source = pick_source(cfg, run_dir, variant, method)
            if source is None:
                log.warning("poisson %s: no input cloud (%s), skipped", key,
                            "clean/<source>/points.npy" if variant == "pre" else "dense/<m> or scale/points3d_cm.npy")
                continue
            loaded = load_cloud(run_dir, source, variant)
            if loaded is None:
                continue
            points, colors = loaded
            log.info("poisson %s: source=%s points=%d depth=%d", key, source, len(points), pcfg["depth"])

            info = {}
            mesh = reconstruct(points, colors, centers, pcfg, info)
            if len(mesh.triangles) == 0:
                log.warning("poisson %s produced no faces (source=%s), nothing saved", key, source)
                continue
            if method is None:
                out_dir, ply = run_dir / "mesh" / "poisson" / variant, f"poisson_{variant}.ply"
            else:
                out_dir, ply = run_dir / "mesh" / f"poisson_{method}" / variant, f"poisson_{method}_{variant}.ply"
            path = save_mesh(mesh, out_dir)
            shutil.copyfile(path, meshlab / ply)

            row = {"variant": key, "source": source, "n_points_in": len(points), "n_vertices": len(mesh.vertices),
                   "n_faces": len(mesh.triangles), "depth": pcfg["depth"],
                   "density_quantile": pcfg["density_quantile"], **info, "t_s": round(time.perf_counter() - t0, 3)}
            stage_io.log_metrics(log, row)
            stage_io.upsert_csv(Path(artifacts_dir) / "poisson.csv", row, key="variant")
            rows[key] = row

    return rows or None
# <<< END IMPLEMENTATION <<<

def synthetic_run(tmp: Path, rng: np.random.Generator) -> tuple[Path, np.ndarray, np.ndarray]:
    """Sphere + box room in cm with far outliers: raw = scale/points3d_cm.npy, pre = clean/sparse (no outliers).
    Also cameras (scale/centers_cm.npy, sfm/{cam_R,K,image_names}.npy). Returns (run_dir, clean pts, raw pts)."""

    run_dir = tmp / "run"
    sphere = rng.normal(size=(6000, 3))
    sphere = 100 * sphere / np.linalg.norm(sphere, axis=1, keepdims=True)
    box = rng.uniform(-150, 150, size=(3000, 3))
    box[np.arange(3000), rng.integers(0, 3, 3000)] = rng.choice([-150, 150], 3000)
    pts = np.vstack([sphere, box])
    outliers = rng.uniform(-600, 600, size=(400, 3))
    raw = np.vstack([pts, outliers])
    for d in ("clean/sparse", "sfm", "scale"):
        (run_dir / d).mkdir(parents=True)
    np.save(run_dir / "clean" / "sparse" / "points.npy", pts)
    np.save(run_dir / "clean" / "sparse" / "colors.npy", rng.integers(0, 256, (len(pts), 3), dtype=np.uint8))
    np.save(run_dir / "scale" / "points3d_cm.npy", raw)
    np.save(run_dir / "sfm" / "colors.npy", rng.integers(0, 256, (len(raw), 3), dtype=np.uint8))
    np.save(run_dir / "scale" / "scale.npy", np.array([1.0]))
    np.save(run_dir / "scale" / "centers_cm.npy", np.array([[0, 0, 130.0], [40, 0, 125], [80, 10, 110]]))
    np.save(run_dir / "sfm" / "cam_R.npy", np.stack([np.eye(3)] * 3))
    np.save(run_dir / "sfm" / "K.npy", np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1.0]]))
    np.save(run_dir / "sfm" / "image_names.npy", np.array(["img_0000.png", "img_0001.png", "img_0002.png"]))

    return run_dir, pts, raw

if __name__ == "__main__":
    import tempfile
    logging.basicConfig(level=logging.INFO)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        run_dir, pts, raw = synthetic_run(tmp, np.random.default_rng(0))
        art = tmp / "art"
        cfg = {"poisson": {"enabled": True, "source": "dense", "variants": ["pre", "raw"], "depth": 7,
                           "density_quantile": 0.05, "normal_knn": 20, "threads": 8},
               "dense": {"methods": ["patch_match", "openmvs"]}}
        rows = run(cfg, str(run_dir), str(art))
        assert rows is not None and set(rows) == {"pre", "raw"}, rows
        ext = {}
        for v in ("pre", "raw"):
            d = run_dir / "mesh" / "poisson" / v
            m = o3d.io.read_triangle_mesh(str(d / "mesh.ply"))
            assert len(m.triangles) > 0 and m.has_vertex_colors(), v
            assert len(np.load(d / "vertices.npy")) == len(m.vertices) and len(np.load(d / "colors.npy")) == len(m.vertices)
            assert (run_dir / "exports" / "meshlab" / f"poisson_{v}.ply").stat().st_size > 0
            ext[v] = np.ptp(np.asarray(m.vertices), axis=0).max()
        csv_rows = stage_io.read_csv(art / "poisson.csv")
        assert [r["variant"] for r in csv_rows] == ["pre", "raw"] and all(r["source"] == "sparse" for r in csv_rows)
        assert int(csv_rows[0]["n_points_in"]) == len(pts) and int(csv_rows[1]["n_points_in"]) == len(raw)
        nv = {r["variant"]: int(r["n_vertices"]) for r in csv_rows}
        # outliers stretch the octree (coarser cells at the same depth) and add blobs around the room
        assert nv["raw"] < 0.8 * nv["pre"] and ext["raw"] >= ext["pre"], (nv, ext)
        run(cfg, str(run_dir), str(art))
        assert len(stage_io.read_csv(art / "poisson.csv")) == 2  # upsert by variant
        assert run({**cfg, "poisson": {**cfg["poisson"], "enabled": False}}, str(run_dir), str(art)) is None
        # poisson.method: per-method output next to the default one, missing method skipped
        before = (run_dir / "mesh" / "poisson" / "pre" / "mesh.ply").stat().st_mtime_ns
        rows_m = run({**cfg, "poisson": {**cfg["poisson"], "method": ["sparse", "casmvsnet"]}}, str(run_dir), str(art))
        assert set(rows_m) == {"sparse/pre", "sparse/raw"}, rows_m
        assert (run_dir / "mesh" / "poisson_sparse" / "pre" / "mesh.ply").exists()
        assert (run_dir / "exports" / "meshlab" / "poisson_sparse_raw.ply").exists()
        assert (run_dir / "mesh" / "poisson" / "pre" / "mesh.ply").stat().st_mtime_ns == before
        assert [r["variant"] for r in stage_io.read_csv(art / "poisson.csv")] == ["pre", "raw", "sparse/pre", "sparse/raw"]
        assert run({**cfg, "poisson": {**cfg["poisson"], "method": "casmvsnet"}}, str(run_dir), str(art)) is None
        # orientation: poisson.device auto (CUDA when available) vs CPU numpy search, same normals
        pcfg = {**cfg["poisson"], "device": "CPU:0"}
        centers = load_centers(run_dir)
        n_dev = {}
        for dev in ("CPU:0", "auto"):
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
            used = estimate_normals(pcd, pts, centers, {**pcfg, "device": dev})
            n_dev[used] = np.asarray(pcd.normals).copy()
        if len(n_dev) == 2:
            dot = (n_dev["CPU:0"] * n_dev[next(k for k in n_dev if k != "CPU:0")]).sum(1)
            assert (dot > 0.999999).mean() > 0.999, (dot > 0.999999).mean()
            print("normals CPU vs GPU orientation: same sign", (dot > 0).mean())
        print(csv_rows, ext)
    print("OK")

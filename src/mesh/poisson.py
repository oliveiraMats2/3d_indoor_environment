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
#      density_quantile, t_s).
# YAML: poisson: {enabled, source: dense|sparse, variants: [pre, raw], depth, density_quantile, normal_knn, threads}
# Rules: source dense = first cfg["dense"]["methods"] entry with output for the variant, else sparse (WARNING).
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

log = logging.getLogger(__name__)

VARIANTS = ("pre", "raw")

def cloud_path(run_dir: Path, source: str, variant: str) -> Path:

    if variant == "pre":
        return run_dir / "clean" / source / "points.npy"

    return run_dir / "scale" / "points3d_cm.npy" if source == "sparse" else run_dir / "dense" / source / "points.npy"

def pick_source(cfg: dict, run_dir: Path, variant: str = "pre") -> str | None:
    """poisson.source dense -> first dense.methods entry with a cloud for the variant, else sparse (WARNING)."""

    run_dir = Path(run_dir)
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

def reconstruct(points: np.ndarray, colors: np.ndarray | None, centers: np.ndarray | None,
                pcfg: dict) -> o3d.geometry.TriangleMesh:
    """Normals (knn, oriented to the nearest camera) -> Poisson(depth, threads) -> density-quantile trim."""

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.astype(np.float64)))
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=pcfg["normal_knn"]))

    if centers is not None and len(centers):
        pcd.normals = o3d.utility.Vector3dVector(orient_to_nearest_camera(points, np.asarray(pcd.normals), centers))
    else:
        log.warning("no camera centers: normals oriented by tangent-plane consistency only")
        pcd.orient_normals_consistent_tangent_plane(pcfg["normal_knn"])

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=pcfg["depth"],
                                                                             n_threads=pcfg["threads"])
    densities = np.asarray(densities)
    if len(densities):
        mesh.remove_vertices_by_mask(densities < np.quantile(densities, pcfg["density_quantile"]))
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    mesh.triangle_normals = o3d.utility.Vector3dVector()

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
    rows = {}
    for variant in pcfg["variants"]:
        t0 = time.perf_counter()
        source = pick_source(cfg, run_dir, variant)
        if source is None:
            log.warning("poisson %s: no input cloud (%s), skipped", variant,
                        "clean/<source>/points.npy" if variant == "pre" else "dense/<m> or scale/points3d_cm.npy")
            continue
        loaded = load_cloud(run_dir, source, variant)
        if loaded is None:
            continue
        points, colors = loaded
        log.info("poisson %s: source=%s points=%d depth=%d", variant, source, len(points), pcfg["depth"])

        mesh = reconstruct(points, colors, centers, pcfg)
        if len(mesh.triangles) == 0:
            log.warning("poisson %s produced no faces (source=%s), nothing saved", variant, source)
            continue
        path = save_mesh(mesh, run_dir / "mesh" / "poisson" / variant)
        shutil.copyfile(path, meshlab / f"poisson_{variant}.ply")

        row = {"variant": variant, "source": source, "n_points_in": len(points), "n_vertices": len(mesh.vertices),
               "n_faces": len(mesh.triangles), "depth": pcfg["depth"], "density_quantile": pcfg["density_quantile"],
               "t_s": round(time.perf_counter() - t0, 3)}
        stage_io.log_metrics(log, row)
        stage_io.upsert_csv(Path(artifacts_dir) / "poisson.csv", row, key="variant")
        rows[variant] = row

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
        print(csv_rows, ext)
    print("OK")

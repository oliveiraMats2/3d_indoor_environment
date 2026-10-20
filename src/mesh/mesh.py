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
# Stage: mesh (E2) + 3D scene export (X1).
# Goal: Poisson surface from the cleaned cloud and a web/MeshLab scene with the camera path and frustums.
# In:  clean/<source>/{points,colors}.npy (working units), scale/centers_cm.npy, sfm/{cam_R,K,image_names}.npy,
#      one frame of cfg["extraction"]["output_dir"] (image size; falls back to 2*cx, 2*cy).
# Out: exports/meshlab/mesh.{ply,obj,stl,off}, exports/meshlab/scene.obj, exports/web/{mesh,scene}.glb,
#      mesh/{vertices,faces,colors}.npy; artifacts_dir/mesh.csv (source, n_vertices, n_faces, poisson_depth, t_s).
# YAML: mesh: {enabled, source: dense|sparse, poisson_depth, density_quantile, normal_knn, poisson_threads, frustum_cm,
#       frustum_every, trajectory_cm}
# Rules: source dense = first cfg["dense"]["methods"] entry with clean/<method>/points.npy, else clean/sparse
#        (WARNING). Normals are oriented per point towards its NEAREST camera center (a point is seen from the
#        camera side, so the normal must face it); this is more robust in a room than one mean center, which
#        sits inside the room and flips walls behind it. *_cm sizes -> working units via stage_io.units_per_cm.
#        poisson_threads: -1 (all cores) oversubscribes on 32 threads (~30x slower with colors); 8 is fast.
#        Every written file is reloaded (trimesh for GLB, Open3D for OBJ/STL/OFF) as a check.
# >>> BEGIN IMPLEMENTATION >>>
"""Poisson mesh of the cleaned cloud plus a GLB/OBJ scene with the camera trajectory and frustums."""
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import trimesh

from src import stage_io

log = logging.getLogger(__name__)

FRUSTUM_RGBA = np.array([230, 60, 40, 255], dtype=np.uint8)
TRAJECTORY_RGBA = np.array([40, 120, 230, 255], dtype=np.uint8)

def pick_source(cfg: dict, run_dir: Path) -> str | None:

    clean = run_dir / "clean"
    if cfg["mesh"]["source"] == "dense":
        for method in cfg.get("dense", {}).get("methods", []):
            if (clean / method / "points.npy").exists():
                return method
        log.warning("mesh.source=dense but no clean/<method>/points.npy for %s: falling back to clean/sparse",
                    cfg.get("dense", {}).get("methods", []))

    return "sparse" if (clean / "sparse" / "points.npy").exists() else None

def load_centers(run_dir: Path) -> np.ndarray | None:

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
                             chunk: int = 100_000) -> np.ndarray:
    """Flip each normal so it points to its nearest camera center."""

    out = normals.copy()
    for i in range(0, len(points), chunk):
        p = points[i:i + chunk]
        d2 = ((p[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        view = centers[d2.argmin(1)] - p
        flip = (out[i:i + chunk] * view).sum(1) < 0
        out[i:i + chunk][flip] *= -1

    return out

def poisson(points: np.ndarray, colors: np.ndarray | None, centers: np.ndarray | None,
            mcfg: dict) -> o3d.geometry.TriangleMesh:

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.astype(np.float64)))
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=mcfg["normal_knn"]))

    normals = np.asarray(pcd.normals)
    if centers is not None and len(centers):
        normals = orient_to_nearest_camera(points, normals, centers)
    else:
        log.warning("no camera centers: normals oriented by tangent-plane consistency only")
        pcd.orient_normals_consistent_tangent_plane(mcfg["normal_knn"])
        normals = np.asarray(pcd.normals)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=mcfg["poisson_depth"],
                                                                             n_threads=mcfg["poisson_threads"])
    densities = np.asarray(densities)
    mesh.remove_vertices_by_mask(densities < np.quantile(densities, mcfg["density_quantile"]))
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    mesh.triangle_normals = o3d.utility.Vector3dVector()

    return mesh

def image_size(cfg: dict, run_dir: Path, K: np.ndarray) -> tuple[int, int]:

    names_path = run_dir / "sfm" / "image_names.npy"
    frames_dir = Path(cfg["extraction"]["output_dir"])
    if names_path.exists():
        names = [str(n) for n in np.load(names_path)]
        for name in names[:1]:
            for path in [frames_dir / name, *frames_dir.glob(f"{Path(name).stem}.*")]:
                image = cv2.imread(str(path)) if path.is_file() else None
                if image is not None:
                    return image.shape[1], image.shape[0]

    log.warning("frame image not found in %s: image size from 2*cx, 2*cy", frames_dir)

    return int(round(2 * K[0, 2])), int(round(2 * K[1, 2]))

def frustums(cam_R: np.ndarray, centers: np.ndarray, K: np.ndarray, wh: tuple[int, int],
             size: float, every: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pyramids (apex at the center, base at depth `size`). Returns vertices (5F,3), side triangles (4F,3)
    and edge segments (8F,2) as vertex indices."""

    w, h = wh
    pix = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    rays = np.c_[(pix[:, 0] - K[0, 2]) / K[0, 0], (pix[:, 1] - K[1, 2]) / K[1, 1], np.ones(4)] * size

    verts, tris, edges = [], [], []
    for i in range(0, len(centers), max(1, every)):
        base = 5 * len(verts)
        corners = rays @ cam_R[i] + centers[i]  # R^T x_cam + C, row form
        verts.append(np.vstack([centers[i], corners]))
        tris += [[base, base + 1 + k, base + 1 + (k + 1) % 4] for k in range(4)]
        edges += [[base, base + 1 + k] for k in range(4)] + [[base + 1 + k, base + 1 + (k + 1) % 4] for k in range(4)]

    return np.vstack(verts) if verts else np.zeros((0, 3)), np.array(tris, int).reshape(-1, 3), \
        np.array(edges, int).reshape(-1, 2)

def tubes(path: np.ndarray, radius: float, sections: int = 6) -> trimesh.Trimesh | None:
    """Thin cylinders along a polyline (visible in viewers that ignore line primitives)."""

    parts = [trimesh.creation.cylinder(radius=radius, segment=path[i:i + 2], sections=sections)
             for i in range(len(path) - 1) if np.linalg.norm(path[i + 1] - path[i]) > 1e-9]

    return trimesh.util.concatenate(parts) if parts else None

def write_scene_obj(path: Path, points: np.ndarray, colors: np.ndarray | None, f_verts: np.ndarray,
                    f_tris: np.ndarray, f_edges: np.ndarray, centers: np.ndarray, tube: trimesh.Trimesh | None) -> None:
    """Points as colored vertices, frustums as triangles + lines, trajectory as tubes + polyline."""

    rgb_pts = (colors / 255.0) if colors is not None else np.full((len(points), 3), 0.7)
    blocks = [(points, rgb_pts),
              (f_verts, np.tile(FRUSTUM_RGBA[:3] / 255.0, (len(f_verts), 1))),
              (centers, np.tile(TRAJECTORY_RGBA[:3] / 255.0, (len(centers), 1)))]
    if tube is not None:
        blocks.append((np.asarray(tube.vertices), np.tile(TRAJECTORY_RGBA[:3] / 255.0, (len(tube.vertices), 1))))

    o_f = len(points) + 1  # OBJ is 1-based
    o_c = o_f + len(f_verts)
    o_t = o_c + len(centers)
    with open(path, "w") as fh:
        fh.write("# point cloud (vertices), camera frustums (faces+lines), trajectory (tubes+polyline)\n")
        for v, c in blocks:
            fh.writelines(f"v {a:.6f} {b:.6f} {z:.6f} {r:.4f} {g:.4f} {bl:.4f}\n"
                          for (a, b, z), (r, g, bl) in zip(v, c))
        fh.write("g frustums\n")
        fh.writelines(f"f {a + o_f} {b + o_f} {c + o_f}\n" for a, b, c in f_tris)
        fh.writelines(f"l {a + o_f} {b + o_f}\n" for a, b in f_edges)
        fh.write("g trajectory\n")
        if len(centers) > 1:
            fh.write("l " + " ".join(str(o_c + i) for i in range(len(centers))) + "\n")
        if tube is not None:
            fh.writelines(f"f {a + o_t} {b + o_t} {c + o_t}\n" for a, b, c in np.asarray(tube.faces))

def colored_path(vertices: np.ndarray, segments: np.ndarray, rgba: np.ndarray) -> trimesh.path.Path3D:

    entities = [trimesh.path.entities.Line(seg) for seg in segments]  # direct build: load_path needs scipy
    path = trimesh.path.Path3D(entities=entities, vertices=vertices, colors=np.tile(rgba, (len(entities), 1)),
                               process=False)

    return path

def verify(paths: dict[str, Path]) -> None:

    for name, path in paths.items():
        if path.suffix == ".glb":
            loaded = trimesh.load(str(path))
            assert len(getattr(loaded, "geometry", {"m": loaded})) > 0, f"{path} reloads empty"
        elif name != "scene_obj":
            loaded = o3d.io.read_triangle_mesh(str(path))
            assert len(loaded.vertices) > 0 and len(loaded.triangles) > 0, f"{path} reloads empty"
        else:
            assert sum(1 for ln in open(path) if ln.startswith("v ")) > 0, f"{path} has no vertices"
        log.info("verified %s", path)

def run(cfg: dict, run_dir: str, artifacts_dir: str) -> Path | None:

    mcfg = cfg["mesh"]
    if not mcfg["enabled"]:
        log.info("mesh disabled")
        return None

    t0 = time.perf_counter()
    run_dir = Path(run_dir)
    source = pick_source(cfg, run_dir)
    if source is None:
        log.warning("mesh: no clean/<source>/points.npy found, skipping")
        return None

    points = np.load(run_dir / "clean" / source / "points.npy").reshape(-1, 3)
    colors_path = run_dir / "clean" / source / "colors.npy"
    colors = np.load(colors_path).reshape(-1, 3).astype(np.uint8) if colors_path.exists() else None
    centers = load_centers(run_dir)
    log.info("mesh source=%s points=%d", source, len(points))

    mesh = poisson(points, colors, centers, mcfg)
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    if len(faces) == 0:
        log.warning("Poisson produced no faces (source=%s), nothing exported", source)
        return None
    vcolors = (np.asarray(mesh.vertex_colors) * 255).round().astype(np.uint8) if mesh.has_vertex_colors() else None

    out_npy = run_dir / "mesh"
    out_npy.mkdir(parents=True, exist_ok=True)
    np.save(out_npy / "vertices.npy", vertices)
    np.save(out_npy / "faces.npy", faces)
    if vcolors is not None:
        np.save(out_npy / "colors.npy", vcolors)

    meshlab, web = run_dir / "exports" / "meshlab", run_dir / "exports" / "web"
    meshlab.mkdir(parents=True, exist_ok=True)
    web.mkdir(parents=True, exist_ok=True)
    files = {ext: meshlab / f"mesh.{ext}" for ext in ("ply", "obj", "off", "stl")}  # stl last: needs triangle normals
    for path in files.values():
        if path.suffix == ".stl":
            mesh.compute_triangle_normals()  # STL requires them; OBJ/OFF writers warn if present
        o3d.io.write_triangle_mesh(str(path), mesh, write_vertex_colors=path.suffix != ".stl")

    tmesh = trimesh.Trimesh(vertices, faces, vertex_colors=vcolors, process=False)
    files["glb"] = web / "mesh.glb"
    tmesh.export(str(files["glb"]))

    # X1: scene with point cloud, trajectory and frustums
    cam_R_path, K_path = run_dir / "sfm" / "cam_R.npy", run_dir / "sfm" / "K.npy"
    upc = stage_io.units_per_cm(run_dir, points, room_extent_cm=cfg["scale"]["provisional_extent_cm"])
    scene = trimesh.Scene()
    pc_colors = np.c_[colors, np.full(len(colors), 255, np.uint8)] if colors is not None else None
    scene.add_geometry(trimesh.PointCloud(points, colors=pc_colors), geom_name="points")
    f_verts, f_tris, f_edges = np.zeros((0, 3)), np.zeros((0, 3), int), np.zeros((0, 2), int)
    tube = None
    cams = centers if centers is not None else np.zeros((0, 3))
    if centers is not None and cam_R_path.exists() and K_path.exists():
        cam_R, K = np.load(cam_R_path), np.load(K_path)
        f_verts, f_tris, f_edges = frustums(cam_R, centers, K, image_size(cfg, run_dir, K),
                                            mcfg["frustum_cm"] * upc, mcfg["frustum_every"])
        tube = tubes(centers, mcfg["trajectory_cm"] * upc / 2)
        if len(f_edges):
            scene.add_geometry(colored_path(f_verts, f_edges, FRUSTUM_RGBA), geom_name="frustums")
        if tube is not None:
            tube.visual.vertex_colors = np.tile(TRAJECTORY_RGBA, (len(tube.vertices), 1))  # face colors need scipy
            scene.add_geometry(tube, geom_name="trajectory")
    else:
        log.warning("sfm/cam_R.npy, sfm/K.npy or camera centers missing: scene without cameras")

    files["scene_glb"] = web / "scene.glb"
    scene.export(str(files["scene_glb"]))
    files["scene_obj"] = meshlab / "scene.obj"
    write_scene_obj(files["scene_obj"], points, colors, f_verts, f_tris, f_edges, cams, tube)
    verify(files)

    row = {"source": source, "n_vertices": len(vertices), "n_faces": len(faces),
           "poisson_depth": mcfg["poisson_depth"], "t_s": round(time.perf_counter() - t0, 3)}
    stage_io.log_metrics(log, row)
    stage_io.upsert_csv(Path(artifacts_dir) / "mesh.csv", row)

    return files["glb"]
# <<< END IMPLEMENTATION <<<

if __name__ == "__main__":
    import tempfile
    logging.basicConfig(level=logging.INFO)
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        run_dir, art, frames = tmp / "run", tmp / "art", tmp / "frames"
        sphere = rng.normal(size=(6000, 3))
        sphere = 100 * sphere / np.linalg.norm(sphere, axis=1, keepdims=True)
        box = rng.uniform(-150, 150, size=(3000, 3))
        box[np.arange(3000), rng.integers(0, 3, 3000)] = rng.choice([-150, 150], 3000)
        pts = np.vstack([sphere, box])
        (run_dir / "clean" / "sparse").mkdir(parents=True)
        np.save(run_dir / "clean" / "sparse" / "points.npy", pts)
        np.save(run_dir / "clean" / "sparse" / "colors.npy", rng.integers(0, 256, (len(pts), 3), dtype=np.uint8))
        centers = np.array([[0, 0, 130.0], [40, 0, 125], [80, 10, 110]])
        cam_R = np.stack([np.eye(3)] * 3)
        (run_dir / "sfm").mkdir()
        (run_dir / "scale").mkdir()
        np.save(run_dir / "scale" / "scale.npy", np.array([1.0]))
        np.save(run_dir / "scale" / "centers_cm.npy", centers)
        np.save(run_dir / "sfm" / "cam_R.npy", cam_R)
        np.save(run_dir / "sfm" / "K.npy", np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1.0]]))
        np.save(run_dir / "sfm" / "image_names.npy", np.array(["img_0000.png", "img_0001.png", "img_0002.png"]))
        frames.mkdir()
        cv2.imwrite(str(frames / "img_0000.png"), np.zeros((480, 640, 3), np.uint8))
        cfg = {"mesh": {"enabled": True, "source": "dense", "poisson_depth": 7, "density_quantile": 0.05,
                        "normal_knn": 20, "poisson_threads": 8, "frustum_cm": 20, "frustum_every": 1, "trajectory_cm": 1.0},
               "dense": {"methods": ["patch_match", "openmvs"]}, "scale": {"provisional_extent_cm": 500},
               "extraction": {"output_dir": str(frames)}}
        assert run(cfg, str(run_dir), str(art)) is not None
        for rel in ["exports/meshlab/mesh.ply", "exports/meshlab/mesh.obj", "exports/meshlab/mesh.stl",
                    "exports/meshlab/mesh.off", "exports/meshlab/scene.obj", "exports/web/mesh.glb",
                    "exports/web/scene.glb", "mesh/vertices.npy", "mesh/faces.npy"]:
            assert (run_dir / rel).stat().st_size > 0, rel
        glb = trimesh.load(str(run_dir / "exports/web/mesh.glb"))
        g = glb.geometry[next(iter(glb.geometry))] if hasattr(glb, "geometry") else glb
        assert len(g.faces) > 0 and g.visual.kind == "vertex"
        sc = trimesh.load(str(run_dir / "exports/web/scene.glb"))
        assert {"points", "frustums", "trajectory"} <= set(sc.geometry), set(sc.geometry)
        assert len(sc.geometry["points"].vertices) == len(pts)
        row = stage_io.read_csv(art / "mesh.csv")[0]
        assert row["source"] == "sparse" and int(row["n_vertices"]) > 0 and int(row["n_faces"]) > 0, row
        print(row)
    print("OK")

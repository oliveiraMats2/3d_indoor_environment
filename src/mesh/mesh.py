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
# Goal: export ONE chosen mesh to every format and a web/MeshLab scene with the camera path and frustums.
#       Poisson itself runs in src/mesh/poisson.py, hole closing in src/mesh/close_holes.py.
# In:  mesh.input close_holes -> mesh/close_holes/pre_pre/mesh.ply (falls back to mesh/poisson/pre, WARNING);
#      liepa -> mesh/liepa/mesh.ply (same fallback); poisson -> mesh/poisson/pre/mesh.ply. Scene: clean/<source>/{points,colors}.npy (source as in
#      poisson.pick_source), scale/centers_cm.npy, sfm/{cam_R,K,image_names}.npy, one frame of
#      cfg["extraction"]["output_dir"] (image size; falls back to 2*cx, 2*cy).
# Out: exports/meshlab/mesh.{ply,obj,stl,off}, exports/meshlab/scene.obj, exports/web/{mesh,scene}.glb,
#      mesh/{vertices,faces,colors}.npy; artifacts_dir/mesh.csv (source, input, n_vertices, n_faces,
#      poisson_depth, t_s).
# YAML: mesh: {enabled, input: close_holes|liepa|poisson, frustum_cm, frustum_every, trajectory_cm};
#       poisson.{source, depth} (source of the scene cloud, depth logged in the CSV).
# Rules: *_cm sizes -> working units via stage_io.units_per_cm. Every written file is reloaded (trimesh for
#        GLB, Open3D for OBJ/STL/OFF) as a check.
# >>> BEGIN IMPLEMENTATION >>>
"""Exports the chosen mesh (PLY/OBJ/STL/OFF/GLB) plus a GLB/OBJ scene with the camera trajectory and frustums."""
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import trimesh

from src import stage_io
from src.mesh import poisson

log = logging.getLogger(__name__)

FRUSTUM_RGBA = np.array([230, 60, 40, 255], dtype=np.uint8)
TRAJECTORY_RGBA = np.array([40, 120, 230, 255], dtype=np.uint8)

def pick_input(cfg: dict, run_dir: Path) -> Path | None:
    """mesh.input close_holes -> close_holes/pre_pre, liepa -> liepa (else poisson/pre, WARNING);
    poisson -> poisson/pre."""

    pre = run_dir / "mesh" / "poisson" / "pre" / "mesh.ply"
    chosen = {"close_holes": ("close_holes", "pre_pre"), "liepa": ("liepa",)}.get(cfg["mesh"]["input"])
    if chosen:
        path = run_dir.joinpath("mesh", *chosen, "mesh.ply")
        if path.exists():
            return path
        log.warning("mesh.input=%s but %s missing: falling back to %s", cfg["mesh"]["input"], path, pre)

    return pre if pre.exists() else None

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
    in_path = pick_input(cfg, run_dir)
    if in_path is None:
        log.warning("mesh: no mesh/poisson/pre/mesh.ply (run the poisson stage), skipping")
        return None
    mesh = o3d.io.read_triangle_mesh(str(in_path))
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    if len(faces) == 0:
        log.warning("mesh: %s has no faces, nothing exported", in_path)
        return None
    mesh.compute_vertex_normals()
    mesh.triangle_normals = o3d.utility.Vector3dVector()
    vcolors = (np.asarray(mesh.vertex_colors) * 255).round().astype(np.uint8) if mesh.has_vertex_colors() else None
    input_rel = in_path.parent.relative_to(run_dir / "mesh").as_posix()
    log.info("mesh input=%s vertices=%d faces=%d", input_rel, len(vertices), len(faces))

    source = poisson.pick_source(cfg, run_dir, "pre")
    if source is not None:
        points, colors = poisson.load_cloud(run_dir, source, "pre")
    else:
        log.warning("mesh: no clean/<source>/points.npy for the scene: using the mesh vertices as points")
        points, colors = vertices, vcolors
    centers = poisson.load_centers(run_dir)

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

    row = {"source": source or "", "input": input_rel, "n_vertices": len(vertices), "n_faces": len(faces),
           "poisson_depth": cfg["poisson"]["depth"], "t_s": round(time.perf_counter() - t0, 3)}
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
        run_dir, pts, _ = poisson.synthetic_run(tmp, rng)
        art, frames = tmp / "art", tmp / "frames"
        frames.mkdir()
        cv2.imwrite(str(frames / "img_0000.png"), np.zeros((480, 640, 3), np.uint8))
        cfg = {"poisson": {"enabled": True, "source": "dense", "variants": ["pre"], "depth": 7,
                           "density_quantile": 0.05, "normal_knn": 20, "threads": 8},
               "mesh": {"enabled": True, "input": "close_holes", "frustum_cm": 20, "frustum_every": 1,
                        "trajectory_cm": 1.0},
               "dense": {"methods": ["patch_match", "openmvs"]}, "scale": {"provisional_extent_cm": 500},
               "extraction": {"output_dir": str(frames)}}
        assert run(cfg, str(run_dir), str(art)) is None  # no Poisson mesh yet
        assert poisson.run(cfg, str(run_dir), str(art)) is not None
        assert run(cfg, str(run_dir), str(art)) is not None  # close_holes missing -> poisson/pre
        files = ["exports/meshlab/mesh.ply", "exports/meshlab/mesh.obj", "exports/meshlab/mesh.stl",
                 "exports/meshlab/mesh.off", "exports/meshlab/scene.obj", "exports/web/mesh.glb", "exports/web/scene.glb"]
        for rel in files + ["mesh/vertices.npy", "mesh/faces.npy"]:
            assert (run_dir / rel).stat().st_size > 0, rel
        for rel in files[:4]:
            assert len(o3d.io.read_triangle_mesh(str(run_dir / rel)).triangles) > 0, rel
        assert sum(1 for ln in open(run_dir / files[4]) if ln.startswith("v ")) > len(pts)
        glb = trimesh.load(str(run_dir / "exports/web/mesh.glb"))
        g = glb.geometry[next(iter(glb.geometry))] if hasattr(glb, "geometry") else glb
        assert len(g.faces) > 0 and g.visual.kind == "vertex"
        sc = trimesh.load(str(run_dir / "exports/web/scene.glb"))
        assert {"points", "frustums", "trajectory"} <= set(sc.geometry), set(sc.geometry)
        assert len(sc.geometry["points"].vertices) == len(pts)
        row = stage_io.read_csv(art / "mesh.csv")[0]
        n_pre = len(np.load(run_dir / "mesh/poisson/pre/vertices.npy"))
        assert row["source"] == "sparse" and row["input"] == "poisson/pre" and int(row["n_vertices"]) == n_pre, row
        assert int(row["n_faces"]) > 0 and row["poisson_depth"] == "7", row
        print(row)
    print("OK")

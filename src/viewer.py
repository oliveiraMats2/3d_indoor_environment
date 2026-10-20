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

"""Viewer: Open3D window (or offscreen render) of a cloud with the camera trajectory and frustums;
optionally records an orbit video."""
# PROMPT COMPLEMENT
# Stage: viewer (standalone, `python view.py --config <yaml>`; not called by main.py).
# Goal : show the reconstruction (cloud or mesh) + camera trajectory + frustums; record: true renders an
#        orbit around the scene centre into an mp4 without user interaction.
# In   : viewer.cloud -> sparse: clean/sparse/cloud_clean.ply (fallback exports/colmap/sparse.ply);
#        patch_match|openmvs: clean/<m>/cloud_clean.ply (fallback dense/<m>/fused.ply);
#        mesh: exports/meshlab/mesh.ply. scale/centers_cm.npy, sfm/{K,cam_R,cam_t}.npy,
#        planes/T_floor.npy (optional, gives the up axis).
# Out  : record true -> <run_dir>/videos/navigation_<cloud>_<seconds>s.mp4 (mp4v). No CSV (videos.csv
#        belongs to the videos stage).
# YAML : viewer.{cloud, record, seconds, fps, offscreen, width, height, frustum_cm, point_size, fov_deg,
#        orbit_radius_factor, orbit_elevation_deg, background}; scale.provisional_extent_cm.
# Rules: clean clouds and mesh are in working units (frame of scale/centers_cm.npy); raw sparse.ply and
#        fused.ply are in model units -> multiplied by s when stage_io.scale_info says unit=cm.
#        Frustum size = frustum_cm * stage_io.units_per_cm. offscreen true -> rendering.OffscreenRenderer
#        (EGL/Vulkan, no display needed); false -> hidden legacy Visualizer for recording (Open3D falls
#        back to EGL when no display), interactive window when record is false.
#        On this machine (Open3D 0.20, RTX 5080, SSH session with DISPLAY=localhost:10.0) GLFW cannot
#        open an X11 window; both OffscreenRenderer and the hidden Visualizer (EGL fallback) render fine.
# >>> BEGIN IMPLEMENTATION >>>
import ctypes, ctypes.util, logging, os
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm

from src import stage_io

log = logging.getLogger(__name__)

TRAJECTORY_RGB = (1.0, 0.0, 0.0)
FRUSTUM_RGB = (0.0, 0.3, 1.0)

def cloud_path(run_dir: Path, cloud: str) -> tuple[Path | None, bool]:
    """(path, in_model_units) of the first existing candidate for viewer.cloud."""

    candidates = {
        "sparse": [("clean/sparse/cloud_clean.ply", False), ("exports/colmap/sparse.ply", True)],
        "patch_match": [("clean/patch_match/cloud_clean.ply", False), ("dense/patch_match/fused.ply", True)],
        "openmvs": [("clean/openmvs/cloud_clean.ply", False), ("dense/openmvs/fused.ply", True)],
        "mesh": [("exports/meshlab/mesh.ply", False)],
    }[cloud]

    for rel, raw in candidates:
        if (run_dir / rel).exists():
            return run_dir / rel, raw

    return None, False

def load_cloud(run_dir: Path, cloud: str):

    path, raw = cloud_path(run_dir, cloud)
    if path is None:
        log.warning("viewer: no file for cloud=%s under %s", cloud, run_dir)
        return None

    geom = o3d.io.read_triangle_mesh(str(path)) if cloud == "mesh" else o3d.io.read_point_cloud(str(path))
    if cloud == "mesh":
        geom.compute_vertex_normals()

    s, unit = stage_io.scale_info(run_dir)
    if raw and unit == "cm":
        geom.scale(s, center=np.zeros(3))
    elif raw:
        log.warning("viewer: %s in model units (scale unknown)", path.name)

    log.info("viewer: %s (%s)", path, "mesh" if cloud == "mesh" else f"{len(geom.points)} points")

    return geom

def cameras(run_dir: Path):
    """(K, R (N,3,3), t_working (N,3), centers_working (N,3)) or None."""

    sfm = run_dir / "sfm"
    if not all((sfm / f).exists() for f in ("K.npy", "cam_R.npy", "cam_t.npy")):
        log.warning("viewer: sfm/{K,cam_R,cam_t}.npy missing, no trajectory or frustums")
        return None

    K, R, t = (np.load(sfm / f) for f in ("K.npy", "cam_R.npy", "cam_t.npy"))
    s, unit = stage_io.scale_info(run_dir)
    t = t * s if unit == "cm" else t
    centers_path = run_dir / "scale" / "centers_cm.npy"
    centers = np.load(centers_path) if centers_path.exists() else -np.einsum("nji,nj->ni", R, t)

    return K, R, t, centers

def camera_geometries(run_dir: Path, cams, frustum_size: float) -> list:

    K, R, t, centers = cams
    w, h = int(round(2 * K[0, 2])), int(round(2 * K[1, 2]))
    geoms = []
    if len(centers) > 1:
        traj = o3d.geometry.LineSet(o3d.utility.Vector3dVector(centers),
                                    o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(centers) - 1)]))
        traj.paint_uniform_color(TRAJECTORY_RGB)
        geoms.append(traj)

    for Ri, ti in zip(R, t):
        ext = np.eye(4)
        ext[:3, :3], ext[:3, 3] = Ri, ti
        frustum = o3d.geometry.LineSet.create_camera_visualization(w, h, K, ext, frustum_size)
        frustum.paint_uniform_color(FRUSTUM_RGB)
        geoms.append(frustum)

    return geoms

def up_axis(run_dir: Path, cams) -> np.ndarray:
    """World up: floor normal from planes/T_floor.npy, else the mean camera -y (COLMAP: y down)."""

    t_floor = run_dir / "planes" / "T_floor.npy"
    if t_floor.exists():
        up = np.load(t_floor)[:3, :3].T @ np.array([0.0, 0.0, 1.0])
    elif cams is not None:
        up = -cams[1][:, 1, :].mean(axis=0)
    else:
        up = np.array([0.0, 0.0, 1.0])

    return up / (np.linalg.norm(up) or 1.0)

def orbit_eyes(centre: np.ndarray, radius: float, up: np.ndarray, elevation_deg: float, n: int) -> np.ndarray:

    e1 = np.cross(up, [1.0, 0.0, 0.0])
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(up, [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    el = np.radians(elevation_deg)
    a = np.linspace(0.0, 2 * np.pi, n, endpoint=False)[:, None]
    ring = np.cos(a) * e1 + np.sin(a) * e2

    return centre + radius * (np.cos(el) * ring + np.sin(el) * up)

def look_at(eye: np.ndarray, centre: np.ndarray, up: np.ndarray) -> np.ndarray:
    """4x4 world->camera extrinsic, OpenCV axes (x right, y down, z forward)."""

    z = centre - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    ext = np.eye(4)
    ext[:3, :3] = np.stack([x, y, z])
    ext[:3, 3] = -ext[:3, :3] @ eye

    return ext

def scene_frame(geoms: list) -> tuple[np.ndarray, float]:
    """(centre, robust radius) from every vertex of the geometries."""

    pts = [np.asarray(g.points if hasattr(g, "points") else g.vertices) for g in geoms]
    pts = np.concatenate([p for p in pts if len(p)])
    lo, hi = np.percentile(pts, [2, 98], axis=0)

    return np.median(pts, axis=0), float(np.linalg.norm(hi - lo)) / 2 or 1.0

def render_offscreen(geoms: list, v: dict, eyes: np.ndarray, centre: np.ndarray, up: np.ndarray):

    renderer = o3d.visualization.rendering.OffscreenRenderer(v["width"], v["height"])
    renderer.scene.set_background([*v["background"], 1.0])
    renderer.scene.view.set_post_processing(False)  # no tone mapping: true colours and background
    for i, g in enumerate(geoms):
        mat = o3d.visualization.rendering.MaterialRecord()
        is_lines = isinstance(g, o3d.geometry.LineSet)
        is_mesh = isinstance(g, o3d.geometry.TriangleMesh)
        mat.shader = "unlitLine" if is_lines else "defaultLit" if is_mesh else "defaultUnlit"
        mat.point_size = float(v["point_size"])
        mat.line_width = 1.0
        renderer.scene.add_geometry(f"g{i}", g, mat)

    for eye in eyes:
        renderer.setup_camera(float(v["fov_deg"]), centre, eye, up)
        yield cv2.cvtColor(np.asarray(renderer.render_to_image()), cv2.COLOR_RGB2BGR)

def render_hidden(geoms: list, v: dict, eyes: np.ndarray, centre: np.ndarray, up: np.ndarray):

    w, h = v["width"], v["height"]
    vis = o3d.visualization.Visualizer()
    if not vis.create_window(width=w, height=h, visible=False):
        log.warning("viewer: hidden Open3D window failed, falling back to OffscreenRenderer")
        yield from render_offscreen(geoms, v, eyes, centre, up)
        return

    for g in geoms:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.point_size = float(v["point_size"])
    opt.background_color = np.asarray(v["background"], dtype=float)
    ctr = vis.get_view_control()
    f = (h / 2) / np.tan(np.radians(v["fov_deg"]) / 2)
    params = ctr.convert_to_pinhole_camera_parameters()
    params.intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, f, f, w / 2 - 0.5, h / 2 - 0.5)

    try:
        for eye in eyes:
            params.extrinsic = look_at(eye, centre, up)
            ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
            rgb = np.asarray(vis.capture_screen_float_buffer(do_render=True))
            frame = cv2.cvtColor((np.clip(rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            yield frame if frame.shape[:2] == (h, w) else cv2.resize(frame, (w, h))
    finally:
        vis.destroy_window()

def record(geoms: list, v: dict, run_dir: Path, up: np.ndarray) -> Path:

    centre, radius = scene_frame(geoms)
    n = int(round(v["seconds"] * v["fps"]))
    eyes = orbit_eyes(centre, radius * v["orbit_radius_factor"], up, v["orbit_elevation_deg"], n)
    out = run_dir / "videos" / f"navigation_{v['cloud']}_{v['seconds']}s.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), float(v["fps"]), (v["width"], v["height"]))
    frames = (render_offscreen if v["offscreen"] else render_hidden)(geoms, v, eyes, centre, up)

    for frame in tqdm(frames, total=n, desc="viewer orbit"):
        writer.write(frame)
    writer.release()
    log.info("viewer: %d frames -> %s", n, out)

    return out

def has_display() -> bool:
    """True when a Wayland socket is announced or XOpenDisplay succeeds (Open3D silently goes headless
    otherwise and an interactive window would block forever, invisible)."""

    if os.environ.get("WAYLAND_DISPLAY"):
        return True

    lib = ctypes.util.find_library("X11")
    if not os.environ.get("DISPLAY") or lib is None:
        return False

    x11 = ctypes.CDLL(lib)
    x11.XOpenDisplay.restype = ctypes.c_void_p
    x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
    x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
    handle = x11.XOpenDisplay(None)
    if handle:
        x11.XCloseDisplay(handle)

    return bool(handle)

def run(cfg: dict, run_dir: str) -> Path | None:

    v = cfg["viewer"]
    run_dir = Path(run_dir)
    geoms = []
    cloud = load_cloud(run_dir, v["cloud"])
    if cloud is not None:
        geoms.append(cloud)

    cams = cameras(run_dir)
    if cams is not None:
        points = np.asarray(cloud.points if hasattr(cloud, "points") else cloud.vertices) if cloud is not None else cams[3]
        size = v["frustum_cm"] * stage_io.units_per_cm(run_dir, points, cfg["scale"]["provisional_extent_cm"])
        geoms += camera_geometries(run_dir, cams, size)

    if not geoms:
        log.warning("viewer: nothing to show in %s", run_dir)
        return None

    if v["record"]:
        return record(geoms, v, run_dir, up_axis(run_dir, cams))

    if not has_display():
        log.warning("viewer: no usable display (DISPLAY=%s); set viewer.record: true (offscreen) instead",
                    os.environ.get("DISPLAY"))
        return None

    o3d.visualization.draw_geometries(geoms, window_name=f"{run_dir.name} - {v['cloud']}",
                                      width=v["width"], height=v["height"])

    return None
# <<< END IMPLEMENTATION <<<

if __name__ == "__main__":
    import tempfile
    logging.basicConfig(level=logging.INFO)
    rng = np.random.default_rng(0)
    base = {"cloud": "sparse", "record": True, "seconds": 2, "fps": 10, "width": 320, "height": 240,
            "frustum_cm": 20, "point_size": 2.0, "fov_deg": 60.0, "orbit_radius_factor": 1.5,
            "orbit_elevation_deg": 30.0, "background": [1.0, 1.0, 1.0]}

    with tempfile.TemporaryDirectory() as tmp:
        rd = Path(tmp)
        (rd / "sfm").mkdir()
        (rd / "scale").mkdir()
        pts = rng.uniform(-100, 100, (2000, 3))
        stage_io.write_ply(rd / "exports/colmap/sparse.ply", pts / 50.0, rng.integers(0, 255, (2000, 3)))
        np.save(rd / "scale/scale.npy", np.array([50.0]))
        K = np.array([[300.0, 0, 160], [0, 300, 120], [0, 0, 1]])
        R = np.stack([cv2.Rodrigues(np.array([0.0, a, 0.0]))[0] for a in np.linspace(0, 1, 5)])
        C = np.stack([np.linspace(-1, 1, 5), np.zeros(5), np.zeros(5)], axis=1)
        t = -np.einsum("nij,nj->ni", R, C)
        for name, arr in (("K", K), ("cam_R", R), ("cam_t", t)):
            np.save(rd / "sfm" / f"{name}.npy", arr)
        np.save(rd / "scale/centers_cm.npy", C * 50.0)

        assert cloud_path(rd, "sparse") == (rd / "exports/colmap/sparse.ply", True)
        g = load_cloud(rd, "sparse")
        assert np.abs(np.asarray(g.points)).max() > 50  # scaled back to cm
        cams = cameras(rd)
        assert np.allclose(-np.einsum("nji,nj->ni", cams[1], cams[2]), cams[3])
        ext = look_at(np.array([0.0, 0, -5]), np.zeros(3), np.array([0.0, -1, 0]))
        assert np.allclose(ext[:3, :3], np.eye(3)) and np.allclose(ext[:3, 3], [0, 0, 5])

        for offscreen in (False, True):  # hidden window first: after an OffscreenRenderer it may fail
            cfg = {"viewer": {**base, "offscreen": offscreen}, "scale": {"provisional_extent_cm": 500}}
            out = run(cfg, tmp)
            assert out == rd / "videos/navigation_sparse_2s.mp4" and out.exists()
            cap = cv2.VideoCapture(str(out))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            ok, frame = cap.read()
            cap.release()
            print(f"offscreen={offscreen}: {n} frames, shape {frame.shape}, mean {frame.mean():.1f}")
            assert ok and n == 20 and frame.shape == (240, 320, 3) and frame.std() > 1
        assert run({"viewer": {**base, "record": False}, "scale": {"provisional_extent_cm": 500}}, tmp) is None
        print("has_display:", has_display())
    print("OK")

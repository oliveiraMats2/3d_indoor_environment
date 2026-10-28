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
records one orbit video per cloud and two still figures of the sparse cloud with the camera poses."""
# PROMPT COMPLEMENT
# Stage: viewer (last 3D stage of main.py, before videos: run_safe records the orbit video in a child
#        process; `python view.py --config <yaml>` opens the interactive window on a finished run).
# Goal : show the reconstruction (cloud or mesh) + camera trajectory + frustums; main.py renders an
#        orbit around the scene centre into an mp4 per cloud without user interaction, plus two still
#        figures of the sparse cloud with the camera poses (T2 E2 / 4.A.1).
# In   : each cloud of viewer.clouds (list; empty/null -> [viewer.cloud]):
#        sparse: clean/sparse/cloud_clean.ply (fallback exports/colmap/sparse.ply);
#        mesh: exports/meshlab/mesh.ply; any other name = a dense method: clean/<m>/cloud_clean.ply
#        (fallback dense/<m>/fused.ply). scale/centers_cm.npy, sfm/{K,cam_R,cam_t}.npy,
#        planes/{T_floor,ceiling}.npy (optional: up axis / floor frame, ceiling crop of the top view).
# Out  : <run_dir>/videos/navigation_<cloud>_<seconds>s.mp4 per cloud (videos.codec through
#        videos.open_writer); viewer.figures -> <run_dir>/figures/sparse_poses.png (3/4 view) and
#        figures/sparse_poses_top.png (top view in the floor frame, scale bar in cm when metric).
#        No CSV (videos.csv belongs to the videos stage; main.py rewrites it after this stage).
# YAML : viewer.{enabled, cloud, clouds, figures, seconds, fps, offscreen (auto|true|false), timeout_s,
#        width, height, frustum_cm, point_size, fov_deg, orbit_radius_factor, orbit_elevation_deg,
#        background}; videos.codec; scale.provisional_extent_cm; run.log_level.
# Rules: clean clouds and mesh are in working units (frame of scale/centers_cm.npy); raw sparse.ply and
#        fused.ply are in model units -> multiplied by s when stage_io.scale_info says unit=cm.
#        Frustum size = frustum_cm * stage_io.units_per_cm. offscreen true -> rendering.OffscreenRenderer
#        (EGL on Linux, no display needed); false -> hidden legacy Visualizer (Open3D falls back to EGL
#        without a display); auto -> hidden Visualizer when has_display(), else OffscreenRenderer. A blank
#        first frame (std < 1) switches to the other renderer once. OffscreenRenderer always returns a
#        black first image on a fresh scene, and ~2% black frames later -> each re-rendered up to
#        WARMUP_RENDERS times (then the previous frame is repeated). Orbit centre/radius = 2nd-98th
#        percentile box of the cloud/mesh (not the median: pulled onto the floor), near/far from the orbit
#        distance, at most MAX_FRUSTUMS frustums, coloured mesh drawn unlit. Recording runs in a "spawn"
#        child (run_safe) so a native renderer abort (no EGL, no macOS WindowServer) cannot kill main.py.
#        Linux + Open3D 0.20 over SSH (DISPLAY=localhost:10.0): GLFW cannot open an X11 window; both
#        OffscreenRenderer and the hidden Visualizer (EGL fallback) render fine.
#        Figures: OffscreenRenderer (EGL, GPU when available), perspective 3/4 view and orthographic top
#        view (exact pixels per unit -> scale bar); blank image, exception or a crashed child -> matplotlib
#        (Agg) fallback in the floor frame. Top view drops points above 95% of the ceiling height (from
#        planes/ceiling.npy) so the ceiling does not hide the room. Each video and the figures run in their
#        own spawn child, each with viewer.timeout_s.
# >>> BEGIN IMPLEMENTATION >>>
import ctypes, ctypes.util, logging, multiprocessing, os, re, sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm

from src import stage_io, videos

log = logging.getLogger(__name__)

TRAJECTORY_RGB = (1.0, 0.0, 0.0)
FRUSTUM_RGB = (0.0, 0.3, 1.0)
MAX_FRUSTUMS = 60      # frustums drawn (evenly spaced along the trajectory): 800+ merge into an opaque band
WARMUP_RENDERS = 5     # OffscreenRenderer re-renders of a black image (first of a fresh scene, ~2% of the rest)
FIGURE_AZIMUTH_DEG = 45.0    # 3/4 view of figures/sparse_poses.png: azimuth on the orbit ring, elevation 35 deg
FIGURE_ELEVATION_DEG = 35.0
FIGURE_LINE_WIDTH = 2.0      # trajectory/frustum width in the still figures (videos keep 1 px)
CEILING_KEEP_FRAC = 0.95     # top view: keep points below this fraction of the ceiling height
MPL_MAX_POINTS = 150_000     # matplotlib fallback: random subsample of the cloud
DEFAULTS = {"clouds": None, "figures": True}  # keys newer than the YAML schema

def settings(cfg: dict) -> dict:

    return {**DEFAULTS, **cfg["viewer"]}

def cloud_names(v: dict) -> list[str]:
    """viewer.clouds (order kept, duplicates dropped), else [viewer.cloud]."""

    names = v.get("clouds") or [v["cloud"]]

    return list(dict.fromkeys(str(n) for n in names))

def cloud_path(run_dir: Path, cloud: str) -> tuple[Path | None, bool]:
    """(path, in_model_units) of the first existing candidate for a cloud name: sparse, mesh or any dense
    method written under dense/<method>/."""

    if not re.fullmatch(r"[A-Za-z0-9_\-]+", cloud):
        log.warning("viewer: invalid cloud name %r", cloud)
        return None, False

    candidates = {
        "sparse": [("clean/sparse/cloud_clean.ply", False), ("exports/colmap/sparse.ply", True)],
        "mesh": [("exports/meshlab/mesh.ply", False)],
    }.get(cloud, [(f"clean/{cloud}/cloud_clean.ply", False), (f"dense/{cloud}/fused.ply", True)])

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

    step = int(np.ceil(len(R) / MAX_FRUSTUMS))
    for Ri, ti in zip(R[::step], t[::step]):
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
    """(centre, radius) of the 2nd-98th percentile box of the cloud/mesh vertices (the line sets only when
    there is no cloud): far outliers do not move the orbit, and unlike the median the centre is not pulled
    onto the floor / the densest wall."""

    solid = [g for g in geoms if not isinstance(g, o3d.geometry.LineSet)] or geoms
    pts = [np.asarray(g.points if hasattr(g, "points") else g.vertices) for g in solid]
    pts = np.concatenate([p for p in pts if len(p)])
    lo, hi = np.percentile(pts, [2, 98], axis=0)

    return (lo + hi) / 2, float(np.linalg.norm(hi - lo)) / 2 or 1.0

def clip_planes(eyes: np.ndarray, centre: np.ndarray) -> tuple[float, float]:
    """(near, far) from the orbit distance d (= orbit_radius_factor * scene radius): everything within
    ~4 scene radii of the eye is kept, far outliers are clipped, the depth buffer keeps its precision."""

    d = float(np.linalg.norm(eyes[0] - centre)) or 1.0

    return 0.01 * d, 4.0 * d

def add_geometries(scene, geoms: list, v: dict, line_width: float = 1.0) -> None:
    """Add geoms to an Open3D rendering scene as g0, g1, ... (lines unlit, coloured clouds/meshes unlit)."""

    for i, g in enumerate(geoms):
        mat = o3d.visualization.rendering.MaterialRecord()
        is_lines = isinstance(g, o3d.geometry.LineSet)
        is_mesh = isinstance(g, o3d.geometry.TriangleMesh)
        lit = is_mesh and not g.has_vertex_colors()  # coloured mesh: unlit, true colours (lit is too dark)
        mat.shader = "unlitLine" if is_lines else "defaultLit" if lit else "defaultUnlit"
        mat.point_size = float(v["point_size"])
        mat.line_width = float(line_width)
        scene.add_geometry(f"g{i}", g, mat)

def offscreen_renderer(geoms: list, v: dict, line_width: float = 1.0):

    renderer = o3d.visualization.rendering.OffscreenRenderer(v["width"], v["height"])
    renderer.scene.set_background([*v["background"], 1.0])
    renderer.scene.view.set_post_processing(False)  # no tone mapping: true colours and background
    add_geometries(renderer.scene, geoms, v, line_width)

    return renderer

def render_image(renderer) -> np.ndarray:
    """RGB image; a black one (first of a fresh scene, ~2% of later ones) is re-rendered up to
    WARMUP_RENDERS times."""

    frame = np.asarray(renderer.render_to_image())
    for _ in range(WARMUP_RENDERS):
        if float(frame.std()) >= 1.0:
            break
        frame = np.asarray(renderer.render_to_image())

    return frame

def render_offscreen(geoms: list, v: dict, eyes: np.ndarray, centre: np.ndarray, up: np.ndarray):

    renderer = offscreen_renderer(geoms, v)

    near, far = clip_planes(eyes, centre)
    cam = renderer.scene.camera
    vertical = o3d.visualization.rendering.Camera.FovType.Vertical
    last = None
    for eye in eyes:
        renderer.setup_camera(float(v["fov_deg"]), centre, eye, up)
        cam.set_projection(float(v["fov_deg"]), v["width"] / v["height"], near, far, vertical)
        frame = render_image(renderer)
        if float(frame.std()) >= 1.0:
            last = frame
        elif last is not None:  # still black after the retries: repeat the previous frame
            frame = last
        yield cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

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
    near, far = clip_planes(eyes, centre)
    ctr.set_constant_z_near(near)
    ctr.set_constant_z_far(far)
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

def video_out(run_dir: Path, v: dict, cloud: str | None = None) -> Path:

    return Path(run_dir) / "videos" / f"navigation_{cloud or cloud_names(v)[0]}_{v['seconds']}s.mp4"

def use_offscreen(v: dict) -> bool:
    """viewer.offscreen auto -> OffscreenRenderer only when there is no usable display."""

    return (not has_display()) if v["offscreen"] == "auto" else bool(v["offscreen"])

def first_frame(frames):
    """First frame of a renderer generator, None when it is blank (std < 1) or the renderer raised."""

    try:
        frame = next(frames, None)
    except Exception as exc:  # noqa: BLE001 - any renderer failure: try the other one
        log.warning("viewer: renderer failed (%s)", exc)
        return None

    return frame if frame is not None and float(frame.std()) >= 1.0 else None

def record(geoms: list, v: dict, run_dir: Path, up: np.ndarray, codec: str = "auto",
           cloud: str | None = None) -> Path | None:

    centre, radius = scene_frame(geoms)
    n = int(round(v["seconds"] * v["fps"]))
    eyes = orbit_eyes(centre, radius * v["orbit_radius_factor"], up, v["orbit_elevation_deg"], n)
    out = video_out(run_dir, v, cloud)
    offscreen = use_offscreen(v)
    renderers = [render_offscreen, render_hidden] if offscreen else [render_hidden, render_offscreen]
    log.info("viewer: offscreen=%s -> %s", v["offscreen"], renderers[0].__name__)
    frames = renderers[0](geoms, v, eyes, centre, up)
    first = first_frame(frames)
    if first is None:
        log.warning("viewer: %s gave no usable first frame, switching to %s", renderers[0].__name__,
                    renderers[1].__name__)
        frames.close()
        frames = renderers[1](geoms, v, eyes, centre, up)
        first = first_frame(frames)
        if first is None:
            log.warning("viewer: both renderers gave a blank frame, no video")
            return None

    writer = videos.open_writer(out, v["fps"], (v["width"], v["height"]), codec)
    if writer is None:
        return None

    writer.write(first)
    for frame in tqdm(frames, total=n - 1, desc="viewer orbit"):
        writer.write(frame)
    writer.release()
    log.info("viewer: %d frames -> %s", n, out)

    return out

def has_display() -> bool:
    """macOS: True unless in an SSH session (no WindowServer access). Linux: True when a Wayland socket is
    announced or XOpenDisplay succeeds (Open3D silently goes headless otherwise and an interactive window
    would block forever, invisible)."""

    if sys.platform == "darwin":
        return not os.environ.get("SSH_CONNECTION")

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

def scene(cfg: dict, run_dir: Path, cloud: str) -> tuple[list, tuple | None]:
    """(geoms = [cloud?, trajectory, frustums...], cams) of one cloud name."""

    v = settings(cfg)
    geoms = []
    geom = load_cloud(run_dir, cloud)
    if geom is not None:
        geoms.append(geom)

    cams = cameras(run_dir)
    if cams is not None:
        points = np.asarray(geom.points if hasattr(geom, "points") else geom.vertices) if geom is not None else cams[3]
        size = v["frustum_cm"] * stage_io.units_per_cm(run_dir, points, cfg["scale"]["provisional_extent_cm"])
        geoms += camera_geometries(run_dir, cams, size)

    return geoms, cams

def run(cfg: dict, run_dir: str, interactive: bool = False, cloud: str | None = None) -> Path | None:
    """interactive False (main.py): record the orbit video of `cloud` (default: first of viewer.clouds);
    True (view.py): Open3D window."""

    v = settings(cfg)
    run_dir = Path(run_dir)
    cloud = cloud or cloud_names(v)[0]
    geoms, cams = scene(cfg, run_dir, cloud)
    if not geoms:
        log.warning("viewer: nothing to show in %s", run_dir)
        return None

    if not interactive:
        return record(geoms, v, run_dir, up_axis(run_dir, cams), cfg.get("videos", {}).get("codec", "auto"), cloud)

    if not has_display():
        log.warning("viewer: no usable display (DISPLAY=%s); main.py records the orbit video instead",
                    os.environ.get("DISPLAY"))
        return None

    o3d.visualization.draw_geometries(geoms, window_name=f"{run_dir.name} - {cloud}",
                                      width=v["width"], height=v["height"])

    return None

def floor_transform(run_dir: Path, cams) -> np.ndarray:
    """4x4 world -> floor frame (z up): planes/T_floor.npy, else a rotation taking up_axis to +z."""

    t_floor = run_dir / "planes" / "T_floor.npy"
    if t_floor.exists():
        return np.load(t_floor)

    up = up_axis(run_dir, cams)
    e1 = np.cross(up, [1.0, 0.0, 0.0])
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(up, [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    T = np.eye(4)
    T[:3, :3] = np.stack([e1, np.cross(up, e1), up])

    return T

def to_frame(T: np.ndarray, pts: np.ndarray) -> np.ndarray:

    return np.asarray(pts, float).reshape(-1, 3) @ T[:3, :3].T + T[:3, 3]

def ceiling_z(run_dir: Path, T: np.ndarray) -> float | None:
    """Ceiling height in the floor frame from planes/ceiling.npy (a, b, c, d), None when unknown."""

    path = run_dir / "planes" / "ceiling.npy"
    if not path.exists():
        return None

    plane = np.load(path).ravel()
    if len(plane) != 4 or not np.all(np.isfinite(plane)):
        return None

    z = float(to_frame(T, -plane[3] * plane[:3] / np.linalg.norm(plane[:3]))[0, 2])

    return z if z > 0 else None

def nice_length(span: float) -> float:
    """Largest 1/2/5 x 10^k not above a fifth of span (scale bar length)."""

    target = span / 5
    base = 10 ** np.floor(np.log10(target))

    return float(max(m * base for m in (1, 2, 5) if m * base <= target))

def scale_label(length: float, metric: bool) -> str:

    return f"{length:g} cm" if metric else f"{length:.3g} model units (no metric scale)"

def split_geoms(geoms: list):
    """(cloud points, cloud colours or None, [(segments (S,2,3), rgb)]) of a scene."""

    pts, cols, segs = np.zeros((0, 3)), None, []
    for g in geoms:
        if isinstance(g, o3d.geometry.LineSet):
            p, l = np.asarray(g.points), np.asarray(g.lines)
            rgb = tuple(np.asarray(g.colors)[0]) if g.has_colors() and len(l) else FRUSTUM_RGB
            segs.append((p[l], rgb))
        elif hasattr(g, "points"):
            pts = np.asarray(g.points)
            cols = np.asarray(g.colors) if g.has_colors() else None
        else:
            pts = np.asarray(g.vertices)
            cols = np.asarray(g.vertex_colors) if g.has_vertex_colors() else None

    return pts, cols, segs

def top_bounds(xy: np.ndarray, traj_xy: np.ndarray, w: int, h: int) -> tuple[float, float, float, float]:
    """(cx, cy, half_width, half_height) of the top view: 1-99th percentile of the cloud plus the whole
    trajectory, 5% margin, aspect of the image."""

    box = [np.percentile(xy, [1, 99], axis=0)] if len(xy) else []
    if len(traj_xy):
        box.append(np.stack([traj_xy.min(0), traj_xy.max(0)]))
    box = np.concatenate(box)
    lo, hi = box.min(0), box.max(0)
    (cx, cy), (hw, hh) = (lo + hi) / 2, np.maximum((hi - lo) / 2 * 1.05, 1e-6)
    hw, hh = max(hw, hh * w / h), max(hh, hw * h / w)

    return float(cx), float(cy), float(hw), float(hh)

def draw_scale_bar(img: np.ndarray, length_px: float, label: str, colour=(0, 0, 0)) -> None:

    h = img.shape[0]
    x0, y0 = 30, h - 30
    x1 = int(round(x0 + length_px))
    cv2.line(img, (x0, y0), (x1, y0), colour, 4)
    for x in (x0, x1):
        cv2.line(img, (x, y0 - 8), (x, y0 + 8), colour, 2)
    cv2.putText(img, label, (x0, y0 - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2, cv2.LINE_AA)

def draw_caption(img: np.ndarray, text: str, colour=(0, 0, 0)) -> None:

    cv2.putText(img, text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, colour, 2, cv2.LINE_AA)

def figures_open3d(geoms: list, v: dict, T: np.ndarray, zc: float | None, metric: bool,
                   caption: str) -> tuple[np.ndarray, np.ndarray] | None:
    """(3/4 view, top view) BGR images from the OffscreenRenderer (EGL), None when a render is blank."""

    w, h = v["width"], v["height"]
    rendering = o3d.visualization.rendering
    renderer = offscreen_renderer(geoms, v, FIGURE_LINE_WIDTH)
    cam = renderer.scene.camera
    up = T[2, :3] / np.linalg.norm(T[2, :3])

    # 3/4 view: one eye of the orbit ring, perspective
    centre, radius = scene_frame(geoms)
    n = int(round(360 / FIGURE_AZIMUTH_DEG))
    eye = orbit_eyes(centre, radius * v["orbit_radius_factor"], up, FIGURE_ELEVATION_DEG, n)[1]
    near, far = clip_planes(eye[None], centre)
    renderer.setup_camera(float(v["fov_deg"]), centre, eye, up)
    cam.set_projection(float(v["fov_deg"]), w / h, near, far, rendering.Camera.FovType.Vertical)
    three_quarter = render_image(renderer)
    if float(three_quarter.std()) < 1.0:
        return None

    # top view: orthographic along -z of the floor frame (exact pixels per unit), ceiling cropped; only the
    # cloud is rendered, trajectory and frustums are drawn over it in 2D (else hidden under walls/furniture)
    pts, cols, segs = split_geoms(geoms)
    fpts = to_frame(T, pts)
    has_cloud = not isinstance(geoms[0], o3d.geometry.LineSet)
    for i in range(1 if has_cloud else 0, len(geoms)):
        renderer.scene.remove_geometry(f"g{i}")
    if zc is not None and len(fpts) and has_cloud:
        keep = fpts[:, 2] < CEILING_KEEP_FRAC * zc
        renderer.scene.remove_geometry("g0")
        top_cloud = geoms[0].select_by_index(np.flatnonzero(keep)) if hasattr(geoms[0], "points") else geoms[0]
        add_geometries(renderer.scene, [top_cloud], v, FIGURE_LINE_WIDTH)
        fpts = fpts[keep]
    traj = to_frame(T, np.concatenate([s.reshape(-1, 3) for s, rgb in segs if rgb == TRAJECTORY_RGB] or [np.zeros((0, 3))]))
    cx, cy, hw, hh = top_bounds(fpts[:, :2], traj[:, :2], w, h)
    allz = np.concatenate([fpts[:, 2], traj[:, 2], [0.0]])
    depth = float(allz.max() - allz.min()) + 2 * max(hw, hh)
    Rinv = T[:3, :3].T
    centre_w = Rinv @ (np.array([cx, cy, float(allz.min())]) - T[:3, 3])
    eye_w = centre_w + up * depth
    cam.look_at(centre_w, eye_w, T[1, :3])
    cam.set_projection(rendering.Camera.Projection.Ortho, -hw, hw, -hh, hh, 1e-3 * depth, 2 * depth)
    top = render_image(renderer) if has_cloud else np.full((h, w, 3), np.uint8(255 * np.asarray(v["background"])))
    if float(top.std()) < 1.0 and has_cloud:
        return None

    three_quarter = cv2.cvtColor(three_quarter, cv2.COLOR_RGB2BGR)
    top = np.ascontiguousarray(cv2.cvtColor(top, cv2.COLOR_RGB2BGR))

    def px(p):  # floor frame xy -> top view pixel
        return int(round((p[0] - cx + hw) / (2 * hw) * w)), int(round((cy + hh - p[1]) / (2 * hh) * h))

    for s, rgb in sorted(segs, key=lambda sr: sr[1] == TRAJECTORY_RGB):  # trajectory drawn last
        bgr = tuple(int(255 * c) for c in rgb[::-1])
        for a, b in to_frame(T, s.reshape(-1, 3)).reshape(-1, 2, 3):
            cv2.line(top, px(a), px(b), bgr, 2 if rgb == TRAJECTORY_RGB else 1, cv2.LINE_AA)
    if len(traj):
        centres = to_frame(T, geoms_centres(segs))
        cv2.circle(top, px(centres[0]), 9, (0, 160, 0), -1)   # start: green
        cv2.circle(top, px(centres[-1]), 9, (0, 0, 0), 2)     # end: black ring
    length = nice_length(2 * hw)
    draw_scale_bar(top, length / (2 * hw) * w, scale_label(length, metric))
    draw_caption(three_quarter, caption)
    draw_caption(top, caption + " - top view, floor frame (start green, end black)")

    return three_quarter, top

def geoms_centres(segs: list) -> np.ndarray:
    """Camera centres in trajectory order from the trajectory segments (i -> i+1)."""

    traj = [s for s, rgb in segs if rgb == TRAJECTORY_RGB]
    if not traj:
        return np.zeros((0, 3))

    s = traj[0]

    return np.vstack([s[:, 0], s[-1:, 1]])

def figures_matplotlib(geoms: list, v: dict, T: np.ndarray, zc: float | None, metric: bool,
                       caption: str) -> tuple[np.ndarray, np.ndarray]:
    """Fallback (no renderer): (3/4 view, top view) BGR images drawn with matplotlib Agg in the floor frame."""

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.collections import LineCollection
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    w, h = v["width"], v["height"]
    bg = tuple(v["background"])
    pts, cols, segs = split_geoms(geoms)
    fpts = to_frame(T, pts)
    cols = np.full((len(fpts), 3), 0.4) if cols is None else cols
    rng = np.random.default_rng(0)
    sub = rng.choice(len(fpts), MPL_MAX_POINTS, replace=False) if len(fpts) > MPL_MAX_POINTS else slice(None)
    fpts, cols = fpts[sub], cols[sub]
    fsegs = [(to_frame(T, s.reshape(-1, 3)).reshape(-1, 2, 3), rgb) for s, rgb in segs]
    centres = to_frame(T, geoms_centres(segs))
    unit = "cm" if metric else "model units"

    def image(fig) -> np.ndarray:
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba())
        return cv2.resize(cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR), (w, h))

    # 3/4 view
    fig = Figure(figsize=(w / 100, h / 100), dpi=100, facecolor=bg)
    ax = fig.add_subplot(projection="3d", facecolor=bg)
    ax.scatter(*fpts.T, c=cols, s=0.5, depthshade=False, linewidths=0)
    for s, rgb in fsegs:
        ax.add_collection3d(Line3DCollection(s, colors=[rgb], linewidths=1.5 if rgb == TRAJECTORY_RGB else 0.6))
    box = np.concatenate([np.percentile(fpts, [2, 98], axis=0) if len(fpts) else np.zeros((0, 3)), centres])
    lo, hi = box.min(0), box.max(0)
    ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(lo[2], hi[2]),
           xlabel=f"x ({unit})", ylabel=f"y ({unit})", zlabel=f"z ({unit})")
    ax.set_box_aspect(np.maximum(hi - lo, 1e-6))
    ax.view_init(elev=FIGURE_ELEVATION_DEG, azim=FIGURE_AZIMUTH_DEG - 90)
    ax.set_title(caption)
    three_quarter = image(fig)

    # top view
    if zc is not None:
        keep = fpts[:, 2] < CEILING_KEEP_FRAC * zc
        fpts, cols = fpts[keep], cols[keep]
    order = np.argsort(fpts[:, 2])  # highest points drawn last, as seen from above
    cx, cy, hw, hh = top_bounds(fpts[:, :2], centres[:, :2], w, h)
    fig = Figure(figsize=(w / 100, h / 100), dpi=100, facecolor=bg)
    ax = fig.add_axes([0.06, 0.08, 0.92, 0.86], facecolor=bg)
    ax.scatter(fpts[order, 0], fpts[order, 1], c=cols[order], s=0.5, linewidths=0)
    for s, rgb in fsegs:
        ax.add_collection(LineCollection(s[:, :, :2], colors=[rgb], linewidths=1.5 if rgb == TRAJECTORY_RGB else 0.6))
    if len(centres):
        ax.plot(*centres[0, :2], "o", color=(0, 0.63, 0), ms=8)
        ax.plot(*centres[-1, :2], "o", mfc="none", mec="k", ms=8)
    ax.set(xlim=(cx - hw, cx + hw), ylim=(cy - hh, cy + hh), aspect="equal",
           xlabel=f"x ({unit})", ylabel=f"y ({unit})", title=caption + " - top view, floor frame")
    length = nice_length(2 * hw)
    x0, y0 = cx - hw + 0.05 * hw, cy - hh + 0.08 * hh
    ax.plot([x0, x0 + length], [y0, y0], "k-", lw=3)
    ax.text(x0, y0 + 0.03 * hh, scale_label(length, metric), fontsize=11)
    top = image(fig)

    return three_quarter, top

def figures(cfg: dict, run_dir: str, renderer: str = "auto", out_dir: str | None = None) -> list[Path]:
    """figures/sparse_poses.png (3/4 view) and figures/sparse_poses_top.png (top view in the floor frame,
    scale bar in cm when metric) of the sparse cloud + trajectory + frustums. renderer: auto (Open3D
    OffscreenRenderer, matplotlib fallback) | matplotlib."""

    v = settings(cfg)
    run_dir = Path(run_dir)
    geoms, cams = scene(cfg, run_dir, "sparse")
    if not geoms:
        log.warning("viewer: no sparse cloud nor cameras in %s, no figures", run_dir)
        return []

    T = floor_transform(run_dir, cams)
    zc = ceiling_z(run_dir, T)
    metric = stage_io.scale_info(run_dir)[1] == "cm"
    n_cams = 0 if cams is None else len(cams[1])
    caption = f"sparse cloud + {n_cams} camera poses"
    images, used = None, "matplotlib"
    if renderer != "matplotlib":
        try:
            images, used = figures_open3d(geoms, v, T, zc, metric, caption), "open3d"
        except Exception as exc:  # noqa: BLE001 - any renderer failure: matplotlib
            log.warning("viewer: OffscreenRenderer figures failed (%s)", exc)
        if images is None:
            log.warning("viewer: OffscreenRenderer gave a blank figure, matplotlib fallback")
            used = "matplotlib"
    if images is None:
        images = figures_matplotlib(geoms, v, T, zc, metric, caption)

    out = Path(out_dir) if out_dir else run_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / "sparse_poses.png", out / "sparse_poses_top.png"]
    for path, img in zip(paths, images):
        cv2.imwrite(str(path), img)
    log.info("viewer: figures (%s) -> %s", used, ", ".join(str(p) for p in paths))

    return paths

def _record_child(cfg: dict, run_dir: str, job: str = "video", arg: str | None = None) -> None:
    """Child process body of run_safe: its own logging (stdout + run.log), exit 0 even without output.
    job "video" -> orbit of cloud `arg`; "figures" -> still figures with renderer `arg`."""

    handlers = [logging.StreamHandler(sys.stdout)]
    if (Path(run_dir) / "run.log").exists():
        handlers.append(logging.FileHandler(Path(run_dir) / "run.log"))
    logging.basicConfig(level=cfg.get("run", {}).get("log_level", "INFO"), handlers=handlers,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    if job == "figures":
        figures(cfg, run_dir, renderer=arg or "auto")
    else:
        run(cfg, run_dir, cloud=arg)

def _spawn(cfg: dict, run_dir: str, job: str, arg: str | None, timeout_s: float) -> int | None:
    """Run _record_child in a "spawn" child; exit code (None or != 0 = crashed / killed)."""

    child = multiprocessing.get_context("spawn").Process(target=_record_child, args=(cfg, str(run_dir), job, arg))
    child.start()
    child.join(timeout_s)
    if child.is_alive():
        log.warning("viewer: %s %s still running after %s s (viewer.timeout_s): killed", job, arg, timeout_s)
        child.kill()
        child.join()
        return None

    return child.exitcode

def run_safe(cfg: dict, run_dir: str) -> Path | None:
    """main.py stage: one "spawn" child per cloud of viewer.clouds (orbit video) and one for the figures,
    so a native Open3D abort (no EGL / no WindowServer) or a hang (viewer.timeout_s) only costs that
    output. Returns the first video written, or None."""

    v = settings(cfg)
    if not v["enabled"]:
        log.info("viewer disabled")
        return None

    videos_written = []
    for cloud in cloud_names(v):
        out = video_out(Path(run_dir), v, cloud)
        out.unlink(missing_ok=True)
        if cloud_path(Path(run_dir), cloud)[0] is None:
            log.warning("viewer: no file for cloud=%s, no video", cloud)
            continue
        code = _spawn(cfg, run_dir, "video", cloud, v["timeout_s"])
        if code != 0:
            log.warning("viewer: recording of %s exited with code %s (renderer unavailable?): no video", cloud, code)
        elif not out.exists():
            log.warning("viewer: no video written (%s)", out)
        else:
            log.info("viewer: video %s", out)
            videos_written.append(out)

    if v["figures"]:
        code = _spawn(cfg, run_dir, "figures", "auto", v["timeout_s"])
        if code != 0:  # native renderer abort: matplotlib only
            log.warning("viewer: figures child exited with code %s, retrying with matplotlib", code)
            _spawn(cfg, run_dir, "figures", "matplotlib", v["timeout_s"])

    return videos_written[0] if videos_written else None
# <<< END IMPLEMENTATION <<<

if __name__ == "__main__":
    import tempfile
    logging.basicConfig(level=logging.INFO)
    rng = np.random.default_rng(0)
    base = {"enabled": True, "cloud": "sparse", "timeout_s": 300, "seconds": 2, "fps": 10, "width": 320, "height": 240,
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
        stage_io.write_ply(rd / "dense/my_mvs/fused.ply", pts / 50.0, rng.integers(0, 255, (2000, 3)))
        assert cloud_path(rd, "my_mvs") == (rd / "dense/my_mvs/fused.ply", True)  # any dense method
        assert cloud_path(rd, "openmvs") == (None, False) and cloud_path(rd, "../x") == (None, False)
        assert cloud_names({"cloud": "mesh", "clouds": None}) == ["mesh"]
        assert cloud_names({"cloud": "mesh", "clouds": ["sparse", "my_mvs", "sparse"]}) == ["sparse", "my_mvs"]
        assert nice_length(1000) == 200 and nice_length(37) == 5 and scale_label(50, True) == "50 cm"
        g = load_cloud(rd, "sparse")
        assert np.abs(np.asarray(g.points)).max() > 50  # scaled back to cm
        cams = cameras(rd)
        assert np.allclose(-np.einsum("nji,nj->ni", cams[1], cams[2]), cams[3])
        ext = look_at(np.array([0.0, 0, -5]), np.zeros(3), np.array([0.0, -1, 0]))
        assert np.allclose(ext[:3, :3], np.eye(3)) and np.allclose(ext[:3, 3], [0, 0, 5])
        assert first_frame(iter([np.zeros((4, 4, 3), np.uint8)])) is None and first_frame(iter([])) is None
        assert first_frame(iter([rng.integers(0, 255, (4, 4, 3)).astype(np.uint8)])) is not None
        far = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack([pts, [[1e5, 1e5, 1e5]]])))
        c0, r0 = scene_frame([far, *camera_geometries(rd, cams, 1e4)])  # one far outlier, huge frustums
        assert np.abs(c0).max() < 20 and r0 < 200, (c0, r0)
        many = (cams[0], np.repeat(cams[1], 40, 0), np.repeat(cams[2], 40, 0), np.repeat(cams[3], 40, 0))
        assert MAX_FRUSTUMS // 2 < len(camera_geometries(rd, many, 1.0)) - 1 <= MAX_FRUSTUMS  # capped frustums

        for offscreen in (False, True):  # hidden window first: after an OffscreenRenderer it may fail
            cfg = {"viewer": {**base, "offscreen": offscreen}, "scale": {"provisional_extent_cm": 500},
                   "videos": {"codec": "auto"}}
            out = run(cfg, tmp)
            assert out == rd / "videos/navigation_sparse_2s.mp4" and out.exists()
            cap = cv2.VideoCapture(str(out))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frames = [f for ok, f in iter(cap.read, (False, None)) if ok]
            cap.release()
            frame = frames[0]
            print(f"offscreen={offscreen}: {n} frames, shape {frame.shape}, mean {frame.mean():.1f}")
            assert n == len(frames) == 20 and frame.shape == (240, 320, 3)
            assert min(float(f.std()) for f in frames) > 1  # no black frame anywhere in the orbit
        cfg = {"viewer": {**base, "offscreen": "auto", "clouds": ["sparse", "my_mvs", "openmvs"]},
               "scale": {"provisional_extent_cm": 500}, "run": {"log_level": "INFO"}}
        out = run_safe(cfg, tmp)  # spawn children, as main.py calls it (openmvs missing: skipped)
        assert out == rd / "videos/navigation_sparse_2s.mp4" and out.exists()
        assert (rd / "videos/navigation_my_mvs_2s.mp4").exists()
        assert not (rd / "videos/navigation_openmvs_2s.mp4").exists()
        for name in ("sparse_poses.png", "sparse_poses_top.png"):  # written by the figures child
            img = cv2.imread(str(rd / "figures" / name))
            assert img is not None and img.shape == (240, 320, 3) and img.std() > 1, name
        for renderer in ("matplotlib", "auto"):
            paths = figures(cfg, tmp, renderer=renderer, out_dir=str(rd / f"fig_{renderer}"))
            assert [p.name for p in paths] == ["sparse_poses.png", "sparse_poses_top.png"]
            assert all(cv2.imread(str(p)).std() > 1 for p in paths)
        print("figures OK")
        print("run_safe (offscreen=auto ->", "offscreen" if use_offscreen(cfg["viewer"]) else "hidden window", "): OK")
        assert run_safe({**cfg, "viewer": {**base, "enabled": False}}, tmp) is None
        if has_display() is False:
            assert run({"viewer": base, "scale": {"provisional_extent_cm": 500}}, tmp, interactive=True) is None
        print("has_display:", has_display())
    print("OK")

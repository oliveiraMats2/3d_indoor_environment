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
# Stage: room_fill, after close_holes/liepa, before mesh/export. Optional, off by default: room_fill.enabled.
# Goal: textureless floor/ceiling (walls) are missing from the Poisson mesh. Fit the room outline measured by hand
#   (room_fill.footprint_cm) to the reconstruction, then build flat floor/ceiling (optionally walls) meshes that
#   fill ONLY where the reconstruction has no surface; the fit also yields a metric scale (cm per working unit).
# In:  clean/<room_fill.source>/{points,colors}.npy (working units; missing -> clean/<planes.source>, then
#      clean/sparse, WARNING), planes/{T_floor,floor,ceiling,labels}.npy (missing T_floor or no floor -> WARNING,
#      None), the mesh selected by mesh.input (src.mesh.mesh.pick_input; missing -> fill only, WARNING),
#      scale/scale.npy (stage_io.scale_info, only to compare scales).
# Out: mesh/room_fill/{room_fill.ply (only the filled parts), room_scene.ply (selected mesh + fill),
#      T_room.npy (4,4: room outline frame in cm -> model frame in working units), footprint_cm.npy (fitted
#      outline, CCW, mirrored when the mirrored one won)}, copies exports/meshlab/{room_fill,room_scene}.ply;
#      artifacts room_fill.csv (one row: fit residual median/p90 cm, inlier fraction, rotation, s, cm_per_unit,
#      scale comparison, ceiling height cm, polygon area m2, per surface area/filled m2 and filled fraction,
#      n_vertices, n_faces, t_s), every value also logged as `metric name=value`.
# YAML: room_fill: {enabled, footprint_cm: [[x, y], ...], ceiling_height_cm: null, surfaces: [floor, ceiling],
#       source: patch_match, grid_cm, max_gap_cm, fit_rotations}
# Rules:
# - Floor frame: p' = T_floor p (z up, floor at z = 0). Wall points = planes labels >= 2 when labels.npy is
#   aligned with the source cloud and holds >= MIN_WALL_POINTS, else points between 20% and 80% of the robust
#   (1-99 percentile) height range; at most FIT_POINTS of them (seeded subsample) enter the fit.
# - Fit: 2D similarity q = s R(theta) p + t (s = working units per cm) of the outline p (cm) onto the wall points
#   q; robust point-to-edge Gauss-Newton (Huber weights, residuals above the TRIM quantile ignored), started
#   from fit_rotations angles x {outline, mirrored outline}, initial s from the robust extents (area ratio) and t
#   from the box centres. The best start (trimmed mean distance) is refined; fit_ambiguity = best cost / cost of
#   the best clearly different pose (close to 1 = the outline fits two ways, check the result).
# - Residual = point-to-segment distance in cm (median, p90); inlier = distance < INLIER_CM; cm_per_unit = 1/s.
#   With scale known (unit cm) the working unit is already the cm and cm_per_unit should be ~1
#   (scale_ratio = cm_per_unit); cm_per_model_unit = cm_per_unit * scale.npy (or cm_per_unit when unit model).
# - Ceiling height: ceiling_height_cm when given, else the planes ceiling plane, else the 99th percentile of z of
#   the points inside the fitted outline.
# - Fill: grid_cm cells inside the outline (centre inside) at z = 0 (floor, normal up) and z = h (ceiling, normal
#   down); walls = per edge vertical grid 0..h (cells shrunk to fit the edge and h exactly, normal inward). A cell
#   whose centre lies within max_gap_cm of the source cloud or of the selected mesh vertices is covered and
#   dropped; the others become 2 triangles each (lattice vertices shared). Colour = median colour of the
#   source points of that surface (labels, else points within max_gap_cm of it), else neutral gray.
# - Outputs in the model frame and working units (inverse of T_floor). Nothing to fill -> WARNING, no
#   room_fill.ply, room_scene.ply = selected mesh. Never crashes main.py on missing inputs.
# >>> BEGIN IMPLEMENTATION >>>
"""Room fill: fit the hand-measured room outline and fill the missing floor/ceiling (walls) with flat meshes."""
import logging, shutil, time
from pathlib import Path

import numpy as np
import open3d as o3d

from src import stage_io
from src.mesh import mesh as mesh_stage

log = logging.getLogger(__name__)

SURFACES = ("floor", "ceiling", "walls")
GRAY = np.array([0.5, 0.5, 0.5])   # colour of a surface without source points
MIN_WALL_POINTS = 200              # fewer labelled wall points -> height band fallback
FIT_POINTS = 20000                 # wall points used by the fit (seeded subsample)
TRIM = 0.8                         # residual quantile kept in every Gauss-Newton step
INLIER_CM = 10.0                   # wall point inlier of the fitted outline
COARSE_ITERS, FINE_ITERS = 8, 60   # Gauss-Newton steps per start / for the best start
CHUNK = 500_000                    # points per vectorised polygon test (bounds memory on dense clouds)

def rot2(theta: float) -> np.ndarray:

    c, s = np.cos(theta), np.sin(theta)

    return np.array([[c, -s], [s, c]])

def signed_area(poly: np.ndarray) -> float:

    x, y = poly[:, 0], poly[:, 1]

    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))

def ccw(poly: np.ndarray) -> np.ndarray:
    """Same outline, counter-clockwise (interior on the left of every edge)."""

    return poly if signed_area(poly) >= 0 else poly[::-1].copy()

def inside(poly: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Even-odd point-in-polygon test (vectorised ray casting along +x, in chunks of CHUNK points)."""

    if len(xy) > CHUNK:
        return np.concatenate([inside(poly, xy[i:i + CHUNK]) for i in range(0, len(xy), CHUNK)])

    x, y = xy[:, 0:1], xy[:, 1:2]
    a, b = poly, np.roll(poly, -1, axis=0)
    crosses = (a[:, 1] > y) != (b[:, 1] > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_cross = a[:, 0] + (y - a[:, 1]) * (b[:, 0] - a[:, 0]) / (b[:, 1] - a[:, 1])

    return (np.sum(crosses & (x < x_cross), axis=1) % 2) == 1

def segment_distance(xy: np.ndarray, poly: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distance of each point to the closed polyline and the index of the nearest edge (chunked)."""

    if len(xy) > CHUNK:
        parts = [segment_distance(xy[i:i + CHUNK], poly) for i in range(0, len(xy), CHUNK)]
        return np.concatenate([d for d, _ in parts]), np.concatenate([j for _, j in parts])

    a, b = poly, np.roll(poly, -1, axis=0)
    ab = b - a
    rel = xy[:, None, :] - a[None]
    u = np.clip(np.sum(rel * ab, axis=2) / np.maximum(np.sum(ab * ab, axis=1), 1e-12), 0.0, 1.0)
    d = np.linalg.norm(rel - u[..., None] * ab, axis=2)
    j = np.argmin(d, axis=1)

    return d[np.arange(len(xy)), j], j

def place(poly: np.ndarray, s: float, theta: float, t: np.ndarray) -> np.ndarray:

    return s * poly @ rot2(theta).T + t

def trimmed_cost(q: np.ndarray, poly: np.ndarray, s: float, theta: float, t: np.ndarray) -> float:

    d, _ = segment_distance(q, place(poly, s, theta, t))

    return float(np.mean(np.sort(d)[:max(1, int(TRIM * len(d)))]))

def gauss_newton(q: np.ndarray, poly: np.ndarray, s: float, theta: float, t: np.ndarray,
                 iters: int) -> tuple[float, float, np.ndarray]:
    """Robust point-to-edge fit of q = s R(theta) p + t. Residual of q_i on its nearest edge j (inward unit
    normal n_j, offset c_j = n_j . a_j): r = (R n_j).(q - t) - s c_j; d/dtheta = (J R n_j).(q - t),
    d/dlog s = -s c_j, d/dt = -R n_j. Huber weights, residuals above the TRIM quantile ignored."""

    a, b = poly, np.roll(poly, -1, axis=0)
    d_edge = (b - a) / np.linalg.norm(b - a, axis=1, keepdims=True)
    n_edge = np.c_[-d_edge[:, 1], d_edge[:, 0]]   # left normal = inward for a CCW outline
    c_edge = np.sum(n_edge * a, axis=1)
    t = np.asarray(t, dtype=np.float64).copy()
    for _ in range(iters):
        _, j = segment_distance(q, place(poly, s, theta, t))
        R = rot2(theta)
        N = n_edge[j] @ R.T                         # rotated normals per point
        qt = q - t
        r = np.sum(N * qt, axis=1) - s * c_edge[j]
        absr = np.abs(r)
        delta = max(1.345 * 1.4826 * float(np.median(absr)), 1e-9 * s)
        w = np.where(absr <= delta, 1.0, delta / np.maximum(absr, 1e-30))
        w[absr > np.quantile(absr, TRIM)] = 0.0
        JN = np.c_[-N[:, 1], N[:, 0]]               # J R n, J = 90 deg rotation
        Jm = np.c_[np.sum(JN * qt, axis=1), -s * c_edge[j], -N]
        H = Jm.T @ (Jm * w[:, None])
        g = Jm.T @ (w * r)
        step = np.linalg.solve(H + 1e-9 * np.trace(H) * np.eye(4), -g)
        step[1] = np.clip(step[1], -0.5, 0.5)       # at most x1.65 in scale per step
        theta, s, t = theta + step[0], s * float(np.exp(step[1])), t + step[2:]
        if np.abs(step[0]) < 1e-9 and np.abs(step[1]) < 1e-9:
            break

    return s, theta, t

def fit_outline(q: np.ndarray, footprint: np.ndarray, n_rot: int) -> dict:
    """Best similarity over n_rot initial angles x {outline, mirrored}. Returns the pose in the frame of the
    centred outline (`poly`, cm, CCW) plus its fit statistics."""

    q_lo, q_hi = np.percentile(q, [2, 98], axis=0)
    starts = []
    for mirrored in (False, True):
        poly = footprint * ([-1.0, 1.0] if mirrored else [1.0, 1.0])
        poly = ccw(poly)
        poly = poly - poly.mean(axis=0)
        p_lo, p_hi = poly.min(axis=0), poly.max(axis=0)
        p_ext, p_mid = np.maximum(p_hi - p_lo, 1e-9), (p_lo + p_hi) / 2
        for theta0 in np.linspace(0.0, 2 * np.pi, max(1, n_rot), endpoint=False):
            R = rot2(theta0)
            local = (q - q.mean(axis=0)) @ R          # wall points in the outline orientation
            l_lo, l_hi = np.percentile(local, [2, 98], axis=0)
            s0 = float(np.sqrt(np.prod(np.maximum(l_hi - l_lo, 1e-12)) / np.prod(p_ext)))
            mid = q.mean(axis=0) + R @ ((l_lo + l_hi) / 2)
            t0 = mid - s0 * R @ p_mid
            try:
                s, theta, t = gauss_newton(q, poly, s0, theta0, t0, COARSE_ITERS)
            except np.linalg.LinAlgError:
                continue                                # singular start (e.g. every point on one edge)
            if not (np.isfinite(s) and s > 0 and np.all(np.isfinite(t))):
                continue
            starts.append((trimmed_cost(q, poly, s, theta, t), mirrored, poly, s, theta, t))

    if not starts:
        raise RuntimeError("outline fit diverged from every start")

    starts.sort(key=lambda x: x[0])
    _, mirrored, poly, s, theta, t = starts[0]
    s, theta, t = gauss_newton(q, poly, s, theta, t, FINE_ITERS)
    cost = trimmed_cost(q, poly, s, theta, t)
    theta = float(np.mod(theta, 2 * np.pi))

    def different(x):
        dth = abs((x[4] - theta + np.pi) % (2 * np.pi) - np.pi)
        return x[1] != mirrored or dth > np.radians(5) or abs(np.log(x[3] / s)) > 0.05

    rival = next((x for x in starts[1:] if different(x)), None)
    d, _ = segment_distance(q, place(poly, s, theta, t))
    d_cm = d / s

    return {"poly": poly, "s": float(s), "theta": theta, "t": np.asarray(t), "mirrored": mirrored,
            "residual_median_cm": float(np.median(d_cm)), "residual_p90_cm": float(np.percentile(d_cm, 90)),
            "inlier_frac": float(np.mean(d_cm < INLIER_CM)),
            "ambiguity": float(cost / rival[0]) if rival is not None and rival[0] > 0 else float("nan")}

def to_frame(T: np.ndarray, points: np.ndarray) -> np.ndarray:

    return points @ T[:3, :3].T + T[:3, 3]

def lattice_mesh(nodes: np.ndarray, quads: np.ndarray, color: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Kept quads (K,4 node indices, counter-clockwise seen from the normal side) -> (V, F, C) with only
    the used nodes, 2 triangles per quad."""

    if len(quads) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64), np.zeros((0, 3))

    used, inv = np.unique(quads.ravel(), return_inverse=True)
    q = inv.reshape(-1, 4)
    faces = np.vstack([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])

    return nodes[used], faces, np.tile(color, (len(used), 1))

def horizontal_cells(poly: np.ndarray, grid: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Grid over the outline bbox (cm): nodes (2D), quads (CCW from +z) and centres of the cells whose
    centre is inside the outline."""

    lo, hi = poly.min(axis=0), poly.max(axis=0)
    nx, ny = (np.ceil((hi - lo) / grid).astype(int) + 1)
    gx, gy = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1), indexing="ij")
    nodes = lo + np.c_[gx.ravel(), gy.ravel()] * grid
    ci, cj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    ci, cj = ci.ravel(), cj.ravel()
    centres = lo + (np.c_[ci, cj] + 0.5) * grid
    keep = inside(poly, centres)
    ci, cj, centres = ci[keep], cj[keep], centres[keep]
    node = lambda i, j: i * (ny + 1) + j
    quads = np.c_[node(ci, cj), node(ci + 1, cj), node(ci + 1, cj + 1), node(ci, cj + 1)]

    return nodes, quads, centres

def wall_cells(poly: np.ndarray, height: float, grid: float) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    """Per edge: nodes (3D, cm), quads (CCW seen from inside the room), centres and cell area."""

    nz = max(1, int(np.ceil(height / grid)))
    walls = []
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        length = float(np.linalg.norm(b - a))
        nu = max(1, int(np.ceil(length / grid)))
        gu, gz = np.meshgrid(np.linspace(0, 1, nu + 1), np.linspace(0, height, nz + 1), indexing="ij")
        nodes = np.c_[a + gu.ravel()[:, None] * (b - a), gz.ravel()]
        ci, cj = np.meshgrid(np.arange(nu), np.arange(nz), indexing="ij")
        ci, cj = ci.ravel(), cj.ravel()
        cu, cz = (ci + 0.5) / nu, (cj + 0.5) * height / nz
        centres = np.c_[a + cu[:, None] * (b - a), cz]
        node = lambda i, j: i * (nz + 1) + j
        # (u0,z0) (u0,z1) (u1,z1) (u1,z0): normal = left of the edge = inside of a CCW outline
        quads = np.c_[node(ci, cj), node(ci, cj + 1), node(ci + 1, cj + 1), node(ci + 1, cj)]
        walls.append((nodes, quads, centres, (length / nu) * (height / nz)))

    return walls

def nearest_distance(queries: np.ndarray, targets: list[np.ndarray]) -> np.ndarray:
    """Distance of each query to the nearest point of any target cloud (Open3D KD-tree)."""

    best = np.full(len(queries), np.inf)
    if len(queries) == 0:
        return best

    qp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(queries))
    for pts in targets:
        if pts is not None and len(pts):
            tp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(pts, dtype=np.float64)))
            best = np.minimum(best, np.asarray(qp.compute_point_cloud_distance(tp)))

    return best

def median_color(colors: np.ndarray | None, mask: np.ndarray) -> np.ndarray:

    if colors is None or mask.sum() == 0:
        return GRAY

    return np.median(colors[mask], axis=0) / 255.0

def load_source(cfg: dict, run_dir: Path) -> tuple[str, np.ndarray, np.ndarray | None] | None:
    """(source, points, colors) of clean/<room_fill.source>, else clean/<planes.source>, else clean/sparse."""

    wanted = cfg["room_fill"]["source"]
    for source in dict.fromkeys([wanted, cfg.get("planes", {}).get("source", "sparse"), "sparse"]):
        path = run_dir / "clean" / source / "points.npy"
        if not path.exists():
            continue
        if source != wanted:
            log.warning("room_fill: clean/%s/points.npy missing, using clean/%s", wanted, source)
        points = np.load(path).astype(np.float64).reshape(-1, 3)
        c_path = path.parent / "colors.npy"
        colors = np.load(c_path).astype(np.float64).reshape(-1, 3) if c_path.exists() else None
        if colors is not None and len(colors) != len(points):
            log.warning("room_fill: clean/%s colors misaligned, ignored", source)
            colors = None
        return source, points, colors

    return None

def run(cfg: dict, run_dir: str, artifacts_dir: str) -> Path | None:

    rcfg = cfg["room_fill"]
    if not rcfg["enabled"]:
        log.info("room_fill disabled")
        return None

    t0 = time.perf_counter()
    run_dir, artifacts_dir = Path(run_dir), Path(artifacts_dir)
    footprint = np.asarray(rcfg["footprint_cm"], dtype=np.float64).reshape(-1, 2)
    if len(footprint) < 3 or abs(signed_area(footprint)) < 1e-6:
        log.warning("room_fill: room_fill.footprint_cm needs >= 3 non-collinear [x, y] corners in cm, stage skipped")
        return None

    surfaces = [s for s in SURFACES if s in rcfg["surfaces"]]
    planes = run_dir / "planes"
    T_path, floor_path = planes / "T_floor.npy", planes / "floor.npy"
    if not T_path.exists() or not floor_path.exists() or not np.all(np.isfinite(np.load(floor_path))):
        log.warning("room_fill: no floor in %s (run the planes stage), stage skipped", planes)
        return None

    loaded = load_source(cfg, run_dir)
    if loaded is None:
        log.warning("room_fill: no clean/<source>/points.npy under %s (run cleanup), stage skipped", run_dir)
        return None
    source, points, colors = loaded
    T = np.load(T_path)
    T_inv = np.linalg.inv(T)
    pf = to_frame(T, points)                         # floor frame, working units
    labels = np.load(planes / "labels.npy") if (planes / "labels.npy").exists() else None
    if labels is not None and len(labels) != len(points):
        log.info("room_fill: planes labels belong to another cloud (%d vs %d points), not used", len(labels), len(points))
        labels = None

    # 1. wall points: planes labels, else the room_fill.wall_band fraction of the height range
    z_lo, z_hi = np.percentile(pf[:, 2], [1, 99])
    wall_mask, wall_from = None, "labels"
    if labels is not None and (labels >= 2).sum() >= MIN_WALL_POINTS:
        wall_mask = labels >= 2
    else:
        wall_from = "height_band"
        b_lo, b_hi = rcfg.get("wall_band", [0.2, 0.8])
        wall_mask = (pf[:, 2] > z_lo + b_lo * (z_hi - z_lo)) & (pf[:, 2] < z_lo + b_hi * (z_hi - z_lo))
    q = pf[wall_mask, :2]
    if len(q) < 10:
        log.warning("room_fill: only %d wall points, outline fit impossible, stage skipped", len(q))
        return None
    rng = np.random.default_rng(int(cfg.get("run", {}).get("seed", 0)))
    if len(q) > FIT_POINTS:
        q = q[rng.choice(len(q), FIT_POINTS, replace=False)]

    # 2. outline fit: q = s R p + t (s = working units per cm)
    try:
        fit = fit_outline(q, footprint, int(rcfg["fit_rotations"]))
    except (RuntimeError, np.linalg.LinAlgError) as exc:
        log.warning("room_fill: %s, stage skipped", exc)
        return None
    s, theta, t, poly = fit["s"], fit["theta"], fit["t"], fit["poly"]
    R2 = rot2(theta)

    # 3. ceiling height (working units)
    ceiling = np.load(planes / "ceiling.npy") if (planes / "ceiling.npy").exists() else np.full(4, np.nan)
    if rcfg["ceiling_height_cm"] is not None:
        h, ceiling_from = float(rcfg["ceiling_height_cm"]) * s, "config"
    elif np.all(np.isfinite(ceiling)):
        h, ceiling_from = float(to_frame(T, (-ceiling[3] * ceiling[:3])[None])[0, 2]), "planes"
    else:
        room = inside(place(poly, s, theta, t), pf[:, :2]) & (pf[:, 2] > 0)
        h = float(np.percentile(pf[room, 2], 99)) if room.any() else float("nan")
        ceiling_from = "p99"
    if not np.isfinite(h) or h <= 0:
        log.warning("room_fill: ceiling height not found (%s), stage skipped", ceiling_from)
        return None
    h_cm = h / s

    # 4. cells (outline frame, cm) -> floor frame -> coverage test against the cloud and the selected mesh
    selected = mesh_stage.pick_input(cfg, run_dir) if "mesh" in cfg else None
    sel_mesh = o3d.io.read_triangle_mesh(str(selected)) if selected is not None else None
    if sel_mesh is None or len(sel_mesh.triangles) == 0:
        log.warning("room_fill: no selected mesh (mesh.input), coverage from the cloud only, scene = fill only")
        sel_mesh = None
    targets = [pf] + ([to_frame(T, np.asarray(sel_mesh.vertices))] if sel_mesh is not None else [])
    gap, grid = rcfg["max_gap_cm"] * s, rcfg["grid_cm"]

    def to_floor(xy_cm: np.ndarray, z_cm: np.ndarray) -> np.ndarray:
        return np.c_[s * xy_cm @ R2.T + t, s * z_cm]

    items = []                                      # (surface, nodes, quads, centres, cell area, colour), cm
    for surf in surfaces:
        if surf in ("floor", "ceiling"):
            nodes, quads, centres = horizontal_cells(poly, grid)
            z = 0.0 if surf == "floor" else h_cm
            cells = [(np.c_[nodes, np.full(len(nodes), z)], quads if surf == "floor" else quads[:, ::-1],
                      np.c_[centres, np.full(len(centres), z)], grid * grid)]
            label, near = (0, np.abs(pf[:, 2]) < gap) if surf == "floor" else (1, np.abs(pf[:, 2] - h) < gap)
        else:
            cells = wall_cells(poly, h_cm, grid)
            label = None
            d_wall, _ = segment_distance(pf[:, :2], place(poly, s, theta, t))
            near = (d_wall < gap) & (pf[:, 2] > gap) & (pf[:, 2] < h - gap)
        mask = (labels >= 2 if label is None else labels == label) if labels is not None else near
        color = median_color(colors, mask)
        items += [(surf, *cell, color) for cell in cells]

    centres_all = np.vstack([to_floor(c[:, :2], c[:, 2]) for _, _, _, c, _, _ in items])
    covered_all = nearest_distance(centres_all, targets) < gap   # one KD-tree per target cloud
    parts, totals, start = [], {surf: [0.0, 0.0] for surf in surfaces}, 0
    for surf, nodes, quads, centres, cell_area, color in items:
        covered = covered_all[start:start + len(centres)]
        start += len(centres)
        kept = quads[~covered]
        totals[surf][0] += len(quads) * cell_area
        totals[surf][1] += len(kept) * cell_area
        V, F, C = lattice_mesh(to_floor(nodes[:, :2], nodes[:, 2]), kept, color)
        if len(F):
            parts.append((V, F, C))
    row_surf = {}
    for surf, (total, filled) in totals.items():
        row_surf.update({f"{surf}_area_m2": round(total / 1e4, 4), f"{surf}_filled_m2": round(filled / 1e4, 4),
                         f"{surf}_filled_frac": round(filled / total, 4) if total else ""})

    # 5. back to the model frame; write fill and scene
    out_dir = run_dir / "mesh" / "room_fill"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    meshlab = run_dir / "exports" / "meshlab"
    meshlab.mkdir(parents=True, exist_ok=True)
    for name in ("room_fill.ply", "room_scene.ply"):
        (meshlab / name).unlink(missing_ok=True)
    fill = o3d.geometry.TriangleMesh()
    offset = 0
    V_all, F_all, C_all = [], [], []
    for V, F, C in parts:
        V_all.append(to_frame(T_inv, V))
        F_all.append(F + offset)
        C_all.append(C)
        offset += len(V)
    if parts:
        fill.vertices = o3d.utility.Vector3dVector(np.vstack(V_all))
        fill.triangles = o3d.utility.Vector3iVector(np.vstack(F_all).astype(np.int32))
        fill.vertex_colors = o3d.utility.Vector3dVector(np.vstack(C_all))
        fill.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(out_dir / "room_fill.ply"), fill, write_ascii=False)
        shutil.copy2(out_dir / "room_fill.ply", meshlab / "room_fill.ply")
    else:
        log.warning("room_fill: every cell is covered by the reconstruction, nothing to fill")

    scene = o3d.geometry.TriangleMesh()
    if sel_mesh is not None:
        if not sel_mesh.has_vertex_colors():
            sel_mesh.vertex_colors = o3d.utility.Vector3dVector(np.tile(GRAY, (len(sel_mesh.vertices), 1)))
        scene += sel_mesh
    if parts:
        scene += fill
    if len(scene.triangles):
        scene.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(out_dir / "room_scene.ply"), scene, write_ascii=False)
        shutil.copy2(out_dir / "room_scene.ply", meshlab / "room_scene.ply")

    # outline frame (cm, z up from the floor) -> model frame (working units)
    T_room = np.eye(4)
    T_room[:2, :2], T_room[2, 2], T_room[:2, 3] = s * R2, s, t
    np.save(out_dir / "T_room.npy", T_inv @ T_room)
    np.save(out_dir / "footprint_cm.npy", poly)

    s_scale, unit = stage_io.scale_info(run_dir)
    cm_per_unit = 1.0 / s
    row = {"source": source, "wall_points_from": wall_from, "n_wall_points": int(wall_mask.sum()),
           "mirrored": fit["mirrored"], "rotation_deg": round(float(np.degrees(theta)), 3),
           "fit_residual_median_cm": round(fit["residual_median_cm"], 3),
           "fit_residual_p90_cm": round(fit["residual_p90_cm"], 3), "fit_inlier_frac": round(fit["inlier_frac"], 4),
           "fit_ambiguity": round(fit["ambiguity"], 4) if np.isfinite(fit["ambiguity"]) else "",
           "s_units_per_cm": round(s, 8), "cm_per_unit": round(cm_per_unit, 6), "scale_unit": unit,
           "scale_cm_per_model_unit": round(s_scale, 6) if unit == "cm" else "",
           "cm_per_model_unit": round(cm_per_unit * (s_scale if unit == "cm" else 1.0), 6),
           "scale_ratio": round(cm_per_unit, 4) if unit == "cm" else "",
           "ceiling_from": ceiling_from, "ceiling_height_cm": round(h_cm, 2),
           "polygon_area_m2": round(abs(signed_area(footprint)) / 1e4, 4), **row_surf,
           "mesh_input": str(selected.relative_to(run_dir)) if sel_mesh is not None else "",
           "n_vertices": len(fill.vertices), "n_faces": len(fill.triangles), "t_s": round(time.perf_counter() - t0, 3)}
    stage_io.upsert_csv(artifacts_dir / "room_fill.csv", row)
    stage_io.log_metrics(log, row)
    if fit["inlier_frac"] < 0.5 or fit["residual_median_cm"] > INLIER_CM:
        log.warning("room_fill: poor outline fit (median %.1f cm, inliers %.0f%%): check footprint_cm and the planes",
                    fit["residual_median_cm"], 100 * fit["inlier_frac"])
    log.info("room_fill: %s, cm_per_unit=%.4g, ceiling %.1f cm (%s), +%d faces -> %s", source, cm_per_unit, h_cm,
             ceiling_from, len(fill.triangles), out_dir)

    return out_dir / "room_scene.ply" if len(scene.triangles) else None
# <<< END IMPLEMENTATION <<<

KITCHEN_CM = np.array([[0, 0], [273, 0], [273, 429], [81.1, 424.5], [0, 332]], dtype=np.float64)

def synthetic_run(tmp: Path, rng: np.random.Generator, s_true: float = 0.0213, theta_deg: float = 37.0,
                  t_true=(1.7, -0.9), height_cm: float = 250.0, hole=((150.0, 200.0), 60.0),
                  ceiling_y_max: float = 200.0, wall_hole=((0, 120.0, 150.0), 40.0), source: str = "patch_match",
                  labels: bool = False, mesh_patch=None) -> tuple[Path, dict]:
    """Kitchen outline room in model units: points on the floor (minus a disc hole), on the ceiling only where
    y < ceiling_y_max, on the walls (minus a disc on one wall), plus a cabinet-like clutter box; floor frame
    q = s R p + t, then a random rigid 3D motion (T_floor = its inverse). mesh_patch = (x0, y0, x1, y1) cm adds a
    poisson/pre mesh covering that floor rectangle. Returns (run_dir, truth)."""

    poly = KITCHEN_CM
    density, noise = 0.25, 0.5                      # points per cm^2, cm

    def floor_like(z, keep):
        lo, hi = poly.min(axis=0), poly.max(axis=0)
        n = int(density * np.prod(hi - lo))
        xy = rng.uniform(lo, hi, (n, 2))
        xy = xy[inside(poly, xy) & keep(xy)]
        return np.c_[xy, np.full(len(xy), z) + rng.normal(0, noise, len(xy))]

    (hc, hr) = hole
    floor = floor_like(0.0, lambda xy: np.linalg.norm(xy - hc, axis=1) > hr)
    ceil_pts = floor_like(height_cm, lambda xy: xy[:, 1] < ceiling_y_max)
    walls = []
    for k, (a, b) in enumerate(zip(poly, np.roll(poly, -1, axis=0))):
        length = np.linalg.norm(b - a)
        n = int(density * length * height_cm)
        u, z = rng.random(n), rng.uniform(0, height_cm, n)
        if wall_hole is not None and k == wall_hole[0][0]:
            keep = np.hypot(u * length - wall_hole[0][1], z - wall_hole[0][2]) > wall_hole[1]
            u, z = u[keep], z[keep]
        normal = np.array([-(b - a)[1], (b - a)[0]]) / length
        xy = a + u[:, None] * (b - a) + rng.normal(0, noise, (len(u), 1)) * normal
        walls.append(np.c_[xy, z])
    walls = np.vstack(walls)
    clutter = rng.uniform([200, 60, 0], [270, 300, 90], (3000, 3))   # counter along the sink wall
    room = np.vstack([floor, ceil_pts, walls, clutter])
    lab = np.r_[np.zeros(len(floor)), np.ones(len(ceil_pts)), np.full(len(walls), 2), np.full(len(clutter), -1)]
    cols = np.r_[np.tile([200, 180, 150], (len(floor), 1)), np.tile([240, 240, 235], (len(ceil_pts), 1)),
                 np.tile([180, 200, 220], (len(walls), 1)), np.tile([90, 60, 40], (len(clutter), 1))]

    theta = np.radians(theta_deg)
    t_true = np.asarray(t_true, dtype=np.float64)
    qf = np.c_[s_true * room[:, :2] @ rot2(theta).T + t_true, s_true * room[:, 2]]
    A, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    Rw = A * np.sign(np.linalg.det(A))
    tw = rng.normal(0, 3, 3)
    world = qf @ Rw.T + tw
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = Rw.T, -Rw.T @ tw

    run_dir = tmp / "run"
    for d in (f"clean/{source}", "planes"):
        (run_dir / d).mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "clean" / source / "points.npy", world)
    np.save(run_dir / "clean" / source / "colors.npy", cols.astype(np.uint8))
    np.save(run_dir / "planes" / "T_floor.npy", T)
    np.save(run_dir / "planes" / "floor.npy", np.r_[Rw[:, 2], -Rw[:, 2] @ tw])
    np.save(run_dir / "planes" / "ceiling.npy", np.full(4, np.nan))
    if labels:
        np.save(run_dir / "planes" / "labels.npy", lab.astype(np.int32))
    if mesh_patch is not None:
        x0, y0, x1, y1 = mesh_patch
        gx, gy = np.meshgrid(np.linspace(x0, x1, 41), np.linspace(y0, y1, 41), indexing="ij")
        pv = np.c_[gx.ravel(), gy.ravel(), np.zeros(gx.size)]
        idx = lambda i, j: i * 41 + j
        ii, jj = np.meshgrid(np.arange(40), np.arange(40), indexing="ij")
        ii, jj = ii.ravel(), jj.ravel()
        pfaces = np.vstack([np.c_[idx(ii, jj), idx(ii + 1, jj), idx(ii + 1, jj + 1)],
                            np.c_[idx(ii, jj), idx(ii + 1, jj + 1), idx(ii, jj + 1)]])
        pw = np.c_[s_true * pv[:, :2] @ rot2(theta).T + t_true, pv[:, 2]] @ Rw.T + tw
        m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(pw), o3d.utility.Vector3iVector(pfaces))
        (run_dir / "mesh/poisson/pre").mkdir(parents=True)
        o3d.io.write_triangle_mesh(str(run_dir / "mesh/poisson/pre/mesh.ply"), m)

    truth = {"s": s_true, "theta": theta, "t": t_true, "Rw": Rw, "tw": tw, "height_cm": height_cm, "hole": hole,
             "ceiling_y_max": ceiling_y_max, "poly": poly}

    return run_dir, truth

def to_room_cm(points: np.ndarray, truth: dict) -> np.ndarray:
    """Model-frame points back to the true outline frame (cm) of synthetic_run."""

    qf = (points - truth["tw"]) @ truth["Rw"]
    xy = (qf[:, :2] - truth["t"]) @ rot2(truth["theta"]) / truth["s"]

    return np.c_[xy, qf[:, 2] / truth["s"]]

def default_cfg(**overrides) -> dict:

    rcfg = {"enabled": True, "footprint_cm": KITCHEN_CM.tolist(), "ceiling_height_cm": None,
            "surfaces": ["floor", "ceiling"], "source": "patch_match", "grid_cm": 5.0, "max_gap_cm": 5.0,
            "fit_rotations": 72, **overrides}

    return {"run": {"seed": 42}, "planes": {"source": "sparse"}, "mesh": {"input": "close_holes"}, "room_fill": rcfg}

if __name__ == "__main__":
    import tempfile
    logging.basicConfig(level=logging.INFO)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        run_dir, truth = synthetic_run(tmp, np.random.default_rng(0), mesh_patch=(120, 170, 150, 200))
        art = tmp / "art"
        assert run({"room_fill": {"enabled": False}}, str(run_dir), str(art)) is None
        assert run(default_cfg(footprint_cm=[]), str(run_dir), str(art)) is None
        out = run(default_cfg(), str(run_dir), str(art))
        assert out is not None and out.exists() and (run_dir / "exports/meshlab/room_fill.ply").exists()
        row = stage_io.read_csv(art / "room_fill.csv")[0]
        print(row)
        assert abs(float(row["s_units_per_cm"]) / truth["s"] - 1) < 0.02, row
        assert abs((float(row["rotation_deg"]) - 37.0 + 180) % 360 - 180) < 2.0, row
        assert row["mirrored"] == "False" and float(row["fit_residual_median_cm"]) < 2.0, row
        assert abs(float(row["ceiling_height_cm"]) - 250) < 5 and row["ceiling_from"] == "p99", row
        fill = o3d.io.read_triangle_mesh(str(run_dir / "mesh/room_fill/room_fill.ply"))
        c = to_room_cm(np.asarray(fill.vertices)[np.asarray(fill.triangles)].mean(axis=1), truth)
        floor_c, ceil_c = c[c[:, 2] < 125], c[c[:, 2] >= 125]
        assert np.linalg.norm(floor_c[:, :2] - truth["hole"][0], axis=1).max() < 60 + 8   # only the hole
        assert not (inside(np.array([[120, 170], [150, 170], [150, 200], [120, 200.0]]), floor_c[:, :2])).any()
        assert ceil_c[:, 1].min() > 200 - 8                                                # only the bare half
        hole_m2 = (np.pi * 55 ** 2 - (30 ** 2 + 4 * 30 * 5 + np.pi * 5 ** 2)) / 1e4  # hole minus gap, minus patch
        assert abs(float(row["floor_filled_m2"]) - hole_m2) < 0.15 * hole_m2, row
        assert (run_dir / "mesh/room_fill/room_scene.ply").exists() and row["mesh_input"] == "mesh/poisson/pre/mesh.ply"
        # mirrored outline given clockwise: same scale, mirrored flag set
        out2 = run(default_cfg(footprint_cm=(KITCHEN_CM * [-1, 1]).tolist()), str(run_dir), str(art))
        row2 = stage_io.read_csv(art / "room_fill.csv")[0]
        assert out2 is not None and row2["mirrored"] == "True" and abs(float(row2["s_units_per_cm"]) / truth["s"] - 1) < 0.02
        shutil.rmtree(run_dir / "planes")
        assert run(default_cfg(), str(run_dir), str(art)) is None                         # no planes -> None
    print("OK")

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
# Stage: planes (after cleanup, before floorplan).
# Goal: find floor, ceiling and walls by iterative RANSAC, the floor frame T_floor and the wall corner angles.
# In:  clean/<planes.source>/points.npy (working units; fallback scale/points3d_cm.npy), sfm/cam_R.npy,
#      scale/centers_cm.npy (fallback sfm/centers.npy).
# Out: planes/{floor,ceiling}.npy (4,) a,b,c,d (unit normal, NaN if not found), planes/walls.npy (W,4),
#      planes/T_floor.npy (4,4), planes/labels.npy (P,) int (-1 none, 0 floor, 1 ceiling, 2+k wall k);
#      artifacts: planes.csv (plane, type, a, b, c, d, n_inliers, inlier_frac, height, height_cm),
#      wall_angles.csv (wall_i, wall_j, angle_deg, error_to_90_deg).
# YAML: planes: {enabled, source: sparse, distance_cm, ransac_n, iterations, max_walls, min_inlier_frac,
#       angle_tol_deg, max_planes}
# Rules:
# - up = normalised mean of -R_i^T [0,1,0] (OpenCV camera y points down); [0,-1,0] + WARNING without cameras.
# - Planes are peeled one by one (Open3D segment_plane, inliers removed) up to max_planes rounds, stopping
#   when a plane has fewer than min_inlier_frac * P inliers. |n.up| > cos(tol) -> horizontal,
#   |n.up| < sin(tol) -> vertical, else other.
# - Floor = LOWEST horizontal plane (median inlier height along up) below the median camera height. The
#   spec says "first plane nearly perpendicular to gravity"; the lowest one is preferred because a large
#   table or the ceiling may be peeled first. Ceiling = horizontal plane farthest above the floor and above
#   the cameras. Walls = vertical planes, largest first, up to max_walls.
# - Normals: floor and ceiling point to +up; walls point to the camera centroid (room interior).
# - T_floor: rotation taking the floor normal to +z, then a rotation about z putting the largest wall
#   normal on +x, translation putting the floor at z = 0. Identity + WARNING without a floor.
# - Wall angles: inward wall normals projected on the floor plane, sorted by their azimuth; neighbours are
#   consecutive in that cyclic order. angle_deg = interior corner angle = 180 - angle between the two
#   inward normals, in [0, 180]; error_to_90_deg = |angle_deg - 90|.
# - height_cm is empty when the scale is unknown (unit = model).

# >>> BEGIN IMPLEMENTATION >>>
"""Planes: floor, ceiling and walls by iterative RANSAC, floor frame and wall corner angles."""
import logging
from pathlib import Path

import numpy as np
import open3d as o3d
from tqdm import tqdm

from src import stage_io

log = logging.getLogger(__name__)

def load_points(run_dir: Path, source: str) -> np.ndarray | None:

    candidates = [run_dir / "clean" / source / "points.npy", run_dir / "clean" / "sparse" / "points.npy",
                  run_dir / "scale" / "points3d_cm.npy"]
    for path in candidates:
        if path.exists():
            if path != candidates[0]:
                log.warning("planes: %s missing, using %s", candidates[0], path)
            return np.load(path).astype(np.float64).reshape(-1, 3)

    return None

def load_cameras(run_dir: Path) -> tuple[np.ndarray | None, np.ndarray | None]:

    r_path = run_dir / "sfm" / "cam_R.npy"
    cam_R = np.load(r_path) if r_path.exists() else None
    centers = None
    for path in (run_dir / "scale" / "centers_cm.npy", run_dir / "sfm" / "centers.npy"):
        if path.exists():
            centers = np.load(path).astype(np.float64).reshape(-1, 3)
            break

    return cam_R, centers

def gravity_up(cam_R: np.ndarray | None) -> np.ndarray:
    """Mean camera 'up' in the world: -R^T [0,1,0] (OpenCV y points down)."""

    if cam_R is None or len(cam_R) == 0:
        log.warning("planes: no cameras, assuming up = [0,-1,0] (first COLMAP camera frame)")
        return np.array([0.0, -1.0, 0.0])

    ups = -np.asarray(cam_R)[:, 1, :]  # row 1 of R = camera y axis in world
    up = ups.mean(axis=0)

    return up / np.linalg.norm(up)

def peel_planes(points: np.ndarray, distance: float, ransac_n: int, iterations: int, max_planes: int,
                min_inliers: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Iterative segment_plane removing inliers: [(plane (4,), original indices)]."""

    remaining = np.arange(len(points))
    planes = []
    for _ in tqdm(range(max_planes), desc="planes", leave=False):
        if len(remaining) < max(ransac_n, min_inliers):
            break

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[remaining]))
        model, inliers = pcd.segment_plane(distance, ransac_n, iterations)
        if len(inliers) < min_inliers:
            break

        model = np.asarray(model, dtype=np.float64)
        model /= np.linalg.norm(model[:3])
        idx = remaining[np.asarray(inliers)]
        planes.append((model, idx))
        remaining = np.setdiff1d(remaining, idx, assume_unique=True)

    return planes

def orient(plane: np.ndarray, direction: np.ndarray) -> np.ndarray:

    return plane if plane[:3] @ direction >= 0 else -plane

def orient_to_point(plane: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Flip so that `point` is on the positive side (n.x + d > 0)."""

    return plane if plane[:3] @ point + plane[3] >= 0 else -plane

def rotation_to(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation matrix taking unit vector a to unit vector b."""

    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-12:
        if c > 0:
            return np.eye(3)
        axis = np.cross(a, [1.0, 0.0, 0.0])
        axis = axis if np.linalg.norm(axis) > 1e-6 else np.cross(a, [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)

    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    return np.eye(3) + vx + vx @ vx / (1 + c)

def floor_transform(floor: np.ndarray, walls: np.ndarray) -> np.ndarray:

    T = np.eye(4)
    if not np.all(np.isfinite(floor)):
        log.warning("planes: no floor found, T_floor = identity")
        return T

    R = rotation_to(floor[:3], np.array([0.0, 0.0, 1.0]))
    if len(walls):
        m = R @ walls[0, :3]
        yaw = -np.arctan2(m[1], m[0])
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        R = Rz @ R

    T[:3, :3] = R
    T[2, 3] = floor[3]  # z' = n.x + d

    return T

def wall_angles(walls: np.ndarray, T: np.ndarray) -> list[dict]:
    """Interior corner angle between cyclic neighbours (walls sorted by normal azimuth on the floor)."""

    if len(walls) < 2:
        return []

    m = (T[:3, :3] @ walls[:, :3].T).T[:, :2]
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    order = np.argsort(np.arctan2(m[:, 1], m[:, 0]))
    pairs = [(order[k], order[(k + 1) % len(order)]) for k in range(len(order))]
    pairs = pairs[:1] if len(walls) == 2 else pairs

    rows = []
    for i, j in pairs:
        between = np.degrees(np.arccos(np.clip(m[i] @ m[j], -1.0, 1.0)))
        angle = 180.0 - between
        rows.append({"wall_i": f"wall_{i}", "wall_j": f"wall_{j}", "angle_deg": round(float(angle), 3),
                     "error_to_90_deg": round(abs(float(angle) - 90.0), 3)})

    return rows

def run(cfg: dict, run_dir: str, artifacts_dir: str) -> dict | None:

    pc = cfg["planes"]
    if not pc["enabled"]:
        log.info("planes disabled")
        return None

    run_dir, artifacts_dir = Path(run_dir), Path(artifacts_dir)
    points = load_points(run_dir, pc["source"])
    if points is None or len(points) < pc["ransac_n"]:
        log.warning("planes: no point cloud under %s, skipping", run_dir)
        return None

    o3d.utility.random.seed(int(cfg["run"]["seed"]))
    cam_R, centers = load_cameras(run_dir)
    up = gravity_up(cam_R)
    upc = stage_io.units_per_cm(run_dir, points, room_extent_cm=cfg["scale"]["provisional_extent_cm"])
    _, unit = stage_io.scale_info(run_dir)
    n_total = len(points)
    min_inliers = max(int(np.ceil(pc["min_inlier_frac"] * n_total)), pc["ransac_n"])

    found = peel_planes(points, pc["distance_cm"] * upc, pc["ransac_n"], pc["iterations"], pc["max_planes"],
                        min_inliers)
    cos_tol, sin_tol = np.cos(np.radians(pc["angle_tol_deg"])), np.sin(np.radians(pc["angle_tol_deg"]))
    cam_height = float(np.median(centers @ up)) if centers is not None else None
    center = centers.mean(axis=0) if centers is not None else points.mean(axis=0)

    horizontal, vertical = [], []
    for model, idx in found:
        alignment = abs(model[:3] @ up)
        if alignment > cos_tol:
            horizontal.append((orient(model, up), idx, float(np.median(points[idx] @ up))))
        elif alignment < sin_tol:
            vertical.append((orient_to_point(model, center), idx))

    floor_cands = [h for h in horizontal if cam_height is None or h[2] < cam_height]
    floor = min(floor_cands, key=lambda h: h[2]) if floor_cands else None
    ceiling = None
    if floor is not None:
        ceil_cands = [h for h in horizontal if h is not floor and h[2] > floor[2]
                      and (cam_height is None or h[2] > cam_height)]
        ceiling = max(ceil_cands, key=lambda h: h[2]) if ceil_cands else None
    else:
        log.warning("planes: no horizontal plane below the cameras, floor not found")
    if floor is not None and ceiling is None:
        log.warning("planes: ceiling not found")

    vertical.sort(key=lambda v: -len(v[1]))
    vertical = vertical[:pc["max_walls"]]
    nan4 = np.full(4, np.nan)
    floor_plane = floor[0] if floor is not None else nan4
    ceiling_plane = ceiling[0] if ceiling is not None else nan4
    walls = np.array([v[0] for v in vertical]).reshape(-1, 4)
    T = floor_transform(floor_plane, walls)

    labels = np.full(n_total, -1, dtype=np.int32)
    if floor is not None:
        labels[floor[1]] = 0
    if ceiling is not None:
        labels[ceiling[1]] = 1
    for k, (_, idx) in enumerate(vertical):
        labels[idx] = 2 + k

    out = run_dir / "planes"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "floor.npy", floor_plane)
    np.save(out / "ceiling.npy", ceiling_plane)
    np.save(out / "walls.npy", walls)
    np.save(out / "T_floor.npy", T)
    np.save(out / "labels.npy", labels)

    def height_of(idx):
        if floor is None:
            return "", ""
        h = float(np.median(points[idx] @ up) - floor[2])
        return round(h, 4), (round(h, 2) if unit == "cm" else "")

    entries = ([("floor", "horizontal", floor)] if floor is not None else []) + \
              ([("ceiling", "horizontal", ceiling)] if ceiling is not None else []) + \
              [(f"wall_{k}", "vertical", v) for k, v in enumerate(vertical)]
    planes_csv = artifacts_dir / "planes.csv"
    planes_csv.unlink(missing_ok=True)
    for name, kind, item in entries:
        model, idx = item[0], item[1]
        height, height_cm = height_of(idx)
        row = {"plane": name, "type": kind, "a": round(float(model[0]), 6), "b": round(float(model[1]), 6),
               "c": round(float(model[2]), 6), "d": round(float(model[3]), 6), "n_inliers": len(idx),
               "inlier_frac": round(len(idx) / n_total, 4), "height": height, "height_cm": height_cm}
        stage_io.upsert_csv(planes_csv, row, key="plane")
        stage_io.log_metrics(log, {f"{name}_{k}": v for k, v in row.items() if k not in ("plane", "type")})

    angles_csv = artifacts_dir / "wall_angles.csv"
    angles_csv.unlink(missing_ok=True)
    angles = wall_angles(walls, T)
    for row in angles:
        stage_io.upsert_csv(angles_csv, row, key="wall_i")
        stage_io.log_metrics(log, {f"{row['wall_i']}__{row['wall_j']}_{k}": row[k]
                                   for k in ("angle_deg", "error_to_90_deg")})

    errors = [r["error_to_90_deg"] for r in angles]
    room_height = height_of(ceiling[1]) if ceiling is not None else ("", "")
    summary = {"n_planes_found": len(found), "n_horizontal": len(horizontal), "n_walls": len(vertical),
               "floor_found": floor is not None, "ceiling_found": ceiling is not None,
               "room_height": room_height[0], "room_height_cm": room_height[1],
               "wall_angle_error_mean_deg": round(float(np.mean(errors)), 3) if errors else "",
               "wall_angle_error_max_deg": round(float(np.max(errors)), 3) if errors else "",
               "unlabelled_frac": round(float((labels < 0).mean()), 4)}
    stage_io.log_metrics(log, summary)

    return summary

# <<< END IMPLEMENTATION <<<

if __name__ == "__main__":
    import tempfile

    logging.basicConfig(level=logging.INFO)
    rng = np.random.default_rng(0)
    L, W, H, noise = 400.0, 300.0, 250.0, 0.5

    def rect(n, origin, u, v):
        a, b = rng.random((n, 1)), rng.random((n, 1))
        return origin + a * u + b * v + rng.normal(0, noise, (n, 3))

    ex, ey, ez = np.eye(3)
    yaw5 = np.radians(5.0)
    tilted = np.array([np.sin(yaw5), np.cos(yaw5), 0.0]) * W  # wall x = L rotated 5 deg about its centre
    parts = [rect(4000, np.zeros(3), L * ex, W * ey),                               # floor z = 0
             rect(4000, H * ez, L * ex, W * ey),                                    # ceiling z = H
             rect(2000, np.zeros(3), L * ex, H * ez),                               # wall y = 0
             rect(2000, W * ey, L * ex, H * ez),                                    # wall y = W
             rect(1800, np.zeros(3), W * ey, H * ez),                               # wall x = 0
             rect(1800, np.array([L, W / 2, 0]) - tilted / 2, tilted, H * ez),      # wall x ~ L, 5 deg
             rng.uniform([50, 50, 30], [350, 250, 200], (300, 3))]                  # clutter
    room = np.vstack(parts)
    floor_idx = np.arange(4000)

    # world = random rotation + translation of the room frame (z up)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    Rw = q * np.sign(np.linalg.det(q))
    tw = np.array([30.0, -20.0, 10.0])
    world = room @ Rw.T + tw

    # cameras at 150 cm looking horizontally; camera y axis = -up (OpenCV)
    cam_R, centers = [], []
    for k in range(20):
        yaw = 2 * np.pi * k / 20
        z_c = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        y_c = -ez + rng.normal(0, 0.02, 3)
        y_c -= (y_c @ z_c) * z_c
        y_c /= np.linalg.norm(y_c)
        x_c = np.cross(y_c, z_c)
        R_room = np.stack([x_c, y_c, z_c])           # rows = camera axes in the room frame
        cam_R.append(R_room @ Rw.T)                  # world -> camera
        centers.append(Rw @ np.array([200 + 50 * np.cos(yaw), 150 + 50 * np.sin(yaw), 150.0]) + tw)

    cfg = {"run": {"seed": 42}, "scale": {"provisional_extent_cm": 500},
           "planes": {"enabled": True, "source": "sparse", "distance_cm": 2.0, "ransac_n": 3,
                      "iterations": 2000, "max_walls": 4, "min_inlier_frac": 0.02, "angle_tol_deg": 15,
                      "max_planes": 10}}

    with tempfile.TemporaryDirectory() as tmp:
        rd, ad = Path(tmp) / "run", Path(tmp) / "art"
        for sub in ("clean/sparse", "sfm", "scale"):
            (rd / sub).mkdir(parents=True)
        np.save(rd / "clean/sparse/points.npy", world)
        np.save(rd / "sfm/cam_R.npy", np.array(cam_R))
        np.save(rd / "scale/centers_cm.npy", np.array(centers))
        np.save(rd / "scale/scale.npy", np.array([1.0]))

        summary = run(cfg, str(rd), str(ad))
        assert summary["floor_found"] and summary["ceiling_found"], summary
        assert summary["n_walls"] == 4, summary
        assert abs(summary["room_height_cm"] - H) < 2.0, summary

        T = np.load(rd / "planes/T_floor.npy")
        z = (world @ T[:3, :3].T + T[:3, 3])[:, 2]
        assert abs(np.median(z[floor_idx])) < 1.0 and np.abs(z[floor_idx]).max() < 5.0
        assert abs(np.median(z[4000:8000]) - H) < 2.0

        angles = stage_io.read_csv(ad / "wall_angles.csv")
        errors = sorted(float(r["error_to_90_deg"]) for r in angles)
        assert len(angles) == 4 and abs(errors[-1] - 5.0) < 1.0, angles
        assert sum(abs(e - 5.0) < 1.0 for e in errors) == 2 and errors[0] < 1.0, angles
        labels = np.load(rd / "planes/labels.npy")
        assert (labels[floor_idx] == 0).mean() > 0.95
        assert len(stage_io.read_csv(ad / "planes.csv")) == 6
        print(angles)
    print("OK")

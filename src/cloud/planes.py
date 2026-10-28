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
# Goal: find floor, ceiling and walls by RANSAC, the floor frame T_floor and the angle between neighbouring walls.
# In:  clean/<planes.source>/points.npy (working units; fallback scale/points3d_cm.npy), sfm/cam_R.npy,
#      scale/centers_cm.npy (fallback sfm/centers.npy).
# Out: planes/{floor,ceiling}.npy (4,) a,b,c,d (unit normal, NaN if not found), planes/walls.npy (W,4),
#      planes/T_floor.npy (4,4), planes/labels.npy (P,) int (-1 none, 0 floor, 1 ceiling, 2+k wall k);
#      artifacts: planes.csv (plane, type, a, b, c, d, n_inliers, inlier_frac, height, height_cm, + wall
#      segment x0 y0 x1 y1 in the floor frame, length, height_span, behind_frac, n_merged),
#      wall_angles.csv (pair, wall_i, wall_j, angle_deg, expected_deg, error_deg, error_to_90_deg, normal_angle_deg,
#      corner_x, corner_y, gap_i, gap_j, link).
# YAML: planes: {enabled, source: sparse, distance_cm, ransac_n, iterations, max_walls, min_inlier_frac,
#       angle_tol_deg, max_planes, wall_distance_cm, wall_rounds, wall_min_inlier_frac, wall_min_rel,
#       wall_band, wall_min_height_frac, wall_max_behind_frac, merge_angle_deg, merge_offset_frac,
#       segment_gap_frac, neighbour_min_deg, corner_gap_frac}
# Rules:
# - up = normalised mean of -R_i^T [0,1,0] (OpenCV camera y points down); [0,-1,0] + WARNING without cameras.
# - Floor/ceiling: planes are peeled one by one (Open3D segment_plane, inliers removed) up to max_planes
#   rounds, stopping when a plane has fewer than min_inlier_frac * P inliers; |n.up| > cos(tol) -> horizontal.
#   Floor = LOWEST horizontal plane (median inlier height along up) below the median camera height (a large
#   table or the ceiling may be peeled first). Ceiling = horizontal plane farthest above the floor and above
#   the cameras. Floor and ceiling normals point to +up.
# - Walls are vertical w.r.t. the FLOOR normal (camera up without floor): a vertical plane is a line in the
#   top view, so walls are peeled by 2D line RANSAC on the floor-projected non-floor/ceiling points of the
#   upper band wall_band (fractions of the room height: counters, tables and chairs stay below it), pairs
#   sampled among near neighbours so that small walls such as a chamfer are found, up to wall_rounds lines,
#   stopping below max(wall_min_inlier_frac * P_band, wall_min_rel * largest line). Each line keeps only the
#   main segment of its inlier band (split at gaps > segment_gap_frac * room size) and is refined by PCA.
#   Near-parallel lines (normal angle < merge_angle_deg) whose offsets differ by < merge_offset_frac * room
#   size are ONE wall (merged, refit). A wall spans >= wall_min_height_frac of the room height, has few
#   upper-band points behind it (behind_frac <= wall_max_behind_frac; a line across the room has the room
#   behind it) and no tall parallel candidate behind it over the same stretch (else it is an inner structure:
#   cabinet front, duplicated surface). Walls point to the camera centroid (room interior); largest first, up
#   to max_walls. Wall inliers (labels) = points of any height on the wall segment.
# - T_floor: rotation taking the floor normal to +z, then a rotation about z putting the largest wall
#   normal on +x, translation putting the floor at z = 0. Identity + WARNING without a floor.
# - Wall angles: only NEIGHBOURING walls: inward normals differ by neighbour_min_deg..180-neighbour_min_deg
#   (never parallel or opposite walls). Each wall has two ends (side of its segment centre the corner falls on)
#   and one neighbour per end: pairs whose inlier segments (2nd-98th percentile along the wall) both reach the
#   corner (intersection of the two lines) within corner_gap_frac * room size are corners (link gap); then the
#   wall ends still free are paired greedily by increasing gap up to END_GAP_FRAC * room size (link end: a short
#   or occluded wall, e.g. a chamfer, sees only part of its length). angle_deg = interior corner angle = 180 -
#   angle between inward normals, in [0, 180]; expected_deg = nearest NOMINAL_DEG (90 or the 135 deg chamfer),
#   error_deg = |angle_deg - expected_deg|; error_to_90_deg = |angle_deg - 90| only for expected 90 (walls that
#   should be orthogonal), empty otherwise.
# - Room size = largest 2nd-98th percentile extent of the wall candidate points in the top view.
# - height_cm is empty when the scale is unknown (unit = model).

# >>> BEGIN IMPLEMENTATION >>>
"""Planes: floor, ceiling and walls by RANSAC, floor frame and angles between neighbouring walls."""
import logging
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from tqdm import tqdm

from src import stage_io

log = logging.getLogger(__name__)

DEFAULTS = {"wall_distance_cm": 4.0, "wall_rounds": 40, "wall_band": [0.4, 0.95], "wall_min_inlier_frac": 0.005,
            "wall_min_rel": 0.08, "wall_min_height_frac": 0.3, "wall_max_behind_frac": 1.5, "merge_angle_deg": 10.0,
            "merge_offset_frac": 0.04, "segment_gap_frac": 0.08, "neighbour_min_deg": 30.0, "corner_gap_frac": 0.15}
NOMINAL_DEG = (90.0, 135.0)   # designed interior corner angles: square corner, 45 deg chamfer
END_GAP_FRAC = 0.5            # max corner gap (x room size) when pairing the free ends of short walls

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

def fit_line(xy: np.ndarray) -> np.ndarray:
    """Least-squares 2D line (nx, ny, c) with unit normal: n.p + c = 0."""

    mean = xy.mean(axis=0)
    _, _, vt = np.linalg.svd(xy - mean, full_matrices=False)
    n = vt[1]

    return np.array([n[0], n[1], -n @ mean])

def main_segment(t: np.ndarray, gap: float) -> np.ndarray:
    """Mask of the most populated run of positions t along a line, split where consecutive points are > gap apart."""

    order = np.argsort(t)
    cuts = np.flatnonzero(np.diff(t[order]) > gap) + 1
    runs = np.split(order, cuts)
    mask = np.zeros(len(t), dtype=bool)
    mask[max(runs, key=len)] = True

    return mask

def segment_inliers(xy: np.ndarray, line: np.ndarray, thr: float, gap: float) -> np.ndarray:
    """Indices of the band |n.p + c| < thr restricted to its main segment (a band crossing the room would
    otherwise collect clutter from the other side)."""

    band = np.flatnonzero(np.abs(xy @ line[:2] + line[2]) < thr)
    if len(band) < 2:
        return band

    return band[main_segment(xy[band] @ np.array([-line[1], line[0]]), gap)]

def ransac_line(xy: np.ndarray, thr: float, iterations: int, radius: float, gap: float,
                rng: np.random.Generator, max_eval: int = 30000, top_k: int = 16) -> np.ndarray | None:
    """2D line RANSAC: pairs = random point + a random neighbour within `radius`; band counts on a subsample,
    the top_k hypotheses re-scored by their main segment, the best refined by PCA on its segment."""

    tree = cKDTree(xy)
    first = rng.integers(0, len(xy), iterations)
    second = np.empty(iterations, dtype=np.int64)
    for k, nb in enumerate(tree.query_ball_point(xy[first], radius)):
        second[k] = nb[rng.integers(len(nb))]
    d = xy[second] - xy[first]
    norm = np.linalg.norm(d, axis=1)
    ok = norm > thr
    if not ok.any():
        return None

    n = np.stack([-d[ok, 1], d[ok, 0]], axis=1) / norm[ok, None]
    c = -(n * xy[first[ok]]).sum(axis=1)
    sub = xy[rng.choice(len(xy), min(len(xy), max_eval), replace=False)]
    counts = np.concatenate([(np.abs(sub @ n[s:s + 256].T + c[s:s + 256]) < thr).sum(axis=0)
                             for s in range(0, len(n), 256)])
    top = np.argsort(-counts)[:top_k]
    lines = [np.array([n[b, 0], n[b, 1], c[b]]) for b in top]
    line = max(lines, key=lambda l: len(segment_inliers(sub, l, thr, gap)))
    for _ in range(2):  # refine on the full set
        inl = segment_inliers(xy, line, thr, gap)
        if len(inl) < 2:
            return None
        line = fit_line(xy[inl])

    return line

def wall_stats(line: np.ndarray, xy: np.ndarray, h: np.ndarray, seg: np.ndarray, upper: np.ndarray, thr: float,
               room_height: float) -> dict:
    """Segment (2nd-98th percentile of the upper-band inliers `seg` along the line), inliers of every height on
    that segment, their height span (fraction of the room height) and upper-band points behind the wall per
    upper-band inlier (a wall has nothing behind it; a line across the room has the room behind it)."""

    n = line[:2]
    u = np.array([-n[1], n[0]])
    lo, hi = np.percentile(xy[seg] @ u, [2, 98])
    s_all = xy @ n + line[2]
    along = xy @ u
    within = (along >= lo) & (along <= hi)
    idx = np.flatnonzero(within & (np.abs(s_all) < thr))
    behind = int((within[upper] & (s_all[upper] < -3 * thr)).sum())
    span = np.percentile(h[idx], 95) - np.percentile(h[idx], 5)
    foot = -line[2] * n

    return {"p0": foot + lo * u, "p1": foot + hi * u, "lo": lo, "hi": hi, "length": hi - lo, "local": idx,
            "height_span": span / room_height if room_height > 0 else 1.0,
            "behind_frac": behind / max(len(seg), 1)}

def outer_of(a: dict, b: dict, cos_par: float, offset: float) -> bool:
    """True when b is parallel to a, at least `offset` behind it (outside the room) and overlapping at least half
    of the shorter segment: a is then an inner structure (cabinet front, counter, duplicated surface)."""

    na, nb = a["line"][:2], b["line"][:2]
    if na @ nb < cos_par:
        return False
    mid_b = (b["p0"] + b["p1"]) / 2
    if na @ mid_b + a["line"][2] > -offset:
        return False
    u = np.array([-na[1], na[0]])
    tb = sorted((b["p0"] @ u, b["p1"] @ u))
    overlap = min(a["hi"], tb[1]) - max(a["lo"], tb[0])

    return overlap >= 0.5 * min(a["length"], b["length"])

def detect_walls(points: np.ndarray, up: np.ndarray, exclude: np.ndarray, floor_h: float, room_height: float,
                 center: np.ndarray, thr: float, iterations: int, wc: dict, rng: np.random.Generator) -> dict:
    """Walls = vertical planes (w.r.t. up) found as top-view lines of the upper band; merged, filtered (tall,
    outermost), largest first."""

    e1 = np.cross(up, [1.0, 0.0, 0.0] if abs(up[0]) < 0.9 else [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    E = np.stack([e1, e2])
    h = points @ up - floor_h
    margin = 0.05 * room_height
    cand = np.flatnonzero(~exclude & (h > margin) & (h < room_height - margin))
    out = {"E": E, "walls": [], "candidates": [], "room_size": float("nan"), "n_candidates": len(cand),
           "n_lines": 0}
    if len(cand) < 10:
        return out

    xy = (points @ E.T)[cand]
    hc = h[cand]
    c2 = center @ E.T
    lo, hi = np.percentile(xy, [2, 98], axis=0)
    room = float(np.max(hi - lo))
    out["room_size"] = room
    band_lo, band_hi = wc["wall_band"]
    upper = np.flatnonzero((hc >= band_lo * room_height) & (hc <= band_hi * room_height))
    if len(upper) < 10:
        upper = np.arange(len(cand))
    xyu = xy[upper]
    min_abs = max(int(np.ceil(wc["wall_min_inlier_frac"] * len(upper))), 10)
    gap = wc["segment_gap_frac"] * room

    # 1) peel lines on the upper band (counters, tables and chairs stay below it)
    remaining = np.arange(len(upper))
    lines = []
    for _ in range(int(wc["wall_rounds"])):
        if len(remaining) < min_abs:
            break
        line = ransac_line(xyu[remaining], thr, iterations, 0.15 * room, gap, rng)
        if line is None:
            break
        inl = segment_inliers(xyu[remaining], line, thr, gap)
        if len(inl) < max(min_abs, wc["wall_min_rel"] * (lines[0][1].size if lines else 0)):
            break
        lines.append((line, remaining[inl]))
        remaining = np.delete(remaining, inl)
    out["n_lines"] = len(lines)

    # 2) orient to the room interior and merge near-parallel close lines (union-find)
    lines = [((l if l[:2] @ c2 + l[2] >= 0 else -l), i) for l, i in lines]
    parent = list(range(len(lines)))

    def root(a):
        while parent[a] != a:
            a = parent[a]
        return a

    cos_merge = np.cos(np.radians(wc["merge_angle_deg"]))
    offset = wc["merge_offset_frac"] * room
    for a in range(len(lines)):
        for b in range(a + 1, len(lines)):
            la, lb = lines[a][0], lines[b][0]
            if la[:2] @ lb[:2] > cos_merge:
                # offsets compared at the centroid of the two segments
                mid = np.vstack([xyu[lines[a][1]], xyu[lines[b][1]]]).mean(axis=0)
                if abs((la[:2] @ mid + la[2]) - (lb[:2] @ mid + lb[2])) < offset:
                    parent[root(b)] = root(a)

    groups = {}
    for k in range(len(lines)):
        groups.setdefault(root(k), []).append(k)

    walls = []
    for members in groups.values():
        seg = np.concatenate([lines[k][1] for k in members])
        if len(members) > 1:  # refit and re-collect the merged wall
            line = fit_line(xyu[seg])
            seg = segment_inliers(xyu, line, thr, gap)
            line = fit_line(xyu[seg]) if len(seg) >= 2 else line
        else:
            line = lines[members[0]][0]
        line = line if line[:2] @ c2 + line[2] >= 0 else -line
        st = wall_stats(line, xy, hc, upper[seg], upper, thr, room_height)
        st.update(line=line, idx=cand[st.pop("local")], n_upper=len(seg), n_merged=len(members))
        walls.append(st)

    # 3) tall enough and outermost (no parallel candidate behind it over the same stretch)
    for w in walls:
        w["tall"] = w["height_span"] >= wc["wall_min_height_frac"] and w["behind_frac"] <= wc["wall_max_behind_frac"]
    for w in walls:
        w["inner"] = any(v is not w and v["tall"] and outer_of(w, v, cos_merge, offset) for v in walls)
        w["ok"] = w["tall"] and not w["inner"]
        log.info("planes: wall candidate n=%d merged=%d len=%.3g height_span=%.2f behind=%.2f -> %s",
                 len(w["idx"]), w["n_merged"], w["length"], w["height_span"], w["behind_frac"],
                 "wall" if w["ok"] else ("inner" if w["inner"] else "short or crossing"))

    out["walls"] = sorted((w for w in walls if w["ok"]), key=lambda w: -w["n_upper"])
    out["candidates"] = walls

    return out

def wall_angles(walls: list[dict], neighbour_min_deg: float, gap_max: float, end_gap_max: float) -> list[dict]:
    """Interior corner angle between neighbouring walls: non-parallel normals, one neighbour per wall end; segments
    meeting at the corner first, then the free ends of short walls by increasing gap."""

    cands = []
    for i in range(len(walls)):
        for j in range(i + 1, len(walls)):
            li, lj = walls[i]["line"], walls[j]["line"]
            between = float(np.degrees(np.arccos(np.clip(li[:2] @ lj[:2], -1.0, 1.0))))
            if not neighbour_min_deg <= between <= 180.0 - neighbour_min_deg:
                continue
            corner = np.linalg.solve(np.stack([li[:2], lj[:2]]), -np.array([li[2], lj[2]]))
            gaps, ends = [], []
            for k in (i, j):
                n = walls[k]["line"][:2]
                t = corner @ np.array([-n[1], n[0]])
                gaps.append(max(0.0, walls[k]["lo"] - t, t - walls[k]["hi"]))
                ends.append((k, bool(t > (walls[k]["lo"] + walls[k]["hi"]) / 2)))   # (wall, end the corner is on)
            cands.append((max(gaps), i, j, between, corner, gaps, ends))
    cands.sort(key=lambda c: c[0])

    rows, used, done = [], set(), set()
    for link, limit in (("gap", gap_max), ("end", end_gap_max)):
        for g, i, j, between, corner, gaps, ends in cands:
            if g > limit or (i, j) in done or (link == "end" and used & set(ends)):
                continue
            done.add((i, j))
            used |= set(ends)
            angle = 180.0 - between
            expected = min(NOMINAL_DEG, key=lambda a: abs(angle - a))
            rows.append({"pair": f"wall_{i}__wall_{j}", "wall_i": f"wall_{i}", "wall_j": f"wall_{j}",
                         "angle_deg": round(angle, 3), "expected_deg": expected,
                         "error_deg": round(abs(angle - expected), 3),
                         "error_to_90_deg": round(abs(angle - 90.0), 3) if expected == 90.0 else "",
                         "normal_angle_deg": round(between, 3), "corner_x": round(float(corner[0]), 4),
                         "corner_y": round(float(corner[1]), 4), "gap_i": round(gaps[0], 4),
                         "gap_j": round(gaps[1], 4), "link": link})

    return rows

def run(cfg: dict, run_dir: str, artifacts_dir: str, debug: dict | None = None) -> dict | None:
    """debug: optional dict filled with the wall candidates (top-view basis E, segments, ok flags) for plots."""

    pc = {**DEFAULTS, **cfg["planes"]}
    if not pc["enabled"]:
        log.info("planes disabled")
        return None

    run_dir, artifacts_dir = Path(run_dir), Path(artifacts_dir)
    points = load_points(run_dir, pc["source"])
    if points is None or len(points) < pc["ransac_n"]:
        log.warning("planes: no point cloud under %s, skipping", run_dir)
        return None

    seed = int(cfg["run"]["seed"])
    o3d.utility.random.seed(seed)
    rng = np.random.default_rng(seed)
    cam_R, centers = load_cameras(run_dir)
    up = gravity_up(cam_R)
    upc = stage_io.units_per_cm(run_dir, points, room_extent_cm=cfg["scale"]["provisional_extent_cm"])
    _, unit = stage_io.scale_info(run_dir)
    n_total = len(points)
    min_inliers = max(int(np.ceil(pc["min_inlier_frac"] * n_total)), pc["ransac_n"])

    # floor and ceiling: largest planes peeled by 3D RANSAC, horizontal w.r.t. the camera up
    found = peel_planes(points, pc["distance_cm"] * upc, pc["ransac_n"], pc["iterations"], pc["max_planes"],
                        min_inliers)
    cos_tol = np.cos(np.radians(pc["angle_tol_deg"]))
    cam_height = float(np.median(centers @ up)) if centers is not None else None
    center = centers.mean(axis=0) if centers is not None else points.mean(axis=0)

    horizontal = [(orient(m, up), idx, float(np.median(points[idx] @ up)))
                  for m, idx in found if abs(m[:3] @ up) > cos_tol]
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

    # walls: vertical w.r.t. the floor normal, between floor and ceiling (robust top of the cloud without one)
    wall_up = floor[0][:3] if floor is not None else up
    hs = points @ wall_up
    floor_h = -floor[0][3] if floor is not None else float(np.percentile(hs, 2))
    top_h = -ceiling[0][3] * np.sign(ceiling[0][:3] @ wall_up) if ceiling is not None \
        else float(np.percentile(hs, 98))
    exclude = np.zeros(n_total, dtype=bool)
    for item in (floor, ceiling):
        if item is not None:
            exclude[item[1]] = True
    wd = detect_walls(points, wall_up, exclude, floor_h, top_h - floor_h, center, pc["wall_distance_cm"] * upc,
                      pc["iterations"], pc, rng)
    vertical = wd["walls"][:pc["max_walls"]]
    if debug is not None:
        debug.update(wd, points=points)
    if len(wd["walls"]) > pc["max_walls"]:
        log.warning("planes: %d walls found, keeping the largest max_walls=%d", len(wd["walls"]), pc["max_walls"])
    for w in vertical:  # 3D plane from the top-view line: n = nx e1 + ny e2, d = c
        w["plane"] = np.append(w["line"][:2] @ wd["E"], w["line"][2])

    nan4 = np.full(4, np.nan)
    floor_plane = floor[0] if floor is not None else nan4
    ceiling_plane = ceiling[0] if ceiling is not None else nan4
    walls = np.array([w["plane"] for w in vertical]).reshape(-1, 4)
    T = floor_transform(floor_plane, walls)

    labels = np.full(n_total, -1, dtype=np.int32)
    if floor is not None:
        labels[floor[1]] = 0
    if ceiling is not None:
        labels[ceiling[1]] = 1
    for k, w in reversed(list(enumerate(vertical))):  # larger walls win shared points
        labels[w["idx"]] = 2 + k

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

    # wall geometry in the floor frame (x, y of T_floor)
    to_floor = T[:2, :3] @ wd["E"].T if floor is not None else None
    for w in vertical:
        w["line_f"] = w["line"].copy()
        if to_floor is not None:  # rotate the 2D line; T's translation is along z only
            w["line_f"][:2] = to_floor @ w["line"][:2]
            w["p0f"], w["p1f"] = to_floor @ w["p0"], to_floor @ w["p1"]
        else:
            w["p0f"], w["p1f"] = w["p0"], w["p1"]
        if to_floor is not None and np.linalg.det(to_floor) < 0:
            w["lo"], w["hi"] = -w["hi"], -w["lo"]

    entries = ([("floor", "horizontal", floor[0], floor[1], {})] if floor is not None else []) + \
              ([("ceiling", "horizontal", ceiling[0], ceiling[1], {})] if ceiling is not None else []) + \
              [(f"wall_{k}", "vertical", w["plane"], w["idx"], w) for k, w in enumerate(vertical)]
    planes_csv = artifacts_dir / "planes.csv"
    planes_csv.unlink(missing_ok=True)
    for name, kind, model, idx, w in entries:
        height, height_cm = height_of(idx)
        row = {"plane": name, "type": kind, "a": round(float(model[0]), 6), "b": round(float(model[1]), 6),
               "c": round(float(model[2]), 6), "d": round(float(model[3]), 6), "n_inliers": len(idx),
               "inlier_frac": round(len(idx) / n_total, 4), "height": height, "height_cm": height_cm,
               "x0": "", "y0": "", "x1": "", "y1": "", "length": "", "height_span": "", "behind_frac": "",
               "n_merged": ""}
        if w:
            row.update(x0=round(float(w["p0f"][0]), 4), y0=round(float(w["p0f"][1]), 4),
                       x1=round(float(w["p1f"][0]), 4), y1=round(float(w["p1f"][1]), 4),
                       length=round(float(w["length"]), 4), height_span=round(float(w["height_span"]), 3),
                       behind_frac=round(float(w["behind_frac"]), 3), n_merged=w["n_merged"])
        stage_io.upsert_csv(planes_csv, row, key="plane")
        stage_io.log_metrics(log, {f"{name}_{k}": row[k] for k in ("a", "b", "c", "d", "n_inliers", "inlier_frac",
                                                                   "height", "height_cm")})

    angles_csv = artifacts_dir / "wall_angles.csv"
    angles_csv.unlink(missing_ok=True)
    room = wd["room_size"]
    angles = wall_angles([{"line": w["line_f"], "lo": w["lo"], "hi": w["hi"]} for w in vertical],
                         pc["neighbour_min_deg"], pc["corner_gap_frac"] * room, END_GAP_FRAC * room) \
        if np.isfinite(room) else []
    for row in angles:
        stage_io.upsert_csv(angles_csv, row, key="pair")
        stage_io.log_metrics(log, {f"{row['pair']}_{k}": row[k] for k in ("angle_deg", "expected_deg", "error_deg")})

    errors = [r["error_deg"] for r in angles]
    errors_90 = [r["error_to_90_deg"] for r in angles if r["error_to_90_deg"] != ""]
    room_height = height_of(ceiling[1]) if ceiling is not None else ("", "")
    summary = {"n_planes_found": len(found), "n_horizontal": len(horizontal), "n_wall_lines": wd["n_lines"],
               "n_walls": len(vertical), "n_wall_pairs": len(angles),
               "floor_found": floor is not None, "ceiling_found": ceiling is not None,
               "room_height": room_height[0], "room_height_cm": room_height[1],
               "wall_angles_deg": " ".join(f"{r['angle_deg']:.1f}" for r in angles),
               "wall_angle_error_mean_deg": round(float(np.mean(errors)), 3) if errors else "",
               "wall_angle_error_max_deg": round(float(np.max(errors)), 3) if errors else "",
               "n_wall_pairs_90": len(errors_90),
               "wall_angle_error90_mean_deg": round(float(np.mean(errors_90)), 3) if errors_90 else "",
               "wall_angle_error90_max_deg": round(float(np.max(errors_90)), 3) if errors_90 else "",
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
        errors = sorted(float(r["error_to_90_deg"]) for r in angles)   # rectangle: every corner expected 90
        assert len(angles) == 4 and abs(errors[-1] - 5.0) < 1.0, angles
        assert sum(abs(e - 5.0) < 1.0 for e in errors) == 2 and errors[0] < 1.0, angles
        labels = np.load(rd / "planes/labels.npy")
        assert (labels[floor_idx] == 0).mean() > 0.95
        assert len(stage_io.read_csv(ad / "planes.csv")) == 6
        print(angles)

    # kitchen-like room: 5 walls (45 deg chamfer -> corners 90, 90, 90, 135, 135), the long wall peeled twice
    # (parallel duplicate 3 cm behind), an upper cabinet front 35 cm in front of a wall, max_walls 8
    C = 100.0
    chamfer = np.array([-C, C, 0.0])
    parts = [rect(5000, np.zeros(3), L * ex, W * ey), rect(5000, H * ez, L * ex, W * ey),
             rect(2000, np.zeros(3), L * ex, H * ez),                                 # y = 0
             rect(1200, W * ey, (L - C) * ex, H * ez),                                # y = W (up to the chamfer)
             rect(1200, W * ey - 3 * ey, (L - C) * ex, H * ez),                       # duplicate, 3 cm inside
             rect(1600, np.zeros(3), W * ey, H * ez),                                 # x = 0
             rect(1300, np.zeros(3) + L * ex, (W - C) * ey, H * ez),                  # x = L (up to the chamfer)
             rect(800, np.array([L, W - C, 0]), chamfer, H * ez),                     # chamfer
             rect(1000, np.array([50, 35, 150]), 200 * ex, 70 * ez),                  # upper cabinet front
             rng.uniform([50, 50, 30], [350, 250, 200], (300, 3))]
    world = np.vstack(parts) @ Rw.T + tw
    cfg["planes"]["max_walls"] = 8
    with tempfile.TemporaryDirectory() as tmp:
        rd, ad = Path(tmp) / "run", Path(tmp) / "art"
        for sub in ("clean/sparse", "sfm", "scale"):
            (rd / sub).mkdir(parents=True)
        np.save(rd / "clean/sparse/points.npy", world)
        np.save(rd / "sfm/cam_R.npy", np.array(cam_R))
        np.save(rd / "scale/centers_cm.npy", np.array(centers))
        np.save(rd / "scale/scale.npy", np.array([1.0]))

        summary = run(cfg, str(rd), str(ad))
        assert summary["n_walls"] == 5, summary
        angles = sorted(float(r["angle_deg"]) for r in stage_io.read_csv(ad / "wall_angles.csv"))
        assert len(angles) == 5, angles
        assert all(abs(a - e) < 2.0 for a, e in zip(angles, [90, 90, 90, 135, 135])), angles
        rows = stage_io.read_csv(ad / "wall_angles.csv")
        assert sorted(float(r["expected_deg"]) for r in rows) == [90, 90, 90, 135, 135], rows
        assert all(float(r["error_deg"]) < 2.0 for r in rows) and sum(r["error_to_90_deg"] == "" for r in rows) == 2
        assert summary["n_wall_pairs_90"] == 3 and summary["wall_angle_error_max_deg"] < 2.0, summary
        print(angles)
    print("OK")

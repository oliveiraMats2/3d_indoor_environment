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

# ==============================================================================
# PROMPT COMPLEMENT
# Stage: floorplan (after planes, before dense).
# Goal: top-down floor plan of the room and its main measures.
#   - Points of clean/sparse are moved to the floor frame (planes/T_floor.npy,
#     floor z = 0, up +z) and projected to XY; a 2D histogram (cell = cell_cm in
#     working units, ceiling points excluded) is rendered with OpenCV: log
#     density, colormap, camera trajectory (start/end markers), room rectangle
#     with its width/length written on the sides (cm when metric), scale bar (only when unit = cm) and axis text.
#   - width/length (method column of floorplan.csv):
#     walls: orientation from cv2.minAreaRect of the XY of the wall points
#     (labels >= 2), sides = extent_percentiles span of those points in the
#     rectangle frame, each side refined to the median of the points within
#     one cell of it (robust to outliers and noise). Used only with >=
#     min_wall_directions distinct wall directions (parallel walls = one) and
#     a non degenerate rectangle (width/length >= min_aspect, width > 1 cell).
#     Otherwise (WARNING) the fallback list, in order: room_fill = minAreaRect
#     of the fitted room_fill outline (mesh/room_fill/{T_room,footprint_cm}.npy,
#     ignored when older than sfm/centers.npy; NOTE its size comes from the
#     hand-measured outline, circular when the scale also comes from it);
#     percentiles = same robust span on all non floor/ceiling points at
#     fallback_percentiles, oriented by the dominant wall when there is one.
#   - unit: scale/scale.npy is read on every run (cm per model unit). The
#     factor already applied to the working cloud = spread of
#     scale/centers_cm.npy / spread of sfm/centers.npy; cm per working unit =
#     scale / applied, so a scale.npy written after the cloud was built still
#     yields cm (WARNING).
#   - area_m2 = area of that rectangle; floor_cells_m2 = occupied cells of the
#     floor points (label 0) times the cell area (extra column; underestimates
#     where the sparse floor has holes, so area_m2 is the main value).
#   - ceiling_height_cm = distance of the ceiling plane to a point of the floor
#     plane (empty when no ceiling).
# In:  clean/sparse/{points,colors}.npy, planes/{T_floor,floor,ceiling,walls,
#      labels}.npy (labels: -1 none, 0 floor, 1 ceiling, 2+k wall k),
#      scale/centers_cm.npy (else sfm/centers.npy) for the trajectory,
#      scale/scale.npy (via stage_io.scale_info), sfm/centers.npy,
#      optional mesh/room_fill/{T_room,footprint_cm}.npy.
# Out: figures/floorplan.png, floorplan/{grid,grid_meta,trajectory_xy}.npy
#      (grid_meta = [x0, y0, cell] in working units, grid[row, col] with row
#      along +y); one row in artifacts/<id>/floorplan.csv.
# YAML: floorplan: {enabled: true, cell_cm: 5.0, dpi: 150, fig_inches: 8.0,
#       extent_percentiles: [1, 99], min_wall_directions: 2, min_aspect: 0.15,
#       wall_angle_tol_deg: 20.0, fallback: [room_fill, percentiles],
#       fallback_percentiles: [5, 95]}; scale.provisional_extent_cm.
# Rules: unit = model -> cm/m2 columns empty + WARNING (the figure is still
#   drawn and the *_model columns are filled); missing optional inputs ->
#   WARNING, never invented values; OpenCV only for figures.
# ==============================================================================

# >>> BEGIN IMPLEMENTATION >>>
"""Floor plan: top-down density map, trajectory, room width/length/area and ceiling height."""
import logging
from pathlib import Path

import cv2
import numpy as np

from src import stage_io

log = logging.getLogger(__name__)

DEFAULTS = {
    "min_wall_directions": 2,          # fewer distinct wall directions -> fallback
    "min_aspect": 0.15,                # width / length below this -> degenerate rectangle, fallback
    "wall_angle_tol_deg": 20.0,        # walls closer than this in direction are one direction (parallel)
    "fallback": ["room_fill", "percentiles"],   # tried in order: room_fill outline, percentile footprint
    "fallback_percentiles": [5, 95],   # percentile footprint of ALL non floor/ceiling points: clutter and points
                                       # seen through windows/doors need a stronger trim than the wall points
}

def load_optional(path: Path, what: str) -> np.ndarray | None:

    if path.exists():
        return np.load(path)

    log.warning("floorplan: %s missing (%s)", what, path)

    return None

def to_floor(T: np.ndarray, points: np.ndarray) -> np.ndarray:

    return points @ T[:3, :3].T + T[:3, 3]

def ceiling_height(floor: np.ndarray | None, ceiling: np.ndarray | None) -> float:
    """Distance from the point of the floor plane closest to the origin to the ceiling plane."""

    if floor is None or ceiling is None or not (np.all(np.isfinite(floor)) and np.all(np.isfinite(ceiling))):
        return float("nan")

    n_f = floor[:3] / np.linalg.norm(floor[:3])
    p0 = -floor[3] / np.linalg.norm(floor[:3]) * n_f
    n_c = np.linalg.norm(ceiling[:3])

    return float(abs(ceiling[:3] @ p0 + ceiling[3]) / n_c)

def robust_rect(pts: np.ndarray, percentiles, band: float, angle: float | None = None) -> tuple:
    """Orientation from cv2.minAreaRect (or the given angle, deg); each side = median of the points within
    +-band of the per-axis percentile in the rectangle frame (removes the noise bias of the percentile)."""

    if angle is None:
        (_, _), (_, _), angle = cv2.minAreaRect(pts.astype(np.float32))
    a = np.deg2rad(angle)
    rot = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    local = pts @ rot.T
    lo, hi = np.percentile(local, percentiles, axis=0)
    for k in range(2):
        for ends in (lo, hi):
            near = local[np.abs(local[:, k] - ends[k]) <= band, k]
            ends[k] = np.median(near) if len(near) else ends[k]
    center = rot.T @ ((lo + hi) / 2)

    return (tuple(center), tuple(hi - lo), float(angle))

def wall_directions(xy: np.ndarray, labels: np.ndarray, tol_deg: float) -> list[tuple[float, int]]:
    """Distinct wall directions in the floor plane: (angle of the wall line in deg mod 180, n points), most
    supported first. Each wall label gives the main axis of its XY points (PCA); labels whose lines differ by
    less than tol_deg (parallel walls) are one direction."""

    walls = []
    for k in np.unique(labels[labels >= 2]):
        p = xy[labels == k]
        if len(p) < 3:
            continue
        _, vec = np.linalg.eigh(np.cov((p - p.mean(0)).T))
        walls.append((float(np.degrees(np.arctan2(vec[1, 1], vec[0, 1])) % 180), len(p)))

    dirs: list[list[float]] = []                      # [angle, n]
    for ang, n in sorted(walls, key=lambda w: -w[1]):
        for d in dirs:
            if min(abs(ang - d[0]), 180 - abs(ang - d[0])) < tol_deg:
                d[1] += n
                break
        else:
            dirs.append([ang, n])

    return [(a, int(n)) for a, n in dirs]

def degenerate(rect: tuple, min_aspect: float, min_side: float) -> bool:

    a, b = sorted(rect[1])

    return not (np.isfinite(a) and a > min_side and b > 0 and a / b >= min_aspect)

def room_fill_outline(run_dir: Path, T: np.ndarray) -> np.ndarray | None:
    """Fitted room_fill outline corners in the floor frame (XY, working units), None when absent or older than
    the SfM model it was fitted to."""

    d = run_dir / "mesh" / "room_fill"
    T_room_p, poly_p, sfm_p = d / "T_room.npy", d / "footprint_cm.npy", run_dir / "sfm" / "centers.npy"
    if not (T_room_p.exists() and poly_p.exists()):
        return None
    if sfm_p.exists() and T_room_p.stat().st_mtime < sfm_p.stat().st_mtime:
        log.warning("floorplan: room_fill outline older than the sfm model, ignored (%s)", T_room_p)
        return None

    poly = np.asarray(np.load(poly_p), np.float64).reshape(-1, 2)
    corners = np.c_[poly, np.zeros(len(poly)), np.ones(len(poly))] @ np.load(T_room_p).T

    return to_floor(T, corners[:, :3])[:, :2]

def room_rect(xy: np.ndarray, labels: np.ndarray, percentiles, band: float, opts: dict,
              outline: np.ndarray | None) -> tuple[tuple, str, int]:
    """Room rectangle and the method used: walls (minAreaRect of the wall points, >= min_wall_directions
    distinct wall directions and a non degenerate rectangle), else the first usable fallback: room_fill
    (minAreaRect of the fitted room_fill outline) or percentiles (robust extent of all non floor/ceiling
    points, oriented by the dominant wall when there is one)."""

    dirs = wall_directions(xy, labels, opts["wall_angle_tol_deg"])
    walls = xy[labels >= 2]
    if len(dirs) >= opts["min_wall_directions"] and len(walls) >= 3:
        rect = robust_rect(walls, percentiles, band)
        if not degenerate(rect, opts["min_aspect"], band):
            return rect, "walls", len(dirs)
        log.warning("floorplan: wall rectangle degenerate (%.4g x %.4g), using fallback", *sorted(rect[1]))
    else:
        log.warning("floorplan: %d wall direction(s) (< %d), using fallback", len(dirs), opts["min_wall_directions"])

    for method in opts["fallback"]:
        if method == "room_fill" and outline is not None and len(outline) >= 3:
            rect = cv2.minAreaRect(outline.astype(np.float32))
            rect = (tuple(map(float, rect[0])), tuple(map(float, rect[1])), float(rect[2]))
            if not degenerate(rect, opts["min_aspect"], band):
                return rect, "room_fill", len(dirs)
        elif method == "percentiles":
            rest = xy[(labels != 0) & (labels != 1)]
            rest = xy if len(rest) < 3 else rest
            angle = dirs[0][0] if dirs else None
            pct = opts["fallback_percentiles"]
            log.warning("floorplan: extent from %d non floor/ceiling points at percentiles %s, orientation %s",
                        len(rest), list(pct), "dominant wall" if dirs else "minAreaRect")
            rect = robust_rect(rest, pct, band, angle)
            if not degenerate(rect, opts["min_aspect"], band) or method == opts["fallback"][-1]:
                return rect, "percentiles", len(dirs)

    log.warning("floorplan: every fallback degenerate, using the wall rectangle")

    return robust_rect(walls if len(walls) >= 3 else xy, percentiles, band), "walls", len(dirs)

def cm_per_unit(run_dir: Path) -> float:
    """cm per working unit of clean/sparse, nan when the scale is unknown. scale/scale.npy (read now, never
    cached) gives cm per model unit; the factor the scale stage applied to the working cloud is the ratio of
    scale/centers_cm.npy to sfm/centers.npy, so a scale.npy written after the cloud was built (still in model
    units) is applied here instead of being taken as cm."""

    s, unit = stage_io.scale_info(run_dir)
    if unit != "cm":
        return float("nan")

    c_p, m_p = run_dir / "scale" / "centers_cm.npy", run_dir / "sfm" / "centers.npy"
    if not (c_p.exists() and m_p.exists()):
        return 1.0
    c, m = np.load(c_p).reshape(-1, 3), np.load(m_p).reshape(-1, 3)
    spread = np.linalg.norm(m - m.mean(0))
    if len(c) != len(m) or spread <= 0:
        return 1.0
    applied = float(np.linalg.norm(c - c.mean(0)) / spread)
    if abs(applied - s) <= 0.01 * abs(s):
        return 1.0
    log.warning("floorplan: working cloud built with factor %.4g but scale.npy = %.4g cm per model unit, "
                "converting with %.4g cm per working unit", applied, s, s / applied)

    return s / applied

def histogram(xy: np.ndarray, x0: float, y0: float, cell: float, shape: tuple[int, int]) -> np.ndarray:

    col = np.floor((xy[:, 0] - x0) / cell).astype(int)
    row = np.floor((xy[:, 1] - y0) / cell).astype(int)
    ok = (col >= 0) & (col < shape[1]) & (row >= 0) & (row < shape[0])
    grid = np.zeros(shape, dtype=np.int32)
    np.add.at(grid, (row[ok], col[ok]), 1)

    return grid

def nice_length(target: float) -> float:

    base = 10 ** np.floor(np.log10(target))
    for m in (5, 2, 1):
        if m * base <= target:
            return float(m * base)

    return float(base)

def render(grid: np.ndarray, x0: float, y0: float, cell: float, traj: np.ndarray | None, rect: tuple,
           unit: str, upc: float, cfg: dict, out_path: Path, title: str, dims: tuple[str, str] = ("", "")) -> None:
    """dims = (width, length) labels drawn at the middle of a short and a long side of the rectangle.
    The canvas size depends only on the grid (src/videos.py rebuilds the same mapping)."""


    long_px = int(cfg["fig_inches"] * cfg["dpi"])
    px = max(long_px / max(grid.shape), 1.0)                  # pixels per cell
    h, w = int(round(grid.shape[0] * px)), int(round(grid.shape[1] * px))
    font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, cfg["dpi"] / 300, max(1, int(cfg["dpi"] / 150))
    margin = int(0.5 * cfg["dpi"])

    dens = np.log1p(grid.astype(np.float64))
    dens = (255 * dens / max(dens.max(), 1e-9)).astype(np.uint8)
    color = cv2.applyColorMap(dens, cv2.COLORMAP_INFERNO)
    color[grid == 0] = (40, 40, 40)
    color = cv2.resize(color[::-1], (w, h), interpolation=cv2.INTER_NEAREST)   # +y up

    canvas = np.full((h + 2 * margin, w + 2 * margin, 3), 255, np.uint8)
    canvas[margin:margin + h, margin:margin + w] = color

    def to_px(pts: np.ndarray) -> np.ndarray:
        u = margin + (pts[:, 0] - x0) / cell * px
        v = margin + h - (pts[:, 1] - y0) / cell * px
        return np.round(np.stack([u, v], 1)).astype(np.int32)

    corners = cv2.boxPoints(rect).astype(np.float64)
    box = to_px(corners)
    cv2.polylines(canvas, [box], True, (255, 255, 0), th, cv2.LINE_AA)
    sides = sorted(range(4), key=lambda i: np.linalg.norm(corners[(i + 1) % 4] - corners[i]))
    for i, text in ((sides[0], dims[0]), (sides[-1], dims[1])):
        if not text:
            continue
        mid = (box[i] + box[(i + 1) % 4]) / 2
        (tw, tht), _ = cv2.getTextSize(text, font, fs, th)
        u = int(np.clip(mid[0] - tw / 2, 2, canvas.shape[1] - tw - 2))
        v = int(np.clip(mid[1] + tht / 2, tht + 2, canvas.shape[0] - 4))
        cv2.rectangle(canvas, (u - 3, v - tht - 3), (u + tw + 3, v + 4), (255, 255, 255), -1)
        cv2.putText(canvas, text, (u, v), font, fs, (160, 90, 0), th, cv2.LINE_AA)
    if traj is not None and len(traj):
        tp = to_px(traj)
        cv2.polylines(canvas, [tp], False, (255, 255, 255), th, cv2.LINE_AA)
        r = max(3, int(cfg["dpi"] / 25))
        cv2.circle(canvas, tuple(int(v) for v in tp[0]), r, (0, 200, 0), -1, cv2.LINE_AA)
        cv2.circle(canvas, tuple(int(v) for v in tp[-1]), r, (0, 0, 230), -1, cv2.LINE_AA)
        cv2.putText(canvas, "start", tuple(int(v) for v in tp[0] + r + 2), font, fs, (0, 200, 0), th, cv2.LINE_AA)
        cv2.putText(canvas, "end", (int(tp[-1][0]) + r + 2, int(tp[-1][1]) - r - 2), font, fs, (0, 0, 230), th, cv2.LINE_AA)

    if unit == "cm":
        bar_cm = nice_length(w / px * cell / upc / 4)
        bar_px = int(round(bar_cm * upc / cell * px))
        y = margin + h + margin // 2
        cv2.line(canvas, (margin, y), (margin + bar_px, y), (0, 0, 0), 2 * th)
        cv2.putText(canvas, f"{bar_cm:g} cm", (margin + bar_px + 8, y + 5), font, fs, (0, 0, 0), th, cv2.LINE_AA)

    u_name = "cm" if unit == "cm" else "model units"
    k = 1.0 / upc if unit == "cm" else 1.0            # axis values in cm when metric
    x1, y1 = (x0 + grid.shape[1] * cell) * k, (y0 + grid.shape[0] * cell) * k
    cv2.putText(canvas, title, (margin, int(margin * 0.6)), font, fs * 1.2, (0, 0, 0), th, cv2.LINE_AA)
    cv2.putText(canvas, f"x [{x0 * k:.3g}, {x1:.3g}] {u_name}", (margin + w // 2, margin + h + margin // 2 + 5),
                font, fs, (0, 0, 0), th, cv2.LINE_AA)
    label = f"y [{y0 * k:.3g}, {y1:.3g}] {u_name}"
    (tw, _), _ = cv2.getTextSize(label, font, fs, th)
    side = np.full((margin, tw + 10, 3), 255, np.uint8)
    cv2.putText(side, label, (5, margin // 2 + 5), font, fs, (0, 0, 0), th, cv2.LINE_AA)
    side = cv2.rotate(side, cv2.ROTATE_90_COUNTERCLOCKWISE)
    sh = min(side.shape[0], h)
    top = margin + (h - sh) // 2
    canvas[top:top + sh, :side.shape[1]] = side[:sh]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)

def run(cfg: dict, run_dir: str, artifacts_dir: str) -> dict | None:

    fp = cfg["floorplan"]
    if not fp["enabled"]:
        log.info("floorplan disabled")
        return None

    run_dir = Path(run_dir)
    points_path = run_dir / "clean" / "sparse" / "points.npy"
    if not points_path.exists():
        log.warning("floorplan: %s missing, skipping", points_path)
        return None

    points = np.load(points_path).astype(np.float64).reshape(-1, 3)
    planes = run_dir / "planes"
    T = load_optional(planes / "T_floor.npy", "T_floor (identity used)")
    T = np.eye(4) if T is None else T
    labels = load_optional(planes / "labels.npy", "plane labels")
    if labels is None or len(labels) != len(points):
        if labels is not None:
            log.warning("floorplan: labels (%d) not aligned with points (%d), ignored", len(labels), len(points))
        labels = np.full(len(points), -1)
    labels = np.asarray(labels).astype(int)
    floor, ceiling = load_optional(planes / "floor.npy", "floor plane"), load_optional(planes / "ceiling.npy", "ceiling plane")
    centers = load_optional(run_dir / "scale" / "centers_cm.npy", "scale/centers_cm (sfm/centers used)")
    if centers is None:
        centers = load_optional(run_dir / "sfm" / "centers.npy", "camera centers")
    opts = {**DEFAULTS, **{k: fp[k] for k in DEFAULTS if fp.get(k) is not None}}

    k_cm = cm_per_unit(run_dir)                       # cm per working unit, nan without scale
    metric = bool(np.isfinite(k_cm))
    unit = "cm" if metric else "model"
    upc = 1.0 / k_cm if metric else stage_io.units_per_cm(run_dir, points, room_extent_cm=cfg["scale"]["provisional_extent_cm"])
    cell = fp["cell_cm"] * upc

    xy = to_floor(T, points)[:, :2]
    traj = None if centers is None else to_floor(T, np.asarray(centers, np.float64).reshape(-1, 3))[:, :2]

    outline = room_fill_outline(run_dir, T) if "room_fill" in opts["fallback"] else None
    rect, method, n_dirs = room_rect(xy, labels, fp["extent_percentiles"], cell, opts, outline)
    side_a, side_b = rect[1]
    width_u, length_u = float(min(side_a, side_b)), float(max(side_a, side_b))
    height_u = ceiling_height(floor, ceiling)
    if not np.isfinite(height_u):
        log.warning("floorplan: no ceiling plane, ceiling height left empty")

    keep = xy[labels != 1]
    extent = keep if traj is None else np.vstack([keep, traj])
    x0, y0 = extent.min(0) - 2 * cell
    x1, y1 = extent.max(0) + 2 * cell
    shape = (int(np.ceil((y1 - y0) / cell)), int(np.ceil((x1 - x0) / cell)))
    grid = histogram(keep, x0, y0, cell, shape)
    floor_cells = int((histogram(xy[labels == 0], x0, y0, cell, shape) > 0).sum())

    out = run_dir / "floorplan"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "grid.npy", grid)
    np.save(out / "grid_meta.npy", np.array([x0, y0, cell]))
    np.save(out / "trajectory_xy.npy", np.zeros((0, 2)) if traj is None else traj)

    cm = lambda v: round(v / upc, 2) if metric and np.isfinite(v) else ""
    row = {
        "width_cm": cm(width_u),
        "length_cm": cm(length_u),
        "area_m2": round(width_u * length_u / upc ** 2 / 1e4, 3) if metric else "",
        "ceiling_height_cm": cm(height_u),
        "floor_cells_m2": round(floor_cells * fp["cell_cm"] ** 2 / 1e4, 3) if metric else "",
        "width_model": round(width_u, 4),
        "length_model": round(length_u, 4),
        "ceiling_height_model": round(height_u, 4) if np.isfinite(height_u) else "",
        "method": method,                     # walls | room_fill | percentiles
        "extent_source": method,              # kept for older readers (same value as method)
        "n_wall_directions": n_dirs,
        "unit": unit,
        "cm_per_unit": round(k_cm, 6) if metric else "",
        "cell_cm": fp["cell_cm"],
        "n_points": len(points),
        "n_wall_points": int((labels >= 2).sum()),
        "n_floor_cells": floor_cells,
        "n_cameras": 0 if traj is None else len(traj),
    }
    if not metric:
        log.warning("floorplan: scale unknown (unit=model), cm/m2 columns left empty")

    if metric:
        title = f"{row['width_cm']:.0f} x {row['length_cm']:.0f} cm, {row['area_m2']:.2f} m2 ({method})"
        dims = (f"W {row['width_cm']:.0f} cm", f"L {row['length_cm']:.0f} cm")
    else:
        title = f"{width_u:.3g} x {length_u:.3g} model units, no scale ({method})"
        dims = (f"W {width_u:.3g} u", f"L {length_u:.3g} u")
    render(grid, x0, y0, cell, traj, rect, unit, upc, fp, run_dir / "figures" / "floorplan.png", title, dims)

    stage_io.upsert_csv(Path(artifacts_dir) / "floorplan.csv", row)
    stage_io.log_metrics(log, row)

    return row

if __name__ == "__main__":
    import tempfile

    logging.basicConfig(level=logging.INFO)
    rng = np.random.default_rng(0)
    W, L, H, n = 400.0, 300.0, 250.0, 4000

    def uniform(a, b, k):
        return rng.uniform(a, b, k)

    # box room in the floor frame, yawed 25 deg in XY
    faces = [
        (np.c_[uniform(0, W, 8 * n), uniform(0, L, 8 * n), np.zeros(8 * n)], 0),
        (np.c_[uniform(0, W, n), uniform(0, L, n), np.full(n, H)], 1),
        (np.c_[np.zeros(n), uniform(0, L, n), uniform(0, H, n)], 2),
        (np.c_[np.full(n, W), uniform(0, L, n), uniform(0, H, n)], 3),
        (np.c_[uniform(0, W, n), np.zeros(n), uniform(0, H, n)], 4),
        (np.c_[uniform(0, W, n), np.full(n, L), uniform(0, H, n)], 5),
        (np.c_[uniform(50, 150, 500), uniform(50, 100, 500), uniform(0, 80, 500)], -1),
    ]
    pts_f = np.vstack([f for f, _ in faces])
    pts_f = pts_f + rng.normal(0, 0.5, pts_f.shape)
    labels = np.concatenate([np.full(len(f), k) for f, k in faces])
    yaw = np.deg2rad(25)
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    pts_f = pts_f @ Rz.T
    t_ = np.linspace(0, 2 * np.pi, 60)
    cen_f = np.c_[W / 2 + 120 * np.cos(t_), L / 2 + 80 * np.sin(t_), np.full(60, 150.0)] @ Rz.T

    # world frame: floor frame = A @ world, A = tilt + translation
    rv = np.array([0.3, -0.2, 0.1])
    R, _ = cv2.Rodrigues(rv)
    A = np.eye(4)
    A[:3, :3], A[:3, 3] = R, [10.0, -20.0, 30.0]
    Ainv = np.linalg.inv(A)
    pts_w, cen_w = to_floor(Ainv, pts_f), to_floor(Ainv, cen_f)
    plane_w = lambda n_, d_: np.r_[R.T @ n_, n_ @ A[:3, 3] + d_]
    cfg = {"floorplan": {"enabled": True, "cell_cm": 5.0, "dpi": 150, "fig_inches": 8.0,
                         "extent_percentiles": [1, 99]},
           "scale": {"provisional_extent_cm": 500.0}}

    def make(root: Path, with_scale: bool):
        for d in ("clean/sparse", "planes", "scale"):
            (root / d).mkdir(parents=True, exist_ok=True)
        np.save(root / "clean/sparse/points.npy", pts_w)
        np.save(root / "clean/sparse/colors.npy", np.full(pts_w.shape, 128, np.uint8))
        np.save(root / "planes/T_floor.npy", A)
        np.save(root / "planes/labels.npy", labels)
        np.save(root / "planes/floor.npy", plane_w(np.array([0, 0, 1.0]), 0.0))
        np.save(root / "planes/ceiling.npy", plane_w(np.array([0, 0, -1.0]), H))
        np.save(root / "planes/walls.npy", np.zeros((4, 4)))
        np.save(root / "scale/centers_cm.npy", cen_w)
        if with_scale:
            np.save(root / "scale/scale.npy", np.array([1.0]))

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        make(tmp / "a", True)
        row = run(cfg, str(tmp / "a"), str(tmp / "art_a"))
        print(row)
        assert abs(row["width_cm"] - L) < 2 and abs(row["length_cm"] - W) < 2, row
        assert abs(row["ceiling_height_cm"] - H) < 2, row
        assert abs(row["area_m2"] - 12.0) < 0.2 and abs(row["floor_cells_m2"] - 12.0) < 1.5, row
        assert (tmp / "a/figures/floorplan.png").exists()
        assert cv2.imread(str(tmp / "a/figures/floorplan.png")) is not None
        assert np.load(tmp / "a/floorplan/trajectory_xy.npy").shape == (60, 2)
        assert np.load(tmp / "a/floorplan/grid.npy").sum() == (labels != 1).sum()

        make(tmp / "b", False)
        run(cfg, str(tmp / "b"), str(tmp / "art_b"))
        csv_b = stage_io.read_csv(tmp / "art_b/floorplan.csv")[0]
        assert csv_b["width_cm"] == csv_b["length_cm"] == csv_b["area_m2"] == csv_b["ceiling_height_cm"] == ""
        assert csv_b["unit"] == "model" and float(csv_b["width_model"]) > 0
        assert (tmp / "b/figures/floorplan.png").exists()

        # one wall direction (walls 3..5 unlabelled): percentile fallback, method recorded
        make(tmp / "c", True)
        np.save(tmp / "c/planes/labels.npy", np.where(labels >= 3, -1, labels))
        row_c = run(cfg, str(tmp / "c"), str(tmp / "art_c"))
        assert row_c["method"] == "percentiles" and row_c["n_wall_directions"] == 1, row_c
        assert abs(row_c["width_cm"] - L) < 10 and abs(row_c["length_cm"] - W) < 10, row_c

        # scale.npy written after the cloud was built (centers_cm == sfm centers): converted, not read as cm
        make(tmp / "d", False)
        (tmp / "d/sfm").mkdir()
        np.save(tmp / "d/sfm/centers.npy", cen_w)
        np.save(tmp / "d/scale/scale.npy", np.array([2.0]))
        row_d = run(cfg, str(tmp / "d"), str(tmp / "art_d"))
        assert row_d["unit"] == "cm" and abs(row_d["width_cm"] - 2 * L) < 4 and abs(row_d["length_cm"] - 2 * W) < 4, row_d
        assert stage_io.read_csv(tmp / "art_d/floorplan.csv")[0]["method"] == "walls"
    print("OK")
# <<< END IMPLEMENTATION <<<

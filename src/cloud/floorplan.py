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
#     density, colormap, camera trajectory (start/end markers), room rectangle,
#     scale bar (only when unit = cm) and axis text.
#   - width/length: orientation from cv2.minAreaRect of the XY of the wall
#     points (labels >= 2), sides = extent_percentiles span of those points in
#     the rectangle frame, each side refined to the median of the points within
#     one cell of it (robust to outliers and noise); fallback (WARNING): the
#     same on all non floor/ceiling points.
#   - area_m2 = area of that rectangle; floor_cells_m2 = occupied cells of the
#     floor points (label 0) times the cell area (extra column; underestimates
#     where the sparse floor has holes, so area_m2 is the main value).
#   - ceiling_height_cm = distance of the ceiling plane to a point of the floor
#     plane (empty when no ceiling).
# In:  clean/sparse/{points,colors}.npy, planes/{T_floor,floor,ceiling,walls,
#      labels}.npy (labels: -1 none, 0 floor, 1 ceiling, 2+k wall k),
#      scale/centers_cm.npy, scale/scale.npy (via stage_io.scale_info).
# Out: figures/floorplan.png, floorplan/{grid,grid_meta,trajectory_xy}.npy
#      (grid_meta = [x0, y0, cell] in working units, grid[row, col] with row
#      along +y); one row in artifacts/<id>/floorplan.csv.
# YAML: floorplan: {enabled: true, cell_cm: 5.0, dpi: 150, fig_inches: 8.0,
#       extent_percentiles: [1, 99]}; scale.provisional_extent_cm.
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

def robust_rect(pts: np.ndarray, percentiles, band: float) -> tuple:
    """Orientation from cv2.minAreaRect; each side = median of the points within +-band of the
    per-axis percentile in the rectangle frame (removes the noise bias of the percentile)."""

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

def room_rect(xy: np.ndarray, labels: np.ndarray, percentiles, band: float) -> tuple[tuple, str]:

    walls = xy[labels >= 2]
    if len(walls) >= 3:
        return robust_rect(walls, percentiles, band), "walls"

    rest = xy[(labels != 0) & (labels != 1)]
    if len(rest) < 3:
        rest = xy
    log.warning("floorplan: no wall points, extent from %d non floor/ceiling points at percentiles %s",
                len(rest), list(percentiles))

    return robust_rect(rest, percentiles, band), "percentiles"

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
           unit: str, upc: float, cfg: dict, out_path: Path, title: str) -> None:

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

    box = to_px(cv2.boxPoints(rect).astype(np.float64))
    cv2.polylines(canvas, [box], True, (255, 255, 0), th, cv2.LINE_AA)
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
    x1, y1 = x0 + grid.shape[1] * cell, y0 + grid.shape[0] * cell
    cv2.putText(canvas, title, (margin, int(margin * 0.6)), font, fs * 1.2, (0, 0, 0), th, cv2.LINE_AA)
    cv2.putText(canvas, f"x [{x0:.3g}, {x1:.3g}] {u_name}", (margin + w // 2, margin + h + margin // 2 + 5),
                font, fs, (0, 0, 0), th, cv2.LINE_AA)
    label = f"y [{y0:.3g}, {y1:.3g}] {u_name}"
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
    centers = load_optional(run_dir / "scale" / "centers_cm.npy", "camera centers")

    _, unit = stage_io.scale_info(run_dir)
    upc = stage_io.units_per_cm(run_dir, points, room_extent_cm=cfg["scale"]["provisional_extent_cm"])
    cell = fp["cell_cm"] * upc

    xy = to_floor(T, points)[:, :2]
    traj = None if centers is None else to_floor(T, np.asarray(centers, np.float64).reshape(-1, 3))[:, :2]

    rect, source = room_rect(xy, labels, fp["extent_percentiles"], cell)
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

    metric = unit == "cm"
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
        "extent_source": source,
        "unit": unit,
        "cell_cm": fp["cell_cm"],
        "n_points": len(points),
        "n_wall_points": int((labels >= 2).sum()),
        "n_floor_cells": floor_cells,
    }
    if not metric:
        log.warning("floorplan: scale unknown (unit=model), cm/m2 columns left empty")

    if metric:
        title = f"{row['width_cm']:.0f} x {row['length_cm']:.0f} cm, {row['area_m2']:.2f} m2"
    else:
        title = f"{width_u:.3g} x {length_u:.3g} model units (no scale)"
    render(grid, x0, y0, cell, traj, rect, unit, upc, fp, run_dir / "figures" / "floorplan.png", title)

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
    print("OK")
# <<< END IMPLEMENTATION <<<

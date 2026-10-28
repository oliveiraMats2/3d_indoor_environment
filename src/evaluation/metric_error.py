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
# Stage: Stage 5, Evaluation (metric_error, after room_fill).
# Goal: metric error of every dimension measured by tape: real L, model L_hat, |L_hat - L| cm and
#   e = |L_hat - L| / L (T2 items 5.2, 5.A.1).
# In:  evaluation.measured_cm (list of {name, value_cm}) and/or evaluation.measurements_file (YAML with
#      paredes_cm|walls_cm: {name: cm} and pe_direito_cm|ceiling_height_cm: cm or null);
#      planes/{T_floor,floor,ceiling}.npy, mesh/room_fill/{T_room,footprint_cm}.npy,
#      clean/<room_fill.source>/points.npy (fallback clean/<planes.source>, clean/sparse), scale/scale.npy,
#      artifacts floorplan.csv (only for the legacy names room_width / room_length).
# Out: artifacts metric_error.csv (name, value_cm, model_cm, abs_error_cm, rel_error, scale_source, held_out,
#      model_units, cm_per_unit, method, model_extent_cm); results/<run_id>/evaluation/metric_errors.npy (K, 2):
#      abs_error_cm, rel_error (NaN when unknown) and metric_model_cm.npy (K,).
# YAML: evaluation.{measured_cm, measurements_file: null, metric_scale: auto|scale|leave_one_out}
# Rules:
# - Kinds: room_width / room_length -> floorplan.csv width_cm / length_cm (legacy, scale_source floorplan);
#   ceiling_height / pe_direito -> model floor-ceiling distance; any other name is a wall, matched to an
#   edge of the fitted room_fill outline (footprint_cm.npy) by its length (assignment, |dL|/L < MATCH_TOL).
# - Wall model length (model units, floor frame p' = T_floor p): wall points = room_fill.wall_band of the
#   robust height range; each is assigned to the nearest outline edge (outline placed with T_floor @ T_room)
#   when closer than BAND_CM (tolerance only); per edge a robust line (PCA + MAD trimming) is refit from its
#   own points; length = distance between its corners with the two neighbour lines (method corners), else
#   (near-parallel neighbour or neighbour without points) the p2-p98 extent along the line (method extent).
#   model_extent_cm (always the p2-p98 extent) is a diagnostic: it is short where corners are occluded.
# - Scale (cm per working unit), in order: scale/scale.npy known (working unit = cm, k = 1, source scale)
#   > leave-one-out: for each wall, k from the OTHER matched walls measured corner to corner, k = median(L_j / u_j)
#   (robust to one badly reconstructed wall), source leave_one_out, held_out true. The room_fill scale is fitted
#   on these same walls, so it is never used. With scale.npy from a measurement-based method (scale.method
#   outline/room_fill/measurements) held_out is false (WARNING); metric_scale forces one source.
# - Ceiling: planes ceiling plane at the floor frame origin, else p99 z of the points inside the outline;
#   scale = metric, else k of all matched walls (held_out true: the height is not in the fit). Height not
#   measured (null) -> row with empty value_cm / errors, model_cm filled, WARNING.
# - Missing inputs -> WARNING and rows with empty model values; empty measurement list -> header only.
# >>> BEGIN IMPLEMENTATION >>>
"""Metric error of the tape-measured dimensions (walls, ceiling height, legacy floorplan sizes)."""
import csv, logging, math
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment

from src import stage_io

log = logging.getLogger(__name__)

COLUMNS = {"room_width": "width_cm", "room_length": "length_cm"}   # legacy floorplan.csv names
CEILING = ("ceiling_height", "pe_direito")
FIELDS = ["name", "value_cm", "model_cm", "abs_error_cm", "rel_error", "scale_source", "held_out",
          "model_units", "cm_per_unit", "method", "model_extent_cm"]
MEASURED_SCALE = ("outline", "room_fill", "measurements")   # scale.method values fitted on these walls
MATCH_TOL = 0.10        # max |edge - L| / L when matching a measured wall to an outline edge
BAND_CM = 15.0          # wall point -> edge assignment distance (cm, through the room_fill fit scale)
MIN_EDGE_POINTS = 30    # fewer points -> no model length for that wall
MIN_SIN = 0.3           # neighbour lines closer than ~17 deg to parallel -> extent instead of corners

def load_measures(ev: dict) -> list[dict]:
    """{name, value_cm (float or None)} from measured_cm and measurements_file (file wins on repeated names)."""

    measures = {str(m["name"]): (None if m.get("value_cm") is None else float(m["value_cm"]))
                for m in (dict(m) for m in ev.get("measured_cm") or [])}
    path = ev.get("measurements_file")
    if path:
        path = Path(path).expanduser()
        if not path.exists():
            log.warning("metric_error: measurements_file %s not found", path)
        else:
            data = yaml.safe_load(path.read_text()) or {}
            walls = data.get("paredes_cm") or data.get("walls_cm") or {}
            measures.update({str(k): float(v) for k, v in walls.items() if v is not None})
            height = data.get("pe_direito_cm", data.get("ceiling_height_cm"))
            measures["ceiling_height"] = None if height is None else float(height)
            measures.pop("pe_direito", None)

    return [{"name": k, "value_cm": v} for k, v in measures.items()]

def floorplan_value(row: dict, name: str) -> float:
    """Legacy L_hat from the floorplan row; NaN when empty or unparsable."""

    try:
        v = float(row.get(COLUMNS[name], ""))
    except ValueError:
        return float("nan")

    return v if math.isfinite(v) else float("nan")

def load_points(cfg: dict, run_dir: Path) -> np.ndarray | None:

    sources = [cfg.get("room_fill", {}).get("source", "patch_match"), cfg.get("planes", {}).get("source", "sparse"),
               "sparse"]
    for source in dict.fromkeys(sources):
        path = run_dir / "clean" / source / "points.npy"
        if path.exists():
            return np.load(path).astype(np.float64).reshape(-1, 3)

    return None

def fit_line(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Robust 2D line (centre, unit direction): PCA, then points beyond 3 MAD of the normal residual dropped."""

    keep = np.ones(len(xy), bool)
    for _ in range(4):
        if keep.sum() < MIN_EDGE_POINTS:
            return None
        c = np.median(xy[keep], axis=0)
        _, _, vt = np.linalg.svd(xy[keep] - c, full_matrices=False)
        d = vt[0]
        r = (xy - c) @ np.array([-d[1], d[0]])
        r -= np.median(r[keep])
        mad = 1.4826 * np.median(np.abs(r[keep])) + 1e-12
        keep = np.abs(r) < 3 * mad
    c = c + np.median(r[keep]) * np.array([-d[1], d[0]])

    return c, d

def intersect(l1, l2) -> np.ndarray | None:

    (c1, d1), (c2, d2) = l1, l2
    A = np.c_[d1, -d2]
    if abs(np.linalg.det(A)) < MIN_SIN:
        return None

    a, _ = np.linalg.solve(A, c2 - c1)

    return c1 + a * d1

def wall_lengths(cfg: dict, run_dir: Path) -> dict | None:
    """Per outline edge: (cm length of the hand outline edge, model units length, method, extent units),
    the fit scale (units per cm) and the floor-frame points inside the outline; None when inputs are missing."""

    rf, pl = run_dir / "mesh" / "room_fill", run_dir / "planes"
    need = [rf / "T_room.npy", rf / "footprint_cm.npy", pl / "T_floor.npy"]
    if not all(p.exists() for p in need):
        log.warning("metric_error: %s missing (enable room_fill and planes), no wall model lengths",
                    ", ".join(str(p) for p in need if not p.exists()))
        return None
    points = load_points(cfg, run_dir)
    if points is None:
        log.warning("metric_error: no clean/<source>/points.npy under %s, no wall model lengths", run_dir)
        return None

    from src.mesh.room_fill import inside, segment_distance, to_frame
    T = np.load(pl / "T_floor.npy")
    M = T @ np.load(rf / "T_room.npy")                  # outline cm -> floor frame, working units
    poly_cm = np.load(rf / "footprint_cm.npy").reshape(-1, 2)
    poly = to_frame(M, np.c_[poly_cm, np.zeros(len(poly_cm))])[:, :2]
    s_fit = float(np.sqrt(abs(np.linalg.det(M[:2, :2]))))   # units per cm, only for the tolerance
    pf = to_frame(T, points)

    z_lo, z_hi = np.percentile(pf[:, 2], [1, 99])
    b_lo, b_hi = cfg.get("room_fill", {}).get("wall_band", [0.2, 0.8])
    wall = pf[(pf[:, 2] > z_lo + b_lo * (z_hi - z_lo)) & (pf[:, 2] < z_lo + b_hi * (z_hi - z_lo)), :2]
    dist, edge = segment_distance(wall, poly)
    near = dist < BAND_CM * s_fit

    n = len(poly)
    lines = [fit_line(wall[near & (edge == i)]) for i in range(n)]
    edges = []
    for i in range(n):
        L_cm = float(np.linalg.norm(poly_cm[(i + 1) % n] - poly_cm[i]))
        if lines[i] is None:
            edges.append((L_cm, float("nan"), "", float("nan")))
            continue
        c, d = lines[i]
        proj = (wall[near & (edge == i)] - c) @ d
        extent = float(np.subtract(*np.percentile(proj, [98, 2])))
        prev, nxt = lines[i - 1], lines[(i + 1) % n]
        a = intersect(lines[i], prev) if prev is not None else None
        b = intersect(lines[i], nxt) if nxt is not None else None
        if a is not None and b is not None:
            edges.append((L_cm, float(np.linalg.norm(b - a)), "corners", extent))
        else:
            edges.append((L_cm, extent, "extent", extent))

    room = inside(poly, pf[:, :2]) & (pf[:, 2] > 0)
    return {"edges": edges, "s_fit": s_fit, "T": T, "z_room": pf[room, 2]}

def ceiling_units(run_dir: Path, geo: dict | None) -> tuple[float, str]:
    """Floor-ceiling distance in working units: planes ceiling, else p99 z inside the outline."""

    pl = run_dir / "planes"
    ceiling = np.load(pl / "ceiling.npy") if (pl / "ceiling.npy").exists() else np.full(4, np.nan)
    if np.all(np.isfinite(ceiling)) and (pl / "T_floor.npy").exists():
        T = np.load(pl / "T_floor.npy")
        p0 = -ceiling[3] * ceiling[:3]
        n = T[:3, :3] @ ceiling[:3]
        z = (p0 @ T[:3, :3].T + T[:3, 3])[2]
        return float(z / abs(n[2])) if abs(n[2]) > 1e-6 else float(z), "planes"
    if geo is not None and len(geo["z_room"]):
        return float(np.percentile(geo["z_room"], 99)), "p99"

    return float("nan"), ""

def scale_from(L: np.ndarray, u: np.ndarray) -> float:
    """cm per unit, median of the per-wall ratios L/u (one badly reconstructed wall cannot drag the others);
    NaN without data."""

    ok = np.isfinite(L) & np.isfinite(u) & (u > 0)

    return float(np.median(L[ok] / u[ok])) if ok.sum() else float("nan")

def run(cfg: dict, run_dir: str, artifacts_dir: str) -> Path | None:

    ev = cfg["evaluation"]
    if not ev.get("enabled", False):
        log.info("metric_error: evaluation disabled")
        return None

    run_dir, artifacts_dir = Path(run_dir), Path(artifacts_dir)
    csv_path = artifacts_dir / "metric_error.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    out_dir = run_dir / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    measures = load_measures(ev)
    if not measures:
        log.warning("metric_error: evaluation.measured_cm and measurements_file are empty (tape measurements missing)")
        with open(csv_path, "w", newline="") as fh:
            csv.writer(fh).writerow(FIELDS)
        np.save(out_dir / "metric_errors.npy", np.empty((0, 2)))
        np.save(out_dir / "metric_model_cm.npy", np.empty(0))
        return csv_path

    kind = {m["name"]: "floorplan" if m["name"] in COLUMNS else "ceiling" if m["name"] in CEILING else "wall"
            for m in measures}
    rows = {m["name"]: {"name": m["name"], "value_cm": "" if m["value_cm"] is None else m["value_cm"],
                        **{k: "" for k in FIELDS[2:]}} for m in measures}
    model_cm = {m["name"]: float("nan") for m in measures}

    # scale: scale.npy (working unit = cm) unless forced, else leave-one-out over the walls
    s_npy, unit = stage_io.scale_info(run_dir)
    mode = ev.get("metric_scale") or "auto"
    metric = unit == "cm" and mode in ("auto", "scale")
    if mode == "scale" and unit != "cm":
        log.warning("metric_error: metric_scale scale but scale/scale.npy unknown, using leave-one-out")
    scale_method = str(cfg.get("scale", {}).get("method", ""))
    circular = metric and any(k in scale_method for k in MEASURED_SCALE)
    if circular:
        log.warning("metric_error: scale.npy comes from %s (same tape measurements): errors are not held out",
                    scale_method)

    # walls: model length per matched outline edge
    walls = [m for m in measures if kind[m["name"]] == "wall" and m["value_cm"]]
    need_geo = bool(walls) or any(k == "ceiling" for k in kind.values())
    geo = wall_lengths(cfg, run_dir) if need_geo else None
    u = {}                                       # name -> (model units, method, extent units)
    if walls and geo is not None:
        e_len = np.array([e[0] for e in geo["edges"]])
        L = np.array([m["value_cm"] for m in walls])
        cost = np.abs(e_len[None, :] - L[:, None]) / L[:, None]
        ri, ci = linear_sum_assignment(cost)
        for r, c in zip(ri, ci):
            if cost[r, c] < MATCH_TOL:
                u[walls[r]["name"]] = geo["edges"][c][1:]
        for m in walls:
            if m["name"] not in u:
                log.warning("metric_error: wall %s (%.1f cm) matches no room_fill outline edge", m["name"], m["value_cm"])

    names = [m["name"] for m in walls if m["name"] in u and np.isfinite(u[m["name"]][0])]
    corner = [n for n in names if u[n][1] == "corners"]  # both ends found: the extent fallback is not a length
    names = corner if len(corner) >= 2 else names
    Lw = {m["name"]: m["value_cm"] for m in walls}
    k_all = 1.0 if metric else scale_from(np.array([Lw[n] for n in names]), np.array([u[n][0] for n in names]))
    for name in u:
        units, method, extent = u[name]
        if metric:
            k, src = 1.0, "scale"
        else:
            others = [n for n in names if n != name]
            if len(others) < 1:
                log.warning("metric_error: %s: no other measured wall for the leave-one-out scale", name)
                k = float("nan")
            else:
                k = scale_from(np.array([Lw[n] for n in others]), np.array([u[n][0] for n in others]))
            src = "leave_one_out"
        model_cm[name] = units * k
        rows[name].update(scale_source=src, held_out=str(not circular).lower(), method=method,
                          model_units=f"{units:.6g}" if np.isfinite(units) else "",
                          cm_per_unit=f"{k:.6g}" if np.isfinite(k) else "",
                          model_extent_cm=f"{extent * k:.2f}" if np.isfinite(extent * k) else "")

    # ceiling height
    for name in [n for n, k in kind.items() if k == "ceiling"]:
        h, h_from = ceiling_units(run_dir, geo)
        model_cm[name] = h * k_all
        rows[name].update(scale_source="scale" if metric else "all_walls", held_out=str(not circular).lower(),
                          method=h_from, model_units=f"{h:.6g}" if np.isfinite(h) else "",
                          cm_per_unit=f"{k_all:.6g}" if np.isfinite(k_all) else "")
        if rows[name]["value_cm"] == "":
            log.warning("metric_error: %s not measured (null): model %.1f cm reported without error", name, model_cm[name])

    # legacy floorplan sizes
    fp_names = [n for n, k in kind.items() if k == "floorplan"]
    if fp_names:
        fp_rows = stage_io.read_csv(artifacts_dir / "floorplan.csv")
        if not fp_rows:
            log.warning("metric_error: floorplan.csv missing or empty, model_cm unknown")
        for name in fp_names:
            model_cm[name] = floorplan_value(fp_rows[-1] if fp_rows else {}, name)
            rows[name].update(scale_source="floorplan", method="floorplan")

    out, errors, model = [], np.full((len(measures), 2), np.nan), np.full(len(measures), np.nan)
    for i, m in enumerate(measures):
        row, L_hat, L = rows[m["name"]], model_cm[m["name"]], m["value_cm"]
        if math.isfinite(L_hat):
            row["model_cm"] = round(L_hat, 2)
            model[i] = L_hat
        else:
            log.warning("metric_error: %s has no model value (scale or geometry unknown)", m["name"])
        if math.isfinite(L_hat) and L:
            abs_err = abs(L_hat - L)
            errors[i] = abs_err, abs_err / L
            row.update(abs_error_cm=round(abs_err, 2), rel_error=round(abs_err / L, 5))
        out.append(row)
        stage_io.log_metrics(log, {f"{m['name']}_{k}": row[k] for k in ("value_cm", "model_cm", "abs_error_cm", "rel_error")})

    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out)
    np.save(out_dir / "metric_errors.npy", errors)
    np.save(out_dir / "metric_model_cm.npy", model)
    rel = errors[np.isfinite(errors[:, 1]), 1]
    log.info("metric_error: %d measures, mean rel error %s -> %s", len(out),
             f"{rel.mean():.4f}" if len(rel) else "n/a", csv_path)

    return csv_path

def synthetic_run(tmp: Path, units_per_cm: float, noise_cm: float = 0.5, seed: int = 0) -> tuple[Path, Path, np.ndarray]:
    """Kitchen-shaped room (walls 0..250 cm) in a rotated, scaled model frame with room_fill outputs."""

    from src.mesh.room_fill import ccw
    rng = np.random.default_rng(seed)
    poly = np.array([[0, 0], [273, 0], [273, 429], [81.1, 424.5], [0, 332]], float)
    pts = []
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        t = rng.uniform(0.05, 0.95, 3000)[:, None]
        xy = a + t * (b - a)
        nrm = np.array([-(b - a)[1], (b - a)[0]]) / np.linalg.norm(b - a)
        pts.append(np.c_[xy + rng.normal(0, noise_cm, (len(xy), 1)) * nrm, rng.uniform(0, 250, len(xy))])
    floor = np.c_[rng.uniform(0, 273, 3000), rng.uniform(0, 330, 3000), np.zeros(3000)]
    room_cm = np.vstack(pts + [floor, floor + [0, 0, 250]])
    th = 0.7
    Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    T_floor = np.eye(4)
    T_floor[:3, :3], T_floor[:3, 3] = Rz.T, [1.0, -2.0, 0.0]          # model -> floor frame
    T_inv = np.linalg.inv(T_floor)
    model = (room_cm * units_per_cm) @ T_inv[:3, :3].T + T_inv[:3, 3]
    run_dir, art = tmp / "res", tmp / "art"
    for d in ("clean/patch_match", "planes", "mesh/room_fill"):
        (run_dir / d).mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "clean/patch_match/points.npy", model)
    np.save(run_dir / "planes/T_floor.npy", T_floor)
    np.save(run_dir / "planes/ceiling.npy", np.full(4, np.nan))
    fp = ccw(poly) - poly.mean(axis=0)
    T_room = np.diag([units_per_cm, units_per_cm, units_per_cm, 1.0])
    T_room[:2, 3] = poly.mean(axis=0) * units_per_cm
    np.save(run_dir / "mesh/room_fill/T_room.npy", T_inv @ T_room)
    np.save(run_dir / "mesh/room_fill/footprint_cm.npy", fp)

    return run_dir, art, poly

if __name__ == "__main__":
    import tempfile
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # 1. legacy floorplan names
        run_dir, art = tmp / "legacy", tmp / "legacy_art"
        meas = [{"name": "room_width", "value_cm": 400.0}, {"name": "room_length", "value_cm": 500.0}]
        cfg = {"evaluation": {"enabled": True, "measured_cm": meas}}
        stage_io.upsert_csv(art / "floorplan.csv", {"width_cm": 410.0, "length_cm": 490.0})
        rows = stage_io.read_csv(run(cfg, str(run_dir), str(art)))
        assert [r["name"] for r in rows] == ["room_width", "room_length"] and rows[0]["scale_source"] == "floorplan"
        assert np.allclose(np.load(run_dir / "evaluation/metric_errors.npy"), [[10, 0.025], [10, 0.02]])
        (art / "floorplan.csv").unlink()
        rows = stage_io.read_csv(run(cfg, str(run_dir), str(art)))
        assert all(r["model_cm"] == "" for r in rows)
        p = run({"evaluation": {"enabled": True, "measured_cm": []}}, str(run_dir), str(art))
        assert p.read_text().strip() == ",".join(FIELDS)

        # 2. walls from a measurements file, leave-one-out scale, ceiling not measured
        run_dir, art, poly = synthetic_run(tmp / "syn", units_per_cm=0.0367)
        mfile = tmp / "medidas.yaml"
        mfile.write_text("paredes_cm: {janela: 273, pia_fogao: 429, porta_lado_pia: 192, porta: 123, geladeiras: 332}\n"
                         "pe_direito_cm: null\n")
        cfg = {"evaluation": {"enabled": True, "measured_cm": [], "measurements_file": str(mfile)},
               "room_fill": {"source": "patch_match", "wall_band": [0.2, 0.8]}}
        rows = {r["name"]: r for r in stage_io.read_csv(run(cfg, str(run_dir), str(art)))}
        for n in ("janela", "pia_fogao", "porta_lado_pia", "porta", "geladeiras"):
            assert rows[n]["scale_source"] == "leave_one_out" and rows[n]["held_out"] == "true", rows[n]
            assert rows[n]["method"] == "corners" and float(rows[n]["abs_error_cm"]) < 2.0, rows[n]
            assert abs(float(rows[n]["cm_per_unit"]) - 1 / 0.0367) < 0.3
        h = rows["ceiling_height"]
        assert h["value_cm"] == "" and h["abs_error_cm"] == "" and abs(float(h["model_cm"]) - 250) < 3, h

        # 3. metric scale.npy: working unit is cm, k = 1; a scale stretched by 2% shows up as ~2% error
        run_dir, art, _ = synthetic_run(tmp / "metric", units_per_cm=1.02)
        (run_dir / "scale").mkdir()
        np.save(run_dir / "scale/scale.npy", np.array([3.0]))
        rows = {r["name"]: r for r in stage_io.read_csv(run(cfg, str(run_dir), str(art)))}
        assert rows["pia_fogao"]["scale_source"] == "scale" and abs(float(rows["pia_fogao"]["rel_error"]) - 0.02) < 0.005
        cfg["evaluation"]["metric_scale"] = "leave_one_out"
        rows = {r["name"]: r for r in stage_io.read_csv(run(cfg, str(run_dir), str(art)))}
        assert rows["pia_fogao"]["scale_source"] == "leave_one_out" and float(rows["pia_fogao"]["rel_error"]) < 0.005

        # 4. missing room_fill -> rows without model values, no crash
        (run_dir / "mesh/room_fill/T_room.npy").unlink()
        rows = stage_io.read_csv(run(cfg, str(run_dir), str(art)))
        assert all(r["model_cm"] == "" for r in rows if r["name"] != "ceiling_height")
    print("OK")
# <<< END IMPLEMENTATION <<<

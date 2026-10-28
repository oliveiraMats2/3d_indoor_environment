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
# Stage: scale_outline (right after scale, before cleanup_sparse: every later stage then works in cm).
# Goal: metric scale s = cm per model unit from the tape-measured room outline (no marker in the scene): fit the
#   outline (cm) to the wall points of the raw sparse cloud in its own floor frame, with the outline fit of
#   src/mesh/room_fill.py (imported, not copied). Never invent a value: a poor fit leaves s = nan (unit model).
# In:  sfm/{points3d,centers,cam_R}.npy (model units), scale/scale.npy (stage_io.scale_info), the outline from
#      scale.outline.footprint_cm, else the YAML file scale.outline.measurements (key contorno_cm or footprint_cm),
#      else room_fill.footprint_cm.
# Out: same files as src/sfm/scale.py: scale/{scale,points3d_cm,centers_cm}.npy (+ scale/outline_model.npy, the
#      fitted outline at floor height in the model frame), artifacts_dir/scale.csv (method outline, scale, unit,
#      fit residual median/p90 cm, inlier fraction, ambiguity, mirrored, rotation, n points, accepted, t_s).
# YAML: scale.method: aruco|manual|outline; scale.outline: {footprint_cm: null, measurements: null,
#       fallback: true, wall_band: null (-> room_fill.wall_band, else [0.2, 0.8]), fit_rotations: null
#       (-> room_fill.fit_rotations, else 72), voxel_cm, nb_neighbors, std_ratio, crop_percentiles,
#       max_residual_cm, max_p90_cm, min_inlier_frac, min_support}; planes.{distance_cm, ransac_n, iterations,
#       max_planes, min_inlier_frac, angle_tol_deg}; scale.provisional_extent_cm.
# Rules:
# - method aruco|manual with a finite scale.npy -> kept untouched; with nan and outline.fallback -> outline fit
#   (fallback_of = that method); method outline -> always fitted (a stale scale.npy is never trusted).
# - Cloud: statistical outliers (nb_neighbors, std_ratio), voxel_cm and per-axis crop_percentiles, thresholds in
#   cm through the provisional factor of stage_io.units_per_cm (robust extent / provisional_extent_cm),
#   computed from the cloud itself (scale.npy may be stale or nan here).
# - Floor frame: lowest horizontal RANSAC plane below the median camera height (src.cloud.planes helpers),
#   else (WARNING) camera gravity with the floor at the 1st height percentile. Wall points = height band
#   wall_band of the robust (1-99 percentile) height range, at most room_fill.FIT_POINTS (seeded).
# - room_fill.fit_outline gives s_units_per_cm; scale = 1 / s. Accepted only when residual median <=
#   max_residual_cm, p90 <= max_p90_cm (the trimmed fit keeps the median low on a wrong outline), inlier
#   fraction >= min_inlier_frac and outline support (fraction of the fitted outline length with a wall point
#   within room_fill.INLIER_CM) >= min_support; else WARNING and scale = nan.
# - Failed fallback: the aruco|manual outputs stay as they are, the attempt goes to scale_outline.csv.

# >>> BEGIN IMPLEMENTATION >>>
"""Metric scale from the hand-measured room outline fitted to the sparse cloud."""
import logging
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml
from scipy.spatial import cKDTree

from src import stage_io
from src.cloud import clean, planes
from src.mesh import room_fill

log = logging.getLogger(__name__)

DEFAULTS = {"footprint_cm": None, "measurements": None, "fallback": True, "wall_band": None,
            "fit_rotations": None, "voxel_cm": 2.0, "nb_neighbors": 20, "std_ratio": 2.0,
            "crop_percentiles": [1.0, 99.0], "max_residual_cm": 10.0, "min_inlier_frac": 0.5,
            "max_p90_cm": 30.0, "min_support": 0.5}
PLANES = {"distance_cm": 2.0, "ransac_n": 3, "iterations": 2000, "max_planes": 10, "min_inlier_frac": 0.02,
          "angle_tol_deg": 15.0}

def outline_cfg(cfg: dict) -> dict:

    return {**DEFAULTS, **(cfg.get("scale", {}).get("outline") or {})}

def load_footprint(cfg: dict, ocfg: dict) -> tuple[np.ndarray | None, str]:
    """(outline (N,2) cm, where it came from); (None, "") when no valid outline is configured."""

    candidates = [("scale.outline.footprint_cm", ocfg["footprint_cm"])]
    if ocfg["measurements"]:
        path = Path(ocfg["measurements"])
        if path.exists():
            data = yaml.safe_load(path.read_text()) or {}
            candidates.append((str(path), data.get("footprint_cm") or data.get("contorno_cm")))
        else:
            log.warning("scale_outline: scale.outline.measurements %s not found", path)
    candidates.append(("room_fill.footprint_cm", (cfg.get("room_fill") or {}).get("footprint_cm")))
    for origin, value in candidates:
        if not value:
            continue
        poly = np.asarray(value, dtype=np.float64).reshape(-1, 2)
        if len(poly) >= 3 and abs(room_fill.signed_area(poly)) > 1e-6:
            return poly, origin
        log.warning("scale_outline: %s needs >= 3 non-collinear [x, y] corners in cm, ignored", origin)

    return None, ""

def prepare_cloud(points: np.ndarray, ocfg: dict, upc: float) -> np.ndarray:
    """Outliers, voxel and percentile crop (reuses src.cloud.clean)."""

    grey = np.zeros((len(points), 3), np.uint8)
    _, (points, _) = clean.filter_legacy(points, grey, int(ocfg["nb_neighbors"]), float(ocfg["std_ratio"]),
                                         float(ocfg["voxel_cm"]) * upc)
    lo, hi = ocfg["crop_percentiles"]

    return points[clean.crop_percentiles(points, lo, hi)]

def floor_frame(points: np.ndarray, cam_R, centers, pcfg: dict, upc: float, seed: int) -> tuple[np.ndarray, str]:
    """T (4,4) to the floor frame (z up, floor at 0): RANSAC floor, else camera gravity + low percentile."""

    up = planes.gravity_up(cam_R)
    o3d.utility.random.seed(seed)
    min_inliers = max(int(np.ceil(pcfg["min_inlier_frac"] * len(points))), pcfg["ransac_n"])
    found = planes.peel_planes(points, pcfg["distance_cm"] * upc, pcfg["ransac_n"], pcfg["iterations"],
                               pcfg["max_planes"], min_inliers)
    cam_h = float(np.median(centers @ up)) if centers is not None and len(centers) else np.inf
    cos_tol = np.cos(np.radians(pcfg["angle_tol_deg"]))
    horizontal = [(planes.orient(m, up), float(np.median(points[idx] @ up))) for m, idx in found
                  if abs(m[:3] @ up) > cos_tol]
    below = [h for h in horizontal if h[1] < cam_h]
    if below:
        return planes.floor_transform(min(below, key=lambda h: h[1])[0], np.zeros((0, 4))), "ransac"

    log.warning("scale_outline: no horizontal plane below the cameras, floor = gravity at the 1st height percentile")
    floor = np.r_[up, -np.percentile(points @ up, 1)]

    return planes.floor_transform(floor, np.zeros((0, 4))), "gravity"

def outline_support(q: np.ndarray, fit: dict, step_cm: float = 2.0) -> float:
    """Fraction of the fitted outline length (sampled every step_cm) with a wall point within INLIER_CM."""

    poly = room_fill.place(fit["poly"], fit["s"], fit["theta"], fit["t"])
    samples = []
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        n = max(2, int(np.linalg.norm(b - a) / (step_cm * fit["s"])))
        samples.append(a + np.linspace(0, 1, n, endpoint=False)[:, None] * (b - a))
    d, _ = cKDTree(q).query(np.vstack(samples))

    return float(np.mean(d / fit["s"] < room_fill.INLIER_CM))

def write_outputs(run_dir: Path, s: float, outline_model: np.ndarray | None) -> None:
    """Same files as src/sfm/scale.py (raw copy in model units when s is nan)."""

    sfm_dir, out_dir = run_dir / "sfm", run_dir / "scale"
    out_dir.mkdir(parents=True, exist_ok=True)
    factor = s if np.isfinite(s) else 1.0
    np.save(out_dir / "scale.npy", np.array([s], dtype=np.float64))
    np.save(out_dir / "points3d_cm.npy", np.load(sfm_dir / "points3d.npy") * factor)
    np.save(out_dir / "centers_cm.npy", np.load(sfm_dir / "centers.npy") * factor)
    if outline_model is not None:
        np.save(out_dir / "outline_model.npy", outline_model)

def run(cfg: dict, run_dir: str, artifacts_dir: str) -> float:

    scfg, ocfg = cfg["scale"], outline_cfg(cfg)
    if not scfg.get("enabled", True):
        log.info("scale disabled")
        return float("nan")

    t0 = time.perf_counter()
    run_dir, artifacts_dir = Path(run_dir), Path(artifacts_dir)
    method = scfg.get("method", "outline")
    fallback_of = ""
    if method != "outline":
        s_old, unit = stage_io.scale_info(run_dir)
        if unit == "cm":
            log.info("scale_outline: scale.method %s gave %.6g cm per model unit, kept", method, s_old)
            return s_old
        if not ocfg["fallback"]:
            log.info("scale_outline: scale unknown and scale.outline.fallback false, skipped")
            return float("nan")
        log.info("scale_outline: scale.method %s left the scale unknown, fitting the room outline", method)
        fallback_of = method

    sfm_dir = run_dir / "sfm"
    if not (sfm_dir / "points3d.npy").exists() or not (sfm_dir / "centers.npy").exists():
        log.warning("scale_outline: %s has no SfM arrays; skipping", sfm_dir)
        return float("nan")

    footprint, footprint_from = load_footprint(cfg, ocfg)
    row = {"method": "outline", "fallback_of": fallback_of, "scale": "", "unit": "model",
           "footprint_from": footprint_from}
    s, outline_model = float("nan"), None
    raw = np.load(sfm_dir / "points3d.npy").astype(np.float64).reshape(-1, 3)
    centers = np.load(sfm_dir / "centers.npy").astype(np.float64).reshape(-1, 3)
    cam_R = np.load(sfm_dir / "cam_R.npy") if (sfm_dir / "cam_R.npy").exists() else None
    if footprint is None:
        log.warning("scale_outline: no room outline (scale.outline.footprint_cm, scale.outline.measurements or "
                    "room_fill.footprint_cm), scale stays unknown")
    elif len(raw) <= ocfg["nb_neighbors"]:
        log.warning("scale_outline: only %d sparse points, scale stays unknown", len(raw))
    else:
        rf = cfg.get("room_fill") or {}
        band = ocfg["wall_band"] or rf.get("wall_band") or [0.2, 0.8]
        n_rot = int(ocfg["fit_rotations"] or rf.get("fit_rotations") or 72)
        pcfg = {**PLANES, **{k: v for k, v in (cfg.get("planes") or {}).items() if k in PLANES}}
        seed = int(cfg.get("run", {}).get("seed", 0))
        # provisional factor of stage_io.units_per_cm, computed here because a stale scale.npy must not be read
        lo, hi = np.percentile(raw, [2, 98], axis=0)
        upc = (float(np.max(hi - lo)) or 1.0) / float(scfg.get("provisional_extent_cm", 500.0))
        points = prepare_cloud(raw, ocfg, upc)
        T, floor_from = floor_frame(points, cam_R, centers, pcfg, upc, seed)
        pf = room_fill.to_frame(T, points)
        z_lo, z_hi = np.percentile(pf[:, 2], [1, 99])
        keep = (pf[:, 2] > z_lo + band[0] * (z_hi - z_lo)) & (pf[:, 2] < z_lo + band[1] * (z_hi - z_lo))
        q = pf[keep, :2]
        row.update(n_points=len(raw), n_points_clean=len(points), floor_from=floor_from, n_wall_points=len(q))
        if len(q) > room_fill.FIT_POINTS:
            q = q[np.random.default_rng(seed).choice(len(q), room_fill.FIT_POINTS, replace=False)]
        fit = None
        if len(q) < 10:
            log.warning("scale_outline: only %d wall points, scale stays unknown", len(q))
        else:
            try:
                fit = room_fill.fit_outline(q, footprint, n_rot)
            except (RuntimeError, np.linalg.LinAlgError) as exc:
                log.warning("scale_outline: %s, scale stays unknown", exc)
        if fit is not None:
            support = outline_support(q, fit)
            ok = (fit["residual_median_cm"] <= ocfg["max_residual_cm"] and
                  fit["residual_p90_cm"] <= ocfg["max_p90_cm"] and
                  fit["inlier_frac"] >= ocfg["min_inlier_frac"] and support >= ocfg["min_support"])
            row.update(mirrored=fit["mirrored"], rotation_deg=round(np.degrees(fit["theta"]), 3),
                       fit_residual_median_cm=round(fit["residual_median_cm"], 3),
                       fit_residual_p90_cm=round(fit["residual_p90_cm"], 3),
                       fit_inlier_frac=round(fit["inlier_frac"], 4), outline_support=round(support, 4),
                       fit_ambiguity=round(fit["ambiguity"], 4) if np.isfinite(fit["ambiguity"]) else "",
                       s_units_per_cm=round(fit["s"], 8), accepted=ok)
            if ok:
                s = 1.0 / fit["s"]
                xy = room_fill.place(fit["poly"], fit["s"], fit["theta"], fit["t"])
                outline_model = room_fill.to_frame(np.linalg.inv(T), np.c_[xy, np.zeros(len(xy))])
            else:
                log.warning("scale_outline: poor outline fit (median %.1f cm, inliers %.0f%%, support %.0f%%), "
                            "scale stays unknown", fit["residual_median_cm"], 100 * fit["inlier_frac"], 100 * support)

    known = bool(np.isfinite(s))
    if known:
        row.update(scale=round(s, 6), unit="cm")
    elif method != "outline":
        # the aruco|manual outputs (nan, raw copy) are already there: only the attempt is recorded
        row["t_s"] = round(time.perf_counter() - t0, 3)
        stage_io.log_metrics(log, row)
        stage_io.upsert_csv(artifacts_dir / "scale_outline.csv", row)
        return s
    write_outputs(run_dir, s, outline_model)
    row["t_s"] = round(time.perf_counter() - t0, 3)
    stage_io.log_metrics(log, row)
    stage_io.upsert_csv(artifacts_dir / "scale.csv", row)

    return s
# <<< END IMPLEMENTATION <<<

if __name__ == "__main__":
    import tempfile

    logging.basicConfig(level=logging.WARNING)

    def sfm_from_synthetic(tmp: Path, seed: int = 0) -> tuple[Path, dict]:
        """room_fill.synthetic_run cloud (model units) moved into sfm/ with cameras looking horizontally."""

        rng = np.random.default_rng(seed)
        run_dir, truth = room_fill.synthetic_run(tmp, rng, labels=False)
        pts = np.load(run_dir / "clean/patch_match/points.npy")
        Rw, tw, s_true = truth["Rw"], truth["tw"], truth["s"]
        up = Rw[:, 2]
        cam_R, centers = [], []
        for k in range(12):
            yaw = 2 * np.pi * k / 12
            xy_cm = np.array([130.0, 200.0]) + 60 * np.array([np.cos(yaw), np.sin(yaw)])
            qf = np.r_[s_true * room_fill.rot2(truth["theta"]) @ xy_cm + truth["t"], s_true * 150.0]
            centers.append(Rw @ qf + tw)
            z = Rw @ np.array([np.cos(yaw), np.sin(yaw), 0.0])
            y = -up
            cam_R.append(np.stack([np.cross(y, z), y, z]))
        (run_dir / "sfm").mkdir()
        np.save(run_dir / "sfm/points3d.npy", pts)
        np.save(run_dir / "sfm/centers.npy", np.array(centers))
        np.save(run_dir / "sfm/cam_R.npy", np.array(cam_R))
        return run_dir, truth

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        run_dir, truth = sfm_from_synthetic(tmp)
        art = tmp / "art"
        base = {"run": {"seed": 0}, "planes": {"distance_cm": 2.0},
                "room_fill": {"footprint_cm": room_fill.KITCHEN_CM.tolist(), "fit_rotations": 36},
                "scale": {"enabled": True, "method": "outline", "provisional_extent_cm": 500}}
        s = run(base, str(run_dir), str(art))
        assert abs(s * truth["s"] - 1) < 0.03, (s, 1 / truth["s"])
        row = stage_io.read_csv(art / "scale.csv")[0]
        assert row["method"] == "outline" and row["unit"] == "cm" and row["accepted"] == "True", row
        assert row["floor_from"] == "ransac" and float(row["fit_residual_median_cm"]) < 3.0, row
        assert stage_io.scale_info(run_dir) == (s, "cm")
        assert np.allclose(np.load(run_dir / "scale/centers_cm.npy"), np.load(run_dir / "sfm/centers.npy") * s)
        assert np.load(run_dir / "scale/outline_model.npy").shape == (5, 3)
        assert float(row["outline_support"]) > 0.8, row
        print(f"outline: {s:.3f} cm/unit (truth {1 / truth['s']:.3f}), residual {row['fit_residual_median_cm']} cm, "
              f"support {row['outline_support']}")

        # aruco/manual with a finite scale -> kept; outline method never trusts a stale scale.npy
        np.save(run_dir / "scale/scale.npy", np.array([7.0]))
        assert run({**base, "scale": {**base["scale"], "method": "aruco"}}, str(run_dir), str(art)) == 7.0
        s2 = run(base, str(run_dir), str(art)); assert abs(s2 - s) < 0.01 * s, (s2, s)

        # aruco left nan: fallback fits; fallback off -> nan untouched
        np.save(run_dir / "scale/scale.npy", np.array([np.nan]))
        acfg = {**base, "scale": {**base["scale"], "method": "aruco"}}
        off = {**acfg, "scale": {**acfg["scale"], "outline": {"fallback": False}}}
        assert np.isnan(run(off, str(run_dir), str(art)))
        assert abs(run(acfg, str(run_dir), str(art)) - s) < 0.01 * s
        assert stage_io.read_csv(art / "scale.csv")[0]["fallback_of"] == "aruco"

        # measurements file overrides room_fill; a wrong outline is rejected by the gate (nan, unit model)
        mfile = tmp / "medidas.yaml"
        mfile.write_text(yaml.safe_dump({"contorno_cm": (room_fill.KITCHEN_CM * [3.0, 1.0]).tolist()}))
        bad = {**base, "scale": {**base["scale"], "outline": {"measurements": str(mfile)}}}
        assert np.isnan(run(bad, str(run_dir), str(art)))
        row = stage_io.read_csv(art / "scale.csv")[0]
        assert row["footprint_from"] == str(mfile) and row["accepted"] == "False" and row["unit"] == "model", row
        assert stage_io.scale_info(run_dir)[1] == "model"

        # no outline / no sfm -> nan, never crashes
        assert np.isnan(run({**base, "room_fill": {}}, str(run_dir), str(art)))
        assert np.isnan(run(base, str(tmp / "missing"), str(art)))
    print("OK")

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
# This block is a prompt complement for the AI assistant that will implement
# this file. Read it fully before writing any code. It is self-contained: do not
# look for other documents.
#
# Stage: Stage 5, Evaluation          Owner: unassigned          Due: TBD
# Project: 3D reconstruction of an indoor environment (variant A) from frames
#          extracted from a phone video.
# Context: final metrics of the reconstruction, required for every group, plus
# the indoor-environment (variant A) metrics. Each result is one row in a CSV
# under artifacts/<run_id>/ and the partial values are saved as .npy.
#
# Goal of this file (single responsibility):
#   Variant A: loop drift. With C1 and CN the centers of the first and the
#   last camera (which should coincide), report delta = ||CN - C1|| in cm and
#   as a percentage of the trajectory length. R1 asks for a closed loop, so
#   when the user did not say whether the video closes, decide it here.
#
# What to implement:
#   - Camera centers from scale/centers_cm.npy (working units: cm when the
#     scale is known, raw model units otherwise), ordered by image name
#     (sfm/image_names.npy, img_0000.. = video order). Fallback sfm/centers.npy
#     (model units, times s when scale.npy is known) with WARNING.
#   - delta = ||CN - C1||; trajectory = sum of ||C(i+1) - Ci||.
#   - drift_pct = 100 * delta / trajectory (scale-free: always filled).
#   - cm conversion: stage_io.scale_info first (scale/scale.npy, written by
#     src/sfm/scale.py or the outline scale stage src/sfm/scale_outline.py;
#     scale_method = method column of artifacts scale.csv), else the room_fill
#     outline fit (artifacts room_fill.csv cm_per_model_unit), else cm empty.
#   - Loop closure (evaluation.loop_closed: null | true | false):
#     * always computed (loop_closed_auto + evidence columns):
#       geometry: gap_steps = delta / median step, gap_extent_frac = delta /
#       max ||Ci - C1||, geometry closes when drift_pct <= max_gap_pct or
#       gap_steps <= max_gap_steps;
#       visual (cheap, when the frames exist): ORB + Lowe ratio + RANSAC F
#       between n_samples frames of the first and of the last window_frac of
#       the registered frames; best pair recorded (visual_first/last, inliers)
#       with the camera distance of that pair (pair_delta_*, only when it
#       overlaps: the drift where the video really meets its start); visual
#       closes when inliers >= min_inliers.
#       loop_closed_auto = geometry OR visual (few matches on low texture
#       prove nothing; loop_closed_basis: geometry | visual |
#       geometry+visual | none). sfm_shared_points = 3D points of sfm/obs.npy
#       seen in both windows (0 with a closed loop -> WARNING: the SfM did not
#       close the loop, the drift is accumulated).
#     * loop_closed = YAML value when set (loop_closed_source yaml), else the
#       automatic one (source auto). false -> WARNING (drift is just the
#       start-end distance).
#
# Inputs / outputs:
#   Inputs: results/<run_id>/scale/{centers_cm,scale}.npy, sfm/image_names.npy
#   (fallback sfm/centers.npy), frames in extraction.output_dir, artifacts
#   scale.csv / room_fill.csv.
#   Outputs under results/<run_id>/evaluation/: centers.npy (N, 3), video order.
#   One row in artifacts/<run_id>/loop_drift.csv (COLUMNS below).
#
# Defaults (read from the YAML block evaluation, never hardcoded):
#   enabled: false
#   loop_closed: null      # null | true | false
#   loop_auto: {max_gap_pct: 5.0, max_gap_steps: 10.0, visual: true,
#               window_frac: 0.05, n_samples: 8, orb_features: 2000,
#               ratio: 0.8, ransac_px: 2.0, min_inliers: 30, max_width: 960}
#
# Rules:
#   - Write code ONLY between the BEGIN and END markers below. Do not create
#     other regions or other files for this task.
#   - Configuration comes from the YAML loaded by main.py. No argparse.
#   - Save every partial result as .npy under results/<run_id>/<stage>/.
#   - Keep this block after implementing; update it if the behavior changes.
#   - Expose a function run(cfg: dict, run_dir: str, ...) that main.py can call
#     when the YAML block of this stage has enabled: true (default false).
# ==============================================================================

# >>> BEGIN IMPLEMENTATION >>>
import logging
from pathlib import Path

import cv2
import numpy as np

from src import stage_io

log = logging.getLogger(__name__)

LOOP_AUTO = {"max_gap_pct": 5.0, "max_gap_steps": 10.0, "visual": True, "window_frac": 0.05, "n_samples": 8,
             "orb_features": 2000, "ratio": 0.8, "ransac_px": 2.0, "min_inliers": 30, "max_width": 960}
COLUMNS = ["delta_cm", "trajectory_cm", "drift_pct", "delta_model", "trajectory_model", "loop_closed",
           "scale_source", "loop_closed_source", "loop_closed_auto", "loop_closed_basis", "median_step_model",
           "gap_steps", "extent_model", "gap_extent_frac", "visual_first", "visual_last", "visual_matches",
           "visual_inliers", "sfm_shared_points", "pair_delta_model", "pair_delta_cm", "scale_method",
           "cm_per_model_unit"]

def _load_centers(run_dir: Path, s: float) -> tuple[np.ndarray, np.ndarray | None]:
    """Centers in working units and image names, sorted by image name (= video order)."""

    path = run_dir / "scale" / "centers_cm.npy"
    factor = 1.0
    if not path.exists():
        path = run_dir / "sfm" / "centers.npy"
        factor = s if np.isfinite(s) else 1.0
        log.warning("scale/centers_cm.npy missing: using %s%s", path, " * s" if np.isfinite(s) else " (model units)")
    centers = np.load(path).reshape(-1, 3).astype(np.float64) * factor

    names = None
    names_path = run_dir / "sfm" / "image_names.npy"
    if names_path.exists():
        names = np.load(names_path).astype(str)
        if len(names) == len(centers):
            order = np.argsort(names, kind="stable")
            centers, names = centers[order], names[order]
        else:
            log.warning("image_names (%d) != centers (%d): keeping file order", len(names), len(centers))
            names = None

    return centers, names

def drift(centers: np.ndarray) -> tuple[float, float, float]:
    """(delta, trajectory, drift_pct) in the units of centers."""

    if len(centers) < 2:
        return float("nan"), float("nan"), float("nan")
    delta = float(np.linalg.norm(centers[-1] - centers[0]))
    trajectory = float(np.linalg.norm(np.diff(centers, axis=0), axis=1).sum())
    pct = 100.0 * delta / trajectory if trajectory > 0 else float("nan")

    return delta, trajectory, pct

def room_fill_scale(artifacts_dir: Path) -> float:
    """cm per model unit fitted by room_fill (last row of room_fill.csv), nan when absent."""

    rows = stage_io.read_csv(artifacts_dir / "room_fill.csv") if (artifacts_dir / "room_fill.csv").exists() else []
    try:
        return float(rows[-1]["cm_per_model_unit"]) if rows else float("nan")
    except (KeyError, ValueError):
        return float("nan")

def _scale(run_dir: Path, artifacts_dir: Path) -> tuple[float, float, str, str]:
    """(s, to_cm, source, method): s = cm per model unit; to_cm converts working units (centers) to cm.
    scale.npy (outline / ArUco / manual, working units are cm) > room_fill outline fit > unknown."""

    s, unit = stage_io.scale_info(run_dir)
    if unit == "cm":
        rows = stage_io.read_csv(artifacts_dir / "scale.csv") if (artifacts_dir / "scale.csv").exists() else []
        return s, 1.0, "scale.npy", (rows[-1].get("method") or "") if rows else ""
    s = room_fill_scale(artifacts_dir)
    if np.isfinite(s):
        return s, s, "room_fill", "outline_fit"
    return float("nan"), float("nan"), "", ""

def _geometry(centers: np.ndarray, delta: float) -> dict:
    """Start-end gap relative to the median step and to the farthest point reached from the start."""

    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    med = float(np.median(steps)) if len(steps) else float("nan")
    extent = float(np.linalg.norm(centers - centers[0], axis=1).max()) if len(centers) else float("nan")

    return {"median_step_model": med, "gap_steps": delta / med if med > 0 else float("nan"),
            "extent_model": extent, "gap_extent_frac": delta / extent if extent > 0 else float("nan")}

def _orb(path: Path, orb, max_width: int):

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    f = max_width / max(image.shape)
    if f < 1:
        image = cv2.resize(image, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    kp, desc = orb.detectAndCompute(image, None)

    return (np.float32([k.pt for k in kp]), desc) if desc is not None and len(kp) >= 8 else None

def _match(a, b, ratio: float, ransac_px: float) -> tuple[int, int]:
    """(Lowe matches, RANSAC F inliers) between two ORB feature sets."""

    if a is None or b is None:
        return 0, 0
    knn = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(a[1], b[1], k=2)
    good = [p[0] for p in knn if len(p) == 2 and p[0].distance < ratio * p[1].distance]
    if len(good) < 8:
        return len(good), 0
    pa = a[0][[m.queryIdx for m in good]]
    pb = b[0][[m.trainIdx for m in good]]
    F, mask = cv2.findFundamentalMat(pa, pb, cv2.FM_RANSAC, ransac_px, 0.999)
    if F is None or mask is None:  # the mask is garbage when no F is found
        return len(good), 0

    return len(good), int((mask.ravel() > 0).sum())

def _visual(frames_dir: Path, names: np.ndarray, la: dict) -> tuple[int, int, int, int] | None:
    """Best ORB overlap between n_samples frames of the first and of the last window of the registered
    frames: (index first, index last, matches, inliers); None when the frames are not available."""

    n = len(names)
    window = max(1, int(round(la["window_frac"] * n)))
    k = max(1, int(la["n_samples"]))
    first = np.unique(np.linspace(0, min(window, n) - 1, k).round().astype(int))
    last = np.unique(np.linspace(max(n - window, 0), n - 1, k).round().astype(int))
    if not all((frames_dir / names[i]).exists() for i in (first[0], last[-1])):
        log.warning("frames not found in %s: visual loop check skipped", frames_dir)
        return None

    orb = cv2.ORB_create(int(la["orb_features"]))
    feats = {i: _orb(frames_dir / names[i], orb, int(la["max_width"])) for i in np.union1d(first, last)}
    best = (int(first[0]), int(last[-1]), 0, -1)
    for i in first:
        for j in last:
            if j <= i:
                continue
            matches, inliers = _match(feats[i], feats[j], la["ratio"], la["ransac_px"])
            if inliers > best[3]:
                best = (int(i), int(j), matches, inliers)

    return best if best[3] >= 0 else None

def _shared_points(run_dir: Path, names: np.ndarray, window_frac: float) -> int | None:
    """3D points observed both in the first and in the last window of the registered frames (sfm/obs.npy:
    point_idx, cam_idx in image_names file order); None when obs.npy is missing."""

    path = run_dir / "sfm" / "obs.npy"
    if not path.exists():
        return None
    obs = np.load(path, mmap_mode="r")
    file_names = np.load(run_dir / "sfm" / "image_names.npy").astype(str)
    rank = np.empty(len(file_names), dtype=np.int64)
    rank[np.argsort(file_names, kind="stable")] = np.arange(len(file_names))
    point, cam = np.asarray(obs[:, 0], dtype=np.int64), rank[np.asarray(obs[:, 1], dtype=np.int64)]
    n = len(names)
    window = max(1, int(round(window_frac * n)))

    return int(len(np.intersect1d(point[cam < window], point[cam >= n - window])))

def _fmt(value, nd: int = 4):
    return "" if value is None or (isinstance(value, float) and not np.isfinite(value)) else round(float(value), nd)

def run(cfg: dict, run_dir: str, artifacts_dir: str) -> dict | None:

    ev = cfg["evaluation"]
    if not ev.get("enabled", False):
        log.info("evaluation disabled: skipping loop drift")
        return None

    run_dir, artifacts_dir = Path(run_dir), Path(artifacts_dir)
    if not (run_dir / "scale" / "centers_cm.npy").exists() and not (run_dir / "sfm" / "centers.npy").exists():
        log.warning("no camera centers under %s: skipping loop drift", run_dir)
        return None

    s, to_cm, scale_source, scale_method = _scale(run_dir, artifacts_dir)
    centers, names = _load_centers(run_dir, s)
    out_dir = run_dir / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "centers.npy", centers)

    delta, trajectory, pct = drift(centers)
    if not scale_source:
        log.warning("scale unknown: delta_cm and trajectory_cm left empty (drift_pct is scale-free)")
    elif scale_source == "room_fill":
        log.info("loop drift in cm from the room_fill outline fit (%.4g cm per model unit)", to_cm)
    else:
        log.info("loop drift in cm from scale/scale.npy (%s, %.4g cm per model unit)", scale_method or "?", s)

    # automatic closure: geometry always, visual overlap of the first/last frames when the images exist
    la = {**LOOP_AUTO, **(ev.get("loop_auto") or {})}
    geo = _geometry(centers, delta)
    geo_closed = bool(pct <= la["max_gap_pct"] or geo["gap_steps"] <= la["max_gap_steps"])
    frames_dir = (cfg.get("extraction") or {}).get("output_dir")
    vis = None
    if la["visual"] and names is not None and frames_dir and len(centers) >= 2:
        vis = _visual(Path(frames_dir), names, la)
    vis_closed = vis is not None and vis[3] >= la["min_inliers"]
    # either evidence that start and end coincide is enough: few matches on low texture prove nothing
    auto = geo_closed or vis_closed
    basis = "+".join(b for b, ok in (("geometry", geo_closed), ("visual", vis_closed)) if ok) or "none"
    shared = _shared_points(run_dir, names, la["window_frac"]) if names is not None else None
    if auto and shared == 0:
        log.warning("loop closes (%s) but the SfM shares no 3D point between the first and the last %.0f%% of "
                    "the frames: the reconstruction did not close the loop (drift is accumulated)", basis,
                    100 * la["window_frac"])
    # drift where the video visibly meets its start (only when the best pair really overlaps)
    pair_delta = float(np.linalg.norm(centers[vis[1]] - centers[vis[0]])) if vis_closed else float("nan")

    loop_closed = ev.get("loop_closed")
    source = "yaml" if loop_closed is not None else "auto"
    if loop_closed is None:
        loop_closed = auto
        log.info("loop_closed not set: automatic decision %s (%s)", str(auto).lower(), basis)
    elif bool(loop_closed) != auto:
        log.warning("evaluation.loop_closed=%s but the automatic check says %s (%s)",
                    str(bool(loop_closed)).lower(), str(auto).lower(), basis)
    if not loop_closed:
        log.warning("loop not closed: drift measures start-end distance, not drift")

    row = {
        "delta_cm": _fmt(delta * to_cm) if scale_source else "",
        "trajectory_cm": _fmt(trajectory * to_cm) if scale_source else "",
        "drift_pct": _fmt(pct),
        "delta_model": _fmt(delta, 6),
        "trajectory_model": _fmt(trajectory, 6),
        "loop_closed": str(bool(loop_closed)).lower(),
        "scale_source": scale_source,  # scale.npy (outline / ArUco / manual) | room_fill | "" unknown
        "loop_closed_source": source,
        "loop_closed_auto": str(bool(auto)).lower(),
        "loop_closed_basis": basis,
        "median_step_model": _fmt(geo["median_step_model"], 6),
        "gap_steps": _fmt(geo["gap_steps"], 2),
        "extent_model": _fmt(geo["extent_model"], 6),
        "gap_extent_frac": _fmt(geo["gap_extent_frac"]),
        "visual_first": str(names[vis[0]]) if vis is not None else "",
        "visual_last": str(names[vis[1]]) if vis is not None else "",
        "visual_matches": vis[2] if vis is not None else "",
        "visual_inliers": vis[3] if vis is not None else "",
        "sfm_shared_points": "" if shared is None else shared,
        "pair_delta_model": _fmt(pair_delta, 6),
        "pair_delta_cm": _fmt(pair_delta * to_cm) if scale_source else "",
        "scale_method": scale_method,
        "cm_per_model_unit": _fmt(s, 6),
    }
    stage_io.log_metrics(log, row)
    stage_io.upsert_csv(artifacts_dir / "loop_drift.csv", row)

    return row

if __name__ == "__main__":
    import tempfile

    logging.basicConfig(level=logging.INFO)
    side = 100.0
    corners = np.array([[0, 0, 0], [side, 0, 0], [side, side, 0], [0, side, 0], [0, 10, 0]], dtype=np.float64)
    pts = np.concatenate([np.linspace(corners[i], corners[i + 1], 10, endpoint=False) for i in range(4)]
                         + [corners[-1:]])
    names = np.array([f"img_{i:04d}.png" for i in range(len(pts))])
    perm = np.random.default_rng(0).permutation(len(pts))
    expected_traj = 3 * side + (side - 10)

    with tempfile.TemporaryDirectory() as tmp:
        rd, ad = Path(tmp) / "results", Path(tmp) / "artifacts"
        (rd / "scale").mkdir(parents=True); (rd / "sfm").mkdir()
        np.save(rd / "sfm" / "image_names.npy", names[perm])
        np.save(rd / "scale" / "centers_cm.npy", pts[perm])
        np.save(rd / "scale" / "scale.npy", np.array([2.0]))
        stage_io.upsert_csv(ad / "scale.csv", {"method": "outline", "scale": 2.0, "unit": "cm"})
        cfg = {"evaluation": {"enabled": True, "loop_closed": True}}

        # 1) metric scale from scale.npy (outline stage), YAML says closed
        row = run(cfg, str(rd), str(ad))
        assert abs(row["delta_cm"] - 10.0) < 1e-6 and abs(row["trajectory_cm"] - expected_traj) < 1e-6
        assert abs(row["drift_pct"] - 100 * 10 / expected_traj) < 1e-3 and row["loop_closed"] == "true"
        assert row["scale_source"] == "scale.npy" and row["scale_method"] == "outline"
        assert row["loop_closed_source"] == "yaml" and row["loop_closed_basis"] == "geometry"
        assert row["sfm_shared_points"] == ""  # no obs.npy
        assert row["loop_closed_auto"] == "true" and abs(row["gap_steps"] - 1.0) < 1e-6  # gap = one 10 cm step
        assert np.allclose(np.load(rd / "evaluation" / "centers.npy"), pts)
        assert list(stage_io.read_csv(ad / "loop_drift.csv")[0]) == COLUMNS

        # 2) no scale, no room_fill: cm empty, closure decided automatically
        (rd / "scale" / "scale.npy").unlink()
        cfg["evaluation"]["loop_closed"] = None
        row = run(cfg, str(rd), str(ad))
        assert row["delta_cm"] == "" and row["trajectory_cm"] == "" and row["scale_source"] == ""
        assert row["loop_closed"] == "true" and row["loop_closed_source"] == "auto"
        assert abs(row["delta_model"] - 10.0) < 1e-6

        # 3) room_fill fallback
        stage_io.upsert_csv(ad / "room_fill.csv", {"cm_per_model_unit": 3.0})
        row = run(cfg, str(rd), str(ad))
        assert row["scale_source"] == "room_fill" and abs(row["delta_cm"] - 30.0) < 1e-6
        (ad / "room_fill.csv").unlink()

        # 4) open path (ends at the opposite corner): geometry says not closed
        np.save(rd / "scale" / "centers_cm.npy", np.linspace([0, 0, 0], [side, side, 0], len(pts))[perm])
        row = run(cfg, str(rd), str(ad))
        assert row["loop_closed"] == "false" and row["loop_closed_auto"] == "false"

        # 5) visual check: textured frames, last frame = first frame shifted (closed) vs unrelated noise;
        #    obs.npy: point 0 seen by the first and the last camera
        rng = np.random.default_rng(1)
        frames = Path(tmp) / "frames"; frames.mkdir()
        base = cv2.GaussianBlur(rng.integers(0, 255, (480, 640), dtype=np.uint8), (5, 5), 0)
        for i, name in enumerate(names):
            img = np.roll(base, 3 * i, axis=1) if i in (0, len(names) - 1) else \
                cv2.GaussianBlur(rng.integers(0, 255, (480, 640), dtype=np.uint8), (5, 5), 0)
            cv2.imwrite(str(frames / name), img)
        cfg["extraction"] = {"output_dir": str(frames)}
        cfg["evaluation"]["loop_auto"] = {"window_frac": 0.05, "n_samples": 2}
        first_cam, last_cam = (int(np.flatnonzero(names[perm] == names[k])[0]) for k in (0, -1))
        np.save(rd / "sfm" / "obs.npy", np.array([[0, first_cam, 1, 1], [0, last_cam, 2, 2], [1, first_cam, 3, 3]],
                                                 dtype=np.float64))
        row = run(cfg, str(rd), str(ad))
        assert row["loop_closed_basis"] == "visual" and row["loop_closed_auto"] == "true", row
        assert row["sfm_shared_points"] == 1, row
        assert row["visual_first"] == names[0] and row["visual_last"] == names[-1] and row["visual_inliers"] >= 30
        assert abs(row["pair_delta_model"] - row["delta_model"]) < 1e-9
        cv2.imwrite(str(frames / names[-1]), rng.integers(0, 255, (480, 640), dtype=np.uint8))
        assert run(cfg, str(rd), str(ad))["loop_closed_auto"] == "false"

        cfg["evaluation"]["loop_closed"] = False
        assert run(cfg, str(rd), str(ad))["loop_closed"] == "false"

        # 6) centers_cm.npy missing with a known scale: sfm/centers.npy (model units) times s
        (rd / "scale" / "centers_cm.npy").unlink()
        np.save(rd / "sfm" / "centers.npy", pts[perm] / 4.0)
        np.save(rd / "scale" / "scale.npy", np.array([4.0]))
        row = run(cfg, str(rd), str(ad))
        assert abs(row["delta_cm"] - 10.0) < 1e-6 and row["scale_source"] == "scale.npy"
        assert run({"evaluation": {"enabled": False}}, str(rd), str(ad)) is None
    print("OK")
# <<< END IMPLEMENTATION <<<

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
#   as a percentage of the trajectory length.
#
# What to implement:
#   - Camera centers from scale/centers_cm.npy (working units: cm when the
#     scale is known, raw model units otherwise), ordered by image name
#     (sfm/image_names.npy, img_0000.. = video order).
#   - delta = ||CN - C1||; trajectory = sum of ||C(i+1) - Ci||.
#   - drift_pct = 100 * delta / trajectory (scale-free: always filled).
#   - unit = model: delta_cm / trajectory_cm written empty; delta_model and
#     trajectory_model always filled.
#   - evaluation.loop_closed (user answer: does the video start and end at the
#     same spot?): false -> WARNING (drift is just the start-end distance);
#     null -> WARNING (unknown). Value written in column loop_closed.
#
# Inputs / outputs:
#   Inputs: results/<run_id>/scale/centers_cm.npy, results/<run_id>/sfm/
#   image_names.npy (fallback sfm/centers.npy with WARNING).
#   Outputs under results/<run_id>/evaluation/: centers.npy (N, 3), video order.
#   One row in artifacts/<run_id>/loop_drift.csv: delta_cm, trajectory_cm,
#   drift_pct, delta_model, trajectory_model, loop_closed, scale_source. Without a metric scale
#   (scale/scale.npy) the cm columns use the room_fill outline fit (artifacts room_fill.csv
#   cm_per_model_unit, scale_source room_fill); with neither they stay empty.
#
# Defaults (read from the YAML block evaluation, never hardcoded):
#   enabled: false
#   loop_closed: null      # null | true | false
#
# Rules:
#   - Write code ONLY between the BEGIN and END markers below. Do not create
#     other regions or other files for this task.
#   - Configuration comes from the YAML loaded by main.py. No argparse.
#   - Save every partial result as .npy under results/<run_id>/<stage>/.
#   - Keep this block after implementing; update it if the behavior changes.
#   - Expose a function run(cfg: dict, run_dir: str, ...) that main.py can call
#     when the YAML block of this stage has enabled: true (default false).
#   - Pipeline already available: frames are extracted and quality-filtered, and
#     SIFT, ORB and AKAZE keypoints/descriptors are saved per frame under
#     results/<run_id>/detection/<detector>/<frame>/ as keypoints.npy (N, 6:
#     x, y, size, angle, response, octave) and descriptors.npy (N, D).
# ==============================================================================

# >>> BEGIN IMPLEMENTATION >>>
import logging
from pathlib import Path

import numpy as np

from src import stage_io

log = logging.getLogger(__name__)

def _load_centers(run_dir: Path) -> np.ndarray:
    """Centers in working units, sorted by image name (= video order)."""

    path = run_dir / "scale" / "centers_cm.npy"
    if not path.exists():
        path = run_dir / "sfm" / "centers.npy"
        log.warning("scale/centers_cm.npy missing: using %s (model units)", path)
    centers = np.load(path).reshape(-1, 3).astype(np.float64)

    names_path = run_dir / "sfm" / "image_names.npy"
    if names_path.exists():
        names = np.load(names_path).astype(str)
        if len(names) == len(centers):
            centers = centers[np.argsort(names, kind="stable")]
        else:
            log.warning("image_names (%d) != centers (%d): keeping file order", len(names), len(centers))

    return centers

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

def run(cfg: dict, run_dir: str, artifacts_dir: str) -> dict | None:

    ev = cfg["evaluation"]
    if not ev.get("enabled", False):
        log.info("evaluation disabled: skipping loop drift")
        return None

    run_dir = Path(run_dir)
    if not (run_dir / "scale" / "centers_cm.npy").exists() and not (run_dir / "sfm" / "centers.npy").exists():
        log.warning("no camera centers under %s: skipping loop drift", run_dir)
        return None

    centers = _load_centers(run_dir)
    out_dir = run_dir / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "centers.npy", centers)

    _, unit = stage_io.scale_info(run_dir)
    delta, trajectory, pct = drift(centers)
    to_cm, scale_source = (1.0, "scale") if unit == "cm" else (room_fill_scale(Path(artifacts_dir)), "room_fill")
    if not np.isfinite(to_cm):
        to_cm, scale_source = float("nan"), ""
        log.warning("scale unknown: delta_cm and trajectory_cm left empty (drift_pct is scale-free)")
    elif scale_source == "room_fill":
        log.info("loop drift in cm from the room_fill outline fit (%.4g cm per model unit)", to_cm)

    loop_closed = ev.get("loop_closed")
    if loop_closed is None:
        log.warning("loop_closed unknown (evaluation.loop_closed: null): drift assumes start and end coincide")
    elif not loop_closed:
        log.warning("loop not closed: drift measures start-end distance, not drift")

    row = {
        "delta_cm": round(delta * to_cm, 4) if scale_source else "",
        "trajectory_cm": round(trajectory * to_cm, 4) if scale_source else "",
        "drift_pct": round(pct, 4),
        "delta_model": round(delta, 6),
        "trajectory_model": round(trajectory, 6),
        "loop_closed": "" if loop_closed is None else str(bool(loop_closed)).lower(),
        "scale_source": scale_source,  # scale (ArUco / manual) | room_fill (hand-measured outline) | "" unknown
    }
    stage_io.log_metrics(log, row)
    stage_io.upsert_csv(Path(artifacts_dir) / "loop_drift.csv", row)

    return row

if __name__ == "__main__":
    import tempfile

    logging.basicConfig(level=logging.INFO)
    side = 100.0
    corners = np.array([[0, 0, 0], [side, 0, 0], [side, side, 0], [0, side, 0], [0, 10, 0]], dtype=np.float64)
    pts = np.concatenate([np.linspace(corners[i], corners[i + 1], 10, endpoint=False) for i in range(4)]
                         + [corners[-1:]])
    names = np.array([f"img_{i:04d}" for i in range(len(pts))])
    perm = np.random.default_rng(0).permutation(len(pts))

    with tempfile.TemporaryDirectory() as tmp:
        rd, ad = Path(tmp) / "results", Path(tmp) / "artifacts"
        (rd / "scale").mkdir(parents=True); (rd / "sfm").mkdir()
        np.save(rd / "sfm" / "image_names.npy", names[perm])
        np.save(rd / "scale" / "centers_cm.npy", pts[perm])
        np.save(rd / "scale" / "scale.npy", np.array([2.0]))
        cfg = {"evaluation": {"enabled": True, "loop_closed": True}}

        row = run(cfg, str(rd), str(ad))
        expected_traj = 3 * side + (side - 10)
        assert abs(row["delta_cm"] - 10.0) < 1e-6 and abs(row["trajectory_cm"] - expected_traj) < 1e-6
        assert abs(row["drift_pct"] - 100 * 10 / expected_traj) < 1e-3 and row["loop_closed"] == "true"
        assert np.allclose(np.load(rd / "evaluation" / "centers.npy"), pts)
        csv_row = stage_io.read_csv(ad / "loop_drift.csv")[0]
        assert list(csv_row) == ["delta_cm", "trajectory_cm", "drift_pct", "delta_model", "trajectory_model",
                                 "loop_closed", "scale_source"]

        (rd / "scale" / "scale.npy").unlink()
        cfg["evaluation"]["loop_closed"] = None
        row = run(cfg, str(rd), str(ad))
        assert row["delta_cm"] == "" and row["trajectory_cm"] == "" and row["loop_closed"] == ""
        assert abs(row["delta_model"] - 10.0) < 1e-6 and abs(row["drift_pct"] - 100 * 10 / expected_traj) < 1e-3
        assert stage_io.read_csv(ad / "loop_drift.csv")[0]["delta_cm"] == ""

        stage_io.upsert_csv(ad / "room_fill.csv", {"cm_per_model_unit": 3.0})
        row = run(cfg, str(rd), str(ad))
        assert row["scale_source"] == "room_fill" and abs(row["delta_cm"] - 30.0) < 1e-6
        (ad / "room_fill.csv").unlink()

        cfg["evaluation"]["loop_closed"] = False
        assert run(cfg, str(rd), str(ad))["loop_closed"] == "false"
        assert run({"evaluation": {"enabled": False}}, str(rd), str(ad)) is None
    print("OK")
# <<< END IMPLEMENTATION <<<

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
#   - Camera center C = -R^T t for each camera, in scaled (cm) units.
#   - delta_cm = ||CN - C1||; trajectory length = sum of ||C(i+1) - Ci||.
#   - drift_pct = 100 * delta_cm / trajectory length.
#
# Inputs / outputs:
#   Inputs: camera poses (R, t) of all cameras from results/<run_id>/.
#   Outputs under results/<run_id>/evaluation/: centers.npy (N, 3).
#   One row in artifacts/<run_id>/loop_drift.csv: delta_cm, trajectory_cm,
#   drift_pct.
#
# Defaults (read from the YAML block evaluation, never hardcoded):
#   enabled: false
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
"""Loop-closure drift from global camera centers or chained two-view poses."""
import logging
from pathlib import Path

import numpy as np

from src import stage_io

log = logging.getLogger(__name__)


def drift(centers: np.ndarray) -> tuple[float, float, float]:
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    if len(centers) < 2:
        return float("nan"), float("nan"), float("nan")
    delta = float(np.linalg.norm(centers[-1] - centers[0]))
    trajectory = float(np.linalg.norm(np.diff(centers, axis=0), axis=1).sum())
    percentage = 100.0 * delta / trajectory if trajectory > 0 else float("nan")
    return delta, trajectory, percentage


def _global_centers(root: Path) -> tuple[np.ndarray, bool] | None:
    scaled = root / "scale" / "centers_cm.npy"
    model = root / "sfm" / "centers.npy"
    path = scaled if scaled.exists() else model
    if not path.exists():
        return None
    centers = np.load(path).reshape(-1, 3)
    names = root / "sfm" / "image_names.npy"
    if names.exists():
        image_names = np.load(names).astype(str)
        if len(image_names) == len(centers):
            centers = centers[np.argsort(image_names, kind="stable")]
    return centers, path == scaled


def run(cfg: dict, run_dir: str, artifacts_dir: str) -> list[dict] | None:
    if not cfg["evaluation"]["enabled"]:
        return None

    root = Path(run_dir)
    out = root / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    for path in out.glob("centers*.npy"):
        path.unlink()
    csv_path = Path(artifacts_dir) / "loop_drift.csv"
    csv_path.unlink(missing_ok=True)
    global_result = _global_centers(root)
    if global_result is None:
        log.warning("loop_drift: global camera centers unavailable; pairwise poses cannot measure loop drift")
        return None
    groups = [("global", *global_result)]

    loop_closed = cfg["evaluation"].get("loop_closed")
    if loop_closed is not True:
        log.warning("loop_drift: loop_closed is not true; start-end distance must be interpreted cautiously")

    rows = []
    for index, (detector, centers, scaled) in enumerate(groups):
        delta, trajectory, percentage = drift(centers)
        row = {
            "detector": detector,
            "source": "sfm",
            "delta_cm": delta if scaled else "",
            "trajectory_cm": trajectory if scaled else "",
            "drift_pct": percentage,
            "delta_model": delta,
            "trajectory_model": trajectory,
            "loop_closed": "" if loop_closed is None else str(bool(loop_closed)).lower(),
        }
        stage_io.upsert_csv(csv_path, row, key="detector")
        np.save(out / f"centers_{detector}.npy", centers)
        if index == 0:
            np.save(out / "centers.npy", centers)
        stage_io.log_metrics(log, {f"{detector}.{key}": value for key, value in row.items()
                                   if key not in ("detector", "source")})
        rows.append(row)
    return rows

# <<< END IMPLEMENTATION <<<

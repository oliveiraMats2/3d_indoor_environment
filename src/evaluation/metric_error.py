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
#   Metric error of every dimension measured by hand at collection time.
#   With L the real measure and L_hat the measure in the model, report
#   |L_hat - L| and the relative error e = |L_hat - L| / L.
#   Variant A: report the errors of the room dimensions in cm.
#
# What to implement:
#   - For each entry in measured_cm, read its L_hat from the scaled model.
#   - Compute abs_error_cm = |L_hat - L| and rel_error = abs_error_cm / L.
#
# Inputs / outputs:
#   Inputs: scaled model measures from results/<run_id>/.
#   Outputs under results/<run_id>/evaluation/: metric_errors.npy (K, 2):
#   abs_error_cm, rel_error. One row per dimension in
#   artifacts/<run_id>/metric_error.csv: name, value_cm, model_cm,
#   abs_error_cm, rel_error.
#
# Defaults (read from the YAML block evaluation.measured_cm, never hardcoded):
#   measured_cm: a list of {name, value_cm}
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
"""Compare manually measured room dimensions with the reconstructed floor plan."""
import csv
import logging
import math
from pathlib import Path

import numpy as np

from src import stage_io

log = logging.getLogger(__name__)

MODEL_COLUMNS = {
    "room_width": "width_cm",
    "room_length": "length_cm",
    "ceiling_height": "ceiling_height_cm",
}
FIELDS = ["name", "value_cm", "model_cm", "abs_error_cm", "rel_error"]


def _number(row: dict, key: str) -> float:
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def run(cfg: dict, run_dir: str, artifacts_dir: str) -> Path | None:
    if not cfg["evaluation"]["enabled"]:
        return None

    measures = [dict(item) for item in cfg["evaluation"].get("measured_cm", [])]
    csv_path = Path(artifacts_dir) / "metric_error.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    out = Path(run_dir) / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    floorplan_rows = stage_io.read_csv(Path(artifacts_dir) / "floorplan.csv")
    floorplan = floorplan_rows[-1] if floorplan_rows else {}
    if measures and not floorplan:
        log.warning("metric_error: floorplan.csv unavailable; model measurements stay empty")

    rows = []
    errors = np.full((len(measures), 2), np.nan, dtype=np.float64)
    for index, measure in enumerate(measures):
        name = measure["name"]
        real = float(measure["value_cm"])
        model = _number(floorplan, MODEL_COLUMNS[name])
        row = {"name": name, "value_cm": real, "model_cm": "", "abs_error_cm": "", "rel_error": ""}
        if np.isfinite(model):
            absolute = abs(model - real)
            relative = absolute / real if real else float("nan")
            errors[index] = absolute, relative
            row.update(model_cm=model, abs_error_cm=absolute,
                       rel_error=relative if np.isfinite(relative) else "")
        rows.append(row)

    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    np.save(out / "metric_errors.npy", errors)
    if not measures:
        log.warning("metric_error: evaluation.measured_cm is empty")
    return csv_path

# <<< END IMPLEMENTATION <<<

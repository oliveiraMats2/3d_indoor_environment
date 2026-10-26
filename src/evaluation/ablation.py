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
#   Variant A: ablation study. Compare exhaustive vs. sequential matching
#   (with and without loop detection), or the number of photos/frames used.
#   One YAML per variant (one experiment per config), and this file only reads
#   their results and builds the comparison table.
#
# What to implement:
#   - Read the evaluation CSVs of each listed run id under artifacts/.
#   - Build one table, one row per variant, with the same metric columns.
#   - Run nothing itself; experiments are run by main.py with their own YAML.
#
# Inputs / outputs:
#   Inputs: artifacts/<variant_run_id>/*.csv for each listed variant.
#   Outputs: artifacts/<run_id>/ablation.csv (one row per variant) and
#   results/<run_id>/evaluation/ablation.npy with the numeric table.
#
# Defaults (read from the YAML block evaluation, never hardcoded):
#   enabled: false
#   ablation_runs: a list of run ids to compare
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
"""Build an ablation table from results already produced by independent YAML runs."""
import csv
import logging
from pathlib import Path

import numpy as np

from src import stage_io

log = logging.getLogger(__name__)

COLUMNS = ["run_id", "detector", "reproj_mean_px", "reproj_median_px", "n_sparse",
           "n_dense", "drift_pct", "metric_error_mean_rel"]
NUMERIC = COLUMNS[2:]


def _float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _rows(run_id: str, artifacts_root: Path) -> list[dict]:
    folder = artifacts_root / run_id
    evaluation_rows = stage_io.read_csv(folder / "evaluation.csv") or [{}]
    drift_rows = stage_io.read_csv(folder / "loop_drift.csv")
    drift_by_detector = {row.get("detector", "global"): row for row in drift_rows}
    relative = [_float(row.get("rel_error")) for row in stage_io.read_csv(folder / "metric_error.csv")]
    relative = [value for value in relative if np.isfinite(value)]
    relative_mean = float(np.mean(relative)) if relative else ""
    rows = []
    for evaluation in evaluation_rows:
        detector = evaluation.get("detector", "global")
        drift_row = drift_by_detector.get(detector, drift_by_detector.get("global", {}))
        rows.append({
            "run_id": run_id,
            "detector": detector,
            "reproj_mean_px": evaluation.get("reproj_mean_px", ""),
            "reproj_median_px": evaluation.get("reproj_median_px", ""),
            "n_sparse": evaluation.get("n_sparse", ""),
            "n_dense": evaluation.get("n_dense", ""),
            "drift_pct": drift_row.get("drift_pct", ""),
            "metric_error_mean_rel": relative_mean,
        })
    return rows


def run(cfg: dict, run_dir: str, artifacts_dir: str) -> Path | None:
    run_ids = cfg["evaluation"].get("ablation_runs", [])
    if not run_ids:
        return None

    artifacts = Path(artifacts_dir)
    artifacts_root = artifacts.parent
    rows = [row for run_id in run_ids for row in _rows(run_id, artifacts_root)]
    csv_path = artifacts / "ablation.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    out = Path(run_dir) / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    table = np.array([[_float(row[column]) for column in NUMERIC] for row in rows], dtype=np.float64)
    np.save(out / "ablation.npy", table.reshape(len(rows), len(NUMERIC)))
    np.save(out / "ablation_columns.npy", np.asarray(NUMERIC))
    log.info("ablation: %d rows from %d runs", len(rows), len(run_ids))
    return csv_path

# <<< END IMPLEMENTATION <<<

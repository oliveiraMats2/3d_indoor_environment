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
# This block is a prompt complement for the AI assistant that will implement
# this file. Read it fully before writing any code. It is self-contained: do not
# look for other documents.
#
# Stage: Stage 3, part 1 (Matching)          Owner: Emily          Due: Wed 21/10
# Project: 3D reconstruction of an indoor environment (variant A) from frames
#          extracted from a phone video.
# Context: consecutive frames from the video are matched by their descriptors
# (SIFT gives float32 descriptors, ORB and AKAZE give uint8 binary ones). Pairs
# feed Stage 3 part 2 (fundamental/essential matrices, pose, triangulation).
# Fixed rules for this stage:
#   - Same nfeatures for every detector, the same matcher and the same Lowe
#     ratio; otherwise the comparison measures keypoint budget, not quality.
#   - knnMatch(k=2) is required by Lowe's test, so crossCheck must stay False
#     (they are mutually exclusive). Reject crossCheck: true in the YAML.
#   - The descriptor dtype decides the branch: uint8 -> binary, float32 -> float.
# The owner also writes the first part of the Stage 3 report section; this
# package must output the numbers and figures that text needs.
#
# Goal of this file (single responsibility):
#   count matches and compute the inlier rate per pair.
#
# What to implement:
#   One row per pair x detector in matching_metrics.csv:
#   raw_matches       = len(knn)                 (before Lowe)
#   filtered_matches  = matches after Lowe
#   n_inliers         = inlier_mask.sum()        (after RANSAC on E or F)
#   inlier_ratio      = n_inliers / filtered_matches
#   matching_score    = n_inliers / min(keypoints in frame A, keypoints in frame B)
#   success           = n_inliers >= 8            (enough to estimate the pose)
#   match_time_ms     = mean and std over N repetitions, warmup runs discarded
#
# Inputs / outputs:
#   Partial logs (.npy), under results/<run_id>/matching/<detector>/<frameA>__<frameB>/:
#     matches_raw.npy  (M, 3): queryIdx, trainIdx, distance
#     matches_lowe.npy (K, 3)
#     inlier_mask.npy  (K,) bool
#     times_ms.npy     (R,)
#   Out: matching_metrics.csv under artifacts/<run_id>/.
#   The CSV is built only from these .npy files.
#
# Defaults (read from the YAML block filters, never hardcoded):
#   min_matches: 10            # pairs below it are success=False
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

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from src.matching.filters import select_inliers


FIELDS = (
    "detector",
    "pair",
    "raw_matches",
    "filtered_matches",
    "n_inliers",
    "inlier_ratio",
    "matching_score",
    "success",
    "match_time_ms_mean",
    "match_time_ms_std",
)


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else float("nan")


def measure_pair(
    raw_matches: np.ndarray,
    matches_lowe: np.ndarray,
    inlier_mask: np.ndarray,
    times_ms: np.ndarray,
    keypoints_first: int,
    keypoints_second: int,
) -> dict[str, float | int | bool]:
    """Calculate matching quality and runtime metrics for one frame pair."""

    inliers = select_inliers(matches_lowe, inlier_mask)
    raw_count = len(raw_matches)
    filtered_count = len(matches_lowe)
    inlier_count = len(inliers)
    minimum_keypoints = min(keypoints_first, keypoints_second)
    times = np.asarray(times_ms, dtype=np.float64).reshape(-1)

    return {
        "raw_matches": raw_count,
        "filtered_matches": filtered_count,
        "n_inliers": inlier_count,
        "inlier_ratio": _safe_ratio(inlier_count, filtered_count),
        "matching_score": _safe_ratio(inlier_count, minimum_keypoints),
        "success": inlier_count >= 8,
        "match_time_ms_mean": float(times.mean()) if len(times) else float("nan"),
        "match_time_ms_std": float(times.std()) if len(times) else float("nan"),
    }


def _load_keypoint_count(run_dir: Path, detector: str, frame: str) -> int:
    path = run_dir / "detection" / detector / frame / "keypoints.npy"
    return len(np.load(path))


def _pair_rows(run_dir: Path) -> list[dict[str, object]]:
    matching_root = run_dir / "matching"
    if not matching_root.is_dir():
        raise FileNotFoundError(f"matching results not found: {matching_root}")

    rows = []
    detector_dirs = sorted(path for path in matching_root.iterdir() if path.is_dir())
    for detector_dir in detector_dirs:
        pair_dirs = sorted(path for path in detector_dir.iterdir() if path.is_dir())
        for pair_dir in pair_dirs:
            mask_path = pair_dir / "inlier_mask.npy"
            if not mask_path.exists():
                continue

            first, second = pair_dir.name.split("__", maxsplit=1)
            measurements = measure_pair(
                np.load(pair_dir / "matches_raw.npy"),
                np.load(pair_dir / "matches_lowe.npy"),
                np.load(mask_path),
                np.load(pair_dir / "times_ms.npy"),
                _load_keypoint_count(run_dir, detector_dir.name, first),
                _load_keypoint_count(run_dir, detector_dir.name, second),
            )
            rows.append(
                {
                    "detector": detector_dir.name,
                    "pair": pair_dir.name,
                    **measurements,
                }
            )

    return rows


def run(run_dir: str, artifacts_dir: str) -> Path:
    """Build matching_metrics.csv from saved NumPy artifacts."""

    rows = _pair_rows(Path(run_dir))
    output_dir = Path(artifacts_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "matching_metrics.csv"
    with output_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    return output_path

# <<< END IMPLEMENTATION <<<

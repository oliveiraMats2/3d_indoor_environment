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
# Stage: 3, part 2 (Epipolar geometry)          Owner: Maxime          Due: Fri 23/10
# Project: 3D reconstruction of an indoor environment (variant A) from frames
#          extracted from a phone video.
# Context (plain words): a point in frame 1 only gives a direction, like a laser
# beam leaving camera 1 through that pixel. Seen from camera 2 the beam is a
# straight line, the epipolar line; the correct match of the point must lie on
# it. F (fundamental matrix) states this in pixels: x2^T F x1 = 0. E (essential
# matrix) is the same rule after removing the lens with the intrinsics K:
# E = K^T F K, and it keeps only the motion between the cameras.
# RANSAC: sample a few pairs (8 for F, 5 for E), fit, count how many other
# pairs fall close to their epipolar lines, repeat, keep the model with the
# most votes. Voters are inliers; the rest are discarded outliers.
# t comes out only as a direction, without size: metric scale is applied later
# from an object of known size in the scene.
# Input: the Lowe-filtered pairs from the matching package, plus K from the YAML.
# The owner also writes the second part of the Stage 3 report section.
#
# Goal of this file (single responsibility):
#   estimate F and E with RANSAC for each pair.
#
# What to implement:
#   - cv2.findFundamentalMat and cv2.findEssentialMat with the method from the
#     YAML (resolve the string with getattr(cv2, name)).
#   - Return the matrix and the inlier mask; the mask is what matching/metrics.py
#     counts as n_inliers.
#   - Inlier epipolar error: mean Sampson distance in pixels.
#
# Inputs / outputs:
#   In: pts1, pts2 (K, 2) float32 Lowe-filtered matched pixels of one pair; K (3, 3).
#   Out: F.npy (3,3), E.npy (3,3), inlier_mask.npy (K,) bool under
#   results/<run_id>/epipolar/<detector>/<frameA>__<frameB>/
#
# Defaults (read from the YAML block geometry, never hardcoded):
#   ransac.method: USAC_MAGSAC   # less sensitive to the hard threshold than RANSAC
#   ransac.threshold_px: 1.0
#   ransac.confidence: 0.999
#   ransac.max_iters: 5000
#   intrinsics: {fx: null, fy: null, cx: null, cy: null}  # from calibration
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

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class EstimateResult:
    fundamental: np.ndarray
    essential: np.ndarray
    inlier_mask: np.ndarray
    sampson_errors: np.ndarray


def _matrix_or_nan(matrix: np.ndarray | None) -> np.ndarray:
    if matrix is None or matrix.size < 9:
        return np.full((3, 3), np.nan, dtype=np.float64)
    return np.asarray(matrix, dtype=np.float64).reshape(-1, 3, 3)[0]


def _mask_or_false(mask: np.ndarray | None, size: int) -> np.ndarray:
    if mask is None:
        return np.zeros(size, dtype=bool)
    flattened = np.asarray(mask).reshape(-1).astype(bool)
    if len(flattened) != size:
        raise ValueError("RANSAC returned an inlier mask with an unexpected size")
    return flattened


def _sampson_errors(
    fundamental: np.ndarray,
    first_points: np.ndarray,
    second_points: np.ndarray,
) -> np.ndarray:
    if len(first_points) == 0 or np.isnan(fundamental).any():
        return np.empty(0, dtype=np.float64)

    first_h = np.column_stack((first_points, np.ones(len(first_points))))
    second_h = np.column_stack((second_points, np.ones(len(second_points))))
    lines_second = first_h @ fundamental.T
    lines_first = second_h @ fundamental
    numerator = np.sum(second_h * lines_second, axis=1) ** 2
    denominator = (
        lines_second[:, 0] ** 2
        + lines_second[:, 1] ** 2
        + lines_first[:, 0] ** 2
        + lines_first[:, 1] ** 2
    )
    valid = denominator > np.finfo(np.float64).eps
    errors = np.full(len(first_points), np.nan, dtype=np.float64)
    errors[valid] = np.sqrt(numerator[valid] / denominator[valid])
    return errors


def estimate(cfg: dict, first_points: np.ndarray, second_points: np.ndarray, intrinsics: np.ndarray) -> EstimateResult:
    """Estimate F and E and keep correspondences accepted by both models."""

    first = np.asarray(first_points, dtype=np.float32).reshape(-1, 2)
    second = np.asarray(second_points, dtype=np.float32).reshape(-1, 2)
    if len(first) != len(second):
        raise ValueError("first_points and second_points must have the same length")

    size = len(first)
    if size < 8:
        empty_mask = np.zeros(size, dtype=bool)
        nan_matrix = np.full((3, 3), np.nan, dtype=np.float64)
        return EstimateResult(nan_matrix, nan_matrix.copy(), empty_mask, np.empty(0))

    ransac = cfg["ransac"]
    try:
        method = getattr(cv2, ransac["method"])
    except AttributeError as exc:
        raise ValueError(f"OpenCV does not provide RANSAC method {ransac['method']!r}") from exc

    try:
        fundamental_raw, fundamental_mask_raw = cv2.findFundamentalMat(
            first,
            second,
            method,
            ransac["threshold_px"],
            ransac["confidence"],
            ransac["max_iters"],
        )
    except cv2.error:
        fundamental_raw, fundamental_mask_raw = None, None

    try:
        essential_raw, essential_mask_raw = cv2.findEssentialMat(
            first,
            second,
            intrinsics,
            method=method,
            prob=ransac["confidence"],
            threshold=ransac["threshold_px"],
            maxIters=ransac["max_iters"],
        )
    except cv2.error:
        essential_raw, essential_mask_raw = None, None

    fundamental = _matrix_or_nan(fundamental_raw)
    essential = _matrix_or_nan(essential_raw)
    fundamental_mask = _mask_or_false(fundamental_mask_raw, size)
    essential_mask = _mask_or_false(essential_mask_raw, size)
    inlier_mask = fundamental_mask & essential_mask
    errors = _sampson_errors(fundamental, first[inlier_mask], second[inlier_mask])
    return EstimateResult(fundamental, essential, inlier_mask, errors)


def run(
    cfg: dict,
    output_dir: str | Path,
    first_points: np.ndarray,
    second_points: np.ndarray,
    intrinsics: np.ndarray,
) -> EstimateResult:
    """Estimate and persist epipolar geometry for one frame pair."""

    result = estimate(cfg, first_points, second_points, intrinsics)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    np.save(destination / "F.npy", result.fundamental)
    np.save(destination / "E.npy", result.essential)
    np.save(destination / "K.npy", intrinsics)
    np.save(destination / "inlier_mask.npy", result.inlier_mask)
    np.save(destination / "sampson_errors.npy", result.sampson_errors)
    return result

# <<< END IMPLEMENTATION <<<

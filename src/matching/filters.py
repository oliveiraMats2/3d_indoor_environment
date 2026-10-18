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
#   Lowe's ratio test over knnMatch(k=2) results.
#
# What to implement:
#   - Keep the best neighbor only if best.distance < ratio * second.distance.
#   - Skip entries with fewer than 2 neighbors.
#   - Pairs with fewer than min_matches survivors do not go to RANSAC
#     (mark them success=False, reason="below_min_matches").
#
# Inputs / outputs:
#   In: knnMatch(k=2) result for one pair (list of lists of cv2.DMatch).
#   Out: surviving matches, saved as matches_lowe.npy (K, 3): queryIdx,
#   trainIdx, distance, under results/<run_id>/matching/<detector>/<frameA>__<frameB>/
#
# Defaults (read from the YAML block filters, never hardcoded):
#   lowe_ratio: 0.75           # lower = cleaner and fewer matches; 0.7 to 0.8 is usual
#   min_matches: 10
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
from typing import Optional, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class LoweResult:
    matches: list[cv2.DMatch]
    success: bool
    reason: Optional[str] = None


def run(cfg: dict, neighbors: Sequence[Sequence[cv2.DMatch]]) -> LoweResult:
    """Apply Lowe's ratio test to the two nearest matches per descriptor."""

    ratio = float(cfg["lowe_ratio"])
    min_matches = int(cfg["min_matches"])

    if not 0.0 < ratio < 1.0:
        raise ValueError("lowe_ratio must be between 0 and 1")
    if min_matches < 1:
        raise ValueError("min_matches must be at least 1")

    accepted = [
        pair[0]
        for pair in neighbors
        if len(pair) >= 2 and pair[0].distance < ratio * pair[1].distance
    ]

    if len(accepted) < min_matches:
        return LoweResult(
            matches=accepted,
            success=False,
            reason="below_min_matches",
        )

    return LoweResult(matches=accepted, success=True)


def select_inliers(matches_lowe: np.ndarray, inlier_mask: np.ndarray) -> np.ndarray:
    """Select geometrically consistent matches using a RANSAC mask."""

    matches = np.asarray(matches_lowe)
    mask = np.asarray(inlier_mask).reshape(-1)

    if matches.ndim != 2 or matches.shape[1] != 3:
        raise ValueError("matches_lowe must have shape (n_matches, 3)")
    if len(mask) != len(matches):
        raise ValueError(
            "inlier_mask must contain one value for every Lowe-filtered match"
        )
    if not np.isin(mask, (0, 1, False, True)).all():
        raise ValueError("inlier_mask must contain only boolean or 0/1 values")

    return matches[mask.astype(bool)]

# <<< END IMPLEMENTATION <<<

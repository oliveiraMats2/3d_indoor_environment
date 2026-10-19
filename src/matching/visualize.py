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
#   show matches before and after filtering.
#
# What to implement:
#   - Raw panel: all matches after Lowe, outliers in red (0, 0, 255) and
#     inliers in green (0, 255, 0), drawn on the same canvas with two
#     cv2.drawMatches passes (the second with DRAW_OVER_OUTIMG).
#   - Filtered panel: only the RANSAC inliers.
#   - Cap the drawn lines at max_lines with a seeded random sample.
#   - Consistent pairs form roughly parallel lines; spurious ones cross the
#     figure and must be gone after filtering.
#   - Save a figure ONLY for pairs whose registration was found (passed RANSAC
#     and became a pose), one per such pair. Failed pairs stay only as .npy;
#     their figure can be drawn later from those files.
#
# Inputs / outputs:
#   In: the two frames, keypoints.npy of each and matches_lowe.npy from matching,
#   plus inlier_mask.npy from epipolar/<detector>/<frameA>__<frameB>/.
#   Out: one figure per registered pair under results/<run_id>/figures/matching/.
#
# Defaults (read from the YAML block visualize.matches, never hardcoded):
#   max_lines: 100
#   dpi: 150
#   format: png
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

from pathlib import Path

import cv2
import numpy as np

from src.matching.filters import select_inliers


def _keypoints(rows: np.ndarray) -> list[cv2.KeyPoint]:
    return [
        cv2.KeyPoint(float(x), float(y), float(size), float(angle), float(response), int(octave))
        for x, y, size, angle, response, octave in rows
    ]


def _matches(rows: np.ndarray) -> list[cv2.DMatch]:
    return [
        cv2.DMatch(int(query), int(train), float(distance))
        for query, train, distance in rows
    ]


def _sample_indices(size: int, limit: int, rng: np.random.Generator) -> np.ndarray:
    if limit < 1:
        raise ValueError("max_lines must be at least 1")
    if size <= limit:
        return np.arange(size)
    return np.sort(rng.choice(size, size=limit, replace=False))


def _add_title(image: np.ndarray, title: str) -> np.ndarray:
    titled = image.copy()
    cv2.rectangle(titled, (0, 0), (titled.shape[1], 48), (0, 0, 0), -1)
    cv2.putText(
        titled,
        title,
        (16, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return titled


def draw_pair(
    first_image: np.ndarray,
    second_image: np.ndarray,
    first_keypoints: np.ndarray,
    second_keypoints: np.ndarray,
    matches_lowe: np.ndarray,
    inlier_mask: np.ndarray,
    max_lines: int,
    seed: int,
) -> np.ndarray:
    """Create before/after RANSAC panels for one registered image pair."""

    select_inliers(matches_lowe, inlier_mask)
    mask = np.asarray(inlier_mask).reshape(-1).astype(bool)
    matches = np.asarray(matches_lowe)
    rng = np.random.default_rng(seed)
    sampled = _sample_indices(len(matches), max_lines, rng)
    sampled_matches = matches[sampled]
    sampled_mask = mask[sampled]

    keypoints_first = _keypoints(first_keypoints)
    keypoints_second = _keypoints(second_keypoints)
    outliers = _matches(sampled_matches[~sampled_mask])
    inliers = _matches(sampled_matches[sampled_mask])
    flags = cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS

    raw_panel = cv2.drawMatches(
        first_image,
        keypoints_first,
        second_image,
        keypoints_second,
        outliers,
        None,
        matchColor=(0, 0, 255),
        flags=flags,
    )
    raw_panel = cv2.drawMatches(
        first_image,
        keypoints_first,
        second_image,
        keypoints_second,
        inliers,
        raw_panel,
        matchColor=(0, 255, 0),
        flags=flags | cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG,
    )

    all_inliers = matches[mask]
    filtered_indices = _sample_indices(len(all_inliers), max_lines, rng)
    filtered_matches = _matches(all_inliers[filtered_indices])
    filtered_panel = cv2.drawMatches(
        first_image,
        keypoints_first,
        second_image,
        keypoints_second,
        filtered_matches,
        None,
        matchColor=(0, 255, 0),
        flags=flags,
    )

    raw_title = f"Depois de Lowe: verde=inlier, vermelho=outlier ({len(matches)} matches)"
    filtered_title = f"Depois de RANSAC: {int(mask.sum())} inliers"
    return np.vstack(
        (
            _add_title(raw_panel, raw_title),
            _add_title(filtered_panel, filtered_title),
        )
    )


def _load_image(frames_dir: Path, frame: str) -> np.ndarray:
    for extension in (".png", ".jpg", ".jpeg"):
        image = cv2.imread(str(frames_dir / f"{frame}{extension}"))
        if image is not None:
            return image
    raise FileNotFoundError(f"frame image not found: {frame}")


def run(cfg: dict, run_dir: str, frames_dir: str, seed: int) -> list[Path]:
    """Save one matching figure for every pair with a valid RANSAC pose."""

    root = Path(run_dir)
    matching_root = root / "matching"
    outputs = []
    pair_number = 0
    for detector_dir in sorted(path for path in matching_root.iterdir() if path.is_dir()):
        for pair_dir in sorted(path for path in detector_dir.iterdir() if path.is_dir()):
            geometry_dir = root / "epipolar" / detector_dir.name / pair_dir.name
            mask_path = geometry_dir / "inlier_mask.npy"
            if not mask_path.exists():
                continue

            mask = np.load(mask_path).reshape(-1).astype(bool)
            if int(mask.sum()) < 8:
                continue

            first, second = pair_dir.name.split("__", maxsplit=1)
            detection = root / "detection" / detector_dir.name
            figure = draw_pair(
                _load_image(Path(frames_dir), first),
                _load_image(Path(frames_dir), second),
                np.load(detection / first / "keypoints.npy"),
                np.load(detection / second / "keypoints.npy"),
                np.load(pair_dir / "matches_lowe.npy"),
                np.load(mask_path),
                cfg["max_lines"],
                seed + pair_number,
            )
            extension = cfg["format"]
            output = root / "figures" / "matching" / detector_dir.name / f"{pair_dir.name}.{extension}"
            output.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output), figure):
                raise OSError(f"could not save matching figure: {output}")
            outputs.append(output)
            pair_number += 1

    return outputs

# <<< END IMPLEMENTATION <<<

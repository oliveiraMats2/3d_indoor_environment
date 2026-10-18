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
#   build the descriptor matcher described in the YAML (FLANN or Brute Force).
#
# What to implement:
#   - FLANN: KDTREE index for float descriptors, LSH index for binary ones
#     (KDTREE does not work with Hamming distance). Search param: checks.
#   - Brute Force: NORM_L2 for float, NORM_HAMMING for binary, crossCheck=False.
#   - Return the matcher; the caller runs knnMatch(query, train, k=2) and times it.
#
# Inputs / outputs:
#   In: descriptors.npy (N, D) of one detector; its dtype picks the branch.
#   Out: a cv2 DescriptorMatcher (no files written by this file).
#
# Defaults (read from the YAML block matching, never hardcoded):
#   name: brute_force          # exact; FLANN is approximate and mixes index error
#                              # with detector quality
#   strategy: sequential       # consecutive frames; all_pairs is the ablation
#   flann.checks: 50
#   flann.float_index:  {algorithm: 1, trees: 5}                  # KDTREE
#   flann.binary_index: {algorithm: 6, table_number: 6, key_size: 12,
#                        multi_probe_level: 1}                    # LSH
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

from typing import Literal

import cv2
import numpy as np


DescriptorKind = Literal["binary", "float"]


def descriptor_kind(descriptors: np.ndarray) -> DescriptorKind:
    """Classify an OpenCV descriptor matrix from its NumPy dtype."""

    if not isinstance(descriptors, np.ndarray):
        raise TypeError("descriptors must be a NumPy array")
    if descriptors.ndim != 2:
        raise ValueError("descriptors must have shape (n_keypoints, descriptor_size)")

    if descriptors.dtype == np.uint8:
        return "binary"
    if descriptors.dtype == np.float32:
        return "float"

    raise TypeError(
        f"unsupported descriptor dtype {descriptors.dtype}; "
        "expected uint8 (ORB/AKAZE) or float32 (SIFT)"
    )


def _brute_force(kind: DescriptorKind) -> cv2.DescriptorMatcher:
    norm = cv2.NORM_HAMMING if kind == "binary" else cv2.NORM_L2
    return cv2.BFMatcher(normType=norm, crossCheck=False)


def _flann(cfg: dict, kind: DescriptorKind) -> cv2.DescriptorMatcher:
    flann_cfg = cfg["flann"]
    index_key = "binary_index" if kind == "binary" else "float_index"
    index_params = dict(flann_cfg[index_key])
    search_params = {"checks": flann_cfg["checks"]}
    return cv2.FlannBasedMatcher(index_params, search_params)


def run(cfg: dict, descriptors: np.ndarray) -> cv2.DescriptorMatcher:
    """Build the configured matcher for one descriptor representation."""

    if cfg.get("cross_check", False):
        raise ValueError("cross_check must be false when Lowe's knnMatch(k=2) is used")

    kind = descriptor_kind(descriptors)
    factories = {
        "brute_force": _brute_force,
        "flann": lambda value: _flann(cfg, value),
    }

    try:
        factory = factories[cfg["name"]]
    except KeyError as exc:
        supported = ", ".join(factories)
        raise ValueError(f"unsupported matcher {cfg.get('name')!r}; choose {supported}") from exc

    return factory(kind)

# <<< END IMPLEMENTATION <<<

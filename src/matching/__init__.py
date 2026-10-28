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

"""Matching package: Stage 3, part 1 (descriptor matching, Lowe filter, metrics, figures)."""

from __future__ import annotations

import itertools
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.matching import filters, matcher


def _frame_directories(detector_dir: Path) -> list[Path]:
    return sorted(path for path in detector_dir.iterdir() if path.is_dir())


def _pair_indices(n_frames: int, strategy: str) -> list[tuple[int, int]]:
    if strategy == "sequential":
        return [(index, index + 1) for index in range(n_frames - 1)]
    if strategy == "all_pairs":
        return list(itertools.combinations(range(n_frames), 2))
    raise ValueError(f"unsupported matching strategy: {strategy!r}")


def _as_array(matches: list[cv2.DMatch]) -> np.ndarray:
    rows = [
        (match.queryIdx, match.trainIdx, match.distance)
        for match in matches
    ]
    return np.asarray(rows, dtype=np.float32).reshape(-1, 3)


def _knn_with_timing(
    descriptor_matcher: cv2.DescriptorMatcher,
    query: np.ndarray,
    train: np.ndarray,
    reps: int,
    warmup: int,
) -> tuple[list[list[cv2.DMatch]], np.ndarray]:
    if len(query) == 0 or len(train) < 2:
        return [], np.empty(0, dtype=np.float64)

    neighbors: list[list[cv2.DMatch]] = []
    times = []
    for repetition in range(warmup + reps):
        started = time.perf_counter()
        neighbors = descriptor_matcher.knnMatch(query, train, k=2)
        elapsed_ms = (time.perf_counter() - started) * 1_000
        if repetition >= warmup:
            times.append(elapsed_ms)

    return neighbors, np.asarray(times, dtype=np.float64)


def _save_pair(
    pair_dir: Path,
    raw_matches: list[cv2.DMatch],
    lowe_matches: list[cv2.DMatch],
    times_ms: np.ndarray,
) -> None:
    pair_dir.mkdir(parents=True, exist_ok=True)
    np.save(pair_dir / "matches_raw.npy", _as_array(raw_matches))
    np.save(pair_dir / "matches_lowe.npy", _as_array(lowe_matches))
    np.save(pair_dir / "times_ms.npy", times_ms)


def run(cfg: dict, filters_cfg: dict, run_dir: str) -> None:
    """Match descriptors for every configured detector and frame pair."""

    detection_root = Path(run_dir) / "detection"
    if not detection_root.is_dir():
        raise FileNotFoundError(f"detection results not found: {detection_root}")

    detector_dirs = sorted(path for path in detection_root.iterdir() if path.is_dir())
    for detector_dir in detector_dirs:
        frames = _frame_directories(detector_dir)
        pairs = _pair_indices(len(frames), cfg["strategy"])
        progress = tqdm(pairs, desc=f"{detector_dir.name} matching", unit="pair")

        for first_index, second_index in progress:
            first = frames[first_index]
            second = frames[second_index]
            query = np.load(first / "descriptors.npy")
            train = np.load(second / "descriptors.npy")
            descriptor_matcher = matcher.run(cfg, query)
            neighbors, times_ms = _knn_with_timing(
                descriptor_matcher,
                query,
                train,
                cfg["reps"],
                cfg["warmup"],
            )

            raw_matches = [pair[0] for pair in neighbors if pair]
            lowe_result = filters.run(filters_cfg, neighbors)
            pair_name = f"{first.name}__{second.name}"
            pair_dir = Path(run_dir) / "matching" / detector_dir.name / pair_name
            _save_pair(pair_dir, raw_matches, lowe_result.matches, times_ms)
    __import__("src.matching.cross_check", fromlist=["run"]).run(cfg, filters_cfg, run_dir)  # filters.cross_check

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

"""Stage 3 part 2: epipolar geometry (F, E, pose, epipolar lines, triangulation)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.epipolar import estimate


def _frame_path(frames_dir: Path, frame: str) -> Path:
    for extension in (".png", ".jpg", ".jpeg"):
        candidate = frames_dir / f"{frame}{extension}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"frame image not found: {frame}")


def _intrinsics_matrix(cfg: dict, width: int, height: int) -> np.ndarray:
    intrinsics = cfg["intrinsics"]
    default_focal = max(width, height)
    fx = intrinsics["fx"] or default_focal
    fy = intrinsics["fy"] or default_focal
    cx = intrinsics["cx"] if intrinsics["cx"] is not None else width / 2
    cy = intrinsics["cy"] if intrinsics["cy"] is not None else height / 2
    return np.array(((fx, 0, cx), (0, fy, cy), (0, 0, 1)), dtype=np.float64)


def _matched_points(
    first_keypoints: np.ndarray,
    second_keypoints: np.ndarray,
    matches: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if matches.ndim != 2 or matches.shape[1] != 3:
        raise ValueError("matches_lowe must have shape (n_matches, 3)")
    if len(matches) == 0:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty.copy()

    first_indices = matches[:, 0].astype(int)
    second_indices = matches[:, 1].astype(int)
    if (
        (first_indices < 0).any()
        or (second_indices < 0).any()
        or (first_indices >= len(first_keypoints)).any()
        or (second_indices >= len(second_keypoints)).any()
    ):
        raise IndexError("a match references a keypoint outside its frame")
    return (
        first_keypoints[first_indices, :2].astype(np.float32),
        second_keypoints[second_indices, :2].astype(np.float32),
    )


def run(cfg: dict, run_dir: str, frames_dir: str) -> list[Path]:
    """Estimate epipolar geometry for every saved detector/frame pair."""

    root = Path(run_dir)
    matching_root = root / "matching"
    if not matching_root.is_dir():
        raise FileNotFoundError(f"matching results not found: {matching_root}")

    outputs = []
    detector_dirs = sorted(path for path in matching_root.iterdir() if path.is_dir())
    for detector_dir in detector_dirs:
        pair_dirs = sorted(path for path in detector_dir.iterdir() if path.is_dir())
        progress = tqdm(pair_dirs, desc=f"{detector_dir.name} geometry", unit="pair")
        for pair_dir in progress:
            first, second = pair_dir.name.split("__", maxsplit=1)
            image = cv2.imread(str(_frame_path(Path(frames_dir), first)))
            if image is None:
                raise ValueError(f"could not read frame: {first}")
            height, width = image.shape[:2]
            intrinsics = _intrinsics_matrix(cfg, width, height)

            detection = root / "detection" / detector_dir.name
            first_keypoints = np.load(detection / first / "keypoints.npy")
            second_keypoints = np.load(detection / second / "keypoints.npy")
            matches = np.load(pair_dir / "matches_lowe.npy")
            first_points, second_points = _matched_points(
                first_keypoints,
                second_keypoints,
                matches,
            )

            output_dir = root / "epipolar" / detector_dir.name / pair_dir.name
            estimate.run(cfg, output_dir, first_points, second_points, intrinsics)
            outputs.append(output_dir)

    if outputs:
        np.save(root / "K.npy", np.load(outputs[0] / "K.npy"))
    return outputs

"""Per-frame metrics: Laplacian sharpness, brightness, SIFT keypoints, density and grid occupancy."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from frame_analyzer.config import AnalyzerConfig

log = logging.getLogger(__name__)


class SiftUnavailableError(RuntimeError):
    """Raised when this OpenCV build has no SIFT (no fallback detector is used)."""


def laplacian_variance(gray: np.ndarray) -> float:
    """Variance of the Laplacian (CV_64F): higher means more high-frequency detail."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness_stats(gray: np.ndarray, dark_threshold: int, saturated_threshold: int) -> dict:
    """Mean/std of gray levels and the fractions of very dark and saturated pixels."""
    g = gray.astype(np.float64)
    n = max(g.size, 1)
    return {
        "brightness_mean": float(g.mean()),
        "brightness_std": float(g.std()),
        "dark_frac": float(np.count_nonzero(gray <= dark_threshold) / n),
        "saturated_frac": float(np.count_nonzero(gray >= saturated_threshold) / n),
    }


def create_sift(cfg: AnalyzerConfig) -> cv2.SIFT:
    """SIFT detector configured from cfg; raises SiftUnavailableError if this OpenCV lacks SIFT."""
    factory = getattr(cv2, "SIFT_create", None)
    if factory is None:
        raise SiftUnavailableError(
            f"cv2.SIFT_create is not available in OpenCV {cv2.__version__}. Install opencv-contrib-python>=4.4 "
            "(or opencv-python>=4.4, where SIFT is in the main module). No fallback detector is used."
        )
    try:
        return factory(nfeatures=cfg.sift_nfeatures, contrastThreshold=cfg.sift_contrast_threshold,
                       edgeThreshold=cfg.sift_edge_threshold)
    except cv2.error as exc:
        raise SiftUnavailableError(f"cv2.SIFT_create failed: {exc}. opencv-contrib-python>=4.4 is required.") from exc


def detect_sift(sift: cv2.SIFT, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Keypoint coordinates (K,2) float32 and descriptors (K,128) float32 (None when K == 0)."""
    keypoints, desc = sift.detectAndCompute(gray, None)
    if not keypoints:
        return np.zeros((0, 2), np.float32), None
    pts = np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 2)
    return pts, (desc.astype(np.float32) if desc is not None else None)


def keypoint_density(n_keypoints: int, width: int, height: int) -> float:
    """Keypoints per megapixel of the analysed image."""
    area = width * height
    if area <= 0:
        raise ValueError(f"image area must be positive, got {width}x{height}")
    return float(n_keypoints / (area / 1e6))


def grid_counts(points: np.ndarray, width: int, height: int, rows: int, cols: int) -> np.ndarray:
    """Keypoints per cell of a rows x cols grid; points on the right/bottom border go to the last cell."""
    counts = np.zeros((rows, cols), np.int32)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if pts.size == 0 or width <= 0 or height <= 0:
        return counts
    c = np.clip(np.floor(pts[:, 0] * cols / width).astype(np.int64), 0, cols - 1)
    r = np.clip(np.floor(pts[:, 1] * rows / height).astype(np.int64), 0, rows - 1)
    np.add.at(counts, (r, c), 1)
    return counts


def grid_occupancy(counts: np.ndarray) -> tuple[int, int, float]:
    """(occupied cells, total cells, occupied/total)."""
    total = int(counts.size)
    occupied = int(np.count_nonzero(counts))
    return occupied, total, (occupied / total if total else float("nan"))

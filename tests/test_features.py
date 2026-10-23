"""Tests of per-frame metrics: sharpness, brightness, SIFT, density and grid occupancy."""
import cv2
import numpy as np
import pytest

from frame_analyzer.config import AnalyzerConfig
from frame_analyzer.features import (brightness_stats, create_sift, detect_sift, grid_counts, grid_occupancy,
                                     keypoint_density, laplacian_variance)


def test_laplacian_constant_is_zero() -> None:
    assert laplacian_variance(np.full((50, 60), 128, np.uint8)) == 0.0


def test_blur_lowers_laplacian(textured: np.ndarray) -> None:
    blurred = cv2.GaussianBlur(textured, (0, 0), 3)
    assert laplacian_variance(blurred) < laplacian_variance(textured) * 0.5


def test_brightness_stats() -> None:
    img = np.zeros((10, 10), np.uint8)
    img[:, 5:] = 255
    s = brightness_stats(img, 10, 245)
    assert s["dark_frac"] == pytest.approx(0.5) and s["saturated_frac"] == pytest.approx(0.5)
    assert s["brightness_mean"] == pytest.approx(127.5)


def test_density_formula() -> None:
    assert keypoint_density(1000, 1000, 1000) == pytest.approx(1000.0)
    assert keypoint_density(300, 640, 480) == pytest.approx(300 / 0.3072)
    with pytest.raises(ValueError):
        keypoint_density(10, 0, 100)


def test_grid_counts_and_edges() -> None:
    pts = np.array([[0, 0], [99.9, 0], [100, 100], [50, 50], [0, 100], [100, 0]], np.float32)
    counts = grid_counts(pts, 100, 100, 2, 2)
    assert counts.dtype == np.int32 and counts.shape == (2, 2)
    assert counts.sum() == len(pts)
    assert counts[0, 0] == 1  # (0,0)
    assert counts[0, 1] == 2  # (99.9,0), (100,0) on the right border
    assert counts[1, 1] == 2  # (100,100) corner, (50,50) on the cell boundary
    assert counts[1, 0] == 1  # (0,100) on the bottom border
    assert grid_occupancy(counts) == (4, 4, 1.0)
    empty = grid_counts(np.zeros((0, 2), np.float32), 100, 100, 3, 4)
    assert empty.shape == (3, 4) and grid_occupancy(empty) == (0, 12, 0.0)


def test_detect_sift(textured: np.ndarray, cfg: AnalyzerConfig) -> None:
    sift = create_sift(cfg)
    pts, desc = detect_sift(sift, textured)
    assert pts.shape[1] == 2 and pts.dtype == np.float32 and len(pts) > 100
    assert desc is not None and desc.shape == (len(pts), 128) and desc.dtype == np.float32
    pts0, desc0 = detect_sift(sift, np.full((100, 100), 128, np.uint8))
    assert pts0.shape == (0, 2) and desc0 is None

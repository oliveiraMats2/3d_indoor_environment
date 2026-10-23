"""Shared fixtures: synthetic textured images written under tmp_path (no real photos)."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frame_analyzer.config import AnalyzerConfig  # noqa: E402


def make_textured(height: int = 480, width: int = 640, seed: int = 0) -> np.ndarray:
    """Random blobs, rectangles and lines: plenty of distinctive SIFT features."""
    rng = np.random.default_rng(seed)
    img = np.full((height, width), 127, np.uint8)
    for _ in range(250):
        x, y = int(rng.integers(0, width)), int(rng.integers(0, height))
        color = int(rng.integers(0, 256))
        kind = rng.integers(0, 3)
        if kind == 0:
            cv2.circle(img, (x, y), int(rng.integers(3, 25)), color, -1)
        elif kind == 1:
            cv2.rectangle(img, (x, y), (x + int(rng.integers(5, 40)), y + int(rng.integers(5, 40))), color, -1)
        else:
            cv2.line(img, (x, y), (int(rng.integers(0, width)), int(rng.integers(0, height))), color, 2)
    noise = rng.normal(0, 4, img.shape)
    return np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)


@pytest.fixture
def textured() -> np.ndarray:
    return make_textured()


@pytest.fixture
def cfg(tmp_path: Path) -> AnalyzerConfig:
    return AnalyzerConfig(input_dir=tmp_path / "in", output_dir=tmp_path / "out")

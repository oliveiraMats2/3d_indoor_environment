"""Tests of frame listing, natural sort and image reading."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from frame_analyzer.config import DEFAULT_EXTENSIONS
from frame_analyzer.io import ImageReadError, list_frames, natural_sort_key, read_gray


def test_natural_sort() -> None:
    names = ["frame_10.jpg", "frame_2.jpg", "Frame_1.jpg", "frame_100.jpg", "frame_20.jpg"]
    assert sorted(names, key=natural_sort_key) == ["Frame_1.jpg", "frame_2.jpg", "frame_10.jpg", "frame_20.jpg",
                                                   "frame_100.jpg"]


def test_list_frames_filters_extensions(tmp_path: Path) -> None:
    img = np.zeros((10, 10), np.uint8)
    for name in ["b_10.PNG", "b_2.jpg", "c.TIFF", "d.bmp", "e.jpeg"]:
        cv2.imwrite(str(tmp_path / name.lower()), img)
        (tmp_path / name.lower()).rename(tmp_path / name)
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / "video.mp4").write_bytes(b"\x00")
    (tmp_path / "sub.jpg").mkdir()
    names = [p.name for p in list_frames(tmp_path, DEFAULT_EXTENSIONS)]
    assert names == ["b_2.jpg", "b_10.PNG", "c.TIFF", "d.bmp", "e.jpeg"]


def test_list_frames_empty_and_missing(tmp_path: Path) -> None:
    assert list_frames(tmp_path, DEFAULT_EXTENSIONS) == []
    with pytest.raises(FileNotFoundError):
        list_frames(tmp_path / "missing", DEFAULT_EXTENSIONS)
    f = tmp_path / "file.jpg"
    f.write_bytes(b"x")
    with pytest.raises(NotADirectoryError):
        list_frames(f, DEFAULT_EXTENSIONS)


def test_read_gray_resizes_and_reports_original(tmp_path: Path) -> None:
    path = tmp_path / "a.png"
    cv2.imwrite(str(path), np.zeros((300, 800, 3), np.uint8))
    gray, w, h = read_gray(path, 400)
    assert (w, h) == (800, 300)
    assert gray.dtype == np.uint8 and gray.ndim == 2 and max(gray.shape) == 400
    gray_full, _, _ = read_gray(path, 0)
    assert gray_full.shape == (300, 800)


def test_read_gray_corrupt_and_truncated(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"this is not an image at all" * 10)
    with pytest.raises(ImageReadError):
        read_gray(bad, 0)
    empty = tmp_path / "empty.png"
    empty.write_bytes(b"")
    with pytest.raises(ImageReadError):
        read_gray(empty, 0)
    ok, buf = cv2.imencode(".jpg", np.random.default_rng(0).integers(0, 255, (200, 200), dtype=np.uint8))
    assert ok
    trunc = tmp_path / "trunc.jpg"
    trunc.write_bytes(buf.tobytes()[: len(buf) // 2])
    with pytest.raises(ImageReadError, match="truncated"):
        read_gray(trunc, 0)

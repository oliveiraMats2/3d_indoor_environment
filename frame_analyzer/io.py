"""Frame listing (natural sort) and streaming grayscale reading with corruption checks."""
from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

_DIGITS = re.compile(r"(\d+)")
_TAIL_BYTES = 64


class ImageReadError(Exception):
    """Raised when a frame cannot be decoded or looks truncated."""


def natural_sort_key(name: str) -> list:
    """Key that orders 'frame_2' before 'frame_10' (case-insensitive on the text parts)."""
    return [(0, int(tok), "") if tok.isdigit() else (1, 0, tok.lower()) for tok in _DIGITS.split(name) if tok != ""]


def list_frames(input_dir: Path, extensions: Iterable[str]) -> list[Path]:
    """Files of input_dir (non-recursive) whose extension is in `extensions`, natural-sorted."""
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"input path is not a directory: {input_dir}")
    exts = {"." + e.lower().lstrip(".") for e in extensions}
    files = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(files, key=lambda p: natural_sort_key(p.name))


def _check_truncated(data: np.ndarray, suffix: str) -> None:
    tail = data[-_TAIL_BYTES:].tobytes().rstrip(b"\x00")
    if suffix in (".jpg", ".jpeg") and b"\xff\xd9" not in tail:
        raise ImageReadError("truncated JPEG (missing end-of-image marker)")
    if suffix == ".png" and b"IEND" not in tail:
        raise ImageReadError("truncated PNG (missing IEND chunk)")


def _resize_max_side(gray: np.ndarray, max_side: int) -> np.ndarray:
    h, w = gray.shape[:2]
    longest = max(h, w)
    if max_side <= 0 or longest <= max_side:
        return gray
    scale = max_side / longest
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(gray, size, interpolation=cv2.INTER_AREA)


def read_gray(path: Path, max_side: int) -> tuple[np.ndarray, int, int]:
    """Decode one frame as uint8 gray, resized so its longest side <= max_side (0 = keep). Returns gray, orig_w, orig_h."""
    path = Path(path)
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError as exc:
        raise ImageReadError(f"cannot read file: {exc}") from exc
    if data.size == 0:
        raise ImageReadError("empty file")
    _check_truncated(data, path.suffix.lower())
    try:
        gray = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    except cv2.error as exc:
        raise ImageReadError(f"OpenCV could not decode the image: {exc}") from exc
    if gray is None or gray.size == 0:
        raise ImageReadError("OpenCV could not decode the image (imdecode returned None)")
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    orig_h, orig_w = gray.shape[:2]
    return _resize_max_side(gray, max_side), int(orig_w), int(orig_h)

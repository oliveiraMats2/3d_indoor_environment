"""Transparent per-frame classification rules (good / review / poor), no combined score."""
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from frame_analyzer.config import AnalyzerConfig

log = logging.getLogger(__name__)

LEVELS = ("good", "review", "poor")

CLASSIFICATION_RULES: list[dict] = [
    {"flag": "unreadable", "level": "poor", "description": "file could not be decoded or is truncated"},
    {"flag": "blur_poor", "level": "poor",
     "description": "sharpness_rel = laplacian_var / median < sharpness_poor_rel"},
    {"flag": "blur_abs_poor", "level": "poor",
     "description": "laplacian_var < sharpness_abs_min (only when a calibrated absolute floor is set)"},
    {"flag": "keypoints_poor", "level": "poor", "description": "sift_keypoints < keypoints_poor_min"},
    {"flag": "blur_review", "level": "review",
     "description": "sharpness_poor_rel <= sharpness_rel < sharpness_review_rel"},
    {"flag": "keypoints_review", "level": "review",
     "description": "keypoints_poor_min <= sift_keypoints < keypoints_review_min"},
    {"flag": "low_occupancy", "level": "review", "description": "occupancy_ratio < occupancy_review_min"},
    {"flag": "isolated", "level": "review",
     "description": "degree < degree_review_min and degree == 0 (no accepted edge in the connectivity graph)"},
    {"flag": "low_degree", "level": "review", "description": "0 < degree < degree_review_min"},
    {"flag": "dark", "level": "review", "description": "dark_frac > dark_frac_review"},
    {"flag": "saturated", "level": "review", "description": "saturated_frac > saturated_frac_review"},
]
_LEVEL_OF = {r["flag"]: r["level"] for r in CLASSIFICATION_RULES}


def _num(value: object) -> float:
    try:
        v = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return v


def _below(value: float, threshold: float) -> bool:
    return not math.isnan(value) and value < threshold


def _above(value: float, threshold: float) -> bool:
    return not math.isnan(value) and value > threshold


def frame_flags(row: pd.Series | dict, cfg: AnalyzerConfig) -> list[str]:
    """Flags raised by one frame, in CLASSIFICATION_RULES order."""
    if not bool(row.get("readable", False)):
        return ["unreadable"]
    rel, lap = _num(row.get("sharpness_rel")), _num(row.get("laplacian_var"))
    kp, occ, deg = _num(row.get("sift_keypoints")), _num(row.get("occupancy_ratio")), _num(row.get("degree"))
    flags: list[str] = []
    if _below(rel, cfg.sharpness_poor_rel):
        flags.append("blur_poor")
    if cfg.sharpness_abs_min is not None and _below(lap, cfg.sharpness_abs_min):
        flags.append("blur_abs_poor")
    if _below(kp, cfg.keypoints_poor_min):
        flags.append("keypoints_poor")
    if "blur_poor" not in flags and _below(rel, cfg.sharpness_review_rel):
        flags.append("blur_review")
    if "keypoints_poor" not in flags and _below(kp, cfg.keypoints_review_min):
        flags.append("keypoints_review")
    if _below(occ, cfg.occupancy_review_min):
        flags.append("low_occupancy")
    if _below(deg, cfg.degree_review_min):
        flags.append("isolated" if deg == 0 else "low_degree")
    if _above(_num(row.get("dark_frac")), cfg.dark_frac_review):
        flags.append("dark")
    if _above(_num(row.get("saturated_frac")), cfg.saturated_frac_review):
        flags.append("saturated")
    return flags


def quality_from_flags(flags: list[str]) -> str:
    """Worst level among the flags (good when there are none)."""
    worst = 0
    for f in flags:
        worst = max(worst, LEVELS.index(_LEVEL_OF.get(f, "review")))
    return LEVELS[worst]


def classify_frames(frames_df: pd.DataFrame, cfg: AnalyzerConfig) -> pd.DataFrame:
    """Return a copy of frames_df with the `flags` (';'-joined) and `quality` columns filled."""
    out = frames_df.copy()
    all_flags = [frame_flags(row, cfg) for _, row in out.iterrows()]
    out["flags"] = [";".join(f) for f in all_flags]
    out["quality"] = [quality_from_flags(f) for f in all_flags]
    log.info("classification: %s", out["quality"].value_counts().to_dict() if len(out) else {})
    return out

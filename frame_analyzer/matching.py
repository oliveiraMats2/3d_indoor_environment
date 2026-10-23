"""Pairwise SIFT matching (Lowe ratio) and geometric verification with a RANSAC fundamental matrix."""
from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from frame_analyzer.config import AnalyzerConfig

log = logging.getLogger(__name__)

MIN_F_INLIERS = 8
MIN_H_POINTS = 4
F_METHOD = getattr(cv2, "USAC_MAGSAC", cv2.FM_RANSAC)
F_METHOD_NAME = "USAC_MAGSAC" if hasattr(cv2, "USAC_MAGSAC") else "FM_RANSAC"

_MATCHER = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)


def inlier_ratio(inliers: int, lowe_matches: int) -> float:
    """inliers / lowe_matches, NaN when there are no Lowe matches."""
    if not lowe_matches:
        return float("nan")
    return float(inliers / lowe_matches)


def lowe_filter(knn: Sequence[Sequence[cv2.DMatch]], ratio: float) -> list[cv2.DMatch]:
    """Keep best matches whose distance < ratio * second-best distance (pairs with < 2 neighbours are dropped)."""
    return [m[0] for m in knn if len(m) >= 2 and m[0].distance < ratio * m[1].distance]


def _empty_result(status: str) -> dict:
    return {
        "raw_matches": 0, "lowe_matches": 0, "inliers": 0, "inlier_ratio": float("nan"), "model": "fundamental",
        "verified": False, "status": status, "homography_inliers": -1, "homography_ratio": float("nan"),
        "median_disp_px": float("nan"),
    }


def _fundamental(pa: np.ndarray, pb: np.ndarray, cfg: AnalyzerConfig) -> np.ndarray | None:
    """Inlier mask (bool) of the F estimation, or None when it failed."""
    cv2.setRNGSeed(cfg.seed)
    try:
        F, mask = cv2.findFundamentalMat(pa, pb, F_METHOD, cfg.ransac_threshold_px, cfg.ransac_confidence,
                                         cfg.ransac_max_iters)
    except cv2.error as exc:
        log.debug("findFundamentalMat failed: %s", exc)
        return None
    if F is None or mask is None or F.shape[0] < 3:
        return None
    return mask.ravel().astype(bool)


def _homography_inliers(pa: np.ndarray, pb: np.ndarray, cfg: AnalyzerConfig) -> int:
    """Number of RANSAC homography inliers, 0 when the estimation fails."""
    cv2.setRNGSeed(cfg.seed)
    try:
        H, mask = cv2.findHomography(pa, pb, cv2.RANSAC, cfg.ransac_threshold_px, maxIters=cfg.ransac_max_iters,
                                     confidence=cfg.ransac_confidence)
    except cv2.error as exc:
        log.debug("findHomography failed: %s", exc)
        return 0
    if H is None or mask is None:
        return 0
    return int(mask.ravel().astype(bool).sum())


def match_pair(pts_a: np.ndarray, desc_a: np.ndarray | None, pts_b: np.ndarray, desc_b: np.ndarray | None,
               cfg: AnalyzerConfig) -> dict:
    """Match frame a against frame b; returns the pair_quality.csv fields raw_matches..median_disp_px."""
    if desc_a is None or desc_b is None or len(desc_a) < 1 or len(desc_b) < 2:
        return _empty_result("no_descriptors")
    try:
        knn = _MATCHER.knnMatch(desc_a, desc_b, k=2)
    except cv2.error as exc:
        log.debug("knnMatch failed: %s", exc)
        return _empty_result("no_descriptors")
    result = _empty_result("too_few_matches")
    result["raw_matches"] = sum(1 for m in knn if len(m) >= 2)
    good = lowe_filter(knn, cfg.lowe_ratio)
    n_lowe = len(good)
    result["lowe_matches"] = n_lowe
    result["inlier_ratio"] = inlier_ratio(0, n_lowe)
    if n_lowe < max(cfg.min_lowe_matches, MIN_F_INLIERS):
        return result
    pa = np.float32([pts_a[m.queryIdx] for m in good]).reshape(-1, 2)
    pb = np.float32([pts_b[m.trainIdx] for m in good]).reshape(-1, 2)
    if cfg.compute_homography and n_lowe >= MIN_H_POINTS:
        h_in = _homography_inliers(pa, pb, cfg)
        result["homography_inliers"] = h_in
        result["homography_ratio"] = inlier_ratio(h_in, n_lowe)
    mask = _fundamental(pa, pb, cfg)
    if mask is None:
        result["status"] = "estimation_failed"
        return result
    n_in = int(mask.sum())
    result["inliers"] = n_in
    result["inlier_ratio"] = inlier_ratio(n_in, n_lowe)
    result["verified"] = n_in >= MIN_F_INLIERS
    result["status"] = "ok" if result["verified"] else "estimation_failed"
    if n_in:
        result["median_disp_px"] = float(np.median(np.linalg.norm(pb[mask] - pa[mask], axis=1)))
    return result


def accept_edge(row_or_dict: Mapping, cfg: AnalyzerConfig) -> bool:
    """Graph edge rule: verified F, inliers >= edge_min_inliers and inlier_ratio >= edge_min_inlier_ratio."""
    ratio = row_or_dict.get("inlier_ratio", float("nan"))
    if ratio is None or (isinstance(ratio, float) and math.isnan(ratio)):
        return False
    return bool(row_or_dict.get("verified", False)) and int(row_or_dict.get("inliers", 0)) >= cfg.edge_min_inliers \
        and float(ratio) >= cfg.edge_min_inlier_ratio


def edge_criteria(cfg: AnalyzerConfig) -> str:
    """Human-readable acceptance rule, used in edges_df.criteria."""
    return f"verified=fundamental; inliers>={cfg.edge_min_inliers}; inlier_ratio>={cfg.edge_min_inlier_ratio:g}"

"""Tests of matching, geometric verification and edge acceptance."""
import math

import cv2
import numpy as np

from frame_analyzer.config import AnalyzerConfig
from frame_analyzer.features import create_sift, detect_sift
from frame_analyzer.matching import accept_edge, inlier_ratio, lowe_filter, match_pair

from conftest import make_textured


def test_inlier_ratio_zero_denominator() -> None:
    assert math.isnan(inlier_ratio(0, 0))
    assert inlier_ratio(30, 60) == 0.5


def test_lowe_filter() -> None:
    knn = [[cv2.DMatch(0, 0, 1.0), cv2.DMatch(0, 1, 10.0)], [cv2.DMatch(1, 0, 9.0), cv2.DMatch(1, 1, 10.0)],
           [cv2.DMatch(2, 0, 1.0)]]
    good = lowe_filter(knn, 0.75)
    assert [m.queryIdx for m in good] == [0]


def _warp(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[12, 8], [w - 5, 15], [w - 10, h - 6], [6, h - 12]])
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h), borderValue=127)


def test_match_pair_warped_copy_verified(textured: np.ndarray, cfg: AnalyzerConfig) -> None:
    sift = create_sift(cfg)
    pa, da = detect_sift(sift, textured)
    pb, db = detect_sift(sift, _warp(textured))
    r = match_pair(pa, da, pb, db, cfg)
    assert r["status"] == "ok" and r["verified"]
    assert r["inliers"] >= 100 and r["inlier_ratio"] > 0.5
    assert r["raw_matches"] >= r["lowe_matches"] >= r["inliers"]
    assert r["homography_inliers"] > 0 and 0 < r["homography_ratio"] <= 1
    assert r["median_disp_px"] > 0
    assert accept_edge(r, cfg)
    assert match_pair(pa, da, pb, db, cfg) == r  # deterministic


def test_match_pair_shifted_copy(textured: np.ndarray, cfg: AnalyzerConfig) -> None:
    shifted = np.roll(textured, 15, axis=1)
    sift = create_sift(cfg)
    pa, da = detect_sift(sift, textured)
    pb, db = detect_sift(sift, shifted)
    r = match_pair(pa, da, pb, db, cfg)
    assert r["verified"] and r["inliers"] >= 100
    assert abs(r["median_disp_px"] - 15) < 1.5


def test_match_pair_unrelated(textured: np.ndarray, cfg: AnalyzerConfig) -> None:
    other = np.random.default_rng(123).integers(0, 256, textured.shape, dtype=np.uint8)
    sift = create_sift(cfg)
    pa, da = detect_sift(sift, textured)
    pb, db = detect_sift(sift, other)
    r = match_pair(pa, da, pb, db, cfg)
    assert not accept_edge(r, cfg)
    assert r["inliers"] < cfg.edge_min_inliers or r["inlier_ratio"] < cfg.edge_min_inlier_ratio
    r2 = match_pair(pa, da, *detect_sift(sift, make_textured(seed=99)), cfg)
    assert not accept_edge(r2, cfg)


def test_match_pair_no_descriptors(cfg: AnalyzerConfig) -> None:
    r = match_pair(np.zeros((0, 2), np.float32), None, np.zeros((0, 2), np.float32), None, cfg)
    assert r["status"] == "no_descriptors" and not r["verified"] and math.isnan(r["inlier_ratio"])
    assert r["homography_inliers"] == -1


def test_accept_edge_thresholds() -> None:
    cfg = AnalyzerConfig(input_dir=".", edge_min_inliers=30, edge_min_inlier_ratio=0.25)
    assert accept_edge({"verified": True, "inliers": 30, "inlier_ratio": 0.25}, cfg)
    assert not accept_edge({"verified": True, "inliers": 29, "inlier_ratio": 0.9}, cfg)
    assert not accept_edge({"verified": True, "inliers": 300, "inlier_ratio": 0.2}, cfg)
    assert not accept_edge({"verified": False, "inliers": 300, "inlier_ratio": 0.9}, cfg)
    assert not accept_edge({"verified": True, "inliers": 300, "inlier_ratio": float("nan")}, cfg)

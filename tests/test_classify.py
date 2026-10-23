"""Tests of the transparent classification rules."""
import pandas as pd

from frame_analyzer.classify import CLASSIFICATION_RULES, classify_frames
from frame_analyzer.config import AnalyzerConfig

GOOD = {"readable": True, "laplacian_var": 500.0, "sharpness_rel": 1.0, "sift_keypoints": 2000,
        "occupancy_ratio": 0.9, "degree": 3, "dark_frac": 0.0, "saturated_frac": 0.0}


def _run(rows: list[dict], **overrides) -> pd.DataFrame:
    cfg = AnalyzerConfig(input_dir=".", **overrides)
    return classify_frames(pd.DataFrame([{**GOOD, **r} for r in rows]), cfg)


def test_rules_exported() -> None:
    flags = {r["flag"] for r in CLASSIFICATION_RULES}
    assert {"unreadable", "blur_poor", "keypoints_poor", "isolated", "low_degree", "low_occupancy"} <= flags
    assert all(r["level"] in {"poor", "review"} and r["description"] for r in CLASSIFICATION_RULES)


def test_classification_levels() -> None:
    df = _run([
        {},
        {"readable": False, "sharpness_rel": float("nan"), "degree": 0},
        {"sharpness_rel": 0.2},
        {"sharpness_rel": 0.5},
        {"sift_keypoints": 50},
        {"sift_keypoints": 200},
        {"occupancy_ratio": 0.3},
        {"degree": 0},
        {"dark_frac": 0.5},
        {"saturated_frac": 0.5},
        {"sharpness_rel": 0.5, "sift_keypoints": 50, "degree": 0},
    ])
    assert list(df["quality"]) == ["good", "poor", "poor", "review", "poor", "review", "review", "review", "review",
                                   "review", "poor"]
    assert df.loc[0, "flags"] == ""
    assert df.loc[1, "flags"] == "unreadable"
    assert df.loc[2, "flags"] == "blur_poor"
    assert df.loc[3, "flags"] == "blur_review"
    assert df.loc[4, "flags"] == "keypoints_poor"
    assert df.loc[5, "flags"] == "keypoints_review"
    assert df.loc[6, "flags"] == "low_occupancy"
    assert df.loc[7, "flags"] == "isolated"
    assert df.loc[8, "flags"] == "dark"
    assert df.loc[9, "flags"] == "saturated"
    assert df.loc[10, "flags"] == "keypoints_poor;blur_review;isolated"


def test_low_degree_and_absolute_floor() -> None:
    df = _run([{"degree": 1}, {"laplacian_var": 10.0}], degree_review_min=2, sharpness_abs_min=50.0)
    assert df.loc[0, "flags"] == "low_degree" and df.loc[0, "quality"] == "review"
    assert df.loc[1, "flags"] == "blur_abs_poor" and df.loc[1, "quality"] == "poor"
    df2 = _run([{"laplacian_var": 10.0}])
    assert df2.loc[0, "quality"] == "good"

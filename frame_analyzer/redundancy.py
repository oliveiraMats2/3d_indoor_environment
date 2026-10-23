"""Temporal redundancy candidates: consecutive frames that are nearly identical."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from frame_analyzer.config import AnalyzerConfig

log = logging.getLogger(__name__)

REDUNDANT_COLUMNS = ["i", "j", "filename_i", "filename_j", "inliers", "inlier_ratio", "median_disp_px", "reason"]


def find_redundant_pairs(pairs_df: pd.DataFrame, frames_df: pd.DataFrame, cfg: AnalyzerConfig) -> pd.DataFrame:
    """offset==1 verified pairs with inlier_ratio >= thr and median_disp_px <= thr (candidates, not removals)."""
    if pairs_df is None or pairs_df.empty:
        return pd.DataFrame(columns=REDUNDANT_COLUMNS)
    p = pairs_df
    mask = (p["offset"] == 1) & p["verified"].astype(bool) \
        & (p["inlier_ratio"] >= cfg.redundancy_min_inlier_ratio) \
        & (p["median_disp_px"] <= cfg.redundancy_max_median_disp_px)
    sel = p.loc[mask, ["i", "j", "filename_i", "filename_j", "inliers", "inlier_ratio", "median_disp_px"]].copy()
    sel["reason"] = [
        f"inlier_ratio={r:.3f}>={cfg.redundancy_min_inlier_ratio:g}; "
        f"median_disp_px={d:.2f}<={cfg.redundancy_max_median_disp_px:g}"
        for r, d in zip(sel["inlier_ratio"], sel["median_disp_px"])
    ]
    log.info("temporal redundancy: %d candidate pairs", len(sel))
    return sel.reset_index(drop=True)[REDUNDANT_COLUMNS]

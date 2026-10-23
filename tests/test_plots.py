"""Tests of the diagnostic charts using synthetic dataframes only."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from frame_analyzer import plots
from frame_analyzer.config import AnalyzerConfig
from frame_analyzer.graph import build_graph, graph_stats

CFG = AnalyzerConfig(input_dir=Path("."))


def _frames(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lap = rng.lognormal(5, 0.6, n)
    df = pd.DataFrame({
        "index": np.arange(n), "filename": [f"frame_{i:05d}.png" for i in range(n)], "readable": True,
        "laplacian_var": lap, "sift_keypoints": rng.integers(50, 4000, n),
        "sift_density_per_mp": rng.uniform(100, 5000, n), "occupancy_ratio": rng.uniform(0.2, 1.0, n),
    })
    df.loc[3, ["readable"]] = False
    df.loc[3, ["laplacian_var", "sift_keypoints", "sift_density_per_mp", "occupancy_ratio"]] = np.nan
    return df


def _pairs(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        for d in (1, 2, 5):
            if i + d < n:
                inl = int(rng.integers(0, 400))
                ratio = float(rng.uniform(0, 1))
                rows.append({"i": i, "j": i + d, "offset": d, "inliers": inl, "inlier_ratio": ratio,
                             "median_disp_px": float(rng.uniform(0.5, 40)),
                             "edge_accepted": inl >= 30 and ratio >= 0.25})
    return pd.DataFrame(rows)


def _grid(n: int) -> np.ndarray:
    g = np.random.default_rng(1).integers(0, 60, (n, 8, 8)).astype(np.int32)
    g[3] = 0
    return g


def _calls(frames: pd.DataFrame, pairs: pd.DataFrame, grid: np.ndarray, stats: dict, redundant):
    return {
        "sharpness": lambda p: plots.plot_sharpness(frames, CFG, p),
        "sharpness_dist": lambda p: plots.plot_sharpness_distribution(frames, CFG, p),
        "sift_kp": lambda p: plots.plot_sift_keypoints(frames, CFG, p),
        "sift_density": lambda p: plots.plot_sift_density(frames, p),
        "sift_dist": lambda p: plots.plot_sift_distribution(frames, p),
        "occupancy": lambda p: plots.plot_grid_occupancy(frames, CFG, p),
        "heatmap": lambda p: plots.plot_keypoint_heatmap(grid, frames, p),
        "examples": lambda p: plots.plot_grid_examples(grid, frames, p, n=6),
        "inliers": lambda p: plots.plot_matching_inliers(pairs, CFG, p),
        "ratio": lambda p: plots.plot_matching_inlier_ratio(pairs, CFG, p),
        "scatter": lambda p: plots.plot_matching_scatter(pairs, CFG, p),
        "redundancy": lambda p: plots.plot_temporal_redundancy(pairs, redundant, CFG, p),
        "degree": lambda p: plots.plot_degree_distribution(stats, p),
    }


def _is_png(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 1000 and p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.parametrize("n", [12, 1250])
def test_all_plots_with_data(tmp_path: Path, n: int) -> None:
    frames, pairs = _frames(n), _pairs(n)
    stats = graph_stats(build_graph([i for i in range(n) if i != 3], pairs))
    redundant = pairs[pairs["offset"] == 1].head(3)[["i", "j"]]
    for name, fn in _calls(frames, pairs, _grid(n), stats, redundant).items():
        out = fn(tmp_path / f"{name}.png")
        assert _is_png(out), name


def test_all_plots_with_empty_input(tmp_path: Path) -> None:
    empty_grid = np.zeros((0, 8, 8), dtype=np.int32)
    for name, fn in _calls(pd.DataFrame(), pd.DataFrame(), empty_grid, {}, None).items():
        assert _is_png(fn(tmp_path / f"{name}.png")), name


def test_all_plots_with_all_nan(tmp_path: Path) -> None:
    n = 6
    frames = _frames(n)
    for c in ("laplacian_var", "sift_keypoints", "sift_density_per_mp", "occupancy_ratio"):
        frames[c] = np.nan
    frames["readable"] = False
    pairs = _pairs(n)
    pairs["inliers"] = np.nan
    pairs["inlier_ratio"] = np.nan
    pairs["median_disp_px"] = np.nan
    stats = graph_stats(build_graph([], pd.DataFrame()))
    for name, fn in _calls(frames, pairs, np.zeros((n, 8, 8), np.int32), stats, pd.DataFrame()).items():
        assert _is_png(fn(tmp_path / f"{name}.png")), name

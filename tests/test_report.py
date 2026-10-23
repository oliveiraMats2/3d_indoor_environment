"""Tests of the HTML quality report and the output README."""
from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from frame_analyzer.config import AnalyzerConfig
from frame_analyzer.report import (
    REPORT_NAME,
    SECTIONS,
    build_command,
    build_recommendations,
    component_gaps,
    write_output_readme,
    write_report,
)

HEADINGS = [name for _, name in SECTIONS]


def _png(path: Path, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    img = (rng.random((12, 16)) * 255).astype(np.uint8)
    assert cv2.imwrite(str(path), img)
    return path


def _frames_df(n: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "index": np.arange(n), "filename": [f"frame_{k:04d}.jpg" for k in range(n)],
        "readable": [True] * (n - 1) + [False], "error": [""] * (n - 1) + ["cannot decode"],
        "width": 1920, "height": 1080, "aspect_ratio": 1920 / 1080, "analysis_width": 1280, "analysis_height": 720,
        "laplacian_var": rng.uniform(10, 300, n), "sift_keypoints": rng.integers(50, 2000, n),
        "occupancy_ratio": rng.uniform(0.2, 1.0, n), "dark_frac": rng.uniform(0, 0.5, n),
        "saturated_frac": rng.uniform(0, 0.2, n), "degree": rng.integers(0, 4, n),
        "component_id": [0] * 5 + [1] * 4 + [-1], "component_size": [5] * 5 + [4] * 4 + [0],
    })
    df["sharpness_rel"] = df["laplacian_var"] / df["laplacian_var"].median()
    df.loc[n - 1, ["laplacian_var", "sharpness_rel", "occupancy_ratio"]] = np.nan
    df["quality"] = ["good", "review", "poor", "good", "review", "good", "good", "poor", "good", "poor"][:n]
    df["flags"] = ["", "low_occupancy", "blur_poor", "", "dark", "", "", "isolated", "", "unreadable"][:n]
    return df


def _pairs_df() -> pd.DataFrame:
    rows = []
    for i in range(8):
        for d in (1, 2):
            j = i + d
            if j > 8:
                continue
            inl = 0 if (i, j) == (4, 5) else 40 + 5 * i
            rows.append({"i": i, "j": j, "offset": d, "lowe_matches": 100, "inliers": inl,
                         "inlier_ratio": inl / 100, "verified": inl >= 8, "status": "ok" if inl else "too_few_matches",
                         "homography_inliers": inl // 2, "median_disp_px": 3.0, "edge_accepted": inl >= 30})
    return pd.DataFrame(rows)


def _summary() -> dict:
    stats = {m: {"mean": 1.0, "std": 0.1, "min": 0.5, "p05": 0.6, "median": 1.1, "p95": 1.5, "max": 2.0}
             for m in ("laplacian_var", "sift_keypoints", "sift_density_per_mp", "occupancy_ratio", "inliers",
                       "inlier_ratio", "brightness_mean", "dark_frac", "saturated_frac")}
    stats["laplacian_var"]["median"] = 123.456
    return {
        "version": "0.1.0", "started_at": "2026-10-20T10:00:00", "finished_at": "2026-10-20T10:01:00",
        "elapsed_s": 60.5, "timings": {"read": 10.0, "match": 40.0}, "input_dir": "frames", "output_dir": "out",
        "n_files": 10, "n_readable": 9, "n_unreadable": 1,
        "unreadable": [{"filename": "frame_0009.jpg", "error": "cannot decode"}],
        "n_pairs": 15, "n_pairs_verified": 14, "n_edges_accepted": 14, "offsets_used": [1, 2],
        "resolution": {"unique": [{"width": 1920, "height": 1080, "count": 9}], "consistent": True,
                       "width_min": 1920, "width_max": 1920, "height_min": 1080, "height_max": 1080},
        "stats": stats,
        "thresholds": {"sharpness_median": 123.456, "sharpness_poor_abs": 37.0, "sharpness_review_abs": 74.1},
        "quality_counts": {"good": 5, "review": 2, "poor": 3},
        "flag_counts": {"blur_poor": 1, "low_occupancy": 1, "dark": 1, "isolated": 1, "unreadable": 1},
        "graph": {"n_nodes": 9, "n_edges": 14, "n_components": 2, "n_isolated": 1, "isolated_nodes": [7],
                  "largest_component_size": 5, "largest_component_pct": 55.6, "component_sizes": [5, 3, 1],
                  "degree_histogram": {0: 1, 1: 2, 2: 6}, "component_ranges": [[0, 4, 5], [5, 8, 3], [7, 7, 1]]},
        "redundancy": {"enabled": True, "n_candidates": 1},
        "warnings": ["offset 50 dropped: only 10 frames"],
    }


def _redundant_df() -> pd.DataFrame:
    return pd.DataFrame([{"i": 0, "j": 1, "filename_i": "frame_0000.jpg", "filename_j": "frame_0001.jpg",
                          "inliers": 90, "inlier_ratio": 0.95, "median_disp_px": 0.8, "reason": "near duplicate"}])


def _edges_df() -> pd.DataFrame:
    return pd.DataFrame([{"source": 0, "target": 1, "offset": 1, "inliers": 40, "inlier_ratio": 0.4,
                          "criteria": "verified=fundamental; inliers>=30; inlier_ratio>=0.25"}])


@pytest.fixture
def figures(tmp_path: Path) -> dict[str, Path]:
    fig_dir = tmp_path / "figs"
    fig_dir.mkdir()
    keys = ["sharpness", "sift_keypoints", "keypoint_heatmap", "matching_scatter", "graph", "degree_distribution"]
    return {k: _png(fig_dir / f"{k}.png", n) for n, k in enumerate(keys)}


def _cfg(tmp_path: Path, **kw) -> AnalyzerConfig:
    return AnalyzerConfig(input_dir=tmp_path / "frames", output_dir=tmp_path / "out", **kw)


def _write_full(tmp_path: Path, figures: dict[str, Path], cfg: AnalyzerConfig, graph_html: Path | None) -> str:
    out = cfg.output_dir
    path = write_report(out, _summary(), _frames_df(), _pairs_df(), _edges_df(), _redundant_df(), figures, cfg,
                        graph_html)
    assert path == out / REPORT_NAME and path.exists()
    return path.read_text(encoding="utf-8")


def test_report_has_all_sections_in_order(tmp_path: Path, figures: dict[str, Path]) -> None:
    html = _write_full(tmp_path, figures, _cfg(tmp_path), None)
    assert len(HEADINGS) == 13
    assert "Connectivity Graph: Visual Reference" in html
    positions = [html.index(f">{h}</h2>") for h in HEADINGS]
    assert positions == sorted(positions)


def test_report_is_self_contained(tmp_path: Path, figures: dict[str, Path]) -> None:
    html = _write_full(tmp_path, figures, _cfg(tmp_path), None)
    assert html.count("data:image/png;base64,") == len(figures)
    assert "http" not in re.sub(r"data:image/png;base64,[A-Za-z0-9+/=]+", "", html).lower()
    assert "<script" not in html.lower()
    assert "@import" not in html and "<link" not in html.lower()
    assert "URL pública" in html


def test_reference_url_is_the_only_external_link(tmp_path: Path, figures: dict[str, Path]) -> None:
    url = "https://example.org/reference-graph"
    html = _write_full(tmp_path, figures, _cfg(tmp_path, reference_url=url), None)
    assert f'<a href="{url}">' in html
    others = [m for m in re.findall(r"https?://[^\s\"'<>]+", html) if m != url]
    assert others == []


def test_graph_html_linked_relatively(tmp_path: Path, figures: dict[str, Path]) -> None:
    cfg = _cfg(tmp_path)
    cfg.output_dir.mkdir(parents=True)
    graph = cfg.output_dir / "connectivity_graph.html"
    graph.write_text("<html><body>graph</body></html>", encoding="utf-8")
    html = _write_full(tmp_path, figures, cfg, graph)
    assert 'href="connectivity_graph.html"' in html
    assert "graph</body>" not in html


def test_numbers_come_from_summary(tmp_path: Path, figures: dict[str, Path]) -> None:
    html = _write_full(tmp_path, figures, _cfg(tmp_path), None)
    assert "123.5" in html                       # sharpness median
    assert "55.6%" in html                       # largest component pct
    assert "offset 50 dropped" in html           # warnings
    assert "frame_0009.jpg" in html              # unreadable
    assert "garante" in html                     # diagnosis disclaimer


def test_recommendations_list_component_gaps(tmp_path: Path) -> None:
    summary = _summary()
    assert component_gaps(summary) == [(4, 5)]
    recs = build_recommendations(summary, _frames_df(), _cfg(tmp_path))
    text = " ".join(recs)
    assert "4 (frame_0004.jpg)" in text and "5 (frame_0005.jpg)" in text
    assert "7 (frame_0007.jpg)" in text          # isolated frame


def test_empty_dataframes_and_missing_keys(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    empty = pd.DataFrame()
    path = write_report(cfg.output_dir, {}, empty, empty, empty, None, {}, cfg, None)
    html = path.read_text(encoding="utf-8")
    for h in HEADINGS:
        assert f">{h}</h2>" in html
    assert "http" not in re.sub(r"data:image/png;base64,[A-Za-z0-9+/=]+", "", html).lower()


def test_minimal_summary_with_none_inputs(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    summary = {"n_files": 3, "n_readable": 3, "graph": {"n_components": 1, "largest_component_pct": 100.0}}
    figs = {"missing": tmp_path / "nope.png"}
    path = write_report(cfg.output_dir, summary, None, None, None, None, figs, cfg, tmp_path / "absent.html")
    html = path.read_text(encoding="utf-8")
    assert "única componente" in html
    assert "data:image/png" not in html


def test_html_is_escaped(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    summary = _summary()
    summary["warnings"] = ["<b>bad</b>"]
    path = write_report(cfg.output_dir, summary, None, None, None, None, {}, cfg, None)
    html = path.read_text(encoding="utf-8")
    assert "<b>bad</b>" not in html and "&lt;b&gt;bad&lt;/b&gt;" in html


def test_build_command_only_non_defaults(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, stride=3, temporal_offsets=(1, 3), grid_rows=4, grid_cols=6, redundancy=False)
    cmd = build_command(cfg)
    assert cmd.startswith("python analyze_frames.py --input ")
    assert "--stride 3" in cmd and "--offsets 1,3" in cmd and "--grid 4x6" in cmd and "--no-redundancy" in cmd
    assert "--seed" not in cmd and "--lowe-ratio" not in cmd


def test_output_readme(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, limit=50)
    path = write_output_readme(cfg.output_dir, _summary(), cfg)
    text = path.read_text(encoding="utf-8")
    assert path.name == "README.md"
    for name in ("quality_report.html", "frame_quality.csv", "pair_quality.csv", "connectivity_edges.csv",
                 "redundant_pairs.csv", "summary.json", "connectivity_graph.html"):
        assert name in text
    assert "--limit 50" in text
    assert write_output_readme(cfg.output_dir, {}, cfg).exists()

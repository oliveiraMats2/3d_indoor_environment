"""End-to-end run on synthetic frames: every output file is written and consistent."""
from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from conftest import make_textured
from frame_analyzer import pipeline

EXPECTED = [
    "quality_report.html", "frame_quality.csv", "pair_quality.csv", "connectivity_edges.csv", "summary.json",
    "sharpness.png", "sharpness_distribution.png", "sift_keypoints.png", "sift_density.png", "sift_distribution.png",
    "grid_occupancy.png", "keypoint_heatmap.png", "grid_examples.png", "matching_inliers.png",
    "matching_inlier_ratio.png", "matching_scatter.png", "connectivity_graph.png", "connectivity_graph.html",
    "temporal_redundancy.png", "redundant_pairs.csv", "README.md",
]


def write_sequence(folder: Path, n: int) -> None:
    """A camera panning over one large synthetic texture, plus one corrupt file."""
    folder.mkdir(parents=True)
    canvas = make_textured(height=360, width=640 + 12 * n, seed=3)
    for k in range(n):
        cv2.imwrite(str(folder / f"img_{k}.png"), canvas[:, 12 * k:12 * k + 480])
    (folder / f"img_{n}.png").write_bytes(b"not an image")


@pytest.fixture
def outputs(cfg) -> tuple[Path, dict]:
    write_sequence(cfg.input_dir, 12)
    cfg = replace(cfg, temporal_offsets=(1, 2, 5), analysis_max_side=0)
    return cfg.output_dir, pipeline.run(cfg)


def test_every_output_written(outputs):
    out, _ = outputs
    for name in EXPECTED:
        assert (out / name).is_file() and (out / name).stat().st_size > 0, name


def test_tables_consistent_with_summary(outputs):
    out, summary = outputs
    frames = pd.read_csv(out / "frame_quality.csv")
    pairs = pd.read_csv(out / "pair_quality.csv")
    edges = pd.read_csv(out / "connectivity_edges.csv")
    assert len(frames) == summary["n_files"] == 13
    assert summary["n_unreadable"] == 1 and frames.loc[12, "quality"] == "poor"
    assert list(frames["filename"][:3]) == ["img_0.png", "img_1.png", "img_2.png"]  # natural order
    assert len(pairs) == summary["n_pairs"]
    assert len(edges) == int(pairs["edge_accepted"].sum()) == summary["graph"]["n_edges"]
    assert set(pairs["offset"]) == {1, 2, 5}
    json.loads((out / "summary.json").read_text())  # valid JSON (NaN written as null)


def test_report_has_no_photos(outputs):
    out, _ = outputs
    html = (out / "quality_report.html").read_text()
    assert "Connectivity Graph: Visual Reference" in html
    text = re.sub(r"data:image/png;base64,[A-Za-z0-9+/=]+", "", html)  # base64 may contain "http" by chance
    assert "http" not in text and "image/jpeg" not in html


def test_optional_outputs_disabled(cfg):
    write_sequence(cfg.input_dir, 6)
    cfg = replace(cfg, redundancy=False, interactive_graph=False, temporal_offsets=(1,))
    pipeline.run(cfg)
    assert not (cfg.output_dir / "redundant_pairs.csv").exists()
    assert not (cfg.output_dir / "temporal_redundancy.png").exists()
    assert not (cfg.output_dir / "connectivity_graph.html").exists()


def test_empty_directory_fails_cleanly(cfg):
    cfg.input_dir.mkdir()
    with pytest.raises(pipeline.EmptyInputError):
        pipeline.run(cfg)

"""Configuration of the frame quality analyzer (CLI arguments -> AnalyzerConfig)."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


@dataclass
class AnalyzerConfig:
    input_dir: Path
    output_dir: Path = Path("quality_report")
    extensions: tuple[str, ...] = DEFAULT_EXTENSIONS
    limit: int | None = None              # analyze only the first N frames (after natural sort); None = all
    stride: int = 1                       # analyze 1 of every N frames
    analysis_max_side: int = 1280         # longest side used for every metric; 0 = full resolution
    seed: int = 42

    # grid occupancy
    grid_rows: int = 8
    grid_cols: int = 8

    # SIFT
    sift_nfeatures: int = 4000            # 0 = unlimited
    sift_contrast_threshold: float = 0.04
    sift_edge_threshold: float = 10.0

    # matching and geometric verification
    temporal_offsets: tuple[int, ...] = (1, 2, 5)
    lowe_ratio: float = 0.75
    min_lowe_matches: int = 15            # fewer -> status too_few_matches, no RANSAC
    ransac_threshold_px: float = 1.5      # in analysis-resolution pixels
    ransac_confidence: float = 0.999
    ransac_max_iters: int = 5000
    compute_homography: bool = True       # complementary diagnostic only

    # graph edge acceptance (both must hold)
    edge_min_inliers: int = 30
    edge_min_inlier_ratio: float = 0.50   # F-RANSAC on unrelated images can reach ~0.25 by chance

    # per-frame classification
    sharpness_poor_rel: float = 0.30      # lapvar / median lapvar below this -> poor
    sharpness_review_rel: float = 0.60    # below this -> review
    sharpness_abs_min: float | None = None  # optional calibrated absolute floor -> poor
    keypoints_poor_min: int = 100
    keypoints_review_min: int = 300
    occupancy_review_min: float = 0.50
    degree_review_min: int = 1            # fewer accepted edges -> review
    dark_threshold: int = 10              # gray <= this counts as very dark
    saturated_threshold: int = 245        # gray >= this counts as saturated
    dark_frac_review: float = 0.30
    saturated_frac_review: float = 0.10

    # temporal redundancy (optional)
    redundancy: bool = True
    redundancy_min_inlier_ratio: float = 0.90
    redundancy_max_median_disp_px: float = 2.0

    # report
    interactive_graph: bool = True
    reference_url: str | None = None      # real URL of the visual reference, if one exists
    review_table_max_rows: int = 200

    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["input_dir"] = str(self.input_dir)
        d["output_dir"] = str(self.output_dir)
        d.pop("warnings")
        return d


def _int_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(t) for t in text.replace(",", " ").split())


def _ext_tuple(text: str) -> tuple[str, ...]:
    return tuple("." + t.strip().lower().lstrip(".") for t in text.replace(",", " ").split())


def build_parser() -> argparse.ArgumentParser:
    d = AnalyzerConfig(input_dir=Path("."))
    p = argparse.ArgumentParser(
        prog="analyze_frames.py",
        description="Frame quality diagnosis before 3D reconstruction (no reconstruction is run).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, type=Path, help="directory with the extracted frames")
    p.add_argument("--output", default=d.output_dir, type=Path, help="output directory")
    p.add_argument("--extensions", default=",".join(e.lstrip(".") for e in d.extensions), type=_ext_tuple)
    p.add_argument("--limit", default=None, type=int)
    p.add_argument("--stride", default=d.stride, type=int)
    p.add_argument("--analysis-max-side", default=d.analysis_max_side, type=int)
    p.add_argument("--seed", default=d.seed, type=int)
    p.add_argument("--grid", default=f"{d.grid_rows}x{d.grid_cols}", help="ROWSxCOLS")
    p.add_argument("--sift-nfeatures", default=d.sift_nfeatures, type=int)
    p.add_argument("--sift-contrast-threshold", default=d.sift_contrast_threshold, type=float)
    p.add_argument("--sift-edge-threshold", default=d.sift_edge_threshold, type=float)
    p.add_argument("--offsets", default=",".join(map(str, d.temporal_offsets)), type=_int_tuple,
                   help="temporal distances of the compared pairs")
    p.add_argument("--lowe-ratio", default=d.lowe_ratio, type=float)
    p.add_argument("--min-lowe-matches", default=d.min_lowe_matches, type=int)
    p.add_argument("--ransac-threshold", default=d.ransac_threshold_px, type=float)
    p.add_argument("--ransac-confidence", default=d.ransac_confidence, type=float)
    p.add_argument("--ransac-max-iters", default=d.ransac_max_iters, type=int)
    p.add_argument("--no-homography", action="store_true")
    p.add_argument("--edge-min-inliers", default=d.edge_min_inliers, type=int)
    p.add_argument("--edge-min-inlier-ratio", default=d.edge_min_inlier_ratio, type=float)
    p.add_argument("--sharpness-poor-rel", default=d.sharpness_poor_rel, type=float)
    p.add_argument("--sharpness-review-rel", default=d.sharpness_review_rel, type=float)
    p.add_argument("--sharpness-abs-min", default=None, type=float)
    p.add_argument("--keypoints-poor-min", default=d.keypoints_poor_min, type=int)
    p.add_argument("--keypoints-review-min", default=d.keypoints_review_min, type=int)
    p.add_argument("--occupancy-review-min", default=d.occupancy_review_min, type=float)
    p.add_argument("--degree-review-min", default=d.degree_review_min, type=int)
    p.add_argument("--no-redundancy", action="store_true")
    p.add_argument("--redundancy-min-inlier-ratio", default=d.redundancy_min_inlier_ratio, type=float)
    p.add_argument("--redundancy-max-median-disp", default=d.redundancy_max_median_disp_px, type=float)
    p.add_argument("--no-interactive-graph", action="store_true")
    p.add_argument("--reference-url", default=None, help="real URL of the graph visual reference, if any")
    return p


def parse_args(argv: list[str] | None = None) -> AnalyzerConfig:
    a = build_parser().parse_args(argv)
    rows, cols = (int(v) for v in a.grid.lower().split("x"))
    return AnalyzerConfig(
        input_dir=a.input, output_dir=a.output, extensions=a.extensions, limit=a.limit, stride=a.stride,
        analysis_max_side=a.analysis_max_side, seed=a.seed, grid_rows=rows, grid_cols=cols,
        sift_nfeatures=a.sift_nfeatures, sift_contrast_threshold=a.sift_contrast_threshold,
        sift_edge_threshold=a.sift_edge_threshold, temporal_offsets=a.offsets, lowe_ratio=a.lowe_ratio,
        min_lowe_matches=a.min_lowe_matches, ransac_threshold_px=a.ransac_threshold,
        ransac_confidence=a.ransac_confidence, ransac_max_iters=a.ransac_max_iters,
        compute_homography=not a.no_homography, edge_min_inliers=a.edge_min_inliers,
        edge_min_inlier_ratio=a.edge_min_inlier_ratio, sharpness_poor_rel=a.sharpness_poor_rel,
        sharpness_review_rel=a.sharpness_review_rel, sharpness_abs_min=a.sharpness_abs_min,
        keypoints_poor_min=a.keypoints_poor_min, keypoints_review_min=a.keypoints_review_min,
        occupancy_review_min=a.occupancy_review_min, degree_review_min=a.degree_review_min,
        redundancy=not a.no_redundancy, redundancy_min_inlier_ratio=a.redundancy_min_inlier_ratio,
        redundancy_max_median_disp_px=a.redundancy_max_median_disp, interactive_graph=not a.no_interactive_graph,
        reference_url=a.reference_url,
    )

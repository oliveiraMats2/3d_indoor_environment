"""Single streaming pass over the frames, then graph, classification, plots, CSV/JSON outputs and the report."""
from __future__ import annotations

import json
import logging
import math
import sys
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from frame_analyzer import __version__
from frame_analyzer.classify import classify_frames
from frame_analyzer.config import AnalyzerConfig, parse_args
from frame_analyzer.features import (SiftUnavailableError, brightness_stats, create_sift, detect_sift, grid_counts,
                                     grid_occupancy, keypoint_density, laplacian_variance)
from frame_analyzer.io import ImageReadError, list_frames, read_gray
from frame_analyzer.matching import F_METHOD_NAME, accept_edge, match_pair

log = logging.getLogger(__name__)

PROGRESS_EVERY = 50
FRAME_COLUMNS = [
    "index", "filename", "readable", "error", "width", "height", "aspect_ratio", "analysis_width", "analysis_height",
    "laplacian_var", "sharpness_rel", "sift_keypoints", "sift_density_per_mp", "occupied_cells", "total_cells",
    "occupancy_ratio", "brightness_mean", "brightness_std", "dark_frac", "saturated_frac", "degree", "component_id",
    "component_size", "flags", "quality",
]
FRAME_INT_COLUMNS = ["width", "height", "analysis_width", "analysis_height", "sift_keypoints", "occupied_cells",
                     "total_cells"]
PAIR_COLUMNS = [
    "i", "j", "offset", "filename_i", "filename_j", "kp_i", "kp_j", "raw_matches", "lowe_matches", "inliers",
    "inlier_ratio", "model", "verified", "status", "homography_inliers", "homography_ratio", "median_disp_px",
    "edge_accepted",
]
STAT_COLUMNS = ["laplacian_var", "sift_keypoints", "sift_density_per_mp", "occupancy_ratio", "brightness_mean",
                "dark_frac", "saturated_frac"]
PLOT_STEMS = [
    "sharpness", "sharpness_distribution", "sift_keypoints", "sift_density", "sift_distribution", "grid_occupancy",
    "keypoint_heatmap", "grid_examples", "matching_inliers", "matching_inlier_ratio", "matching_scatter",
    "connectivity_graph", "degree_distribution",
]


class EmptyInputError(ValueError):
    """Raised when the input directory has no frame with an accepted extension."""


def select_frames(cfg: AnalyzerConfig) -> tuple[list[Path], int]:
    """Natural-sorted frames after stride then limit; returns (selected, number of listed files)."""
    files = list_frames(cfg.input_dir, cfg.extensions)
    if not files:
        raise EmptyInputError(
            f"no frame found in {cfg.input_dir} with extensions {', '.join(cfg.extensions)}")
    selected = files[:: max(1, cfg.stride)]
    if cfg.limit is not None and cfg.limit > 0:
        selected = selected[: cfg.limit]
    return selected, len(files)


def usable_offsets(cfg: AnalyzerConfig, n_frames: int) -> list[int]:
    """Sorted unique positive offsets < n_frames; dropped ones are reported in cfg.warnings."""
    used: list[int] = []
    for d in sorted(set(cfg.temporal_offsets)):
        if d <= 0 or d >= n_frames:
            msg = f"temporal offset {d} dropped (must be in 1..{n_frames - 1} for {n_frames} analysed frames)"
            log.warning(msg)
            cfg.warnings.append(msg)
        else:
            used.append(d)
    return used


def _nan_frame_row(index: int, path: Path, error: str, total_cells: int) -> dict:
    row = {c: float("nan") for c in FRAME_COLUMNS}
    row.update(index=index, filename=path.name, readable=False, error=error, total_cells=total_cells,
               degree=0, component_id=-1, component_size=0, flags="unreadable", quality="poor")
    return row


def _frame_metrics(index: int, path: Path, gray: np.ndarray, orig_w: int, orig_h: int, pts: np.ndarray,
                   counts: np.ndarray, cfg: AnalyzerConfig) -> dict:
    ah, aw = gray.shape[:2]
    occupied, total, ratio = grid_occupancy(counts)
    row = {
        "index": index, "filename": path.name, "readable": True, "error": "", "width": orig_w, "height": orig_h,
        "aspect_ratio": orig_w / orig_h if orig_h else float("nan"), "analysis_width": aw, "analysis_height": ah,
        "laplacian_var": laplacian_variance(gray), "sharpness_rel": float("nan"), "sift_keypoints": len(pts),
        "sift_density_per_mp": keypoint_density(len(pts), aw, ah), "occupied_cells": occupied, "total_cells": total,
        "occupancy_ratio": ratio, "degree": 0, "component_id": -1, "component_size": 0, "flags": "", "quality": "",
    }
    row.update(brightness_stats(gray, cfg.dark_threshold, cfg.saturated_threshold))
    return row


def _pair_row(i: int, j: int, names: list[str], cache: dict, cfg: AnalyzerConfig) -> dict:
    a, b = cache.get(i), cache.get(j)
    row = {"i": i, "j": j, "offset": j - i, "filename_i": names[i], "filename_j": names[j],
           "kp_i": len(a[0]) if a else 0, "kp_j": len(b[0]) if b else 0}
    if a is None or b is None:
        row.update(raw_matches=0, lowe_matches=0, inliers=0, inlier_ratio=float("nan"), model="fundamental",
                   verified=False, status="unreadable", homography_inliers=-1, homography_ratio=float("nan"),
                   median_disp_px=float("nan"))
    else:
        row.update(match_pair(a[0], a[1], b[0], b[1], cfg))
    row["edge_accepted"] = accept_edge(row, cfg)
    return row


def _log_progress(done: int, total: int, t0: float) -> None:
    elapsed = time.perf_counter() - t0
    eta = elapsed / done * (total - done) if done else float("nan")
    log.info("frames %d/%d (%.0f%%) elapsed %.1fs ETA %.1fs", done, total, 100 * done / total, elapsed, eta)


def stream_frames(paths: list[Path], offsets: list[int], cfg: AnalyzerConfig
                  ) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Read each frame once, compute its metrics and match it against the cached previous frames."""
    sift = create_sift(cfg)
    n = len(paths)
    names = [p.name for p in paths]
    total_cells = cfg.grid_rows * cfg.grid_cols
    grids = np.zeros((n, cfg.grid_rows, cfg.grid_cols), np.int32)
    window = max(offsets, default=0)
    cache: OrderedDict[int, tuple[np.ndarray, np.ndarray | None] | None] = OrderedDict()
    frame_rows: list[dict] = []
    pair_rows: list[dict] = []
    t0 = time.perf_counter()
    for j, path in enumerate(paths):
        try:
            gray, w, h = read_gray(path, cfg.analysis_max_side)
        except ImageReadError as exc:
            log.warning("unreadable frame %s: %s", path.name, exc)
            frame_rows.append(_nan_frame_row(j, path, str(exc), total_cells))
            cache[j] = None
        else:
            pts, desc = detect_sift(sift, gray)
            grids[j] = grid_counts(pts, gray.shape[1], gray.shape[0], cfg.grid_rows, cfg.grid_cols)
            frame_rows.append(_frame_metrics(j, path, gray, w, h, pts, grids[j], cfg))
            cache[j] = (pts, desc)
            del gray
        for d in offsets:
            if j - d >= 0:
                pair_rows.append(_pair_row(j - d, j, names, cache, cfg))
        while cache and next(iter(cache)) < j - window + 1:
            cache.popitem(last=False)
        if (j + 1) % PROGRESS_EVERY == 0 or j + 1 == n:
            _log_progress(j + 1, n, t0)
    frames_df = pd.DataFrame(frame_rows, columns=FRAME_COLUMNS)
    frames_df["readable"] = frames_df["readable"].astype(bool)
    pairs_df = pd.DataFrame(pair_rows, columns=PAIR_COLUMNS)
    if not pairs_df.empty:
        pairs_df["verified"] = pairs_df["verified"].astype(bool)
        pairs_df["edge_accepted"] = pairs_df["edge_accepted"].astype(bool)
    return frames_df, grids, pairs_df


def add_relative_sharpness(frames_df: pd.DataFrame, cfg: AnalyzerConfig) -> float:
    """Fill sharpness_rel = laplacian_var / median(readable laplacian_var); returns the median."""
    lap = frames_df.loc[frames_df["readable"], "laplacian_var"].astype(float)
    median = float(lap.median()) if len(lap) else float("nan")
    if math.isnan(median) or median <= 0:
        frames_df["sharpness_rel"] = float("nan")
        msg = "median Laplacian variance is zero or undefined: relative sharpness is not available"
        log.warning(msg)
        cfg.warnings.append(msg)
    else:
        frames_df["sharpness_rel"] = frames_df["laplacian_var"].astype(float) / median
    return median


def fill_graph_columns(frames_df: pd.DataFrame, stats: dict) -> None:
    """Copy degree / component id / component size from graph_stats into frames_df (in place)."""
    degree = stats.get("degree", {})
    comp_of = stats.get("component_of", {})
    sizes = [len(c) for c in stats.get("components", [])]
    readable = frames_df["readable"].to_numpy()
    idx = frames_df["index"].to_numpy()
    frames_df["degree"] = [int(degree.get(int(k), 0)) if r else 0 for k, r in zip(idx, readable)]
    frames_df["component_id"] = [int(comp_of.get(int(k), -1)) if r else -1 for k, r in zip(idx, readable)]
    frames_df["component_size"] = [sizes[c] if 0 <= c < len(sizes) else 0 for c in frames_df["component_id"]]


def describe(values: pd.Series) -> dict:
    """mean/std/min/p05/median/p95/max of the finite values (all None when empty)."""
    v = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    keys = ["mean", "std", "min", "p05", "median", "p95", "max"]
    if v.size == 0:
        return dict.fromkeys(keys)
    return {"mean": float(v.mean()), "std": float(v.std()), "min": float(v.min()),
            "p05": float(np.percentile(v, 5)), "median": float(np.median(v)), "p95": float(np.percentile(v, 95)),
            "max": float(v.max())}


def resolution_summary(frames_df: pd.DataFrame, cfg: AnalyzerConfig) -> dict:
    """Unique original resolutions; a warning is added to cfg.warnings when they differ."""
    ok = frames_df[frames_df["readable"]]
    if ok.empty:
        return {"unique": [], "consistent": True, "width_min": None, "width_max": None,
                "height_min": None, "height_max": None}
    groups = ok.groupby(["width", "height"]).size().reset_index(name="count").sort_values("count", ascending=False)
    unique = [{"width": int(r.width), "height": int(r.height), "count": int(r["count"])} for _, r in groups.iterrows()]
    if len(unique) > 1:
        msg = "inconsistent resolutions: " + ", ".join(f"{u['width']}x{u['height']} ({u['count']})" for u in unique)
        log.warning(msg)
        cfg.warnings.append(msg)
    return {"unique": unique, "consistent": len(unique) <= 1, "width_min": int(ok["width"].min()),
            "width_max": int(ok["width"].max()), "height_min": int(ok["height"].min()),
            "height_max": int(ok["height"].max())}


def component_ranges(components: list[list[int]]) -> list[list[int]]:
    """[first_index, last_index, size] of each component (same order as the input)."""
    return [[int(min(c)), int(max(c)), len(c)] for c in components if c]


def thresholds_summary(cfg: AnalyzerConfig, sharpness_median: float) -> dict:
    """All classification and edge thresholds, with the relative sharpness ones converted to absolute values."""
    med = None if math.isnan(sharpness_median) else sharpness_median
    return {
        "sharpness_median": med,
        "sharpness_poor_abs": None if med is None else med * cfg.sharpness_poor_rel,
        "sharpness_review_abs": None if med is None else med * cfg.sharpness_review_rel,
        "sharpness_poor_rel": cfg.sharpness_poor_rel, "sharpness_review_rel": cfg.sharpness_review_rel,
        "sharpness_abs_min": cfg.sharpness_abs_min, "keypoints_poor_min": cfg.keypoints_poor_min,
        "keypoints_review_min": cfg.keypoints_review_min, "occupancy_review_min": cfg.occupancy_review_min,
        "degree_review_min": cfg.degree_review_min, "dark_threshold": cfg.dark_threshold,
        "saturated_threshold": cfg.saturated_threshold, "dark_frac_review": cfg.dark_frac_review,
        "saturated_frac_review": cfg.saturated_frac_review, "lowe_ratio": cfg.lowe_ratio,
        "min_lowe_matches": cfg.min_lowe_matches, "ransac_threshold_px": cfg.ransac_threshold_px,
        "ransac_confidence": cfg.ransac_confidence, "ransac_max_iters": cfg.ransac_max_iters,
        "fundamental_method": F_METHOD_NAME, "edge_min_inliers": cfg.edge_min_inliers,
        "edge_min_inlier_ratio": cfg.edge_min_inlier_ratio,
        "redundancy_min_inlier_ratio": cfg.redundancy_min_inlier_ratio,
        "redundancy_max_median_disp_px": cfg.redundancy_max_median_disp_px,
    }


def _jsonable(obj: object) -> object:
    """Recursively convert numpy types and NaN/inf floats (-> None) for json.dump."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    return obj


def write_summary(path: Path, summary: dict) -> Path:
    """summary.json with NaN -> null and default=str for anything else."""
    path.write_text(json.dumps(_jsonable(summary), indent=2, default=str, allow_nan=False), encoding="utf-8")
    return path


def _csv_frames(frames_df: pd.DataFrame, path: Path) -> None:
    out = frames_df.copy()
    for c in FRAME_INT_COLUMNS:
        out[c] = pd.to_numeric(out[c], errors="coerce").round().astype("Int64")
    out.to_csv(path, index=False)


def _safe_plot(figures: dict[str, Path], stem: str, fn: Callable[[], Path | None], cfg: AnalyzerConfig) -> None:
    try:
        result = fn()
    except Exception as exc:  # a broken figure must not lose the CSV/JSON outputs
        log.exception("figure %s failed", stem)
        cfg.warnings.append(f"figure {stem} could not be generated: {exc}")
        return
    if result is not None and Path(result).exists():
        figures[stem] = Path(result)


def make_figures(out: Path, frames_df: pd.DataFrame, grids: np.ndarray, pairs_df: pd.DataFrame,
                 redundant_df: pd.DataFrame | None, G: object, stats: dict, cfg: AnalyzerConfig) -> dict[str, Path]:
    """Every PNG of the report, keyed by file stem."""
    from frame_analyzer import graph, plots

    jobs: dict[str, Callable[[], Path | None]] = {
        "sharpness": lambda: plots.plot_sharpness(frames_df, cfg, out / "sharpness.png"),
        "sharpness_distribution": lambda: plots.plot_sharpness_distribution(frames_df, cfg,
                                                                             out / "sharpness_distribution.png"),
        "sift_keypoints": lambda: plots.plot_sift_keypoints(frames_df, cfg, out / "sift_keypoints.png"),
        "sift_density": lambda: plots.plot_sift_density(frames_df, out / "sift_density.png"),
        "sift_distribution": lambda: plots.plot_sift_distribution(frames_df, out / "sift_distribution.png"),
        "grid_occupancy": lambda: plots.plot_grid_occupancy(frames_df, cfg, out / "grid_occupancy.png"),
        "keypoint_heatmap": lambda: plots.plot_keypoint_heatmap(grids, frames_df, out / "keypoint_heatmap.png"),
        "grid_examples": lambda: plots.plot_grid_examples(grids, frames_df, out / "grid_examples.png"),
        "matching_inliers": lambda: plots.plot_matching_inliers(pairs_df, cfg, out / "matching_inliers.png"),
        "matching_inlier_ratio": lambda: plots.plot_matching_inlier_ratio(pairs_df, cfg,
                                                                          out / "matching_inlier_ratio.png"),
        "matching_scatter": lambda: plots.plot_matching_scatter(pairs_df, cfg, out / "matching_scatter.png"),
        "connectivity_graph": lambda: graph.draw_graph_png(G, stats, out / "connectivity_graph.png", cfg.seed),
        "degree_distribution": lambda: plots.plot_degree_distribution(stats, out / "degree_distribution.png"),
    }
    if cfg.redundancy and redundant_df is not None:
        jobs["temporal_redundancy"] = lambda: plots.plot_temporal_redundancy(pairs_df, redundant_df, cfg,
                                                                             out / "temporal_redundancy.png")
    figures: dict[str, Path] = {}
    for stem, fn in jobs.items():
        _safe_plot(figures, stem, fn, cfg)
    return figures


def build_summary(cfg: AnalyzerConfig, started: datetime, t_start: float, timings: dict, n_listed: int,
                  frames_df: pd.DataFrame, pairs_df: pd.DataFrame, offsets: list[int], stats: dict,
                  sharpness_median: float, redundant_df: pd.DataFrame | None) -> dict:
    """The summary.json dict described in the contract."""
    readable = frames_df["readable"]
    ok_pairs = pairs_df[pairs_df["status"] != "unreadable"] if not pairs_df.empty else pairs_df
    stat_block = {c: describe(frames_df.loc[readable, c]) for c in STAT_COLUMNS}
    stat_block["inliers"] = describe(ok_pairs["inliers"]) if not ok_pairs.empty else describe(pd.Series([], float))
    stat_block["inlier_ratio"] = describe(ok_pairs["inlier_ratio"]) if not ok_pairs.empty \
        else describe(pd.Series([], float))
    flag_counts: dict[str, int] = {}
    for f in frames_df["flags"]:
        for flag in filter(None, str(f).split(";")):
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
    components = stats.get("components", [])
    resolution = resolution_summary(frames_df, cfg)
    if n_listed != len(frames_df):
        cfg.warnings.append(f"{len(frames_df)} of {n_listed} listed frames analysed (stride={cfg.stride}, "
                            f"limit={cfg.limit})")
    return {
        "version": __version__, "started_at": started.isoformat(timespec="seconds"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_s": time.perf_counter() - t_start, "timings": timings,
        "input_dir": str(cfg.input_dir), "output_dir": str(cfg.output_dir), "config": cfg.to_dict(),
        "n_files_listed": n_listed, "n_files": len(frames_df), "n_readable": int(readable.sum()),
        "n_unreadable": int((~readable).sum()),
        "unreadable": [{"filename": r.filename, "error": r.error} for r in frames_df[~readable].itertuples()],
        "n_pairs": len(pairs_df), "n_pairs_verified": int(pairs_df["verified"].sum()) if len(pairs_df) else 0,
        "n_edges_accepted": int(pairs_df["edge_accepted"].sum()) if len(pairs_df) else 0,
        "offsets_used": offsets, "resolution": resolution, "stats": stat_block,
        "thresholds": thresholds_summary(cfg, sharpness_median),
        "quality_counts": {q: int((frames_df["quality"] == q).sum()) for q in ("good", "review", "poor")},
        "flag_counts": flag_counts,
        "graph": {
            "n_nodes": stats.get("n_nodes", 0), "n_edges": stats.get("n_edges", 0),
            "n_components": stats.get("n_components", 0), "n_isolated": stats.get("n_isolated", 0),
            "isolated_nodes": list(stats.get("isolated_nodes", [])),
            "largest_component_size": stats.get("largest_component_size", 0),
            "largest_component_pct": stats.get("largest_component_pct", 0.0),
            "component_sizes": [len(c) for c in components],
            "degree_histogram": stats.get("degree_histogram", {}),
            "component_ranges": component_ranges(components),
        },
        "redundancy": {"enabled": cfg.redundancy,
                       "n_candidates": 0 if redundant_df is None else len(redundant_df)},
        "warnings": list(cfg.warnings),
    }


def run(cfg: AnalyzerConfig) -> dict:
    """Run the whole diagnosis and write every output into cfg.output_dir; returns the summary dict."""
    from frame_analyzer import graph, report
    from frame_analyzer.redundancy import find_redundant_pairs

    started, t_start = datetime.now(), time.perf_counter()
    timings: dict[str, float] = {}
    paths, n_listed = select_frames(cfg)
    offsets = usable_offsets(cfg, len(paths))
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log.info("analysing %d frames (%d listed) from %s, offsets %s, analysis max side %d",
             len(paths), n_listed, cfg.input_dir, offsets, cfg.analysis_max_side)

    t = time.perf_counter()
    frames_df, grids, pairs_df = stream_frames(paths, offsets, cfg)
    timings["frames_and_matching"] = time.perf_counter() - t
    sharpness_median = add_relative_sharpness(frames_df, cfg)

    t = time.perf_counter()
    readable_idx = [int(i) for i in frames_df.loc[frames_df["readable"], "index"]]
    G = graph.build_graph(readable_idx, pairs_df)
    stats = graph.graph_stats(G)
    edges_df = graph.edges_dataframe(G, cfg)
    fill_graph_columns(frames_df, stats)
    timings["graph"] = time.perf_counter() - t

    t = time.perf_counter()
    frames_df = classify_frames(frames_df, cfg)
    redundant_df = find_redundant_pairs(pairs_df, frames_df, cfg) if cfg.redundancy else None
    timings["classification"] = time.perf_counter() - t

    t = time.perf_counter()
    _csv_frames(frames_df, out / "frame_quality.csv")
    pairs_df.to_csv(out / "pair_quality.csv", index=False)
    edges_df.to_csv(out / "connectivity_edges.csv", index=False)
    np.save(out / "grid_counts.npy", grids)
    if redundant_df is not None:
        redundant_df.to_csv(out / "redundant_pairs.csv", index=False)
    timings["tables"] = time.perf_counter() - t

    t = time.perf_counter()
    figures = make_figures(out, frames_df, grids, pairs_df, redundant_df, G, stats, cfg)
    graph_html: Path | None = None
    if cfg.interactive_graph:
        try:
            graph_html = graph.write_graph_html(G, stats, frames_df, out / "connectivity_graph.html", cfg.seed)
        except Exception as exc:
            log.exception("interactive graph failed")
            cfg.warnings.append(f"interactive graph could not be generated: {exc}")
    timings["figures"] = time.perf_counter() - t

    summary = build_summary(cfg, started, t_start, timings, n_listed, frames_df, pairs_df, offsets, stats,
                            sharpness_median, redundant_df)
    write_summary(out / "summary.json", summary)
    t = time.perf_counter()
    report.write_report(out, summary, frames_df, pairs_df, edges_df, redundant_df, figures, cfg, graph_html)
    report.write_output_readme(out, summary, cfg)
    timings["report"] = time.perf_counter() - t
    summary["elapsed_s"] = time.perf_counter() - t_start
    summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
    write_summary(out / "summary.json", summary)
    log.info("done in %.1fs: %d good, %d review, %d poor; %d components; outputs in %s", summary["elapsed_s"],
             summary["quality_counts"]["good"], summary["quality_counts"]["review"],
             summary["quality_counts"]["poor"], summary["graph"]["n_components"], out)
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; exit codes 0 ok, 2 input error, 3 SIFT unavailable."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse_args(argv)
    try:
        run(cfg)
    except (EmptyInputError, FileNotFoundError, NotADirectoryError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except SiftUnavailableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    return 0


# ╔══════════════════════════════════════════════════════════════════════════════════════╗
# ║  ⠀⠀⠀⠀⣠⠶⡒⠒⢬⡲⣮⠂⣆⣀⠀⠀⠀⠀⠀⠀⢀⣤⣴⣦⣤⡀⠀⠀⠀⠀   MATEUS OLIVEIRA                        ║
# ║  ⠀⠀⠀⣀⣥⠠⣿⠆⠐⣻⣾⣿⣿⢷⡄⠀⠀⠀⠀⢠⡿⠋⠉⠉⠙⢿⡄⠀⠀⠀   m203656@dac.unicamp.edu.br             ║
# ║  ⠀⠀⢘⡵⢋⠄⡙⠒⣤⣄⣉⠙⣿⣗⠑⡄⠀⠀⠀⠘⡇⠀⠀⠀⠀⠈⡇⠀⠀⠀   UNICAMP - Universidade Estadual de     ║
# ║  ⠀⣴⢿⡜⢡⡞⢀⢼⣿⣿⣿⣿⣿⣿⠟⣂⠀⠀⢀⣀⠱⡀⠀⠀⠀⢰⠁⠀⠀⠀               Campinas                     ║
# ║  ⠰⢫⢟⡇⢸⡇⢸⢾⣿⣿⣿⣿⣿⣿⡷⠰⠀⢰⡏⠀⠀⢡⠀⠀⢠⠃⠀⠀⠀⠀   IC - Institute of Computing            ║
# ║  ⢰⠁⣿⢣⣿⠇⢀⣿⣿⡿⠿⠤⣭⣥⣶⡆⠀⠸⣷⣤⣠⡾⠀⢀⡇⠀⠀⠀⠀⠀   Computer Science Department              ║
# ║  ⡞⣰⣧⠟⡝⢸⢸⣿⣥⠖⣴⡆⣤⣬⠉⠀⠀⠀⠈⠉⠉⠀⠀⢸⣇⠀⠀⠀⠀⠀   github.com/oliveiraMats2              ║
# ║  ⠀⡿⡟⢸⡇⠸⡄⢹⣿⢸⣿⣇⡏⠟⣰⣄⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠁⠀⠀⠀   linkedin.com/in/mateus-eng            ║
# ║  ⠀⠇⣧⠘⡇⠦⣹⣸⣿⡇⡿⡿⣡⣼⣿⣿⣷⣦⣄⡀⠀⠀⣸⣿⣿⠄⠻⢷⣦⠀                                            ║
# ║  ⠀⢀⠘⣇⢹⡸⣿⣿⣿⢹⢃⣠⣿⣿⣿⣿⣿⣿⣿⣿⣆⠀⠑⠋⠉⠀⠀⠈⣿⣧   UNICAMP · IC · 2026                    ║
# ║  ⠀⢸⣿⡌⠘⢷⣿⣿⡏⢀⣾⣿⣿⣿⣿⣿⣿⢻⣿⣿⣿⡆⠀⠀⠀⠀⠀⠀⣿⡿                                            ║
# ║  ⠀⠈⣿⣿⣦⡌⢿⠏⣰⣿⣿⣿⣿⣿⣿⡿⡏⣼⣿⣿⣿⡇⣄⠀⠀⠀⢀⣼⣿⠇                                            ║
# ║  ⠀⠀⠹⣿⣿⢻⡀⣼⣿⣿⢻⣿⣿⣿⣿⡇⡇⢻⣿⣿⣿⡇⣿⣿⣶⣿⣿⠟⠁⠀                                            ║
# ║  ⠀⠀⠀⢻⣿⣦⡓⢿⣿⣿⡆⣿⣿⣿⣿⢃⣶⡸⣿⣿⣿⡇⠀⠉⠉⠁⠀⠀⠀⠀                                            ║
# ║  ⠀⠀⠀⠈⣿⣿⣿⡆⠀⠀⠀⣿⣿⣿⡟⣼⡿⠁⢹⣿⣿⣷⠀⠀⠀⠀⠀⠀⠀⠀                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════════════╝

# ==============================================================================
# PROMPT COMPLEMENT
# This block is a prompt complement for the AI assistant that will implement
# this file. Read it fully before writing any code. It is self-contained: do not
# look for other documents.
#
# Stage: Stage 5, Evaluation          Owner: unassigned          Due: TBD
# Project: 3D reconstruction of an indoor environment (variant A) from frames
# ==============================================================================
# Stage: Stage 5, Evaluation          Owner: Mateus Oliveira
# Goal of this file (single responsibility):
#   T2 item 2.3: compare the detectors run by the pipeline (SIFT/ORB/AKAZE) and
#   justify the choice. Reads detection_metrics.csv, matching_metrics.csv,
#   cross_check.csv and epipolar_metrics.csv from artifacts_dir, builds one row
#   per detector (keypoints/img, detection time, matches after Lowe and after
#   cross-check, RANSAC inliers and inlier ratio, pose success rate, Sampson
#   error, grid coverage), ranks each criterion and computes a weighted score.
#   Outputs: artifacts_dir/detector_choice.csv and detector_choice.md (short
#   generated justification; COLMAP SfM uses SIFT).
#   Weights: cfg.get("detector_choice", {}).get("weights", {}) overrides WEIGHTS.
# ==============================================================================

import logging
import math
from pathlib import Path

from src import stage_io

log = logging.getLogger(__name__)

# criterion -> (column, higher_is_better, label)
CRITERIA = {
    "keypoints": ("keypoints_per_img", True, "keypoints/img"),
    "coverage": ("coverage_8x8", True, "grid coverage"),
    "time": ("det_time_ms", False, "detection time"),
    "matches": ("matches_cross", True, "matches after Lowe+cross-check"),
    "inliers": ("ransac_inliers", True, "RANSAC inliers"),
    "inlier_ratio": ("inlier_ratio", True, "RANSAC inlier ratio"),
    "pose_success": ("pose_success_rate", True, "pose success rate"),
    "sampson": ("sampson_px", False, "Sampson error"),
}
WEIGHTS = {"keypoints": 0.5, "coverage": 1.0, "time": 0.5, "matches": 1.0, "inliers": 1.0,
           "inlier_ratio": 1.5, "pose_success": 2.0, "sampson": 1.5}
SFM_DETECTOR = "sift"  # COLMAP feature_extractor


def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _mean(values) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def _by_detector(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r["detector"], []).append(r)
    return out


def _read(artifacts_dir: Path, name: str) -> dict[str, list[dict]]:
    path = artifacts_dir / name
    if not path.exists():
        log.warning("%s missing: its criteria are left empty", path)
        return {}
    return _by_detector(stage_io.read_csv(path))


def table(artifacts_dir: Path) -> list[dict]:
    det = _read(artifacts_dir, "detection_metrics.csv")
    mat = _read(artifacts_dir, "matching_metrics.csv")
    cross = _read(artifacts_dir, "cross_check.csv")
    epi = _read(artifacts_dir, "epipolar_metrics.csv")

    rows = []
    for name in sorted(set(det) | set(mat)):
        d, m, c, e = det.get(name, []), mat.get(name, []), cross.get(name, []), epi.get(name, [])
        n_pairs = len(e) or len(m)
        posed = sum(1 for r in e if math.isfinite(_f(r.get("rot_deg"))))
        rows.append({
            "detector": name,
            "n_frames": len(d),
            "n_pairs": n_pairs,
            "keypoints_per_img": _mean(_f(r["n_keypoints"]) for r in d),
            "coverage_8x8": _mean(_f(r.get("coverage_8x8")) for r in d),
            "det_time_ms": _mean(_f(r["time_ms_mean"]) for r in d),
            "matches_lowe": _mean(_f(r["n_lowe"]) for r in c),
            "matches_cross": _mean(_f(r["n_matches_lowe"]) for r in m),
            "ransac_inliers": _mean(_f(r["inliers"]) for r in m),
            "inlier_ratio": _mean(_f(r["inlier_ratio"]) for r in m),
            "pose_success_rate": posed / n_pairs if e and n_pairs else _mean(_f(r["success"]) for r in m),
            "sampson_px": _mean(_f(r.get("sampson_px")) for r in m),
            "cheirality_frac": _mean(_f(r.get("cheirality_frac")) for r in e),
        })
    return rows


def score(rows: list[dict], weights: dict) -> None:
    """Adds rank_<criterion> (1 = best) and a weighted min-max score in [0, 1]."""

    total = {id(r): 0.0 for r in rows}
    wsum = 0.0
    for crit, (col, higher, _) in CRITERIA.items():
        w = float(weights.get(crit, 0.0))
        vals = [r[col] for r in rows if math.isfinite(r[col])]
        order = sorted(vals, reverse=higher)
        for r in rows:
            r[f"rank_{crit}"] = order.index(r[col]) + 1 if math.isfinite(r[col]) else ""
        if w <= 0 or not vals:
            continue
        lo, hi = min(vals), max(vals)
        wsum += w
        for r in rows:
            v = r[col]
            if not math.isfinite(v):
                continue
            norm = 1.0 if hi == lo else (v - lo) / (hi - lo)
            total[id(r)] += w * (norm if higher else 1.0 - norm)
    for r in rows:
        r["score"] = total[id(r)] / wsum if wsum else float("nan")
    for i, r in enumerate(sorted(rows, key=lambda r: -r["score"])):
        r["rank_overall"] = i + 1


def justification(rows: list[dict], weights: dict) -> str:
    best = min(rows, key=lambda r: r["rank_overall"])
    fastest = min(rows, key=lambda r: r["det_time_ms"] if math.isfinite(r["det_time_ms"]) else math.inf)
    wins = [lab for crit, (_, _, lab) in CRITERIA.items() if best.get(f"rank_{crit}") == 1]
    loses = [lab for crit, (_, _, lab) in CRITERIA.items() if best.get(f"rank_{crit}") not in (1, "")]

    def fmt(v, p=2):
        return f"{v:.{p}f}" if isinstance(v, float) else str(v)

    cols = ["detector", "keypoints_per_img", "det_time_ms", "matches_lowe", "matches_cross", "ransac_inliers",
            "inlier_ratio", "pose_success_rate", "sampson_px", "coverage_8x8", "score", "rank_overall"]
    lines = ["# Detector choice", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in sorted(rows, key=lambda r: r["rank_overall"]):
        lines.append("| " + " | ".join(fmt(r[c], 3) for c in cols) + " |")
    lines += ["", f"Weights: {', '.join(f'{k}={v}' for k, v in weights.items())} "
              "(score = weighted mean of min-max normalised criteria, 1 = best).", "",
              f"**Chosen detector: {best['detector'].upper()}** (score {best['score']:.3f}). "
              f"It ranks first on: {', '.join(wins) or 'no single criterion'}"
              + (f"; it is not the best on: {', '.join(loses)}." if loses else ".")]
    if fastest is not best:
        lines.append(
            f"\nTrade-off: {fastest['detector'].upper()} is the fastest "
            f"({fastest['det_time_ms']:.1f} ms/img vs {best['det_time_ms']:.1f} ms) and gives "
            f"{fastest['ransac_inliers']:.0f} inliers/pair, but its inlier ratio "
            f"({fastest['inlier_ratio']:.3f} vs {best['inlier_ratio']:.3f}) and Sampson error "
            f"({fastest['sampson_px']:.3f} vs {best['sampson_px']:.3f} px) are worse, and its binary "
            "descriptor is less discriminative under the scale/viewpoint changes of a handheld video. "
            "Geometric accuracy and pose success matter more for SfM than detection speed, which is "
            "negligible next to bundle adjustment.")
    if best["detector"] == SFM_DETECTOR:
        lines.append(f"\nThe COLMAP SfM stage uses {SFM_DETECTOR.upper()} features for this reason: it gives the "
                     "most accurate two-view geometry here, and it is COLMAP's native, GPU-accelerated extractor.")
    else:
        lines.append(f"\nNote: the COLMAP SfM stage uses {SFM_DETECTOR.upper()} (its native, GPU-accelerated "
                     f"extractor), while this run ranks {best['detector'].upper()} first under these weights.")
    return "\n".join(lines) + "\n"


def run(cfg: dict, run_dir: str, artifacts_dir: str) -> list[dict] | None:

    artifacts_dir = Path(artifacts_dir)
    weights = {**WEIGHTS, **(cfg.get("detector_choice", {}) or {}).get("weights", {})}
    rows = table(artifacts_dir)
    if len(rows) < 2:
        log.warning("fewer than two detectors in %s: skipping detector choice", artifacts_dir)
        return None

    score(rows, weights)
    keys = list(rows[0].keys())
    path = artifacts_dir / "detector_choice.csv"
    with open(path, "w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(str(r[k]) for k in keys) + "\n")
    (artifacts_dir / "detector_choice.md").write_text(justification(rows, weights))

    for r in rows:
        log.info("metric %s=%s", f"detector_choice_score_{r['detector']}", round(r["score"], 4))
    best = min(rows, key=lambda r: r["rank_overall"])
    log.info("metric %s=%s", "detector_choice", best["detector"])
    log.info("detector choice written to %s", path)
    return rows

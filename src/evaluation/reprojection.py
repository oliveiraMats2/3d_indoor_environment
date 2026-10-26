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
#          extracted from a phone video.
# Context: final metrics of the reconstruction, required for every group, plus
# the indoor-environment (variant A) metrics. Each result is one row in a CSV
# under artifacts/<run_id>/ and the partial values are saved as .npy.
#
# Goal of this file (single responsibility):
#   Reprojection error (mean and median), number of sparse and dense points,
#   and density (points per square meter, approximate).
#
# What to implement:
#   - Project every 3D point with each camera that observes it (K, R, t) and
#     measure the pixel distance to the observed keypoint.
#   - Report mean and median reprojection error in pixels.
#   - Count sparse points and dense points (0 if no dense cloud exists).
#   - Density: points divided by the approximate covered floor area in m^2.
#
# Inputs / outputs:
#   Inputs: 3D points (N, 3), observations, K, R, t from results/<run_id>/.
#   Outputs under results/<run_id>/evaluation/: reproj_errors.npy (M,).
#   One row in artifacts/<run_id>/evaluation.csv: reproj_mean_px,
#   reproj_median_px, n_sparse, n_dense, density_pts_m2.
#
# Defaults (read from the YAML block evaluation, never hardcoded):
#   enabled: false
#
# Rules:
#   - Write code ONLY between the BEGIN and END markers below. Do not create
#     other regions or other files for this task.
#   - Configuration comes from the YAML loaded by main.py. No argparse.
#   - Save every partial result as .npy under results/<run_id>/<stage>/.
#   - Keep this block after implementing; update it if the behavior changes.
#   - Expose a function run(cfg: dict, run_dir: str, ...) that main.py can call
#     when the YAML block of this stage has enabled: true (default false).
#   - Pipeline already available: frames are extracted and quality-filtered, and
#     SIFT, ORB and AKAZE keypoints/descriptors are saved per frame under
#     results/<run_id>/detection/<detector>/<frame>/ as keypoints.npy (N, 6:
#     x, y, size, angle, response, octave) and descriptors.npy (N, D).
# ==============================================================================

# >>> BEGIN IMPLEMENTATION >>>
"""Reprojection and point-count metrics for global SfM or current two-view clouds."""
import logging
from pathlib import Path

import cv2
import numpy as np

from src import stage_io

log = logging.getLogger(__name__)


def project(points: np.ndarray, R: np.ndarray, t: np.ndarray, K: np.ndarray,
            dist: np.ndarray | None = None) -> np.ndarray:
    """Project one 3D point per camera pose; return NaN for points behind a camera."""

    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    R = np.broadcast_to(np.asarray(R, dtype=np.float64), (len(points), 3, 3))
    t = np.broadcast_to(np.asarray(t, dtype=np.float64).reshape(-1, 3), (len(points), 3))
    xc = np.einsum("nij,nj->ni", R, points) + t
    with np.errstate(divide="ignore", invalid="ignore"):
        x = xc[:, 0] / xc[:, 2]
        y = xc[:, 1] / xc[:, 2]

    coeffs = list(np.ravel(np.zeros(4) if dist is None else dist)) + [0.0] * 4
    k1, k2, p1, p2 = coeffs[:4]
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    uv = np.column_stack([
        K[0, 0] * xd + K[0, 1] * yd + K[0, 2],
        K[1, 1] * yd + K[1, 2],
    ])
    uv[xc[:, 2] <= 0] = np.nan
    return uv


def reprojection_errors(points3d, observations, K, dist, camera_R, camera_t) -> np.ndarray:
    """Pixel error for observations stored as point_index, camera_index, x, y."""

    observations = np.asarray(observations, dtype=np.float64).reshape(-1, 4)
    if not len(observations):
        return np.empty(0, dtype=np.float64)
    point_index = observations[:, 0].astype(int)
    camera_index = observations[:, 1].astype(int)
    uv = project(np.asarray(points3d)[point_index], np.asarray(camera_R)[camera_index],
                 np.asarray(camera_t)[camera_index], np.asarray(K), np.asarray(dist))
    return np.linalg.norm(uv - observations[:, 2:4], axis=1)


def _global_sfm(root: Path) -> tuple[np.ndarray, np.ndarray] | None:
    sfm = root / "sfm"
    names = ("points3d", "obs", "K", "dist", "cam_R", "cam_t")
    paths = [sfm / f"{name}.npy" for name in names]
    if not all(path.exists() for path in paths):
        return None
    points, obs, K, dist, camera_R, camera_t = (np.load(path) for path in paths)
    return points, reprojection_errors(points, obs, K, dist, camera_R, camera_t)


def _pair_errors(root: Path, pair_dir: Path, detector: str) -> tuple[np.ndarray, int] | None:
    """Rebuild the observations retained by two-view triangulation and evaluate them."""

    required = ("points3d.npy", "K.npy", "R.npy", "t.npy", "pose_mask.npy")
    if not all((pair_dir / name).exists() for name in required):
        return None
    first, second = pair_dir.name.split("__", maxsplit=1)
    matching = root / "matching" / detector / pair_dir.name / "matches_lowe.npy"
    kp1 = root / "detection" / detector / first / "keypoints.npy"
    kp2 = root / "detection" / detector / second / "keypoints.npy"
    if not matching.exists() or not kp1.exists() or not kp2.exists():
        return None

    points = np.load(pair_dir / "points3d.npy")
    matches = np.load(matching)
    mask = np.load(pair_dir / "pose_mask.npy").astype(bool)
    if len(mask) != len(matches) or not len(points):
        return None
    p1 = np.load(kp1)[matches[:, 0].astype(int), :2][mask]
    p2 = np.load(kp2)[matches[:, 1].astype(int), :2][mask]
    K, R, t = (np.load(pair_dir / f"{name}.npy") for name in ("K", "R", "t"))

    # Recreate the cheirality filter used by triangulate.py so observations stay aligned.
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t.reshape(3, 1)])
    homogeneous = cv2.triangulatePoints(P1, P2, p1.T, p2.T)
    with np.errstate(divide="ignore", invalid="ignore"):
        rebuilt = (homogeneous[:3] / homogeneous[3]).T
    keep = (np.isfinite(rebuilt).all(axis=1) & (rebuilt[:, 2] > 0)
            & ((rebuilt @ R.T + t.reshape(1, 3))[:, 2] > 0))
    if int(keep.sum()) != len(points):
        log.warning("reprojection %s/%s: observations and points do not align; skipped", detector, pair_dir.name)
        return None

    p1, p2 = p1[keep], p2[keep]
    uv1 = project(points, np.eye(3), np.zeros(3), K)
    uv2 = project(points, R, t, K)
    errors = np.concatenate([np.linalg.norm(uv1 - p1, axis=1), np.linalg.norm(uv2 - p2, axis=1)])
    return errors, len(points)


def _two_view(root: Path) -> list[tuple[str, np.ndarray, int]]:
    rows = []
    epipolar = root / "epipolar"
    if not epipolar.is_dir():
        return rows
    for detector_dir in sorted(path for path in epipolar.iterdir() if path.is_dir()):
        errors, count = [], 0
        for pair_dir in sorted(path for path in detector_dir.iterdir() if path.is_dir()):
            result = _pair_errors(root, pair_dir, detector_dir.name)
            if result is not None:
                pair_errors, n_points = result
                errors.append(pair_errors)
                count += n_points
        if errors:
            rows.append((detector_dir.name, np.concatenate(errors), count))
    return rows


def _dense_count(root: Path) -> int:
    counts = [len(np.load(path, mmap_mode="r")) for path in sorted((root / "dense").glob("*/points.npy"))]
    return max(counts, default=0)


def _floor_area(artifacts: Path) -> float:
    rows = stage_io.read_csv(artifacts / "floorplan.csv")
    try:
        area = float(rows[-1]["area_m2"]) if rows else float("nan")
    except (KeyError, TypeError, ValueError):
        return float("nan")
    return area if np.isfinite(area) and area > 0 else float("nan")


def run(cfg: dict, run_dir: str, artifacts_dir: str) -> list[dict] | None:
    if not cfg["evaluation"]["enabled"]:
        return None

    root, artifacts = Path(run_dir), Path(artifacts_dir)
    out = root / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    csv_path = artifacts / "evaluation.csv"
    csv_path.unlink(missing_ok=True)

    data = _global_sfm(root)
    groups = [("global", data[1], len(data[0]))] if data is not None else _two_view(root)
    if not groups:
        log.warning("reprojection: no global SfM arrays or two-view clouds found")
        return None

    rows, all_errors = [], []
    dense_count = _dense_count(root) if data is not None else 0
    area_m2 = _floor_area(artifacts)
    for detector, errors, n_sparse in groups:
        valid = errors[np.isfinite(errors)]
        point_count = dense_count or n_sparse
        density = point_count / area_m2 if np.isfinite(area_m2) else ""
        row = {
            "detector": detector,
            "source": "sfm" if detector == "global" else "two_view",
            "reproj_mean_px": float(valid.mean()) if len(valid) else "",
            "reproj_median_px": float(np.median(valid)) if len(valid) else "",
            "n_sparse": int(n_sparse),
            "n_dense": dense_count,
            "density_pts_m2": density,
        }
        stage_io.upsert_csv(csv_path, row, key="detector")
        stage_io.log_metrics(log, {f"{detector}.{key}": value for key, value in row.items()
                                   if key not in ("detector", "source")})
        np.save(out / f"reproj_errors_{detector}.npy", errors)
        rows.append(row)
        all_errors.append(errors)

    np.save(out / "reproj_errors.npy", np.concatenate(all_errors))
    return rows

# <<< END IMPLEMENTATION <<<

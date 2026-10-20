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

# PROMPT COMPLEMENT
# This block is a prompt complement for the AI assistant that will implement
# this file. Read it fully before writing any code. It is self-contained: do not
# look for other documents.
#
# Stage: 3, part 2 (Epipolar geometry)          Owner: Maxime          Due: Fri 23/10
# Project: 3D reconstruction of an indoor environment (variant A) from frames
#          extracted from a phone video.
# Context (plain words): a point in frame 1 only gives a direction, like a laser
# beam leaving camera 1 through that pixel. Seen from camera 2 the beam is a
# straight line, the epipolar line; the correct match of the point must lie on
# it. F (fundamental matrix) states this in pixels: x2^T F x1 = 0. E (essential
# matrix) is the same rule after removing the lens with the intrinsics K:
# E = K^T F K, and it keeps only the motion between the cameras.
# RANSAC: sample a few pairs (8 for F, 5 for E), fit, count how many other
# pairs fall close to their epipolar lines, repeat, keep the model with the
# most votes. Voters are inliers; the rest are discarded outliers.
# t comes out only as a direction, without size: metric scale is applied later
# from an object of known size in the scene.
# Input: the Lowe-filtered pairs from the matching package, plus K from the YAML.
# The owner also writes the second part of the Stage 3 report section.
#
# Goal of this file (single responsibility):
#   triangulate an initial 3D point cloud from the inliers.
#
# What to implement:
#   - Projection matrices P1 = K [I | 0], P2 = K [R | t].
#   - cv2.triangulatePoints on the inliers, convert from homogeneous.
#   - The two rays almost never cross exactly (noise); the result is the point
#     closest to both. Drop points behind either camera.
#   - Export the cloud as .ply for viewing (Open3D).
#
# Inputs / outputs:
#   Signature (called by two_view, cfg = geometry block):
#     run(cfg, run_dir, pair_dir, K, R, t, pts1, pts2, mask, image1_bgr) -> dict
#   In: K (3, 3), R (3, 3), t (3, 1), pts1, pts2 (M, 2) = all Lowe matches,
#   mask (M,) bool (pose_mask: rows to triangulate), frame 1 (BGR) for colors.
#   Out: points3d.npy (N, 3) float64, colors.npy (N, 3) uint8 RGB and cloud.ply
#   under results/<run_id>/epipolar/<detector>/<frameA>__<frameB>/ (= pair_dir).
#   Returns {n_points3d, parallax_med_deg, cheirality_frac}; cheirality_frac =
#   kept / triangulated (NaN if nothing triangulated). The CSV is written by
#   two_view, not here. Empty input -> empty arrays saved, 0 / NaN metrics.
#
# YAML keys: none of its own (K comes already built from geometry.intrinsics).
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
import logging
from pathlib import Path

import cv2
import numpy as np

from src import stage_io

log = logging.getLogger(__name__)


def _parallax_deg(points: np.ndarray, center2: np.ndarray) -> np.ndarray:
    """Angle between the rays camera1 -> X and camera2 -> X (camera 1 at the origin)."""

    ray1 = points
    ray2 = points - center2
    cos = np.sum(ray1 * ray2, axis=1) / (np.linalg.norm(ray1, axis=1) * np.linalg.norm(ray2, axis=1))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def run(cfg: dict, run_dir: str, pair_dir: Path, K, R, t, pts1, pts2, mask, image1_bgr) -> dict:
    """Triangulate the masked matches with P1 = K[I|0], P2 = K[R|t]; keep points in front of both cameras."""

    pair_dir = Path(pair_dir)
    pair_dir.mkdir(parents=True, exist_ok=True)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3, 1)
    pts1 = np.asarray(pts1, dtype=np.float64).reshape(-1, 2)
    pts2 = np.asarray(pts2, dtype=np.float64).reshape(-1, 2)
    mask = np.asarray(mask).reshape(-1).astype(bool)

    points = np.empty((0, 3), dtype=np.float64)
    colors = np.empty((0, 3), dtype=np.uint8)
    n_tri = 0
    parallax_med = float("nan")

    if len(mask) == len(pts1) == len(pts2) and mask.any() and np.isfinite(np.hstack([K, R, t])).all():
        x1, x2 = pts1[mask], pts2[mask]
        n_tri = len(x1)
        P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = K @ np.hstack([R, t])
        X_h = cv2.triangulatePoints(P1, P2, x1.T, x2.T)
        with np.errstate(divide="ignore", invalid="ignore"):
            X = (X_h[:3] / X_h[3]).T
        depth1 = X[:, 2]
        depth2 = (X @ R.T + t.T)[:, 2]
        keep = np.isfinite(X).all(axis=1) & (depth1 > 0) & (depth2 > 0)
        points, kept_px = X[keep], x1[keep]

        if len(points):
            parallax_med = float(np.median(_parallax_deg(points, (-R.T @ t).ravel())))
            if image1_bgr is not None:
                h, w = image1_bgr.shape[:2]
                u = np.clip(np.round(kept_px[:, 0]).astype(int), 0, w - 1)
                v = np.clip(np.round(kept_px[:, 1]).astype(int), 0, h - 1)
                bgr = np.asarray(image1_bgr)[v, u]
                bgr = np.repeat(bgr[:, None], 3, axis=1) if bgr.ndim == 1 else bgr[:, :3]
                colors = bgr[:, ::-1].astype(np.uint8)
            else:
                colors = np.full((len(points), 3), 255, dtype=np.uint8)
    elif mask.any():
        log.warning("triangulate: invalid inputs for %s (sizes or NaN pose), nothing triangulated", pair_dir.name)

    np.save(pair_dir / "points3d.npy", points)
    np.save(pair_dir / "colors.npy", colors)
    stage_io.write_ply(pair_dir / "cloud.ply", points, colors)

    metrics = {
        "n_points3d": int(len(points)),
        "parallax_med_deg": parallax_med,
        "cheirality_frac": float(len(points) / n_tri) if n_tri else float("nan"),
    }
    stage_io.log_metrics(log, metrics)
    return metrics


if __name__ == "__main__":
    import tempfile

    import open3d as o3d

    rng = np.random.default_rng(0)
    K = np.array([[800.0, 0, 320], [0, 800.0, 240], [0, 0, 1]])
    angle = np.radians(5.0)
    R = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    t = np.array([[-1.0], [0.0], [0.0]])

    front = np.column_stack([rng.uniform(-1, 1, 60), rng.uniform(-1, 1, 60), rng.uniform(4, 8, 60)])
    behind = np.column_stack([rng.uniform(-1, 1, 5), rng.uniform(-1, 1, 5), rng.uniform(-8, -4, 5)])
    X = np.vstack([front, behind])

    def project(P, pts):
        x = (P @ np.column_stack([pts, np.ones(len(pts))]).T).T
        return x[:, :2] / x[:, 2:]

    pts1 = project(K @ np.hstack([np.eye(3), np.zeros((3, 1))]), X)
    pts2 = project(K @ np.hstack([R, t]), X)
    mask = np.ones(len(X), dtype=bool)
    mask[0] = False  # one masked-out row must be ignored
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[..., 0] = 200  # blue in BGR

    with tempfile.TemporaryDirectory() as tmp:
        pair = Path(tmp) / "epipolar" / "sift" / "img_0000__img_0001"
        m = run({}, tmp, pair, K, R, t, pts1, pts2, mask, image)
        pts3d = np.load(pair / "points3d.npy")
        cols = np.load(pair / "colors.npy")
        assert m["n_points3d"] == 59 == len(pts3d) == len(cols), m
        assert np.abs(pts3d - front[1:]).max() < 1e-6
        assert abs(m["cheirality_frac"] - 59 / 64) < 1e-9
        assert 0 < m["parallax_med_deg"] < 30
        assert cols.dtype == np.uint8 and (cols == [0, 0, 200]).all()
        pc = o3d.io.read_point_cloud(str(pair / "cloud.ply"))
        assert len(pc.points) == 59

        empty = run({}, tmp, Path(tmp) / "empty", K, R, t, np.empty((0, 2)), np.empty((0, 2)),
                    np.empty(0, dtype=bool), image)
        assert empty["n_points3d"] == 0 and np.isnan(empty["cheirality_frac"])
        assert np.load(Path(tmp) / "empty" / "points3d.npy").shape == (0, 3)
        nan_pose = run({}, tmp, Path(tmp) / "nan", K, np.full((3, 3), np.nan), t, pts1, pts2, mask, image)
        assert nan_pose["n_points3d"] == 0
    print("OK")
# <<< END IMPLEMENTATION <<<

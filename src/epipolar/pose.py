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
#   recover the rotation R and the translation t of camera 2 relative to camera 1.
#
# What to implement:
#   - cv2.recoverPose(E, pts1, pts2, K). E has four decompositions; the right
#     one is the only one that puts the points in front of both cameras
#     (recoverPose does this cheirality check).
#   - t has unit norm (direction only).
#
# Inputs / outputs:
#   Signature: run(cfg, run_dir, pair_dir, E, pts1, pts2, K, inlier_mask) -> dict
#     cfg = geometry block (no keys of its own); called per pair by two_view.py.
#   In: E (3, 3); pts1, pts2 (M, 2) ALL Lowe matches; K (3, 3); inlier_mask (M,)
#     bool from estimate.py (aligned with matches_lowe rows).
#   Out: R.npy (3,3), t.npy (3,1) unit norm, pose_mask.npy (M,) bool (inlier AND
#   passed the cheirality check) under
#   results/<run_id>/epipolar/<detector>/<frameA>__<frameB>/ (= pair_dir)
#   Returns {rot_deg, t_x, t_y, t_z, n_pose_inliers}; all NaN (and nothing saved)
#   when E has NaN or fewer than 5 inliers. CSV is written by two_view.py.
#
# Defaults (read from the YAML block geometry, never hardcoded):
#   intrinsics: {fx: null, fy: null, cx: null, cy: null}  # null fx/fy -> max(w, h); null cx/cy -> w/2, h/2
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

log = logging.getLogger(__name__)

MIN_POINTS = 5  # minimal sample of the essential matrix, not a tunable


def _nan_result() -> dict:
    return {k: float("nan") for k in ("rot_deg", "t_x", "t_y", "t_z", "n_pose_inliers")}


def run(cfg: dict, run_dir: str, pair_dir: Path, E, pts1, pts2, K, inlier_mask) -> dict:
    """Recover R, t (camera 2 relative to camera 1) from E with the cheirality check."""

    pair_dir = Path(pair_dir)
    E = np.asarray(E, dtype=np.float64)
    pts1 = np.asarray(pts1, dtype=np.float64).reshape(-1, 2)
    pts2 = np.asarray(pts2, dtype=np.float64).reshape(-1, 2)
    K = np.asarray(K, dtype=np.float64)
    inlier_mask = np.asarray(inlier_mask, dtype=bool).reshape(-1)

    if E.shape != (3, 3) or not np.all(np.isfinite(E)) or inlier_mask.sum() < MIN_POINTS:
        log.warning("pose %s: skipped (E finite=%s, inliers=%d, min=%d)", pair_dir.name,
                    bool(E.shape == (3, 3) and np.all(np.isfinite(E))), int(inlier_mask.sum()), MIN_POINTS)
        return _nan_result()

    mask_in = inlier_mask.astype(np.uint8).reshape(-1, 1) * 255
    n, R, t, mask_out = cv2.recoverPose(E, pts1, pts2, K, mask=mask_in)
    t_norm = float(np.linalg.norm(t))
    if n < 1 or not np.isfinite(t_norm) or t_norm == 0:
        log.warning("pose %s: recoverPose found no point in front of both cameras", pair_dir.name)
        return _nan_result()

    t = (t / t_norm).reshape(3, 1)
    pose_mask = (mask_out.reshape(-1) > 0) & inlier_mask

    pair_dir.mkdir(parents=True, exist_ok=True)
    np.save(pair_dir / "R.npy", R)
    np.save(pair_dir / "t.npy", t)
    np.save(pair_dir / "pose_mask.npy", pose_mask)

    rot_deg = float(np.degrees(np.linalg.norm(cv2.Rodrigues(R)[0])))
    result = {"rot_deg": rot_deg, "t_x": float(t[0, 0]), "t_y": float(t[1, 0]), "t_z": float(t[2, 0]),
              "n_pose_inliers": int(pose_mask.sum())}
    log.debug("pose %s: %s", pair_dir.name, result)
    return result


def _skew(v) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=np.float64).reshape(3)
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


if __name__ == "__main__":
    import tempfile

    rng = np.random.default_rng(0)
    K = np.array([[1000.0, 0, 640], [0, 1000.0, 360], [0, 0, 1]])
    R_gt = cv2.Rodrigues(np.array([0.05, -0.1, 0.02]))[0]
    t_gt = np.array([[1.0], [0.2], [0.1]]); t_gt /= np.linalg.norm(t_gt)
    X = np.c_[rng.uniform(-2, 2, (200, 2)), rng.uniform(4, 8, 200)]
    x1 = (K @ X.T).T; x1 = x1[:, :2] / x1[:, 2:]
    X2 = (R_gt @ X.T + t_gt).T
    x2 = (K @ X2.T).T; x2 = x2[:, :2] / x2[:, 2:]
    E_gt = _skew(t_gt) @ R_gt
    mask = np.ones(len(X), bool); mask[:10] = False

    with tempfile.TemporaryDirectory() as tmp:
        pair = Path(tmp) / "epipolar" / "akaze" / "a__b"
        res = run({}, tmp, pair, E_gt, x1, x2, K, mask)
        R = np.load(pair / "R.npy"); t = np.load(pair / "t.npy"); pm = np.load(pair / "pose_mask.npy")
        err_deg = np.degrees(np.linalg.norm(cv2.Rodrigues(R @ R_gt.T)[0]))
        print("synthetic:", res, "R err deg=%.2e" % err_deg, "t dot=%.6f" % float(t.ravel() @ t_gt.ravel()))
        assert err_deg < 1e-3 and float(t.ravel() @ t_gt.ravel()) > 1 - 1e-6
        assert t.shape == (3, 1) and abs(np.linalg.norm(t) - 1) < 1e-9
        assert pm.shape == (len(X),) and pm.dtype == bool and not pm[:10].any() and res["n_pose_inliers"] == 190

        bad = Path(tmp) / "bad"
        r_nan = run({}, tmp, bad, np.full((3, 3), np.nan), x1, x2, K, mask)
        r_few = run({}, tmp, bad, E_gt, x1, x2, K, np.arange(len(X)) < 4)
        assert all(np.isnan(v) for v in r_nan.values()) and all(np.isnan(v) for v in r_few.values())
        assert not bad.exists()

        root = Path(__file__).resolve().parents[2] / "results" / "default"
        pairs = sorted((root / "epipolar" / "akaze").glob("*__*")) if (root / "epipolar" / "akaze").exists() else []
        if pairs and (pairs[0] / "E.npy").exists():
            p = pairs[0]; a, b = p.name.split("__")
            m = np.load(root / "matching" / "akaze" / p.name / "matches_lowe.npy")
            kp1 = np.load(root / "detection" / "akaze" / a / "keypoints.npy")
            kp2 = np.load(root / "detection" / "akaze" / b / "keypoints.npy")
            p1, p2 = kp1[m[:, 0].astype(int), :2], kp2[m[:, 1].astype(int), :2]
            im = np.load(p / "inlier_mask.npy")
            out = Path(tmp) / "real" / p.name
            r = run({}, tmp, out, np.load(p / "E.npy"), p1, p2, np.load(p / "K.npy"), im)
            print("real %s: inliers=%d %s" % (p.name, int(im.sum()), r))
            assert np.isfinite(r["rot_deg"]) and r["n_pose_inliers"] <= im.sum()
    print("OK")
# <<< END IMPLEMENTATION <<<

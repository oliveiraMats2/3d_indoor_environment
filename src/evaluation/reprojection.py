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
#   run(cfg, run_dir, artifacts_dir) with the FULL cfg, gated by
#   evaluation.enabled. Projection: cv2.projectPoints semantics (OPENCV model),
#   vectorised in numpy over all observations.
#
# What to implement:
#   - Project every 3D point with each camera that observes it (K, R, t) and
#     measure the pixel distance to the observed keypoint.
#   - Report mean and median reprojection error in pixels.
#   - Count sparse points and dense points (0 if no dense cloud exists).
#   - Density: points divided by the approximate covered floor area in m^2.
#
# Inputs / outputs:
#   Inputs (results/<run_id>/sfm/, MODEL units, unscaled): points3d.npy (P,3),
#   obs.npy (O,4: point_idx, cam_idx, x, y), K.npy (3,3), dist.npy (4,) OPENCV
#   k1,k2,p1,p2, cam_R.npy (N,3,3), cam_t.npy (N,3) world->camera. Optional:
#   results/<run_id>/dense/<method>/points.npy and
#   artifacts/<run_id>/floorplan.csv column area_m2.
#   Outputs under results/<run_id>/evaluation/: reproj_errors.npy (O,), aligned
#   with obs.npy rows (NaN for points behind the camera).
#   One row in artifacts/<run_id>/evaluation.csv: reproj_mean_px,
#   reproj_median_px, n_sparse, n_dense, density_pts_m2.
#   n_dense = largest point count among dense methods (0 if none; each method
#   logged as metric n_dense_<method>). density_pts_m2 = n_dense (n_sparse when
#   there is no dense cloud) / area_m2; empty when area_m2 is missing.
#
# Defaults (read from the YAML block evaluation, never hardcoded):
#   enabled: false   (no new keys)
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

import numpy as np

from src import stage_io

log = logging.getLogger(__name__)

def project(points: np.ndarray, R: np.ndarray, t: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Vectorised cv2.projectPoints (OPENCV k1,k2,p1,p2) with one camera per point.
    points (O,3), R (O,3,3), t (O,3) -> (O,2) pixels; NaN where z <= 0."""

    xc = np.einsum("oij,oj->oi", R, points) + t
    z = xc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        x, y = xc[:, 0] / z, xc[:, 1] / z
    k1, k2, p1, p2 = (list(np.ravel(dist)) + [0.0] * 4)[:4]
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 * r2
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    uv = np.stack([K[0, 0] * xd + K[0, 1] * yd + K[0, 2], K[1, 1] * yd + K[1, 2]], axis=1)
    uv[z <= 0] = np.nan

    return uv

def reprojection_errors(points3d, obs, K, dist, cam_R, cam_t) -> np.ndarray:
    """(O,) pixel distance between the projection of obs[:,0] in camera obs[:,1] and obs[:,2:4]."""

    pi, ci = obs[:, 0].astype(int), obs[:, 1].astype(int)
    uv = project(points3d[pi], cam_R[ci], cam_t[ci], K, dist)

    return np.linalg.norm(uv - obs[:, 2:4], axis=1)

def run(cfg: dict, run_dir: str, artifacts_dir: str) -> dict | None:

    if not cfg["evaluation"]["enabled"]:
        log.info("evaluation disabled")
        return None

    run_dir, artifacts_dir = Path(run_dir), Path(artifacts_dir)
    sfm = run_dir / "sfm"
    names = ["points3d", "obs", "K", "dist", "cam_R", "cam_t"]
    if not all((sfm / f"{n}.npy").exists() for n in names):
        log.warning("reprojection: missing sfm arrays in %s, skipping", sfm)
        return None

    points3d, obs, K, dist, cam_R, cam_t = (np.load(sfm / f"{n}.npy") for n in names)
    errors = reprojection_errors(points3d, obs, K, dist, cam_R, cam_t)
    out = run_dir / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "reproj_errors.npy", errors)
    valid = errors[np.isfinite(errors)]
    if len(valid) < len(errors):
        log.warning("reprojection: %d observations behind the camera (NaN)", len(errors) - len(valid))

    n_dense = 0
    for path in sorted((run_dir / "dense").glob("*/points.npy")):
        n = len(np.load(path, mmap_mode="r"))
        log.info("metric n_dense_%s=%s", path.parent.name, n)
        n_dense = max(n_dense, n)

    n_sparse = len(points3d)
    rows = stage_io.read_csv(artifacts_dir / "floorplan.csv")
    try:
        area_m2 = float(rows[-1]["area_m2"]) if rows else float("nan")
    except (KeyError, ValueError):
        area_m2 = float("nan")
    if np.isfinite(area_m2) and area_m2 > 0:
        density = (n_dense or n_sparse) / area_m2
    else:
        log.warning("reprojection: floorplan area_m2 unavailable, density_pts_m2 left empty")
        density = ""

    row = {"reproj_mean_px": float(np.mean(valid)) if len(valid) else "",
           "reproj_median_px": float(np.median(valid)) if len(valid) else "",
           "n_sparse": n_sparse, "n_dense": n_dense, "density_pts_m2": density}
    stage_io.log_metrics(log, row)
    stage_io.upsert_csv(artifacts_dir / "evaluation.csv", row)

    return row

def _colmap_check(model_dir: Path) -> float:
    """Read-only check against a COLMAP model: convert to TXT in a temp dir and return the mean error."""

    import shutil, subprocess, tempfile
    if shutil.which("colmap") is None or not model_dir.exists():
        return float("nan")

    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["colmap", "model_converter", "--input_path", str(model_dir), "--output_path", tmp,
                        "--output_type", "TXT"], check=True, capture_output=True)
        lines = lambda f: [l.split() for l in open(Path(tmp) / f) if l.strip() and not l.startswith("#")]
        cam = lines("cameras.txt")[0]
        fx, fy, cx, cy, *d = map(float, cam[4:])
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        pts = {int(l[0]): list(map(float, l[1:4])) for l in lines("points3D.txt")}
        pid_index = {pid: i for i, pid in enumerate(pts)}
        P = np.array(list(pts.values()))
        img = [l.split() for l in open(Path(tmp) / "images.txt") if not l.startswith("#")]  # keep empty 2D lines
        Rs, ts, obs = [], [], []
        for ci, (head, kp) in enumerate(zip(img[0::2], img[1::2])):
            qw, qx, qy, qz, tx, ty, tz = map(float, head[1:8])
            Rs.append(np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                                [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]]))
            ts.append([tx, ty, tz])
            kp = np.array(kp, dtype=float).reshape(-1, 3)
            kp = kp[kp[:, 2] >= 0]
            obs += [[pid_index[int(p)], ci, x, y] for x, y, p in kp if int(p) in pid_index]
        err = reprojection_errors(P, np.array(obs), K, np.array(d[:4] if d else [0, 0, 0, 0]),
                                  np.array(Rs), np.array(ts))

    return float(np.nanmean(err))  # per observation; COLMAP's own figure is the mean of per-point errors

if __name__ == "__main__":
    import sys, tempfile, time
    import cv2

    rng = np.random.default_rng(0)
    K = np.array([[1500.0, 0, 960], [0, 1500.0, 540], [0, 0, 1]])
    dist = np.array([0.05, -0.02, 0.001, -0.0005])
    n_cam, n_pts = 13, 70000
    cam_R = np.stack([cv2.Rodrigues(rng.normal(0, 0.1, 3))[0] for _ in range(n_cam)])
    cam_t = rng.normal(0, 0.3, (n_cam, 3))
    points3d = rng.uniform([-2, -2, 4], [2, 2, 8], (n_pts, 3))
    pi, ci = np.meshgrid(np.arange(n_pts), np.arange(n_cam), indexing="ij")
    pi, ci = pi.ravel(), ci.ravel()
    uv = np.empty((len(pi), 2))
    for c in range(n_cam):  # ground truth with OpenCV itself
        m = ci == c
        uv[m] = cv2.projectPoints(points3d[pi[m]], cv2.Rodrigues(cam_R[c])[0], cam_t[c], K, dist)[0].reshape(-1, 2)
    obs = np.column_stack([pi, ci, uv]).astype(np.float64)

    with tempfile.TemporaryDirectory() as tmp:
        run_dir, art = Path(tmp) / "results", Path(tmp) / "artifacts"
        (run_dir / "sfm").mkdir(parents=True)
        for n, a in dict(points3d=points3d, obs=obs, K=K, dist=dist, cam_R=cam_R, cam_t=cam_t).items():
            np.save(run_dir / "sfm" / f"{n}.npy", a)
        cfg = {"evaluation": {"enabled": True}}
        t0 = time.time()
        row = run(cfg, str(run_dir), str(art))
        print(f"exact: {row}  ({time.time() - t0:.2f}s, {len(obs)} obs)")
        assert row["reproj_mean_px"] < 1e-6 and row["n_sparse"] == n_pts and row["n_dense"] == 0
        assert row["density_pts_m2"] == ""
        assert np.load(run_dir / "evaluation" / "reproj_errors.npy").shape == (len(obs),)

        noisy = obs.copy()
        noisy[:, 2:4] += rng.normal(0, 1.0, (len(obs), 2))  # |N(0,I)| is Rayleigh: mean sqrt(pi/2)
        np.save(run_dir / "sfm" / "obs.npy", noisy)
        (run_dir / "dense" / "patch_match").mkdir(parents=True)
        np.save(run_dir / "dense" / "patch_match" / "points.npy", np.zeros((123456, 3)))
        stage_io.upsert_csv(art / "floorplan.csv", {"area_m2": 12.5})
        row = run(cfg, str(run_dir), str(art))
        print(f"noisy: {row}  expected mean {np.sqrt(np.pi / 2):.4f}, median {np.sqrt(2 * np.log(2)):.4f}")
        assert abs(row["reproj_mean_px"] - np.sqrt(np.pi / 2)) < 0.01
        assert abs(row["reproj_median_px"] - np.sqrt(2 * np.log(2))) < 0.01
        assert row["n_dense"] == 123456 and abs(row["density_pts_m2"] - 123456 / 12.5) < 1e-9
        assert stage_io.read_csv(art / "evaluation.csv")[0]["n_dense"] == "123456"
        assert run({"evaluation": {"enabled": False}}, str(run_dir), str(art)) is None

    if "--colmap" in sys.argv:
        print(f"colmap_gpu/sparse/0 mean reprojection: {_colmap_check(Path('colmap_gpu/sparse/0')):.4f} px")
    print("OK")

# <<< END IMPLEMENTATION <<<

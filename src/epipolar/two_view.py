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
# Stage: 3, part 2 (two-view wiring: pose -> triangulation -> epipolar lines).
# Goal: for every <run_dir>/epipolar/<det>/<A>__<B>/ written by src.epipolar.run (F, E, K, inlier_mask),
#   rebuild pts1/pts2 from detection keypoints + matching matches_lowe, then call
#   pose.run -> (if a pose was found) triangulate.run -> visualize.run, and collect per-pair metrics.
# In:  epipolar/<det>/<pair>/{E,F,K,inlier_mask}.npy, detection/<det>/<frame>/keypoints.npy,
#      matching/<det>/<pair>/matches_lowe.npy, frames in frames_dir.
# Out: epipolar/<det>/<pair>/{R,t,pose_mask,points3d,colors}.npy + cloud.ply (written by pose/triangulate),
#      figures/epilines/<det>/<pair>.png (visualize), <artifacts_dir>/epipolar_metrics.csv.
# CSV: one row per (detector, pair), key column id = "<det>/<pair>". The file is REWRITTEN on every run
#   (stale pairs from earlier runs are dropped). Per-detector medians go to the log only.
# YAML keys: none of its own. Uses geometry (pose/triangulate), visualize.epilines and run.seed; gated by
#   geometry.enabled (main.py, and checked here too).
# Rules: never edit src/epipolar/__init__.py (Emily); only import _matched_points and _frame_path from it.
# ==============================================================================

# >>> BEGIN IMPLEMENTATION >>>
"""Stage 3 wiring: relative pose, two-view triangulation and epipolar-line figures for every pair."""
import logging
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src import stage_io
from src.epipolar import _frame_path, _matched_points, pose, triangulate, visualize

log = logging.getLogger(__name__)

COLUMNS = ["id", "detector", "pair", "n_matches", "n_inliers", "rot_deg", "t_x", "t_y", "t_z",
           "n_pose_inliers", "n_points3d", "parallax_med_deg", "cheirality_frac"]
MEDIAN_KEYS = ["n_matches", "n_inliers", "rot_deg", "n_pose_inliers", "n_points3d",
               "parallax_med_deg", "cheirality_frac"]
NAN = float("nan")

def _read_image(frames_dir: Path, frame: str) -> np.ndarray:

    image = cv2.imread(str(_frame_path(frames_dir, frame)))
    if image is None:
        raise ValueError(f"could not read frame: {frame}")

    return image

def _process_pair(cfg: dict, root: Path, frames_dir: Path, det: str, pair_dir: Path) -> dict:

    first, second = pair_dir.name.split("__", maxsplit=1)
    E, F, K = (np.load(pair_dir / f"{name}.npy") for name in ("E", "F", "K"))
    inlier_mask = np.load(pair_dir / "inlier_mask.npy").astype(bool)

    detection = root / "detection" / det
    matches = np.load(root / "matching" / det / pair_dir.name / "matches_lowe.npy")
    pts1, pts2 = _matched_points(np.load(detection / first / "keypoints.npy"),
                                 np.load(detection / second / "keypoints.npy"), matches)
    img1, img2 = _read_image(frames_dir, first), _read_image(frames_dir, second)

    row = {"id": f"{det}/{pair_dir.name}", "detector": det, "pair": pair_dir.name,
           "n_matches": len(matches), "n_inliers": int(inlier_mask.sum()), "n_points3d": 0,
           "parallax_med_deg": NAN, "cheirality_frac": NAN}
    row.update(pose.run(cfg["geometry"], str(root), pair_dir, E, pts1, pts2, K, inlier_mask))

    if np.isfinite(float(row["rot_deg"])) and (pair_dir / "R.npy").exists():
        R, t = np.load(pair_dir / "R.npy"), np.load(pair_dir / "t.npy")
        pose_mask = np.load(pair_dir / "pose_mask.npy").astype(bool)
        row.update(triangulate.run(cfg["geometry"], str(root), pair_dir, K, R, t, pts1, pts2,
                                   pose_mask, img1))
    else:
        log.warning("%s: no pose recovered, triangulation skipped", row["id"])

    out_path = root / "figures" / "epilines" / det / f"{pair_dir.name}.png"
    visualize.run(cfg["visualize"]["epilines"], str(root), out_path, img1, img2, F, pts1, pts2,
                  inlier_mask, cfg["run"]["seed"])

    return {k: row.get(k, NAN) for k in COLUMNS}

def _log_detector(det: str, rows: list[dict]) -> None:

    summary = {f"{det}.n_pairs": len(rows),
               f"{det}.n_pairs_posed": int(sum(np.isfinite(float(r["rot_deg"])) for r in rows))}
    for key in MEDIAN_KEYS:
        values = np.array([float(r[key]) for r in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        summary[f"{det}.{key}_med"] = round(float(np.median(finite)), 4) if len(finite) else NAN
    stage_io.log_metrics(log, summary)

def run(cfg: dict, run_dir: str, frames_dir: str, artifacts_dir: str) -> Path:
    """Pose + triangulation + epilines for every epipolar/<det>/<pair>; returns the metrics CSV path."""

    root, frames = Path(run_dir), Path(frames_dir)
    csv_path = Path(artifacts_dir) / "epipolar_metrics.csv"
    if not cfg["geometry"]["enabled"]:
        log.info("geometry disabled, two-view stage skipped")
        return csv_path

    epipolar_root = root / "epipolar"
    if not epipolar_root.is_dir():
        log.warning("epipolar results not found: %s (run src.epipolar first)", epipolar_root)
        return csv_path

    csv_path.unlink(missing_ok=True)
    for det_dir in sorted(p for p in epipolar_root.iterdir() if p.is_dir()):
        rows = []
        pair_dirs = sorted(p for p in det_dir.iterdir() if p.is_dir() and (p / "E.npy").exists())
        for pair_dir in tqdm(pair_dirs, desc=f"{det_dir.name} two-view", unit="pair"):
            row = _process_pair(cfg, root, frames, det_dir.name, pair_dir)
            stage_io.upsert_csv(csv_path, row, key="id")
            rows.append(row)
        if rows:
            _log_detector(det_dir.name, rows)

    return csv_path
# <<< END IMPLEMENTATION <<<

if __name__ == "__main__":
    import tempfile, inspect, sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    def _synthetic(root: Path, frames_dir: Path, seed: int) -> None:
        """Random 3D points seen by two cameras; writes frames, keypoints, matches_lowe and F/E/K/mask."""
        rng = np.random.default_rng(seed)
        w, h = 640, 480
        K = np.array([[600.0, 0, w / 2], [0, 600.0, h / 2], [0, 0, 1]])
        X = np.c_[rng.uniform(-1, 1, (200, 2)), rng.uniform(4, 6, 200)]
        R, _ = cv2.Rodrigues(np.array([0.0, 0.08, 0.0]))
        t = np.array([[-0.5], [0.0], [0.0]])
        x1 = (K @ X.T).T; x1 = x1[:, :2] / x1[:, 2:]
        x2 = (K @ (R @ X.T + t)).T; x2 = x2[:, :2] / x2[:, 2:]
        x2[:20] += rng.uniform(-40, 40, (20, 2))  # outliers
        frames_dir.mkdir(parents=True)
        for name in ("img_0000", "img_0001"):
            cv2.imwrite(str(frames_dir / f"{name}.png"), rng.integers(0, 255, (h, w, 3), dtype=np.uint8))
        for name, pts in (("img_0000", x1), ("img_0001", x2)):
            d = root / "detection" / "sift" / name
            d.mkdir(parents=True)
            np.save(d / "keypoints.npy", np.c_[pts, np.zeros((len(pts), 4))].astype(np.float32))
        pair = "img_0000__img_0001"
        m = root / "matching" / "sift" / pair
        m.mkdir(parents=True)
        np.save(m / "matches_lowe.npy", np.c_[np.arange(200), np.arange(200), np.zeros(200)].astype(np.float32))
        e = root / "epipolar" / "sift" / pair
        e.mkdir(parents=True)
        E, mask = cv2.findEssentialMat(x1, x2, K, cv2.RANSAC, 0.999, 1.0)
        F = np.linalg.inv(K).T @ E @ np.linalg.inv(K)
        for k, v in (("E", E), ("F", F), ("K", K), ("inlier_mask", mask.ravel().astype(bool))):
            np.save(e / f"{k}.npy", v)

    def _cfg() -> dict:
        return {"run": {"seed": 0},
                "geometry": {"enabled": True,
                             "ransac": {"method": "RANSAC", "threshold": 1.0, "confidence": 0.999},
                             "intrinsics": {"fx": None, "fy": None, "cx": None, "cy": None}},
                "visualize": {"epilines": {"n_lines": 20, "dpi": 150}}}

    def _check(tmp: Path, real: bool) -> None:
        root, frames, art = tmp / "run", tmp / "frames", tmp / "art"
        _synthetic(root, frames, 0)
        csv_path = run(_cfg(), str(root), str(frames), str(art))
        rows = stage_io.read_csv(csv_path)
        assert len(rows) == 1 and list(rows[0]) == COLUMNS, rows
        r = rows[0]
        assert r["id"] == "sift/img_0000__img_0001" and int(r["n_matches"]) == 200
        assert 150 <= int(r["n_inliers"]) <= 200, r
        assert int(r["n_points3d"]) > 0 and np.isfinite(float(r["rot_deg"])), r
        pair_dir = root / "epipolar" / "sift" / "img_0000__img_0001"
        if real:
            assert abs(float(r["rot_deg"]) - np.degrees(0.08)) < 1.0, r
            assert float(r["t_x"]) < -0.9, r  # t direction ~ (-1, 0, 0)
            for f in ("R", "t", "pose_mask", "points3d", "colors"):
                assert (pair_dir / f"{f}.npy").exists(), f
            assert (pair_dir / "cloud.ply").exists()
            assert (root / "figures" / "epilines" / "sift" / "img_0000__img_0001.png").exists()
        # rerun rewrites the CSV (no duplicates)
        run(_cfg(), str(root), str(frames), str(art))
        assert len(stage_io.read_csv(csv_path)) == 1
        assert run({**_cfg(), "geometry": {"enabled": False}}, str(root), str(frames), str(art)) == csv_path

    # 1) stubbed modules: tests the wiring alone
    def _pose_stub(cfg, run_dir, pair_dir, E, pts1, pts2, K, inlier_mask):
        assert pts1.shape == pts2.shape == (len(inlier_mask), 2)
        _, R, t, m = cv2.recoverPose(E, pts1.astype(np.float64), pts2.astype(np.float64), K,
                                     mask=inlier_mask.astype(np.uint8).copy())
        np.save(pair_dir / "R.npy", R); np.save(pair_dir / "t.npy", t)
        np.save(pair_dir / "pose_mask.npy", m.ravel() > 0)
        return {"rot_deg": float(np.degrees(np.linalg.norm(cv2.Rodrigues(R)[0]))), "t_x": float(t[0, 0]),
                "t_y": float(t[1, 0]), "t_z": float(t[2, 0]), "n_pose_inliers": int((m > 0).sum())}

    def _tri_stub(cfg, run_dir, pair_dir, K, R, t, pts1, pts2, mask, image1_bgr):
        assert mask.dtype == bool and image1_bgr.ndim == 3
        return {"n_points3d": int(mask.sum()), "parallax_med_deg": 1.0, "cheirality_frac": 1.0}

    def _vis_stub(cfg, run_dir, out_path, image1_bgr, image2_bgr, F, pts1, pts2, mask, seed):
        assert set(cfg) == {"n_lines", "dpi"} and isinstance(seed, int)
        return None

    saved = {m: getattr(m, "run", None) for m in (pose, triangulate, visualize)}
    pose.run, triangulate.run, visualize.run = _pose_stub, _tri_stub, _vis_stub
    with tempfile.TemporaryDirectory() as tmp:
        _check(Path(tmp), real=False)
    for m, fn in saved.items():
        if fn is None:
            delattr(m, "run")
        else:
            m.run = fn
    print("stub test OK")

    # 2) real integration test when the three modules are implemented
    implemented = all("def run" in inspect.getsource(m) for m in (pose, triangulate, visualize))
    if implemented:
        with tempfile.TemporaryDirectory() as tmp:
            _check(Path(tmp), real=True)
        print("integration test OK")
    else:
        print("integration test SKIPPED: pose/triangulate/visualize not implemented yet", file=sys.stderr)
    print("OK")

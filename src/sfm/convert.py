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
# Stage: sfm (model conversion), runs after src/sfm COLMAP reconstruction.
# Goal: turn the COLMAP text model of exports/colmap/sparse/0 into numpy arrays and SfM metrics.
# In: <run_dir>/exports/colmap/sparse/0/{cameras,images,points3D}.txt (converted from *.bin with
#     `colmap model_converter --output_type TXT` when only the binary model exists); frames dir
#     cfg["extraction"]["output_dir"] (only counted).
# Out: <run_dir>/sfm/{points3d,colors,point_error,track_len,cam_R,cam_t,centers,image_names,K,dist,obs}.npy
#      (contract "Array conventions"); row keyed by run_id merged into <artifacts_dir>/sfm_metrics.csv
#      (n_frames, n_registered, registered_frac, n_sparse, track_len_mean, obs_per_image_mean, reproj_mean_px).
# YAML keys: none of its own; uses sfm.enabled and sfm.colmap_bin.
# Rules: hand-written parser of the documented COLMAP text format (no new dependency); colmap missing ->
#        WARNING and return None; never writes outside run_dir / artifacts_dir.

# >>> BEGIN IMPLEMENTATION >>>
"""COLMAP text model -> numpy arrays (world->camera poses, points, tracks, intrinsics) and SfM metrics."""
import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src import stage_io

log = logging.getLogger(__name__)

IMAGE_EXT = {".png", ".jpg", ".jpeg"}
SINGLE_FOCAL = {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"}
OPENCV_MODELS = {"OPENCV", "FULL_OPENCV"}

def data_lines(path: Path) -> list[str]:
    """Lines without comments; empty lines are kept (images.txt may have an empty POINTS2D line)."""

    return [line for line in path.read_text().splitlines() if not line.startswith("#")]

def qvec_to_R(q) -> np.ndarray:
    """Quaternion (w, x, y, z) -> rotation matrix."""

    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)

    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])

def read_cameras(path: Path) -> dict[int, tuple[str, np.ndarray]]:

    cameras = {}
    for line in data_lines(path):
        parts = line.split()
        if parts:
            cameras[int(parts[0])] = (parts[1], np.array(parts[4:], dtype=np.float64))

    return cameras

def read_images(path: Path) -> dict[int, dict]:
    """image_id -> {qvec, tvec, camera_id, name, xy (K,2)}; two lines per image."""

    lines, images, i = data_lines(path), {}, 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue

        parts = lines[i].split()
        pts = np.array(lines[i + 1].split() if i + 1 < len(lines) else [], dtype=np.float64).reshape(-1, 3)
        images[int(parts[0])] = {"qvec": np.array(parts[1:5], dtype=np.float64),
                                 "tvec": np.array(parts[5:8], dtype=np.float64),
                                 "camera_id": int(parts[8]), "name": " ".join(parts[9:]), "xy": pts[:, :2]}
        i += 2

    return images

def read_points(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    """(xyz (P,3), rgb (P,3) uint8, error (P,), tracks [(L,2) image_id, point2d_idx])."""

    xyz, rgb, err, tracks = [], [], [], []
    for line in tqdm(data_lines(path), desc="points3D", leave=False):
        parts = line.split()
        if not parts:
            continue

        xyz.append([float(v) for v in parts[1:4]])
        rgb.append([int(v) for v in parts[4:7]])
        err.append(float(parts[7]))
        tracks.append(np.array(parts[8:], dtype=np.int64).reshape(-1, 2))

    return (np.array(xyz, dtype=np.float64).reshape(-1, 3), np.array(rgb, dtype=np.uint8).reshape(-1, 3),
            np.array(err, dtype=np.float64), tracks)

def intrinsics(model: str, params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """COLMAP camera params -> K (3,3), dist (4,) OPENCV k1,k2,p1,p2."""

    if model in SINGLE_FOCAL:
        (f, cx, cy), extra = params[:3], params[3:]
        fx = fy = f
    else:
        (fx, fy, cx, cy), extra = params[:4], params[4:]

    dist = np.zeros(4)
    if model in OPENCV_MODELS:
        dist[:] = extra[:4]
    else:
        n = min(len(extra), 2)
        dist[:n] = extra[:n]
        log.warning("camera model %s is not OPENCV: dist filled with %d radial term(s) %s and zeros",
                    model, n, extra.tolist())

    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

    return K, dist

def ensure_txt(model_dir: Path, colmap_bin: str) -> bool:
    """Make sure {cameras,images,points3D}.txt exist in model_dir (converting the .bin model if needed)."""

    names = ("cameras", "images", "points3D")
    if all((model_dir / f"{n}.txt").exists() for n in names):
        return True

    if not all((model_dir / f"{n}.bin").exists() for n in names):
        log.warning("no COLMAP model (txt or bin) in %s", model_dir)
        return False

    if shutil.which(colmap_bin) is None:
        log.warning("%s not found: cannot convert %s to TXT", colmap_bin, model_dir)
        return False

    cmd = [colmap_bin, "model_converter", "--input_path", str(model_dir), "--output_path", str(model_dir),
           "--output_type", "TXT"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not all((model_dir / f"{n}.txt").exists() for n in names):
        log.warning("model_converter failed (%d): %s", proc.returncode, proc.stderr.strip()[-500:])
        return False

    return True

def count_frames(frames_dir) -> int | str:

    path = Path(frames_dir) if frames_dir else None
    if path is None or not path.is_dir():
        log.warning("frames dir %s not found: n_frames empty", frames_dir)
        return ""

    return sum(1 for p in path.iterdir() if p.suffix.lower() in IMAGE_EXT)

def run(cfg: dict, run_dir: str, artifacts_dir: str) -> dict | None:

    sfm_cfg = cfg["sfm"]
    if not sfm_cfg.get("enabled", True):
        log.info("sfm disabled: skipping model conversion")
        return None

    model_dir = Path(run_dir) / "exports" / "colmap" / "sparse" / "0"
    if not ensure_txt(model_dir, sfm_cfg.get("colmap_bin", "colmap")):
        return None

    cameras = read_cameras(model_dir / "cameras.txt")
    images = read_images(model_dir / "images.txt")
    xyz, rgb, err, tracks = read_points(model_dir / "points3D.txt")

    ids = sorted(images, key=lambda k: images[k]["name"])
    cam_idx = {image_id: i for i, image_id in enumerate(ids)}
    cam_R = np.stack([qvec_to_R(images[k]["qvec"]) for k in ids]) if ids else np.zeros((0, 3, 3))
    cam_t = np.stack([images[k]["tvec"] for k in ids]) if ids else np.zeros((0, 3))
    centers = -np.einsum("nji,nj->ni", cam_R, cam_t)
    names = np.array([images[k]["name"] for k in ids], dtype=str)

    used = [images[k]["camera_id"] for k in ids]
    main_cam = max(set(used), key=used.count) if used else next(iter(cameras))
    if len(set(used)) > 1:
        log.warning("%d cameras in the model: K/dist from camera %d (most images)", len(set(used)), main_cam)
    K, dist = intrinsics(*cameras[main_cam])

    track_len = np.array([len(t) for t in tracks], dtype=np.int64)
    obs = np.zeros((int(track_len.sum()), 4), dtype=np.float64)
    row = 0
    for p, track in enumerate(tqdm(tracks, desc="tracks", leave=False)):
        for image_id, p2d in track:
            obs[row] = (p, cam_idx[int(image_id)], *images[int(image_id)]["xy"][p2d])
            row += 1

    out = Path(run_dir) / "sfm"
    out.mkdir(parents=True, exist_ok=True)
    arrays = {"points3d": xyz, "colors": rgb, "point_error": err, "track_len": track_len, "cam_R": cam_R,
              "cam_t": cam_t, "centers": centers, "image_names": names, "K": K, "dist": dist, "obs": obs}
    for name, array in arrays.items():
        np.save(out / f"{name}.npy", array)

    log.info("estimated intrinsics, paste into the YAML: geometry.intrinsics: {fx: %.4f, fy: %.4f, "
             "cx: %.4f, cy: %.4f}", K[0, 0], K[1, 1], K[0, 2], K[1, 2])

    n_frames, n_reg = count_frames(cfg["extraction"].get("output_dir")), len(ids)
    metrics = {"n_frames": n_frames, "n_registered": n_reg,
               "registered_frac": round(n_reg / n_frames, 4) if n_frames else "",
               "n_sparse": len(xyz),
               "track_len_mean": round(float(track_len.mean()), 4) if len(xyz) else "",
               "obs_per_image_mean": round(len(obs) / n_reg, 4) if n_reg else "",
               "reproj_mean_px": round(float(err.mean()), 4) if len(err) else ""}
    stage_io.log_metrics(log, metrics)

    run_id = str(cfg["run"]["id"])
    csv_path = Path(artifacts_dir) / "sfm_metrics.csv"
    previous = next((r for r in stage_io.read_csv(csv_path) if r.get("run_id") == run_id), {})
    stage_io.upsert_csv(csv_path, {**previous, "run_id": run_id, **metrics}, key="run_id")

    return metrics

# <<< END IMPLEMENTATION <<<

if __name__ == "__main__":
    import tempfile

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        model = tmp / "run" / "exports" / "colmap" / "sparse" / "0"
        model.mkdir(parents=True)
        frames = tmp / "frames"
        frames.mkdir()
        for i in range(3):
            (frames / f"img_{i:04d}.png").write_bytes(b"")
        (frames / "selecao.csv").write_text("x\n")

        (model / "cameras.txt").write_text("# Camera list\n1 OPENCV 640 480 500 510 320 240 0.1 -0.05 0.001 0.002\n")
        s = np.sqrt(0.5)  # 90 deg about z for image_id 1 (named b, so it sorts second)
        (model / "images.txt").write_text(
            "# Image list\n# two lines per image\n"
            f"1 {s} 0 0 {s} 1 2 3 1 img_b.png\n10 20 0 30 40 1 50 60 -1\n"
            "2 1 0 0 0 0 0 0 1 img_a.png\n11 21 0 31 41 1 51 61 2\n")
        (model / "points3D.txt").write_text(
            "# 3D point list\n0 1 2 3 255 0 0 0.5 1 0 2 0\n1 4 5 6 0 255 0 1.0 1 1 2 1\n2 7 8 9 0 0 255 1.5 2 2\n")
        arts = tmp / "arts"
        stage_io.upsert_csv(arts / "sfm_metrics.csv", {"run_id": "other", "n_sparse": 9}, key="run_id")
        stage_io.upsert_csv(arts / "sfm_metrics.csv", {"run_id": "t", "time_mapper_s": 12.5}, key="run_id")

        cfg = {"run": {"id": "t"}, "sfm": {"enabled": True, "colmap_bin": "colmap"},
               "extraction": {"output_dir": str(frames)}}
        m = run(cfg, str(tmp / "run"), str(arts))
        d = {n: np.load(tmp / "run" / "sfm" / f"{n}.npy") for n in
             ("points3d", "colors", "point_error", "track_len", "cam_R", "cam_t", "centers", "image_names",
              "K", "dist", "obs")}
        assert d["image_names"].tolist() == ["img_a.png", "img_b.png"]
        assert np.allclose(d["cam_R"][0], np.eye(3))
        assert np.allclose(d["cam_R"][1], [[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        assert np.allclose(d["cam_t"], [[0, 0, 0], [1, 2, 3]])
        assert np.allclose(d["centers"][1], -d["cam_R"][1].T @ [1, 2, 3])
        assert np.allclose(d["centers"][1], [-2, 1, -3])
        assert d["points3d"].dtype == np.float64 and np.allclose(d["points3d"][2], [7, 8, 9])
        assert d["colors"].dtype == np.uint8 and d["colors"][1].tolist() == [0, 255, 0]
        assert np.allclose(d["point_error"], [0.5, 1.0, 1.5])
        assert d["track_len"].tolist() == [2, 2, 1]
        assert np.allclose(d["K"], [[500, 0, 320], [0, 510, 240], [0, 0, 1]])
        assert np.allclose(d["dist"], [0.1, -0.05, 0.001, 0.002])
        # point 0 seen by image 1 (cam 1, kp 0) and image 2 (cam 0, kp 0); point 2 by image 2 kp 2
        assert np.allclose(d["obs"], [[0, 1, 10, 20], [0, 0, 11, 21], [1, 1, 30, 40], [1, 0, 31, 41],
                                      [2, 0, 51, 61]])
        assert m["n_frames"] == 3 and m["n_registered"] == 2 and m["n_sparse"] == 3
        assert m["registered_frac"] == round(2 / 3, 4) and m["track_len_mean"] == round(5 / 3, 4)
        assert m["obs_per_image_mean"] == 2.5 and m["reproj_mean_px"] == 1.0
        rows = {r["run_id"]: r for r in stage_io.read_csv(arts / "sfm_metrics.csv")}
        assert rows["t"]["time_mapper_s"] == "12.5" and rows["t"]["n_sparse"] == "3"
        assert rows["other"]["n_sparse"] == "9"
        K_fisheye, dist_fisheye = intrinsics("SIMPLE_RADIAL", np.array([400.0, 100, 50, 0.2]))
        assert K_fisheye[1, 1] == 400 and np.allclose(dist_fisheye, [0.2, 0, 0, 0])
        assert run({**cfg, "sfm": {"enabled": False}}, str(tmp / "run"), str(arts)) is None

    real = Path(__file__).resolve().parents[2] / "colmap_gpu" / "sparse" / "0"
    if real.is_dir() and shutil.which("colmap"):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            model = tmp / "run" / "exports" / "colmap" / "sparse" / "0"
            shutil.copytree(real, model)  # conversion happens inside the temp copy only
            cfg = {"run": {"id": "real"}, "sfm": {"enabled": True, "colmap_bin": "colmap"},
                   "extraction": {"output_dir": str(real.parents[1] / "images")}}
            m = run(cfg, str(tmp / "run"), str(tmp / "arts"))
            K = np.load(tmp / "run" / "sfm" / "K.npy")
            print("real:", m)
            print(f"real intrinsics: fx={K[0, 0]:.2f} fy={K[1, 1]:.2f} cx={K[0, 2]:.2f} cy={K[1, 2]:.2f}")
            assert m["n_registered"] == 194
    print("OK")

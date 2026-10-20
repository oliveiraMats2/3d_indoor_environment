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
# Stage: scale (after sfm).
# Goal: metric scale s = cm per model unit, from an ArUco marker of known side or from manual
#       point pairs of known length; never invent a value (s = nan, unit "model" when unknown).
# In:  <run_dir>/sfm/{cam_R,cam_t,K,dist,image_names,points3d,centers}.npy, frames in
#      cfg["extraction"]["output_dir"].
# Out: <run_dir>/scale/{scale,marker_sides,points3d_cm,centers_cm}.npy; artifacts_dir/scale.csv
#      (method, scale, n_frames_marker, side_cv, unit).
# YAML: scale.{enabled, method: aruco|manual, aruco_dict, marker_cm, min_parallax_deg,
#       manual_pairs: [{length_cm, p: [{image,x,y},{image,x,y}], q: [{image,x,y},{image,x,y}]}],
#       provisional_extent_cm (read by other stages via stage_io.units_per_cm)}.
# Rules: corners undistorted with K/dist, triangulated (normalised coords, P = [R|t]) from every pair of
#   registered frames seeing them (parallax >= min_parallax_deg), median per corner.
#   marker_sides.npy = (n_pairs, 4) model-unit side lengths of the per-pair squares (manual: (n,) lengths).
#   side_cv = std/mean of all those per-pair side lengths (manual: cv of the per-entry scale estimates).

# >>> BEGIN IMPLEMENTATION >>>
"""Metric scale from an ArUco marker or manual length measurements."""
import itertools
import logging
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src import stage_io

log = logging.getLogger(__name__)

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".PNG", ".JPG")

def normalize(pixels: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """(N,2) pixels -> (N,2) undistorted normalised camera coordinates."""

    pts = np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2)

    return cv2.undistortPoints(pts, K, np.asarray(dist, dtype=np.float64)).reshape(-1, 2)

def triangulate_pair(R1, t1, R2, t2, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Linear (DLT) triangulation of (N,2) normalised coords from two views -> (N,3)."""

    P1 = np.hstack([R1, np.reshape(t1, (3, 1))])
    P2 = np.hstack([R2, np.reshape(t2, (3, 1))])
    X = cv2.triangulatePoints(P1, P2, x1.T.astype(np.float64), x2.T.astype(np.float64))

    return (X[:3] / X[3]).T

def parallax_deg(R1, t1, R2, t2, X: np.ndarray) -> float:

    c1, c2 = -R1.T @ np.ravel(t1), -R2.T @ np.ravel(t2)
    a, b = X - c1, X - c2
    cos = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)

    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))).min())

def square_sides(corners: np.ndarray) -> np.ndarray:
    """(4,3) corners in order -> (4,) edge lengths."""

    return np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1)

def marker_geometry(observations: dict, cam_R, cam_t, min_parallax_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """observations: {marker_id: {cam_idx: (4,2) normalised corners}}.
    Returns (median_sides (n_ids*4,), pair_sides (n_pairs, 4)) in model units."""

    median_sides, pair_sides = [], []
    for marker_id, views in observations.items():
        squares = []
        for i, j in itertools.combinations(sorted(views), 2):
            X = triangulate_pair(cam_R[i], cam_t[i], cam_R[j], cam_t[j], views[i], views[j])
            if not np.all(np.isfinite(X)) or parallax_deg(cam_R[i], cam_t[i], cam_R[j], cam_t[j], X) < min_parallax_deg:
                continue
            squares.append(X)
            pair_sides.append(square_sides(X))
        if squares:
            corners = np.median(np.stack(squares), axis=0)
            median_sides.extend(square_sides(corners))
        else:
            log.warning("marker %s: no frame pair with parallax >= %.2f deg", marker_id, min_parallax_deg)

    return np.asarray(median_sides, dtype=np.float64), np.asarray(pair_sides, dtype=np.float64).reshape(-1, 4)

def find_frame(frames_dir: Path, name: str) -> Path | None:

    path = frames_dir / name
    if path.suffix and path.exists():
        return path
    for ext in IMAGE_EXTS:
        if (frames_dir / f"{Path(name).stem}{ext}").exists():
            return frames_dir / f"{Path(name).stem}{ext}"

    return None

def make_detector(dict_name: str):

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))

    return cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

def detect_markers(detector, gray: np.ndarray) -> dict:
    """-> {marker_id: (4,2) pixel corners}."""

    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return {}

    return {int(i): np.asarray(c, dtype=np.float64).reshape(4, 2) for i, c in zip(ids.ravel(), corners)}

def aruco_scale(scfg: dict, frames_dir: Path, names, cam_R, cam_t, K, dist) -> dict:

    detector = make_detector(scfg["aruco_dict"])
    observations: dict = {}
    seen = set()
    for idx, name in enumerate(tqdm(names, desc="aruco")):
        path = find_frame(frames_dir, str(name))
        if path is None:
            continue
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        for marker_id, px in detect_markers(detector, gray).items():
            observations.setdefault(marker_id, {})[idx] = normalize(px, K, dist)
            seen.add(idx)

    out = {"n_frames_marker": len(seen), "scale": float("nan"), "side_cv": None, "sides": np.zeros((0, 4))}
    if not seen:
        log.warning("scale: no ArUco marker (%s) detected in %d frames", scfg["aruco_dict"], len(names))
        return out

    log.info("scale: marker ids %s seen in %d frames", sorted(observations), len(seen))
    median_sides, pair_sides = marker_geometry(observations, cam_R, cam_t, scfg["min_parallax_deg"])
    out["sides"] = pair_sides
    if len(pair_sides):
        out["side_cv"] = float(pair_sides.std() / pair_sides.mean())
    if scfg["marker_cm"] is None:
        log.warning("scale: marker detected but scale.marker_cm is null; scale stays unknown")
    elif len(median_sides):
        out["scale"] = float(scfg["marker_cm"]) / float(np.median(median_sides))

    return out

def manual_scale(scfg: dict, names, cam_R, cam_t, K, dist) -> dict:

    index = {}
    for i, n in enumerate(names):
        index[str(n)] = i
        index.setdefault(Path(str(n)).stem, i)

    def endpoint(clicks):
        (a, b) = clicks
        i, j = index[a["image"]], index[b["image"]]
        xa = normalize([[a["x"], a["y"]]], K, dist)
        xb = normalize([[b["x"], b["y"]]], K, dist)
        return triangulate_pair(cam_R[i], cam_t[i], cam_R[j], cam_t[j], xa, xb)[0], {i, j}

    lengths, estimates, used = [], [], set()
    for entry in scfg["manual_pairs"]:
        try:
            p, ip = endpoint(entry["p"])
            q, iq = endpoint(entry["q"])
        except KeyError as exc:
            log.warning("scale: manual pair skipped, image %s not registered", exc)
            continue
        length = float(np.linalg.norm(p - q))
        if not np.isfinite(length) or length <= 0:
            continue
        lengths.append(length)
        estimates.append(float(entry["length_cm"]) / length)
        used |= ip | iq

    out = {"n_frames_marker": len(used), "scale": float("nan"), "side_cv": None, "sides": np.asarray(lengths)}
    if not estimates:
        log.warning("scale: scale.manual_pairs empty or unusable; scale stays unknown")
        return out
    out["scale"] = float(np.median(estimates))
    if len(estimates) > 1:
        out["side_cv"] = float(np.std(estimates) / np.mean(estimates))

    return out

def run(cfg: dict, run_dir: str, artifacts_dir: str) -> float:

    scfg = cfg["scale"]
    if not scfg["enabled"]:
        log.info("scale disabled")
        return float("nan")

    sfm_dir = Path(run_dir) / "sfm"
    if not (sfm_dir / "cam_R.npy").exists():
        log.warning("scale: %s has no SfM arrays; skipping", sfm_dir)
        return float("nan")

    cam_R, cam_t = np.load(sfm_dir / "cam_R.npy"), np.load(sfm_dir / "cam_t.npy")
    K, dist = np.load(sfm_dir / "K.npy"), np.load(sfm_dir / "dist.npy")
    names = np.load(sfm_dir / "image_names.npy", allow_pickle=True)

    if scfg["method"] == "aruco":
        res = aruco_scale(scfg, Path(cfg["extraction"]["output_dir"]), names, cam_R, cam_t, K, dist)
    else:
        res = manual_scale(scfg, names, cam_R, cam_t, K, dist)

    s = res["scale"]
    known = bool(np.isfinite(s))
    if not known:
        log.warning("scale unknown: arrays copied in model units (unit=model)")

    out_dir = Path(run_dir) / "scale"
    out_dir.mkdir(parents=True, exist_ok=True)
    factor = s if known else 1.0
    np.save(out_dir / "scale.npy", np.array([s], dtype=np.float64))
    np.save(out_dir / "marker_sides.npy", res["sides"])
    np.save(out_dir / "points3d_cm.npy", np.load(sfm_dir / "points3d.npy") * factor)
    np.save(out_dir / "centers_cm.npy", np.load(sfm_dir / "centers.npy") * factor)

    row = {"method": scfg["method"], "scale": s if known else "",
           "n_frames_marker": res["n_frames_marker"],
           "side_cv": res["side_cv"] if (known and res["side_cv"] is not None) else "",
           "unit": "cm" if known else "model"}
    stage_io.log_metrics(log, row)
    stage_io.upsert_csv(Path(artifacts_dir) / "scale.csv", row)

    return s
# <<< END IMPLEMENTATION <<<

if __name__ == "__main__":
    import tempfile

    logging.basicConfig(level=logging.WARNING)
    rng = np.random.default_rng(0)
    K = np.array([[800.0, 0, 640], [0, 800, 360], [0, 0, 1]])
    dist = np.zeros(4)

    def look_at(center, target=np.zeros(3)):
        z = target - center; z /= np.linalg.norm(z)
        x = np.cross([0, -1.0, 0], z); x /= np.linalg.norm(x)
        R = np.stack([x, np.cross(z, x), z])
        return R, -R @ center

    centers = np.array([[a, 0.2 * np.sin(a), -4.0] for a in np.linspace(-1.5, 1.5, 6)])
    poses = [look_at(c) for c in centers]
    cam_R = np.stack([p[0] for p in poses]); cam_t = np.stack([p[1] for p in poses])
    side = 0.37  # model units
    square = np.array([[0, 0, 0], [side, 0, 0], [side, side, 0], [0, side, 0]]) - side / 2

    def project(i, X):
        x, _ = cv2.projectPoints(X, cv2.Rodrigues(cam_R[i])[0], cam_t[i], K, dist)
        return x.reshape(-1, 2)

    # 1) triangulation / scale math on projected pixels (+0.3 px noise)
    obs = {7: {i: normalize(project(i, square) + rng.normal(0, 0.3, (4, 2)), K, dist) for i in range(6)}}
    med, pairs = marker_geometry(obs, cam_R, cam_t, 1.0)
    assert pairs.shape == (15, 4) and med.shape == (4,)
    s = 20.0 / np.median(med)
    assert abs(s - 20.0 / side) / (20.0 / side) < 0.01, s
    assert pairs.std() / pairs.mean() < 0.02

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        run_dir, art, frames = tmp / "run", tmp / "art", tmp / "frames"
        (run_dir / "sfm").mkdir(parents=True); frames.mkdir()
        names = np.array([f"img_{i:04d}.png" for i in range(6)])
        np.save(run_dir / "sfm/cam_R.npy", cam_R); np.save(run_dir / "sfm/cam_t.npy", cam_t)
        np.save(run_dir / "sfm/K.npy", K); np.save(run_dir / "sfm/dist.npy", dist)
        np.save(run_dir / "sfm/image_names.npy", names); np.save(run_dir / "sfm/centers.npy", centers)
        np.save(run_dir / "sfm/points3d.npy", rng.normal(size=(50, 3)))

        # 2) real ArUco rendered and warped into synthetic views at the true projected corners
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        tag = cv2.aruco.generateImageMarker(dictionary, 7, 200)
        src = np.float32([[0, 0], [200, 0], [200, 200], [0, 200]])
        for i in range(6):
            img = np.full((720, 1280), 128, np.uint8)
            H = cv2.getPerspectiveTransform(src, project(i, square).astype(np.float32))
            warped = cv2.warpPerspective(tag, H, (1280, 720), borderValue=0)
            mask = cv2.warpPerspective(np.full_like(tag, 255), H, (1280, 720))
            img[mask > 0] = warped[mask > 0]
            # white quiet zone around the marker
            big = (square - square.mean(0)) * 1.5 + square.mean(0)
            quiet = cv2.fillConvexPoly(np.zeros_like(img), project(i, big).astype(np.int32), 255)
            img[(quiet > 0) & (mask == 0)] = 255
            cv2.imwrite(str(frames / names[i]), img)
        det = detect_markers(make_detector("DICT_4X4_50"), cv2.imread(str(frames / names[0]), 0))
        assert list(det) == [7] and np.abs(det[7] - project(0, square)).max() < 2.0

        cfg = {"extraction": {"output_dir": str(frames)},
               "scale": {"enabled": True, "method": "aruco", "aruco_dict": "DICT_4X4_50", "marker_cm": 20.0,
                         "min_parallax_deg": 1.0, "manual_pairs": [], "provisional_extent_cm": 500}}
        s = run(cfg, str(run_dir), str(art))
        assert abs(s - 20.0 / side) / (20.0 / side) < 0.02, s
        row = stage_io.read_csv(art / "scale.csv")[0]
        assert row["unit"] == "cm" and row["n_frames_marker"] == "6" and row["side_cv"] != ""
        assert stage_io.scale_info(run_dir)[1] == "cm"
        assert np.allclose(np.load(run_dir / "scale/centers_cm.npy"), centers * s)

        # 3) manual pairs: two square corners clicked in views 0 and 3, length = side
        pix = {i: project(i, square) for i in (0, 3)}
        entry = {"length_cm": 20.0,
                 "p": [{"image": "img_0000", "x": pix[0][0, 0], "y": pix[0][0, 1]},
                       {"image": "img_0003.png", "x": pix[3][0, 0], "y": pix[3][0, 1]}],
                 "q": [{"image": "img_0000", "x": pix[0][1, 0], "y": pix[0][1, 1]},
                       {"image": "img_0003", "x": pix[3][1, 0], "y": pix[3][1, 1]}]}
        mcfg = {**cfg, "scale": {**cfg["scale"], "method": "manual", "manual_pairs": [entry]}}
        s = run(mcfg, str(run_dir), str(art))
        assert abs(s - 20.0 / side) < 1e-6 * (20.0 / side), s

        # 4) fallbacks: marker_cm null / no marker / empty manual_pairs -> nan, unit model, raw copy
        ncfg = {**cfg, "scale": {**cfg["scale"], "marker_cm": None}}
        assert np.isnan(run(ncfg, str(run_dir), str(art)))
        row = stage_io.read_csv(art / "scale.csv")[0]
        assert row["unit"] == "model" and row["scale"] == "" and row["n_frames_marker"] == "6"
        for f in frames.iterdir():
            cv2.imwrite(str(f), np.full((720, 1280), 128, np.uint8))
        for sub in ({"marker_cm": 20.0}, {"method": "manual"}):
            fcfg = {**cfg, "scale": {**cfg["scale"], **sub}}
            assert np.isnan(run(fcfg, str(run_dir), str(art)))
            row = stage_io.read_csv(art / "scale.csv")[0]
            assert row["unit"] == "model" and row["scale"] == "" and row["side_cv"] == ""
            assert stage_io.scale_info(run_dir)[1] == "model"
            assert np.allclose(np.load(run_dir / "scale/centers_cm.npy"), centers)
    print("OK")

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
# Stage: evaluation (hard regions), after planes and reprojection.
# Goal: compare the reconstruction quality of smooth (low texture) and textured wall regions.
# In:  sfm/{points3d,obs,K,dist,cam_R,cam_t}.npy (model units); evaluation/reproj_errors.npy (O,) aligned
#      with obs rows if present (else computed with cv2.projectPoints); planes/labels.npy (per point of
#      clean/<planes.source>/points.npy, working units: -1 none, 0 floor, 1 ceiling, 2+k wall k);
#      scale via stage_io.scale_info.
# Out: artifacts/<id>/hard_regions.csv, one row per region (all, wall, smooth_wall, textured_wall):
#      region, n_points, reproj_mean_px, inlier_frac.
# YAML: evaluation: {hard_regions: true, texture_radius_px: 30, texture_split_quantile: 0.5, inlier_px: 2.0}
#       (also reads planes.source, planes.distance_cm, scale.provisional_extent_cm).
# Rules:
# - sfm points are scaled by s (raw when unit = model) and take the label of their nearest clean point if it
#   lies within planes.distance_cm * units_per_cm, else -1. Missing planes -> every point counts as wall
#   (WARNING).
# - Texture of an observation = number of OTHER observations of the same camera within texture_radius_px
#   (uniform grid, cell = radius, exact distance check); texture of a point = median over its observations.
# - Wall points split at the texture_split_quantile: texture <= q -> smooth_wall, > q -> textured_wall.
# - Point error = mean of its finite observation errors; reproj_mean_px = mean of point errors in the region;
#   inlier_frac = fraction of region points with point error < inlier_px. Empty cells when a region is empty.

# >>> BEGIN IMPLEMENTATION >>>
"""Hard regions: reprojection error and inlier fraction of smooth vs textured walls."""
import logging
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from src import stage_io

log = logging.getLogger(__name__)

def observation_errors(points3d, obs, K, dist, cam_R, cam_t) -> np.ndarray:
    """(O,) pixel reprojection error per observation, cv2.projectPoints per camera; NaN behind the camera."""

    pi, ci = obs[:, 0].astype(int), obs[:, 1].astype(int)
    err = np.full(len(obs), np.nan)
    order = np.argsort(ci, kind="stable")
    cams, starts = np.unique(ci[order], return_index=True)
    for cam, rows in zip(cams, np.split(order, starts[1:])):
        X = points3d[pi[rows]].astype(np.float64)
        rvec, _ = cv2.Rodrigues(np.asarray(cam_R[cam], dtype=np.float64))
        uv, _ = cv2.projectPoints(X, rvec, np.asarray(cam_t[cam], dtype=np.float64), K, np.ravel(dist))
        e = np.linalg.norm(uv.reshape(-1, 2) - obs[rows, 2:4], axis=1)
        e[(X @ cam_R[cam].T + cam_t[cam])[:, 2] <= 0] = np.nan
        err[rows] = e

    return err

def keypoint_density(obs: np.ndarray, radius: float) -> np.ndarray:
    """(O,) number of other observations of the same camera within radius pixels (grid, 3x3 cells)."""

    cam = obs[:, 1].astype(np.int64)
    cell = np.floor(obs[:, 2:4] / radius).astype(np.int64)
    cell -= cell.min(axis=0) - 1                                   # >= 1, so the -1 neighbours stay valid
    gx, gy = int(cell[:, 0].max()) + 2, int(cell[:, 1].max()) + 2
    key = (cam * gy + cell[:, 1]) * gx + cell[:, 0]
    order = np.argsort(key, kind="stable")
    skey, sxy = key[order], obs[order, 2:4]
    counts = np.zeros(len(obs), dtype=np.int64)
    r2 = radius * radius
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            nk = key + dy * gx + dx
            lo, hi = np.searchsorted(skey, nk, "left"), np.searchsorted(skey, nk, "right")
            n = hi - lo
            if not n.any():
                continue
            src = np.repeat(np.arange(len(obs)), n)
            offs = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
            dst = np.repeat(lo, n) + offs
            d2 = ((obs[src, 2:4] - sxy[dst]) ** 2).sum(axis=1)
            counts += np.bincount(src[d2 <= r2], minlength=len(obs))

    return counts - 1                                              # remove the observation itself

def group_median(groups: np.ndarray, values: np.ndarray, n_groups: int) -> np.ndarray:
    """(n_groups,) median of values per group id; NaN for empty groups."""

    order = np.lexsort((values, groups))
    g, v = groups[order], values[order].astype(np.float64)
    count = np.bincount(g, minlength=n_groups)
    start = np.cumsum(count) - count
    out = np.full(n_groups, np.nan)
    ok = count > 0
    out[ok] = 0.5 * (v[start[ok] + (count[ok] - 1) // 2] + v[start[ok] + count[ok] // 2])

    return out

def point_labels(run_dir: Path, cfg: dict, points_work: np.ndarray) -> np.ndarray | None:
    """Plane label of each sfm point (nearest clean point within planes.distance_cm); None without planes."""

    labels_path = run_dir / "planes" / "labels.npy"
    clean_path = run_dir / "clean" / cfg["planes"]["source"] / "points.npy"
    if not clean_path.exists():
        clean_path = run_dir / "scale" / "points3d_cm.npy"
    if not labels_path.exists() or not clean_path.exists():
        return None

    labels, clean = np.load(labels_path), np.load(clean_path)
    if len(labels) != len(clean) or not len(clean):
        log.warning("hard_regions: planes/labels.npy (%d) does not match %s (%d)", len(labels), clean_path,
                    len(clean))
        return None

    upc = stage_io.units_per_cm(run_dir, clean, room_extent_cm=cfg["scale"]["provisional_extent_cm"])
    max_d = cfg["planes"]["distance_cm"] * upc
    nns = o3d.core.nns.NearestNeighborSearch(o3d.core.Tensor(np.ascontiguousarray(clean, dtype=np.float64)))
    nns.knn_index()
    idx, d2 = nns.knn_search(o3d.core.Tensor(np.ascontiguousarray(points_work, dtype=np.float64)), 1)
    idx, d2 = idx.numpy()[:, 0], d2.numpy()[:, 0]

    return np.where(d2 <= max_d * max_d, labels[idx], -1)

def region_row(name: str, err: np.ndarray, inlier_px: float) -> dict:

    err = err[np.isfinite(err)]
    if not len(err):
        return {"region": name, "n_points": 0, "reproj_mean_px": "", "inlier_frac": ""}

    return {"region": name, "n_points": int(len(err)), "reproj_mean_px": round(float(err.mean()), 4),
            "inlier_frac": round(float((err < inlier_px).mean()), 4)}

def run(cfg: dict, run_dir: str, artifacts_dir: str) -> list[dict] | None:

    ec = cfg["evaluation"]
    if not ec["enabled"] or not ec.get("hard_regions", False):
        log.info("hard_regions disabled")
        return None

    run_dir, artifacts_dir = Path(run_dir), Path(artifacts_dir)
    sfm = run_dir / "sfm"
    names = ["points3d", "obs", "K", "dist", "cam_R", "cam_t"]
    if not all((sfm / f"{n}.npy").exists() for n in names):
        log.warning("hard_regions: missing sfm arrays in %s, skipping", sfm)
        return None

    points3d, obs, K, dist, cam_R, cam_t = (np.load(sfm / f"{n}.npy") for n in names)
    if not len(obs):
        log.warning("hard_regions: no observations, skipping")
        return None

    err_path = run_dir / "evaluation" / "reproj_errors.npy"
    obs_err = np.load(err_path) if err_path.exists() else None
    if obs_err is None or len(obs_err) != len(obs):
        obs_err = observation_errors(points3d, obs, K, dist, cam_R, cam_t)

    n_pts = len(points3d)
    pi = obs[:, 0].astype(np.int64)
    fin = np.isfinite(obs_err)
    with np.errstate(invalid="ignore", divide="ignore"):
        point_err = (np.bincount(pi[fin], obs_err[fin], minlength=n_pts)
                     / np.bincount(pi[fin], minlength=n_pts))
    texture = group_median(pi, keypoint_density(obs, float(ec["texture_radius_px"])), n_pts)

    s, unit = stage_io.scale_info(run_dir)
    labels = point_labels(run_dir, cfg, points3d * s if unit == "cm" else points3d)
    if labels is None:
        log.warning("hard_regions: planes/labels.npy unavailable, every point is treated as wall")
        wall = np.ones(n_pts, dtype=bool)
    else:
        wall = labels >= 2

    wall &= np.isfinite(texture)
    if wall.any():
        threshold = float(np.quantile(texture[wall], ec["texture_split_quantile"]))
    else:
        log.warning("hard_regions: no wall points")
        threshold = float("nan")
    smooth, textured = wall & (texture <= threshold), wall & (texture > threshold)

    rows = [region_row(n, point_err[m], ec["inlier_px"]) for n, m in
            [("all", np.ones(n_pts, dtype=bool)), ("wall", wall), ("smooth_wall", smooth),
             ("textured_wall", textured)]]
    log.info("metric texture_threshold=%s", threshold)
    for row in rows:
        stage_io.log_metrics(log, {f"{row['region']}_{k}": v for k, v in row.items() if k != "region"})
    csv_path = artifacts_dir / "hard_regions.csv"
    csv_path.unlink(missing_ok=True)
    for row in rows:
        stage_io.upsert_csv(csv_path, row, key="region")

    return rows

if __name__ == "__main__":
    import tempfile, time

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    rng = np.random.default_rng(0)
    K = np.array([[800.0, 0, 640], [0, 800.0, 360], [0, 0, 1]])
    dist = np.array([0.01, -0.005, 0.0005, -0.0003])
    n_cam = 6
    cam_R = np.stack([cv2.Rodrigues(np.array([0, 0.03 * i, 0.0]))[0] for i in range(n_cam)])
    cam_t = np.array([[-0.2 * i, 0.0, 0.0] for i in range(n_cam)])

    # wall A (label 2, x < 0): dense keypoints, 0.3 px noise; wall B (label 3, x > 0): sparse, 3 px noise.
    pa = np.column_stack([rng.uniform(-0.6, -0.1, 1500), rng.uniform(-1, 1, 1500), np.full(1500, 5.0)])
    pb = np.column_stack([rng.uniform(0.1, 2.5, 1500), rng.uniform(-1, 1, 1500), np.full(1500, 5.0)])
    pts = np.vstack([pa, pb])
    noise = np.r_[np.full(len(pa), 0.3), np.full(len(pb), 3.0)]
    obs = []
    for c in range(n_cam):
        uv, _ = cv2.projectPoints(pts, cv2.Rodrigues(cam_R[c])[0], cam_t[c], K, dist)
        uv = uv.reshape(-1, 2) + rng.normal(size=(len(pts), 2)) * noise[:, None] / np.sqrt(2)
        obs.append(np.column_stack([np.arange(len(pts)), np.full(len(pts), c), uv]))
    obs = np.vstack(obs)

    cfg = {"evaluation": {"enabled": True, "hard_regions": True, "texture_radius_px": 30,
                          "texture_split_quantile": 0.5, "inlier_px": 2.0},
           "planes": {"source": "sparse", "distance_cm": 5.0}, "scale": {"provisional_extent_cm": 500}}
    with tempfile.TemporaryDirectory() as tmp:
        rd, ad = Path(tmp) / "results", Path(tmp) / "artifacts"
        (rd / "sfm").mkdir(parents=True)
        for n, a in dict(points3d=pts, obs=obs, K=K, dist=dist, cam_R=cam_R, cam_t=cam_t).items():
            np.save(rd / "sfm" / f"{n}.npy", a)

        rows = run(cfg, str(rd), str(ad))                          # planes missing -> all points are wall
        assert rows[1]["n_points"] == len(pts)

        s = 100.0                                                  # cm per model unit
        (rd / "scale").mkdir()
        np.save(rd / "scale" / "scale.npy", np.array([s]))
        keep = rng.permutation(len(pts))[: len(pts) // 2]          # voxelised subset, working units
        (rd / "clean" / "sparse").mkdir(parents=True)
        np.save(rd / "clean" / "sparse" / "points.npy", pts[keep] * s)
        (rd / "planes").mkdir()
        np.save(rd / "planes" / "labels.npy", np.where(pts[keep, 0] < 0, 2, 3).astype(np.int32))

        rows = {r["region"]: r for r in run(cfg, str(rd), str(ad))}
        assert rows["textured_wall"]["reproj_mean_px"] < rows["smooth_wall"]["reproj_mean_px"]
        assert rows["textured_wall"]["inlier_frac"] > rows["smooth_wall"]["inlier_frac"]
        assert rows["wall"]["n_points"] > 0.9 * len(pts)
        csv = {r["region"]: r for r in stage_io.read_csv(ad / "hard_regions.csv")}
        assert list(csv) == ["all", "wall", "smooth_wall", "textured_wall"]

        err = observation_errors(pts, obs, K, dist, cam_R, cam_t)  # precomputed errors path gives the same
        (rd / "evaluation").mkdir()
        np.save(rd / "evaluation" / "reproj_errors.npy", err)
        assert run(cfg, str(rd), str(ad))[3] == rows["textured_wall"]

    tiny = np.array([[0, 0, 0, 0], [1, 0, 10, 0], [2, 0, 100, 0], [3, 1, 5, 0]], dtype=float)
    assert keypoint_density(tiny, 30.0).tolist() == [1, 1, 0, 0]
    assert group_median(np.array([0, 0, 0, 2]), np.array([3, 1, 2, 7]), 3)[[0, 2]].tolist() == [2.0, 7.0]

    big = np.column_stack([np.repeat(np.arange(70000), 13), rng.integers(0, 150, 910000),  # 150 cameras
                           rng.uniform(0, 1920, 910000), rng.uniform(0, 1080, 910000)])
    t0 = time.time()
    tex = group_median(big[:, 0].astype(np.int64), keypoint_density(big, 30.0), 70000)
    dt = time.time() - t0
    print(f"texture for 70k points x 13 obs, 150 cameras: {dt:.2f} s, median {np.median(tex):.0f}")
    assert dt < 10
    print("OK")
# <<< END IMPLEMENTATION <<<

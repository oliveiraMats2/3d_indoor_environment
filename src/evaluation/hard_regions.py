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
# Goal: reconstruction quality in the hard cases of an indoor capture: low texture, glare (specular), poor
#       light, occlusion / depth boundaries, against the points that fall in none of them.
# In:  sfm/{points3d,obs,K,dist,cam_R,cam_t,image_names}.npy (model units); evaluation/reproj_errors.npy (O,)
#      aligned with obs rows if present (else computed with cv2.projectPoints); planes/labels.npy (per point of
#      clean/<planes.source>/points.npy, working units: -1 none, 0 floor, 1 ceiling, 2+k wall k); frames in
#      extraction.output_dir (observing images); dense/<method>/points.npy (model units); scale via
#      stage_io.scale_info, else room_fill.csv cm_per_model_unit, else the provisional extent.
# Out: artifacts/<id>/hard_regions.csv, one row per region: all, wall, smooth_wall, textured_wall (unchanged),
#      then glare, dark, low_texture, depth_boundary, low_visibility, occlusion, normal. Columns: region,
#      n_points, reproj_mean_px, inlier_frac (unchanged), reproj_median_px, frac_of_points, track_len_mean,
#      track_len_median, vis_ratio_median, dense_cover_frac, dense_nbrs_mean.
#      evaluation/hard_classes.npy (uint8 bit mask per sfm point, CLASS_BITS); figures/hard_regions.png.
# YAML: evaluation: {hard_regions: true, texture_radius_px: 30, texture_split_quantile: 0.5, inlier_px: 2.0,
#       hard_cases: {see HARD_DEFAULTS}} (also planes.source, planes.distance_cm, scale.provisional_extent_cm,
#       extraction.output_dir, dense.methods).
# Rules:
# - sfm points are scaled by s (raw when unit = model) and take the label of their nearest clean point if it
#   lies within planes.distance_cm * units_per_cm, else -1. Missing planes -> every point counts as wall
#   (WARNING). Nearest neighbours: open3d core NNS on CUDA:0 when available, CPU otherwise.
# - Texture of an observation = number of OTHER observations of the same camera within texture_radius_px
#   (uniform grid, cell = radius, exact distance check); texture of a point = median over its observations.
# - Wall points split at the texture_split_quantile: texture <= q -> smooth_wall, > q -> textured_wall.
#   low_texture (any point) = texture <= quantile low_texture_quantile of all points.
# - Image classes, per observation from its image (HSV V = max(RGB), S = (max - min) / max):
#   glare = some pixel within glare_radius_px has V >= glare_v and S <= glare_s; dark = mean luminance in a
#   (2 luma_radius_px + 1) box < dark_luma. A point is glare / dark when >= obs_frac of its imaged
#   observations are. Missing images -> those classes stay empty (WARNING).
# - depth_boundary: an observation is at a boundary when >= boundary_min_far neighbours of the same camera
#   within boundary_radius_px (and >= boundary_far_frac of them) differ in camera depth by a factor > 1 +
#   boundary_rel_depth; a point is when >= obs_frac of its observations are.
# - low_visibility: vis_ratio = track length / number of cameras whose frustum contains the point (inside
#   the image, depth within [min, max] observed depth widened by frustum_depth_margin, pinhole, no
#   z-buffer; torch CUDA when available) < vis_ratio. occlusion = depth_boundary | low_visibility.
#   normal = no hard class.
# - Point error = mean of its finite observation errors; reproj_mean/median_px over region points;
#   inlier_frac = fraction of region points with point error < inlier_px; dense_cover_frac = fraction with a
#   dense point within coverage_radius_cm; dense_nbrs_mean = mean dense points within it (capped at
#   coverage_knn). Empty cells when a region (or the dense cloud) is empty.

# >>> BEGIN IMPLEMENTATION >>>
"""Hard regions: reprojection error, track length and dense coverage per hard case (texture, glare, light,
occlusion)."""
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from src import stage_io

log = logging.getLogger(__name__)

HARD_DEFAULTS = {
    "glare_v": 0.95, "glare_s": 0.25, "glare_radius_px": 6,
    "dark_luma": 0.12, "luma_radius_px": 6,
    "obs_frac": 0.5,
    "boundary_radius_px": 20.0, "boundary_rel_depth": 0.3, "boundary_min_far": 2, "boundary_far_frac": 0.2,
    "vis_ratio": 0.03, "frustum_depth_margin": 1.5,
    "low_texture_quantile": 0.1,
    "coverage_radius_cm": 2.0, "coverage_knn": 16, "dense_method": None,
    "figure": True, "examples": 3, "workers": 8, "device": "auto",
}
CLASS_BITS = {"glare": 1, "dark": 2, "low_texture": 4, "depth_boundary": 8, "low_visibility": 16}

def hard_cfg(cfg: dict) -> dict:

    return {**HARD_DEFAULTS, **(cfg["evaluation"].get("hard_cases") or {})}

def use_cuda(hc: dict) -> bool:

    return hc.get("device", "auto") != "cpu" and o3d.core.cuda.is_available()

def knn(ref: np.ndarray, query: np.ndarray, k: int, cuda: bool = True,
        chunk: int = 500_000) -> tuple[np.ndarray, np.ndarray]:
    """(idx (Q,k), squared distance (Q,k)) of the k nearest ref points; open3d core NNS on CUDA:0 when
    available (float32), CPU fallback."""

    k = int(min(k, len(ref)))
    if cuda and o3d.core.cuda.is_available():
        try:
            dev = o3d.core.Device("CUDA:0")
            nns = o3d.core.nns.NearestNeighborSearch(
                o3d.core.Tensor(np.ascontiguousarray(ref, dtype=np.float32), device=dev))
            nns.knn_index()
            out = [nns.knn_search(o3d.core.Tensor(np.ascontiguousarray(query[i:i + chunk], dtype=np.float32),
                                                  device=dev), k) for i in range(0, len(query), chunk)]
            return (np.concatenate([o[0].cpu().numpy() for o in out]).astype(np.int64),
                    np.concatenate([o[1].cpu().numpy() for o in out]).astype(np.float64))
        except RuntimeError as exc:                                 # e.g. out of memory -> CPU
            log.warning("hard_regions: CUDA NNS failed (%s), using CPU", exc)

    nns = o3d.core.nns.NearestNeighborSearch(o3d.core.Tensor(np.ascontiguousarray(ref, dtype=np.float64)))
    nns.knn_index()
    idx, d2 = nns.knn_search(o3d.core.Tensor(np.ascontiguousarray(query, dtype=np.float64)), k)

    return idx.numpy().astype(np.int64), d2.numpy()

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

def observation_depth(points3d, obs, cam_R, cam_t) -> np.ndarray:
    """(O,) camera-frame z of every observed point."""

    pi, ci = obs[:, 0].astype(np.int64), obs[:, 1].astype(np.int64)

    return np.einsum("ij,ij->i", cam_R[ci, 2, :], points3d[pi]) + cam_t[ci, 2]

def neighbour_pairs(obs: np.ndarray, radius: float, chunk: int = 1_000_000):
    """Yields (src, dst) observation index pairs of the same camera within radius pixels (self pairs
    included); uniform grid with cell = radius, 3x3 cells, processed in chunks of src to bound memory."""

    cam = obs[:, 1].astype(np.int64)
    cell = np.floor(obs[:, 2:4] / radius).astype(np.int64)
    cell -= cell.min(axis=0) - 1                                   # >= 1, so the -1 neighbours stay valid
    gx, gy = int(cell[:, 0].max()) + 2, int(cell[:, 1].max()) + 2
    key = (cam * gy + cell[:, 1]) * gx + cell[:, 0]
    order = np.argsort(key, kind="stable")
    skey, sxy = key[order], obs[order, 2:4]
    r2 = radius * radius
    for a in range(0, len(obs), chunk):
        qk, qxy = key[a:a + chunk], obs[a:a + chunk, 2:4]
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                nk = qk + dy * gx + dx
                lo, hi = np.searchsorted(skey, nk, "left"), np.searchsorted(skey, nk, "right")
                n = hi - lo
                if not n.any():
                    continue
                src = np.repeat(np.arange(len(qk)), n)
                dst = np.repeat(lo, n) + np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
                keep = ((qxy[src] - sxy[dst]) ** 2).sum(axis=1) <= r2
                yield src[keep] + a, order[dst[keep]]

def keypoint_density(obs: np.ndarray, radius: float) -> np.ndarray:
    """(O,) number of other observations of the same camera within radius pixels (grid, 3x3 cells)."""

    counts = np.zeros(len(obs), dtype=np.int64)
    for src, _ in neighbour_pairs(obs, radius):
        counts += np.bincount(src, minlength=len(obs))

    return counts - 1                                              # remove the observation itself

def depth_boundary_obs(obs: np.ndarray, depth: np.ndarray, hc: dict) -> np.ndarray:
    """(O,) bool: observation lies next to a depth discontinuity (neighbours at a different depth)."""

    n_nb = np.zeros(len(obs), dtype=np.int64)
    n_far = np.zeros(len(obs), dtype=np.int64)
    thr = np.log1p(float(hc["boundary_rel_depth"]))
    logz = np.log(np.where(depth > 0, depth, np.nan))
    for src, dst in neighbour_pairs(obs, float(hc["boundary_radius_px"])):
        other = src != dst
        src, dst = src[other], dst[other]
        n_nb += np.bincount(src, minlength=len(obs))
        with np.errstate(invalid="ignore"):
            far = np.abs(logz[src] - logz[dst]) > thr
        n_far += np.bincount(src[far], minlength=len(obs))

    with np.errstate(invalid="ignore", divide="ignore"):
        return (n_far >= int(hc["boundary_min_far"])) & (n_far / np.maximum(n_nb, 1) >= hc["boundary_far_frac"])

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

def group_frac(groups: np.ndarray, flag: np.ndarray, n_groups: int) -> np.ndarray:
    """(n_groups,) fraction of the finite flag values (0/1, NaN = unknown) per group; NaN when none."""

    ok = np.isfinite(flag)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (np.bincount(groups[ok], flag[ok], minlength=n_groups)
                / np.bincount(groups[ok], minlength=n_groups))

def image_masks(img: np.ndarray, hc: dict) -> tuple[np.ndarray, np.ndarray]:
    """(glare_near bool, luma float in [0, 1] box mean) of a BGR uint8 image."""

    mx, mn = img.max(axis=2), img.min(axis=2)
    sat = (mx.astype(np.float32) - mn) / np.maximum(mx, 1)
    glare = ((mx >= hc["glare_v"] * 255) & (sat <= hc["glare_s"])).astype(np.uint8)
    r = int(hc["glare_radius_px"])
    if r > 0:
        glare = cv2.dilate(glare, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    w = 2 * int(hc["luma_radius_px"]) + 1
    luma = cv2.blur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0, (w, w))

    return glare.astype(bool), luma

def image_samples(obs: np.ndarray, image_paths: list[Path], hc: dict) -> tuple[np.ndarray, np.ndarray, tuple]:
    """(glare (O,) float 0/1/NaN, luma (O,) NaN when the image is missing, (w, h) of the images)."""

    ci = obs[:, 1].astype(np.int64)
    glare, luma = np.full(len(obs), np.nan), np.full(len(obs), np.nan)
    order = np.argsort(ci, kind="stable")
    cams, starts = np.unique(ci[order], return_index=True)
    size = [None]

    def one(item):
        cam, rows = item
        img = cv2.imread(str(image_paths[cam]), cv2.IMREAD_COLOR) if cam < len(image_paths) else None
        if img is None:
            return
        g, l = image_masks(img, hc)
        h, w = l.shape
        size[0] = (w, h)
        u = np.clip(np.round(obs[rows, 2]).astype(np.int64), 0, w - 1)
        v = np.clip(np.round(obs[rows, 3]).astype(np.int64), 0, h - 1)
        glare[rows], luma[rows] = g[v, u], l[v, u]

    with ThreadPoolExecutor(max_workers=int(hc["workers"])) as pool:
        list(pool.map(one, zip(cams, np.split(order, starts[1:]))))

    return glare, luma, size[0]

def frustum_counts(points: np.ndarray, K: np.ndarray, cam_R: np.ndarray, cam_t: np.ndarray, size: tuple,
                   zmin: np.ndarray, zmax: np.ndarray, cuda: bool) -> np.ndarray:
    """(N,) number of cameras whose pinhole frustum holds the point within [zmin, zmax] (no z-buffer)."""

    w, h = size
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    try:
        import torch
        dev = torch.device("cuda:0") if cuda and torch.cuda.is_available() else None
    except ImportError:
        torch, dev = None, None

    if dev is not None:
        P = torch.as_tensor(points, dtype=torch.float32, device=dev)
        lo, hi = (torch.as_tensor(a, dtype=torch.float32, device=dev) for a in (zmin, zmax))
        count = torch.zeros(len(points), dtype=torch.int32, device=dev)
        batch = max(1, int(2e8 // max(len(points) * 12, 1)))
        for a in range(0, len(cam_R), batch):
            R = torch.as_tensor(cam_R[a:a + batch], dtype=torch.float32, device=dev)
            t = torch.as_tensor(cam_t[a:a + batch], dtype=torch.float32, device=dev)
            X = torch.einsum("bij,nj->bni", R, P) + t[:, None, :]
            z = X[..., 2]
            u, v = fx * X[..., 0] / z + cx, fy * X[..., 1] / z + cy
            inside = (z > lo) & (z < hi) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
            count += inside.sum(dim=0, dtype=torch.int32)
        out = count.cpu().numpy().astype(np.int64)
        del P, lo, hi, count
        torch.cuda.empty_cache()
        return out

    count = np.zeros(len(points), dtype=np.int64)
    for R, t in zip(cam_R, cam_t):
        X = points @ R.T + t
        z = X[:, 2]
        with np.errstate(invalid="ignore", divide="ignore"):
            u, v = fx * X[:, 0] / z + cx, fy * X[:, 1] / z + cy
            count += (z > zmin) & (z < zmax) & (u >= 0) & (u < w) & (v >= 0) & (v < h)

    return count

def point_labels(run_dir: Path, cfg: dict, points_work: np.ndarray, cuda: bool = True) -> np.ndarray | None:
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
    idx, d2 = knn(clean, points_work, 1, cuda)
    idx, d2 = idx[:, 0], d2[:, 0]

    return np.where(d2 <= max_d * max_d, labels[idx], -1)

def model_units_per_cm(run_dir: Path, artifacts_dir: Path, cfg: dict, points: np.ndarray) -> float:
    """Model units per cm: scale.npy, else the room_fill outline fit, else the provisional extent."""

    s, unit = stage_io.scale_info(run_dir)
    if unit == "cm":
        return 1.0 / s
    rows = stage_io.read_csv(artifacts_dir / "room_fill.csv")
    try:
        cm = float(rows[-1]["cm_per_model_unit"]) if rows else float("nan")
    except (KeyError, ValueError):
        cm = float("nan")
    if np.isfinite(cm) and cm > 0:
        log.info("hard_regions: coverage radius from the room_fill fit (%.4g cm per model unit)", cm)
        return 1.0 / cm

    return stage_io.units_per_cm(run_dir, points, room_extent_cm=cfg["scale"]["provisional_extent_cm"])

def dense_points(run_dir: Path, cfg: dict, hc: dict) -> tuple[str, np.ndarray | None]:
    """(method, points in model units) of hard_cases.dense_method or the first dense.methods with output."""

    methods = [hc["dense_method"]] if hc.get("dense_method") else list((cfg.get("dense") or {}).get("methods") or [])
    for m in methods:
        p = run_dir / "dense" / m / "points.npy"
        if p.exists():
            return m, np.load(p)

    return "", None

def region_row(name: str, mask: np.ndarray, stats: dict, inlier_px: float) -> dict:

    keys = ["reproj_mean_px", "inlier_frac", "reproj_median_px", "frac_of_points", "track_len_mean",
            "track_len_median", "vis_ratio_median", "dense_cover_frac", "dense_nbrs_mean"]
    err = stats["err"][mask]
    fin = np.isfinite(err)
    row = {"region": name, "n_points": int(fin.sum()), **{k: "" for k in keys}}
    if not fin.any():
        return row

    def r(x):
        return round(float(x), 4)

    err, m = err[fin], np.flatnonzero(mask)[fin]
    track, vis = stats["track"][m], stats["vis"][m]
    row.update(reproj_mean_px=r(err.mean()), inlier_frac=r((err < inlier_px).mean()),
               reproj_median_px=r(np.median(err)), frac_of_points=r(len(m) / stats["n_valid"]),
               track_len_mean=r(track.mean()), track_len_median=r(np.median(track)))
    if np.isfinite(vis).any():
        row["vis_ratio_median"] = r(np.nanmedian(vis))
    if stats["cover"] is not None:
        row["dense_cover_frac"], row["dense_nbrs_mean"] = r(stats["cover"][m].mean()), r(stats["nbrs"][m].mean())

    return row

def example_figure(path: Path, obs: np.ndarray, classes: np.ndarray, image_paths: list[Path], hc: dict,
                   names: np.ndarray) -> Path | None:
    """Image crops of the cameras that see the most glare, dark and depth boundary points, masks overlaid."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pi, ci = obs[:, 0].astype(np.int64), obs[:, 1].astype(np.int64)
    n_cam = int(ci.max()) + 1
    panels = []
    occl = CLASS_BITS["depth_boundary"]                           # low_visibility is not visible in one image
    for title, bits in [("glare", CLASS_BITS["glare"]), ("dark", CLASS_BITS["dark"]), ("depth boundary", occl)]:
        hit = (classes[pi] & bits) > 0
        per_cam = np.bincount(ci[hit], minlength=n_cam)
        picked = []
        for c in np.argsort(-per_cam, kind="stable"):            # skip near-duplicate neighbouring frames
            if per_cam[c] == 0 or len(picked) >= int(hc["examples"]):
                break
            if all(abs(int(c) - p) > 10 for p in picked):
                picked.append(int(c))
        panels += [(title, bits, c) for c in picked]
    if not panels:
        return None

    cols = int(hc["examples"])
    titles = list(dict.fromkeys(p[0] for p in panels))
    fig, axes = plt.subplots(len(titles), cols, figsize=(4.2 * cols, 3.0 * len(titles)), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for title, bits, c in panels:
        ax = axes[titles.index(title), [p[2] for p in panels if p[0] == title].index(c)]
        img = cv2.imread(str(image_paths[c]), cv2.IMREAD_COLOR)
        if img is None:
            continue
        glare, luma = image_masks(img, hc)
        rgb = img[..., ::-1].astype(np.float32) / 255.0
        rgb[glare] = 0.45 * rgb[glare] + 0.55 * np.array([1.0, 0.1, 0.1])
        dark = luma < hc["dark_luma"]
        rgb[dark] = 0.45 * rgb[dark] + 0.55 * np.array([0.1, 0.3, 1.0])
        rows = ci == c
        uv, hit = obs[rows, 2:4], (classes[pi[rows]] & bits) > 0
        h, w = luma.shape
        cw, chh = min(w, int(0.45 * w)), min(h, int(0.45 * w * 0.66))
        x0 = int(np.clip(np.median(uv[hit, 0]) - cw / 2, 0, w - cw))
        y0 = int(np.clip(np.median(uv[hit, 1]) - chh / 2, 0, h - chh))
        ax.imshow(np.clip(rgb[y0:y0 + chh, x0:x0 + cw], 0, 1))
        for sel, col, sz in [(~hit, "#d0d0d0", 2), (hit, "#ffd400", 6)]:
            q = uv[sel]
            q = q[(q[:, 0] >= x0) & (q[:, 0] < x0 + cw) & (q[:, 1] >= y0) & (q[:, 1] < y0 + chh)]
            ax.scatter(q[:, 0] - x0, q[:, 1] - y0, s=sz, c=col, linewidths=0)
        ax.set_title(f"{title}: {names[c] if c < len(names) else c} ({int(hit.sum())} pts)", fontsize=9)
    fig.suptitle("hard cases: red = glare mask, blue = dark mask, yellow = class keypoints, grey = others",
                 fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)

    return path

def run(cfg: dict, run_dir: str, artifacts_dir: str) -> list[dict] | None:

    ec = cfg["evaluation"]
    if not ec["enabled"] or not ec.get("hard_regions", False):
        log.info("hard_regions disabled")
        return None

    hc = hard_cfg(cfg)
    cuda = use_cuda(hc)
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

    tic = time.time()
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
    labels = point_labels(run_dir, cfg, points3d * s if unit == "cm" else points3d, cuda)
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
    t_texture = time.time() - tic

    # hard classes ------------------------------------------------------------------------------------------
    tic = time.time()
    track = np.bincount(pi, minlength=n_pts)
    classes = np.zeros(n_pts, dtype=np.uint8)
    obs_frac = float(hc["obs_frac"])
    has_tex = np.isfinite(texture)
    low_tex_thr = float(np.quantile(texture[has_tex], hc["low_texture_quantile"])) if has_tex.any() else np.nan
    classes[has_tex & (texture <= low_tex_thr)] |= CLASS_BITS["low_texture"]

    image_names = np.load(sfm / "image_names.npy") if (sfm / "image_names.npy").exists() else np.array([])
    img_dir = Path(hc.get("images_dir") or (cfg.get("extraction") or {}).get("output_dir") or "")
    image_paths = [img_dir / str(n) for n in image_names]
    size = None
    if len(image_paths) and img_dir != Path("") and image_paths[0].exists():
        g_obs, l_obs, size = image_samples(obs, image_paths, hc)
        with np.errstate(invalid="ignore"):
            dark_obs = np.where(np.isfinite(l_obs), (l_obs < hc["dark_luma"]).astype(float), np.nan)
        g_pt, d_pt = group_frac(pi, g_obs, n_pts), group_frac(pi, dark_obs, n_pts)
        with np.errstate(invalid="ignore"):
            classes[g_pt >= obs_frac] |= CLASS_BITS["glare"]
            classes[d_pt >= obs_frac] |= CLASS_BITS["dark"]
        log.info("hard_regions: images sampled (%d/%d observations) in %.1f s", int(np.isfinite(l_obs).sum()),
                 len(obs), time.time() - tic)
    else:
        log.warning("hard_regions: observing images not found in '%s' (sfm/image_names.npy), glare/dark empty",
                    img_dir)
    t_images = time.time() - tic

    tic = time.time()
    depth = observation_depth(points3d, obs, cam_R, cam_t)
    b_pt = group_frac(pi, depth_boundary_obs(obs, depth, hc).astype(float), n_pts)
    with np.errstate(invalid="ignore"):
        classes[b_pt >= obs_frac] |= CLASS_BITS["depth_boundary"]

    zmin, zmax = np.full(n_pts, np.inf), np.zeros(n_pts)
    ok = depth > 0
    np.minimum.at(zmin, pi[ok], depth[ok])
    np.maximum.at(zmax, pi[ok], depth[ok])
    margin = float(hc["frustum_depth_margin"])
    if size is None:
        size = (int(round(2 * K[0, 2])), int(round(2 * K[1, 2])))
    n_frustum = frustum_counts(points3d, K, cam_R, cam_t, size, zmin / margin, zmax * margin, cuda)
    with np.errstate(invalid="ignore", divide="ignore"):
        vis = np.where(track > 0, track / np.maximum(n_frustum, track), np.nan)
        classes[vis < hc["vis_ratio"]] |= CLASS_BITS["low_visibility"]
    t_occlusion = time.time() - tic

    tic = time.time()
    method, dense = dense_points(run_dir, cfg, hc)
    cover = nbrs = None
    if dense is not None and len(dense):
        r = float(hc["coverage_radius_cm"]) * model_units_per_cm(run_dir, artifacts_dir, cfg, points3d)
        _, d2 = knn(dense, points3d, int(hc["coverage_knn"]), cuda)
        cover, nbrs = d2[:, 0] <= r * r, (d2 <= r * r).sum(axis=1)
        log.info("hard_regions: dense coverage from %s (%d points, radius %.4g model units)", method, len(dense), r)
    else:
        log.warning("hard_regions: no dense cloud, dense coverage columns empty")
    t_dense = time.time() - tic

    stats = {"err": point_err, "track": track, "vis": vis, "cover": cover, "nbrs": nbrs,
             "n_valid": max(int(np.isfinite(point_err).sum()), 1)}
    bit = {k: (classes & v) > 0 for k, v in CLASS_BITS.items()}
    regions = [("all", np.ones(n_pts, dtype=bool)), ("wall", wall), ("smooth_wall", smooth),
               ("textured_wall", textured), ("glare", bit["glare"]), ("dark", bit["dark"]),
               ("low_texture", bit["low_texture"]), ("depth_boundary", bit["depth_boundary"]),
               ("low_visibility", bit["low_visibility"]),
               ("occlusion", bit["depth_boundary"] | bit["low_visibility"]), ("normal", classes == 0)]
    rows = [region_row(n, m, stats, ec["inlier_px"]) for n, m in regions]

    log.info("metric texture_threshold=%s", threshold)
    log.info("metric low_texture_threshold=%s", low_tex_thr)
    log.info("hard_regions: times texture %.1f s, images %.1f s, occlusion %.1f s, dense %.1f s (%s)",
             t_texture, t_images, t_occlusion, t_dense, "CUDA" if cuda else "CPU")
    for row in rows:
        stage_io.log_metrics(log, {f"{row['region']}_{k}": v for k, v in row.items() if k != "region"})
    csv_path = artifacts_dir / "hard_regions.csv"
    csv_path.unlink(missing_ok=True)
    for row in rows:
        stage_io.upsert_csv(csv_path, row, key="region")

    (run_dir / "evaluation").mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "evaluation" / "hard_classes.npy", classes)
    if hc["figure"] and image_paths and (classes & (CLASS_BITS["glare"] | CLASS_BITS["dark"])).any():
        fig = example_figure(run_dir / "figures" / "hard_regions.png", obs, classes, image_paths, hc, image_names)
        if fig:
            log.info("hard_regions: figure %s", fig)

    return rows

if __name__ == "__main__":
    import tempfile

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

        rows = run(cfg, str(rd), str(ad))                          # planes, images, dense missing
        assert rows[1]["n_points"] == len(pts)
        assert {r["region"]: r for r in rows}["glare"]["n_points"] == 0

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
        assert list(csv)[:4] == ["all", "wall", "smooth_wall", "textured_wall"]
        assert list(next(iter(csv.values())))[:4] == ["region", "n_points", "reproj_mean_px", "inlier_frac"]

        err = observation_errors(pts, obs, K, dist, cam_R, cam_t)  # precomputed errors path gives the same
        (rd / "evaluation").mkdir(exist_ok=True)
        np.save(rd / "evaluation" / "reproj_errors.npy", err)
        assert run(cfg, str(rd), str(ad))[3] == rows["textured_wall"]

        # images: the projection of the world strip -0.5 < x < -0.3 is saturated white (glare), the image
        # below the projection of y = 0.6 is dark, the rest is textured grey.
        (rd / "frames").mkdir()
        names = np.array([f"img_{c:04d}.png" for c in range(n_cam)])
        np.save(rd / "sfm" / "image_names.npy", names)
        for c, n in enumerate(names):
            ref, _ = cv2.projectPoints(np.array([[-0.5, 0.6, 5.0], [-0.3, 0.6, 5.0]]), cv2.Rodrigues(cam_R[c])[0],
                                       cam_t[c], K, dist)
            (u0, v0), (u1, _) = ref.reshape(-1, 2).astype(int)
            im = rng.integers(60, 200, (720, 1280, 3)).astype(np.uint8)
            im[:, u0:u1] = 255
            im[v0:] //= 8
            cv2.imwrite(str(rd / "frames" / n), im)
        # dense cloud only on wall A (x < 0)
        (rd / "dense" / "patch_match").mkdir(parents=True)
        np.save(rd / "dense" / "patch_match" / "points.npy",
                np.column_stack([rng.uniform(-0.6, -0.1, 40000), rng.uniform(-1, 1, 40000), np.full(40000, 5.0)]))
        cfg.update(extraction={"output_dir": str(rd / "frames")}, dense={"methods": ["patch_match"]})
        rows = {r["region"]: r for r in run(cfg, str(rd), str(ad))}
        classes = np.load(rd / "evaluation" / "hard_classes.npy")
        in_glare = (pts[:, 0] > -0.48) & (pts[:, 0] < -0.32) & (pts[:, 1] < 0.55)
        in_dark = (pts[:, 1] > 0.7) & ((pts[:, 0] < -0.55) | (pts[:, 0] > -0.25))  # dimmed white = 0.12
        plain = ((pts[:, 0] < -0.55) | (pts[:, 0] > -0.25)) & (pts[:, 1] < 0.55)
        assert ((classes[in_glare] & CLASS_BITS["glare"]) > 0).mean() > 0.9
        assert ((classes[plain] & CLASS_BITS["glare"]) > 0).mean() < 0.05
        assert ((classes[in_dark] & CLASS_BITS["dark"]) > 0).mean() > 0.9
        assert ((classes[plain] & CLASS_BITS["dark"]) > 0).mean() < 0.05
        assert rows["glare"]["n_points"] > 0 and rows["dark"]["n_points"] > 0
        assert rows["textured_wall"]["dense_cover_frac"] > 0.9 > rows["smooth_wall"]["dense_cover_frac"]
        assert rows["normal"]["n_points"] > 0
        assert (rd / "figures" / "hard_regions.png").exists()

    # depth boundary: a near cluster (z = 2) in front of a far wall (z = 10), one camera
    near = np.column_stack([rng.uniform(100, 200, 50), rng.uniform(100, 200, 50)])
    far = np.column_stack([rng.uniform(0, 400, 400), rng.uniform(0, 400, 400)])
    bobs = np.column_stack([np.arange(450), np.zeros(450), np.vstack([near, far])])
    bz = np.r_[np.full(50, 2.0), np.full(400, 10.0)]
    bnd = depth_boundary_obs(bobs, bz, HARD_DEFAULTS)
    xy = bobs[50:, 2:4]
    edge = (np.abs(xy - 150).max(axis=1) < 60)                     # far points around the near cluster
    away = (np.abs(xy - 150).max(axis=1) > 100)
    assert bnd[50:][edge].mean() > 0.5 and not bnd[50:][away].any()

    # frustum counts: CUDA (if present) == CPU
    P = rng.uniform(-1, 1, (2000, 3)) + [0, 0, 5]
    zl, zh = np.full(2000, 1.0), np.full(2000, 20.0)
    c_cpu = frustum_counts(P, K, cam_R, cam_t, (1280, 720), zl, zh, cuda=False)
    c_gpu = frustum_counts(P, K, cam_R, cam_t, (1280, 720), zl, zh, cuda=True)
    assert np.abs(c_cpu - c_gpu).sum() <= 2 and c_cpu.max() == n_cam
    i_c, d_c = knn(P, P[:100] + 1e-3, 1, cuda=True)
    i_n, _ = knn(P, P[:100] + 1e-3, 1, cuda=False)
    assert (i_c[:, 0] == np.arange(100)).all() and (i_n[:, 0] == np.arange(100)).all()

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

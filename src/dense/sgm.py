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
# Stage: dense (method sgm) - classical Semi-Global Matching (Hirschmuller) on rectified image pairs.
# Goal: per reference view, pick 1-2 sources with a good baseline, stereo-rectify each pair, match with a census
#      cost + SGM path aggregation (torch on the GPU, or OpenCV StereoSGBM on the CPU), left-right check, map
#      the disparity back to the reference camera as z depth and fuse with the shared COLMAP stereo_fusion.
# In:  <run_dir>/dense/undistorted (shared PINHOLE workspace), sparse depth ranges and ranked sources.
# Out: <run_dir>/dense/sgm/{ws/stereo/{depth_maps,normal_maps}, fused.ply, points.npy, colors.npy};
#      row method=sgm in <artifacts_dir>/dense_metrics.csv (t_depth_s, vram_peak_gb).
# YAML: dense.sgm.{backend (torch|opencv), max_size, num_pairs, p1, p2, census_window, paths, lr_max_diff,
#      min_angle_deg, max_angle_deg, max_disparities, min_valid_frac, depth_rel_tol, speckle_size,
#      normal_radius}; dense.{ref_step, fusion.*, rerun, gpu_index}.
# Rules: reject degenerate pairs (rectified valid area or reference coverage < min_valid_frac, epipole inside
#      the image); depth is reference-camera z (0 = invalid); pixels seen by >= 2 pairs keep only the
#      consistent depths; no CUDA for the torch backend -> WARNING and the OpenCV backend.

# >>> BEGIN IMPLEMENTATION >>>
"""Semi-Global Matching densifier: rectified pairs, census + SGM aggregation, LR check, depth-map fusion."""
import logging
import math
import warnings
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from src.dense import depth_fusion

log = logging.getLogger(__name__)

METHOD = "sgm"
DEFAULTS = {"backend": "torch", "max_size": 800, "num_pairs": 2, "p1": 10, "p2": 120, "census_window": 7,
            "paths": 8, "lr_max_diff": 1, "min_angle_deg": 3.0, "max_angle_deg": 20.0, "max_disparities": 256,
            "min_valid_frac": 0.5, "depth_rel_tol": 0.02, "speckle_size": 100, "normal_radius": 6}
P2_GRAD = 25.0 / 255.0  # intensity step that halves the adaptive P2
MAX_TRIES = 60  # source candidates tried per reference
MAX_AXIS_DEG = 45.0

def triangulation_angle(ref: depth_fusion.View, src: depth_fusion.View) -> float:
    """Angle (deg) at the point on the reference optical axis at the geometric-mean sparse depth."""

    z = math.sqrt(ref.depth_min * ref.depth_max)
    p = ref.center + z * ref.R[2]
    a, b = p - ref.center, p - src.center
    return float(np.degrees(np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12), -1, 1))))

def rectify(K1, R1w, t1, K2, R2w, t2, size, min_valid_frac=0.5) -> dict | None:
    """Rectification of the pair (image 1 = reference) at image size (w, h); None when degenerate.
    Returns remap tables, the rectified intrinsics, rotation R1 (ref camera -> rectified), fB (focal x baseline)
    and the layout (vertical epipolar lines -> transpose, source on the left -> flip)."""

    w, h = size
    c2 = -R2w.T @ t2
    e = K1 @ (R1w @ c2 + t1)
    if abs(e[2]) > 1e-9 * np.linalg.norm(e):
        ex, ey = e[:2] / e[2]
        if 0 <= ex < w and 0 <= ey < h:
            return None
    R = R2w @ R1w.T
    T = t2 - R @ t1
    zero = np.zeros(5)
    Rr1, Rr2, P1, P2, _, _, _ = cv2.stereoRectify(K1, zero, K2, zero, (w, h), R, T.reshape(3, 1),
                                                  flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1)
    vertical = abs(P2[1, 3]) > abs(P2[0, 3])
    tx = P2[1, 3] if vertical else P2[0, 3]
    if abs(tx) < 1e-12:
        return None
    m1 = cv2.initUndistortRectifyMap(K1, zero, Rr1, P1, (w, h), cv2.CV_32FC1)
    m2 = cv2.initUndistortRectifyMap(K2, zero, Rr2, P2, (w, h), cv2.CV_32FC1)
    ones = np.ones((h, w), np.float32)
    valid1 = cv2.remap(ones, *m1, cv2.INTER_NEAREST, borderValue=0) > 0
    # forward map of the reference grid into the rectified image (also used to map depth back)
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    rays = np.stack([u, v, np.ones_like(u)], -1) @ (Rr1 @ np.linalg.inv(K1)).T
    p = rays @ P1[:, :3].T
    with np.errstate(divide="ignore", invalid="ignore"):
        ur, vr = p[..., 0] / p[..., 2], p[..., 1] / p[..., 2]
    inside = (p[..., 2] > 0) & (ur >= 0) & (ur <= w - 1) & (vr >= 0) & (vr <= h - 1)
    if valid1.mean() < min_valid_frac or inside.mean() < min_valid_frac:
        return None
    return {"map1": m1, "map2": m2, "valid1": valid1, "Kn": P1[:, :3], "fB": float(abs(tx)),
            "vertical": bool(vertical), "flip": bool(tx > 0), "ur": ur.astype(np.float32),
            "vr": vr.astype(np.float32), "ray_z": rays[..., 2].astype(np.float32), "inside": inside}

def _to_match(a: np.ndarray, rect: dict) -> np.ndarray:
    """Rectified image -> matching layout (horizontal epipolar lines, reference = left image)."""

    a = a.T if rect["vertical"] else a
    return np.ascontiguousarray(a[:, ::-1] if rect["flip"] else a)

def _from_match(a: np.ndarray, rect: dict) -> np.ndarray:

    a = a[:, ::-1] if rect["flip"] else a
    return np.ascontiguousarray(a.T if rect["vertical"] else a)

def census(img, window: int):
    """Census transform of a (H, W) torch image packed into (n_bytes, H, W) uint8 (bit = neighbour < centre)."""

    import torch
    import torch.nn.functional as F

    r = window // 2
    h, w = img.shape
    pad = F.pad(img[None, None], (r, r, r, r), mode="replicate")[0, 0]
    bits = [pad[r + dy:r + dy + h, r + dx:r + dx + w] < img
            for dy in range(-r, r + 1) for dx in range(-r, r + 1) if dy or dx]
    bits += [torch.zeros_like(bits[0])] * (-len(bits) % 8)
    b = torch.stack(bits).view(-1, 8, h, w).to(torch.uint8)
    weights = (2 ** torch.arange(8, device=img.device, dtype=torch.uint8)).view(1, 8, 1, 1)
    return (b * weights).sum(1).to(torch.uint8), window * window - 1

def cost_volume(cl, cr, valid_l, valid_r, disps, n_bits):
    """(H, W, D) Hamming cost of the left census against the right one shifted by each disparity; pixels whose
    match falls outside the right image (or its valid area) get the maximum cost."""

    import torch

    lut = torch.tensor([bin(i).count("1") for i in range(256)], device=cl.device, dtype=torch.uint8)
    _, h, w = cl.shape
    C = torch.full((len(disps), h, w), float(n_bits), device=cl.device)
    for k, d in enumerate(disps):
        if d >= w:
            continue
        hd = lut[(cl[:, :, d:] ^ cr[:, :, :w - d]).long()].sum(0, dtype=torch.int16).float()
        C[k, :, d:] = torch.where(valid_r[:, :w - d], hd, float(n_bits))
    C[:, ~valid_l] = float(n_bits)
    return C.permute(1, 2, 0).contiguous()

def _scan(C, S, I, p1, p2, reverse: bool, shift: int) -> None:
    """One SGM path over axis 1 of (A, B, D) views: predecessor of (a, b) is (a - shift, b -/+ 1); S += L."""

    import torch

    A, B, D = C.shape
    prev = prev_i = None
    for b in (range(B - 1, -1, -1) if reverse else range(B)):
        c, i = C[:, b], I[:, b]
        if prev is None:
            L = c.clone()
        else:
            if shift:
                prev, prev_i = torch.roll(prev, shift, 0), torch.roll(prev_i, shift, 0)
                edge = slice(0, shift) if shift > 0 else slice(A + shift, A)
                prev[edge] = 0  # no predecessor: L = C
            m = prev.amin(1, keepdim=True)
            p2e = torch.clamp(p2 / (1 + (i - prev_i).abs()[:, None] / P2_GRAD), min=p1)
            best = torch.minimum(prev, m + p2e)
            best[:, 1:] = torch.minimum(best[:, 1:], prev[:, :-1] + p1)
            best[:, :-1] = torch.minimum(best[:, :-1], prev[:, 1:] + p1)
            L = c + best - m
        S[:, b] += L
        prev, prev_i = L, i

def aggregate(C, img, p1: float, p2: float, paths: int):
    """Sum of the 4 (horizontal + vertical) or 8 (+ diagonal) SGM path costs of an (H, W, D) volume."""

    import torch

    S = torch.zeros_like(C)
    shifts = (0, 1, -1) if paths >= 8 else (0,)
    for reverse in (False, True):
        for s in shifts:
            _scan(C, S, img, p1, p2, reverse, s)
        _scan(C.transpose(0, 1), S.transpose(0, 1), img.T, p1, p2, reverse, 0)
    return S

def sgm_torch(left, right, valid_l, valid_r, d0: int, step: int, n: int, p: dict, device) -> np.ndarray:
    """Disparity (float32, NaN invalid) of the left image: census + SGM + WTA/parabola + left-right check."""

    import torch

    with torch.no_grad():
        L = torch.from_numpy(left).to(device)
        Rt = torch.from_numpy(right).to(device)
        vl = torch.from_numpy(valid_l).to(device)
        vr = torch.from_numpy(valid_r).to(device)
        cl, n_bits = census(L, int(p["census_window"]))
        cr, _ = census(Rt, int(p["census_window"]))
        disps = [d0 + k * step for k in range(n)]
        C = cost_volume(cl, cr, vl, vr, disps, n_bits)
        del cl, cr
        S = aggregate(C, L, float(p["p1"]), float(p["p2"]), int(p["paths"]))
        del C
        h, w, D = S.shape
        k = S.argmin(-1)
        kc = k.clamp(1, D - 2)[..., None]
        s0, sm, sp = S.gather(2, kc).squeeze(-1), S.gather(2, kc - 1).squeeze(-1), S.gather(2, kc + 1).squeeze(-1)
        den = sm + sp - 2 * s0
        off = torch.where((den > 0) & (k > 0) & (k < D - 1), (sm - sp) / (2 * den.clamp(min=1e-6)), 0.0)
        disp = d0 + (k.float() + off.clamp(-0.5, 0.5)) * step
        # right disparity from the same aggregated volume: S_R(y, xr, d) = S(y, xr + d, d)
        dk = torch.tensor(disps, device=S.device)
        xr = torch.arange(w, device=S.device)
        idx = xr[:, None] + dk[None, :]
        inside = idx < w
        SR = S.gather(1, idx.clamp(max=w - 1)[None].expand(h, w, D))
        SR = torch.where(inside[None], SR, torch.inf)
        dR = dk[SR.argmin(-1)]
        del S, SR
        dl = dk[k]
        x = xr[None].expand(h, w)
        xm = x - dl
        ok = vl & (xm >= 0)
        xm = xm.clamp(min=0)
        ok &= vr.gather(1, xm) & ((dl - dR.gather(1, xm)).abs() <= max(int(p["lr_max_diff"]), step))
        disp = torch.where(ok, disp, torch.nan)
        return disp.cpu().numpy().astype(np.float32)

def sgm_opencv(left, right, valid_l, valid_r, d0: int, step: int, n: int, p: dict) -> np.ndarray:
    """Reference CPU path: OpenCV StereoSGBM (8 paths, block 5, its own LR check); NaN invalid."""

    bs = 5
    nd = int(min(math.ceil(n * step / 16) * 16, 512)) or 16
    sgbm = cv2.StereoSGBM_create(minDisparity=int(d0), numDisparities=nd, blockSize=bs, P1=8 * bs * bs,
                                 P2=32 * bs * bs, disp12MaxDiff=int(p["lr_max_diff"]), uniquenessRatio=10,
                                 speckleWindowSize=0, mode=cv2.STEREO_SGBM_MODE_HH)
    to8 = lambda a: np.clip(a * 255, 0, 255).astype(np.uint8)
    disp = sgbm.compute(to8(left), to8(right)).astype(np.float32) / 16.0
    bad = (disp < d0) | ~valid_l
    xm = np.clip(np.arange(left.shape[1])[None] - np.round(np.nan_to_num(disp)).astype(int), 0, left.shape[1] - 1)
    bad |= ~np.take_along_axis(valid_r, xm, 1)
    return np.where(bad, np.nan, disp).astype(np.float32)

def _speckles(disp: np.ndarray, size: int) -> np.ndarray:
    """Remove small connected disparity blobs (cv2.filterSpeckles on 1/16 px fixed point)."""

    if size <= 0:
        return disp
    d16 = np.where(np.isfinite(disp), np.round(disp * 16) + 16, 0).astype(np.int16)
    cv2.filterSpeckles(d16, 0, int(size), 2 * 16)
    return np.where(d16 > 0, disp, np.nan)

def pair_depth(ref_img, src_img, ref, src, K1, K2, drange, p: dict, device=None) -> np.ndarray | None:
    """Reference-camera z depth (0 invalid) of one pair at the reference image grid; None when degenerate."""

    h, w = ref_img.shape
    rect = rectify(K1, ref.R, ref.t, K2, src.R, src.t, (w, h), float(p["min_valid_frac"]))
    if rect is None:
        return None
    left = cv2.remap(ref_img, *rect["map1"], cv2.INTER_LINEAR, borderValue=0)
    right = cv2.remap(src_img, *rect["map2"], cv2.INTER_LINEAR, borderValue=0)
    valid_r = cv2.remap(np.ones_like(src_img), *rect["map2"], cv2.INTER_NEAREST, borderValue=0) > 0
    left, right = _to_match(left, rect), _to_match(right, rect)
    vl, vr = _to_match(rect["valid1"], rect), _to_match(valid_r, rect)
    width = left.shape[1]
    lo = max(int(math.floor(rect["fB"] / (drange[1] * 1.15))), 0)
    hi = min(int(math.ceil(rect["fB"] / max(drange[0] * 0.85, 1e-9))), width // 2)
    if hi - lo < 2:
        return None
    step = max(1, math.ceil((hi - lo + 1) / int(p["max_disparities"])))
    n = (hi - lo) // step + 1
    if p["backend"] == "opencv" or device is None:
        disp = sgm_opencv(left, right, vl, vr, lo, step, n, p)
    else:
        disp = sgm_torch(left, right, vl, vr, lo, step, n, p, device)
    disp = _speckles(disp, int(p["speckle_size"] * (max(h, w) / 800) ** 2))
    disp = _from_match(disp, rect)
    d = cv2.remap(np.nan_to_num(disp, nan=0.0), rect["ur"], rect["vr"], cv2.INTER_NEAREST, borderValue=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where((d > 0) & rect["inside"], rect["fB"] / d / rect["ray_z"], 0.0)
    z[(z < drange[0] * 0.5) | (z > drange[1] * 2)] = 0
    return z.astype(np.float32)

def smooth_normals(depth: np.ndarray, K: np.ndarray, radius: int) -> np.ndarray:
    """Camera-frame normals of an edge-aware smoothed depth (bilateral filter on log depth, 3% range sigma):
    SGM depth is too noisy for 1 px finite differences, which fail the 10 deg normal check of stereo_fusion."""

    valid = depth > 0
    if radius <= 0 or not valid.any():
        return depth_fusion.normals_from_depth(depth, K)
    logd = np.where(valid, np.log(np.where(valid, depth, 1.0)), -10.0).astype(np.float32)
    sm = cv2.bilateralFilter(logd, 2 * radius + 1, 0.03, radius)
    return depth_fusion.normals_from_depth(np.where(valid, np.exp(sm), 0).astype(np.float32), K)

def combine(depths: list[np.ndarray], rel_tol: float) -> np.ndarray:
    """One pair -> its depth; several -> per pixel the median of the depths that agree (|dz|/z <= rel_tol)
    with at least one other pair; single-pair pixels are dropped (not consistent)."""

    if len(depths) == 1:
        return depths[0]
    D = np.stack(depths)
    valid = D > 0
    agree = np.zeros_like(valid)
    for i in range(len(D)):
        for j in range(len(D)):
            if i != j:
                agree[i] |= valid[i] & valid[j] & (np.abs(D[i] - D[j]) <= rel_tol * np.maximum(D[i], D[j]))
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = np.nanmedian(np.where(agree, D, np.nan), 0)
    return np.nan_to_num(out, nan=0.0).astype(np.float32)

def candidates(views, name: str) -> list[str]:
    """Source candidates: the ranked sources first, then the other views looking the same way (optical axes
    < MAX_AXIS_DEG apart) by camera-centre distance (dense video: the top sources may all have a tiny baseline)."""

    ref = views[name]
    rest = [(np.linalg.norm(v.center - ref.center), n) for n, v in views.items()
            if n != name and n not in ref.sources and ref.R[2] @ v.R[2] > math.cos(math.radians(MAX_AXIS_DEG))]
    return list(ref.sources) + [n for _, n in sorted(rest)]

def depth_for_view(views, name: str, p: dict, device, load) -> tuple[np.ndarray, np.ndarray] | None:
    """Depth map of one reference from its best num_pairs rectifiable sources (angle in [min, max] deg)."""

    ref = views[name]
    if ref.depth_min <= 0 or ref.depth_max <= ref.depth_min:
        return None
    s = min(1.0, int(p["max_size"]) / max(ref.width, ref.height))
    w, h = round(ref.width * s), round(ref.height * s)
    K1 = depth_fusion.scale_K(ref.K, w / ref.width, h / ref.height)
    ref_img = load(name, w, h)
    depths, used = [], []
    for src_name in candidates(views, name)[:MAX_TRIES]:
        if len(depths) >= int(p["num_pairs"]):
            break
        src = views[src_name]
        if not p["min_angle_deg"] <= triangulation_angle(ref, src) <= p["max_angle_deg"]:
            continue
        sw, sh = round(src.width * s), round(src.height * s)
        K2 = depth_fusion.scale_K(src.K, sw / src.width, sh / src.height)
        src_img = load(src_name, sw, sh)
        if src_img.shape != ref_img.shape:
            src_img = cv2.resize(src_img, (w, h), interpolation=cv2.INTER_AREA)
            K2 = depth_fusion.scale_K(K2, w / sw, h / sh)
        z = pair_depth(ref_img, src_img, ref, src, K1, K2, (ref.depth_min, ref.depth_max), p, device)
        if z is not None and np.mean(z > 0) > 0.01:
            depths.append(z)
            used.append(src_name)
    if not depths:
        log.info("sgm %s: no usable pair", name)
        return None
    log.debug("sgm %s pairs %s", name, used)
    return combine(depths, float(p["depth_rel_tol"])), K1

def run(cfg: dict, run_dir: str, artifacts_dir: str) -> Path | None:

    if not depth_fusion.enabled(cfg, METHOD):
        return None
    p = depth_fusion.method_cfg(cfg, METHOD, DEFAULTS)
    device = None
    torch = None
    if p["backend"] == "torch":
        try:
            import torch
        except ImportError:
            torch = None
        if torch is None or not torch.cuda.is_available():
            log.warning("dense sgm: torch/CUDA not available, using the OpenCV StereoSGBM backend")
            p = {**p, "backend": "opencv"}
        else:
            device = f"cuda:{int(str(cfg['dense'].get('gpu_index', 0)).split(',')[0] or 0)}"
    elif p["backend"] != "opencv":
        log.warning("dense sgm: unknown backend %r, using opencv", p["backend"])
        p = {**p, "backend": "opencv"}
    row = {**depth_fusion.base_row(cfg, METHOD), "backend": p["backend"]}
    reused = depth_fusion.reuse(cfg, run_dir, artifacts_dir, METHOD, row)
    if reused is not None:
        return reused

    with depth_fusion.Timer() as t:
        und = depth_fusion.undistorted_workspace(cfg, run_dir)
    if und is None:
        return None
    row["t_undistort_s"] = round(t.s, 2)
    refs = depth_fusion.references(cfg, und)
    views = depth_fusion.load_views(und)
    ws = depth_fusion.method_workspace(run_dir, METHOD, und)
    for old in (ws / "stereo").glob("*_maps/*.bin"):
        old.unlink()

    @lru_cache(maxsize=64)
    def load(name, w, h):
        img = depth_fusion.read_image(und, name, gray=True)
        return img if img.shape == (h, w) else cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    if device:
        torch.zeros(1, device=device)  # init the CUDA context before the peak reset
        torch.cuda.reset_peak_memory_stats(device)
    log.info("sgm: %d references, %s", len(refs), {k: p[k] for k in DEFAULTS})
    with depth_fusion.Timer() as t:
        for i, name in enumerate(refs):
            if name not in views:
                continue
            out = depth_for_view(views, name, p, device, load)
            if out is None:
                continue
            depth, K = out
            depth_fusion.save_depth(ws, name, depth, K, smooth_normals(depth, K, int(p["normal_radius"])))
            if i % 10 == 0:
                log.info("sgm %d/%d %s valid %.0f%%", i + 1, len(refs), name, 100 * np.mean(depth > 0))
    row["t_depth_s"] = round(t.s, 2)
    if device:
        row["vram_peak_gb"] = round(torch.cuda.max_memory_allocated(device) / 2 ** 30, 2)
        torch.cuda.empty_cache()
    return depth_fusion.fuse_and_export(cfg, run_dir, artifacts_dir, METHOD, ws, refs, row)

# <<< END IMPLEMENTATION <<<

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    h, w = 120, 200
    tex = cv2.GaussianBlur(rng.random((h, w + 64)).astype(np.float32), (0, 0), 1.0)
    tex = (tex - tex.min()) / (tex.max() - tex.min())
    # synthetic rectified pair: background disparity 10, square disparity 22 (right(x - d) = left(x))
    true = np.full((h, w), 10, np.int64)
    true[30:90, 60:140] = 22
    left = tex[:, 32:32 + w].copy()
    right = np.zeros_like(left)
    for y in range(h):
        for x in range(w):
            if x - true[y, x] >= 0:
                right[y, x - true[y, x]] = left[y, x]
    hole = right == 0  # disoccluded: fill with fresh texture
    right[hole] = tex[:, :w][hole]
    ones = np.ones((h, w), bool)
    p = {**DEFAULTS, "speckle_size": 0}
    for backend in ("torch", "opencv"):
        if backend == "torch":
            disp = sgm_torch(left, right, ones, ones, 4, 1, 28, p, dev)
        else:
            disp = sgm_opencv(left, right, ones, ones, 4, 1, 28, p)
        roi = np.s_[5:-5, 40:-5]
        ok = np.isfinite(disp[roi]) & (np.abs(disp[roi] - true[roi]) <= 1)
        frac, valid = ok.mean(), np.isfinite(disp[roi]).mean()
        print(f"{backend}: within 1 px {frac:.3f}, valid {valid:.3f}")
        assert frac > 0.85, (backend, frac)

    # end-to-end geometry: a tilted plane seen by two posed cameras -> reference z after rectify + map back
    from src.dense.depth_fusion import View
    W, H = 320, 240
    K = np.array([[300.0, 0, 160], [0, 300.0, 120], [0, 0, 1]])
    big = cv2.GaussianBlur(rng.random((H, W)).astype(np.float32), (0, 0), 1.2)
    big = (big - big.min()) / (big.max() - big.min())
    a = np.radians(4)
    R2 = cv2.Rodrigues(np.array([0.0, a, 0.02]))[0]
    t2 = np.array([-0.35, 0.03, 0.02])
    n_pl, d_pl = np.array([0.15, 0.0, 1.0]) / np.linalg.norm([0.15, 0.0, 1.0]), 3.0  # n.X = d (ref frame)
    ref = View("r", K, np.eye(3), np.zeros(3), W, H, 2.0, 5.0)
    src = View("s", K, R2, t2, W, H, 2.0, 5.0)
    print(f"triangulation angle {triangulation_angle(ref, src):.1f} deg")
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z_true = d_pl / (np.stack([u, v, np.ones_like(u)], -1) @ np.linalg.inv(K).T @ n_pl)
    # source to the right (standard), to the left (flipped) and below (vertical epipolar lines)
    for t2 in (np.array([-0.35, 0.03, 0.02]), np.array([0.35, 0.03, 0.02]), np.array([0.02, -0.3, 0.02])):
        Hm = K @ (R2 + np.outer(t2, n_pl) / d_pl) @ np.linalg.inv(K)
        src_img = cv2.warpPerspective(big, Hm, (W, H), borderValue=0)
        src = View("s", K, R2, t2, W, H, 2.0, 5.0)
        for backend in ("torch", "opencv"):
            z = pair_depth(big, src_img, ref, src, K, K, (2.0, 5.0), {**DEFAULTS, "backend": backend},
                           dev if backend == "torch" else None)
            m = z > 0
            err = np.median(np.abs(z[m] - z_true[m]) / z_true[m])
            print(f"t={t2} {backend}: plane valid {m.mean():.3f}, median rel z error {err:.4f}")
            assert m.mean() > 0.4 and err < 0.01, (backend, m.mean(), err)
    # forward motion (epipole inside the image) is rejected
    assert rectify(K, np.eye(3), np.zeros(3), K, np.eye(3), np.array([0.0, 0.0, -0.5]), (W, H)) is None
    d1 = np.full((4, 4), 2.0, np.float32)
    assert np.allclose(combine([d1, d1 * 1.01], 0.02), 2.01, atol=0.011)
    assert combine([d1, d1 * 1.2], 0.02).max() == 0
    print("sgm self-test OK")

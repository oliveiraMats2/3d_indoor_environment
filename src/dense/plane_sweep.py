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
# Stage: dense (method plane_sweep) - classical plane-sweep multi-view stereo in torch on the GPU.
# Goal: per reference view, sweep D fronto-parallel planes (uniform in inverse depth over the sparse depth
#      range), warp the K best source views with the plane-induced homography, score ZNCC over a (2w+1)^2
#      window, keep the best-k sources per pixel, winner-take-all + parabola sub-plane refinement, then filter
#      and fuse with COLMAP stereo_fusion (its geometric check enforces multi-view consistency).
# In:  <run_dir>/dense/undistorted (shared PINHOLE workspace, depth_fusion), dense.ref_step references.
# Out: <run_dir>/dense/plane_sweep/{ws/stereo/{depth_maps,normal_maps,fusion.cfg}, fused.ply, points.npy,
#      colors.npy, colmap_dense.log}; row method=plane_sweep in <artifacts_dir>/dense_metrics.csv.
# YAML: dense.{enabled, rerun, methods, gpu_index, ref_step, fusion.*}, dense.plane_sweep.{max_size,
#      num_depths, num_sources, best_k, window_radius, min_zncc, min_std, peak_ratio, chunk_mb}.
# Rules: no CUDA -> WARNING and return None; depth maps at the sweep resolution (fusion rescales K);
#      depth = camera z in model units, 0 = invalid.

# >>> BEGIN IMPLEMENTATION >>>
"""Plane-sweep stereo: inverse-depth fronto-parallel sweep, ZNCC best-k cost, WTA + parabola, COLMAP fusion."""
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.dense import depth_fusion

log = logging.getLogger(__name__)

METHOD = "plane_sweep"
DEFAULTS = {"max_size": 800, "num_depths": 192, "num_sources": 6, "best_k": 3, "window_radius": 4,
            "min_zncc": 0.4, "min_std": 0.01, "peak_ratio": 0.95, "chunk_mb": 2048}

def _box(x, r: int):
    """Mean over a (2r+1)^2 window, same size (zero padding is masked by the validity weight)."""

    import torch.nn.functional as F

    return F.avg_pool2d(x, 2 * r + 1, stride=1, padding=r, count_include_pad=False)

def sweep(ref, K_ref, sources, depth_min: float, depth_max: float, p: dict, device: str = "cuda"):
    """Plane-sweep depth of one reference view.

    ref: (h, w) gray float32; sources: [(gray (h_s, w_s), K_s, R_rel, t_rel)] with X_s = R_rel X_ref + t_rel.
    Returns (depth (h, w), score (h, w)) as numpy; depth 0 where filtered."""

    import torch
    import torch.nn.functional as F

    h, w = ref.shape
    r, D = int(p["window_radius"]), int(p["num_depths"])
    k = max(1, min(int(p["best_k"]), len(sources)))
    I = torch.from_numpy(ref).to(device)[None, None]
    mu_i = _box(I, r)
    var_i = (_box(I * I, r) - mu_i ** 2).clamp_min(0)

    # pixel centres in COLMAP convention (corner = 0): p = (u + 0.5, v + 0.5, 1)
    v, u = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32),
                          torch.arange(w, device=device, dtype=torch.float32), indexing="ij")
    pix = torch.stack([u + 0.5, v + 0.5, torch.ones_like(u)], 0).reshape(3, -1)
    Kinv = torch.from_numpy(np.linalg.inv(K_ref)).float().to(device)
    S = len(sources)
    A = torch.empty(S, 3, h * w, device=device)
    b = torch.empty(S, 3, 1, device=device)
    imgs, sizes = [], []
    for s, (img, K_s, R_rel, t_rel) in enumerate(sources):
        Ks = torch.from_numpy(K_s).float().to(device)
        A[s] = Ks @ torch.from_numpy(R_rel).float().to(device) @ Kinv @ pix
        b[s] = Ks @ torch.from_numpy(np.asarray(t_rel, np.float64).reshape(3, 1)).float().to(device)
        imgs.append(torch.from_numpy(img).to(device)[None, None])
        sizes.append(img.shape)

    inv = torch.linspace(1.0 / depth_min, 1.0 / depth_max, D, device=device)
    score = torch.empty(D, h, w, device=device)
    # ~8 live (S, chunk, h, w) float32 tensors per chunk
    chunk = int(max(1, min(D, p["chunk_mb"] * 2 ** 20 // (8 * 4 * S * h * w))))
    for d0 in range(0, D, chunk):
        z = 1.0 / inv[d0:d0 + chunk]
        c = len(z)
        zncc = torch.empty(S, c, h, w, device=device)
        for s in range(S):
            hs, ws_ = sizes[s]
            q = A[s][None] * z[:, None, None] + b[s][None]           # (c, 3, hw): p_s ~ z A p + b
            zq = q[:, 2]
            ok = zq > 1e-6
            x = q[:, 0] / zq.clamp_min(1e-6)
            y = q[:, 1] / zq.clamp_min(1e-6)
            ok &= (x >= 0) & (x <= ws_) & (y >= 0) & (y <= hs)
            grid = torch.stack([2 * x / ws_ - 1, 2 * y / hs - 1], -1).reshape(c, h, w, 2)
            J = F.grid_sample(imgs[s].expand(c, -1, -1, -1), grid, mode="bilinear", padding_mode="zeros",
                              align_corners=False)                    # (c, 1, h, w)
            m = ok.reshape(c, 1, h, w).float()
            J = J * m
            mu_j = _box(J, r)
            cov = _box(I * J, r) - mu_i * mu_j
            var_j = (_box(J * J, r) - mu_j ** 2).clamp_min(0)
            nz = cov / torch.sqrt(var_i * var_j + 1e-8)
            # window must lie fully inside the source image and the source patch must have some texture
            good = (_box(m, r) > 0.999) & (var_j > 1e-6)
            zncc[s] = torch.where(good, nz, torch.full_like(nz, -1.0))[:, 0]
            del q, zq, ok, x, y, grid, J, m, mu_j, cov, var_j, nz, good
        # mean of the best-k valid sources (sources that do not see the window do not count)
        top = zncc.topk(k, dim=0).values
        n_ok = (top > -1).sum(0)
        score[d0:d0 + c] = torch.where(n_ok > 0, top.where(top > -1, 0).sum(0) / n_ok.clamp_min(1),
                                       torch.full_like(top[0], -1.0))
        del zncc

    best, idx = score.max(0)
    # parabola through the neighbouring planes (uniform in inverse depth)
    i0 = idx.clamp(1, D - 2)
    s_m = score.gather(0, (i0 - 1)[None])[0]
    s_0 = score.gather(0, i0[None])[0]
    s_p = score.gather(0, (i0 + 1)[None])[0]
    den = s_m - 2 * s_0 + s_p
    off = torch.where(den < -1e-6, 0.5 * (s_m - s_p) / den.clamp(max=-1e-6), torch.zeros_like(den))
    off = torch.where((idx > 0) & (idx < D - 1), off.clamp(-0.5, 0.5), torch.zeros_like(off))
    step = (inv[1] - inv[0]) if D > 1 else inv[0] * 0
    inv_ref = inv[idx] + off * step
    depth = 1.0 / inv_ref.clamp_min(1e-12)

    # peak ratio: best cost vs the best cost outside +-2 planes
    near = (torch.arange(D, device=device)[:, None, None] - idx[None]).abs() <= 2
    second = score.masked_fill(near, -1.0).max(0).values
    ratio = (1 - best) / (1 - second).clamp_min(1e-6)
    valid = (best >= p["min_zncc"]) & (torch.sqrt(var_i[0, 0]) >= p["min_std"]) & (ratio <= p["peak_ratio"])
    depth = torch.where(valid, depth, torch.zeros_like(depth))
    return depth.cpu().numpy().astype(np.float32), best.cpu().numpy().astype(np.float32)

def _scaled_size(view, max_size: int) -> tuple[int, int, float, float]:

    s = min(1.0, float(max_size) / max(view.width, view.height))
    w, h = max(1, round(view.width * s)), max(1, round(view.height * s))
    return w, h, w / view.width, h / view.height

def depth_for_view(ws, views: dict, name: str, p: dict, device: str, load):
    """Depth map + intrinsics at the sweep resolution of reference `name` (None when it cannot be swept)."""

    v = views[name]
    srcs = [s for s in v.sources if s in views][:int(p["num_sources"])]
    if v.depth_max <= v.depth_min or v.depth_min <= 0 or not srcs:
        log.info("plane_sweep: skip %s (no depth range or sources)", name)
        return None
    w, h, sx, sy = _scaled_size(v, p["max_size"])
    K = depth_fusion.scale_K(v.K, sx, sy)
    sources = []
    for s in srcs:
        o = views[s]
        ow, oh, osx, osy = _scaled_size(o, p["max_size"])
        R_rel = o.R @ v.R.T
        sources.append((load(s, ow, oh), depth_fusion.scale_K(o.K, osx, osy), R_rel, o.t - R_rel @ v.t))
    depth, _ = sweep(load(name, w, h), K, sources, v.depth_min, v.depth_max, p, device)
    return depth, K

def run(cfg: dict, run_dir: str, artifacts_dir: str) -> Path | None:

    if not depth_fusion.enabled(cfg, METHOD):
        return None
    p = depth_fusion.method_cfg(cfg, METHOD, DEFAULTS)
    row = {**depth_fusion.base_row(cfg, METHOD), "backend": "torch"}
    reused = depth_fusion.reuse(cfg, run_dir, artifacts_dir, METHOD, row)
    if reused is not None:
        return reused
    try:
        import torch
    except ImportError:
        log.warning("dense plane_sweep skipped: torch not installed")
        return None
    if not torch.cuda.is_available():
        log.warning("dense plane_sweep skipped: CUDA not available")
        return None
    device = f"cuda:{int(str(cfg['dense'].get('gpu_index', 0)).split(',')[0] or 0)}"

    with depth_fusion.Timer() as t:
        und = depth_fusion.undistorted_workspace(cfg, run_dir)
    if und is None:
        return None
    row["t_undistort_s"] = round(t.s, 2)
    refs = depth_fusion.references(cfg, und)
    views = depth_fusion.load_views(und, max(int(p["num_sources"]), 1))
    ws = depth_fusion.method_workspace(run_dir, METHOD, und)
    for old in (ws / "stereo").glob("*_maps/*.bin"):
        old.unlink()

    @lru_cache(maxsize=64)
    def load(name, w, h):
        import cv2

        img = depth_fusion.read_image(und, name, gray=True)
        return img if img.shape == (h, w) else cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    torch.cuda.set_device(device)  # initialises the context (peak stats need it)
    torch.cuda.reset_peak_memory_stats(device)
    log.info("plane_sweep: %d references, %s", len(refs), {k: p[k] for k in DEFAULTS})
    with depth_fusion.Timer() as t:
        for i, name in enumerate(refs):
            if name not in views:
                continue
            out = depth_for_view(und, views, name, p, device, load)
            if out is None:
                continue
            depth, K = out
            depth_fusion.save_depth(ws, name, depth, K)
            if i % 10 == 0:
                log.info("plane_sweep %d/%d %s valid %.0f%%", i + 1, len(refs), name, 100 * np.mean(depth > 0))
    row["t_depth_s"] = round(t.s, 2)
    row["vram_peak_gb"] = round(torch.cuda.max_memory_allocated(device) / 2 ** 30, 2)
    torch.cuda.empty_cache()
    return depth_fusion.fuse_and_export(cfg, run_dir, artifacts_dir, METHOD, ws, refs, row)

# <<< END IMPLEMENTATION <<<

if __name__ == "__main__":
    import tempfile

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    def texture(X, Y):
        """Smooth random texture on the plane (sum of random sines)."""

        rng = np.random.default_rng(1)
        out = np.zeros_like(X)
        for _ in range(24):
            f, ph, ang = rng.uniform(3, 25), rng.uniform(0, 2 * np.pi), rng.uniform(0, np.pi)
            out += np.sin(f * (X * np.cos(ang) + Y * np.sin(ang)) + ph)
        return (0.5 + 0.08 * out).clip(0, 1).astype(np.float32)

    def render(K, R, t, h, w, plane_z):
        """Image of the textured slanted plane z = plane_z(x) (world = ref camera) seen by camera (R, t)."""

        v, u = np.meshgrid(np.arange(h) + 0.5, np.arange(w) + 0.5, indexing="ij")
        rays = np.stack([u, v, np.ones_like(u)], -1) @ np.linalg.inv(K).T @ R  # world directions
        C = -R.T @ t
        a, z0 = plane_z  # plane: z = z0 + a x  ->  (C + l d)_z = z0 + a (C + l d)_x
        lam = (z0 + a * C[0] - C[2]) / (rays[..., 2] - a * rays[..., 0])
        P = C + lam[..., None] * rays
        return texture(P[..., 0], P[..., 1]), P

    dev = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    h, w = 120, 160
    K = np.array([[150.0, 0, 80], [0, 150.0, 60], [0, 0, 1]])
    plane = (0.15, 3.0)
    ref, P_ref = render(K, np.eye(3), np.zeros(3), h, w, plane)
    srcs = []
    for tx, ty in ((-0.4, 0.0), (0.4, 0.05), (0.0, -0.35)):
        ang = np.radians(3.0 * np.sign(tx))
        R = np.array([[np.cos(ang), 0, np.sin(ang)], [0, 1, 0], [-np.sin(ang), 0, np.cos(ang)]])
        t = np.array([tx, ty, 0.0])
        srcs.append((render(K, R, t, h, w, plane)[0], K, R, t))
    p = {**DEFAULTS, "num_depths": 96, "window_radius": 3}
    gt = P_ref[..., 2].astype(np.float32)
    for n_src in (1, 3):  # two views, then best-k over three sources
        depth, score = sweep(ref, K, srcs[:n_src], 2.0, 4.5, p, dev)
        ok = depth > 0
        rel = np.abs(depth[ok] - gt[ok]) / gt[ok]
        print(f"{n_src} source(s): valid {ok.mean():.2f} median rel err {np.median(rel):.4f} "
              f"p90 {np.percentile(rel, 90):.4f}")
        assert ok.mean() > 0.5, ok.mean()
        assert np.median(rel) < 0.005 and np.percentile(rel, 90) < 0.02, (np.median(rel), np.percentile(rel, 90))

    # disabled / reuse paths without COLMAP or CUDA
    with tempfile.TemporaryDirectory() as tmp:
        cfg = {"dense": {"enabled": True, "methods": ["patch_match"], "rerun": False, "max_image_size": 800}}
        assert run(cfg, tmp, tmp) is None
        from src import stage_io
        out = Path(tmp) / "dense" / METHOD
        stage_io.write_ply(out / "fused.ply", np.random.default_rng(0).random((20, 3)), np.full((20, 3), 9))
        cfg["dense"]["methods"] = [METHOD]
        assert run(cfg, tmp, tmp) == out / "fused.ply" and np.load(out / "points.npy").shape == (20, 3)
    print("OK")

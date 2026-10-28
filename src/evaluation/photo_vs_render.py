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
# Stage: photo_vs_render (4.B.3), after mesh/export (and gaussian_splatting when it ran), before the evaluation
#      block. Optional: photo_vs_render.enabled.
# Goal: the real photo next to a render of the reconstructed model from the SAME camera pose, for n_views
#      evenly spread registered views; plus the Gaussian Splatting render of that view when GS was evaluated.
# In:  <run_dir>/sfm/{cam_R,cam_t,K,dist,image_names}.npy (world->camera x_c = R x_w + t, model units, OPENCV
#      dist k1,k2,p1,p2); photos from cfg["extraction"]["output_dir"] (else exports/colmap/images); model, first
#      available from photo_vs_render.source on: mesh (mesh/room_fill/room_scene.ply, else mesh/{vertices,faces,
#      colors}.npy) -> dense (clean/<dense.methods[i]>/{points,colors}.npy) -> sparse (clean/sparse); working
#      units (cm when scale/scale.npy is known: t is multiplied by s). GS: gaussian_splatting/model/test/
#      ours_<N>/{renders,gt}/<k:05d>.png, test k = sorted(gaussian_splatting/data/images)[8 k] (Inria llffhold).
# Out: <run_dir>/figures/photo_vs_render/<frame>.png (photo | render [| GS render]), <run_dir>/figures/
#      photo_vs_render.png (grid of every view); <artifacts_dir>/photo_vs_render.csv (one row per frame: model,
#      psnr, ssim, coverage, gs_psnr, gs_ssim), mean values logged as `metric name=value`.
# YAML: photo_vs_render: {enabled: true, n_views: 6, source: mesh|dense|sparse, max_size: 960, point_px: 4,
#      gs: true}.
# Rules: photo undistorted with K, dist (render is pinhole); render at max_size (longest side), unlit vertex
#      colours. Mesh: Open3D RaycastingScene (Embree) + barycentric colour of the hit triangle. Points: z-buffer
#      splat of point_px x point_px pixels in torch (CUDA when available). PSNR and SSIM (Gaussian 11x11,
#      sigma 1.5, mean over RGB) only over the covered pixels; coverage = covered / all pixels. When GS test
#      views exist, the views are picked among them (held-out views, honest comparison) and the GS metrics are
#      computed against the GS gt image. Missing inputs -> WARNING and return None.

# >>> BEGIN IMPLEMENTATION >>>
"""Real photo vs render of the reconstructed model (mesh or cloud) and of Gaussian Splatting from the same pose."""
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from src import stage_io

log = logging.getLogger(__name__)

DEFAULTS = {"enabled": True, "n_views": 6, "source": "mesh", "max_size": 960, "point_px": 4, "gs": True}
SOURCES = ("mesh", "dense", "sparse")
LLFFHOLD = 8  # Inria --eval split (src/gs/gaussian_splatting.py)
IMAGE_EXT = {".png", ".jpg", ".jpeg"}
BG = 255  # background of uncovered render pixels (white)

def pvr_cfg(cfg: dict) -> dict:

    return {**DEFAULTS, **(cfg.get("photo_vs_render") or {})}

# ---------------------------------------------------------------- model loading
def load_mesh(run_dir: Path):
    """(vertices, faces, colors uint8 | None, label) of the final mesh, or None."""

    scene = run_dir / "mesh" / "room_fill" / "room_scene.ply"
    if scene.exists():
        import open3d as o3d
        m = o3d.io.read_triangle_mesh(str(scene))
        if len(m.triangles):
            col = (np.asarray(m.vertex_colors) * 255).round().astype(np.uint8) if m.has_vertex_colors() else None
            return np.asarray(m.vertices), np.asarray(m.triangles), col, "mesh/room_fill/room_scene.ply"

    v_path, f_path = run_dir / "mesh" / "vertices.npy", run_dir / "mesh" / "faces.npy"
    if v_path.exists() and f_path.exists():
        c_path = run_dir / "mesh" / "colors.npy"
        col = np.load(c_path).astype(np.uint8) if c_path.exists() else None
        return np.load(v_path), np.load(f_path), col, "mesh/vertices.npy"

    return None

def load_points(run_dir: Path, source: str):
    """(points, colors uint8 | None, label) of clean/<source>, or None."""

    p_path = run_dir / "clean" / source / "points.npy"
    if not p_path.exists():
        return None

    points = np.load(p_path).astype(np.float64).reshape(-1, 3)
    c_path = run_dir / "clean" / source / "colors.npy"
    colors = np.load(c_path).astype(np.uint8).reshape(-1, 3) if c_path.exists() else None
    if colors is not None and len(colors) != len(points):
        colors = None

    return points, colors, f"clean/{source}"

def dense_sources(cfg: dict, run_dir: Path) -> list[str]:
    """dense.methods order first, then any other clean/<method> (not sparse)."""

    methods = list((cfg.get("dense") or {}).get("methods") or [])
    clean_dir = run_dir / "clean"
    others = sorted(p.name for p in clean_dir.iterdir() if p.is_dir()) if clean_dir.is_dir() else []

    return [m for m in methods + others if m != "sparse"]

def pick_model(cfg: dict, run_dir: Path, source: str) -> dict | None:
    """First available model from `source` on (mesh -> dense -> sparse)."""

    start = SOURCES.index(source) if source in SOURCES else 0
    if source not in SOURCES:
        log.warning("photo_vs_render.source=%s unknown: using mesh", source)
    for kind in SOURCES[start:]:
        if kind == "mesh":
            m = load_mesh(run_dir)
            if m is not None:
                return {"kind": "mesh", "V": m[0], "F": m[1], "C": m[2], "label": m[3]}
            continue
        for src in (dense_sources(cfg, run_dir) if kind == "dense" else ["sparse"]):
            p = load_points(run_dir, src)
            if p is not None:
                return {"kind": "points", "P": p[0], "C": p[1], "label": p[2]}
        log.info("photo_vs_render: no %s model", kind)

    return None

# ---------------------------------------------------------------- rendering
class MeshRenderer:
    """Ray casting (Open3D RaycastingScene) with barycentric vertex colours."""

    def __init__(self, V: np.ndarray, F: np.ndarray, C: np.ndarray | None):
        import open3d as o3d
        self.o3d = o3d
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32)))
        self.V, self.F = V.astype(np.float32), F.astype(np.int64)
        self.C = C.astype(np.float32) if C is not None else None

    def render(self, K: np.ndarray, R: np.ndarray, t: np.ndarray, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
        o3d = self.o3d
        E = np.eye(4)
        E[:3, :3], E[:3, 3] = R, t
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            o3d.core.Tensor(K, dtype=o3d.core.float64), o3d.core.Tensor(E, dtype=o3d.core.float64), w, h)
        ans = self.scene.cast_rays(rays)
        hit = np.isfinite(ans["t_hit"].numpy())
        tri = ans["primitive_ids"].numpy()[hit].astype(np.int64)
        uv = ans["primitive_uvs"].numpy()[hit]
        bary = np.stack([1.0 - uv[:, 0] - uv[:, 1], uv[:, 0], uv[:, 1]], 1)[:, :, None]
        img = np.full((h, w, 3), BG, np.uint8)
        if self.C is not None:
            col = (self.C[self.F[tri]] * bary).sum(1)
        else:  # no colours: Lambert shading with the light at the camera
            v = self.V[self.F[tri]]
            n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
            n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
            dirs = rays.numpy()[hit][:, 3:]
            shade = np.abs((n * dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)).sum(1))
            col = np.repeat(60 + 180 * shade[:, None], 3, 1)
        img[hit] = np.clip(col, 0, 255).astype(np.uint8)

        return img, hit

class PointRenderer:
    """Z-buffer splat of point_px x point_px pixel squares in torch (CUDA when available)."""

    def __init__(self, P: np.ndarray, C: np.ndarray | None, point_px: int):
        import torch
        self.torch = torch
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.P = torch.as_tensor(P, dtype=torch.float32, device=self.dev)
        col = C if C is not None else np.full((len(P), 3), 128, np.uint8)
        self.C = torch.as_tensor(col, dtype=torch.uint8, device=self.dev)
        self.px = max(1, int(point_px))

    def render(self, K: np.ndarray, R: np.ndarray, t: np.ndarray, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        Rt = torch.as_tensor(R, dtype=torch.float32, device=self.dev)
        tt = torch.as_tensor(t, dtype=torch.float32, device=self.dev)
        Kt = torch.as_tensor(K, dtype=torch.float32, device=self.dev)
        Xc = self.P @ Rt.T + tt
        front = Xc[:, 2] > 1e-6
        Xc, col = Xc[front], self.C[front]
        uv = (Xc @ Kt.T)[:, :2] / Xc[:, 2:3]
        z = Xc[:, 2]
        off = torch.arange(self.px, device=self.dev) - (self.px - 1) // 2
        oy, ox = torch.meshgrid(off, off, indexing="ij")
        u = torch.floor(uv[:, 0]).long()[:, None] + ox.reshape(1, -1)
        v = torch.floor(uv[:, 1]).long()[:, None] + oy.reshape(1, -1)
        idx = torch.arange(len(z), device=self.dev)[:, None].expand_as(u)
        ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        pix, idx = (v * w + u)[ok], idx[ok]
        zbuf = torch.full((h * w,), float("inf"), device=self.dev)
        zbuf.scatter_reduce_(0, pix, z[idx], reduce="amin")
        win = z[idx] <= zbuf[pix]  # ties: any of them (same depth)
        img = torch.full((h * w, 3), BG, dtype=torch.uint8, device=self.dev)
        img[pix[win]] = col[idx[win]]
        hit = torch.isfinite(zbuf)

        return img.reshape(h, w, 3).cpu().numpy(), hit.reshape(h, w).cpu().numpy()

# ---------------------------------------------------------------- metrics
def psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return float("nan")
    mse = float(((a.astype(np.float64) - b.astype(np.float64))[mask] ** 2).mean())

    return float("inf") if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)

def ssim(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """Wang et al. SSIM (Gaussian 11x11, sigma 1.5), mean over RGB and over the masked pixels."""

    if not mask.any():
        return float("nan")
    a, b = a.astype(np.float64), b.astype(np.float64)
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a, mu_b = blur(a), blur(b)
    s_aa, s_bb, s_ab = blur(a * a) - mu_a ** 2, blur(b * b) - mu_b ** 2, blur(a * b) - mu_a * mu_b
    m = ((2 * mu_a * mu_b + c1) * (2 * s_ab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (s_aa + s_bb + c2))

    return float(m.mean(axis=2)[mask].mean())

# ---------------------------------------------------------------- views and Gaussian Splatting
def gs_test_views(run_dir: Path) -> tuple[dict, Path | None]:
    """{frame name: ours_<N>/ index} of the GS test views and the ours_<N> dir (highest N), or ({}, None)."""

    test = run_dir / "gaussian_splatting" / "model" / "test"
    data = run_dir / "gaussian_splatting" / "data" / "images"
    its = sorted((int(p.name.split("_")[-1]), p) for p in test.glob("ours_*") if p.name.split("_")[-1].isdigit()) \
        if test.is_dir() else []
    if not its or not data.is_dir():
        return {}, None
    ours = its[-1][1]
    names = sorted(p.name for p in data.iterdir() if p.suffix.lower() in IMAGE_EXT)
    test_names = [n for i, n in enumerate(names) if i % LLFFHOLD == 0]
    n_renders = len(list((ours / "renders").glob("*.png")))
    if n_renders != len(test_names):
        log.warning("photo_vs_render: %d GS renders but %d test names (llffhold %d): GS column skipped",
                    n_renders, len(test_names), LLFFHOLD)
        return {}, None

    return {n: k for k, n in enumerate(test_names)}, ours

def pick_views(names: list[str], n: int, prefer: dict) -> list[int]:
    """n evenly spread indices of names; among the `prefer` names when enough of them are registered."""

    pool = [i for i, nm in enumerate(names) if nm in prefer]
    if len(pool) < n:
        pool = list(range(len(names)))
    sel = np.unique(np.linspace(0, len(pool) - 1, min(n, len(pool))).round().astype(int))

    return [pool[i] for i in sel]

def read_photo(name: str, dirs: list[Path]) -> np.ndarray | None:
    for d in dirs:
        p = d / name
        if p.exists():
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is not None:
                return img
    return None

def label(img: np.ndarray, text: str) -> np.ndarray:
    """Image with a white title bar on top."""

    bar = np.full((28, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    return np.vstack([bar, img])

def fmt(x: float, nd: int = 2):
    return round(float(x), nd) if np.isfinite(x) else ""

def run(cfg: dict, run_dir: str, artifacts_dir: str) -> Path | None:

    pcfg = pvr_cfg(cfg)
    if not pcfg["enabled"]:
        log.info("photo_vs_render disabled")
        return None

    t0 = time.perf_counter()
    run_dir, artifacts_dir = Path(run_dir), Path(artifacts_dir)
    sfm = run_dir / "sfm"
    need = ["cam_R", "cam_t", "K", "image_names"]
    if not all((sfm / f"{n}.npy").exists() for n in need):
        log.warning("photo_vs_render: missing sfm/{%s}.npy, skipping", ",".join(need))
        return None
    cam_R, cam_t, K = np.load(sfm / "cam_R.npy"), np.load(sfm / "cam_t.npy"), np.load(sfm / "K.npy")
    names = [str(n) for n in np.load(sfm / "image_names.npy")]
    dist = np.load(sfm / "dist.npy") if (sfm / "dist.npy").exists() else np.zeros(4)
    s, unit = stage_io.scale_info(run_dir)
    t_scale = s if unit == "cm" else 1.0  # models are in working units (cm when the scale is known)

    model = pick_model(cfg, run_dir, pcfg["source"])
    if model is None:
        log.warning("photo_vs_render: no mesh, dense or sparse model, skipping")
        return None
    t_load = time.perf_counter()
    if model["kind"] == "mesh":
        renderer, device = MeshRenderer(model["V"], model["F"], model["C"]), "cpu (Embree ray casting)"
    else:
        renderer = PointRenderer(model["P"], model["C"], pcfg["point_px"])
        device = str(renderer.dev)
    log.info("photo_vs_render: model %s (%s) on %s, setup %.1f s", model["label"], model["kind"], device,
             time.perf_counter() - t_load)

    gs_map, ours = gs_test_views(run_dir) if pcfg["gs"] else ({}, None)
    views = pick_views(names, int(pcfg["n_views"]), gs_map)
    photo_dirs = [Path(d) for d in [(cfg.get("extraction") or {}).get("output_dir")] if d] + \
                 [run_dir / "exports" / "colmap" / "images"]

    fig_dir = run_dir / "figures" / "photo_vs_render"
    fig_dir.mkdir(parents=True, exist_ok=True)
    csv_path = artifacts_dir / "photo_vs_render.csv"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    csv_path.unlink(missing_ok=True)  # one run = one set of views
    rows, panels, t_render = [], [], 0.0
    for i in views:
        name = names[i]
        photo = read_photo(name, photo_dirs)
        if photo is None:
            log.warning("photo_vs_render: photo %s not found in %s", name, [str(d) for d in photo_dirs])
            continue
        H, W = photo.shape[:2]
        if np.any(dist):
            photo = cv2.undistort(photo, K, np.asarray(dist, np.float64))
        f = min(1.0, pcfg["max_size"] / max(W, H))
        w, h = int(round(W * f)), int(round(H * f))
        Kr = K.copy()
        Kr[0] *= w / W
        Kr[1] *= h / H
        photo = cv2.resize(photo, (w, h), interpolation=cv2.INTER_AREA) if (w, h) != (W, H) else photo

        tr = time.perf_counter()
        rgb, hit = renderer.render(Kr, cam_R[i], cam_t[i] * t_scale, w, h)
        t_render += time.perf_counter() - tr
        render = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        row = {"frame": name, "model": model["label"], "width": w, "height": h,
               "psnr": fmt(psnr(photo, render, hit)), "ssim": fmt(ssim(photo, render, hit), 4),
               "coverage": round(float(hit.mean()), 4), "gs_psnr": "", "gs_ssim": ""}
        cols = [label(photo, f"photo {name}"),
                label(render, f"{model['kind']} render  PSNR {row['psnr']}  cov {row['coverage']:.2f}")]

        if ours is not None and name in gs_map:
            k = f"{gs_map[name]:05d}.png"
            gs_r, gs_gt = cv2.imread(str(ours / "renders" / k)), cv2.imread(str(ours / "gt" / k))
            if gs_r is not None and gs_gt is not None:
                full = np.ones(gs_r.shape[:2], bool)
                row["gs_psnr"], row["gs_ssim"] = fmt(psnr(gs_gt, gs_r, full)), fmt(ssim(gs_gt, gs_r, full), 4)
                cols.append(label(cv2.resize(gs_r, (w, h), interpolation=cv2.INTER_AREA),
                                  f"Gaussian Splatting (test view)  PSNR {row['gs_psnr']}"))
        elif ours is not None:
            cols.append(label(np.full((h, w, 3), 235, np.uint8), "Gaussian Splatting: training view (not shown)"))

        panel = np.hstack(cols)
        cv2.imwrite(str(fig_dir / f"{Path(name).stem}.png"), panel)
        panels.append(panel)
        stage_io.upsert_csv(csv_path, row, key="frame")
        rows.append(row)

    if not rows:
        log.warning("photo_vs_render: no view rendered")
        return None

    width = max(p.shape[1] for p in panels)
    grid = np.vstack([np.pad(p, ((0, 4), (0, width - p.shape[1]), (0, 0)), constant_values=255) for p in panels])
    scale_g = min(1.0, 2400 / grid.shape[1])
    if scale_g < 1.0:
        grid = cv2.resize(grid, None, fx=scale_g, fy=scale_g, interpolation=cv2.INTER_AREA)
    out = run_dir / "figures" / "photo_vs_render.png"
    cv2.imwrite(str(out), grid)

    def mean(key):
        vals = [r[key] for r in rows if r[key] != ""]
        return round(float(np.mean(vals)), 4) if vals else ""

    stage_io.log_metrics(log, {"pvr_model": model["label"], "pvr_n_views": len(rows), "pvr_psnr_mean": mean("psnr"),
                               "pvr_ssim_mean": mean("ssim"), "pvr_coverage_mean": mean("coverage"),
                               "pvr_gs_psnr_mean": mean("gs_psnr"), "pvr_gs_ssim_mean": mean("gs_ssim"),
                               "pvr_render_s": round(t_render, 2), "pvr_t_s": round(time.perf_counter() - t0, 1)})

    return out
# <<< END IMPLEMENTATION <<<

if __name__ == "__main__":
    import sys
    import tempfile

    import yaml

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # real run (read only): python -m src.evaluation.photo_vs_render <config.resolved.yaml> <run_dir> <artifacts_dir>
    if len(sys.argv) == 4:
        cfg_real = yaml.safe_load(Path(sys.argv[1]).read_text())
        out = run(cfg_real, sys.argv[2], sys.argv[3])
        print("photo_vs_render ->", out)
        sys.exit(0 if out is not None else 1)

    # synthetic: a coloured box room seen from inside; the "photo" is the render itself -> PSNR inf, coverage 1
    import open3d as o3d
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        run_dir, art, frames = tmp / "run", tmp / "art", tmp / "frames"
        for d in (run_dir / "sfm", run_dir / "mesh", run_dir / "clean" / "sparse", frames):
            d.mkdir(parents=True)
        box = o3d.geometry.TriangleMesh.create_box(4, 4, 3).translate((-2, -2, -1.5)).subdivide_midpoint(3)
        V, F = np.asarray(box.vertices), np.asarray(box.triangles)
        C = ((V - V.min(0)) / np.ptp(V, 0) * 255).astype(np.uint8)
        np.save(run_dir / "mesh" / "vertices.npy", V)
        np.save(run_dir / "mesh" / "faces.npy", F)
        np.save(run_dir / "mesh" / "colors.npy", C)
        Ks = np.array([[200.0, 0, 160], [0, 200.0, 120], [0, 0, 1]])
        Rs, ts, names = [], [], []
        for k, yaw in enumerate(np.linspace(0, 2 * np.pi, 8, endpoint=False)):
            Rw = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
            Rs.append(Rw)
            ts.append(-Rw @ np.array([0.2 * np.cos(yaw), 0.1, 0.2 * np.sin(yaw)]))
            names.append(f"img_{k:04d}.png")
        np.save(run_dir / "sfm" / "cam_R.npy", np.array(Rs))
        np.save(run_dir / "sfm" / "cam_t.npy", np.array(ts))
        np.save(run_dir / "sfm" / "K.npy", Ks)
        np.save(run_dir / "sfm" / "dist.npy", np.zeros(4))
        np.save(run_dir / "sfm" / "image_names.npy", np.array(names))
        mr = MeshRenderer(V, F, C)
        for k, n in enumerate(names):
            img, hit = mr.render(Ks, Rs[k], ts[k], 320, 240)
            assert hit.all(), "camera inside the box must hit everywhere"
            cv2.imwrite(str(frames / n), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cfg_t = {"extraction": {"output_dir": str(frames)}, "dense": {"methods": []},
                 "photo_vs_render": {"n_views": 4, "max_size": 320}}
        out = run(cfg_t, str(run_dir), str(art))
        assert out is not None and out.exists()
        rows = stage_io.read_csv(art / "photo_vs_render.csv")
        assert len(rows) == 4 and all(float(r["coverage"]) == 1.0 for r in rows), rows
        assert all(r["psnr"] == "" or float(r["psnr"]) > 40 for r in rows), rows  # inf -> ""
        assert all(float(r["ssim"]) > 0.99 for r in rows), rows
        assert len(list((run_dir / "figures" / "photo_vs_render").glob("*.png"))) == 4

        # points: the box vertices splatted; same pose must give a valid, partially covered image
        P = np.asarray(box.sample_points_uniformly(1_000_000).points)
        pr = PointRenderer(P, None, 2)
        img, hit = pr.render(Ks, Rs[0], ts[0], 320, 240)
        assert img.shape == (240, 320, 3) and 0.6 < hit.mean() <= 1.0, hit.mean()

        # sparse fallback when the mesh is missing; GS view mapping
        for n in ("vertices", "faces", "colors"):
            (run_dir / "mesh" / f"{n}.npy").unlink()
        np.save(run_dir / "clean" / "sparse" / "points.npy", P)
        assert pick_model(cfg_t, run_dir, "mesh")["label"] == "clean/sparse"
        gsd = run_dir / "gaussian_splatting"
        (gsd / "data" / "images").mkdir(parents=True)
        for n in names + ["img_0008.png"]:
            (gsd / "data" / "images" / n).write_bytes(b"x")
        (gsd / "model" / "test" / "ours_300" / "renders").mkdir(parents=True)
        (gsd / "model" / "test" / "ours_300" / "gt").mkdir(parents=True)
        for k in range(2):
            img = cv2.imread(str(frames / names[0]))
            for sub in ("renders", "gt"):
                cv2.imwrite(str(gsd / "model" / "test" / "ours_300" / sub / f"{k:05d}.png"), img)
        gmap, ours = gs_test_views(run_dir)
        assert gmap == {"img_0000.png": 0, "img_0008.png": 1} and ours.name == "ours_300", gmap
        assert pick_views(names, 2, gmap) == [0, 7]  # one registered GS test view < 2 -> all views
        out = run(cfg_t, str(run_dir), str(art))
        rows = stage_io.read_csv(art / "photo_vs_render.csv")
        assert out is not None and rows[0]["model"] == "clean/sparse" and rows[0]["gs_psnr"] == "", rows[0]
        print("photo_vs_render OK")

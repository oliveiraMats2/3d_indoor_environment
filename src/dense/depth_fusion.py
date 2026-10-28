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
# Stage: dense (shared by every depth-map densifier: plane_sweep, sgm, casmvsnet, mvsnet, patchmatchnet).
# Goal: one undistorted COLMAP workspace for all methods, per-view camera/neighbour data, COLMAP depth/normal
#      map I/O and the common fusion + export, so every method writes the same fused cloud as patch_match.
# In:  <run_dir>/exports/colmap/{images,sparse/0}, <frames_dir>/mvs_exclude.txt, dense.ref_step.
# Out: <run_dir>/dense/undistorted/{images,sparse,stereo/patch-match.cfg} (shared, built once);
#      <run_dir>/dense/<method>/{ws/stereo/{depth_maps,normal_maps,fusion.cfg}, fused.ply, points.npy (model
#      units, COLMAP frame), colors.npy (uint8 RGB), colmap_dense.log}; row method=<method> in
#      <artifacts_dir>/dense_metrics.csv.
# YAML: dense.{max_image_size, ref_step, fusion.*, rerun, <method>.*}; sfm.colmap_bin.
# Rules: depth maps are camera-frame z in model units (0 = invalid) and normals are camera-frame unit vectors
#      facing the camera; maps may be smaller than the undistorted image (fusion rescales K); the
#      references follow the same mvs_exclude.txt + ref_step rule as patch_match.

# >>> BEGIN IMPLEMENTATION >>>
"""Shared plumbing of the depth-map densifiers: workspace, views, COLMAP depth maps and stereo fusion."""
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src import stage_io
from src.dense import patch_match

log = logging.getLogger(__name__)

UNDISTORTED = "undistorted"
N_SOURCES = 10

@dataclass
class View:
    """One undistorted PINHOLE view: world->camera X_c = R X_w + t, sparse depth range, ranked sources."""

    name: str
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    width: int
    height: int
    depth_min: float = 0.0
    depth_max: float = 0.0
    sources: list[str] = field(default_factory=list)

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.t

def method_cfg(cfg: dict, method: str, defaults: dict) -> dict:
    """dense.<method> block over the module defaults (missing keys keep the default)."""

    return {**defaults, **(cfg["dense"].get(method) or {})}

def enabled(cfg: dict, method: str) -> bool:

    dense = cfg["dense"]
    if not dense["enabled"] or method not in dense["methods"]:
        log.info("dense %s disabled", method)
        return False
    return True

def undistorted_workspace(cfg: dict, run_dir) -> Path | None:
    """Build <run_dir>/dense/undistorted once with COLMAP image_undistorter (PINHOLE cameras, max_image_size).
    Returns None when there is no sparse model or the undistortion fails."""

    colmap_dir = Path(run_dir) / "exports" / "colmap"
    ws = Path(run_dir) / "dense" / UNDISTORTED
    if (ws / "sparse" / "cameras.bin").exists() and (ws / "stereo" / "patch-match.cfg").exists():
        return ws
    if not (colmap_dir / "sparse" / "0").is_dir():
        log.warning("dense: no sparse model in %s", colmap_dir / "sparse" / "0")
        return None
    dense = cfg["dense"]
    pm = {**dense["patch_match"], "num_src_images": max(int(dense["patch_match"]["num_src_images"]), N_SOURCES)}
    cmd = patch_match.build_commands(cfg["sfm"]["colmap_bin"], colmap_dir / "images", colmap_dir / "sparse" / "0",
                                     ws, ws / "unused.ply", {**dense, "patch_match": pm}, False)["undistort"]
    ws.mkdir(parents=True, exist_ok=True)
    try:
        patch_match.run_cmd(cmd, ws.parent / "undistort.log")
    except RuntimeError as exc:
        log.warning("dense: image_undistorter failed: %s", exc)
        return None
    return ws

def references(cfg: dict, ws) -> list[str]:
    """Reference image names (sorted): registered images minus mvs_exclude.txt and the ref_step thinning."""

    names = patch_match.registered_names(Path(ws) / "sparse")
    exclude = patch_match.read_exclude(cfg["extraction"]["output_dir"])
    exclude |= patch_match.ref_step_exclude(names, cfg["dense"].get("ref_step", 1))
    return [n for n in names if Path(n).stem not in exclude]

def load_views(ws, n_sources: int = N_SOURCES) -> dict[str, View]:
    """Every view of the undistorted model with its sparse depth range (p1/p99 x 0.8/1.2) and its sources
    (patch-match.cfg order when listed, else ranked by shared 3D points with a 5 deg triangulation-angle prior)."""

    import pycolmap

    rec = pycolmap.Reconstruction(str(Path(ws) / "sparse"))
    views, obs = {}, {}
    xyz = {pid: p.xyz for pid, p in rec.points3D.items()}
    for img in rec.images.values():
        cam = rec.cameras[img.camera_id]
        pose = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
        M = pose.matrix()
        R, t = M[:, :3].astype(np.float64), M[:, 3].astype(np.float64)
        pids = [p.point3D_id for p in img.points2D if p.has_point3D()]
        obs[img.name] = set(pids)
        z = (np.array([xyz[p] for p in pids]) @ R.T + t)[:, 2] if pids else np.zeros(0)
        z = z[z > 0]
        lo, hi = (np.percentile(z, 1) * 0.8, np.percentile(z, 99) * 1.2) if len(z) >= 10 else (0.0, 0.0)
        views[img.name] = View(img.name, cam.calibration_matrix().astype(np.float64), R, t, cam.width, cam.height,
                               float(lo), float(hi))

    listed = _patch_match_sources(Path(ws) / "stereo" / "patch-match.cfg")
    names = list(views)
    covis = _covisibility(names, obs)
    for i, name in enumerate(names):
        v = views[name]
        if listed.get(name):
            v.sources = [s for s in listed[name] if s in views][:n_sources]
            continue
        # angle prior only on the 3k most covisible candidates (sparse product, not all pairs)
        row = covis.getrow(i)
        cand = row.indices[np.argsort(-row.data)][:3 * n_sources]
        scores = []
        for j in cand:
            other = names[j]
            if other == name:
                continue
            shared = obs[name] & obs[other]
            angle = _angle_deg(v.center, views[other].center, xyz, shared)
            scores.append((len(shared) * np.exp(-((angle - 5.0) ** 2) / (2 * (1.0 if angle < 5 else 10.0) ** 2)),
                           other))
        v.sources = [o for _, o in sorted(scores, reverse=True)[:n_sources]]
    return views

def _covisibility(names: list[str], obs: dict[str, set]):
    """Sparse (n, n) count of 3D points shared by every pair of views (image x point incidence product)."""

    from scipy import sparse

    pids = sorted(set().union(*obs.values())) if obs else []
    col = {p: k for k, p in enumerate(pids)}
    rows = np.concatenate([np.full(len(obs[n]), i) for i, n in enumerate(names)]) if names else np.zeros(0)
    cols = np.fromiter((col[p] for n in names for p in obs[n]), dtype=np.int64)
    A = sparse.csr_matrix((np.ones(len(cols), np.int32), (rows.astype(np.int64), cols)),
                          shape=(len(names), len(pids)))
    return (A @ A.T).tocsr()

def _angle_deg(c1, c2, xyz, pids) -> float:

    pts = np.array([xyz[p] for p in list(pids)[:200]])
    a, b = pts - c1, pts - c2
    cos = np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
    return float(np.degrees(np.median(np.arccos(np.clip(cos, -1, 1)))))

def _patch_match_sources(path) -> dict[str, list[str]]:
    """patch-match.cfg as {reference: [sources]}; '__auto__, k' entries are left for the covisibility ranking."""

    if not Path(path).exists():
        return {}
    lines = [l.strip() for l in Path(path).read_text().splitlines() if l.strip()]
    out = {}
    for name, src in zip(lines[0::2], lines[1::2]):
        if not src.startswith("__"):
            out[name] = [s.strip() for s in src.split(",") if s.strip()]
    return out

def read_image(ws, name: str, gray: bool = False) -> np.ndarray:
    """Undistorted image as float32 in [0, 1]: (H, W) when gray, else (H, W, 3) RGB."""

    import cv2

    img = cv2.imread(str(Path(ws) / "images" / name), cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(Path(ws) / "images" / name)
    if not gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0

def write_array(path, arr: np.ndarray) -> None:
    """COLMAP Mat<float> file: ASCII 'W&H&C&' header, then float32 column-major (read_write_dense.py)."""

    arr = np.asarray(arr, np.float32)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    h, w, c = arr.shape
    with open(path, "wb") as fh:
        fh.write(f"{w}&{h}&{c}&".encode())
        np.transpose(arr, (1, 0, 2)).reshape(-1, order="F").astype("<f4").tofile(fh)

def read_array(path) -> np.ndarray:
    """Inverse of write_array: (H, W) for one channel, else (H, W, C)."""

    with open(path, "rb") as fh:
        w, h, c = (int(x) for x in fh.read(64).split(b"&")[:3])
        fh.seek(0)
        amps = 0
        while amps < 3:
            amps += fh.read(1) == b"&"
        data = np.fromfile(fh, np.float32)
    arr = np.transpose(data.reshape((w, h, c), order="F"), (1, 0, 2))
    return arr[:, :, 0] if c == 1 else arr

def scale_K(K: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """Intrinsics of the image resized by (sx, sy) (pixel-centre convention of COLMAP)."""

    S = np.diag([sx, sy, 1.0])
    return S @ K

def normals_from_depth(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Camera-frame unit normals from a depth map (cross product of the back-projected neighbours), oriented
    towards the camera; zero where the depth or a neighbour is invalid."""

    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    Kinv = np.linalg.inv(K).astype(np.float32)
    rays = np.stack([u, v, np.ones_like(u)], -1) @ Kinv.T
    P = rays * depth[..., None]
    dx = np.zeros_like(P)
    dy = np.zeros_like(P)
    dx[:, 1:-1] = P[:, 2:] - P[:, :-2]
    dy[1:-1] = P[2:] - P[:-2]
    n = np.cross(dx, dy)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    n = np.where(norm > 1e-12, n / np.maximum(norm, 1e-12), 0.0)
    n = np.where((np.sum(n * rays, -1, keepdims=True) > 0), -n, n)
    valid = depth > 0
    valid[:, 1:-1] &= (depth[:, 2:] > 0) & (depth[:, :-2] > 0)
    valid[1:-1] &= (depth[2:] > 0) & (depth[:-2] > 0)
    valid[[0, -1], :] = False
    valid[:, [0, -1]] = False
    return np.where(valid[..., None], n, 0.0).astype(np.float32)

def method_workspace(run_dir, method: str, undistorted) -> Path:
    """dense/<method>/ws with images/ and sparse/ linked to the shared undistorted workspace."""

    ws = Path(run_dir) / "dense" / method / "ws"
    (ws / "stereo" / "depth_maps").mkdir(parents=True, exist_ok=True)
    (ws / "stereo" / "normal_maps").mkdir(parents=True, exist_ok=True)
    for sub in ("images", "sparse"):
        link = ws / sub
        if not link.exists():
            link.symlink_to(Path(undistorted).resolve() / sub, target_is_directory=True)
    return ws

def save_depth(ws, name: str, depth: np.ndarray, K: np.ndarray, normals: np.ndarray | None = None) -> None:
    """Write <name>.geometric.bin depth + normal maps (K is the intrinsics at the depth-map resolution)."""

    depth = np.where(np.isfinite(depth) & (depth > 0), depth, 0).astype(np.float32)
    normals = normals_from_depth(depth, K) if normals is None else normals
    write_array(Path(ws) / "stereo" / "depth_maps" / f"{name}.geometric.bin", depth)
    write_array(Path(ws) / "stereo" / "normal_maps" / f"{name}.geometric.bin", normals)

def reuse(cfg: dict, run_dir, artifacts_dir, method: str, row: dict) -> Path | None:
    """With dense.rerun false and an existing fused.ply: re-export points/colors and keep the csv row."""

    out_dir = Path(run_dir) / "dense" / method
    fused = out_dir / "fused.ply"
    if not fused.exists() or cfg["dense"]["rerun"]:
        return None
    log.info("reusing %s (dense.rerun=false)", fused)
    points, colors = patch_match.load_ply(fused)
    np.save(out_dir / "points.npy", points)
    np.save(out_dir / "colors.npy", colors)
    csv_path = Path(artifacts_dir) / "dense_metrics.csv"
    old = [r for r in stage_io.read_csv(csv_path) if r.get("method") == method]
    row = {**row, **(old[0] if old else {}), "n_dense": len(points)}
    stage_io.upsert_csv(csv_path, row, key="method")
    stage_io.log_metrics(log, row)
    return fused

def fuse_and_export(cfg: dict, run_dir, artifacts_dir, method: str, ws, names: list[str], row: dict) -> Path | None:
    """COLMAP stereo_fusion (geometric input, dense.fusion options) over the written depth maps, then export
    fused.ply / points.npy / colors.npy and upsert the dense_metrics.csv row (t_fusion_s, n_dense, t_s)."""

    out_dir = Path(run_dir) / "dense" / method
    fused, log_path = out_dir / "fused.ply", out_dir / "colmap_dense.log"
    written = [n for n in names if (Path(ws) / "stereo" / "depth_maps" / f"{n}.geometric.bin").exists()]
    if not written:
        log.warning("dense %s: no depth maps written, nothing to fuse", method)
        return None
    (Path(ws) / "stereo" / "fusion.cfg").write_text("".join(f"{n}\n" for n in written))
    cmd = patch_match.build_commands(cfg["sfm"]["colmap_bin"], "", "", ws, fused,
                                     {**cfg["dense"], "patch_match": {**cfg["dense"]["patch_match"],
                                                                      "geom_consistency": True}}, True)["fusion"]
    log_path.touch()
    try:
        row["t_fusion_s"] = round(patch_match.run_cmd(cmd, log_path), 2)
    except RuntimeError as exc:
        log.warning("dense %s: fusion failed: %s", method, exc)
        return None
    points, colors = patch_match.load_ply(fused)
    np.save(out_dir / "points.npy", points)
    np.save(out_dir / "colors.npy", colors)
    row.update(n_dense=len(points), n_ref_images=len(written), method=method)
    row["t_s"] = round(sum(float(row.get(k) or 0) for k in ("t_undistort_s", "t_depth_s", "t_fusion_s")), 2)
    stage_io.upsert_csv(Path(artifacts_dir) / "dense_metrics.csv", row, key="method")
    stage_io.log_metrics(log, row)
    return fused

def base_row(cfg: dict, method: str) -> dict:
    """dense_metrics.csv row shared by every densifier (patch_match columns + t_depth_s, vram_peak_gb)."""

    dense = cfg["dense"]
    return {"method": method, "n_dense": "", "t_s": "", "max_image_size": dense["max_image_size"],
            "t_undistort_s": "", "t_patch_match_s": "", "t_fusion_s": "", "n_ref_images": "", "n_excluded": "",
            "backend": "", "ref_step": dense.get("ref_step", 1), "t_depth_s": "", "vram_peak_gb": ""}

class Timer:
    """with Timer() as t: ...; t.s -> elapsed seconds."""

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.s = time.perf_counter() - self.t0

# <<< END IMPLEMENTATION <<<

if __name__ == "__main__":
    import tempfile

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    with tempfile.TemporaryDirectory() as tmp:
        rng = np.random.default_rng(0)
        a = rng.random((7, 5, 3)).astype(np.float32)
        write_array(Path(tmp) / "a.bin", a)
        assert np.allclose(read_array(Path(tmp) / "a.bin"), a)
        d = rng.random((6, 4)).astype(np.float32)
        write_array(Path(tmp) / "d.bin", d)
        assert np.allclose(read_array(Path(tmp) / "d.bin"), d)
        # fronto-parallel plane at z=2 -> normals (0, 0, -1) facing the camera
        K = np.array([[100, 0, 32], [0, 100, 24], [0, 0, 1]], float)
        n = normals_from_depth(np.full((48, 64), 2.0, np.float32), K)
        assert np.allclose(n[10:-10, 10:-10], [0, 0, -1], atol=1e-5), n[20, 20]
        assert np.allclose(n[0, 0], 0)
    print("depth_fusion self-test OK")

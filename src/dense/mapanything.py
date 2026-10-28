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
# Stage: dense (method mapanything) - MapAnything (Keetha et al. 2025, facebookresearch/map-anything):
#      feed-forward metric 3D reconstruction that also takes geometric inputs (intrinsics, poses).
# Goal: a dense cloud of the SfM scene from one MapAnything pass over up to max_frames registered frames, fed
#      with the undistorted PINHOLE intrinsics and the SfM cam2world poses (non-metric), in the SfM frame.
# In:  <run_dir>/dense/undistorted/{images,sparse} (depth_fusion.undistorted_workspace), mvs_exclude.txt.
# Out: <run_dir>/dense/mapanything/{fused.ply, points.npy (model units, COLMAP frame of exports/colmap/sparse/0),
#      colors.npy (uint8 RGB), mapanything.log}; row method=mapanything in <artifacts_dir>/dense_metrics.csv
#      with t_load_s, t_depth_s (inference), vram_peak_gb, metric_scaling_factor, align_scale, align_rmse,
#      cm_per_unit_mapanything.
# YAML: dense.mapanything.{weights, max_frames, use_sfm_poses, use_intrinsics, conf_percentile, voxel}.
# Rules: official weights only (facebook/map-anything, CC BY-NC 4.0, Hugging Face hub). The network runs in a
#      spawned child (bf16 autocast, memory_efficient_inference) so a CUDA abort cannot kill main.py and the
#      VRAM is released. Frames = registered minus mvs_exclude.txt, uniform subsample (as upstream
#      scripts/demo_inference_on_colmap_outputs.py: undistorted RGB, K, cam2world, is_metric_scale False).
#      Points kept = non-ambiguous/edge mask AND confidence >= the conf_percentile of the masked pixels. The
#      output is metric and anchored on view 0, so a Umeyama Sim3 of the predicted camera centres onto the SfM
#      centres brings it to the SfM frame; 100 / align_scale is the cm per SfM unit implied by the network
#      (an independent metric scale estimate). voxel auto = 1% of 2x the largest IQR axis (upstream export).

# >>> BEGIN IMPLEMENTATION >>>
"""MapAnything densifier: one feed-forward pass with SfM intrinsics and poses, merged into a cloud in the SfM frame."""
import importlib
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

from src import stage_io
from src.dense import depth_fusion, patch_match
from src.sfm import gpu

log = logging.getLogger(__name__)

METHOD = "mapanything"
DEFAULTS = {"weights": "facebook/map-anything", "max_frames": 80, "use_sfm_poses": True, "use_intrinsics": True,
            "conf_percentile": 50, "voxel": "auto"}
VOXEL_FRACTION = 0.01

def upstream(module: str):
    """Import a module of the installed mapanything package; this file's folder is dropped from sys.path
    first (run as a script it would shadow the package with itself)."""

    here = str(Path(__file__).resolve().parent)
    saved = sys.path[:]
    sys.path[:] = [p for p in sys.path if p and Path(p).resolve().as_posix() != here]
    try:
        return importlib.import_module(module)
    finally:
        sys.path[:] = saved

def frames(cfg: dict, ws, max_frames: int) -> list[dict]:
    """Registered undistorted views (minus mvs_exclude.txt) uniformly subsampled: {name, K, cam2world}."""

    import pycolmap

    rec = pycolmap.Reconstruction(str(Path(ws) / "sparse"))
    exclude = patch_match.read_exclude(cfg["extraction"]["output_dir"])
    views = []
    for img in sorted(rec.images.values(), key=lambda im: im.name):
        if Path(img.name).stem in exclude or not img.has_pose:
            continue
        pose = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
        w2c = np.eye(4)
        w2c[:3] = pose.matrix()
        views.append({"name": img.name, "K": rec.cameras[img.camera_id].calibration_matrix(),
                      "cam2world": np.linalg.inv(w2c)})
    return gpu.subsample(views, max_frames)

def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity (s, R, t) minimising ||dst - (s R src + t)||^2 (Umeyama 1991)."""

    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd(b.T @ a / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / (a ** 2).sum(1).mean())
    return s, R, mu_d - s * R @ mu_s

def voxel_size(points: np.ndarray, voxel) -> float:
    """voxel 'auto' -> VOXEL_FRACTION of 2x the largest IQR axis; a number -> itself; 0/None -> 0 (off)."""

    if voxel in (None, 0, False):
        return 0.0
    if voxel == "auto":
        iqr = (np.percentile(points, 75, axis=0) - np.percentile(points, 25, axis=0)).max()
        return float(2 * iqr * VOXEL_FRACTION)
    return float(voxel)

def voxel_down(points: np.ndarray, colors: np.ndarray, size: float) -> tuple[np.ndarray, np.ndarray]:
    """Mean point and colour per occupied voxel (no downsample when size <= 0)."""

    if size <= 0 or not len(points):
        return points, colors
    _, inv, counts = np.unique(np.floor(points / size).astype(np.int64), axis=0, return_inverse=True,
                               return_counts=True)
    inv = inv.reshape(-1)
    mean = lambda x: np.stack([np.bincount(inv, x[:, k]) for k in range(3)], 1) / counts[:, None]
    return mean(points.astype(np.float64)).astype(np.float32), np.round(mean(colors.astype(np.float64))).astype(np.uint8)

def _child(ws: str, views: list[dict], mcfg: dict, out: str, state_path: str) -> None:
    """Spawned child: MapAnything inference -> out (npz: masked points, colours, confidence, predicted
    cam2world, metric_scaling_factor)."""

    out = Path(out)
    fd = os.open(out.parent / "mapanything.log", os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    from PIL import Image

    MapAnything = upstream("mapanything.models").MapAnything
    preprocess_inputs = upstream("mapanything.utils.image").preprocess_inputs
    state = gpu.State(state_path)
    print(f"mapanything: {len(views)} views, weights {mcfg['weights']}", flush=True)

    def load():
        model = MapAnything.from_pretrained(mcfg["weights"]).to("cuda").eval()
        raw = [{"img": torch.from_numpy(np.asarray(Image.open(Path(ws) / "images" / v["name"]).convert("RGB")).copy()),
                "intrinsics": torch.from_numpy(np.asarray(v["K"], np.float32)),
                "camera_poses": torch.from_numpy(np.asarray(v["cam2world"], np.float32)),
                "is_metric_scale": torch.tensor([False])} for v in views]
        return model, preprocess_inputs(raw, verbose=False)

    model, inputs = state.step("load", load)

    def infer():
        with torch.no_grad():
            return model.infer(inputs, memory_efficient_inference=True, use_amp=True, amp_dtype="bf16",
                               apply_mask=True, mask_edges=True, ignore_calibration_inputs=not mcfg["use_intrinsics"],
                               ignore_depth_inputs=True, ignore_pose_inputs=not mcfg["use_sfm_poses"],
                               ignore_depth_scale_inputs=True, ignore_pose_scale_inputs=True)

    preds = state.step("depth", infer)
    pts, cols, conf, poses = [], [], [], []
    for p in preds:
        m = p["mask"][0, ..., 0].bool() & torch.isfinite(p["pts3d"][0]).all(-1)
        pts.append(p["pts3d"][0][m].float().cpu().numpy())
        cols.append((p["img_no_norm"][0][m].float().clamp(0, 1) * 255).round().byte().cpu().numpy())
        conf.append(p["conf"][0][m].float().cpu().numpy())
        poses.append(p["camera_poses"][0].float().cpu().numpy())
    scale = float(torch.stack([p["metric_scaling_factor"].float().reshape(-1)[0] for p in preds]).mean())
    np.savez(out, points=np.concatenate(pts), colors=np.concatenate(cols), conf=np.concatenate(conf),
             poses=np.stack(poses), metric_scaling_factor=scale)
    state.done()

def run(cfg: dict, run_dir: str, artifacts_dir: str) -> Path | None:
    """dense/mapanything/fused.ply in the SfM frame, or None when disabled/skipped/failed."""

    if not depth_fusion.enabled(cfg, METHOD):
        return None
    mcfg = depth_fusion.method_cfg(cfg, METHOD, DEFAULTS)
    row = {**depth_fusion.base_row(cfg, METHOD), "backend": "mapanything", "metric_scaling_factor": "",
           "align_scale": "", "align_rmse": "", "cm_per_unit_mapanything": ""}
    if (reused := depth_fusion.reuse(cfg, run_dir, artifacts_dir, METHOD, row)) is not None:
        return reused
    if not gpu.cuda_ok():
        log.warning("dense %s: no CUDA device, skipped", METHOD)
        return None
    try:
        upstream("mapanything.models")
    except ImportError as exc:
        log.warning("dense %s: mapanything not importable (%s; install requirements-gpu): skipped", METHOD, exc)
        return None

    t0 = time.perf_counter()
    ws = depth_fusion.undistorted_workspace(cfg, run_dir)
    if ws is None:
        return None
    row["t_undistort_s"] = round(time.perf_counter() - t0, 2)
    views = frames(cfg, ws, int(mcfg["max_frames"]))
    if len(views) < 2:
        log.warning("dense %s: %d registered views, need >= 2: skipped", METHOD, len(views))
        return None
    out_dir = Path(run_dir) / "dense" / METHOD
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = out_dir / "raw.npz"
    raw.unlink(missing_ok=True)
    log.info("dense %s: %d views, weights %s (poses=%s, intrinsics=%s)", METHOD, len(views), mcfg["weights"],
             mcfg["use_sfm_poses"], mcfg["use_intrinsics"])
    exitcode, state = gpu.spawn(_child, (str(ws), views, mcfg, str(raw)), out_dir)
    if exitcode != 0 or not raw.exists():
        log.warning("dense %s: failed at %s (exit %s, see %s)", METHOD, state.get("running", "start"), exitcode,
                    out_dir / "mapanything.log")
        return None

    data = np.load(raw)
    points, colors, conf = data["points"], data["colors"], data["conf"]
    keep = conf >= np.percentile(conf, float(mcfg["conf_percentile"])) if len(conf) else conf > 0
    points, colors = points[keep], colors[keep]
    # MapAnything returns a metric frame anchored on view 0: bring it to the SfM frame through the cameras
    sfm_c = np.array([v["cam2world"][:3, 3] for v in views])
    s, R, t = umeyama(data["poses"][:, :3, 3].astype(np.float64), sfm_c)
    rmse = float(np.sqrt(np.mean(np.sum((s * data["poses"][:, :3, 3] @ R.T + t - sfm_c) ** 2, 1))))
    points = (s * points.astype(np.float64) @ R.T + t).astype(np.float32)
    size = voxel_size(points, mcfg["voxel"])
    points, colors = voxel_down(points, colors, size)

    stage_io.write_ply(out_dir / "fused.ply", points, colors)
    np.save(out_dir / "points.npy", points)
    np.save(out_dir / "colors.npy", colors)
    raw.unlink(missing_ok=True)
    row.update({k: v for k, v in state.items() if k in ("t_load_s", "vram_peak_gb")})
    row.update(t_depth_s=state.get("t_depth_s", ""), n_dense=len(points), n_ref_images=len(views),
               metric_scaling_factor=round(float(data["metric_scaling_factor"]), 4), align_scale=round(s, 4),
               align_rmse=round(rmse, 4), cm_per_unit_mapanything=round(100.0 / s, 2), voxel=round(size, 5),
               t_s=round(time.perf_counter() - t0, 2))
    stage_io.upsert_csv(Path(artifacts_dir) / "dense_metrics.csv", row, key="method")
    stage_io.log_metrics(log, row)
    return out_dir / "fused.ply"

# <<< END IMPLEMENTATION <<<

if __name__ == "__main__":
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(0)
    src = rng.normal(size=(20, 3))
    R0 = Rotation.from_rotvec([0.3, -0.2, 1.0]).as_matrix()
    s, R, t = umeyama(src, 3.5 * src @ R0.T + [1, 2, 3])
    assert abs(s - 3.5) < 1e-9 and np.allclose(R, R0) and np.allclose(t, [1, 2, 3])
    p = rng.random((1000, 3)).astype(np.float32)
    c = rng.integers(0, 255, (1000, 3)).astype(np.uint8)
    q, qc = voxel_down(p, c, 0.5)
    assert len(q) == 8 and qc.dtype == np.uint8 and np.all((q > 0) & (q < 1))
    assert voxel_down(p, c, 0)[0] is p and voxel_size(p, None) == 0 and voxel_size(p, 0.2) == 0.2
    assert 0.005 < voxel_size(p, "auto") < 0.015
    print("mapanything self-test OK")

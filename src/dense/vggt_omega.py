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
# Stage: dense, method vggt_omega (neural multi-view reconstruction; VGGT-Omega, Wang et al. 2026,
#        facebookresearch/vggt-omega; fallback VGGT-1B, facebookresearch/vggt).
# Goal: a dense cloud from one feed-forward pass (camera + depth heads) over up to max_frames registered frames,
#       expressed in the COLMAP frame of exports/colmap/sparse/0, so mesh + evaluation do not change.
# In:  <run_dir>/exports/colmap/{images, sparse/0}; dense.vggt_omega.{model, checkpoint, max_frames,
#      conf_percentile, image_size, use_sfm_poses, edge_rtol, ransac_thresh, voxel_size, dtype}.
# Out: <run_dir>/dense/vggt_omega/{fused.ply, points.npy (model units, COLMAP frame), colors.npy uint8,
#      vggt_omega.log}; row method=vggt_omega in <artifacts_dir>/dense_metrics.csv (n_dense, t_s, t_depth_s,
#      vram_peak_gb, model, n_frames, align_rmse, align_inliers, align_scale).
# Rules: official weights only: facebook/VGGT-Omega vggt_omega_1b_512.pt (gated, FAIR NC license; a local
#      checkpoint path may be given) or facebook/VGGT-1B (CC BY-NC 4.0). No access / no package / no GPU ->
#      warning and None. The network runs in a spawned child (gpu.spawn) so the VRAM is freed when it ends.
#      Depth is kept where conf > the conf_percentile of all pixels and away from depth edges; the network
#      cameras are aligned to the SfM ones by a RANSAC Umeyama similarity on the camera centres;
#      use_sfm_poses true unprojects the scaled depth with the SfM poses (network intrinsics), false moves the
#      network cloud with the similarity.

# >>> BEGIN IMPLEMENTATION >>>
"""vggt_omega densifier: feed-forward depth + cameras, aligned to the SfM frame by a camera-centre similarity."""
import logging
import os
from pathlib import Path

import numpy as np

from src import stage_io
from src.dense import depth_fusion
from src.sfm import gpu

log = logging.getLogger(__name__)

METHOD = "vggt_omega"
MODELS = ("vggt_1b", "vggt_omega")
OMEGA_REPO, OMEGA_FILE = "facebook/VGGT-Omega", "vggt_omega_1b_512.pt"
VGGT_REPO = "facebook/VGGT-1B"
DEFAULTS = {"model": "vggt_1b", "checkpoint": None, "max_frames": 80, "conf_percentile": 50.0,
            "image_size": None, "use_sfm_poses": True, "edge_rtol": 0.03, "ransac_thresh": 0.05,
            "voxel_size": None, "dtype": "auto"}

def sfm_cameras(model_dir) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """{image name: (R, t)} world->camera of the registered images of a COLMAP model."""

    import pycolmap

    rec = pycolmap.Reconstruction(str(model_dir))
    out = {}
    for img in rec.images.values():
        if not img.has_pose:
            continue
        pose = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
        M = pose.matrix()
        out[img.name] = (M[:, :3].astype(np.float64), M[:, 3].astype(np.float64))
    return out

def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity (s, R, t) minimising ||dst - (s R src + t)||^2 (Umeyama 1991)."""

    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd(xd.T @ xs / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var = (xs ** 2).sum() / len(src)
    s = float(np.trace(np.diag(D) @ S) / var) if var > 0 else 1.0
    return s, R, mu_d - s * R @ mu_s

def ransac_umeyama(src: np.ndarray, dst: np.ndarray, thresh_frac: float, iters: int = 2000,
                   seed: int = 0) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Robust similarity: 3-point samples, inliers within thresh_frac x the median spread of dst around its
    centroid, refit on the largest inlier set. Returns (s, R, t, inlier mask)."""

    n = len(src)
    if n < 3:
        raise ValueError("need >= 3 camera centres for a similarity")
    thresh = thresh_frac * float(np.median(np.linalg.norm(dst - dst.mean(0), axis=1)))
    rng = np.random.default_rng(seed)
    best = np.ones(n, bool)
    best_n = -1
    for _ in range(iters if n > 3 else 1):
        idx = rng.choice(n, 3, replace=False)
        s, R, t = umeyama(src[idx], dst[idx])
        inl = np.linalg.norm(s * src @ R.T + t - dst, axis=1) < thresh
        if inl.sum() > best_n:
            best, best_n = inl, int(inl.sum())
    if best_n < 3:
        best = np.ones(n, bool)
    for _ in range(3):  # refit + re-score on the inliers
        s, R, t = umeyama(src[best], dst[best])
        inl = np.linalg.norm(s * src @ R.T + t - dst, axis=1) < thresh
        if inl.sum() < 3 or np.array_equal(inl, best):
            break
        best = inl
    return s, R, t, best

def depth_edges(depth: np.ndarray, rtol: float) -> np.ndarray:
    """(S, H, W) mask of pixels whose 3x3 depth range exceeds rtol x depth (flying pixels at occlusions)."""

    import cv2

    k = np.ones((3, 3), np.uint8)
    out = np.zeros(depth.shape, bool)
    for i, d in enumerate(depth.astype(np.float32)):
        out[i] = (cv2.dilate(d, k) - cv2.erode(d, k)) > rtol * np.maximum(d, 1e-6)
    return out

def unproject(depth: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """World points X = R^T (z K^-1 [u v 1] - t) of the masked pixels of one depth map (world->camera R, t)."""

    v, u = np.nonzero(mask)
    z = depth[v, u]
    cam = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], 1)
    return (cam - t) @ R

def voxel_down(points: np.ndarray, colors: np.ndarray, voxel: float) -> tuple[np.ndarray, np.ndarray]:
    """One point per voxel (mean position and colour)."""

    import open3d as o3d

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.astype(np.float64)))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    pcd = pcd.voxel_down_sample(float(voxel))
    return np.asarray(pcd.points), (np.clip(np.asarray(pcd.colors), 0, 1) * 255).round().astype(np.uint8)

def resolve_weights(vcfg: dict) -> str | None:
    """Checkpoint of the chosen model: VGGT-1B hub id, or a local / downloaded VGGT-Omega .pt (None + warning
    when the gated repo is not accessible)."""

    if vcfg["model"] == "vggt_1b":
        return vcfg["checkpoint"] or VGGT_REPO
    if vcfg["checkpoint"]:
        path = Path(os.path.expanduser(str(vcfg["checkpoint"])))
        if path.is_file():
            return str(path)
        log.warning("dense %s: checkpoint %s not found", METHOD, path)
        return None
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(OMEGA_REPO, OMEGA_FILE)
    except Exception as exc:  # noqa: BLE001 - gated repo, no login, offline
        log.warning("dense %s: cannot get %s/%s (%s: %s). Request access at https://huggingface.co/%s and log "
                    "in (hf auth login), or set dense.vggt_omega.checkpoint to a local file, or use model: "
                    "vggt_1b", METHOD, OMEGA_REPO, OMEGA_FILE, type(exc).__name__, str(exc).splitlines()[0],
                    OMEGA_REPO)
        return None

def _child(vcfg: dict, weights: str, frames: list[str], out_npz: str, seed: int, state_path: str) -> None:
    """Spawned child: network forward (camera + depth heads) -> npz of depth, conf, extrinsic, intrinsic, rgb."""

    out_dir = Path(out_npz).parent
    fd = os.open(out_dir / f"{METHOD}.log", os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    import torch

    torch.manual_seed(seed)
    state = gpu.State(state_path)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(vcfg["dtype"]) or \
        (torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16)
    omega = vcfg["model"] == "vggt_omega"
    size = int(vcfg["image_size"] or (512 if omega else 518))
    print(f"{vcfg['model']}: {len(frames)} frames, long side {size}, weights {weights}, dtype {dtype}", flush=True)

    def load():
        if omega:
            from vggt_omega.models import VGGTOmega
            from vggt_omega.utils.load_fn import load_and_preprocess_images

            model = VGGTOmega(autocast=dtype != torch.float32)
            model.load_state_dict(torch.load(weights, map_location="cpu"))
            images = load_and_preprocess_images(frames, mode="max_size", image_resolution=size)
        else:
            from vggt.models.vggt import VGGT

            from src.sfm import vggt as sfm_vggt

            model = VGGT.from_pretrained(weights)
            model.point_head = model.track_head = None  # cameras + depth only: less VRAM
            images = sfm_vggt.load_images([Path(f) for f in frames],
                                          sfm_vggt.input_size(*gpu.image_size(Path(frames[0])), size))
        return model.eval().to("cuda"), images.to("cuda")

    model, images = state.step("load", load)

    def forward():
        with torch.inference_mode():
            if omega:
                from vggt_omega.utils.pose_enc import encoding_to_camera as decode
                pred = model(images)
            else:
                from vggt.utils.pose_enc import pose_encoding_to_extri_intri as decode
                with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                    pred = model(images)
            extr, intr = decode(pred["pose_enc"], images.shape[-2:])
        return {"depth": pred["depth"][0, ..., 0].float().cpu().numpy(),
                "conf": pred["depth_conf"][0].float().cpu().numpy(),
                "extrinsic": extr[0].float().cpu().numpy(), "intrinsic": intr[0].float().cpu().numpy()}

    out = state.step("depth", forward)
    rgb = (images.permute(0, 2, 3, 1).clamp(0, 1) * 255).round().byte().cpu().numpy()
    np.savez(out_npz, rgb=rgb, **out)
    state.done()

def to_points(pred: dict, names: list[str], sfm: dict, vcfg: dict, seed: int = 0) -> tuple:
    """(points in the SfM frame, colours uint8, alignment info) from the network prediction."""

    depth, conf, extr, intr, rgb = (pred[k] for k in ("depth", "conf", "extrinsic", "intrinsic", "rgb"))
    c_net = np.stack([-E[:, :3].T @ E[:, 3] for E in extr])
    c_sfm = np.stack([-sfm[n][0].T @ sfm[n][1] for n in names])
    s, R, t, inl = ransac_umeyama(c_net, c_sfm, float(vcfg["ransac_thresh"]), seed=seed)
    res = np.linalg.norm(s * c_net @ R.T + t - c_sfm, axis=1)
    info = {"align_rmse": float(np.sqrt(np.mean(res[inl] ** 2))), "align_rmse_all": float(np.sqrt(np.mean(res ** 2))),
            "align_inliers": int(inl.sum()), "align_scale": float(s)}

    thr = np.percentile(conf, float(vcfg["conf_percentile"]))
    mask = (conf > thr) & (depth > 0) & np.isfinite(depth)
    if vcfg["edge_rtol"]:
        mask &= ~depth_edges(depth, float(vcfg["edge_rtol"]))
    pts, cols = [], []
    for i, name in enumerate(names):
        if vcfg["use_sfm_poses"]:
            if not inl[i]:  # network depth of a view whose camera disagrees with the SfM: scale less trusted
                continue
            Rs, ts = sfm[name]
            pts.append(unproject(depth[i] * s, intr[i], Rs, ts, mask[i]))
        else:
            pts.append(unproject(depth[i], intr[i], extr[i][:, :3], extr[i][:, 3], mask[i]) @ (s * R).T + t)
        cols.append(rgb[i][mask[i]])
    return np.concatenate(pts), np.concatenate(cols), info

def run(cfg: dict, run_dir: str, artifacts_dir: str) -> Path | None:
    """dense/vggt_omega/fused.ply in the COLMAP frame of exports/colmap/sparse/0, or None when skipped."""

    if not depth_fusion.enabled(cfg, METHOD):
        return None
    vcfg = depth_fusion.method_cfg(cfg, METHOD, DEFAULTS)
    row = {**depth_fusion.base_row(cfg, METHOD), "model": vcfg["model"], "n_frames": "", "align_rmse": "",
           "align_inliers": "", "align_scale": ""}
    reused = depth_fusion.reuse(cfg, run_dir, artifacts_dir, METHOD, row)
    if reused:
        return reused
    if vcfg["model"] not in MODELS:
        log.warning("dense %s: model %r not in %s: skipped", METHOD, vcfg["model"], MODELS)
        return None
    colmap_dir = Path(run_dir) / "exports" / "colmap"
    model_dir = colmap_dir / "sparse" / "0"
    if not (model_dir / "images.bin").exists() and not (model_dir / "images.txt").exists():
        log.warning("dense %s: no sparse model in %s: skipped", METHOD, model_dir)
        return None
    sfm = sfm_cameras(model_dir)
    names = gpu.subsample(sorted(n for n in sfm if (colmap_dir / "images" / n).exists()), vcfg["max_frames"])
    if len(names) < 3:
        log.warning("dense %s: %d registered frames with images, need >= 3: skipped", METHOD, len(names))
        return None
    if not gpu.cuda_ok():
        return None
    package = "vggt_omega" if vcfg["model"] == "vggt_omega" else "vggt"
    try:
        __import__(package)
    except ImportError as exc:
        log.warning("dense %s: %s not importable (%s; install requirements-nodeps.txt): skipped", METHOD, package, exc)
        return None
    weights = resolve_weights(vcfg)
    if weights is None:
        return None

    out_dir = Path(run_dir) / "dense" / METHOD
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / "prediction.npz"
    npz.unlink(missing_ok=True)
    log.info("dense %s: %s on %d/%d registered frames", METHOD, vcfg["model"], len(names), len(sfm))
    with depth_fusion.Timer() as total:
        exitcode, state = gpu.spawn(_child, (vcfg, weights, [str(colmap_dir / "images" / n) for n in names],
                                             str(npz), int(cfg["run"].get("seed", 0))), out_dir)
        if exitcode != 0 or not npz.exists():
            log.warning("dense %s: network failed at %s (exit code %s, see %s)", METHOD, state.get("running", "start"),
                        exitcode, out_dir / f"{METHOD}.log")
            return None
        with np.load(npz) as f:
            pred = {k: f[k] for k in f.files}
        npz.unlink()
        points, colors, info = to_points(pred, names, sfm, vcfg, int(cfg["run"].get("seed", 0)))
        voxel = vcfg["voxel_size"]
        if voxel is None:  # 0.1 % of the robust bounding-box diagonal
            lo, hi = np.percentile(points, [2, 98], axis=0)
            voxel = 1e-3 * float(np.linalg.norm(hi - lo))
        if voxel and voxel > 0:
            points, colors = voxel_down(points, colors, voxel)
        fused = out_dir / "fused.ply"
        stage_io.write_ply(fused, points, colors)
        np.save(out_dir / "points.npy", points)
        np.save(out_dir / "colors.npy", colors)
    if info["align_inliers"] < len(names) // 2:
        log.warning("dense %s: only %d/%d network cameras agree with the SfM ones (rmse %.4f): check the cloud",
                    METHOD, info["align_inliers"], len(names), info["align_rmse"])
    row.update(n_dense=len(points), t_s=round(total.s, 2), t_depth_s=state.get("t_depth_s", ""),
               vram_peak_gb=state.get("vram_peak_gb", ""), n_frames=len(names), n_ref_images=len(names),
               align_rmse=round(info["align_rmse"], 5), align_inliers=info["align_inliers"],
               align_scale=round(info["align_scale"], 5), backend=vcfg["model"])
    stage_io.upsert_csv(Path(artifacts_dir) / "dense_metrics.csv", row, key="method")
    stage_io.log_metrics(log, row)
    return fused

if __name__ == "__main__":
    import tempfile

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    rng = np.random.default_rng(0)

    def rot(a, b, c):
        import cv2
        return cv2.Rodrigues(np.array([a, b, c], float))[0]

    # Umeyama recovers a similarity; RANSAC ignores two gross outliers
    src = rng.normal(size=(30, 3))
    R0, s0, t0 = rot(0.3, -0.2, 0.5), 2.5, np.array([1.0, -2.0, 0.5])
    dst = s0 * src @ R0.T + t0
    s, R, t = umeyama(src, dst)
    assert abs(s - s0) < 1e-9 and np.allclose(R, R0) and np.allclose(t, t0)
    dst[[3, 7]] += 10.0
    s, R, t, inl = ransac_umeyama(src, dst, 0.05)
    assert abs(s - s0) < 1e-6 and inl.sum() == 28 and not inl[3] and not inl[7]

    # synthetic scene: plane z=5 seen by 4 cameras; network frame = SfM frame / 3, rotated and shifted
    K = np.array([[50.0, 0, 32], [0, 50.0, 24], [0, 0, 1]])
    names, sfm, extr, depth = [], {}, [], []
    Rn, sn, tn = rot(0.1, 0.4, -0.2), 1 / 3.0, np.array([0.2, 0.1, -0.3])  # SfM world -> network world
    for i in range(4):
        R_c, t_c = rot(0, 0.05 * i, 0), np.array([-0.3 * i, 0.1 * i, 0.0])
        names.append(f"img_{i:04d}.png")
        sfm[names[-1]] = (R_c, t_c)
        C = -R_c.T @ t_c
        v, u = np.mgrid[0:48, 0:64]
        rays = np.stack([(u - 32) / 50.0, (v - 24) / 50.0, np.ones_like(u, float)], -1) @ R_c  # world dirs
        lam = (5.0 - C[2]) / rays[..., 2]
        depth.append(((C + lam[..., None] * rays - C) @ R_c.T)[..., 2] * sn)  # network depth
        Rnc = R_c @ Rn.T  # network world -> camera rotation
        extr.append(np.hstack([Rnc, (sn * t_c - Rnc @ tn)[:, None]]))
    pred = {"depth": np.array(depth), "conf": np.full((4, 48, 64), 2.0) + rng.random((4, 48, 64)),
            "extrinsic": np.array(extr), "intrinsic": np.repeat(K[None], 4, 0),
            "rgb": np.zeros((4, 48, 64, 3), np.uint8)}
    for use_sfm in (True, False):
        vcfg = {**DEFAULTS, "use_sfm_poses": use_sfm, "conf_percentile": 0}
        pts, cols, info = to_points(pred, names, sfm, vcfg)
        assert info["align_rmse"] < 1e-6 and abs(info["align_scale"] - 3.0) < 1e-6, info
        assert len(pts) == len(cols) > 4000 and np.allclose(pts[:, 2], 5.0, atol=1e-6), use_sfm
    vcfg = {**DEFAULTS, "conf_percentile": 50}
    assert len(to_points(pred, names, sfm, vcfg)[0]) < 4 * 48 * 64 * 0.55
    p, c = voxel_down(rng.random((1000, 3)), np.zeros((1000, 3), np.uint8), 0.5)
    assert 0 < len(p) <= 27

    # gated model without a checkpoint and without access -> None, no crash (network only)
    with tempfile.TemporaryDirectory() as tmp:
        assert resolve_weights({**DEFAULTS, "model": "vggt_omega", "checkpoint": str(Path(tmp) / "x.pt")}) is None
        assert resolve_weights(DEFAULTS) == VGGT_REPO
    print("vggt_omega self-test OK")
# <<< END IMPLEMENTATION <<<

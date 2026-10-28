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
# Stage: dense (method patchmatchnet) - PatchmatchNet (Wang et al., CVPR 2021) learned multi-view depth maps.
# Goal: run the official PatchmatchNet checkpoint on CUDA per reference view (ref + num_views ranked sources),
#      keep pixels with photometric confidence >= conf_threshold and fuse them with COLMAP like every densifier.
# In:  shared dense/undistorted workspace (depth_fusion), FangjinhuaWang/PatchmatchNet@REPO_SHA with
#      checkpoints/params_000007.ckpt (eager model) or module_000007.pt (TorchScript fallback).
# Out: <run_dir>/dense/patchmatchnet/{ws/stereo/{depth_maps,normal_maps}, fused.ply, points.npy, colors.npy,
#      colmap_dense.log}; row method=patchmatchnet in <artifacts_dir>/dense_metrics.csv (t_depth_s, vram_peak_gb).
# YAML: dense.patchmatchnet.{max_size, num_views, conf_threshold, patchmatch_interval_scale, patchmatch_range,
#      patchmatch_iteration, patchmatch_num_sample, propagate_neighbors, evaluate_neighbors, checkpoint}.
# Rules: inputs as their eval.py / datasets/mvs.py (RGB in [0, 1], K scaled with the image, 4x4 world->camera
#      extrinsics, depth range of the reference); width/height are rounded to multiples of 8 here with K
#      adjusted so the depth map and its K match; no CUDA / no torch / download failure -> WARNING and None.

# >>> BEGIN IMPLEMENTATION >>>
"""PatchmatchNet depth maps (official weights) fused with COLMAP stereo_fusion."""
import logging
from pathlib import Path

import numpy as np

from src.dense import depth_fusion, third_party

log = logging.getLogger(__name__)

METHOD = "patchmatchnet"
REPO = ("FangjinhuaWang", "PatchmatchNet", "8dc6cb40bdb7053e856598b378425c76a9dcf5e0")
DEFAULTS = {
    "max_size": 1600,          # longest image side fed to the network (px)
    "num_views": 5,            # source views per reference
    "conf_threshold": 0.5,     # photometric confidence below this -> depth 0 (their photo_thres)
    "patchmatch_interval_scale": [0.005, 0.0125, 0.025],
    "patchmatch_range": [6, 4, 2],
    "patchmatch_iteration": [1, 2, 2],
    "patchmatch_num_sample": [8, 8, 16],
    "propagate_neighbors": [0, 8, 16],
    "evaluate_neighbors": [9, 9, 9],
    "checkpoint": "params",    # params (eager model + params_000007.ckpt) | module (TorchScript)
}

def load_model(repo: Path, p: dict, device: str):
    """(model, kind): the eager PatchmatchNet with params_000007.ckpt, else the TorchScript module_000007.pt."""

    import torch

    if p["checkpoint"] == "params":
        try:
            net_cls = third_party.import_attr(repo, "models.net", "PatchmatchNet")
            model = net_cls(patchmatch_interval_scale=p["patchmatch_interval_scale"],
                            propagation_range=p["patchmatch_range"], patchmatch_iteration=p["patchmatch_iteration"],
                            patchmatch_num_sample=p["patchmatch_num_sample"],
                            propagate_neighbors=p["propagate_neighbors"], evaluate_neighbors=p["evaluate_neighbors"])
            state = torch.load(repo / "checkpoints" / "params_000007.ckpt", map_location="cpu", weights_only=True)
            state = {k.removeprefix("module."): v for k, v in state["model"].items()}
            model.load_state_dict(state, strict=True)
            return model.to(device).eval(), "params"
        except Exception as exc:  # noqa: BLE001 - any import/load problem falls back to the scripted module
            log.warning("patchmatchnet: eager model failed (%s), using the TorchScript module", exc)
    model = torch.jit.load(str(repo / "checkpoints" / "module_000007.pt"), map_location=device)
    return model.eval(), "module"

def network_input(img: np.ndarray, K: np.ndarray, max_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Image resized to max_size (as their scale_to_max_dim) and to multiples of 8, with K scaled to match."""

    import cv2

    h, w = img.shape[:2]
    s = min(1.0, max_size / max(h, w))
    nw, nh = max(8, int(round(w * s / 8)) * 8), max(8, int(round(h * s / 8)) * 8)
    if (nw, nh) != (w, h):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    return np.ascontiguousarray(img), depth_fusion.scale_K(K, nw / w, nh / h)

def extrinsic(view) -> np.ndarray:

    E = np.eye(4)
    E[:3, :3], E[:3, 3] = view.R, view.t
    return E

def infer(model, imgs: list[np.ndarray], Ks: list[np.ndarray], Es: list[np.ndarray], dmin: float, dmax: float,
          device: str) -> tuple[np.ndarray, np.ndarray]:
    """One PatchmatchNet forward (reference first): depth (H, W) and photometric confidence (H, W)."""

    import torch

    images = [torch.from_numpy(i.transpose(2, 0, 1)).float()[None].to(device) for i in imgs]
    K = torch.from_numpy(np.stack(Ks)).float()[None].to(device)
    E = torch.from_numpy(np.stack(Es)).float()[None].to(device)
    lo = torch.tensor([[dmin]], dtype=torch.float32, device=device)
    hi = torch.tensor([[dmax]], dtype=torch.float32, device=device)
    with torch.no_grad():
        depth, conf, _ = model(images, K, E, lo, hi)
    return depth[0, 0].float().cpu().numpy(), conf[0].float().cpu().numpy()

def run(cfg: dict, run_dir: str, artifacts_dir: str) -> Path | None:

    if not depth_fusion.enabled(cfg, METHOD):
        return None
    p = depth_fusion.method_cfg(cfg, METHOD, DEFAULTS)
    row = depth_fusion.base_row(cfg, METHOD)
    row["backend"] = "pytorch"
    if (fused := depth_fusion.reuse(cfg, run_dir, artifacts_dir, METHOD, row)) is not None:
        return fused
    try:
        import torch
    except ImportError:
        log.warning("dense %s skipped: torch not installed", METHOD)
        return None
    if not torch.cuda.is_available():
        log.warning("dense %s skipped: CUDA not available", METHOD)
        return None
    ws_u = depth_fusion.undistorted_workspace(cfg, run_dir)
    if ws_u is None:
        return None
    try:
        repo = third_party.fetch_repo(*REPO)
        model, kind = load_model(repo, p, "cuda")
    except Exception as exc:  # noqa: BLE001 - network / checkpoint problems must not stop the pipeline
        log.warning("dense %s skipped: could not load PatchmatchNet (%s)", METHOD, exc)
        return None
    row["backend"] = f"pytorch-{kind}"

    views = depth_fusion.load_views(ws_u, int(p["num_views"]))
    refs = depth_fusion.references(cfg, ws_u)
    ws = depth_fusion.method_workspace(run_dir, METHOD, ws_u)
    for old in (ws / "stereo").glob("*_maps/*.bin"):
        old.unlink()
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def view_input(name):
        if name not in cache:
            cache[name] = network_input(depth_fusion.read_image(ws_u, name), views[name].K, int(p["max_size"]))
        return cache[name]

    torch.cuda.reset_peak_memory_stats()
    done, kept = [], []
    with depth_fusion.Timer() as t:
        for i, name in enumerate(refs):
            v = views[name]
            for n in set(cache) - {name, *v.sources}:  # keep only the images this reference reuses
                del cache[n]
            if not v.sources or v.depth_max <= v.depth_min or v.depth_min <= 0:
                log.warning("patchmatchnet: skipping %s (no sources or no sparse depth range)", name)
                continue
            ids = [name] + v.sources
            inputs = [view_input(n) for n in ids]
            depth, conf = infer(model, [x[0] for x in inputs], [x[1] for x in inputs],
                                [extrinsic(views[n]) for n in ids], v.depth_min, v.depth_max, "cuda")
            mask = (conf >= float(p["conf_threshold"])) & (depth > v.depth_min) & (depth < v.depth_max)
            depth = np.where(mask, depth, 0.0).astype(np.float32)
            depth_fusion.save_depth(ws, name, depth, inputs[0][1])
            done.append(name)
            kept.append(float(mask.mean()))
            log.info("patchmatchnet %d/%d %s: %dx%d, kept %.1f%%", i + 1, len(refs), name, depth.shape[1],
                     depth.shape[0], 100 * kept[-1])
    row["t_depth_s"] = round(t.s, 2)
    row["vram_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    del model
    torch.cuda.empty_cache()
    if kept:
        log.info("patchmatchnet: %d depth maps, mean confident fraction %.1f%%", len(done), 100 * np.mean(kept))
    return depth_fusion.fuse_and_export(cfg, run_dir, artifacts_dir, METHOD, ws, done, row)

# <<< END IMPLEMENTATION <<<

if __name__ == "__main__":
    import tempfile

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    K0 = np.array([[500.0, 0, 330], [0, 500.0, 245], [0, 0, 1]])
    img, K1 = network_input(np.zeros((490, 661, 3), np.float32), K0, 1600)
    assert img.shape == (488, 664, 3) and np.isclose(K1[0, 0], 500 * 664 / 661), (img.shape, K1)
    img, K1 = network_input(np.zeros((900, 1600, 3), np.float32), K0, 800)
    assert img.shape == (448, 800, 3) and np.isclose(K1[1, 2], 245 * 448 / 900), img.shape
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and torch.cuda.is_available():
        # fronto-parallel textured plane at z=2 seen by a ref and 2 sideways-shifted sources (tiny input)
        repo = third_party.fetch_repo(*REPO)
        model, kind = load_model(repo, DEFAULTS, "cuda")
        h, w, z = 128, 160, 2.0
        K = np.array([[150.0, 0, w / 2], [0, 150.0, h / 2], [0, 0, 1]])
        rng = np.random.default_rng(0)
        tex = rng.random((h // 4, (w + 80) // 4)).repeat(4, 0).repeat(4, 1)
        imgs, Es = [], []
        for dx in (0.0, 0.1, -0.1):
            shift = int(round(150 * dx / z))
            g = tex[:, 40 + shift:40 + shift + w]
            imgs.append(np.repeat(g[..., None], 3, 2).astype(np.float32))
            E = np.eye(4)
            E[0, 3] = -dx
            Es.append(E)
        depth, conf = infer(model, imgs, [K] * 3, Es, 1.0, 4.0, "cuda")
        assert depth.shape == (h, w) and conf.shape == (h, w), (depth.shape, conf.shape)
        print(f"{kind}: median depth {np.median(depth):.3f} (true {z}), median conf {np.median(conf):.2f}")
    with tempfile.TemporaryDirectory() as tmp:
        cfg = {"dense": {"enabled": True, "methods": ["patch_match"]}}
        assert run(cfg, tmp, tmp) is None
    print("patchmatchnet self-test OK")

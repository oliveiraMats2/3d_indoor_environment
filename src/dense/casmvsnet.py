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
# Stage: dense (method casmvsnet) - CasMVSNet (Gu et al. CVPR 2020) learned cascade MVS on the GPU.
# Goal: per reference view, infer a depth map with the official CasMVSNet weights (DTU, 48/32/8 planes,
#      intervals 4/2/1) from N ranked source views, keep the confident pixels and fuse with COLMAP like
#      every other depth-map densifier.
# In:  <run_dir>/dense/undistorted (shared PINHOLE workspace), its sparse model (poses, depth range,
#      sources), alibaba/cascade-stereo@2ae7da65 (CasMVSNet/) + casmvsnet.ckpt (fetched to third_party/).
# Out: <run_dir>/dense/casmvsnet/{ws/stereo/depth_maps, fused.ply, points.npy, colors.npy, colmap_dense.log};
#      row method=casmvsnet in <artifacts_dir>/dense_metrics.csv (t_depth_s, vram_peak_gb).
# YAML: dense.casmvsnet.{max_size, num_views, num_depth, conf_threshold, ndepths, depth_interval_ratio,
#      cr_base_chs}; dense.{enabled, methods, rerun, ref_step, fusion.*}.
# Rules: no CUDA / torch / download failure -> WARNING and return None; images in [0, 1] (no mean/std, as
#      the official general_eval dataset), resized so the long side <= max_size and both sides are multiples
#      of 32 (K scaled); stage k uses K / 2^(3-k); depth range = the view's sparse range split in num_depth;
#      pixels with photometric confidence < conf_threshold are set to 0; third_party code is never edited.

# >>> BEGIN IMPLEMENTATION >>>
"""CasMVSNet depth maps (official weights) fused with COLMAP stereo_fusion."""
import logging
from pathlib import Path

import numpy as np

from src.dense import depth_fusion as df

log = logging.getLogger(__name__)

METHOD = "casmvsnet"
REPO = ("alibaba", "cascade-stereo", "2ae7da65515c82025383a0fbee8d6fe063dc7101")
WEIGHTS = ("https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/cascade-stereo/CasMVSNet/"
           "48_32_8-4-2-1_dlossw-0.5-1.0-2.0/casmvsnet.ckpt")
WEIGHTS_SHA256 = "ac343e1a6fcec1f6f1cf13a8fd04a9831c0ce09854b73e396dc1d82926eb35dd"
BASE = 32
DEFAULTS = {"max_size": 1152, "num_views": 5, "num_depth": 192, "conf_threshold": 0.9,
            "ndepths": [48, 32, 8], "depth_interval_ratio": [4, 2, 1], "cr_base_chs": [8, 8, 8]}

def input_size(width: int, height: int, max_size: int, base: int = BASE) -> tuple[int, int]:
    """Network input (w, h): long side <= max_size, both floored to a multiple of base (scale_mvs_input)."""

    s = min(1.0, max_size / max(width, height))
    return max(base, int(width * s) // base * base), max(base, int(height * s) // base * base)

def stage_projections(Ks: list[np.ndarray], Rts: list[np.ndarray], n_stages: int = 3) -> dict[str, np.ndarray]:
    """{stageK: (N, 2, 4, 4)} as the official dataset: [0] = world->camera 4x4, [1][:3, :3] = K at the
    stage resolution (stage 1 = 1/4, last stage = full input resolution)."""

    out = {}
    for s in range(n_stages):
        f = 2.0 ** (s - n_stages + 1)
        mats = np.zeros((len(Ks), 2, 4, 4), np.float32)
        for i, (K, Rt) in enumerate(zip(Ks, Rts)):
            mats[i, 0] = Rt
            mats[i, 1, :3, :3] = df.scale_K(K, f, f)
            mats[i, 1, 3, 3] = 1.0
        out[f"stage{s + 1}"] = mats
    return out

def load_model(p: dict, device):
    """CascadeMVSNet with the official test settings and the official checkpoint (eval mode on device)."""

    import torch

    from src.dense import third_party

    repo = third_party.fetch_repo(*REPO) / "CasMVSNet"
    ckpt = third_party.fetch_file(WEIGHTS, "casmvsnet.ckpt", WEIGHTS_SHA256)
    CascadeMVSNet = third_party.import_attr(repo, "models.cas_mvsnet", "CascadeMVSNet")
    model = CascadeMVSNet(refine=False, ndepths=[int(d) for d in p["ndepths"]],
                          depth_interals_ratio=[float(r) for r in p["depth_interval_ratio"]], share_cr=False,
                          cr_base_chs=[int(c) for c in p["cr_base_chs"]], grad_method="detach")
    state = torch.load(ckpt, map_location="cpu", weights_only=True)["model"]
    model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()}, strict=True)
    return model.to(device).eval()

def infer(model, imgs: np.ndarray, projs: dict, depth_values: np.ndarray, device) -> tuple[np.ndarray, np.ndarray]:
    """One reference: imgs (N, 3, H, W) in [0, 1], ref first -> (depth, photometric confidence), both (H, W)."""

    import torch

    with torch.no_grad():
        out = model(torch.from_numpy(imgs)[None].to(device),
                    {k: torch.from_numpy(v)[None].to(device) for k, v in projs.items()},
                    torch.from_numpy(depth_values.astype(np.float32))[None].to(device))
    return out["depth"][0].float().cpu().numpy(), out["photometric_confidence"][0].float().cpu().numpy()

def prepare(ws, views: dict, ref: str, p: dict, cache: dict) -> tuple | None:
    """Network inputs of one reference (None when it has no sparse depth range or no source view)."""

    import cv2

    v = views[ref]
    srcs = [s for s in v.sources if s in views][:int(p["num_views"]) - 1]
    if v.depth_max <= v.depth_min or v.depth_min <= 0 or not srcs:
        return None
    srcs += [srcs[0]] * (int(p["num_views"]) - 1 - len(srcs))
    w, h = input_size(v.width, v.height, int(p["max_size"]))
    imgs, Ks, Rts = [], [], []
    for name in [ref] + srcs:
        u = views[name]
        if (name, w, h) not in cache:
            cache.clear() if len(cache) > 64 else None
            cache[(name, w, h)] = cv2.resize(df.read_image(ws, name), (w, h), interpolation=cv2.INTER_AREA
                                             ).transpose(2, 0, 1).copy()
        imgs.append(cache[(name, w, h)])
        Ks.append(df.scale_K(u.K, w / u.width, h / u.height))
        Rt = np.eye(4)
        Rt[:3, :3], Rt[:3, 3] = u.R, u.t
        Rts.append(Rt)
    n = int(p["num_depth"])
    depth_values = v.depth_min + np.arange(n) * (v.depth_max - v.depth_min) / n
    return np.stack(imgs).astype(np.float32), stage_projections(Ks, Rts, len(p["ndepths"])), depth_values, Ks[0]

def run(cfg: dict, run_dir: str, artifacts_dir: str) -> Path | None:

    if not df.enabled(cfg, METHOD):
        return None
    p = df.method_cfg(cfg, METHOD, DEFAULTS)
    row = df.base_row(cfg, METHOD)
    row["backend"] = "torch"
    if (fused := df.reuse(cfg, run_dir, artifacts_dir, METHOD, row)) is not None:
        return fused
    try:
        import torch
    except ImportError:
        log.warning("dense casmvsnet skipped: torch not installed")
        return None
    if not torch.cuda.is_available():
        log.warning("dense casmvsnet skipped: CUDA not available")
        return None
    undist = df.undistorted_workspace(cfg, run_dir)
    if undist is None:
        return None
    device = torch.device(f"cuda:{str(cfg['dense'].get('gpu_index', 0)).split(',')[0]}")
    try:
        model = load_model(p, device)
    except Exception as exc:  # download, import or checkpoint mismatch
        log.warning("dense casmvsnet skipped: model not available (%s)", exc)
        return None

    views = df.load_views(undist, max(df.N_SOURCES, int(p["num_views"]) - 1))
    refs = [r for r in df.references(cfg, undist) if r in views]
    ws = df.method_workspace(run_dir, METHOD, undist)
    for old in (ws / "stereo" / "depth_maps").glob("*.bin"):
        old.unlink()
    torch.cuda.reset_peak_memory_stats(device)
    cache, kept = {}, []
    with df.Timer() as t:
        for i, ref in enumerate(refs):
            inputs = prepare(undist, views, ref, p, cache)
            if inputs is None:
                log.warning("dense casmvsnet: %s has no depth range or sources, skipped", ref)
                continue
            imgs, projs, depth_values, K = inputs
            try:
                depth, conf = infer(model, imgs, projs, depth_values, device)
            except torch.cuda.OutOfMemoryError:
                log.warning("dense casmvsnet: out of GPU memory at %s (lower dense.casmvsnet.max_size)", ref)
                torch.cuda.empty_cache()
                break
            depth[(conf < float(p["conf_threshold"])) | ~np.isfinite(depth)] = 0
            depth[(depth < depth_values[0]) | (depth > depth_values[-1])] = 0
            df.save_depth(ws, ref, depth, K)
            kept.append(float((depth > 0).mean()))
            if i % 10 == 0:
                log.info("casmvsnet %d/%d %s: %.0f%% confident", i + 1, len(refs), ref, 100 * kept[-1])
    row["t_depth_s"] = round(t.s, 2)
    row["vram_peak_gb"] = round(torch.cuda.max_memory_allocated(device) / 1e9, 2)
    row["n_excluded"] = len(views) - len(refs)
    log.info("casmvsnet: %d depth maps in %.1f s, mean confident fraction %.2f", len(kept), t.s,
             float(np.mean(kept)) if kept else 0.0)
    del model
    torch.cuda.empty_cache()
    return df.fuse_and_export(cfg, run_dir, artifacts_dir, METHOD, ws, refs, row)

# <<< END IMPLEMENTATION <<<

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    assert input_size(1600, 900, 1152) == (1152, 640)
    assert input_size(100, 70, 1152) == (96, 64)
    K = np.array([[400.0, 0, 200], [0, 400, 120], [0, 0, 1]])
    pr = stage_projections([K, K], [np.eye(4), np.eye(4)])
    assert pr["stage1"].shape == (2, 2, 4, 4) and np.isclose(pr["stage1"][0, 1, 0, 0], 100)
    assert np.isclose(pr["stage3"][0, 1, 0, 0], 400) and np.isclose(pr["stage2"][1, 1, 1, 2], 60)

    import torch
    if torch.cuda.is_available():
        # tiny random input through the official network: shapes and confidence range
        dev = torch.device("cuda")
        model = load_model(DEFAULTS, dev)
        w, h, n = 160, 128, 3
        Kt = np.array([[150.0, 0, w / 2], [0, 150, h / 2], [0, 0, 1]])
        Rts = []
        for k in range(n):
            Rt = np.eye(4)
            Rt[0, 3] = -0.1 * k
            Rts.append(Rt)
        imgs = np.random.default_rng(0).random((n, 3, h, w)).astype(np.float32)
        dv = 1.0 + np.arange(192) * (4.0 / 192)
        depth, conf = infer(model, imgs, stage_projections([Kt] * n, Rts), dv, dev)
        assert depth.shape == (h, w) and conf.shape == (h, w), (depth.shape, conf.shape)
        assert np.all(np.isfinite(depth)) and 0.9 <= depth.min() and depth.max() <= 5.1
        assert 0 <= conf.min() and conf.max() <= 1.0 + 1e-4
        print("depth", float(depth.min()), float(depth.max()), "conf mean", float(conf.mean()))
    print("casmvsnet self-test OK")

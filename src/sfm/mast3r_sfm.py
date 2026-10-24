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
# Stage : sfm, technique mast3r_sfm (Duisterhof et al. 2025, naver/mast3r sparse global alignment).
# Goal  : cameras of up to sfm.mast3r_sfm.max_frames frames from MASt3R pair predictions (pairs chosen by
#         retrieval) + sparse global alignment on the GPU, written as a COLMAP model like colmap.py.
# In    : frames cfg["extraction"]["output_dir"]/img_*; cfg["sfm"] and cfg["sfm"]["mast3r_sfm"].{repo_dir,
#         weights, retrieval_url, weights_dir, image_size, max_frames, scene_graph, retrieval_na, retrieval_k,
#         window, lr1, niter1, lr2, niter2, optim_level, matching_conf_thr, shared_intrinsics, subsample, ba,
#         refine_intrinsics}.
# Out   : gpu.py contract (images/ only the used frames, database.db, sparse/0, ...) + <ws>/mast3r_cache/
#         (pair predictions, wiped on every rebuild); sfm_metrics.csv with t_load_s, t_retrieval_s, t_mast3r_s,
#         t_feature_s, t_match_s, t_triangulate_s, t_ba_s, vram_peak_gb.
# Rules : mast3r is not a pip package: repo_dir is the official clone (git clone --recursive, README) put on
#         sys.path. Official weights only: naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric (HF hub)
#         and the retrieval .pth + codebook .pkl downloaded once from retrieval_url into weights_dir
#         (CC BY-NC-SA 4.0). Retrieval needs asmk; scene_graph swin works without it. Intrinsics at 512 px
#         are mapped back to the original frame (resize + centre crop of dust3r load_images). 3D points: SIFT
#         tracks triangulated with the MASt3R poses; ba true = pycolmap bundle adjustment on them.

# >>> BEGIN IMPLEMENTATION >>>
"""MASt3R-SfM sfm technique: retrieval pairs, sparse global alignment, SIFT tracks triangulated with its cameras."""
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from src.sfm import gpu

log = logging.getLogger(__name__)

TECHNIQUE = "mast3r_sfm"
PATCH = 16
RETRIEVAL_FILES = ("{w}_retrieval_trainingfree.pth", "{w}_retrieval_codebook.pkl")

def crop_box(width: int, height: int, size: int) -> tuple[float, float, float, float]:
    """(scale_x, scale_y, x0, y0) of dust3r load_images(size != 224): long side resized to size, then centre
    crop to multiples of PATCH. An original pixel X = (u + x0) * scale_x for a pixel u of the network image."""

    s = size / max(width, height)
    w, h = int(round(width * s)), int(round(height * s))
    cx, cy = w // 2, h // 2
    halfw, halfh = ((2 * cx) // PATCH) * PATCH / 2, ((2 * cy) // PATCH) * PATCH / 2
    if w == h:
        halfh = 3 * halfw / 4

    return width / w, height / h, cx - halfw, cy - halfh

def to_original(K: np.ndarray, box: tuple) -> tuple:
    """(fx, fy, cx, cy) at the original frame size from a 3x3 at the network size."""

    sx, sy, x0, y0 = box

    return float(K[0, 0] * sx), float(K[1, 1] * sy), float((K[0, 2] + x0) * sx), float((K[1, 2] + y0) * sy)

def scene_graph(mcfg: dict, n: int) -> str:
    """dust3r scene graph string: retrieval-Na-k or swin-<window>-noncyclic, clipped to n frames."""

    if mcfg["scene_graph"] == "retrieval":
        return f"retrieval-{min(n, int(mcfg['retrieval_na']))}-{min(n - 1, int(mcfg['retrieval_k']))}"

    return f"swin-{min(n - 1, int(mcfg['window']))}-noncyclic"

def weights_dir(mcfg: dict) -> Path:

    return Path(os.path.expanduser(mcfg["weights_dir"]))

def retrieval_paths(mcfg: dict) -> tuple[Path, Path]:

    name = mcfg["weights"].split("/")[-1]

    return tuple(weights_dir(mcfg) / f.format(w=name) for f in RETRIEVAL_FILES)

def download_retrieval(mcfg: dict) -> Path | None:
    """Official retrieval .pth and codebook in weights_dir (downloaded once with torch.hub); None on failure."""

    import torch

    paths = retrieval_paths(mcfg)
    for path in paths:
        if path.is_file():
            continue
        url = mcfg["retrieval_url"].rstrip("/") + "/" + path.name
        log.info("sfm: downloading %s -> %s", url, path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            torch.hub.download_url_to_file(url, str(path), progress=False)
        except OSError as exc:
            log.warning("sfm: download of %s failed (%s)", url, exc)
            return None

    return paths[0]

def _child(cfg: dict, ws: str, frames: list[str], state_path: str) -> None:
    """Spawned child: MASt3R pairs (retrieval) -> sparse GA -> poses/intrinsics -> SIFT + match ->
    triangulation (+ BA) -> sparse/0."""

    ws = Path(ws)
    gpu.redirect_output(ws)
    scfg, mcfg = cfg["sfm"], cfg["sfm"]["mast3r_sfm"]
    sys.path.insert(0, str(Path(os.path.expanduser(mcfg["repo_dir"]))))
    import torch

    import mast3r.utils.path_to_dust3r  # noqa: F401 - puts dust3r/croco on sys.path
    from dust3r.utils.image import load_images
    from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
    from mast3r.image_pairs import make_pairs
    from mast3r.model import AsymmetricMASt3R

    import scipy.cluster.hierarchy as sch
    import scipy.spatial.distance

    if not hasattr(sch, "distance"):  # sparse_ga calls sch.distance.squareform, an alias newer scipy dropped
        sch.distance = scipy.spatial.distance
    seed = int(cfg["run"]["seed"])
    np.random.seed(seed)  # make_pairs_fps draws with np.random
    torch.manual_seed(seed)
    state = gpu.State(state_path)
    paths = list(frames)
    n = len(paths)
    graph = scene_graph(mcfg, n)
    print(f"mast3r_sfm: {n} frames, scene graph {graph}, weights {mcfg['weights']}", flush=True)

    def load():
        model = AsymmetricMASt3R.from_pretrained(mcfg["weights"]).to("cuda").eval()
        return model, load_images(paths, size=int(mcfg["image_size"]), verbose=False)

    model, imgs = state.step("load", load)

    def retrieval():
        from mast3r.retrieval.processor import Retriever

        load_fn = torch.load
        torch.load = lambda *a, **k: load_fn(*a, **{**k, "weights_only": False})  # ckpt holds argparse.Namespace
        try:
            retriever = Retriever(str(retrieval_paths(mcfg)[0]), backbone=model, device="cuda")
            with torch.no_grad():
                sim = retriever(paths)
        finally:
            torch.load = load_fn
        del retriever
        torch.cuda.empty_cache()
        return sim

    sim = state.step("retrieval", retrieval) if mcfg["scene_graph"] == "retrieval" else None
    pairs = make_pairs(imgs, scene_graph=graph, prefilter=None, symmetrize=True, sim_mat=sim)
    cache = ws / "mast3r_cache"
    shutil.rmtree(cache, ignore_errors=True)  # keyed by path md5, not content
    cache.mkdir()
    niter2 = 0 if mcfg["optim_level"] == "coarse" else int(mcfg["niter2"])
    scene = state.step("mast3r", sparse_global_alignment, paths, pairs, str(cache), model,
                       lr1=float(mcfg["lr1"]), niter1=int(mcfg["niter1"]), lr2=float(mcfg["lr2"]), niter2=niter2,
                       device="cuda", opt_depth="depth" in mcfg["optim_level"],
                       shared_intrinsics=bool(mcfg["shared_intrinsics"]),
                       matching_conf_thr=float(mcfg["matching_conf_thr"]), subsample=int(mcfg["subsample"]))
    cam2w = scene.get_im_poses().detach().float().cpu().numpy()
    Ks = scene.intrinsics.detach().float().cpu().numpy() if torch.is_tensor(scene.intrinsics) \
        else np.stack([k.detach().float().cpu().numpy() for k in scene.intrinsics])
    del scene, model
    torch.cuda.empty_cache()
    shutil.rmtree(cache, ignore_errors=True)

    size = gpu.image_size(Path(paths[0]))
    box = crop_box(*size, int(mcfg["image_size"]))
    names = [Path(p).name for p in paths]
    poses = {name: np.linalg.inv(cam2w[i])[:3] for i, name in enumerate(names)}
    intrinsics = {name: to_original(Ks[i], box) for i, name in enumerate(names)}
    gpu.extract_and_match(scfg, ws, seed, state)
    rec = gpu.posed_reconstruction(ws, poses, intrinsics, shared=bool(scfg["single_camera"]))
    gpu.triangulate(ws, rec, state, ba=bool(mcfg["ba"]), refine_intrinsics=bool(mcfg["refine_intrinsics"]))
    state.done()

def run(cfg: dict, run_dir: str, artifacts_dir: str) -> Path | None:
    """Same contract as colmap.run(): <run_dir>/exports/colmap/sparse/0 or None when skipped/failed."""

    mcfg = cfg["sfm"]["mast3r_sfm"]
    ws, frames, model, reused = gpu.prepare(cfg, run_dir, TECHNIQUE, mcfg["max_frames"])
    if reused:
        return gpu.reuse(cfg, ws, model, TECHNIQUE)

    if len(frames) < 2:
        log.warning("sfm: %d frames in %s, need >= 2: skipped", len(frames), cfg["extraction"]["output_dir"])
        return None

    if not gpu.cuda_ok():
        return None

    repo = Path(os.path.expanduser(mcfg["repo_dir"]))
    if not (repo / "mast3r" / "model.py").exists() or not (repo / "dust3r" / "dust3r").is_dir():
        log.warning("sfm: sfm.mast3r_sfm.repo_dir %s is not a recursive clone of naver/mast3r (README): sfm skipped", repo)
        return None

    if mcfg["scene_graph"] == "retrieval" and download_retrieval(mcfg) is None:
        log.warning("sfm: MASt3R retrieval weights missing: sfm skipped (or set sfm.mast3r_sfm.scene_graph: swin)")
        return None

    log.info("sfm (mast3r_sfm): %d frames, %s, optim %s, ba=%s", len(frames), scene_graph(mcfg, len(frames)),
             mcfg["optim_level"], mcfg["ba"])
    t0 = time.perf_counter()
    exitcode, state = gpu.spawn(_child, (cfg, str(ws), [str(f) for f in frames]), ws)

    return gpu.finish(cfg, ws, TECHNIQUE, len(frames), artifacts_dir, exitcode, state, t0)

DEFAULTS = {"repo_dir": "~/mast3r", "weights": "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
            "retrieval_url": "https://download.europe.naverlabs.com/ComputerVision/MASt3R",
            "weights_dir": "~/.cache/mast3r", "image_size": 512, "max_frames": 200, "scene_graph": "retrieval",
            "retrieval_na": 20, "retrieval_k": 10, "window": 10, "lr1": 0.07, "niter1": 300, "lr2": 0.01,
            "niter2": 300, "optim_level": "refine+depth", "matching_conf_thr": 0.0, "shared_intrinsics": True,
            "subsample": 8, "ba": True, "refine_intrinsics": True}

if __name__ == "__main__":
    assert crop_box(1080, 1920, 512) == (3.75, 3.75, 0.0, 0.0)  # 288x512, already multiples of 16
    sx, sy, x0, y0 = crop_box(1920, 1080, 512)
    assert (sx, sy, x0, y0) == (3.75, 3.75, 0.0, 0.0)
    sx, sy, x0, y0 = crop_box(1000, 1000, 512)  # square -> 4:3 crop
    assert x0 == 0 and y0 == 64 and abs(sx - 1000 / 512) < 1e-12
    fx, fy, cx, cy = to_original(np.array([[400.0, 0, 144], [0, 400.0, 256], [0, 0, 1]]), crop_box(1080, 1920, 512))
    assert (fx, fy, cx, cy) == (1500.0, 1500.0, 540.0, 960.0)
    assert scene_graph(DEFAULTS, 300) == "retrieval-20-10" and scene_graph(DEFAULTS, 5) == "retrieval-5-4"
    assert scene_graph({**DEFAULTS, "scene_graph": "swin"}, 30) == "swin-10-noncyclic"
    assert retrieval_paths(DEFAULTS)[1].name == "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl"
    print("OK")
# <<< END IMPLEMENTATION <<<

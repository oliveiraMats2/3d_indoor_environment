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
# Stage : sfm, technique vggt (Wang et al. 2025, facebookresearch/vggt): feed-forward "SfM in one pass".
# Goal  : cameras of up to sfm.vggt.max_frames frames from one VGGT forward pass on the GPU, written as a
#         COLMAP model in the same workspace as colmap.py.
# In    : frames cfg["extraction"]["output_dir"]/img_*; cfg["sfm"] (matcher, use_gpu, camera_model, ...) and
#         cfg["sfm"]["vggt"].{weights, image_size, max_frames, dtype, matcher, ba, refine_intrinsics,
#         min_points_per_image}.
# Out   : gpu.py contract: <run_dir>/exports/colmap/{images/ (only the used frames), database.db, sparse/0,
#         sparse.ply, project.ini, colmap.log}; sfm_metrics.csv with t_load_s, t_vggt_s, t_feature_s,
#         t_match_s, t_triangulate_s, t_ba_s, vram_peak_gb.
# Rules : official weights only (facebook/VGGT-1B, CC BY-NC 4.0, from the Hugging Face hub). Frames are resized
#         keeping the aspect ratio (long side image_size, both sides multiples of 14, no crop or pad); only the
#         aggregator + camera head run (no depth/point heads). Intrinsics are rescaled to the original frame and
#         the principal point set to the frame centre (VGGT assumes it). 3D points: SIFT tracks triangulated
#         with the VGGT poses; ba true = pycolmap bundle adjustment on them (no VGGSfM tracker / DINOv2 hub).

# >>> BEGIN IMPLEMENTATION >>>
"""VGGT sfm technique: one forward pass for the cameras, SIFT tracks triangulated with them, optional BA."""
import logging
import time
from pathlib import Path

import numpy as np

from src.sfm import gpu

log = logging.getLogger(__name__)

TECHNIQUE = "vggt"
PATCH = 14  # ViT patch size: input sides must be multiples of it

def input_size(width: int, height: int, long_side: int) -> tuple[int, int]:
    """(w, h) for VGGT: long side = long_side, short side rounded to a multiple of PATCH (aspect kept)."""

    scale = long_side / max(width, height)
    w = max(PATCH, round(width * scale / PATCH) * PATCH)
    h = max(PATCH, round(height * scale / PATCH) * PATCH)

    return (long_side, h) if width >= height else (w, long_side)

def to_original(intrinsic: np.ndarray, size_in: tuple[int, int], size_orig: tuple[int, int]) -> tuple:
    """(fx, fy, cx, cy) at the original frame size from a VGGT 3x3 at size_in; cx, cy = frame centre."""

    sx, sy = size_orig[0] / size_in[0], size_orig[1] / size_in[1]

    return float(intrinsic[0, 0] * sx), float(intrinsic[1, 1] * sy), size_orig[0] / 2.0, size_orig[1] / 2.0

def load_images(paths: list[Path], size: tuple[int, int]):
    """(N, 3, h, w) float tensor in [0, 1], bicubic resize, RGB."""

    import torch
    from PIL import Image

    out = []
    for p in paths:
        img = Image.open(p).convert("RGB").resize(size, Image.Resampling.BICUBIC)
        out.append(torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255.0)

    return torch.stack(out)

def _child(cfg: dict, ws: str, frames: list[str], state_path: str) -> None:
    """Spawned child: VGGT forward -> poses/intrinsics -> SIFT + match -> triangulation (+ BA) -> sparse/0."""

    ws = Path(ws)
    gpu.redirect_output(ws)
    import torch
    from vggt.models.vggt import VGGT
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    scfg, vcfg = cfg["sfm"], cfg["sfm"]["vggt"]
    seed = int(cfg["run"]["seed"])
    torch.manual_seed(seed)
    state = gpu.State(state_path)
    frames = [Path(f) for f in frames]
    size_orig = gpu.image_size(frames[0])
    size_in = input_size(*size_orig, int(vcfg["image_size"]))
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[vcfg["dtype"]] \
        if vcfg["dtype"] != "auto" else (torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16)
    print(f"vggt: {len(frames)} frames {size_orig} -> {size_in}, weights {vcfg['weights']}, dtype {dtype}", flush=True)

    def load():
        model = VGGT.from_pretrained(vcfg["weights"])
        model.depth_head = model.point_head = model.track_head = None  # cameras only: less VRAM
        return model.eval().to("cuda"), load_images(frames, size_in).to("cuda")

    model, images = state.step("load", load)

    def forward():
        with torch.no_grad():
            with torch.autocast("cuda", dtype=dtype):
                tokens, _ = model.aggregator(images[None])
            pose_enc = model.camera_head(tokens)[-1]
            extr, intr = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        return extr[0].float().cpu().numpy(), intr[0].float().cpu().numpy()

    extrinsic, intrinsic = state.step("vggt", forward)
    del model, images
    torch.cuda.empty_cache()

    poses = {f.name: extrinsic[i] for i, f in enumerate(frames)}
    intrinsics = {f.name: to_original(intrinsic[i], size_in, size_orig) for i, f in enumerate(frames)}
    gpu.extract_and_match(scfg, ws, seed, state, matcher=vcfg["matcher"])
    rec = gpu.posed_reconstruction(ws, poses, intrinsics, shared=bool(scfg["single_camera"]))
    gpu.triangulate(ws, rec, state, ba=bool(vcfg["ba"]), refine_intrinsics=bool(vcfg["refine_intrinsics"]),
                    min_points=int(vcfg["min_points_per_image"]))
    state.done()

def run(cfg: dict, run_dir: str, artifacts_dir: str) -> Path | None:
    """Same contract as colmap.run(): <run_dir>/exports/colmap/sparse/0 or None when skipped/failed."""

    vcfg = cfg["sfm"]["vggt"]
    ws, frames, model, reused = gpu.prepare(cfg, run_dir, TECHNIQUE, vcfg["max_frames"])
    if reused:
        return gpu.reuse(cfg, ws, model, TECHNIQUE)

    if len(frames) < 2:
        log.warning("sfm: %d frames in %s, need >= 2: skipped", len(frames), cfg["extraction"]["output_dir"])
        return None

    if not gpu.cuda_ok():
        return None

    try:
        import vggt  # noqa: F401
    except ImportError as exc:
        log.warning("sfm: vggt not importable (%s; install requirements.txt): sfm skipped", exc)
        return None

    log.info("sfm (vggt): %d frames, weights %s, ba=%s", len(frames), vcfg["weights"], vcfg["ba"])
    t0 = time.perf_counter()
    exitcode, state = gpu.spawn(_child, (cfg, str(ws), [str(f) for f in frames]), ws)

    return gpu.finish(cfg, ws, TECHNIQUE, len(frames), artifacts_dir, exitcode, state, t0)

DEFAULTS = {"weights": "facebook/VGGT-1B", "image_size": 518, "max_frames": 80, "dtype": "auto",
            "matcher": "exhaustive", "ba": True, "refine_intrinsics": True, "min_points_per_image": 50}

if __name__ == "__main__":
    assert input_size(1080, 1920, 518) == (294, 518)
    assert input_size(1920, 1080, 518) == (518, 294)
    assert input_size(500, 500, 518) == (518, 518)
    fx, fy, cx, cy = to_original(np.array([[300.0, 0, 147], [0, 300.0, 259], [0, 0, 1]]), (294, 518), (1080, 1920))
    assert abs(fx - 300 * 1080 / 294) < 1e-9 and abs(fy - 300 * 1920 / 518) < 1e-9 and (cx, cy) == (540, 960)
    print("OK")
# <<< END IMPLEMENTATION <<<

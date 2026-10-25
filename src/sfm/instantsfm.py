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
# Stage : sfm, technique instantsfm (Zhong et al. 2025, cre185/InstantSfM): classic global SfM (rotation
#         averaging -> global positioning -> sparse bundle adjustment) rewritten in PyTorch on the GPU.
# Goal  : same SIFT features/matches as the pycolmap backend of colmap.py, mapped by InstantSfM instead of the
#         COLMAP mapper, written in the same workspace.
# In    : frames cfg["extraction"]["output_dir"]/img_*; cfg["sfm"] (features/matching keys, intrinsics fixed
#         -> focal prior) and cfg["sfm"]["instantsfm"].{num_iteration_bundle_adjustment, ba_max_num_iterations,
#         skip_view_graph_calibration, skip_retriangulation, skip_pruning, max_frames, min_track_length, ba,
#         min_points_per_image}.
# Out   : gpu.py contract (database.db, sparse/0 = largest cluster, ...); sfm_metrics.csv with t_feature_s,
#         t_match_s, t_read_s, t_mapper_s, t_triangulate_s, t_ba_s, vram_peak_gb.
# Rules : license CC BY-NC 4.0. Its scikit-sparse dependency only provides cholesky(A)(b): when sksparse is not
#         installed (it needs the SuiteSparse headers) a scipy splu shim with the same call is registered.
#         Track ids are renumbered 0..n-1 (upstream keys overflow its int32 container, compact_track_ids()).
#         Its writer leaves inconsistent points2D, so sparse/0 is rebuilt from its cameras (largest cluster)
#         with gpu.triangulate(), the same path as mast3r_sfm and vggt.
#         min_track_length drops short tracks before global positioning / BA: its dense J^T J grows with the
#         tracks (about 400k tracks of 1850 frames at length 3 do not fit in 16 GB). The child sets
#         PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (less fragmentation) unless already set.
#         Never runs the ins-feat / ins-sfm scripts (they shell out to colmap).

# >>> BEGIN IMPLEMENTATION >>>
"""InstantSfM sfm technique: pycolmap SIFT + matching, InstantSfM global mapper on the GPU, COLMAP model."""
import logging
import os
import sys
import time
import types
from pathlib import Path

from src.sfm import gpu

log = logging.getLogger(__name__)

TECHNIQUE = "instantsfm"

def cholmod_shim() -> None:
    """Registers sksparse.cholmod.cholesky (factor once, call with b) backed by scipy when sksparse is absent."""

    try:
        import sksparse.cholmod  # noqa: F401
        return
    except ImportError:
        pass

    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import splu

    def cholesky(A):
        lu = splu(csc_matrix(A))
        return lambda b: lu.solve(b.toarray() if hasattr(b, "toarray") else b)

    pkg, mod = types.ModuleType("sksparse"), types.ModuleType("sksparse.cholmod")
    mod.cholesky = cholesky
    pkg.cholmod = mod
    sys.modules.setdefault("sksparse", pkg)
    sys.modules.setdefault("sksparse.cholmod", mod)

def compact_track_ids() -> None:
    """InstantSfM keys tracks by the union-find root (image_id << 32) | feature and stores them in an int32
    Tracks.ids: any image_id > 0 overflows. The keys are only identifiers, so they are renumbered 0..n-1."""

    from instantsfm.processors.track_establishment import TrackEngine

    establish = TrackEngine.EstablishFullTracks
    if getattr(establish, "compact", False):
        return

    def compact(self, options):
        return dict(enumerate(establish(self, options).values()))

    compact.compact = True
    TrackEngine.EstablishFullTracks = compact

def _child(cfg: dict, ws: str, state_path: str) -> None:
    """Spawned child: SIFT + match (pycolmap) -> InstantSfM ReadColmapDatabase / SolveGlobalMapper /
    WriteGlomapReconstruction -> sparse/<cluster>."""

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # before torch touches CUDA
    ws = Path(ws)
    gpu.redirect_output(ws)
    scfg, icfg = cfg["sfm"], cfg["sfm"]["instantsfm"]
    seed = int(cfg["run"]["seed"])
    state = gpu.State(state_path)
    from src.sfm import colmap as cli

    size = gpu.image_size(cli.list_frames(ws / "images")[0])
    gpu.extract_and_match(scfg, ws, seed, state, cli.resolve_intrinsics(scfg, cfg["geometry"]["intrinsics"], size))

    cholmod_shim()
    import random

    import numpy as np
    import torch
    from instantsfm.controllers.config import Config
    from instantsfm.controllers.data_reader import ReadColmapDatabase
    from instantsfm.controllers.global_mapper import SolveGlobalMapper

    compact_track_ids()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    view_graph, cameras, images, feature_name, rig = state.step("read", ReadColmapDatabase, str(ws / "database.db"))
    config = Config(feature_name)
    config.RUNTIME_OPTIONS["multi_camera_rig"] = rig
    for key in ("num_iteration_bundle_adjustment", "skip_view_graph_calibration", "skip_retriangulation",
                "skip_pruning"):
        config.OPTIONS[key] = icfg[key]
    config.BUNDLE_ADJUSTER_OPTIONS["max_num_iterations"] = int(icfg["ba_max_num_iterations"])
    for options in (config.TRACK_ESTABLISHMENT_OPTIONS, config.GLOBAL_POSITIONER_OPTIONS):
        options["min_num_view_per_track"] = int(icfg["min_track_length"])
    cameras, images, tracks = state.step("mapper", SolveGlobalMapper, view_graph, cameras, images, config,
                                         visualizer=None)
    poses, intrinsics = solved_cameras(cameras, images)
    print(f"instantsfm: {len(poses)} registered images in the largest cluster", flush=True)
    rec = gpu.posed_reconstruction(ws, poses, intrinsics, shared=bool(scfg["single_camera"]))
    gpu.triangulate(ws, rec, state, ba=bool(icfg["ba"]), refine_intrinsics=False,
                    min_points=int(icfg["min_points_per_image"]))
    state.done()

def solved_cameras(cameras, images) -> tuple[dict, dict]:
    """({name: 3x4 cam_from_world}, {name: (fx, fy, cx, cy)}) of the registered images of the largest cluster.
    Its own writer (WriteGlomapReconstruction) leaves points2D pointing at dropped tracks, which pycolmap
    rejects, so the model is rebuilt from these cameras by gpu.triangulate()."""

    import numpy as np

    reg = np.flatnonzero(images.is_registered)
    clusters = np.asarray(getattr(images, "cluster_ids", np.full(len(images.is_registered), -1)))[reg]
    if len(reg) and (clusters >= 0).any():
        values, counts = np.unique(clusters[clusters >= 0], return_counts=True)
        reg = reg[clusters == values[np.argmax(counts)]]
    poses, intrinsics = {}, {}
    for i in reg:
        name = images.filenames[i]
        c = images.cam_ids[i]
        poses[name] = np.asarray(images.world2cams[i])[:3]
        intrinsics[name] = (*map(float, cameras.focal_lengths[c]), *map(float, cameras.principal_points[c]))

    return poses, intrinsics

def run(cfg: dict, run_dir: str, artifacts_dir: str) -> Path | None:
    """Same contract as colmap.run(): <run_dir>/exports/colmap/sparse/0 or None when skipped/failed."""

    icfg = cfg["sfm"]["instantsfm"]
    ws, frames, model, reused = gpu.prepare(cfg, run_dir, TECHNIQUE, icfg["max_frames"])
    if reused:
        return gpu.reuse(cfg, ws, model, TECHNIQUE)

    if len(frames) < 2:
        log.warning("sfm: %d frames in %s, need >= 2: skipped", len(frames), cfg["extraction"]["output_dir"])
        return None

    if not gpu.cuda_ok():
        return None

    try:
        import instantsfm  # noqa: F401
    except ImportError as exc:
        log.warning("sfm: instantsfm not importable (%s; install requirements.txt): sfm skipped", exc)
        return None

    log.info("sfm (instantsfm): %d frames, matcher=%s, BA rounds %d", len(frames), cfg["sfm"]["matcher"],
             icfg["num_iteration_bundle_adjustment"])
    t0 = time.perf_counter()
    exitcode, state = gpu.spawn(_child, (cfg, str(ws)), ws)

    return gpu.finish(cfg, ws, TECHNIQUE, len(frames), artifacts_dir, exitcode, state, t0)

DEFAULTS = {"num_iteration_bundle_adjustment": 3, "ba_max_num_iterations": 100, "skip_view_graph_calibration": False,
            "skip_retriangulation": True, "skip_pruning": True, "max_frames": None, "min_track_length": 3, "ba": False,
            "min_points_per_image": 50}

if __name__ == "__main__":
    import numpy as np
    from scipy.sparse import csc_matrix

    for name in ("sksparse", "sksparse.cholmod"):
        sys.modules.pop(name, None)
    cholmod_shim()
    from sksparse.cholmod import cholesky

    A = csc_matrix(np.array([[4.0, 1.0], [1.0, 3.0]]))
    x = cholesky(A)(np.array([1.0, 2.0]))
    assert np.allclose(A @ x, [1.0, 2.0])
    X = cholesky(A)(csc_matrix(np.array([[1.0], [2.0]])))
    assert np.allclose(A @ X, [[1.0], [2.0]])
    print("OK")
# <<< END IMPLEMENTATION <<<

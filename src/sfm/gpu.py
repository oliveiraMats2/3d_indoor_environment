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
# Stage : sfm, helpers shared by the GPU techniques (instantsfm.py, mast3r_sfm.py, vggt.py).
# Goal  : one workspace and one output contract for every sfm.technique: the same <run_dir>/exports/colmap/
#         as colmap.py, so densification, mesh and evaluation do not change.
# In    : frames cfg["extraction"]["output_dir"]/img_*.{png,jpg}; cfg["sfm"] (technique + its block).
# Out   : <run_dir>/exports/colmap/{images/, database.db, sparse/0/*.bin+*.txt, sparse.ply, project.ini,
#         colmap.log, technique.txt}; sfm_metrics.csv columns t_*_s, t_total_s, technique, n_frames_in,
#         registered, points, mean_reproj_px, vram_peak_gb, backend (= technique), failed_step on failure.
# Rules : the GPU work runs in a spawned child (as pycolmap_backend.py) so a CUDA/C++ abort cannot kill
#         main.py and the VRAM is released when it ends; its stdout/stderr go to colmap.log; no shell commands.
#         A model is reused only when sfm.rerun is false AND technique.txt names the same technique.
#         Networks that output poses (mast3r_sfm, vggt): SIFT + matching through pycolmap on the used frames,
#         the network cameras written into a Reconstruction, pycolmap.triangulate_points, optional BA.

# >>> BEGIN IMPLEMENTATION >>>
"""Shared workspace, child process, pose -> COLMAP model and metrics for the GPU sfm techniques."""
import json
import logging
import multiprocessing as mp
import os
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from src.sfm import colmap as cli
from src.sfm import pycolmap_backend

log = logging.getLogger(__name__)

TECHNIQUES = ("colmap", "glomap", "instantsfm", "mast3r_sfm", "vggt")

def subsample(frames: list, max_frames: int | None) -> list:
    """Uniform subsample (first and last kept) down to max_frames; None or more than len -> all frames."""

    if not max_frames or len(frames) <= max_frames:
        return list(frames)

    idx = np.unique(np.linspace(0, len(frames) - 1, int(max_frames)).round().astype(int))

    return [frames[i] for i in idx]

def prepare(cfg: dict, run_dir: str, technique: str, max_frames: int | None = None):
    """(ws, frames used, model dir, reuse flag). reuse = sfm.rerun false, sparse/0/images.bin present and
    technique.txt == technique; otherwise the workspace is wiped and images/ holds only the used frames."""

    scfg = cfg["sfm"]
    ws = Path(run_dir) / "exports" / "colmap"
    model = ws / "sparse" / "0"
    marker = ws / "technique.txt"
    all_frames = cli.list_frames(Path(cfg["extraction"]["output_dir"]))
    frames = subsample(all_frames, max_frames)
    same = marker.exists() and marker.read_text().strip() == technique
    if not scfg["rerun"] and (model / "images.bin").exists() and same:
        return ws, frames, model, True

    if (model / "images.bin").exists() and not same:
        log.info("sfm: existing model was made by %s, rebuilding with %s",
                 marker.read_text().strip() if marker.exists() else "colmap", technique)
    ws.mkdir(parents=True, exist_ok=True)
    for stale in (ws / "database.db", ws / "sparse.ply", ws / "project.ini", marker):
        stale.unlink(missing_ok=True)
    shutil.rmtree(ws / "sparse", ignore_errors=True)
    (ws / "sparse").mkdir()
    linked = cli.link_images(frames, ws / "images", scfg["link_images"])
    log.info("sfm (%s): %d/%d frames in %s as %s", technique, len(frames), len(all_frames), ws / "images", linked)

    return ws, frames, model, False

def mark(ws: Path, technique: str) -> None:

    (ws / "technique.txt").write_text(technique + "\n")

def redirect_output(ws: Path) -> None:
    """Child process: stdout/stderr (Python, glog, CUDA) appended to colmap.log."""

    fd = os.open(ws / "colmap.log", os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)

class State:
    """Child-side progress file: the step being run, wall time of each finished step and the torch VRAM peak."""

    def __init__(self, path: str):

        self.path, self.data = Path(path), {}

    def save(self) -> None:

        self.path.write_text(json.dumps(self.data))

    def step(self, name: str, fn, *args, **kwargs):

        self.data["running"] = name
        self.save()
        print(f"\n$ {name}", flush=True)
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        self.data[f"t_{name}_s"] = round(time.perf_counter() - t0, 2)
        print(f"# wall {self.data[f't_{name}_s']:.2f}s", flush=True)
        self.data.update(vram_peak())
        self.save()

        return out

    def done(self, **extra) -> None:

        self.data.pop("running", None)
        self.data.update(extra)
        self.save()

def vram_peak() -> dict:
    """{"vram_peak_gb": torch peak allocated on the current device} or {} without torch/CUDA."""

    try:
        import torch
    except ImportError:
        return {}

    if not torch.cuda.is_available():
        return {}

    return {"vram_peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}

def cuda_ok() -> bool:
    """torch imports and sees a CUDA device (every GPU technique needs it)."""

    try:
        import torch
    except ImportError:
        log.warning("sfm: torch is not installed in this interpreter (see README: .venv-gpu)")
        return False

    if not torch.cuda.is_available():
        log.warning("sfm: torch %s sees no CUDA device", torch.__version__)
        return False

    return True

def spawn(target, args: tuple, ws: Path) -> tuple[int, dict]:
    """Runs target(*args, state_path) in a spawned child; returns (exit code, state dict)."""

    state_path = ws / "gpu_state.json"
    state_path.unlink(missing_ok=True)
    child = mp.get_context("spawn").Process(target=target, args=(*args, str(state_path)))
    child.start()
    child.join()
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    state_path.unlink(missing_ok=True)

    return child.exitcode, state

def extract_and_match(scfg: dict, ws: Path, seed: int, state: State, camera_params: list[float] | None = None) -> None:
    """Child side: pycolmap SIFT + matching (sfm.matcher, GPU when available) into ws/database.db; same
    options as the pycolmap backend of colmap.py. camera_params fixed the shared camera (OPENCV/PINHOLE)."""

    import pycolmap as pc

    scfg = {**scfg, "use_gpu": cli.resolve_gpu(scfg["use_gpu"], pycolmap_backend.gpu_info())}
    vocab = pycolmap_backend.resolve_vocab_tree(scfg.get("vocab_tree")) \
        if scfg["matcher"] == "sequential" and scfg["loop_detection"] else None
    opt = pycolmap_backend.build_options(scfg, camera_params, seed, vocab)
    db, images = str(ws / "database.db"), str(ws / "images")
    state.step("feature", pc.extract_features, db, images, camera_mode=opt["camera_mode"],
               reader_options=opt["reader"], extraction_options=opt["extraction"], device=opt["device"])
    fn = pc.match_sequential if scfg["matcher"] == "sequential" else pc.match_exhaustive
    state.step("match", fn, db, matching_options=opt["matching"], pairing_options=opt["pairing"], device=opt["device"])

def camera_params(model: str, f: tuple[float, float], c: tuple[float, float], n_params: int) -> list[float]:
    """Pinhole intrinsics in the parameter layout of a COLMAP model (distortion terms 0)."""

    fx, fy = f
    cx, cy = c
    if model.startswith("SIMPLE") or model in ("RADIAL", "RADIAL_FISHEYE", "FOV"):
        head = [(fx + fy) / 2, cx, cy]
    else:
        head = [fx, fy, cx, cy]

    return head + [0.0] * (n_params - len(head))

def posed_reconstruction(ws: Path, poses: dict, intrinsics: dict, shared: bool):
    """Reconstruction from database.db rigs/frames/images with the network cameras: poses {name: 3x4
    cam_from_world (OpenCV)}, intrinsics {name: (fx, fy, cx, cy)} at the ORIGINAL frame size. shared = one
    camera with the median intrinsics (single_camera). Frames without a pose stay unregistered."""

    import pycolmap as pc

    db = pc.Database.open(str(ws / "database.db"))
    cameras, rigs, frames, images = db.read_all_cameras(), db.read_all_rigs(), db.read_all_frames(), db.read_all_images()
    db.close()
    rec = pc.Reconstruction()
    by_camera = {}
    for image in images:
        by_camera.setdefault(image.camera_id, []).append(image.name)
    for cam in cameras:
        names = [n for n in by_camera.get(cam.camera_id, []) if n in intrinsics] if not shared \
            else [n for n in intrinsics]
        if names:
            k = np.median(np.array([intrinsics[n] for n in names], float), axis=0)
            cam.params = camera_params(cam.model.name, (k[0], k[1]), (k[2], k[3]), len(cam.params))
            cam.has_prior_focal_length = True
        rec.add_camera(cam)
    for rig in rigs:
        rec.add_rig(rig)
    image_of_frame = {}
    for image in images:
        image_of_frame[image.frame_id] = image
    for frame in frames:
        image = image_of_frame.get(frame.frame_id)
        if image is not None and image.name in poses:
            P = np.asarray(poses[image.name], float)
            frame.rig_from_world = pc.Rigid3d(pc.Rotation3d(P[:3, :3]), P[:3, 3])
        rec.add_frame(frame)
    for image in images:
        rec.add_image(image)
    for frame in frames:
        image = image_of_frame.get(frame.frame_id)
        if image is not None and image.name in poses:
            rec.register_frame(frame.frame_id)

    return rec

def triangulate(ws: Path, rec, state: State, ba: bool, refine_intrinsics: bool):
    """Child side: pycolmap.triangulate_points with the posed cameras -> ws/sparse/0; optional bundle
    adjustment (poses + intrinsics when refine_intrinsics), then points over 4 px or under 1.5 deg dropped."""

    import pycolmap as pc

    out = ws / "sparse" / "0"
    out.mkdir(parents=True, exist_ok=True)
    rec = state.step("triangulate", pc.triangulate_points, rec, str(ws / "database.db"), str(ws / "images"),
                     str(out), clear_points=True, refine_intrinsics=False)
    if ba:
        options = pc.BundleAdjustmentOptions()
        options.refine_focal_length = options.refine_principal_point = refine_intrinsics
        options.refine_extra_params = False
        state.step("ba", pc.bundle_adjustment, rec, options)
        pc.ObservationManager(rec).filter_all_points3D(4.0, 1.5)  # max reproj px, min tri angle deg (mapper defaults)
    rec.write(str(out))

    return rec

def model_stats(model: Path) -> dict:
    """Registered images, 3D points and mean reprojection error (px) of a model directory."""

    return pycolmap_backend.model_stats(model)

def finish(cfg: dict, ws: Path, technique: str, n_frames: int, artifacts_dir: str, exitcode: int, state: dict,
           t0: float) -> Path | None:
    """Parent side after the child: best model into sparse/0, exports (TXT, sparse.ply, project.ini),
    sfm_metrics.csv row; returns sparse/0 or None."""

    scfg = cfg["sfm"]
    times = {k: v for k, v in state.items() if k.startswith("t_") or k == "vram_peak_gb"}
    row = {**times, "technique": technique, "n_frames_in": n_frames, "failed_step": ""}  # clears an older failure

    def metrics(extra: dict) -> None:
        cli.write_metrics(artifacts_dir, cfg["run"]["id"], {**row, **extra,
                                                            "t_total_s": round(time.perf_counter() - t0, 2)},
                          backend_name=technique)

    if exitcode != 0:
        step = state.get("running", "start")
        log.warning("sfm: %s failed at %s (exit code %s, see %s): sfm stopped", technique, step, exitcode,
                    ws / "colmap.log")
        metrics({"failed_step": step})
        return None

    best, n_reg = cli.pick_best_model(ws / "sparse")
    if best is None or n_reg == 0:
        log.warning("sfm: %s reconstructed no model (see %s)", technique, ws / "colmap.log")
        metrics({})
        return None

    try:
        pycolmap_backend.export_model(scfg, ws, best, None)
    except Exception as exc:  # noqa: BLE001 - exports are optional, the model is kept
        log.warning("sfm: exporting the model failed (%s); sparse/0 is kept", exc)
    stats = model_stats(best)
    mark(ws, technique)
    log.info("sfm (%s): sparse/0 has %d/%d registered images, %d points, %.3f px", technique, stats["registered"],
             n_frames, stats["points"], stats["mean_reproj_px"])
    metrics(stats)

    return best

def reuse(cfg: dict, ws: Path, model: Path, technique: str) -> Path:

    log.info("sfm: reusing existing %s model %s (%d images; sfm.rerun=false)", technique, model,
             cli.count_registered(model))
    if not (ws / "sparse.ply").exists() or not (ws / "project.ini").exists() or not (model / "images.txt").exists():
        try:
            pycolmap_backend.export_model(cfg["sfm"], ws, model, None)
        except Exception as exc:  # noqa: BLE001 - exports are optional, the model is kept
            log.warning("sfm: exporting the reused model failed (%s)", exc)

    return model

def image_size(frame: Path) -> tuple[int, int]:
    """(width, height) of a frame."""

    img = cv2.imread(str(frame))

    return img.shape[1], img.shape[0]

if __name__ == "__main__":
    import tempfile

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    assert subsample(list(range(10)), None) == list(range(10)) and subsample(list(range(10)), 20) == list(range(10))
    s = subsample(list(range(100)), 5)
    assert s[0] == 0 and s[-1] == 99 and len(s) == 5
    assert camera_params("OPENCV", (10, 11), (5, 6), 8) == [10, 11, 5, 6, 0, 0, 0, 0]
    assert camera_params("SIMPLE_PINHOLE", (10, 12), (5, 6), 3) == [11, 5, 6]
    assert camera_params("SIMPLE_RADIAL", (10, 12), (5, 6), 4) == [11, 5, 6, 0]

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        fdir = tmp / "frames"
        fdir.mkdir()
        for k in range(6):
            cv2.imwrite(str(fdir / f"img_{k:04d}.png"), np.zeros((8, 8, 3), np.uint8))
        cfg = {"extraction": {"output_dir": str(fdir)}, "sfm": {**cli.DEFAULTS}}
        ws, frames, model, reused = prepare(cfg, str(tmp / "run"), "vggt", 3)
        assert not reused and len(frames) == 3 and sorted(p.name for p in (ws / "images").iterdir()) == \
            ["img_0000.png", "img_0002.png", "img_0005.png"]
        (model).mkdir(parents=True)
        (model / "images.bin").write_bytes(b"\0" * 8)
        mark(ws, "vggt")
        assert prepare(cfg, str(tmp / "run"), "vggt", 3)[3]          # same technique -> reuse
        assert not prepare(cfg, str(tmp / "run"), "mast3r_sfm", 3)[3]  # other technique -> rebuilt
        assert not (ws / "sparse" / "0").exists()

        # State: steps timed and saved, done() drops running
        st = State(str(tmp / "s.json"))
        assert st.step("x", lambda a: a + 1, 1) == 2
        st.done(n=1)
        d = json.loads((tmp / "s.json").read_text())
        assert "t_x_s" in d and "running" not in d and d["n"] == 1
    print("OK")
# <<< END IMPLEMENTATION <<<

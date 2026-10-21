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
# Stage : sfm, pycolmap backend (used only when sfm.backend resolves to pycolmap, see colmap.backend()).
# Goal  : same sparse reconstruction as the COLMAP CLI path of colmap.py, through the pycolmap wheels, for
#         machines without a COLMAP install. The CLI stays the primary path.
# In    : frames cfg["extraction"]["output_dir"]/img_*.{png,jpg}; cfg["sfm"]; cfg["geometry"]["intrinsics"].
# Out   : identical to colmap.py: <run_dir>/exports/colmap/{images/, database.db, sparse/0/*.bin+*.txt,
#         sparse.ply, project.ini (relative paths), colmap.log}; sfm_metrics.csv columns t_feature_s,
#         t_match_s, t_mapper_s, t_total_s, matcher, mapper, backend (= pycolmap), failed_step on failure.
# YAML  : every sfm key of colmap.py, mapped to pycolmap options (build_options()).
# Rules : feature/match/mapper run in a spawned child (a C++ abort cannot kill main.py) whose stdout/stderr
#         (glog) go to colmap.log; failure -> WARNING + failed_step, return None. vocab_tree url;name;sha256
#         is downloaded once into the COLMAP cache (~/.cache/colmap/<sha256>-<name>) and checked.

# >>> BEGIN IMPLEMENTATION >>>
"""pycolmap SfM backend: same workspace, exports and CSV columns as the COLMAP CLI path of colmap.py."""
import functools
import hashlib
import json
import logging
import multiprocessing as mp
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

import cv2

from src.sfm import colmap as cli

log = logging.getLogger(__name__)

@functools.lru_cache(maxsize=None)
def gpu_info() -> dict:
    """{"cuda": pycolmap built with CUDA AND `nvidia-smi -L` lists a GPU} (same rule as colmap.colmap_info)."""

    import pycolmap

    gpu = False
    if pycolmap.has_cuda and shutil.which("nvidia-smi"):
        try:
            gpu = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            gpu = False
    log.info("pycolmap %s: CUDA build %s, GPU visible %s", pycolmap.__version__, pycolmap.has_cuda, gpu)

    return {"cuda": bool(pycolmap.has_cuda and gpu)}

def device(use_gpu: bool):
    """pycolmap.Device for a resolved use_gpu flag."""

    import pycolmap

    return pycolmap.Device.cuda if use_gpu else pycolmap.Device.cpu

def cache_dir() -> Path:

    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "colmap"

def resolve_vocab_tree(spec: str | None) -> str | None:
    """Local path of sfm.vocab_tree: a path as is, or COLMAP `url;name;sha256` downloaded once into the
    COLMAP cache and checked against sha256 (pycolmap takes only a local path). None + WARNING on failure."""

    if not spec:
        return None
    if ";" not in spec:
        if Path(spec).is_file():
            return str(spec)
        log.warning("sfm.vocab_tree %s not found", spec)
        return None

    url, name, sha = spec.split(";")
    path = cache_dir() / f"{sha}-{name}"
    if not path.is_file():
        log.info("sfm: downloading the vocabulary tree %s -> %s", url, path)
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_suffix(path.suffix + ".part")
        try:
            urllib.request.urlretrieve(url, part)
        except OSError as exc:
            log.warning("sfm: vocabulary tree download failed (%s)", exc)
            part.unlink(missing_ok=True)
            return None
        digest = hashlib.sha256(part.read_bytes()).hexdigest()
        if digest != sha:
            log.warning("sfm: vocabulary tree sha256 %s != %s: discarded", digest, sha)
            part.unlink()
            return None
        part.rename(path)

    return str(path)

def build_options(scfg: dict, cam_params: list[float] | None, seed: int, vocab: str | None) -> dict:
    """pycolmap option objects equivalent to colmap.build_commands() (scfg["use_gpu"] already resolved)."""

    import pycolmap as pc

    gpu, idx = bool(scfg["use_gpu"]), str(scfg["gpu_index"])
    reader = pc.ImageReaderOptions()
    reader.camera_model = scfg["camera_model"]
    if cam_params is not None:
        reader.camera_params = ",".join(f"{p:.10g}" for p in cam_params)
    extraction = pc.FeatureExtractionOptions()
    extraction.use_gpu, extraction.gpu_index = gpu, idx
    extraction.max_image_size = int(scfg["max_image_size"])
    extraction.sift.max_num_features = int(scfg["max_num_features"])
    matching = pc.FeatureMatchingOptions()
    matching.use_gpu, matching.gpu_index = gpu, idx

    if scfg["matcher"] == "sequential":
        pairing = pc.SequentialPairingOptions()
        pairing.overlap = int(scfg["sequential_overlap"])
        loop = bool(scfg["loop_detection"])
        if loop and vocab is None:
            log.warning("sfm.loop_detection=true needs a usable sfm.vocab_tree: loop detection disabled")
            loop = False
        pairing.loop_detection = loop
        if loop:
            pairing.vocab_tree_path = vocab
    elif scfg["matcher"] == "exhaustive":
        pairing = pc.ExhaustivePairingOptions()
    else:
        raise ValueError(f"sfm.matcher must be sequential|exhaustive, got {scfg['matcher']}")

    fixed, ba_gpu = cam_params is not None, gpu and bool(scfg["ba_use_gpu"])
    if scfg["mapper"] == "incremental":
        mapper = pc.IncrementalPipelineOptions()
        mapper.random_seed = mapper.mapper.random_seed = seed
        mapper.min_num_matches = int(scfg["min_num_matches"])
        mapper.multiple_models = bool(scfg["multiple_models"])
        mapper.ba_use_gpu, mapper.ba_gpu_index = ba_gpu, idx
        for name in ("ba_global_max_num_iterations", "ba_global_max_refinements"):
            if scfg.get(name) is not None:
                setattr(mapper, name, int(scfg[name]))
        if fixed:
            mapper.ba_refine_focal_length = mapper.ba_refine_principal_point = mapper.ba_refine_extra_params = False
    elif scfg["mapper"] == "global":
        mapper = pc.GlobalPipelineOptions()
        mapper.random_seed = mapper.mapper.random_seed = seed
        mapper.min_num_matches = int(scfg["min_num_matches"])
        mapper.multiple_models = bool(scfg["multiple_models"])
        gp, ba = mapper.mapper.global_positioning, mapper.mapper.bundle_adjustment
        gp.use_gpu, gp.gpu_index = gpu, idx
        ba.ceres.use_gpu, ba.ceres.gpu_index, mapper.mapper.ba_gpu_index = ba_gpu, idx, idx
        if fixed:
            ba.refine_focal_length = ba.refine_principal_point = ba.refine_extra_params = False
    else:
        raise ValueError(f"sfm.mapper must be incremental|global, got {scfg['mapper']}")

    camera_mode = pc.CameraMode.SINGLE if scfg["single_camera"] else pc.CameraMode.AUTO

    return {"reader": reader, "extraction": extraction, "matching": matching, "pairing": pairing,
            "mapper": mapper, "camera_mode": camera_mode, "device": device(gpu)}

def _pipeline(scfg: dict, ws: str, cam_params, seed: int, vocab, state_path: str) -> None:
    """Child process: feature -> match -> mapper; stdout/stderr (glog included) appended to colmap.log;
    state_path (json) holds the step being run and the wall time of each finished step."""

    ws = Path(ws)
    fd = os.open(ws / "colmap.log", os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    import pycolmap as pc

    pc.logging.logtostderr, pc.logging.colorlogtostderr, pc.logging.minloglevel = True, False, 0
    pc.set_random_seed(seed)
    opt = build_options(scfg, cam_params, seed, vocab)
    db, images, sparse = str(ws / "database.db"), str(ws / "images"), str(ws / "sparse")
    state = {}

    def feature():
        pc.extract_features(db, images, camera_mode=opt["camera_mode"], reader_options=opt["reader"],
                            extraction_options=opt["extraction"], device=opt["device"])

    def match():
        fn = pc.match_sequential if scfg["matcher"] == "sequential" else pc.match_exhaustive
        fn(db, matching_options=opt["matching"], pairing_options=opt["pairing"], device=opt["device"])

    def mapper():
        fn = pc.incremental_mapping if scfg["mapper"] == "incremental" else pc.global_mapping
        recs = fn(db, images, sparse, options=opt["mapper"])
        state["n_models"] = len(recs)
        if not any(p.is_dir() and p.name.isdigit() for p in Path(sparse).iterdir()):
            for i, rec in recs.items():  # older wheels return the models without writing them
                (Path(sparse) / str(i)).mkdir()
                rec.write(str(Path(sparse) / str(i)))

    for step, fn in (("feature", feature), ("match", match), ("mapper", mapper)):
        state["running"] = step
        Path(state_path).write_text(json.dumps(state))
        print(f"\n$ pycolmap {step} ({scfg['matcher'] if step == 'match' else scfg['mapper'] if step == 'mapper' else 'sift'})",
              flush=True)
        t0 = time.perf_counter()
        fn()
        state[f"t_{step}_s"] = round(time.perf_counter() - t0, 2)
        print(f"# wall {state[f't_{step}_s']:.2f}s", flush=True)
    state.pop("running")
    Path(state_path).write_text(json.dumps(state))

def write_ini(scfg: dict, ws: Path, cam_params: list[float] | None) -> None:
    """project.ini with paths RELATIVE to exports/colmap/: `colmap project_generator` when the CLI is there
    (as colmap.export_model), else the same keys written directly."""

    ini = ws / "project.ini"
    if cli.colmap_info(scfg["colmap_bin"])["path"]:
        try:
            cli._run([scfg["colmap_bin"], "project_generator", "--log_color", "0", "--output_path", str(ini),
                      "--quality", scfg["project_quality"]], ws / "colmap.log")
        except (subprocess.CalledProcessError, OSError):
            ini.unlink(missing_ok=True)
    if not ini.exists():
        ini.write_text("[ImageReader]\n")
    values = {("", "database_path"): "database.db", ("", "image_path"): "images", ("", "log_color"): "false",
              ("ImageReader", "camera_model"): scfg["camera_model"],
              ("ImageReader", "single_camera"): "true" if scfg["single_camera"] else "false"}
    if cam_params is not None:
        values[("ImageReader", "camera_params")] = ",".join(f"{p:.10g}" for p in cam_params)
    cli.patch_ini(ini, values)

def export_model(scfg: dict, ws: Path, model: Path, cam_params: list[float] | None) -> None:
    """TXT next to the .bin, sparse.ply and project.ini, as colmap.export_model() but through pycolmap."""

    import pycolmap

    rec = pycolmap.Reconstruction(str(model))
    rec.write_text(str(model))
    rec.export_PLY(str(ws / "sparse.ply"))
    write_ini(scfg, ws, cam_params)
    log.info("sfm: COLMAP GUI (when installed): cd %s && colmap gui --project_path project.ini", ws)

def run(cfg: dict, run_dir: str, artifacts_dir: str) -> Path | None:
    """Same contract as colmap.run(): <run_dir>/exports/colmap/sparse/0 or None when skipped/failed."""

    scfg = cfg["sfm"]
    try:
        import pycolmap  # noqa: F401
    except ImportError as exc:
        log.warning("sfm: pycolmap not importable (%s): sfm skipped", exc)
        return None

    scfg = {**scfg, "use_gpu": cli.resolve_gpu(scfg["use_gpu"], gpu_info())}
    ws = Path(run_dir) / "exports" / "colmap"
    model = ws / "sparse" / "0"
    frames = cli.list_frames(Path(cfg["extraction"]["output_dir"]))
    first = cv2.imread(str(frames[0])) if frames else None
    size = (first.shape[1], first.shape[0]) if first is not None else None
    cam_params = cli.resolve_intrinsics(scfg, cfg["geometry"]["intrinsics"], size)
    run_id, seed = cfg["run"]["id"], int(cfg["run"]["seed"])

    def metrics(row: dict) -> None:
        cli.write_metrics(artifacts_dir, run_id, {**row, "matcher": scfg["matcher"], "mapper": scfg["mapper"]},
                          backend_name="pycolmap")

    if not scfg["rerun"] and (model / "images.bin").exists():
        log.info("sfm: reusing existing model %s (%d images; sfm.rerun=false)", model, cli.count_registered(model))
        if not (ws / "sparse.ply").exists() or not (ws / "project.ini").exists() or not (model / "images.txt").exists():
            try:
                export_model(scfg, ws, model, cam_params)
            except Exception as exc:  # noqa: BLE001 - exports are optional, the model is kept
                log.warning("sfm: exporting the reused model failed (%s)", exc)
        return model

    if len(frames) < 2:
        log.warning("sfm: %d frames in %s, need >= 2: skipped", len(frames), cfg["extraction"]["output_dir"])
        return None

    ws.mkdir(parents=True, exist_ok=True)
    for stale in (ws / "database.db", ws / "sparse.ply", ws / "project.ini"):
        stale.unlink(missing_ok=True)
    shutil.rmtree(ws / "sparse", ignore_errors=True)
    (ws / "sparse").mkdir()
    linked = cli.link_images(frames, ws / "images", scfg["link_images"])
    log.info("sfm: %d frames in %s as %s", len(frames), ws / "images", linked)
    log.info("sfm (pycolmap): %d frames, matcher=%s, mapper=%s, intrinsics=%s, gpu=%s",
             len(frames), scfg["matcher"], scfg["mapper"], "fixed" if cam_params else "estimate", scfg["use_gpu"])
    vocab = resolve_vocab_tree(scfg.get("vocab_tree")) if scfg["matcher"] == "sequential" and scfg["loop_detection"] else None

    log_path, state_path = ws / "colmap.log", ws / "pycolmap_state.json"
    state_path.unlink(missing_ok=True)
    t0 = time.perf_counter()
    child = mp.get_context("spawn").Process(target=_pipeline,
                                            args=(scfg, str(ws), cam_params, seed, vocab, str(state_path)))
    child.start()
    child.join()
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    state_path.unlink(missing_ok=True)
    times = {k: v for k, v in state.items() if k.startswith("t_")}
    if child.exitcode != 0:
        step = state.get("running", "feature")
        log.warning("sfm: pycolmap %s failed (exit code %s, see %s): sfm stopped", step, child.exitcode, log_path)
        metrics({**times, "t_total_s": round(time.perf_counter() - t0, 2), "failed_step": step})
        return None

    best, n_reg = cli.pick_best_model(ws / "sparse")
    if best is None or n_reg == 0:
        log.warning("sfm: pycolmap reconstructed no model (see %s)", log_path)
        metrics({**times, "t_total_s": round(time.perf_counter() - t0, 2)})
        return None

    try:
        export_model(scfg, ws, best, cam_params)
    except Exception as exc:  # noqa: BLE001 - exports are optional, the model is kept
        log.warning("sfm: exporting the model failed (%s); sparse/0 is kept", exc)
    log.info("sfm: sparse/0 has %d/%d registered images", n_reg, len(frames))
    metrics({**times, "t_total_s": round(time.perf_counter() - t0, 2)})

    return best

def model_stats(model: Path) -> dict:
    """Registered images, 3D points and mean reprojection error (px) of a model directory."""

    import pycolmap

    rec = pycolmap.Reconstruction(str(model))

    return {"registered": rec.num_reg_images(), "points": rec.num_points3D(),
            "mean_reproj_px": round(rec.compute_mean_reprojection_error(), 4)}

if __name__ == "__main__":
    import tempfile

    import numpy as np

    from src import stage_io

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    import pycolmap

    base = dict(cli.DEFAULTS)
    o = build_options({**base, "use_gpu": True}, None, 7, None)
    assert o["reader"].camera_model == "OPENCV" and o["reader"].camera_params == ""
    assert o["camera_mode"] == pycolmap.CameraMode.SINGLE and o["device"] == pycolmap.Device.cuda
    assert o["extraction"].use_gpu and o["extraction"].gpu_index == "0" and o["extraction"].max_image_size == 1920
    assert o["extraction"].sift.max_num_features == 8192 and o["matching"].use_gpu
    assert isinstance(o["pairing"], pycolmap.SequentialPairingOptions) and o["pairing"].overlap == 10
    assert not o["pairing"].loop_detection
    m = o["mapper"]
    assert isinstance(m, pycolmap.IncrementalPipelineOptions) and m.random_seed == 7 and m.min_num_matches == 15
    assert m.multiple_models and not m.ba_use_gpu and m.ba_refine_focal_length

    # loop detection: no vocab -> disabled; vocab -> enabled with the local path
    assert not build_options({**base, "loop_detection": True}, None, 0, None)["pairing"].loop_detection
    p = build_options({**base, "loop_detection": True}, None, 0, "/v.bin")["pairing"]
    assert p.loop_detection and str(p.vocab_tree_path) == "/v.bin"

    # exhaustive + global + fixed intrinsics, CPU
    s = {**base, "matcher": "exhaustive", "mapper": "global", "intrinsics": "fixed", "use_gpu": False,
         "single_camera": False, "fixed_dist": [0.1, -0.05, 0, 0]}
    params = cli.resolve_intrinsics(s, {"fx": 1500.0, "fy": None, "cx": None, "cy": 540.5}, (1920, 1080))
    o = build_options(s, params, 1, None)
    assert o["reader"].camera_params == "1500,1500,960,540.5,0.1,-0.05,0,0" and o["camera_mode"] == pycolmap.CameraMode.AUTO
    assert isinstance(o["pairing"], pycolmap.ExhaustivePairingOptions) and o["device"] == pycolmap.Device.cpu
    g = o["mapper"]
    assert isinstance(g, pycolmap.GlobalPipelineOptions) and g.random_seed == 1
    assert not g.mapper.global_positioning.use_gpu and not g.mapper.bundle_adjustment.refine_focal_length
    assert not g.mapper.bundle_adjustment.refine_extra_params
    o = build_options({**s, "mapper": "incremental", "ba_global_max_refinements": 3, "use_gpu": True,
                       "ba_use_gpu": True}, params, 1, None)
    assert o["mapper"].ba_global_max_refinements == 3 and o["mapper"].ba_use_gpu and o["mapper"].ba_gpu_index == "0"
    assert not o["mapper"].ba_refine_focal_length and not o["mapper"].ba_refine_extra_params

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # vocab tree: a missing path -> None; url spec already cached with the right sha256 -> cache path
        assert resolve_vocab_tree(None) is None and resolve_vocab_tree(str(tmp / "nope.bin")) is None
        os.environ["XDG_CACHE_HOME"] = str(tmp / "cache")
        blob = b"vocab"
        sha = hashlib.sha256(blob).hexdigest()
        (tmp / "cache" / "colmap").mkdir(parents=True)
        (tmp / "cache" / "colmap" / f"{sha}-v.bin").write_bytes(blob)
        assert resolve_vocab_tree(f"https://example.invalid/v.bin;v.bin;{sha}") == str(tmp / "cache" / "colmap" / f"{sha}-v.bin")
        assert resolve_vocab_tree("http://127.0.0.1:9/x.bin;x.bin;00") is None  # download refused
        del os.environ["XDG_CACHE_HOME"]

        # project.ini without the CLI: relative paths, same keys as colmap.export_model
        (tmp / "ws").mkdir()
        write_ini({**base, "colmap_bin": "no_such_colmap_bin"}, tmp / "ws", params)
        ini = (tmp / "ws" / "project.ini").read_text()
        assert "database_path=database.db\n" in ini and "image_path=images\n" in ini and "log_color=false" in ini
        assert "camera_params=1500,1500,960,540.5,0.1,-0.05,0,0" in ini and "single_camera=true" in ini

        # synthetic frames with no texture: the child fails (or builds no model) -> WARNING, None, metrics row
        fdir, art = tmp / "flat", tmp / "art"
        fdir.mkdir()
        for k in range(3):
            cv2.imwrite(str(fdir / f"img_{k:04d}.png"), np.full((64, 64, 3), 127, np.uint8))
        fcfg = {"run": {"id": "flat", "seed": 0}, "extraction": {"output_dir": str(fdir)},
                "geometry": {"intrinsics": {"fx": None}}, "sfm": {**base, "backend": "pycolmap", "use_gpu": False}}
        assert cli.run(fcfg, str(tmp / "frun"), str(art)) is None
        row = stage_io.read_csv(art / "sfm_metrics.csv")[0]
        assert row["backend"] == "pycolmap" and row["matcher"] == "sequential" and row["t_total_s"], row
        assert (tmp / "frun" / "exports" / "colmap" / "colmap.log").stat().st_size > 0

        # a pycolmap error inside the child (unknown camera model) -> failed_step=feature, main.py unharmed
        bad = {**fcfg, "run": {"id": "bad", "seed": 0}, "sfm": {**fcfg["sfm"], "camera_model": "NO_SUCH_MODEL"}}
        assert cli.run(bad, str(tmp / "brun"), str(art)) is None
        row = [r for r in stage_io.read_csv(art / "sfm_metrics.csv") if r["run_id"] == "bad"][0]
        assert row["failed_step"] == "feature" and row["backend"] == "pycolmap", row

        # real smoke (SFM_SMOKE=1): both backends on the same 40 frames (every SFM_SMOKE_STRIDE-th, default 25), copied into tmp
        repo = Path(__file__).resolve().parents[2]
        src = Path(os.environ.get("SFM_SMOKE_FRAMES", repo / "video_preview" / "frames" / "d1e24530"))
        if os.environ.get("SFM_SMOKE") == "1":
            frames = cli.list_frames(src)[::int(os.environ.get("SFM_SMOKE_STRIDE", "25"))][:40]
            assert len(frames) >= 8, f"need frames in {src} (set SFM_SMOKE_FRAMES)"
            sdir = tmp / "smoke_frames"
            sdir.mkdir()
            for f in frames:
                shutil.copy(f, sdir / f.name)
            print(f"smoke: {len(frames)} frames from {src}")
            results = {}
            for name in ("cli", "pycolmap"):
                cfg = {"run": {"id": f"smoke_{name}", "seed": 0}, "extraction": {"output_dir": str(sdir)},
                       "geometry": {"intrinsics": {"fx": None, "fy": None, "cx": None, "cy": None}},
                       "sfm": {**base, "backend": name, "rerun": True}}
                t0 = time.perf_counter()
                model = cli.run(cfg, str(tmp / f"run_{name}"), str(art))
                wall = round(time.perf_counter() - t0, 2)
                assert model is not None, name
                ws_dir = model.parents[1]
                for f in ("database.db", "sparse.ply", "project.ini", "colmap.log"):
                    assert (ws_dir / f).exists(), (name, f)
                assert all((model / f).exists() for f in ("cameras.bin", "images.bin", "points3D.bin",
                                                          "cameras.txt", "images.txt", "points3D.txt"))
                assert "database_path=database.db\n" in (ws_dir / "project.ini").read_text()
                row = [r for r in stage_io.read_csv(art / "sfm_metrics.csv") if r["run_id"] == f"smoke_{name}"][0]
                assert row["backend"] == name
                results[name] = {**model_stats(model), "wall_s": wall,
                                 **{k: row[k] for k in ("t_feature_s", "t_match_s", "t_mapper_s", "t_total_s")}}
            for name, r in results.items():
                print(f"smoke {name:8s}: {r}")
            assert all(r["registered"] >= 2 for r in results.values())  # small subsets: counts vary run to run
    print("OK")
# <<< END IMPLEMENTATION <<<

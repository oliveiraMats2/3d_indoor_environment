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
# Stage : sfm (COLMAP 4.x CLI, CUDA when available).
# Goal  : sparse reconstruction of the selected frames with COLMAP; the best model always ends in sparse/0.
# In    : frames cfg["extraction"]["output_dir"]/img_*.{png,jpg}; cfg["sfm"]; cfg["geometry"]["intrinsics"].
# Out   : <run_dir>/exports/colmap/{images/ (symlinks), database.db, sparse/0/*.bin+*.txt, sparse.ply,
#         project.ini, colmap.log}; artifacts_dir/sfm_metrics.csv (row key run_id, merged with the
#         columns other sfm modules write): t_feature_s, t_match_s, t_mapper_s, t_total_s, matcher, mapper.
# YAML  : sfm.{enabled, rerun, colmap_bin, camera_model, single_camera, intrinsics (estimate|fixed),
#         fixed_dist [k1,k2,p1,p2], matcher (sequential|exhaustive), sequential_overlap, loop_detection,
#         vocab_tree, mapper (incremental|global), use_gpu, gpu_index, ba_use_gpu, max_image_size,
#         max_num_features, min_num_matches, multiple_models, ba_global_max_num_iterations,
#         ba_global_max_refinements, project_quality, backend (auto|cli|pycolmap)}.
# Backend: the COLMAP CLI below is the primary path. backend(cfg) picks it whenever sfm.colmap_bin is a
#         COLMAP >= 4.0 (auto); pycolmap (src/sfm/pycolmap_backend.py, same outputs and CSV columns) only
#         when the binary is missing or with sfm.backend: pycolmap.
# Rules : colmap missing -> WARNING and return None. sfm.rerun false and sparse/0/images.bin present ->
#         reuse. loop_detection without vocab_tree -> WARNING, disabled. intrinsics fixed with fx null ->
#         WARNING, estimate. Every command appends stdout/stderr to colmap.log, check=True, timed.

# >>> BEGIN IMPLEMENTATION >>>
"""COLMAP sparse reconstruction: features, matching, mapping, model selection and exports."""
import functools
import importlib.util
import logging
import os
import re
import shutil
import struct
import subprocess
import time
from pathlib import Path

import cv2

from src import stage_io

log = logging.getLogger(__name__)

IMAGE_GLOBS = ("img_*.png", "img_*.jpg", "img_*.jpeg")
MIN_VERSION = (4, 0)  # FeatureExtraction.* / FeatureMatching.* option names and global_mapper are COLMAP 4.x

def parse_banner(text: str) -> tuple[tuple[int, int] | None, bool]:
    """((major, minor) or None, built with CUDA) from the first lines of `colmap -h`."""

    m = re.search(r"COLMAP\s+(\d+)\.(\d+)", text)

    return ((int(m.group(1)), int(m.group(2))) if m else None), "with CUDA" in text

@functools.lru_cache(maxsize=None)
def colmap_info(exe: str) -> dict:
    """{path, version (major, minor) | None, cuda}: cuda = COLMAP built with CUDA AND `nvidia-smi -L` lists a
    GPU. path None when the binary is not on PATH. Cached: every stage asks the same binary once."""

    path = shutil.which(exe)
    if path is None:
        return {"path": None, "version": None, "cuda": False}

    try:
        out = subprocess.run([path, "-h"], capture_output=True, text=True, timeout=60)
        version, cuda_build = parse_banner(out.stdout + out.stderr)
    except (OSError, subprocess.TimeoutExpired):
        version, cuda_build = None, False

    gpu = False
    if cuda_build and shutil.which("nvidia-smi"):
        try:
            gpu = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            gpu = False

    info = {"path": path, "version": version, "cuda": cuda_build and gpu}
    log.info("colmap_info %s: version %s, CUDA build %s, GPU visible %s", path,
             ".".join(map(str, version)) if version else "unknown", cuda_build, gpu)

    return info

@functools.lru_cache(maxsize=None)
def _backend(requested: str, exe: str) -> str:

    info = colmap_info(exe)
    cli_ok = info["path"] is not None and info["version"] is not None and info["version"] >= MIN_VERSION
    py_ok = importlib.util.find_spec("pycolmap") is not None
    if requested == "cli" or (requested == "auto" and (cli_ok or not py_ok)):
        chosen = "cli"
    elif requested == "pycolmap" and not py_ok:
        log.warning("sfm.backend=pycolmap but pycolmap is not installed (uv sync --extra cpu|cuda): %s",
                    "using the COLMAP CLI" if cli_ok else "sfm will be skipped")
        chosen = "cli"
    else:
        chosen = "pycolmap"
    log.info("sfm backend: %s (requested %s; CLI %s usable %s, pycolmap installed %s)", chosen, requested, exe,
             cli_ok, py_ok)

    return chosen

def backend(cfg: dict) -> str:
    """"cli" | "pycolmap" for cfg["sfm"]["backend"] (auto = CLI when sfm.colmap_bin is a COLMAP >= 4.0, else
    pycolmap when importable, else "cli", whose run() then skips with a WARNING). Resolved once per process."""

    scfg = cfg["sfm"]

    return _backend(scfg.get("backend", "auto"), scfg["colmap_bin"])

def resolve_gpu(value, info: dict) -> bool:
    """sfm.use_gpu true | false | "auto" -> bool. auto = CUDA available; true without CUDA -> WARNING, CPU."""

    if value == "auto":
        return bool(info["cuda"])

    if value and not info["cuda"]:
        log.warning("sfm.use_gpu=true but COLMAP has no CUDA build or no GPU is visible: using CPU")
        return False

    return bool(value)

def _b(value) -> str:

    return "1" if value else "0"

def list_frames(frames_dir: Path) -> list[Path]:

    return sorted({p for g in IMAGE_GLOBS for p in Path(frames_dir).glob(g)})

def link_images(frames: list[Path], images_dir: Path, mode: str = "auto") -> str:
    """images/ holds only the current frames, as hardlinks (auto: hardlink, copy when the file system
    refuses, e.g. another device), RELATIVE symlinks or copies, so exports/colmap/ can be moved or zipped.
    Returns the mode used for the last frame."""

    images_dir.mkdir(parents=True, exist_ok=True)
    for old in images_dir.iterdir():
        if old.is_symlink() or old.is_file():
            old.unlink()

    used = mode
    for frame in frames:
        dst = images_dir / frame.name
        if mode == "symlink":
            dst.symlink_to(os.path.relpath(frame.resolve(), images_dir.resolve()))
            continue

        if mode in ("auto", "hardlink"):
            try:
                os.link(frame, dst)
                used = "hardlink"
                continue
            except OSError as exc:
                if mode == "hardlink":
                    raise
                if used != "copy":
                    log.warning("sfm: hardlink failed (%s): copying the frames into %s", exc, images_dir)

        shutil.copy2(frame, dst)
        used = "copy"

    return used

def resolve_intrinsics(scfg: dict, intr: dict, image_size: tuple[int, int] | None) -> list[float] | None:
    """Camera params for intrinsics=fixed (OPENCV: fx,fy,cx,cy,k1,k2,p1,p2; PINHOLE: fx,fy,cx,cy), else None."""

    if scfg["intrinsics"] != "fixed":
        return None

    if intr.get("fx") is None:
        log.warning("sfm.intrinsics=fixed but geometry.intrinsics.fx is null: falling back to estimate")
        return None

    if scfg["camera_model"] not in ("OPENCV", "PINHOLE"):
        log.warning("sfm.intrinsics=fixed supports OPENCV/PINHOLE only (got %s): falling back to estimate",
                    scfg["camera_model"])
        return None

    w, h = image_size or (0, 0)
    fx = float(intr["fx"])
    fy = float(intr["fy"]) if intr.get("fy") is not None else fx
    cx = float(intr["cx"]) if intr.get("cx") is not None else w / 2.0
    cy = float(intr["cy"]) if intr.get("cy") is not None else h / 2.0
    params = [fx, fy, cx, cy]
    if scfg["camera_model"] == "OPENCV":
        params += [float(d) for d in (scfg.get("fixed_dist") or [0.0, 0.0, 0.0, 0.0])]

    return params

def build_commands(scfg: dict, ws: Path, cam_params: list[float] | None, seed: int) -> dict[str, list[str]]:
    """Argument lists for the feature, match and mapper steps (no execution)."""

    exe = scfg["colmap_bin"]
    db, images, sparse = str(ws / "database.db"), str(ws / "images"), str(ws / "sparse")
    common = ["--log_color", "0", "--default_random_seed", str(seed)]
    gpu, gpu_idx = _b(scfg["use_gpu"]), str(scfg["gpu_index"])

    feature = [exe, "feature_extractor", *common, "--database_path", db, "--image_path", images,
               "--ImageReader.camera_model", scfg["camera_model"],
               "--ImageReader.single_camera", _b(scfg["single_camera"]),
               "--FeatureExtraction.use_gpu", gpu, "--FeatureExtraction.gpu_index", gpu_idx,
               "--FeatureExtraction.max_image_size", str(scfg["max_image_size"]),
               "--SiftExtraction.max_num_features", str(scfg["max_num_features"])]
    if cam_params is not None:
        feature += ["--ImageReader.camera_params", ",".join(f"{p:.10g}" for p in cam_params)]

    match = [exe, f"{scfg['matcher']}_matcher", *common, "--database_path", db,
             "--FeatureMatching.use_gpu", gpu, "--FeatureMatching.gpu_index", gpu_idx]
    if scfg["matcher"] == "sequential":
        loop = bool(scfg["loop_detection"])
        if loop and not scfg.get("vocab_tree"):
            log.warning("sfm.loop_detection=true needs sfm.vocab_tree: loop detection disabled")
            loop = False
        match += ["--SequentialMatching.overlap", str(scfg["sequential_overlap"]),
                  "--SequentialMatching.loop_detection", _b(loop)]
        if loop:
            match += ["--SequentialMatching.vocab_tree_path", str(scfg["vocab_tree"])]
    elif scfg["matcher"] != "exhaustive":
        raise ValueError(f"sfm.matcher must be sequential|exhaustive, got {scfg['matcher']}")

    fixed = cam_params is not None
    if scfg["mapper"] == "incremental":
        mapper = [exe, "mapper", *common, "--database_path", db, "--image_path", images, "--output_path", sparse,
                  "--Mapper.random_seed", str(seed),
                  "--Mapper.min_num_matches", str(scfg["min_num_matches"]),
                  "--Mapper.multiple_models", _b(scfg["multiple_models"]),
                  "--Mapper.ba_use_gpu", _b(scfg["use_gpu"] and scfg["ba_use_gpu"]),
                  "--Mapper.ba_gpu_index", gpu_idx]
        for name in ("ba_global_max_num_iterations", "ba_global_max_refinements"):
            if scfg.get(name) is not None:
                mapper += [f"--Mapper.{name}", str(scfg[name])]
        prefix = "--Mapper."
    elif scfg["mapper"] == "global":
        mapper = [exe, "global_mapper", *common, "--database_path", db, "--image_path", images,
                  "--output_path", sparse, "--GlobalMapper.random_seed", str(seed),
                  "--GlobalMapper.min_num_matches", str(scfg["min_num_matches"]),
                  "--GlobalMapper.multiple_models", _b(scfg["multiple_models"]),
                  "--GlobalMapper.gp_use_gpu", gpu, "--GlobalMapper.gp_gpu_index", gpu_idx,
                  "--GlobalMapper.ba_ceres_use_gpu", _b(scfg["use_gpu"] and scfg["ba_use_gpu"]),
                  "--GlobalMapper.ba_gpu_index", gpu_idx]
        prefix = "--GlobalMapper."
    else:
        raise ValueError(f"sfm.mapper must be incremental|global, got {scfg['mapper']}")

    if fixed:
        for name in ("ba_refine_focal_length", "ba_refine_principal_point", "ba_refine_extra_params"):
            mapper += [prefix + name, "0"]

    return {"feature": feature, "match": match, "mapper": mapper}

def count_registered(model_dir: Path) -> int:
    """Registered images of a model: images.bin header (uint64), else images.txt entries; 0 if absent."""

    model_dir = Path(model_dir)
    if (model_dir / "images.bin").exists():
        with open(model_dir / "images.bin", "rb") as fh:
            head = fh.read(8)
        return struct.unpack("<Q", head)[0] if len(head) == 8 else 0

    if (model_dir / "images.txt").exists():
        lines = [ln for ln in (model_dir / "images.txt").read_text().splitlines() if ln and not ln.startswith("#")]
        return (len(lines) + 1) // 2

    return 0

def pick_best_model(sparse_dir: Path) -> tuple[Path | None, int]:
    """Swap the model with most registered images into sparse/0; returns (sparse/0, n_registered)."""

    models = sorted((d for d in Path(sparse_dir).iterdir() if d.is_dir() and d.name.isdigit()), key=lambda d: int(d.name)) \
        if Path(sparse_dir).exists() else []
    if not models:
        return None, 0

    counts = {d.name: count_registered(d) for d in models}
    best = max(models, key=lambda d: (counts[d.name], -int(d.name)))
    if len(models) > 1:
        log.warning("COLMAP produced %d models %s: keeping %s (%d images) as sparse/0",
                    len(models), counts, best.name, counts[best.name])

    target = Path(sparse_dir) / "0"
    if best != target:
        tmp = Path(sparse_dir) / "_swap"
        if target.exists():
            target.rename(tmp)
        best.rename(target)
        if tmp.exists():
            tmp.rename(best)

    return target, counts[best.name]

def patch_ini(path: Path, values: dict[tuple[str, str], str]) -> None:
    """Set key=value in an INI-like COLMAP project; section "" = keys before the first [section]."""

    lines, section, seen = path.read_text().splitlines(), "", set()
    for i, line in enumerate(lines):
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            continue
        key = line.split("=", 1)[0]
        if (section, key) in values:
            lines[i] = f"{key}={values[(section, key)]}"
            seen.add((section, key))

    for (section, key), value in values.items():
        if (section, key) in seen:
            continue
        if section == "":
            lines.insert(0, f"{key}={value}")
        else:
            idx = lines.index(f"[{section}]") + 1 if f"[{section}]" in lines else len(lines)
            if idx == len(lines) and f"[{section}]" not in lines:
                lines.append(f"[{section}]")
                idx += 1
            lines.insert(idx, f"{key}={value}")

    path.write_text("\n".join(lines) + "\n")

def _run(cmd: list[str], log_path: Path) -> float:
    """Runs one COLMAP command appending stdout/stderr to log_path; returns wall time (s)."""

    log.info("$ %s", " ".join(cmd))
    t0 = time.perf_counter()
    with open(log_path, "a") as fh:
        fh.write(f"\n$ {' '.join(cmd)}\n")
        fh.flush()
        try:
            subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, check=True)
        except subprocess.CalledProcessError:
            log.error("COLMAP failed: %s (see %s)", cmd[1], log_path)
            raise
    dt = time.perf_counter() - t0
    with open(log_path, "a") as fh:
        fh.write(f"# wall {dt:.2f}s\n")

    return dt

def export_model(scfg: dict, ws: Path, model: Path, cam_params: list[float] | None) -> None:
    """TXT next to the .bin, sparse.ply, project.ini with paths RELATIVE to exports/colmap/ (portable:
    open it with `cd exports/colmap && colmap gui --project_path project.ini`)."""

    exe, log_path = scfg["colmap_bin"], ws / "colmap.log"
    _run([exe, "model_converter", "--log_color", "0", "--input_path", str(model), "--output_path", str(model),
          "--output_type", "TXT"], log_path)
    _run([exe, "model_converter", "--log_color", "0", "--input_path", str(model),
          "--output_path", str(ws / "sparse.ply"), "--output_type", "PLY"], log_path)

    ini = ws / "project.ini"
    _run([exe, "project_generator", "--log_color", "0", "--output_path", str(ini),
          "--quality", scfg["project_quality"]], log_path)
    values = {("", "database_path"): "database.db",
              ("", "image_path"): "images",
              ("", "log_color"): "false",
              ("ImageReader", "camera_model"): scfg["camera_model"],
              ("ImageReader", "single_camera"): "true" if scfg["single_camera"] else "false"}
    if cam_params is not None:
        values[("ImageReader", "camera_params")] = ",".join(f"{p:.10g}" for p in cam_params)
    patch_ini(ini, values)
    log.info("sfm: COLMAP GUI: cd %s && %s gui --project_path project.ini", ws, exe)

def write_metrics(artifacts_dir: str, run_id, row: dict, backend_name: str = "cli") -> None:
    """Merges this module's columns into the run_id row of sfm_metrics.csv (other sfm modules add theirs)."""

    path = Path(artifacts_dir) / "sfm_metrics.csv"
    old = next((r for r in stage_io.read_csv(path) if r.get("run_id") == str(run_id)), {})
    row = {"backend": backend_name, **row}
    stage_io.upsert_csv(path, {**old, "run_id": run_id, **row}, key="run_id")
    stage_io.log_metrics(log, row)

def run(cfg: dict, run_dir: str, artifacts_dir: str) -> Path | None:
    """Returns <run_dir>/exports/colmap/sparse/0 or None when skipped/failed to reconstruct."""

    scfg = cfg["sfm"]
    if not scfg["enabled"]:
        log.info("sfm disabled")
        return None

    if backend(cfg) == "pycolmap":
        from src.sfm import pycolmap_backend
        return pycolmap_backend.run(cfg, run_dir, artifacts_dir)

    info = colmap_info(scfg["colmap_bin"])
    if info["path"] is None:
        log.warning("COLMAP binary '%s' not found: sfm skipped", scfg["colmap_bin"])
        return None

    if info["version"] is None or info["version"] < MIN_VERSION:
        log.warning("COLMAP %s at %s: need >= %d.%d (brew/conda-forge ship 4.x): sfm skipped",
                    info["version"] or "of unknown version", info["path"], *MIN_VERSION)
        return None

    scfg = {**scfg, "use_gpu": resolve_gpu(scfg["use_gpu"], info)}

    ws = Path(run_dir) / "exports" / "colmap"
    model = ws / "sparse" / "0"
    frames = list_frames(Path(cfg["extraction"]["output_dir"]))
    first = cv2.imread(str(frames[0])) if frames else None
    size = (first.shape[1], first.shape[0]) if first is not None else None
    cam_params = resolve_intrinsics(scfg, cfg["geometry"]["intrinsics"], size)
    run_id = cfg["run"]["id"]

    if not scfg["rerun"] and (model / "images.bin").exists():
        log.info("sfm: reusing existing model %s (%d images; sfm.rerun=false)", model, count_registered(model))
        if not (ws / "sparse.ply").exists() or not (ws / "project.ini").exists() or not (model / "images.txt").exists():
            try:
                export_model(scfg, ws, model, cam_params)
            except subprocess.CalledProcessError:
                log.warning("sfm: exporting the reused model failed (see %s)", ws / "colmap.log")
        return model

    if len(frames) < 2:
        log.warning("sfm: %d frames in %s, need >= 2: skipped", len(frames), cfg["extraction"]["output_dir"])
        return None

    ws.mkdir(parents=True, exist_ok=True)
    for stale in (ws / "database.db", ws / "sparse.ply", ws / "project.ini"):
        stale.unlink(missing_ok=True)
    shutil.rmtree(ws / "sparse", ignore_errors=True)
    (ws / "sparse").mkdir()
    linked = link_images(frames, ws / "images", scfg["link_images"])
    log.info("sfm: %d frames in %s as %s", len(frames), ws / "images", linked)
    log.info("sfm: %d frames, matcher=%s, mapper=%s, intrinsics=%s, gpu=%s",
             len(frames), scfg["matcher"], scfg["mapper"], "fixed" if cam_params else "estimate", scfg["use_gpu"])

    cmds = build_commands(scfg, ws, cam_params, int(cfg["run"]["seed"]))
    log_path = ws / "colmap.log"
    t0 = time.perf_counter()
    times = {}
    for step in ("feature", "match", "mapper"):
        try:
            times[f"t_{step}_s"] = round(_run(cmds[step], log_path), 2)
        except subprocess.CalledProcessError as exc:
            log.warning("sfm: COLMAP %s failed with exit code %s (see %s): sfm stopped", cmds[step][1],
                        exc.returncode, log_path)
            write_metrics(artifacts_dir, run_id, {**times, "t_total_s": round(time.perf_counter() - t0, 2),
                                                  "matcher": scfg["matcher"], "mapper": scfg["mapper"],
                                                  "failed_step": step})
            return None

    best, n_reg = pick_best_model(ws / "sparse")
    if best is None or n_reg == 0:
        log.warning("sfm: COLMAP reconstructed no model (see %s)", log_path)
        write_metrics(artifacts_dir, run_id, {**times, "t_total_s": round(time.perf_counter() - t0, 2),
                                              "matcher": scfg["matcher"], "mapper": scfg["mapper"]})
        return None

    try:
        export_model(scfg, ws, best, cam_params)
    except subprocess.CalledProcessError:
        log.warning("sfm: exporting the model failed (see %s); sparse/0 is kept", log_path)
    log.info("sfm: sparse/0 has %d/%d registered images", n_reg, len(frames))
    write_metrics(artifacts_dir, run_id, {**times, "t_total_s": round(time.perf_counter() - t0, 2),
                                          "matcher": scfg["matcher"], "mapper": scfg["mapper"]})

    return best

DEFAULTS = {"enabled": True, "rerun": False, "colmap_bin": "colmap", "camera_model": "OPENCV",
            "single_camera": True, "intrinsics": "estimate", "fixed_dist": [0.0, 0.0, 0.0, 0.0],
            "matcher": "sequential", "sequential_overlap": 10, "loop_detection": False, "vocab_tree": None,
            "mapper": "incremental", "use_gpu": "auto", "gpu_index": "0", "ba_use_gpu": False,
            "max_image_size": 1920, "max_num_features": 8192, "min_num_matches": 15, "multiple_models": True,
            "ba_global_max_num_iterations": None, "ba_global_max_refinements": None, "project_quality": "high",
            "link_images": "auto", "backend": "auto"}

if __name__ == "__main__":
    import os
    import tempfile

    import numpy as np

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ws = Path("/w")

    def flag(cmd: list[str], name: str) -> str | None:
        return cmd[cmd.index(name) + 1] if name in cmd else None

    # sequential + incremental + estimate
    assert parse_banner("COLMAP 4.2.0 (Commit Unknown on Unknown with CUDA)\nUsage:") == ((4, 2), True)
    assert parse_banner("COLMAP 3.9.1 (Commit abc on 2024 without CUDA)") == ((3, 9), False)
    assert parse_banner("garbage") == (None, False)
    assert resolve_gpu("auto", {"cuda": True}) and not resolve_gpu("auto", {"cuda": False})
    assert not resolve_gpu(True, {"cuda": False}) and resolve_gpu(True, {"cuda": True}) and not resolve_gpu(False, {"cuda": True})
    assert colmap_info("no_such_colmap_bin") == {"path": None, "version": None, "cuda": False}
    print("colmap_info(colmap):", colmap_info("colmap"))

    c = build_commands({**DEFAULTS, "use_gpu": True}, ws, None, 7)
    assert c["feature"][:2] == ["colmap", "feature_extractor"] and "--ImageReader.camera_params" not in c["feature"]
    assert flag(c["feature"], "--FeatureExtraction.gpu_index") == "0" and flag(c["feature"], "--FeatureExtraction.use_gpu") == "1"
    assert flag(c["feature"], "--FeatureExtraction.max_image_size") == "1920"
    assert flag(c["feature"], "--SiftExtraction.max_num_features") == "8192"
    assert flag(c["feature"], "--ImageReader.camera_model") == "OPENCV" and flag(c["feature"], "--ImageReader.single_camera") == "1"
    assert c["match"][1] == "sequential_matcher" and flag(c["match"], "--SequentialMatching.overlap") == "10"
    assert flag(c["match"], "--SequentialMatching.loop_detection") == "0" and flag(c["match"], "--FeatureMatching.use_gpu") == "1"
    assert c["mapper"][1] == "mapper" and flag(c["mapper"], "--output_path") == "/w/sparse"
    assert flag(c["mapper"], "--Mapper.random_seed") == "7" and "--Mapper.ba_refine_focal_length" not in c["mapper"]
    assert flag(c["mapper"], "--Mapper.ba_use_gpu") == "0"

    # loop detection without vocab tree -> disabled; with vocab tree -> enabled
    c = build_commands({**DEFAULTS, "loop_detection": True}, ws, None, 0)
    assert flag(c["match"], "--SequentialMatching.loop_detection") == "0" and "--SequentialMatching.vocab_tree_path" not in c["match"]
    c = build_commands({**DEFAULTS, "loop_detection": True, "vocab_tree": "/v.bin"}, ws, None, 0)
    assert flag(c["match"], "--SequentialMatching.loop_detection") == "1"
    assert flag(c["match"], "--SequentialMatching.vocab_tree_path") == "/v.bin"

    # exhaustive + global + fixed intrinsics, CPU
    s = {**DEFAULTS, "matcher": "exhaustive", "mapper": "global", "intrinsics": "fixed", "use_gpu": False,
         "fixed_dist": [0.1, -0.05, 0, 0]}
    params = resolve_intrinsics(s, {"fx": 1500.0, "fy": None, "cx": None, "cy": 540.5}, (1920, 1080))
    assert params == [1500.0, 1500.0, 960.0, 540.5, 0.1, -0.05, 0.0, 0.0]
    c = build_commands(s, ws, params, 1)
    assert flag(c["feature"], "--ImageReader.camera_params") == "1500,1500,960,540.5,0.1,-0.05,0,0"
    assert c["match"][1] == "exhaustive_matcher" and "--SequentialMatching.overlap" not in c["match"]
    assert flag(c["match"], "--FeatureMatching.use_gpu") == "0"
    assert c["mapper"][1] == "global_mapper" and flag(c["mapper"], "--GlobalMapper.gp_use_gpu") == "0"
    for n in ("ba_refine_focal_length", "ba_refine_principal_point", "ba_refine_extra_params"):
        assert flag(c["mapper"], f"--GlobalMapper.{n}") == "0"

    # incremental + fixed -> Mapper.ba_refine_* = 0, exposed ba_global keys
    c = build_commands({**s, "mapper": "incremental", "ba_global_max_refinements": 3}, ws, params, 1)
    assert all(flag(c["mapper"], f"--Mapper.{n}") == "0"
               for n in ("ba_refine_focal_length", "ba_refine_principal_point", "ba_refine_extra_params"))
    assert flag(c["mapper"], "--Mapper.ba_global_max_refinements") == "3"
    assert "--Mapper.ba_global_max_num_iterations" not in c["mapper"]

    # fixed with fx null -> estimate; PINHOLE -> 4 params; unsupported model -> estimate
    assert resolve_intrinsics(s, {"fx": None}, (10, 10)) is None
    assert resolve_intrinsics({**s, "camera_model": "PINHOLE"}, {"fx": 2.0}, (10, 8)) == [2.0, 2.0, 5.0, 4.0]
    assert resolve_intrinsics({**s, "camera_model": "SIMPLE_RADIAL"}, {"fx": 2.0}, (10, 8)) is None
    assert resolve_intrinsics(DEFAULTS, {"fx": 2.0}, (10, 8)) is None

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # best model picking: sparse/1 has more images -> swapped into sparse/0
        sp = tmp / "sparse"
        for name, n in (("0", 3), ("1", 9), ("2", 5)):
            (sp / name).mkdir(parents=True)
            (sp / name / "images.bin").write_bytes(struct.pack("<Q", n) + b"\0" * 16)
        best, n = pick_best_model(sp)
        assert best == sp / "0" and n == 9
        assert [count_registered(sp / k) for k in ("0", "1", "2")] == [9, 3, 5]
        txt = tmp / "t"
        txt.mkdir()
        (txt / "images.txt").write_text("# c\n1 0 0 0 1 0 0 0 1 a.png\n1 2 -1\n2 0 0 0 1 0 0 0 1 b.png\n\n")
        assert count_registered(txt) == 2 and pick_best_model(tmp / "none") == (None, 0)

        # ini patch
        ini = tmp / "p.ini"
        ini.write_text("log_color=true\ndatabase_path=\nimage_path=\n[ImageReader]\ncamera_model=SIMPLE_RADIAL\n")
        patch_ini(ini, {("", "database_path"): "/d.db", ("", "image_path"): "/i",
                        ("ImageReader", "camera_model"): "OPENCV", ("ImageReader", "camera_params"): "1,2"})
        text = ini.read_text()
        assert "database_path=/d.db\n" in text and "image_path=/i\n" in text
        assert "camera_model=OPENCV" in text and "[ImageReader]\ncamera_params=1,2" in text

        # images/: hardlinks (same inode), relative symlinks, copies; a rerun replaces the old entries
        src_frames = tmp / "src_frames"
        src_frames.mkdir()
        fr = []
        for k in range(3):
            fr.append(src_frames / f"img_{k:04d}.png")
            fr[-1].write_bytes(bytes([k]) * 10)
        for mode, expect in (("auto", "hardlink"), ("hardlink", "hardlink"), ("copy", "copy"), ("symlink", "symlink")):
            idir = tmp / "ws" / "images"
            assert link_images(fr, idir, mode) == expect and sorted(p.name for p in idir.iterdir()) == [f.name for f in fr]
            first = idir / fr[0].name
            if expect == "hardlink":
                assert first.stat().st_ino == fr[0].stat().st_ino and fr[0].stat().st_nlink > 1
            elif expect == "symlink":
                assert first.is_symlink() and not os.path.isabs(os.readlink(first)) and first.read_bytes() == fr[0].read_bytes()
            else:
                assert not first.is_symlink() and first.stat().st_ino != fr[0].stat().st_ino

        # metrics merge keeps columns written by other modules
        art = tmp / "art"
        stage_io.upsert_csv(art / "sfm_metrics.csv", {"run_id": "r1", "n_registered": 30}, key="run_id")
        write_metrics(str(art), "r1", {"t_total_s": 1.5, "matcher": "sequential"})
        rows = stage_io.read_csv(art / "sfm_metrics.csv")
        assert len(rows) == 1 and rows[0]["n_registered"] == "30" and rows[0]["t_total_s"] == "1.5"
        assert rows[0]["backend"] == "cli"

        # backend dispatch: auto = CLI when the binary is usable; explicit cli is kept
        assert backend({"sfm": {**DEFAULTS, "backend": "cli", "colmap_bin": "no_such_colmap_bin"}}) == "cli"
        py = importlib.util.find_spec("pycolmap") is not None
        assert backend({"sfm": {**DEFAULTS, "colmap_bin": "no_such_colmap_bin"}}) == ("pycolmap" if py else "cli")
        if colmap_info("colmap")["path"]:
            assert backend({"sfm": DEFAULTS}) == "cli"

        # a COLMAP that fails at feature_extractor -> WARNING, metrics written, None (main.py keeps going)
        fake = tmp / "fake_colmap"
        fake.write_text("#!/bin/sh\n[ \"$1\" = -h ] && echo 'COLMAP 4.1.0 (Commit x without CUDA)' && exit 0\nexit 3\n")
        fake.chmod(0o755)
        fdir = tmp / "ff"
        fdir.mkdir()
        for k in range(2):
            cv2.imwrite(str(fdir / f"img_{k:04d}.png"), np.zeros((8, 8, 3), np.uint8))
        fcfg = {"run": {"id": "fail", "seed": 0}, "extraction": {"output_dir": str(fdir)},
                "geometry": {"intrinsics": {"fx": None}}, "sfm": {**DEFAULTS, "colmap_bin": str(fake)}}
        assert run(fcfg, str(tmp / "frun"), str(art)) is None
        row = [r for r in stage_io.read_csv(art / "sfm_metrics.csv") if r["run_id"] == "fail"][0]
        assert row["failed_step"] == "feature", row

        # missing binary (and backend cli) -> None
        assert run({"sfm": {**DEFAULTS, "colmap_bin": "no_such_colmap_bin", "backend": "cli"}}, str(tmp), str(art)) is None

        # optional real smoke run (env SFM_SMOKE=1) on 8 frames from <repo>/frames (sequential + incremental, GPU)
        src = Path(__file__).resolve().parents[2] / "frames"
        if os.environ.get("SFM_SMOKE") == "1" and shutil.which("colmap") and len(list_frames(src)) >= 8:
            fdir = tmp / "frames"
            fdir.mkdir()
            for f in list_frames(src)[:8]:
                shutil.copy(f, fdir / f.name)
            cfg = {"run": {"id": "smoke", "seed": 0}, "extraction": {"output_dir": str(fdir)},
                   "geometry": {"intrinsics": {"fx": None, "fy": None, "cx": None, "cy": None}},
                   "sfm": dict(DEFAULTS)}
            model = run(cfg, str(tmp / "run"), str(art))
            assert model is not None and (model / "images.txt").exists()
            ws_dir = tmp / "run" / "exports" / "colmap"
            ini_text = (ws_dir / "project.ini").read_text()
            assert (ws_dir / "sparse.ply").exists() and "database_path=database.db\n" in ini_text
            assert "image_path=images\n" in ini_text
            print("smoke registered images:", count_registered(model), "/ 8")
            print("smoke metrics:", stage_io.read_csv(art / "sfm_metrics.csv")[-1])
            assert run(cfg, str(tmp / "run"), str(art)) == model  # reuse path
    print("OK")
# <<< END IMPLEMENTATION <<<

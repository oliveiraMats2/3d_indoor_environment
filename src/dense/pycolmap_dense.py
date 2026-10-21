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
# Stage: dense (method patch_match), pycolmap backend - same PatchMatch MVS as the COLMAP CLI path of
#      src/dense/patch_match.py, run in-process through the pycolmap bindings (pycolmap-cuda12).
# Goal: undistort_images + patch_match_stereo + stereo_fusion with exactly the options the CLI path passes,
#      so ws/, fused.ply and the dense_metrics.csv row are identical whichever backend ran.
# In:  <run_dir>/exports/colmap/{images,sparse/0}, the excluded names from mvs_exclude.txt.
# Out: <run_dir>/dense/patch_match/{ws/, fused.ply, colmap_dense.log}; per-step timings written into the row.
# YAML: sfm.backend (auto | pycolmap | cli; auto = CLI when sfm.colmap_bin is on PATH, else pycolmap),
#      the same dense.* keys as the CLI path.
# Rules: the CLI stays the primary path; pycolmap without CUDA (not pycolmap.has_cuda) or no GPU ->
#      WARNING and skip, like the CLI path; COLMAP's own log (stderr) goes to colmap_dense.log.

# >>> BEGIN IMPLEMENTATION >>>
"""PatchMatch stereo through pycolmap: the in-process twin of the COLMAP CLI commands of patch_match.py."""
import contextlib
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

log = logging.getLogger(__name__)

def resolve_backend(cfg: dict) -> str:
    """"pycolmap" | "cli" for sfm.backend; uses src.sfm.colmap.backend when present, else the same rule:
    cli/pycolmap explicit; auto = cli when sfm.colmap_bin is on PATH, else pycolmap when importable."""

    try:
        from src.sfm.colmap import backend
        return backend(cfg)
    except ImportError:
        pass
    sfm = cfg.get("sfm", {})  # local fallback, same rule as src.sfm.colmap.backend
    choice = sfm.get("backend", "auto")
    if choice in ("cli", "pycolmap"):
        return choice
    if shutil.which(sfm.get("colmap_bin", "colmap")):
        return "cli"
    try:
        import pycolmap  # noqa: F401
        return "pycolmap"
    except ImportError:
        return "cli"

def cuda_available() -> bool:
    """pycolmap built with CUDA and `nvidia-smi -L` lists a GPU."""

    try:
        import pycolmap
    except ImportError:
        return False
    if not pycolmap.has_cuda or not shutil.which("nvidia-smi"):
        return False
    try:
        return subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False

def build_options(dense_cfg: dict, allow_missing: bool) -> dict:
    """pycolmap option objects mirroring build_commands() of patch_match.py, key by key."""

    import pycolmap
    pm, fu = dense_cfg["patch_match"], dense_cfg["fusion"]
    size = int(dense_cfg["max_image_size"])
    undistort = pycolmap.UndistortCameraOptions()
    undistort.max_image_size = size
    pmo = pycolmap.PatchMatchOptions()
    pmo.max_image_size, pmo.gpu_index = size, str(dense_cfg["gpu_index"])
    pmo.num_iterations, pmo.window_radius = int(pm["num_iterations"]), int(pm["window_radius"])
    pmo.num_samples, pmo.geom_consistency = int(pm["num_samples"]), bool(pm["geom_consistency"])
    pmo.cache_size, pmo.allow_missing_files = float(pm["cache_size"]), bool(allow_missing)
    fuo = pycolmap.StereoFusionOptions()
    fuo.max_image_size, fuo.min_num_pixels = size, int(fu["min_num_pixels"])
    fuo.max_reproj_error, fuo.max_depth_error = float(fu["max_reproj_error"]), float(fu["max_depth_error"])
    fuo.cache_size = float(fu["cache_size"])

    return {"undistort": undistort, "num_src_images": int(pm["num_src_images"]), "patch_match": pmo,
            "fusion": fuo, "input_type": "geometric" if pmo.geom_consistency else "photometric"}

@contextlib.contextmanager
def stderr_to(log_path: Path):
    """Send fd 2 (COLMAP's glog output from the C++ side) to log_path while the block runs."""

    with open(log_path, "a") as fh:
        saved = os.dup(2)
        os.dup2(fh.fileno(), 2)
        try:
            yield fh
        finally:
            os.dup2(saved, 2)
            os.close(saved)

def timed(name: str, fn, log_path: Path) -> float:

    log.info("running pycolmap.%s", name)
    t0 = time.perf_counter()
    with stderr_to(log_path) as fh:
        fh.write(f"\n$ pycolmap.{name}\n")
        fh.flush()
        try:
            fn()
        except Exception as exc:  # pycolmap raises plain RuntimeError/ValueError from C++
            raise RuntimeError(f"pycolmap.{name} failed: {exc}, see {log_path}") from exc
    dt = time.perf_counter() - t0
    log.info("pycolmap.%s took %.1f s", name, dt)

    return dt

def run_steps(colmap_dir: Path, ws: Path, fused: Path, dense_cfg: dict, exclude: set[str], row: dict,
              log_path: Path) -> None:
    """Undistort -> rewrite stereo cfgs (mvs_exclude) -> patch_match_stereo -> stereo_fusion (PLY at fused).
    Fills row timings and reference counts; raises RuntimeError on failure, like run_cmd of the CLI path."""

    import pycolmap
    from src.dense.patch_match import rewrite_stereo_cfgs

    opt = build_options(dense_cfg, allow_missing=bool(exclude))
    row["t_undistort_s"] = round(timed("undistort_images", lambda: pycolmap.undistort_images(
        str(ws), str(colmap_dir / "sparse" / "0"), str(colmap_dir / "images"), output_type="COLMAP",
        num_patch_match_src_images=opt["num_src_images"], undistort_options=opt["undistort"]), log_path), 2)
    n_ref, n_removed = rewrite_stereo_cfgs(ws / "stereo", exclude)
    row["n_ref_images"], row["n_excluded"] = n_ref, n_removed
    log.info("MVS references: %d (excluded %d: mvs_exclude.txt + ref_step)", n_ref, n_removed)
    row["t_patch_match_s"] = round(timed("patch_match_stereo", lambda: pycolmap.patch_match_stereo(
        str(ws), workspace_format="COLMAP", options=opt["patch_match"]), log_path), 2)
    row["t_fusion_s"] = round(timed("stereo_fusion", lambda: pycolmap.stereo_fusion(
        str(fused), str(ws), workspace_format="COLMAP", input_type=opt["input_type"], options=opt["fusion"],
        output_type="PLY"), log_path), 2)
    if not fused.exists():
        raise RuntimeError(f"pycolmap.stereo_fusion wrote no {fused}")

# <<< END IMPLEMENTATION <<<

if __name__ == "__main__":
    import tempfile

    import numpy as np

    from src import stage_io
    from src.dense import patch_match

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    dense_cfg = {"enabled": True, "rerun": True, "methods": ["patch_match"], "max_image_size": 640, "gpu_index": "0",
                 "patch_match": {"num_iterations": 5, "window_radius": 5, "num_samples": 15, "num_src_images": 20,
                                 "geom_consistency": True, "cache_size": 32},
                 "fusion": {"min_num_pixels": 5, "max_reproj_error": 2.0, "max_depth_error": 0.01, "cache_size": 32}}

    assert resolve_backend({"sfm": {"backend": "cli", "colmap_bin": "colmap"}}) == "cli"
    assert resolve_backend({"sfm": {"backend": "pycolmap", "colmap_bin": "colmap"}}) == "pycolmap"
    if shutil.which("colmap"):  # binary on PATH -> CLI stays primary
        assert resolve_backend({"sfm": {"backend": "auto", "colmap_bin": "colmap"}}) == "cli"
    try:
        import pycolmap
        assert resolve_backend({"sfm": {"backend": "auto", "colmap_bin": "no_such_colmap_bin"}}) == "pycolmap"
        opt = build_options(dense_cfg, allow_missing=True)
        assert opt["undistort"].max_image_size == 640 and opt["num_src_images"] == 20
        assert opt["patch_match"].gpu_index == "0" and opt["patch_match"].allow_missing_files
        assert opt["patch_match"].geom_consistency and opt["input_type"] == "geometric"
        assert abs(opt["fusion"].max_depth_error - 0.01) < 1e-6 and opt["fusion"].min_num_pixels == 5
        photo = {**dense_cfg, "patch_match": {**dense_cfg["patch_match"], "geom_consistency": False}}
        assert build_options(photo, False)["input_type"] == "photometric"
    except ImportError:
        log.warning("pycolmap not installed: option tests skipped")

    # real smoke (PATCH_MATCH_SMOKE=1): every PATCH_MATCH_STEP-th frame (default 30; 20 frames) copied to a temp dir, pycolmap sparse,
    # dense with both backends at 640 px; prints n_dense and timings per backend
    src = Path(os.environ.get("PATCH_MATCH_FRAMES", "video_preview/frames/d1e24530"))
    if os.environ.get("PATCH_MATCH_SMOKE") == "1":
        import pycolmap
        frames = sorted(src.glob("img_*.png"))[int(os.environ.get("PATCH_MATCH_START", "0"))::int(os.environ.get("PATCH_MATCH_STEP", "30"))][:20]
        assert len(frames) >= 8, f"not enough frames in {src}"
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            col = tmp / "run" / "exports" / "colmap"
            (col / "images").mkdir(parents=True)
            for f in frames:
                shutil.copy(f, col / "images" / f.name)
            t0 = time.perf_counter()
            db = str(col / "database.db")
            reader = pycolmap.ImageReaderOptions()
            reader.camera_model = "OPENCV"
            pycolmap.extract_features(db, str(col / "images"), camera_mode=pycolmap.CameraMode.SINGLE,
                                      reader_options=reader)
            pycolmap.match_sequential(db)
            models = pycolmap.incremental_mapping(db, str(col / "images"), str(tmp / "maps"))
            best = max(models.values(), key=lambda r: r.num_reg_images())
            (col / "sparse" / "0").mkdir(parents=True)
            best.write(str(col / "sparse" / "0"))
            registered = sorted(im.name for im in best.images.values())
            print(f"smoke sparse: {len(registered)}/{len(frames)} registered, {len(best.points3D)} points, "
                  f"{time.perf_counter() - t0:.1f} s")
            (tmp / "frames").mkdir()
            (tmp / "frames" / "mvs_exclude.txt").write_text(registered[len(registered) // 2] + "\n")
            results = {}
            for be in ("cli", "pycolmap"):
                cfg = {"dense": dense_cfg, "extraction": {"output_dir": str(tmp / "frames")},
                       "sfm": {"colmap_bin": "colmap", "backend": be}}
                assert patch_match.run(cfg, str(tmp / "run"), str(tmp / f"art_{be}")) is not None
                row = stage_io.read_csv(tmp / f"art_{be}" / "dense_metrics.csv")[0]
                assert row["backend"] == be and int(row["n_dense"]) > 0 and row["n_excluded"] == "1"
                assert int(row["n_ref_images"]) == len(registered) - 1
                assert np.load(tmp / "run" / "dense" / "patch_match" / "points.npy").shape == (int(row["n_dense"]), 3)
                text = (tmp / "run" / "dense" / "patch_match" / "colmap_dense.log").read_text(errors="replace")
                views = [l for l in text.splitlines() if "Processing view" in l or "Elapsed time" in l]
                print(f"smoke {be} log ({len(views)} timing lines):", *views[:3], *views[-3:], sep="\n  ")
                results[be] = row
            n_img = len(registered)
            for be, row in results.items():
                per_img = float(row["t_s"]) / n_img
                print(f"smoke {be}: n_dense={row['n_dense']} t_s={row['t_s']} (undistort {row['t_undistort_s']}, "
                      f"patch_match {row['t_patch_match_s']}, fusion {row['t_fusion_s']}) "
                      f"= {per_img:.2f} s/img at 640 px; ~{per_img * (1600 / 640) ** 2 * 1159 / 60:.0f} min "
                      f"estimated for 1159 imgs at 1600 px")
            ratio = int(results["pycolmap"]["n_dense"]) / int(results["cli"]["n_dense"])
            print(f"smoke n_dense pycolmap/cli = {ratio:.3f}")
            assert 0.9 < ratio < 1.1

    print("OK")

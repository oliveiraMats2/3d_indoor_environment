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
# Stage: dense (method patch_match) - COLMAP PatchMatch MVS on the GPU.
# Goal: undistort the SfM model, run patch_match_stereo + stereo_fusion and export the fused cloud.
# In:  <run_dir>/exports/colmap/{images,sparse/0}, <frames_dir>/mvs_exclude.txt (names never used as
#      MVS references; they can still be source views).
# Out: <run_dir>/dense/patch_match/{ws/, fused.ply, points.npy (model units, COLMAP frame), colors.npy
#      (uint8 RGB), colmap_dense.log}; row method=patch_match in <artifacts_dir>/dense_metrics.csv.
# YAML: dense.{enabled, rerun, methods, max_image_size, gpu_index, patch_match.{num_iterations,
#      window_radius, num_samples, num_src_images, geom_consistency, cache_size},
#      fusion.{min_num_pixels, max_reproj_error, max_depth_error, cache_size}}; sfm.colmap_bin.
# Rules: no CUDA (nvidia-smi missing or colmap built without CUDA) -> WARNING and return; reuse
#      fused.ply when dense.rerun is false; every command is timed and logged.

# >>> BEGIN IMPLEMENTATION >>>
"""COLMAP PatchMatch stereo: undistortion, per-view depth maps on the GPU and depth-map fusion."""
import logging
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from src import stage_io

log = logging.getLogger(__name__)

METHOD = "patch_match"

def cuda_available(colmap: str) -> bool:

    if shutil.which("nvidia-smi") is None or shutil.which(colmap) is None:
        return False
    try:
        out = subprocess.run([colmap, "-h"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return False

    return "with CUDA" in (out.stdout + out.stderr)

def read_exclude(frames_dir) -> set[str]:
    """Stems of the frames listed in mvs_exclude.txt (one name per line, extension optional)."""

    path = Path(frames_dir) / "mvs_exclude.txt"
    if not path.exists():
        return set()

    return {Path(line.strip()).stem for line in path.read_text().splitlines() if line.strip()}

def rewrite_stereo_cfgs(stereo_dir, exclude: set[str]) -> tuple[int, int]:
    """Drop excluded references from patch-match.cfg (pairs: name line + sources line) and fusion.cfg
    (one name per line). Returns (n_references_kept, n_removed)."""

    stereo_dir = Path(stereo_dir)
    pm_path, fu_path = stereo_dir / "patch-match.cfg", stereo_dir / "fusion.cfg"
    lines = [l for l in pm_path.read_text().splitlines() if l.strip()]
    pairs = list(zip(lines[0::2], lines[1::2]))
    kept = [(name, src) for name, src in pairs if Path(name.strip()).stem not in exclude]
    pm_path.write_text("".join(f"{name}\n{src}\n" for name, src in kept))

    if fu_path.exists():
        names = [l for l in fu_path.read_text().splitlines() if l.strip()]
        fu_path.write_text("".join(f"{n}\n" for n in names if Path(n.strip()).stem not in exclude))

    return len(kept), len(pairs) - len(kept)

def build_commands(colmap: str, image_path, sparse_path, ws, out_ply, dense_cfg: dict,
                   allow_missing: bool) -> dict[str, list[str]]:

    pm, fu = dense_cfg["patch_match"], dense_cfg["fusion"]
    size, gpu = str(dense_cfg["max_image_size"]), str(dense_cfg["gpu_index"])
    geom = bool(pm["geom_consistency"])
    undistort = [colmap, "image_undistorter", "--image_path", str(image_path), "--input_path", str(sparse_path),
                 "--output_path", str(ws), "--output_type", "COLMAP", "--max_image_size", size,
                 "--num_patch_match_src_images", str(pm["num_src_images"])]
    patch_match = [colmap, "patch_match_stereo", "--workspace_path", str(ws), "--workspace_format", "COLMAP",
                   "--PatchMatchStereo.max_image_size", size,
                   "--PatchMatchStereo.gpu_index", gpu,
                   "--PatchMatchStereo.num_iterations", str(pm["num_iterations"]),
                   "--PatchMatchStereo.window_radius", str(pm["window_radius"]),
                   "--PatchMatchStereo.num_samples", str(pm["num_samples"]),
                   "--PatchMatchStereo.geom_consistency", str(int(geom)),
                   "--PatchMatchStereo.cache_size", str(pm["cache_size"]),
                   "--PatchMatchStereo.allow_missing_files", str(int(allow_missing))]
    fusion = [colmap, "stereo_fusion", "--workspace_path", str(ws), "--workspace_format", "COLMAP",
              "--input_type", "geometric" if geom else "photometric", "--output_path", str(out_ply),
              "--StereoFusion.max_image_size", size,
              "--StereoFusion.min_num_pixels", str(fu["min_num_pixels"]),
              "--StereoFusion.max_reproj_error", str(fu["max_reproj_error"]),
              "--StereoFusion.max_depth_error", str(fu["max_depth_error"]),
              "--StereoFusion.cache_size", str(fu["cache_size"])]

    return {"undistort": undistort, "patch_match": patch_match, "fusion": fusion}

def run_cmd(cmd: list[str], log_path: Path) -> float:

    log.info("running %s", " ".join(cmd[:2]))
    t0 = time.perf_counter()
    with open(log_path, "a") as fh:
        fh.write(f"\n$ {' '.join(cmd)}\n")
        fh.flush()
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    dt = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd[1]} failed (code {proc.returncode}), see {log_path}")
    log.info("%s took %.1f s", cmd[1], dt)

    return dt

def load_ply(path) -> tuple[np.ndarray, np.ndarray]:

    import open3d as o3d
    pcd = o3d.io.read_point_cloud(str(path))
    points = np.asarray(pcd.points, dtype=np.float64)
    colors = (np.clip(np.asarray(pcd.colors), 0, 1) * 255).round().astype(np.uint8) if pcd.has_colors() \
        else np.zeros((len(points), 3), np.uint8)

    return points, colors

def run(cfg: dict, run_dir: str, artifacts_dir: str) -> Path | None:

    dense = cfg["dense"]
    if not dense["enabled"] or METHOD not in dense["methods"]:
        log.info("dense patch_match disabled")
        return None

    out_dir = Path(run_dir) / "dense" / METHOD
    ws, fused, log_path = out_dir / "ws", out_dir / "fused.ply", out_dir / "colmap_dense.log"
    csv_path = Path(artifacts_dir) / "dense_metrics.csv"
    colmap_dir = Path(run_dir) / "exports" / "colmap"
    row = {"method": METHOD, "n_dense": "", "t_s": "", "max_image_size": dense["max_image_size"],
           "t_undistort_s": "", "t_patch_match_s": "", "t_fusion_s": "", "n_ref_images": "", "n_excluded": ""}

    if fused.exists() and not dense["rerun"]:
        log.info("reusing %s (dense.rerun=false)", fused)
        points, colors = load_ply(fused)
        np.save(out_dir / "points.npy", points)
        np.save(out_dir / "colors.npy", colors)
        old = [r for r in stage_io.read_csv(csv_path) if r.get("method") == METHOD]
        row = {**row, **(old[0] if old else {}), "n_dense": len(points)}
        stage_io.upsert_csv(csv_path, row, key="method")
        stage_io.log_metrics(log, row)
        return fused

    colmap = cfg.get("sfm", {}).get("colmap_bin", "colmap")
    if not cuda_available(colmap):
        log.warning("dense patch_match skipped: CUDA not available (nvidia-smi or CUDA-enabled %s missing)", colmap)
        return None
    if not (colmap_dir / "sparse" / "0").is_dir():
        log.warning("dense patch_match skipped: no sparse model in %s", colmap_dir / "sparse" / "0")
        return None

    if ws.exists():
        shutil.rmtree(ws)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path.write_text("")
    exclude = read_exclude(cfg["extraction"]["output_dir"])
    cmds = build_commands(colmap, colmap_dir / "images", colmap_dir / "sparse" / "0", ws, fused, dense,
                          allow_missing=bool(exclude))

    try:
        row["t_undistort_s"] = round(run_cmd(cmds["undistort"], log_path), 2)
        n_ref, n_removed = rewrite_stereo_cfgs(ws / "stereo", exclude)
        row["n_ref_images"], row["n_excluded"] = n_ref, n_removed
        log.info("MVS references: %d (excluded %d from mvs_exclude.txt)", n_ref, n_removed)
        row["t_patch_match_s"] = round(run_cmd(cmds["patch_match"], log_path), 2)
        row["t_fusion_s"] = round(run_cmd(cmds["fusion"], log_path), 2)
    except RuntimeError as exc:
        log.warning("dense patch_match failed: %s", exc)
        return None

    points, colors = load_ply(fused)
    np.save(out_dir / "points.npy", points)
    np.save(out_dir / "colors.npy", colors)
    row["n_dense"] = len(points)
    row["t_s"] = round(row["t_undistort_s"] + row["t_patch_match_s"] + row["t_fusion_s"], 2)
    stage_io.upsert_csv(csv_path, row, key="method")
    stage_io.log_metrics(log, row)

    return fused

# <<< END IMPLEMENTATION <<<

if __name__ == "__main__":
    import os
    import tempfile

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    dense_cfg = {"enabled": True, "rerun": False, "methods": ["patch_match"], "max_image_size": 640, "gpu_index": "0",
                 "patch_match": {"num_iterations": 5, "window_radius": 5, "num_samples": 15, "num_src_images": 20,
                                 "geom_consistency": True, "cache_size": 32},
                 "fusion": {"min_num_pixels": 5, "max_reproj_error": 2.0, "max_depth_error": 0.01, "cache_size": 32}}

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cmds = build_commands("colmap", "i", "s", "w", "f.ply", dense_cfg, allow_missing=True)
        assert cmds["undistort"][1] == "image_undistorter" and "--max_image_size" in cmds["undistort"]
        pmc = cmds["patch_match"]
        assert pmc[pmc.index("--PatchMatchStereo.gpu_index") + 1] == "0"
        assert pmc[pmc.index("--PatchMatchStereo.geom_consistency") + 1] == "1"
        assert pmc[pmc.index("--PatchMatchStereo.allow_missing_files") + 1] == "1"
        assert cmds["fusion"][cmds["fusion"].index("--input_type") + 1] == "geometric"
        photo = {**dense_cfg, "patch_match": {**dense_cfg["patch_match"], "geom_consistency": False}}
        assert build_commands("colmap", "i", "s", "w", "f", photo, False)["fusion"][7] == "photometric"

        stereo = tmp / "stereo"
        stereo.mkdir()
        names = [f"img_{i:04d}.png" for i in range(4)]
        (stereo / "patch-match.cfg").write_text("".join(f"{n}\n__auto__, 20\n" for n in names))
        (stereo / "fusion.cfg").write_text("".join(f"{n}\n" for n in names))
        (tmp / "mvs_exclude.txt").write_text("img_0001.png\nimg_0003\n")
        exclude = read_exclude(tmp)
        assert exclude == {"img_0001", "img_0003"} and read_exclude(tmp / "nope") == set()
        assert rewrite_stereo_cfgs(stereo, exclude) == (2, 2)
        assert (stereo / "patch-match.cfg").read_text() == "img_0000.png\n__auto__, 20\nimg_0002.png\n__auto__, 20\n"
        assert (stereo / "fusion.cfg").read_text() == "img_0000.png\nimg_0002.png\n"

        cfg = {"dense": {**dense_cfg, "enabled": False}, "extraction": {"output_dir": str(tmp)}}
        assert run(cfg, str(tmp / "run"), str(tmp / "art")) is None

        # reuse path: synthetic fused.ply, no COLMAP call
        out = tmp / "run" / "dense" / METHOD
        stage_io.write_ply(out / "fused.ply", np.random.default_rng(0).random((50, 3)), np.full((50, 3), 200))
        cfg["dense"]["enabled"] = True
        assert run(cfg, str(tmp / "run"), str(tmp / "art")) == out / "fused.ply"
        assert np.load(out / "points.npy").shape == (50, 3) and np.load(out / "colors.npy").dtype == np.uint8
        assert int(np.load(out / "colors.npy")[0, 0]) == 200
        assert stage_io.read_csv(tmp / "art" / "dense_metrics.csv")[0]["n_dense"] == "50"

    # optional real smoke test on ./frames (set PATCH_MATCH_SMOKE=1): sparse on every frame, dense at 640 px
    if os.environ.get("PATCH_MATCH_SMOKE") == "1" and cuda_available("colmap"):
        frames = sorted(Path("frames").glob("img_*.png"))
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            col = tmp / "run" / "exports" / "colmap"
            (col / "images").mkdir(parents=True)
            (col / "sparse").mkdir()
            for f in frames:
                (col / "images" / f.name).symlink_to(f.resolve())
            (tmp / "frames").mkdir()
            db = str(col / "database.db")
            t0 = time.perf_counter()
            for cmd in (["feature_extractor", "--database_path", db, "--image_path", str(col / "images"),
                         "--ImageReader.single_camera", "1", "--ImageReader.camera_model", "OPENCV"],
                        ["exhaustive_matcher", "--database_path", db],
                        ["mapper", "--database_path", db, "--image_path", str(col / "images"),
                         "--output_path", str(col / "sparse")]):
                run_cmd(["colmap", *cmd], tmp / "sparse.log")
            models = sorted((col / "sparse").iterdir(), key=lambda d: (d / "images.bin").stat().st_size)
            if models[-1].name != "0":  # largest model -> sparse/0, as the sfm stage guarantees
                (col / "sparse" / "0").rename(col / "sparse" / "tmp")
                models[-1].rename(col / "sparse" / "0")
            log.info("smoke sparse took %.1f s", time.perf_counter() - t0)
            txt = tmp / "txt"
            txt.mkdir()
            run_cmd(["colmap", "model_converter", "--input_path", str(col / "sparse" / "0"), "--output_path", str(txt),
                     "--output_type", "TXT"], tmp / "sparse.log")
            registered = sorted(l.split()[-1] for l in (txt / "images.txt").read_text().splitlines()
                                if l.strip().endswith(".png"))
            (tmp / "frames" / "mvs_exclude.txt").write_text(registered[len(registered) // 2] + "\n")
            cfg = {"dense": {**dense_cfg, "rerun": True}, "extraction": {"output_dir": str(tmp / "frames")},
                   "sfm": {"colmap_bin": "colmap"}}
            assert run(cfg, str(tmp / "run"), str(tmp / "art")) is not None
            row = stage_io.read_csv(tmp / "art" / "dense_metrics.csv")[0]
            print("smoke", row)
            assert int(row["n_dense"]) > 0 and row["n_excluded"] == "1"
            assert int(row["n_ref_images"]) == len(registered) - 1

    print("OK")

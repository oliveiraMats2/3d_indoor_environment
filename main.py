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

"""Single entry point: python3 main.py [<config>.yaml] (or --config <path>; default configs/default.yaml)."""
import glob, logging, os, random, shutil, sys, time
from pathlib import Path

import cv2
import numpy as np
import yaml

from src import detectores, epipolar, extrair_quadros, frame_quality, matching, relatorio
from src.constants import Config
from src.matching import metrics as matching_metrics
from src.matching import visualize as matching_visualize
from src import stage_io, videos, viewer
from src.cloud import clean, floorplan, planes
from src.dense import compare as dense_compare, openmvs, patch_match
from src.epipolar import two_view
from src.evaluation import ablation, hard_regions, loop_drift, metric_error, reprojection
from src.export import meshlab
from src.gs import gaussian_splatting
from src.mesh import close_holes, liepa, mesh, poisson
from src.sfm import colmap, convert, scale

log = logging.getLogger("main")

def stage(name, fn, *args):

    log.info("stage %s: start", name)
    t = time.perf_counter()
    out = fn(*args)
    elapsed = time.perf_counter() - t
    log.info("stage %s: done in %.1f s", name, elapsed)

    return out

def main():

    config_arg = stage_io.config_arg("Whole pipeline driven by one YAML (one YAML = one experiment).")
    path = str(stage_io.enter_repo(config_arg, os.path.dirname(os.path.abspath(__file__))))  # cwd = repo root
    config_file = open(path)
    raw = yaml.safe_load(config_file) or {}
    cfg = Config.model_validate(raw)
    config_path = Path(path)
    cfg = Config.model_validate(stage_io.resolve_paths(cfg.model_dump(), config_path))  # id, tag, <ws>/... paths
    run = cfg.run
    run_dir = os.path.join(run.output_root, run.id)
    artifacts_dir = os.path.join(run.artifacts_root, run.id)

    if os.path.exists(run_dir) and not run.overwrite and not run.cache:
        sys.exit(f"{run_dir} exists and run.overwrite is false; set overwrite: true or cache: true")

    if run.overwrite:
        shutil.rmtree(run_dir, ignore_errors=True)

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)
    random.seed(run.seed)
    np.random.seed(run.seed)
    cv2.setRNGSeed(run.seed)
    log_format = "%(asctime)s %(name)s %(levelname)s %(message)s"
    stream_handler = logging.StreamHandler(sys.stdout)
    log_path = os.path.join(run_dir, "run.log")
    file_handler = logging.FileHandler(log_path)
    handlers = [stream_handler, file_handler]
    logging.basicConfig(level=run.log_level, format=log_format, handlers=handlers)
    resolved = cfg.model_dump()
    resolved_path = os.path.join(run_dir, "config.resolved.yaml")

    with open(resolved_path, "w") as resolved_file:
        yaml.safe_dump(resolved, resolved_file, sort_keys=False)

    log.info("run %s -> %s, %s", run.id, run_dir, artifacts_dir)
    ext = cfg.extraction
    log.info("frames tag %s -> %s", ext.tag, ext.output_dir)
    os.makedirs(ext.output_dir, exist_ok=True)
    selection_csv = os.path.join(ext.output_dir, "selecao.csv")
    old_pattern = os.path.join(ext.output_dir, "img_*.*")
    old_frames = sorted(glob.glob(old_pattern))
    extract = ext.rerun or not os.path.exists(selection_csv)
    if extract:
        stale = old_frames + [selection_csv] if os.path.exists(selection_csv) else old_frames
        log.info("extraction.rerun: deleting %d old files in %s", len(stale), ext.output_dir)

        for path in stale:
            os.remove(path)

        extraction_cfg = ext.model_dump()
        video_file = stage_io.find_video(ext.video)  # ext.video keeps naming <ws> and the tag
        if video_file is None:
            sys.exit(f"extraction.video {ext.video} not found (relative paths start at the repo root {os.getcwd()})")

        extraction_cfg["video"] = str(video_file)
        frames = stage("extraction", extrair_quadros.run, extraction_cfg)
    else:
        log.info("reusing %d frames in %s (set extraction.rerun: true to extract again)", len(old_frames), ext.output_dir)
        frames = old_frames

    stage_io.check_intrinsics(cfg.geometry.intrinsics.model_dump(), frames[0] if frames else None)
    extraction_metrics = os.path.join(ext.output_dir, "extraction_metrics.csv")
    if os.path.exists(extraction_metrics):
        shutil.copy(extraction_metrics, artifacts_dir)  # extraction may be cached: copy it every run

    quality_cfg = cfg.frame_quality.model_dump()
    metrics_csv = os.path.join(artifacts_dir, "frame_quality_metrics.csv")
    kept = stage("frame_quality", frame_quality.run, quality_cfg, frames, metrics_csv)
    kept = stage_io.uniform_subsample(kept, cfg.detection.max_frames, "detection.max_frames (2D stages; COLMAP uses every frame)")
    detection_marker = os.path.join(run_dir, "detection_complete.npy")
    if run.cache and os.path.exists(detection_marker) and not extract:
        log.info("cache: reusing detection in %s", run_dir)
    else:
        detection_cfg = cfg.detection.model_dump()
        stage("detection", detectores.run, detection_cfg, kept, run_dir)

    if cfg.matching.enabled:
        matching_cfg = cfg.matching.model_dump()
        filters_cfg = cfg.filters.model_dump()
        stage("matching", matching.run, matching_cfg, filters_cfg, run_dir)

    if cfg.geometry.enabled:
        geometry_cfg = cfg.geometry.model_dump()
        stage("geometry", epipolar.run, geometry_cfg, run_dir, ext.output_dir)

    if cfg.matching.enabled and cfg.geometry.enabled:
        stage("matching_metrics", matching_metrics.run, run_dir, artifacts_dir)
        if cfg.visualize.matches.enabled:  # off by default: `samples` figures per detector
            matches_visualize_cfg = cfg.visualize.matches.model_dump()
            stage(
                "matching_visualization",
                matching_visualize.run,
                matches_visualize_cfg,
                run_dir,
                ext.output_dir,
                run.seed,
            )

    # ponytail: evaluation is still a stub; replace the raise once implemented
    stubs = (("evaluation", "src.evaluation"),)

    full_cfg = cfg.model_dump()  # resolved paths and tag; the new stages receive the whole config
    if cfg.geometry.enabled:
        stage("two_view", two_view.run, full_cfg, run_dir, ext.output_dir, artifacts_dir)

    if cfg.sfm.enabled:
        stage("sfm", colmap.run, full_cfg, run_dir, artifacts_dir)
        stage("sfm_convert", convert.run, full_cfg, run_dir, artifacts_dir)

    if cfg.scale.enabled:
        stage("scale", scale.run, full_cfg, run_dir, artifacts_dir)

    if cfg.cleanup.enabled:
        stage("cleanup_sparse", clean.run, full_cfg, run_dir, artifacts_dir, "sparse")

    if cfg.planes.enabled:
        stage("planes", planes.run, full_cfg, run_dir, artifacts_dir)

    if cfg.floorplan.enabled:
        stage("floorplan", floorplan.run, full_cfg, run_dir, artifacts_dir)

    if cfg.dense.enabled:
        dense_runs = {"patch_match": patch_match.run, "openmvs": openmvs.run}
        for method in cfg.dense.methods:
            stage(f"dense_{method}", dense_runs[method], full_cfg, run_dir, artifacts_dir)

        for method in cfg.dense.methods:
            if cfg.cleanup.enabled:
                stage(f"cleanup_{method}", clean.run, full_cfg, run_dir, artifacts_dir, method)

    if cfg.dense_compare.enabled:
        stage("dense_compare", dense_compare.run, full_cfg, run_dir, artifacts_dir)

    if cfg.gaussian_splatting.enabled:  # needs only the sfm model; GPU stages stay sequential
        stage("gaussian_splatting", gaussian_splatting.run, full_cfg, run_dir, artifacts_dir)

    if cfg.poisson.enabled:
        stage("poisson", poisson.run, full_cfg, run_dir, artifacts_dir)

    if cfg.close_holes.enabled:
        stage("close_holes", close_holes.run, full_cfg, run_dir, artifacts_dir)

    if cfg.liepa.enabled:
        stage("liepa", liepa.run, full_cfg, run_dir, artifacts_dir)

    if cfg.mesh.enabled:
        stage("mesh", mesh.run, full_cfg, run_dir, artifacts_dir)

    if cfg.export.enabled:
        stage("export", meshlab.run, full_cfg, run_dir, artifacts_dir)

    if cfg.evaluation.enabled:
        stage("reprojection", reprojection.run, full_cfg, run_dir, artifacts_dir)
        stage("metric_error", metric_error.run, full_cfg, run_dir, artifacts_dir)
        stage("loop_drift", loop_drift.run, full_cfg, run_dir, artifacts_dir)
        if cfg.evaluation.hard_regions:
            stage("hard_regions", hard_regions.run, full_cfg, run_dir, artifacts_dir)

        if cfg.evaluation.ablation_runs:
            stage("ablation", ablation.run, full_cfg, run_dir, artifacts_dir)

    if cfg.viewer.enabled:  # before videos: videos.csv then lists navigation_<cloud>_<s>s.mp4
        stage("viewer", viewer.run_safe, full_cfg, run_dir)
        if not cfg.videos.enabled:
            stage("videos_manifest", videos.write_manifest, full_cfg, run_dir, artifacts_dir)

    if cfg.videos.enabled:
        stage("videos", videos.run, full_cfg, run_dir, artifacts_dir)

    report_cfg = cfg.report.model_dump()
    stage("report", relatorio.run, report_cfg, run_dir, ext.output_dir, artifacts_dir)

if __name__ == "__main__":
    main()

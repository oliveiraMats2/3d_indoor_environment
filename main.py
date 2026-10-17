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

"""Single entry point: python main.py --config configs/default.yaml"""
import argparse, glob, logging, os, random, shutil, sys, time
from pathlib import Path

import cv2
import numpy as np
import yaml

from src import detectores, extrair_quadros, frame_quality, relatorio
from src.constants import Config

log = logging.getLogger("main")

def stage(name, fn, *args):

    log.info("stage %s: start", name)
    t = time.perf_counter()
    out = fn(*args)
    elapsed = time.perf_counter() - t
    log.info("stage %s: done in %.1f s", name, elapsed)

    return out

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    path = args.config
    config_file = open(path)
    raw = yaml.safe_load(config_file) or {}
    cfg = Config.model_validate(raw)
    run = cfg.run
    config_path = Path(path)
    run.id = run.id or config_path.stem
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
        frames = stage("extraction", extrair_quadros.run, extraction_cfg)
    else:
        log.info("reusing %d frames in %s (set extraction.rerun: true to extract again)", len(old_frames), ext.output_dir)
        frames = old_frames

    quality_cfg = cfg.frame_quality.model_dump()
    metrics_csv = os.path.join(artifacts_dir, "frame_quality_metrics.csv")
    kept = stage("frame_quality", frame_quality.run, quality_cfg, frames, metrics_csv)
    k_path = os.path.join(run_dir, "K.npy")
    if run.cache and os.path.exists(k_path) and not extract:
        log.info("cache: reusing detection in %s", run_dir)
    else:
        detection_cfg = cfg.detection.model_dump()
        intrinsics = cfg.geometry.intrinsics.model_dump()
        stage("detection", detectores.run, detection_cfg, kept, run_dir, cfg.filters.lowe_ratio, intrinsics)

    report_cfg = cfg.report.model_dump()
    stage("report", relatorio.run, report_cfg, run_dir, ext.output_dir, artifacts_dir)

    # ponytail: later stages are stubs; replace the raise with their run() once implemented
    stubs = (("matching", "src.matching"), ("geometry", "src.epipolar"), ("evaluation", "src.evaluation"))
    for block, package in stubs:

        block_cfg = getattr(cfg, block)
        if block_cfg.enabled:
            raise NotImplementedError(f"{block}.enabled is true but {package} is not implemented yet")

if __name__ == "__main__":
    main()

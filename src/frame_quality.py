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

"""Per-frame quality: sharpness, exposure, contrast and texture (no relation between frames)."""
import csv
import logging
from collections import Counter

import cv2
import numpy as np
from tqdm import tqdm

log = logging.getLogger(__name__)

# Lens distortion is not a per-frame discard criterion (calibration fixes it);
# motion blur is caught by the blur (Laplacian) criterion.

def load_gray(path, width):

    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    new_h = round(h * width / w)
    size = (width, new_h)
    resized = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)

    return resized

def laplacian_var(gray):

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()

    return float(variance)

def exposure(gray):

    mean = float(gray.mean())
    clipped_mask = (gray <= 5) | (gray >= 250)
    clipped_frac = float(clipped_mask.mean())

    return mean, clipped_frac

def fast_keypoints(gray, threshold):

    detector = cv2.FastFeatureDetector_create(threshold=threshold)
    keypoints = detector.detect(gray)

    return len(keypoints)

def measure(gray, cfg):

    mean, clipped = exposure(gray)
    lap_var = laplacian_var(gray)
    std = float(gray.std())
    fast_threshold = cfg["texture"]["fast_threshold"]
    n_keypoints = fast_keypoints(gray, fast_threshold)

    return {"laplacian_var": lap_var, "mean": mean, "clipped_frac": clipped,
            "std": std, "fast_keypoints": n_keypoints}

def reasons(m, cfg, lap_median):

    b, e = cfg["blur"], cfg["exposure"]
    checks = {"blur_abs": m["laplacian_var"] < b["laplacian_var_min"],
              "blur_rel": m["laplacian_var"] < b["relative_to_median"] * lap_median,
              "dark": m["mean"] < e["mean_min"],
              "bright": m["mean"] > e["mean_max"],
              "clipped": m["clipped_frac"] > e["clipped_max_frac"],
              "low_contrast": m["std"] < cfg["contrast"]["std_min"],
              "low_texture": m["fast_keypoints"] < cfg["texture"]["min_keypoints"]}

    return [k for k, failed in checks.items() if failed]

def write_csv(rows, csv_path):

    fieldnames = list(rows[0])

    with open(csv_path, "w", newline="") as f:

        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

def run(cfg, frame_paths, csv_path):

    metrics = []
    progress = tqdm(frame_paths, desc="frame quality", unit="frame")
    for p in progress:

        gray = load_gray(p, cfg["resize_width"])
        metric = measure(gray, cfg)
        metrics.append(metric)

    lap_values = [m["laplacian_var"] for m in metrics]
    lap_median = float(np.median(lap_values))
    rows, kept, counts = [], [], Counter()

    for p, m in zip(frame_paths, metrics):

        why = reasons(m, cfg, lap_median)
        counts.update(why)

        if not why:
            kept.append(p)

        reason = ";".join(why)
        row = {"frame": p, **m, "keep": not why, "reason": reason}
        rows.append(row)

    write_csv(rows, csv_path)
    n_kept = len(kept)
    n_total = len(frame_paths)
    counts_dict = dict(counts)
    log.info("frame_quality: kept %d/%d, discarded by reason: %s", n_kept, n_total, counts_dict)

    return kept

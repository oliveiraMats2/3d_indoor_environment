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
# Stage: 1 (frame extraction), motion-blur filter applied while the video is analysed.
# Goal: score every sampled frame for blur and directional (motion) blur and decide which to drop.
#   blur_coef    = Laplacian variance / rolling median of the Laplacian variance over `window`
#                  neighbouring sampled frames (centered, truncated at the ends) -> scene independent.
#   motion_ratio = lambda_max / lambda_min of the summed gradient structure tensor
#                  [[sum gx^2, sum gx gy], [sum gx gy, sum gy^2]]; motion blur removes gradients
#                  along the motion direction, so the tensor becomes anisotropic and the ratio rises.
#   reason: "motion" if motion_ratio > motion_ratio_max, else "blur" if blur_coef < coef_min, else "".
# In: gray frames at extraction.analysis_width (from src/extrair_quadros.py).
# Out (written by src/extrair_quadros.py with the helpers below, all in <frames_dir>):
#   discarded/{blur,motion}/<video_frame_idx>.jpg (downscaled to discard_width),
#   mvs_exclude.txt (one image file name per line, always written, may be empty).
# YAML keys (extraction.blur): enabled, coef_min, motion_ratio_max, keep_for_sfm_only, window,
#   discard_width (width of the jpg copies of dropped frames), discard_jpeg_quality.
# Rules: keep_for_sfm_only true -> flagged frames stay selectable but are listed in mvs_exclude.txt.

# >>> BEGIN IMPLEMENTATION >>>
"""Motion-blur coefficient (relative Laplacian variance) and structure-tensor anisotropy per frame."""
import logging
import shutil
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

EPS = 1e-9  # numerical guard for lambda_min and the median, not a tunable

def motion_ratio(gray: np.ndarray) -> float:
    """lambda_max / lambda_min of the summed gradient structure tensor (1 = isotropic)."""

    g = gray.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    jxx, jyy, jxy = float((gx * gx).sum()), float((gy * gy).sum()), float((gx * gy).sum())
    half_tr = (jxx + jyy) / 2
    root = np.sqrt(((jxx - jyy) / 2) ** 2 + jxy ** 2)
    lmax, lmin = half_tr + root, max(half_tr - root, 0.0)

    return float((lmax + EPS) / (lmin + EPS))

def blur_coef(lap_vars, window: int) -> np.ndarray:
    """Laplacian variance divided by the centered rolling median over `window` sampled frames."""

    v = np.asarray(lap_vars, dtype=np.float64)
    half = max(int(window), 1) // 2
    med = np.array([np.median(v[max(0, i - half):i + half + 1]) for i in range(len(v))])

    return v / np.maximum(med, EPS)

def reasons(coef: np.ndarray, ratio: np.ndarray, cfg: dict) -> list[str]:
    """Per frame: "motion", "blur" or "" (kept). All "" when the filter is disabled."""

    if not cfg["enabled"]:
        return [""] * len(coef)

    return ["motion" if r > cfg["motion_ratio_max"] else "blur" if c < cfg["coef_min"] else ""
            for c, r in zip(coef, ratio)]

def reset_discarded(frames_dir) -> None:
    """Empty discarded/blur and discarded/motion so a rerun leaves no stale copies."""

    for sub in ("blur", "motion"):
        d = Path(frames_dir) / "discarded" / sub
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)

def save_discarded(frames_dir, reason: str, frame_idx: int, frame: np.ndarray, cfg: dict) -> Path:
    """Downscaled jpg copy of a dropped frame, named by its video frame index."""

    path = Path(frames_dir) / "discarded" / reason / f"{frame_idx:06d}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    s = min(1.0, cfg["discard_width"] / frame.shape[1])
    small = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else frame
    cv2.imwrite(str(path), small, [cv2.IMWRITE_JPEG_QUALITY, cfg["discard_jpeg_quality"]])

    return path

def write_mvs_exclude(frames_dir, names: list[str]) -> Path:
    """Always written; one image file name per line (empty file when nothing is excluded)."""

    path = Path(frames_dir) / "mvs_exclude.txt"
    path.write_text("".join(f"{n}\n" for n in sorted(names)))

    return path

if __name__ == "__main__":
    import tempfile

    rng = np.random.default_rng(0)
    cfg = {"enabled": True, "coef_min": 0.5, "motion_ratio_max": 4.0, "keep_for_sfm_only": False,
           "window": 9, "discard_width": 64, "discard_jpeg_quality": 90}

    def texture(seed):
        noise = np.random.default_rng(seed).integers(0, 256, (240, 320)).astype(np.uint8)
        return cv2.GaussianBlur(noise, (3, 3), 0.8)  # natural-ish texture, not pure pixel noise

    sharp = [texture(s) for s in range(9)]
    gauss = cv2.GaussianBlur(sharp[0], (0, 0), 3.0)
    kernel = np.zeros((21, 21), np.float32)
    kernel[10, :] = 1.0 / 21  # horizontal line kernel = camera moving sideways
    motion = cv2.filter2D(sharp[1], -1, kernel)

    seq = sharp[:4] + [gauss] + sharp[4:7] + [motion] + sharp[7:]
    lap = [float(cv2.Laplacian(g, cv2.CV_64F).var()) for g in seq]
    coef = blur_coef(lap, cfg["window"])
    ratio = np.array([motion_ratio(g) for g in seq])
    why = reasons(coef, ratio, cfg)
    print("coef", np.round(coef, 3).tolist())
    print("ratio", np.round(ratio, 2).tolist())
    print("reasons", why)

    assert all(why[i] == "" for i in range(len(seq)) if i not in (4, 8)), why
    assert why[4] == "blur" and ratio[4] < cfg["motion_ratio_max"], (why[4], ratio[4])
    assert why[8] == "motion", why[8]
    assert all(r == "" for r in reasons(coef, ratio, {**cfg, "enabled": False}))
    assert motion_ratio(np.zeros((32, 32), np.uint8)) == 1.0

    with tempfile.TemporaryDirectory() as tmp:
        reset_discarded(tmp)
        p = save_discarded(tmp, "motion", 17, cv2.cvtColor(motion, cv2.COLOR_GRAY2BGR), cfg)
        assert p.name == "000017.jpg" and cv2.imread(str(p)).shape[1] == 64
        assert write_mvs_exclude(tmp, []).read_text() == ""
        assert write_mvs_exclude(tmp, ["img_0007.png"]).read_text() == "img_0007.png\n"
    print("OK")
# <<< END IMPLEMENTATION <<<

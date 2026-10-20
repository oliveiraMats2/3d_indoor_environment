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
# This block is a prompt complement for the AI assistant that will implement
# this file. Read it fully before writing any code. It is self-contained: do not
# look for other documents.
#
# Stage: 3, part 2 (Epipolar geometry)          Owner: Maxime          Due: Fri 23/10
# Project: 3D reconstruction of an indoor environment (variant A) from frames
#          extracted from a phone video.
# Context (plain words): a point in frame 1 only gives a direction, like a laser
# beam leaving camera 1 through that pixel. Seen from camera 2 the beam is a
# straight line, the epipolar line; the correct match of the point must lie on
# it. F (fundamental matrix) states this in pixels: x2^T F x1 = 0. E (essential
# matrix) is the same rule after removing the lens with the intrinsics K:
# E = K^T F K, and it keeps only the motion between the cameras.
# RANSAC: sample a few pairs (8 for F, 5 for E), fit, count how many other
# pairs fall close to their epipolar lines, repeat, keep the model with the
# most votes. Voters are inliers; the rest are discarded outliers.
# t comes out only as a direction, without size: metric scale is applied later
# from an object of known size in the scene.
# Input: the Lowe-filtered pairs from the matching package, plus K from the YAML.
# The owner also writes the second part of the Stage 3 report section.
#
# Goal of this file (single responsibility):
#   draw epipolar lines on one or two pairs.
#
# What to implement:
#   - cv2.computeCorrespondEpilines for a sample of inliers, lines drawn on the
#     other frame with the matching point on top, both frames side by side.
#   - Only for pairs whose registration was found (mask.sum() >= 8, else None).
#   - Lines from pts1 are drawn on frame 2 (with pts2 on top) and lines from
#     pts2 on frame 1 (with pts1 on top), same colour per sampled pair.
#   - Sample of n_lines inliers drawn with np.random.default_rng(seed).
#   - Figures with OpenCV only. dpi sets the output width: the side-by-side
#     canvas is treated as a FIGURE_WIDTH_IN (12 in) wide figure, so its width
#     in pixels is 12 * dpi (150 -> 1800 px); the frames are resized before
#     drawing so line thickness stays readable.
#
# Signature (called by src/epipolar/two_view.py, cfg = visualize.epilines):
#   run(cfg, run_dir, out_path, image1_bgr, image2_bgr, F, pts1, pts2, mask,
#       seed) -> Path | None
#
# Inputs / outputs:
#   In: the two frames (BGR), F (3, 3), pts1, pts2 (M, 2) all Lowe matches,
#       mask (M,) bool inliers.
#   Out: results/<run_id>/figures/epilines/<detector>/<A>__<B>.png
#
# Defaults (read from the YAML block visualize.epilines, never hardcoded):
#   n_lines: 20
#   dpi: 150
#
# Rules:
#   - Write code ONLY between the BEGIN and END markers below. Do not create
#     other regions or other files for this task.
#   - Configuration comes from the YAML loaded by main.py. No argparse.
#   - Save every partial result as .npy under results/<run_id>/<stage>/.
#   - Keep this block after implementing; update it if the behavior changes.
#   - Expose a function run(cfg: dict, run_dir: str, ...) that main.py can call
#     when the YAML block of this stage has enabled: true (default false).
#   - Pipeline already available: frames are extracted and quality-filtered, and
#     SIFT, ORB and AKAZE keypoints/descriptors are saved per frame under
#     results/<run_id>/detection/<detector>/<frame>/ as keypoints.npy (N, 6:
#     x, y, size, angle, response, octave) and descriptors.npy (N, D).
# ==============================================================================

# >>> BEGIN IMPLEMENTATION >>>
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

MIN_INLIERS = 8
FIGURE_WIDTH_IN = 12.0


def _colors(count: int, rng: np.random.Generator) -> list[tuple[int, int, int]]:
    hues = rng.permutation(np.linspace(0, 179, max(count, 1), endpoint=False).astype(np.uint8))[:count]
    hsv = np.stack([hues, np.full(count, 255, np.uint8), np.full(count, 230, np.uint8)], axis=1)
    bgr = cv2.cvtColor(hsv.reshape(-1, 1, 3), cv2.COLOR_HSV2BGR).reshape(-1, 3)
    return [tuple(int(v) for v in c) for c in bgr]


def _draw(image: np.ndarray, lines: np.ndarray, points: np.ndarray, colors: list, thickness: int) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    span = 2.0 * (h + w)
    for (a, b, c), (x, y), color in zip(lines, points, colors):
        norm = np.hypot(a, b)
        if norm < 1e-12:
            continue
        x0, y0 = -a * c / norm**2, -b * c / norm**2  # closest point of the line to the origin
        dx, dy = -b / norm, a / norm
        ok, p1, p2 = cv2.clipLine(
            (0, 0, w, h),
            (int(round(x0 - span * dx)), int(round(y0 - span * dy))),
            (int(round(x0 + span * dx)), int(round(y0 + span * dy))),
        )
        if ok:
            cv2.line(out, p1, p2, color, thickness, cv2.LINE_AA)
        center = (int(round(x)), int(round(y)))
        cv2.circle(out, center, 3 * thickness + 2, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(out, center, 3 * thickness, color, -1, cv2.LINE_AA)
    return out


def run(
    cfg: dict,
    run_dir: str,
    out_path: Path,
    image1_bgr: np.ndarray,
    image2_bgr: np.ndarray,
    F: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    mask: np.ndarray,
    seed: int,
) -> Path | None:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    F = np.asarray(F, dtype=np.float64)
    if mask.sum() < MIN_INLIERS or not np.isfinite(F).all():
        log.info("epilines skipped for %s: %d inliers", Path(out_path).stem, int(mask.sum()))
        return None
    rng = np.random.default_rng(seed)
    inliers = np.flatnonzero(mask)
    chosen = np.sort(rng.choice(inliers, size=min(int(cfg["n_lines"]), inliers.size), replace=False))
    p1 = np.asarray(pts1, dtype=np.float64)[chosen]
    p2 = np.asarray(pts2, dtype=np.float64)[chosen]

    height = max(image1_bgr.shape[0], image2_bgr.shape[0])
    native_width = image1_bgr.shape[1] * height / image1_bgr.shape[0] + image2_bgr.shape[1] * height / image2_bgr.shape[0]
    zoom = FIGURE_WIDTH_IN * float(cfg["dpi"]) / native_width
    scale1 = zoom * height / image1_bgr.shape[0]
    scale2 = zoom * height / image2_bgr.shape[0]
    target_h = max(1, int(round(height * zoom)))

    def resize(image: np.ndarray, s: float) -> np.ndarray:
        size = (max(1, int(round(image.shape[1] * s))), target_h)
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)

    img1, img2 = resize(image1_bgr, scale1), resize(image2_bgr, scale2)
    if img1.ndim == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if img2.ndim == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    # a uniform resize by s maps line (a, b, c) to (a, b, c * s)
    lines_on_2 = cv2.computeCorrespondEpilines(p1.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
    lines_on_1 = cv2.computeCorrespondEpilines(p2.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
    lines_on_2[:, 2] *= scale2
    lines_on_1[:, 2] *= scale1
    colors = _colors(len(chosen), rng)
    thickness = max(1, int(round(target_h / 600)))
    left = _draw(img1, lines_on_1, p1 * scale1, colors, thickness)
    right = _draw(img2, lines_on_2, p2 * scale2, colors, thickness)
    canvas = np.hstack([left, right])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    log.info("epilines %s: %d lines -> %s", out_path.stem, len(chosen), out_path)
    return out_path


if __name__ == "__main__":
    import tempfile

    logging.basicConfig(level=logging.INFO)
    rng = np.random.default_rng(0)
    K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
    R, _ = cv2.Rodrigues(np.array([0.0, 0.1, 0.0]))
    t = np.array([1.0, 0.0, 0.1])
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    F_true = np.linalg.inv(K).T @ tx @ R @ np.linalg.inv(K)
    X = np.column_stack([rng.uniform(-2, 2, 60), rng.uniform(-1.5, 1.5, 60), rng.uniform(4, 8, 60)])
    x1 = (K @ X.T).T
    x2 = (K @ (R @ X.T + t[:, None])).T
    pts1, pts2 = x1[:, :2] / x1[:, 2:], x2[:, :2] / x2[:, 2:]
    l2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F_true).reshape(-1, 3)
    assert np.abs(np.sum(l2[:, :2] * pts2, axis=1) + l2[:, 2]).max() < 1e-6
    img1 = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    img2 = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    cfg = {"n_lines": 20, "dpi": 150}
    mask = np.ones(len(pts1), bool)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "figures" / "epilines" / "akaze" / "img_0000__img_0001.png"
        res = run(cfg, tmp, out, img1, img2, F_true, pts1, pts2, mask, seed=0)
        assert res == out and out.is_file()
        fig = cv2.imread(str(out))
        assert fig.shape[1] == int(FIGURE_WIDTH_IN * cfg["dpi"]), fig.shape
        few = np.zeros(len(pts1), bool)
        few[:7] = True
        assert run(cfg, tmp, Path(tmp) / "few.png", img1, img2, F_true, pts1, pts2, few, seed=0) is None
        assert not (Path(tmp) / "few.png").exists()
        assert run(cfg, tmp, Path(tmp) / "nan.png", img1, img2, np.full((3, 3), np.nan), pts1, pts2, mask, 0) is None

        root = Path(__file__).resolve().parents[2]
        pair = root / "results" / "default" / "epipolar" / "akaze" / "img_0000__img_0001"
        match = root / "results" / "default" / "matching" / "akaze" / "img_0000__img_0001" / "matches_lowe.npy"
        if (pair / "F.npy").is_file() and match.is_file() and (root / "frames" / "img_0000.png").is_file():
            m = np.load(match).astype(int)
            k1 = np.load(root / "results" / "default" / "detection" / "akaze" / "img_0000" / "keypoints.npy")
            k2 = np.load(root / "results" / "default" / "detection" / "akaze" / "img_0001" / "keypoints.npy")
            real = run(cfg, tmp, Path(tmp) / "real.png", cv2.imread(str(root / "frames" / "img_0000.png")),
                       cv2.imread(str(root / "frames" / "img_0001.png")), np.load(pair / "F.npy"),
                       k1[m[:, 0], :2], k2[m[:, 1], :2], np.load(pair / "inlier_mask.npy"), seed=0)
            print("real smoke test:", real is not None and real.is_file())
    print("OK")
# <<< END IMPLEMENTATION <<<

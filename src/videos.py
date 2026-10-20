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

# ==============================================================================
# PROMPT COMPLEMENT
# Stage: videos (near the end of main.py, after evaluation, before the report).
# Goal: two diagnostic videos plus the manifest of every video of the run.
#   - trajectory_topdown.mp4 (topdown_fps): camera centers (scale/centers_cm.npy)
#     moved to the floor frame (planes/T_floor.npy) and drawn over
#     figures/floorplan.png, one output frame per camera (past path, current
#     position, viewing direction when sfm/cam_R.npy is aligned). The pixel
#     mapping is rebuilt from floorplan/{grid,grid_meta}.npy and
#     floorplan.{dpi,fig_inches} exactly as src/cloud/floorplan.py renders; if
#     that fails (missing files, size mismatch) a blank canvas with the
#     trajectory's own bounds is used (WARNING).
#   - frames_kept_vs_discarded.mp4 (kept_vs_discarded): the original video
#     downscaled to `width` with a top banner: green "KEPT img_xxxx", red
#     "DROPPED: blur|motion|quality", grey "not sampled" before the first
#     sampled frame. A sampled frame holds its banner until the next one.
#     Sources: <frames_dir>/selecao.csv (imagem, frame_video, tempo_s,
#     blur_coef, motion_ratio, kept; dropped rows have imagem "" and kept 0),
#     <frames_dir>/discarded/{blur,motion,quality}/* (stem = video frame index
#     or contains img_xxxx) and artifacts/<id>/frame_quality_metrics.csv
#     (frame, ..., keep, reason). Dropped overrides kept.
#   - videos.csv: one row per <run_dir>/videos/*.mp4 (file, cloud, seconds,
#     fps, resolution, run_id), read with cv2.VideoCapture; cloud parsed from
#     navigation_<cloud>_<s>s.mp4. This stage is the ONLY writer of it.
#     write_manifest(cfg, run_dir, artifacts_dir) refreshes it alone and is
#     called by main.py after the viewer stage when videos.enabled is false.
# In:  figures/floorplan.png, floorplan/{grid,grid_meta}.npy, scale/centers_cm.npy,
#      planes/T_floor.npy, sfm/{cam_R,image_names}.npy, frames dir, video.
# Out: videos/{trajectory_topdown,frames_kept_vs_discarded}.mp4, artifacts/<id>/videos.csv.
# YAML: videos: {enabled: true, topdown_fps: 10, kept_vs_discarded: true, width: 960, codec: auto}
# Rules: missing inputs -> WARNING and skip that video, never raise; OpenCV only.
# ==============================================================================

# >>> BEGIN IMPLEMENTATION >>>
"""Diagnostic videos (top-down trajectory, kept vs discarded frames) and the videos.csv manifest."""
import csv
import logging
import re
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src import stage_io
from src.extraction import video

log = logging.getLogger(__name__)

MANIFEST_COLUMNS = ["file", "cloud", "seconds", "fps", "resolution", "run_id"]
COLORS = {"kept": (40, 160, 40), "dropped": (40, 40, 200), "none": (110, 110, 110)}
NAV_RE = re.compile(r"^navigation_(.+)_(\d+(?:\.\d+)?)s$")
IMG_RE = re.compile(r"img_\d+")

def even(n: int) -> int:

    return max(2, int(n) // 2 * 2)

AUTO_CODECS = ("avc1", "mp4v")  # H.264 when the OpenCV build has an encoder (macOS AVFoundation), else MPEG-4

def open_writer(path: Path, fps: float, size: tuple[int, int], codec: str) -> cv2.VideoWriter | None:
    """First codec whose writer opens (codec "auto" tries AUTO_CODECS in order); None (WARNING) if none does."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for fourcc in (AUTO_CODECS if codec == "auto" else (codec,)):
        level = cv2.utils.logging.getLogLevel()
        if codec == "auto":  # a missing H.264 encoder is expected on Linux wheels: probe it silently
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
        try:
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), float(fps), size)
        finally:
            cv2.utils.logging.setLogLevel(level)
        if writer.isOpened():
            log.info("videos: %s with codec %s", path.name, fourcc)
            return writer
        writer.release()

    log.warning("videos: cannot open writer %s (codec %s)", path, codec)

    return None

# ---------------------------------------------------------------- top-down trajectory
def floorplan_mapping(cfg: dict, run_dir: Path, image: np.ndarray | None):
    """to_px(xy) of src/cloud/floorplan.render, or None when it cannot be rebuilt for this image."""

    fp = cfg.get("floorplan", {})
    grid_p, meta_p = run_dir / "floorplan" / "grid.npy", run_dir / "floorplan" / "grid_meta.npy"
    if image is None or not grid_p.exists() or not meta_p.exists() or "dpi" not in fp or "fig_inches" not in fp:
        return None

    shape = np.load(grid_p).shape
    x0, y0, cell = np.load(meta_p).astype(np.float64)
    px = max(int(fp["fig_inches"] * fp["dpi"]) / max(shape), 1.0)
    h, w = int(round(shape[0] * px)), int(round(shape[1] * px))
    margin = int(0.5 * fp["dpi"])
    if image.shape[:2] != (h + 2 * margin, w + 2 * margin):
        log.warning("videos: floorplan.png %s does not match grid mapping %s", image.shape[:2],
                    (h + 2 * margin, w + 2 * margin))
        return None

    def to_px(xy: np.ndarray) -> np.ndarray:
        return np.stack([margin + (xy[:, 0] - x0) / cell * px, margin + h - (xy[:, 1] - y0) / cell * px], 1)

    return to_px

def blank_mapping(xy: np.ndarray, width: int):

    lo, hi = xy.min(0), xy.max(0)
    span = np.maximum(hi - lo, 1e-9)
    lo, span = lo - 0.1 * span, 1.2 * span
    ppu = width / span.max()
    size = (int(np.ceil(span[0] * ppu)) + 1, int(np.ceil(span[1] * ppu)) + 1)
    canvas = np.full((size[1], size[0], 3), 40, np.uint8)

    def to_px(p: np.ndarray) -> np.ndarray:
        return np.stack([(p[:, 0] - lo[0]) * ppu, size[1] - (p[:, 1] - lo[1]) * ppu], 1)

    return canvas, to_px

def topdown(cfg: dict, run_dir: Path, out_path: Path) -> Path | None:

    vc = cfg["videos"]
    centers_p, T_p = run_dir / "scale" / "centers_cm.npy", run_dir / "planes" / "T_floor.npy"
    if not centers_p.exists():
        log.warning("videos: %s missing, trajectory_topdown skipped", centers_p)
        return None
    if T_p.exists():
        T = np.load(T_p).astype(np.float64)
    else:
        log.warning("videos: %s missing, identity used for the floor frame", T_p)
        T = np.eye(4)

    centers = np.load(centers_p).astype(np.float64).reshape(-1, 3)
    if len(centers) == 0:
        log.warning("videos: no camera centers, trajectory_topdown skipped")
        return None
    xy = (centers @ T[:3, :3].T + T[:3, 3])[:, :2]

    names_p, R_p = run_dir / "sfm" / "image_names.npy", run_dir / "sfm" / "cam_R.npy"
    names = [str(n) for n in np.load(names_p)] if names_p.exists() else []
    names = names if len(names) == len(xy) else [f"cam {i}" for i in range(len(xy))]
    heading = None
    if R_p.exists():
        R = np.load(R_p).astype(np.float64)
        if len(R) == len(xy):
            fwd = np.einsum("nji,j->ni", R, np.array([0.0, 0.0, 1.0])) @ T[:3, :3].T   # R^T z, floor frame
            heading = fwd[:, :2]

    plan_p = run_dir / "figures" / "floorplan.png"
    image = cv2.imread(str(plan_p)) if plan_p.exists() else None
    to_px = floorplan_mapping(cfg, run_dir, image)
    if to_px is None:
        log.warning("videos: no usable floorplan mapping (%s), drawing on a blank canvas with the trajectory bounds",
                    plan_p)
        image, to_px = blank_mapping(xy, vc["width"])

    s = vc["width"] / image.shape[1]
    size = (even(vc["width"]), even(round(image.shape[0] * s)))
    base = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    pts = to_px(xy) * np.array([size[0] / image.shape[1], size[1] / image.shape[0]])
    pts_i = np.round(pts).astype(np.int32)
    th, r = max(1, size[0] // 400), max(3, size[0] // 120)
    fs = size[0] / 1600
    cv2.polylines(base, [pts_i], False, (160, 160, 160), th, cv2.LINE_AA)

    writer = open_writer(out_path, vc["topdown_fps"], size, vc["codec"])
    if writer is None:
        return None
    for i in tqdm(range(len(pts_i)), desc="trajectory video", unit="frame"):
        frame = base.copy()
        cv2.polylines(frame, [pts_i[:i + 1]], False, (0, 255, 255), 2 * th, cv2.LINE_AA)
        c = tuple(int(v) for v in pts_i[i])
        if heading is not None and np.linalg.norm(heading[i]) > 1e-9:
            d = heading[i] / np.linalg.norm(heading[i]) * 4 * r
            tip = (int(c[0] + d[0]), int(c[1] - d[1]))
            cv2.arrowedLine(frame, c, tip, (255, 128, 0), th + 1, cv2.LINE_AA, tipLength=0.35)
        cv2.circle(frame, c, r, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(frame, f"{names[i]}  {i + 1}/{len(pts_i)}", (10, int(30 * fs) + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 3 * th, cv2.LINE_AA)
        cv2.putText(frame, f"{names[i]}  {i + 1}/{len(pts_i)}", (10, int(30 * fs) + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th, cv2.LINE_AA)
        writer.write(frame)
    writer.release()
    log.info("videos: %s (%d cameras)", out_path, len(pts_i))

    return out_path

# ---------------------------------------------------------------- kept vs discarded
def to_int(value) -> int | None:

    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None

def to_float(value) -> float | None:

    try:
        v = float(value)
    except (TypeError, ValueError):
        return None

    return v if np.isfinite(v) else None

def parse_discarded(path: Path, image_to_frame: dict[str, int]) -> int | None:
    """Video frame index from a discarded file name: an img_xxxx inside it, else its last integer."""

    stem = path.stem
    m = IMG_RE.search(stem)
    if m:
        return image_to_frame.get(m.group(0))
    digits = re.findall(r"\d+", stem)

    return int(digits[-1]) if digits else None

def frame_events(cfg: dict, artifacts_dir: Path) -> dict[int, tuple[str, str]]:
    """video frame index -> (status, banner text) for every sampled frame."""

    frames_dir = Path(cfg["extraction"]["output_dir"])
    rows = stage_io.read_csv(frames_dir / "selecao.csv")
    if not rows:
        log.warning("videos: %s missing or empty, every frame will read 'not sampled'", frames_dir / "selecao.csv")
    blur_cfg = cfg["extraction"].get("blur", {}) if isinstance(cfg["extraction"].get("blur"), dict) else {}

    events, image_to_frame, pending = {}, {}, {}
    for row in rows:
        f = to_int(row.get("frame_video"))
        if f is None:
            continue
        image = Path(row.get("imagem") or "").stem
        kept = to_int(row.get("kept"))
        kept = bool(image) if kept is None else bool(kept)
        if image:
            image_to_frame[image] = f
        if kept:
            events[f] = ("kept", f"KEPT {image}")
            continue
        coef, ratio = to_float(row.get("blur_coef")), to_float(row.get("motion_ratio"))
        reason = ""
        if ratio is not None and "motion_ratio_max" in blur_cfg and ratio > blur_cfg["motion_ratio_max"]:
            reason = "motion"
        elif coef is not None and "coef_min" in blur_cfg and coef < blur_cfg["coef_min"]:
            reason = "blur"
        pending[f] = reason

    for sub in ("blur", "motion", "quality"):
        d = frames_dir / "discarded" / sub
        for p in sorted(d.glob("*")) if d.is_dir() else []:
            f = parse_discarded(p, image_to_frame)
            if f is not None:
                pending[f] = sub
            else:
                log.warning("videos: cannot map discarded file %s to a video frame", p)

    for row in stage_io.read_csv(artifacts_dir / "frame_quality_metrics.csv"):
        if str(row.get("keep", "")).strip().lower() in ("false", "0"):
            f = image_to_frame.get(Path(row.get("frame", "")).stem)
            if f is not None:
                pending[f] = "quality" + (f" ({row['reason']})" if row.get("reason") else "")

    for f, reason in pending.items():
        events[f] = ("dropped", f"DROPPED: {reason or 'unknown'}")

    return events

def video_path(cfg: dict) -> Path:

    p = Path(cfg["extraction"]["video"])  # main.py runs from the repo root: relative paths start there

    return stage_io.find_video(p) or p

def kept_vs_discarded(cfg: dict, artifacts_dir: Path, out_path: Path) -> Path | None:

    vc = cfg["videos"]
    src = video_path(cfg)
    cap = video.open_capture(src)  # FFmpeg backend, same decoding as the extraction
    if not cap.isOpened():
        log.warning("videos: cannot open %s, frames_kept_vs_discarded skipped", src)
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    events = frame_events(cfg, artifacts_dir)
    counts = {s: sum(1 for e in events.values() if e[0] == s) for s in ("kept", "dropped")}
    log.info("videos: %d kept and %d dropped sampled frames over %d video frames", counts["kept"],
             counts["dropped"], total)

    writer, status, text, i = None, "none", "not sampled", 0
    for i in tqdm(range(total), desc="kept vs discarded video", unit="frame"):
        ok, frame = cap.read()
        if not ok:
            break
        if writer is None:
            size = (even(vc["width"]), even(round(frame.shape[0] * vc["width"] / frame.shape[1])))
            banner_h = max(16, size[1] // 12)
            fs = banner_h / 40
            th = max(1, banner_h // 20)
            writer = open_writer(out_path, fps, size, vc["codec"])
            if writer is None:
                cap.release()
                return None
        status, text = events.get(i, (status, text))
        small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        small[:banner_h] = COLORS[status]
        y = int(banner_h * 0.7)
        cv2.putText(small, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th, cv2.LINE_AA)
        stamp = f"#{i}  {i / fps:.2f}s"
        (tw, _), _ = cv2.getTextSize(stamp, cv2.FONT_HERSHEY_SIMPLEX, fs * 0.7, th)
        cv2.putText(small, stamp, (size[0] - tw - 8, y), cv2.FONT_HERSHEY_SIMPLEX, fs * 0.7, (255, 255, 255), th,
                    cv2.LINE_AA)
        writer.write(small)
    cap.release()
    if writer is None:
        log.warning("videos: no frame read from %s", src)
        return None
    writer.release()
    log.info("videos: %s (%d frames)", out_path, i + 1)

    return out_path

# ---------------------------------------------------------------- manifest
def nav_cloud(path: Path) -> str:

    m = NAV_RE.match(path.stem)

    return m.group(1) if m else ""

def write_manifest(cfg: dict, run_dir: str, artifacts_dir: str) -> list[dict]:
    """Rewrite artifacts/<id>/videos.csv from <run_dir>/videos/*.mp4. Safe to call again after
    the viewer stage of main.py records navigation_<cloud>_<s>s.mp4."""

    rows = []
    for p in sorted((Path(run_dir) / "videos").glob("*.mp4")):
        cap = cv2.VideoCapture(str(p))
        fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if fps <= 0:
            log.warning("videos: cannot read %s", p)
        seconds = round(n / fps, 3) if fps > 0 else ""
        rows.append({"file": p.name, "cloud": nav_cloud(p), "seconds": seconds,
                     "fps": round(fps, 3) if fps > 0 else "", "resolution": f"{w}x{h}" if w and h else "",
                     "run_id": cfg["run"]["id"]})
        log.info("metric %s=%s", f"video_{p.stem}_seconds", seconds)

    out = Path(artifacts_dir) / "videos.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    log.info("videos: %d videos listed in %s", len(rows), out)

    return rows

def run(cfg: dict, run_dir: str, artifacts_dir: str) -> list[dict] | None:

    vc = cfg["videos"]
    if not vc["enabled"]:
        log.info("videos disabled")
        return None

    run_dir, artifacts_dir = Path(run_dir), Path(artifacts_dir)
    out = run_dir / "videos"
    topdown(cfg, run_dir, out / "trajectory_topdown.mp4")
    if vc["kept_vs_discarded"]:
        kept_vs_discarded(cfg, artifacts_dir, out / "frames_kept_vs_discarded.mp4")

    return write_manifest(cfg, str(run_dir), str(artifacts_dir))

if __name__ == "__main__":
    import tempfile

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    assert nav_cloud(Path("navigation_patch_match_45s.mp4")) == "patch_match"
    assert nav_cloud(Path("trajectory_topdown.mp4")) == ""
    assert parse_discarded(Path("000045.jpg"), {}) == 45
    assert parse_discarded(Path("q_img_0002.png"), {"img_0002": 60}) == 60

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        run_dir, art, frames = tmp / "results/t", tmp / "artifacts/t", tmp / "frames"
        cfg = {"run": {"id": "t", "seed": 0},
               "extraction": {"video": str(tmp / "v.mp4"), "output_dir": str(frames),
                              "blur": {"coef_min": 0.5, "motion_ratio_max": 4.0}},
               "floorplan": {"dpi": 50, "fig_inches": 4.0},
               "videos": {"enabled": True, "topdown_fps": 10, "kept_vs_discarded": True, "width": 160,
                          "codec": "auto"}}

        # floor plan exactly as src/cloud/floorplan.render sizes it: grid (40, 60), dpi 50 -> 250x183 png
        for d in ("floorplan", "figures", "scale", "planes", "sfm"):
            (run_dir / d).mkdir(parents=True)
        np.save(run_dir / "floorplan/grid.npy", np.zeros((40, 60), np.int32))
        np.save(run_dir / "floorplan/grid_meta.npy", np.array([-300.0, -200.0, 10.0]))
        cv2.imwrite(str(run_dir / "figures/floorplan.png"), np.full((183, 250, 3), 60, np.uint8))
        a = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        np.save(run_dir / "scale/centers_cm.npy", np.stack([200 * np.cos(a), 150 * np.sin(a), np.full(12, 140.0)], 1))
        np.save(run_dir / "planes/T_floor.npy", np.eye(4))
        np.save(run_dir / "sfm/cam_R.npy", np.repeat(np.eye(3)[None], 12, 0))
        np.save(run_dir / "sfm/image_names.npy", np.array([f"img_{i:04d}" for i in range(12)]))

        # 3 s synthetic video, selecao.csv with kept + dropped rows, a discarded file and a quality csv
        vw = cv2.VideoWriter(str(tmp / "v.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 180))
        for i in range(90):
            vw.write(np.full((180, 320, 3), (i * 2) % 255, np.uint8))
        vw.release()
        (frames / "discarded/blur").mkdir(parents=True)
        cv2.imwrite(str(frames / "discarded/blur/000045.jpg"), np.zeros((9, 16, 3), np.uint8))
        with open(frames / "selecao.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["imagem", "frame_video", "tempo_s", "blur_coef", "motion_ratio", "kept"])
            w.writerows([["img_0000.png", 0, 0.0, 1.0, 1.2, 1], ["", 15, 0.5, 1.0, 9.0, 0],
                         ["img_0001.png", 30, 1.0, 1.0, 1.1, 1], ["", 45, 1.5, 0.2, 1.0, 0],
                         ["img_0002.png", 60, 2.0, 1.0, 1.0, 1]])
        art.mkdir(parents=True)
        with open(art / "frame_quality_metrics.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["frame", "laplacian_var", "keep", "reason"])
            w.writerows([[str(frames / "img_0000.png"), 500, True, ""], [str(frames / "img_0001.png"), 5, False, "blur_abs"]])

        ev = frame_events(cfg, art)
        assert ev[0][0] == "kept" and ev[15] == ("dropped", "DROPPED: motion") and ev[45] == ("dropped", "DROPPED: blur")
        assert ev[30] == ("dropped", "DROPPED: quality (blur_abs)") and ev[60] == ("kept", "KEPT img_0002"), ev

        rows = run(cfg, str(run_dir), str(art))
        assert (run_dir / "videos/trajectory_topdown.mp4").exists()
        assert (run_dir / "videos/frames_kept_vs_discarded.mp4").exists()
        by = {r["file"]: r for r in stage_io.read_csv(art / "videos.csv")}
        assert len(by) == 2 and len(rows) == 2, by
        kv, td = by["frames_kept_vs_discarded.mp4"], by["trajectory_topdown.mp4"]
        assert float(kv["fps"]) == 30.0 and kv["resolution"] == "160x90" and abs(float(kv["seconds"]) - 3.0) < 0.05, kv
        assert float(td["fps"]) == 10.0 and td["resolution"] == "160x116" and abs(float(td["seconds"]) - 1.2) < 0.05, td
        assert td["cloud"] == "" and td["run_id"] == "t"

        cap = cv2.VideoCapture(str(run_dir / "videos/frames_kept_vs_discarded.mp4"))
        banners = []
        for i in range(90):
            ok, f = cap.read()
            banners.append(f[0:2, 40:60].reshape(-1, 3).mean(0))
        cap.release()
        green = lambda b: b[1] > b[2] + 50
        red = lambda b: b[2] > b[1] + 50
        assert green(banners[0]) and green(banners[14]) and red(banners[15]) and red(banners[40]), banners[:16]
        assert red(banners[30]) and green(banners[89]), (banners[30], banners[89])

        cfg["videos"]["enabled"] = False
        assert run(cfg, str(run_dir), str(art)) is None
    print("OK")
# <<< END IMPLEMENTATION <<<

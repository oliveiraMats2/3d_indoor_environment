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

"""Shared I/O for the pipeline stages: CSV upsert, metric logging, PLY writing, workspace and scale."""
import csv, hashlib, json, logging, os
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

def read_csv(path) -> list[dict]:

    path = Path(path)
    if not path.exists():
        return []

    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))

def upsert_csv(path, row: dict, key: str | None = None) -> None:
    """key None: the file holds only this row; else rows with the same row[key] are replaced."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [] if key is None else [r for r in read_csv(path) if r.get(key) != str(row[key])]
    rows.append({k: "" if v is None else v for k, v in row.items()})
    fieldnames = list(dict.fromkeys(k for r in rows for k in r))

    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, restval="")
        w.writeheader()
        w.writerows(rows)

def log_metrics(logger: logging.Logger, row: dict) -> None:

    for name, value in row.items():
        logger.info("metric %s=%s", name, value)

def uniform_subsample(items: list, n: int | None, what: str = "max_frames") -> list:
    """Uniform subsample keeping the first and last item; n None or >= len -> all items (logged)."""

    items = list(items)
    if n is not None and len(items) > n:
        pos = np.unique(np.round(np.linspace(0, len(items) - 1, n)).astype(int))
        items_out = [items[i] for i in pos]
    else:
        items_out = items

    log.info("%s: %d of %d items", what, len(items_out), len(items))

    return items_out

def write_ply(path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    """Binary little-endian PLY with float xyz and optional uint8 rgb."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if colors is not None:
        fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]

    data = np.empty(len(points), dtype=fields)
    data["x"], data["y"], data["z"] = points.T
    if colors is not None:
        colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
        data["red"], data["green"], data["blue"] = colors.T

    props = "".join(f"property {'float' if t == '<f4' else 'uchar'} {n}\n" for n, t in fields)
    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {len(points)}\n{props}end_header\n"

    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(data.tobytes())

def workspace(cfg: dict) -> Path:
    """<ws> = extraction.video without the extension."""

    return Path(cfg["extraction"]["video"]).with_suffix("")

def find_video(path) -> Path | None:
    """path if it exists, else a file in the same folder whose name matches ignoring case
    (video.MOV vs video.mov on case-sensitive Linux file systems); None when nothing matches."""

    path = Path(path)
    if path.is_file():
        return path

    folder = path.parent if str(path.parent) else Path(".")
    if not folder.is_dir():
        return None

    match = sorted(p for p in folder.iterdir() if p.is_file() and p.name.lower() == path.name.lower())
    if match:
        log.warning("video %s not found, using %s (same name, different case)", path, match[0])

    return match[0] if match else None

def enter_repo(config_arg, repo_root) -> Path:
    """Absolute config path (relative to the caller's directory, else to the repo root) and chdir to the
    repo root, so every relative path inside the YAML starts there whatever directory main.py runs from."""

    config = Path(config_arg).expanduser()
    if not config.is_absolute():
        config = Path.cwd() / config if (Path.cwd() / config).exists() else Path(repo_root) / config

    os.chdir(repo_root)
    if not config.is_file():
        raise SystemExit(f"config {config_arg} not found (looked in {Path.cwd()} and {repo_root})")

    return config.resolve()

def check_intrinsics(intrinsics: dict, frame_path) -> bool:
    """WARNING when geometry.intrinsics cx/cy do not match the frame centre (another resolution or video:
    fx/fy are then wrong too). True when consistent or not set."""

    import cv2

    image = cv2.imread(str(frame_path)) if frame_path else None
    if image is None or intrinsics.get("cx") is None or intrinsics.get("cy") is None:
        return True

    h, w = image.shape[:2]
    ok = abs(2 * intrinsics["cx"] - w) <= 2 and abs(2 * intrinsics["cy"] - h) <= 2
    if not ok:
        log.warning("geometry.intrinsics cx=%s cy=%s expect %gx%g frames but %s is %dx%d: fix or null them",
                    intrinsics["cx"], intrinsics["cy"], 2 * intrinsics["cx"], 2 * intrinsics["cy"],
                    Path(frame_path).name, w, h)

    return ok

TAG_EXCLUDED = ("rerun", "tag", "output_dir")  # cache/path flags: changing them keeps the same frames tag

def extraction_tag(extraction: dict) -> str:
    """First 8 hex chars of the SHA-1 of the extraction block (sorted-key JSON) without TAG_EXCLUDED."""

    block = {k: v for k, v in extraction.items() if k not in TAG_EXCLUDED}
    text = json.dumps(block, sort_keys=True, default=str)

    return hashlib.sha1(text.encode()).hexdigest()[:8]

def resolve_paths(cfg: dict, config_path) -> dict:
    """Fills the derived paths of a validated cfg dict (in place, also returned): run.id = config stem,
    extraction.tag = extraction_tag, null run.output_root / run.artifacts_root / extraction.output_dir ->
    <ws>/results, <ws>/artifacts, <ws>/frames/<tag>. Explicit values win."""

    ws = workspace(cfg)
    run, ext = cfg["run"], cfg["extraction"]
    run["id"] = run["id"] or Path(config_path).stem
    run["output_root"] = run["output_root"] or str(ws / "results")
    run["artifacts_root"] = run["artifacts_root"] or str(ws / "artifacts")
    ext["tag"] = ext["tag"] or extraction_tag(ext)
    ext["output_dir"] = ext["output_dir"] or str(ws / "frames" / ext["tag"])

    return cfg

def run_dirs(cfg: dict) -> tuple[Path, Path]:
    """(run_dir, artifacts_dir) of a resolved cfg."""

    return Path(cfg["run"]["output_root"]) / cfg["run"]["id"], Path(cfg["run"]["artifacts_root"]) / cfg["run"]["id"]

def scale_info(run_dir) -> tuple[float, str]:
    """(s, unit): s = cm per model unit from scale/scale.npy; (nan, "model") when unknown."""

    path = Path(run_dir) / "scale" / "scale.npy"
    s = float(np.load(path).ravel()[0]) if path.exists() else float("nan")

    return (s, "cm") if np.isfinite(s) else (float("nan"), "model")

def units_per_cm(run_dir, points: np.ndarray | None, room_extent_cm: float) -> float:
    """Working units per cm: 1.0 when the model is in cm; otherwise a PROVISIONAL factor that maps
    the robust extent (2nd to 98th percentile) of the points to room_extent_cm. Used only to turn
    *_cm thresholds into model units; outputs in cm stay empty when the unit is model."""

    _, unit = scale_info(run_dir)
    if unit == "cm":
        return 1.0

    if points is None:
        points = np.load(Path(run_dir) / "sfm" / "points3d.npy")

    lo, hi = np.percentile(np.asarray(points).reshape(-1, 3), [2, 98], axis=0)
    extent = float(np.max(hi - lo)) or 1.0
    factor = extent / room_extent_cm
    log.warning("scale unknown: provisional %.4g model units per cm (extent %.3g -> %.0f cm), not metric",
                factor, extent, room_extent_cm)

    return factor

if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "t.csv"
        upsert_csv(p, {"source": "sparse", "n": 1}, "source")
        upsert_csv(p, {"source": "dense", "n": 2}, "source")
        upsert_csv(p, {"source": "sparse", "n": 3}, "source")
        rows = read_csv(p)
        assert [(r["source"], r["n"]) for r in rows] == [("dense", "2"), ("sparse", "3")]
        write_ply(Path(tmp) / "c.ply", np.zeros((4, 3)), np.full((4, 3), 255))
        assert (Path(tmp) / "c.ply").stat().st_size > 4 * 15
        assert uniform_subsample(list(range(10)), 4) == [0, 3, 6, 9] and uniform_subsample([1, 2], None) == [1, 2]
        assert scale_info(tmp)[1] == "model"
        assert abs(units_per_cm(tmp, np.array([[0, 0, 0], [10, 0, 0.0]]), 500.0) - 0.0192) < 1e-3
        assert workspace({"extraction": {"video": "video_preview.MOV"}}) == Path("video_preview")
        c = resolve_paths({"run": {"id": None, "output_root": None, "artifacts_root": None},
                           "extraction": {"video": "v.MOV", "rerun": False, "tag": None, "output_dir": None, "stride": 2}},
                          "configs/default.yaml")
        tag = c["extraction"]["tag"]
        assert len(tag) == 8 and c["extraction"]["output_dir"] == f"v/frames/{tag}" and c["run"]["id"] == "default"
        assert run_dirs(c) == (Path("v/results/default"), Path("v/artifacts/default"))
        assert extraction_tag({**c["extraction"], "rerun": True, "output_dir": "x"}) == tag
        assert extraction_tag({**c["extraction"], "stride": 3}) != tag
        cwd = os.getcwd()
        (Path(tmp) / "repo/configs").mkdir(parents=True)
        (Path(tmp) / "repo/configs/x.yaml").write_text("run: {}\n")
        (Path(tmp) / "elsewhere").mkdir()
        os.chdir(Path(tmp) / "elsewhere")
        assert enter_repo("configs/x.yaml", Path(tmp) / "repo") == (Path(tmp) / "repo/configs/x.yaml").resolve()
        assert Path.cwd() == (Path(tmp) / "repo").resolve()
        os.chdir(cwd)
        import cv2
        cv2.imwrite(str(Path(tmp) / "f.png"), np.zeros((1080, 1920, 3), np.uint8))
        assert check_intrinsics({"cx": 960, "cy": 540}, Path(tmp) / "f.png")
        assert not check_intrinsics({"cx": 960, "cy": 720}, Path(tmp) / "f.png")
        assert check_intrinsics({"cx": None, "cy": None}, Path(tmp) / "f.png") and check_intrinsics({"cx": 1}, None)
        (Path(tmp) / "clip.mov").write_bytes(b"x")
        assert find_video(Path(tmp) / "clip.MOV") == Path(tmp) / "clip.mov" and find_video(Path(tmp) / "no.MOV") is None
        assert find_video(Path(tmp) / "clip.mov") == Path(tmp) / "clip.mov" and find_video(Path(tmp) / "x/y.mov") is None
    print("OK")

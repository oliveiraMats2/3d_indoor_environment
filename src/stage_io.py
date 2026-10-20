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
import csv, logging
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
        assert scale_info(tmp)[1] == "model"
        assert abs(units_per_cm(tmp, np.array([[0, 0, 0], [10, 0, 0.0]]), 500.0) - 0.0192) < 1e-3
        assert workspace({"extraction": {"video": "video_preview.MOV"}}) == Path("video_preview")
    print("OK")

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

"""Shared I/O for the pipeline stages: CSV upsert, metric logging and PLY writing."""
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

if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "t.csv"
        upsert_csv(p, {"source": "a", "n": 1}, "source")
        upsert_csv(p, {"source": "b", "n": 2}, "source")
        upsert_csv(p, {"source": "a", "n": 3}, "source")
        rows = read_csv(p)
        assert [(r["source"], r["n"]) for r in rows] == [("b", "2"), ("a", "3")]
        upsert_csv(p, {"only": 1})
        assert read_csv(p) == [{"only": "1"}] and read_csv(Path(tmp) / "missing.csv") == []
        log_metrics(log, {"x": 1})
        write_ply(Path(tmp) / "c.ply", np.zeros((4, 3)), np.full((4, 3), 255))
        assert (Path(tmp) / "c.ply").stat().st_size > 4 * 15
    print("OK")

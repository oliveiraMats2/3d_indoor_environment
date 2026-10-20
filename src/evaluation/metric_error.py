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
# This block is a prompt complement for the AI assistant that will implement
# this file. Read it fully before writing any code. It is self-contained: do not
# look for other documents.
#
# Stage: Stage 5, Evaluation          Owner: unassigned          Due: TBD
# Project: 3D reconstruction of an indoor environment (variant A) from frames
#          extracted from a phone video.
# Context: final metrics of the reconstruction, required for every group, plus
# the indoor-environment (variant A) metrics. Each result is one row in a CSV
# under artifacts/<run_id>/ and the partial values are saved as .npy.
#
# Goal of this file (single responsibility):
#   Metric error of every dimension measured by hand at collection time.
#   With L the real measure and L_hat the measure in the model, report
#   |L_hat - L| and the relative error e = |L_hat - L| / L.
#   Variant A: report the errors of the room dimensions in cm.
#
# What to implement:
#   - For each entry in measured_cm, read its L_hat from the scaled model.
#   - Compute abs_error_cm = |L_hat - L| and rel_error = abs_error_cm / L.
#
# Inputs / outputs:
#   Inputs: scaled model measures from artifacts/<run_id>/floorplan.csv (last
#   row): room_width -> width_cm, room_length -> length_cm, ceiling_height ->
#   ceiling_height_cm (empty when the scale is unknown or the file is missing).
#   Outputs under results/<run_id>/evaluation/: metric_errors.npy (K, 2):
#   abs_error_cm, rel_error. One row per dimension in
#   artifacts/<run_id>/metric_error.csv: name, value_cm, model_cm,
#   abs_error_cm, rel_error.
#
# Defaults (read from the YAML block evaluation.measured_cm, never hardcoded):
#   measured_cm: a list of {name, value_cm}; name in room_width, room_length,
#   ceiling_height (pydantic Literal; unknown names also raise ValueError here).
#   Empty list -> WARNING and metric_error.csv with the header only.
#   Unknown model_cm -> WARNING, row with empty model_cm/abs/rel, NaN in .npy.
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
import csv, logging, math
from pathlib import Path

import numpy as np

from src import stage_io

log = logging.getLogger(__name__)

COLUMNS = {"room_width": "width_cm", "room_length": "length_cm", "ceiling_height": "ceiling_height_cm"}
FIELDS = ["name", "value_cm", "model_cm", "abs_error_cm", "rel_error"]

def model_value(row: dict, name: str) -> float:
    """L_hat from the floorplan row; NaN when empty or unparsable."""

    try:
        v = float(row.get(COLUMNS[name], ""))
    except ValueError:
        return float("nan")

    return v if math.isfinite(v) else float("nan")

def run(cfg: dict, run_dir: str, artifacts_dir: str) -> Path | None:

    ev = cfg["evaluation"]
    if not ev.get("enabled", False):
        log.info("metric_error: evaluation disabled")
        return None

    measures = [dict(m) for m in ev.get("measured_cm") or []]
    for m in measures:
        if m["name"] not in COLUMNS:
            raise ValueError(f"evaluation.measured_cm: unknown name {m['name']!r}, expected one of {list(COLUMNS)}")

    csv_path = Path(artifacts_dir) / "metric_error.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    out_dir = Path(run_dir) / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not measures:
        log.warning("evaluation.measured_cm is empty (tape measurements missing)")
        with open(csv_path, "w", newline="") as fh:
            csv.writer(fh).writerow(FIELDS)
        np.save(out_dir / "metric_errors.npy", np.empty((0, 2)))
        return csv_path

    fp_rows = stage_io.read_csv(Path(artifacts_dir) / "floorplan.csv")
    if not fp_rows:
        log.warning("metric_error: floorplan.csv missing or empty, model_cm unknown")
    fp = fp_rows[-1] if fp_rows else {}

    rows, errors = [], np.full((len(measures), 2), np.nan)
    for i, m in enumerate(measures):
        L, L_hat = float(m["value_cm"]), model_value(fp, m["name"])
        row = {"name": m["name"], "value_cm": L, "model_cm": "", "abs_error_cm": "", "rel_error": ""}
        if math.isnan(L_hat):
            log.warning("metric_error: %s has no model value (scale unknown or floorplan missing)", m["name"])
        else:
            abs_err = abs(L_hat - L)
            rel_err = abs_err / L if L else float("nan")
            errors[i] = abs_err, rel_err
            row.update(model_cm=L_hat, abs_error_cm=abs_err, rel_error=rel_err if math.isfinite(rel_err) else "")
        rows.append(row)
        stage_io.log_metrics(log, {f"{m['name']}_{k}": v for k, v in row.items() if k != "name"})

    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    np.save(out_dir / "metric_errors.npy", errors)
    log.info("metric_error: %d measures -> %s", len(rows), csv_path)

    return csv_path

if __name__ == "__main__":
    import tempfile
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    with tempfile.TemporaryDirectory() as tmp:
        run_dir, art = Path(tmp) / "res", Path(tmp) / "art"
        meas = [{"name": "room_width", "value_cm": 400.0}, {"name": "room_length", "value_cm": 500.0},
                {"name": "ceiling_height", "value_cm": 250.0}]
        cfg = {"evaluation": {"enabled": True, "measured_cm": meas}}
        stage_io.upsert_csv(art / "floorplan.csv", {"width_cm": 410.0, "length_cm": 490.0, "ceiling_height_cm": 250.0})
        rows = stage_io.read_csv(run(cfg, str(run_dir), str(art)))
        assert [r["name"] for r in rows] == ["room_width", "room_length", "ceiling_height"]
        e = np.load(run_dir / "evaluation" / "metric_errors.npy")
        assert e.shape == (3, 2) and np.allclose(e, [[10, 0.025], [10, 0.02], [0, 0]])
        assert float(rows[0]["abs_error_cm"]) == 10.0 and abs(float(rows[1]["rel_error"]) - 0.02) < 1e-12

        stage_io.upsert_csv(art / "floorplan.csv", {"width_cm": "", "length_cm": "", "ceiling_height_cm": ""})
        rows = stage_io.read_csv(run(cfg, str(run_dir), str(art)))
        assert all(r["model_cm"] == r["abs_error_cm"] == r["rel_error"] == "" for r in rows)
        assert np.isnan(np.load(run_dir / "evaluation" / "metric_errors.npy")).all()
        (art / "floorplan.csv").unlink()
        rows = stage_io.read_csv(run(cfg, str(run_dir), str(art)))
        assert all(r["model_cm"] == "" for r in rows) and rows[0]["value_cm"] == "400.0"

        p = run({"evaluation": {"enabled": True, "measured_cm": []}}, str(run_dir), str(art))
        assert p.read_text().strip() == ",".join(FIELDS) and stage_io.read_csv(p) == []

        try:
            run({"evaluation": {"enabled": True, "measured_cm": [{"name": "door", "value_cm": 1.0}]}},
                str(run_dir), str(art))
            raise AssertionError("unknown name accepted")
        except ValueError:
            pass
    print("OK")
# <<< END IMPLEMENTATION <<<

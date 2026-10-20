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
#   Variant A: ablation study. Compare exhaustive vs. sequential matching
#   (with and without loop detection), or the number of photos/frames used.
#   One YAML per variant (one experiment per config), and this file only reads
#   their results and builds the comparison table.
#
# What to implement:
#   - Read the evaluation CSVs of each listed run id under artifacts/.
#   - Build one table, one row per variant, with the same metric columns.
#   - Run nothing itself; experiments are run by main.py with their own YAML.
#   - Variant artifacts live at <ws>/artifacts/<id>/ (<ws> = stage_io.workspace,
#     all variants share the video). axis/value come from the header lines
#     '#   axis : ...' and '#   variant : ...' of configs/experiments/<id>.yaml
#     ('default' -> axis baseline, empty value; missing YAML -> empty + WARNING).
#   - Columns: run_id, axis, value, n_frames, n_registered, registered_frac,
#     n_sparse, track_len_mean, reproj_mean_px, reproj_median_px, n_dense,
#     metric_error_mean_rel (mean rel_error of metric_error.csv), drift_pct,
#     delta_cm, t_total_s, t_pipeline_s (sum of 'stage X: done in Y s' of the
#     last run in <ws>/results/<id>/run.log). Sources: sfm_metrics.csv,
#     evaluation.csv, loop_drift.csv, metric_error.csv. Missing CSV -> empty
#     cells + WARNING naming the file; the row is never dropped.
#
# Inputs / outputs:
#   Inputs: artifacts/<variant_run_id>/*.csv for each listed variant.
#   Outputs: artifacts/<run_id>/ablation.csv (one row per variant) and
#   results/<run_id>/evaluation/ablation.npy with the numeric table (float64,
#   NaN for blanks, columns value..t_pipeline_s), plus ablation_columns.npy and
#   ablation_run_ids.npy.
#
# Defaults (read from the YAML block evaluation, never hardcoded):
#   ablation_runs: a list of run ids to compare (empty -> skipped)
#
# Rules:
#   - Write code ONLY between the BEGIN and END markers below. Do not create
#     other regions or other files for this task.
#   - Configuration comes from the YAML loaded by main.py. No argparse.
#   - Save every partial result as .npy under results/<run_id>/<stage>/.
#   - Keep this block after implementing; update it if the behavior changes.
#   - Expose run(cfg: dict, run_dir: str, artifacts_dir: str) (full cfg); it
#     only acts when evaluation.ablation_runs is non-empty.
#   - Pipeline already available: frames are extracted and quality-filtered, and
#     SIFT, ORB and AKAZE keypoints/descriptors are saved per frame under
#     results/<run_id>/detection/<detector>/<frame>/ as keypoints.npy (N, 6:
#     x, y, size, angle, response, octave) and descriptors.npy (N, D).
# ==============================================================================

# >>> BEGIN IMPLEMENTATION >>>
"""Ablation table: one row per variant run id, read from the CSVs each run already wrote."""
import logging, math, re
from pathlib import Path

import numpy as np

from src import stage_io

log = logging.getLogger(__name__)

EXPERIMENTS_DIR = Path(__file__).resolve().parents[2] / "configs" / "experiments"
COLUMNS = ["run_id", "axis", "value", "n_frames", "n_registered", "registered_frac", "n_sparse",
           "track_len_mean", "reproj_mean_px", "reproj_median_px", "n_dense", "metric_error_mean_rel",
           "drift_pct", "delta_cm", "t_total_s", "t_pipeline_s"]
NUMERIC = COLUMNS[2:]
# column -> CSVs searched in order (first non-empty value wins)
SOURCES = {
    "n_frames": ["sfm_metrics.csv"], "n_registered": ["sfm_metrics.csv"],
    "registered_frac": ["sfm_metrics.csv"], "track_len_mean": ["sfm_metrics.csv"],
    "n_sparse": ["evaluation.csv", "sfm_metrics.csv"],
    "reproj_mean_px": ["evaluation.csv", "sfm_metrics.csv"],
    "reproj_median_px": ["evaluation.csv", "sfm_metrics.csv"],
    "n_dense": ["evaluation.csv"],
    "drift_pct": ["loop_drift.csv", "sfm_metrics.csv"], "delta_cm": ["loop_drift.csv", "sfm_metrics.csv"],
    "t_total_s": ["sfm_metrics.csv"],
}
HEADER_RE = re.compile(r"^#\s*(axis|variant)\s*:\s*(.*?)\s*$")
STAGE_RE = re.compile(r"stage (\S+): done in ([0-9.eE+-]+) s")

def _to_float(value) -> float:

    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")

def _axis_value(run_id: str, experiments_dir: Path) -> tuple[str, str]:
    """(axis, value) from the header of configs/experiments/<id>.yaml; default is the baseline."""

    if run_id == "default":
        return "baseline", ""

    path = experiments_dir / f"{run_id}.yaml"
    if not path.exists():
        log.warning("ablation %s: %s missing, axis/value left empty", run_id, path)
        return "", ""

    found = {}
    with open(path) as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            m = HEADER_RE.match(line)
            if m:
                found.setdefault(m.group(1), m.group(2))

    if len(found) < 2:
        log.warning("ablation %s: no axis/variant header in %s", run_id, path)

    return found.get("axis", ""), found.get("variant", "")

def _pipeline_seconds(log_path: Path) -> str:
    """Sum of 'stage X: done in Y s' of the LAST run in run.log (the handler appends across runs)."""

    if not log_path.exists():
        log.warning("ablation: %s missing, t_pipeline_s empty", log_path)
        return ""

    times: dict[str, float] = {}
    for line in log_path.read_text(errors="replace").splitlines():
        m = STAGE_RE.search(line)
        if not m:
            continue
        if m.group(1) in times:
            times = {}
        times[m.group(1)] = float(m.group(2))

    return f"{sum(times.values()):.1f}" if times else ""

def _variant_row(run_id: str, ws: Path, cfg: dict) -> dict:

    art = ws / cfg["run"]["artifacts_root"] / run_id
    res = ws / cfg["run"]["output_root"] / run_id
    axis, value = _axis_value(run_id, EXPERIMENTS_DIR)
    row = {"run_id": run_id, "axis": axis, "value": value}
    names = {n for srcs in SOURCES.values() for n in srcs} | {"metric_error.csv"}
    tables = {}
    for name in sorted(names):
        tables[name] = stage_io.read_csv(art / name)
        if not tables[name]:
            log.warning("ablation %s: %s missing or empty, its columns stay empty", run_id, art / name)

    for col, srcs in SOURCES.items():
        row[col] = ""
        for name in srcs:
            first = tables[name][0] if tables[name] else {}
            if first.get(col, "") not in ("", None):
                row[col] = first[col]
                break

    n_frames, n_reg = _to_float(row["n_frames"]), _to_float(row["n_registered"])
    if row["registered_frac"] == "" and np.isfinite(n_frames) and np.isfinite(n_reg) and n_frames > 0:
        row["registered_frac"] = f"{n_reg / n_frames:.4f}"

    rel = [_to_float(r.get("rel_error")) for r in tables["metric_error.csv"]]
    rel = [v for v in rel if np.isfinite(v)]
    row["metric_error_mean_rel"] = f"{float(np.mean(rel)):.6g}" if rel else ""
    row["t_pipeline_s"] = _pipeline_seconds(res / "run.log")

    return {c: row.get(c, "") for c in COLUMNS}

def run(cfg: dict, run_dir: str, artifacts_dir: str) -> Path | None:

    run_ids = list(cfg["evaluation"].get("ablation_runs") or [])
    if not run_ids:
        log.info("ablation: evaluation.ablation_runs empty, skipped")
        return None

    ws = stage_io.workspace(cfg)
    csv_path = Path(artifacts_dir) / "ablation.csv"
    rows = [_variant_row(run_id, ws, cfg) for run_id in run_ids]

    for i, row in enumerate(rows):
        stage_io.upsert_csv(csv_path, row, None if i == 0 else "run_id")
        stage_io.log_metrics(log, {f"ablation.{row['run_id']}.{c}": row[c] for c in NUMERIC})

    out_dir = Path(run_dir) / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    table = np.array([[_to_float(r[c]) for c in NUMERIC] for r in rows], dtype=np.float64)
    np.save(out_dir / "ablation.npy", table)
    np.save(out_dir / "ablation_columns.npy", np.array(NUMERIC))
    np.save(out_dir / "ablation_run_ids.npy", np.array(run_ids))
    log.info("ablation: %d variants -> %s", len(rows), csv_path)

    return csv_path

if __name__ == "__main__":
    import tempfile
    logging.basicConfig(level=logging.INFO)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ws = tmp / "video_preview"
        EXPERIMENTS_DIR = tmp / "experiments"
        EXPERIMENTS_DIR.mkdir()
        (EXPERIMENTS_DIR / "lowe_070.yaml").write_text(
            "# ====\n# lowe_070\n#   axis     : filters.lowe_ratio\n#   variant  : 0.70\n#   output   : x\nrun: {}\n")
        cfg = {"extraction": {"video": str(tmp / "video_preview.MOV")},
               "run": {"output_root": "results", "artifacts_root": "artifacts"},
               "evaluation": {"ablation_runs": ["default", "lowe_070", "ghost"]}}
        for rid, reproj, rels in [("default", 0.8, [0.1, 0.3]), ("lowe_070", 0.6, [])]:
            art = ws / "artifacts" / rid
            stage_io.upsert_csv(art / "sfm_metrics.csv", {"n_frames": 100, "n_registered": 90,
                                "track_len_mean": 4.5, "t_total_s": 12.0, "reproj_mean_px": 9.9})
            stage_io.upsert_csv(art / "evaluation.csv", {"reproj_mean_px": reproj, "reproj_median_px": 0.5,
                                "n_sparse": 1000, "n_dense": ""})
            for i, r in enumerate(rels):
                stage_io.upsert_csv(art / "metric_error.csv", {"name": f"d{i}", "rel_error": r}, "name")
            stage_io.upsert_csv(art / "loop_drift.csv", {"delta_cm": "", "drift_pct": ""})
        log_dir = ws / "results" / "default"
        log_dir.mkdir(parents=True)
        (log_dir / "run.log").write_text("x main INFO stage extraction: done in 99.0 s\n"
                                         "x main INFO stage extraction: done in 2.5 s\n"
                                         "x main INFO stage sfm: done in 10.0 s\n")
        out = run(cfg, str(ws / "results" / "abl"), str(ws / "artifacts" / "abl"))
        rows = stage_io.read_csv(out)
        assert [r["run_id"] for r in rows] == ["default", "lowe_070", "ghost"]
        assert list(rows[0]) == COLUMNS
        assert rows[0]["axis"] == "baseline" and rows[0]["value"] == ""
        assert rows[1]["axis"] == "filters.lowe_ratio" and rows[1]["value"] == "0.70"
        assert rows[2]["axis"] == "" and rows[2]["reproj_mean_px"] == ""
        assert rows[0]["reproj_mean_px"] == "0.8" and rows[1]["reproj_mean_px"] == "0.6"
        assert rows[0]["registered_frac"] == "0.9000"
        assert abs(float(rows[0]["metric_error_mean_rel"]) - 0.2) < 1e-9
        assert rows[1]["metric_error_mean_rel"] == ""
        assert rows[0]["t_pipeline_s"] == "12.5" and rows[1]["t_pipeline_s"] == ""
        table = np.load(ws / "results" / "abl" / "evaluation" / "ablation.npy")
        cols = list(np.load(ws / "results" / "abl" / "evaluation" / "ablation_columns.npy"))
        assert table.shape == (3, len(NUMERIC)) and table.dtype == np.float64 and cols == NUMERIC
        assert table[1, cols.index("value")] == 0.70 and np.isnan(table[2]).all()
        assert np.isnan(table[0, cols.index("n_dense")]) and table[0, cols.index("n_sparse")] == 1000
    print("OK")
# <<< END IMPLEMENTATION <<<

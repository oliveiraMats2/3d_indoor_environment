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

"""SIFT x ORB x AKAZE tables and figures from the .npy files saved by detectores.py."""
import csv, logging, os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src import stage_io

COLS_DET = ["n_keypoints", "time_ms_mean", "time_ms_std", "time_per_kp_ms", "coverage_8x8",
            "entropy_8x8", "scale_p5", "scale_p95", "descriptor_bytes"]
COLS_MAT = ["n_matches_raw", "n_matches_lowe", "inliers", "inlier_ratio", "matching_score", "success",
            "sampson_px", "match_time_ms_mean", "match_time_ms_std"]

log = logging.getLogger(__name__)

load_pair_file = lambda mbase, pair, f: np.load(os.path.join(mbase, pair, f))

def ls(p):

    if not os.path.isdir(p):
        return []

    names = os.listdir(p)

    return sorted(d for d in names if os.path.isdir(os.path.join(p, d)))

def div(a, b):

    return a / b if b else np.nan

def load_image(folder, q):

    for ext in (".jpg", ".png"):

        path = os.path.join(folder, q + ext)
        if os.path.exists(path):

            return cv2.imread(path)

    return None

def save(path, img, side=1600):

    longest = max(img.shape[:2])
    s = side / longest
    if s < 1:

        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)
    params = [cv2.IMWRITE_JPEG_QUALITY, 90]
    cv2.imwrite(path, img, params)

def kps(kp):

    return [cv2.KeyPoint(float(x), float(y), float(s), float(a), float(r), int(o)) for x, y, s, a, r, o in kp]

def detection_metrics(d):

    kp_path = os.path.join(d, "keypoints.npy")
    desc_path = os.path.join(d, "descriptors.npy")
    times_path = os.path.join(d, "times_ms.npy")
    grid_path = os.path.join(d, "grid_counts.npy")
    kp = np.load(kp_path)
    desc = np.load(desc_path)
    t = np.load(times_path)
    grid = np.load(grid_path)
    g = grid.ravel()
    n = len(kp)
    total = max(g.sum(), 1)
    p = g[g > 0] / total
    t_mean = t.mean()
    t_std = t.std()
    time_per_kp = div(t_mean, n)
    coverage = (g > 0).mean()
    entropy = float(-(p * np.log(p)).sum() / np.log(64))

    if n:

        scales = np.percentile(kp[:, 2], [5, 95])

    else:
        scales = [np.nan, np.nan]

    desc_bytes = div(desc.nbytes, n)

    return kp, [n, t_mean, t_std, time_per_kp, coverage, entropy, *scales, desc_bytes]

def sampson(K, E, p1, p2):

    if len(p1) == 0 or np.isnan(E).any():
        return np.nan

    Ki = np.linalg.inv(K)
    F = Ki.T @ E @ Ki
    ones1 = np.ones(len(p1))
    ones2 = np.ones(len(p2))
    x1 = np.c_[p1, ones1]
    x2 = np.c_[p2, ones2]
    Fx1 = x1 @ F.T
    Ftx2 = x2 @ F
    d = (x2 * Fx1).sum(1) ** 2 / (Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2)
    dist = np.sqrt(d)

    return float(dist.mean())

def write_csv(path, header, rows):

    with open(path, "w", newline="") as fh:

        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)

def run(cfg: dict, run_dir: str, frames_dir: str, metrics_dir: str) -> None:

    k_path = os.path.join(run_dir, "K.npy")
    K = np.load(k_path) if os.path.exists(k_path) else None
    met = metrics_dir
    fig = os.path.join(run_dir, "figures")
    detection_dir = os.path.join(run_dir, "detection")
    os.makedirs(met, exist_ok=True)
    det_rows = []
    mat_rows = []
    summary = []

    for det in ls(detection_dir):

        base = os.path.join(detection_dir, det)
        frames = ls(base)
        n_fig = min(cfg["max_figures"], len(frames))
        idx = np.linspace(0, len(frames) - 1, n_fig)
        with_fig = {frames[int(i)] for i in idx}
        pts = {}
        dr = []
        progress = tqdm(frames, desc=f"{det} report frames", unit="frame")
        for q in progress:

            frame_dir = os.path.join(base, q)
            kp, m = detection_metrics(frame_dir)
            pts[q] = kp
            dr.append(m)
            det_rows.append([det, q, *m])
            img = load_image(frames_dir, q) if q in with_fig else None
            if img is not None:

                kp_list = kps(kp)
                img = cv2.drawKeypoints(img, kp_list, None, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                label = f"{det.upper()} {len(kp)} keypoints"
                cv2.putText(img, label, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 6)
                out = os.path.join(fig, "keypoints", det, q + ".jpg")
                save(out, img)

        mr = []
        mbase = os.path.join(run_dir, "epipolar", det)
        match_dir = os.path.join(run_dir, "matching", det)
        pairs = ls(mbase)
        progress = tqdm(pairs, desc=f"{det} report pairs", unit="pair")
        for pair in progress:

            required = ("inlier_mask.npy", "E.npy")
            if K is None or not all(os.path.exists(os.path.join(mbase, pair, name)) for name in required):
                continue

            qa, qb = pair.split("__")
            raw = load_pair_file(match_dir, pair, "matches_raw.npy")
            lowe = load_pair_file(match_dir, pair, "matches_lowe.npy")
            mask_raw = load_pair_file(mbase, pair, "inlier_mask.npy")
            mask = mask_raw.astype(bool)
            E = load_pair_file(mbase, pair, "E.npy")
            t = load_pair_file(match_dir, pair, "times_ms.npy")
            if len(mask) != len(lowe):
                log.warning("%s %s: inlier_mask (%d) != matches_lowe (%d), skipped", det, pair, len(mask), len(lowe))
                continue

            inl = lowe[mask] if len(lowe) else lowe.reshape(-1, 3)
            ia_idx = inl[:, 0].astype(int)
            ib_idx = inl[:, 1].astype(int)
            p1 = pts[qa][ia_idx, :2]
            p2 = pts[qb][ib_idx, :2]
            ni = len(inl)
            n_lowe = len(lowe)
            inlier_ratio = div(ni, n_lowe)
            min_kp = min(len(pts[qa]), len(pts[qb]))
            score = div(ni, min_kp)
            err = sampson(K, E, p1, p2)
            m = [len(raw), n_lowe, ni, inlier_ratio, score, int(ni >= 8), err, t.mean(), t.std()]
            mr.append(m)
            mat_rows.append([det, pair, *m])

            if ni >= 8:

                ia = load_image(frames_dir, qa)
                ib = load_image(frames_dir, qb)
                if ia is not None and ib is not None:

                    dm = [cv2.DMatch(int(i), int(j), float(d)) for i, j, d in inl]
                    kp_a = kps(pts[qa])
                    kp_b = kps(pts[qb])
                    img = cv2.drawMatches(ia, kp_a, ib, kp_b, dm, None, (0, 255, 0), (0, 0, 255),
                                          flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
                    out = os.path.join(fig, "registration", det, pair + ".jpg")
                    save(out, img)

        mr_arr = np.array(mr, float)
        mr = mr_arr.reshape(-1, len(COLS_MAT))
        dr_arr = np.array(dr, float)
        det_mean = np.nanmean(dr_arr, 0)

        if len(mr):

            mat_mean = np.nanmean(mr[:, [0, 1, 2, 3, 4, 6, 7]], 0)
            success_rate = mr[:, 5].mean()

        else:
            mat_mean = [np.nan] * 7
            success_rate = np.nan

        summary.append([det, len(frames), len(mr), *det_mean, *mat_mean, success_rate])

    det_csv = os.path.join(met, "detection_metrics.csv")
    mat_csv = os.path.join(met, "matching_metrics.csv")
    summary_csv = os.path.join(met, "summary.csv")
    det_header = ["detector", "frame", *COLS_DET]
    mat_header = ["detector", "pair", *COLS_MAT]
    kept_mat = [c for c in COLS_MAT if c not in ("success", "match_time_ms_std")]
    header = ["detector", "n_frames", "n_pairs", *COLS_DET, *kept_mat, "success_rate"]
    write_csv(det_csv, det_header, det_rows)
    write_csv(mat_csv, mat_header, mat_rows)
    write_csv(summary_csv, header, summary)

    # transposed table: one row per metric, one column per detector
    title_cells = "".join(f"{r[0]:>14}" for r in summary)
    print(f"{'metric':<26}" + title_cells)

    for i, c in enumerate(header[1:], 1):

        cells = "".join(f"{r[i]:>14.4g}" for r in summary)
        print(f"{c:<26}" + cells)

    print(f"CSVs in {met}, figures in {fig}")

    write_report(metrics_dir, run_dir, cfg.get("max_rows", 20), cfg["max_figures"])

# report.md: one section per stage CSV (pipeline order), figures, exports and videos; missing -> FALTANDO
REPORT_CSVS = [("extraction_metrics.csv", "extraction"), ("frame_quality_metrics.csv", "frame_quality"),
               ("detection_metrics.csv", "report (detection)"), ("matching_metrics.csv", "report (matching)"),
               ("summary.csv", "report"), ("epipolar_metrics.csv", "two_view"), ("sfm_metrics.csv", "sfm"),
               ("scale.csv", "scale"), ("cleanup.csv", "cleanup"), ("planes.csv", "planes"),
               ("wall_angles.csv", "planes"), ("floorplan.csv", "floorplan"), ("dense_metrics.csv", "dense"),
               ("dense_compare.csv", "dense_compare"), ("poisson.csv", "poisson"), ("close_holes.csv", "close_holes"),
               ("mesh.csv", "mesh"), ("evaluation.csv", "evaluation"),
               ("metric_error.csv", "evaluation"), ("loop_drift.csv", "evaluation"),
               ("hard_regions.csv", "evaluation"), ("ablation.csv", "ablation"), ("videos.csv", "videos")]
REPORT_EXPORTS = [("exports/colmap/project.ini", "sfm"), ("exports/meshlab/project.mlp", "export"),
                  ("exports/web/scene.glb", "mesh"), ("exports/meshlab/mesh.ply", "mesh"),
                  ("exports/meshlab/mesh.obj", "mesh"), ("exports/meshlab/mesh.stl", "mesh"),
                  ("exports/meshlab/mesh.off", "mesh"), ("exports/web/mesh.glb", "mesh")]
REPORT_EXPORT_GLOBS = [("exports/meshlab/poisson_*.ply", "poisson"),  # one file per variant (YAML variants)
                       ("exports/meshlab/close_holes_*.ply", "close_holes")]
REPORT_FIGURES = [("keypoints", "*.jpg", "report"), ("registration", "*.jpg", "report"),
                  ("matching", "*", "matching"), ("epilines", "*.png", "two_view")]

def rel_path(path: Path, ws: Path) -> str:

    path = Path(path).resolve()
    try:
        return str(path.relative_to(ws))
    except ValueError:
        return str(path)

def cell(v) -> str:

    v = "" if v is None else str(v)
    try:
        f = float(v)
        v = v if f.is_integer() and "." not in v and "e" not in v.lower() else f"{f:.4g}"
    except ValueError:
        pass

    return v.replace("|", "\\|").replace("\n", " ")

def md_table(rows: list[dict], max_rows: int, full_path: str) -> list[str]:

    if not rows:
        return ["(CSV vazio)"]

    cols = list(dict.fromkeys(k for r in rows for k in r))
    lines = ["| " + " | ".join(cell(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(cell(r.get(c)) for c in cols) + " |" for r in rows[:max_rows]]
    if len(rows) > max_rows:
        lines.append(f"... ({len(rows)} rows, full file: {full_path})")

    return lines

def missing(what: str, stage: str) -> str:

    return f"FALTANDO: {what} (estágio {stage} desligado ou falhou)"

def write_report(metrics_dir: str, run_dir: str, max_rows: int, max_figures: int) -> Path:
    """Writes metrics_dir/report.md; paths relative to the ws (parent of results/) when possible."""

    met, run = Path(metrics_dir), Path(run_dir)
    ws = run.resolve().parent.parent
    out = [f"# Report {run.name}", "", f"run_dir: {rel_path(run, ws)}  ", f"metrics: {rel_path(met, ws)}", ""]
    n_missing = 0

    for name, stage in REPORT_CSVS:

        path = met / name
        out += [f"## {stage}: {name}", ""]
        if path.exists():
            out += md_table(stage_io.read_csv(path), max_rows, rel_path(path, ws))
        else:
            out.append(missing(name, stage))
            n_missing += 1
        out.append("")

    out += ["## figures", ""]
    floorplan = run / "figures" / "floorplan.png"
    if floorplan.exists():
        out.append(f"- {rel_path(floorplan, ws)}")
    else:
        out.append("- " + missing("figures/floorplan.png", "floorplan"))
        n_missing += 1

    for sub, pattern, stage in REPORT_FIGURES:

        base = run / "figures" / sub
        dets = sorted(d for d in base.iterdir() if d.is_dir()) if base.is_dir() else []
        if not dets:
            out.append("- " + missing(f"figures/{sub}/<det>/{pattern}", stage))
            n_missing += 1
        for d in dets:
            files = sorted(f for f in d.glob(pattern) if f.is_file())
            out += [f"- {rel_path(f, ws)}" for f in files[:max_figures]]
            if len(files) > max_figures:
                out.append(f"- ... ({len(files)} files in {rel_path(d, ws)})")
            if not files:
                out.append("- " + missing(f"figures/{sub}/{d.name}/{pattern}", stage))
                n_missing += 1

    out += ["", "## exports", ""]
    for name, stage in REPORT_EXPORTS:

        if (run / name).exists():
            out.append(f"- {rel_path(run / name, ws)}")
        else:
            out.append("- " + missing(name, stage))
            n_missing += 1

    for pattern, stage in REPORT_EXPORT_GLOBS:
        files = sorted(run.glob(pattern))
        out += [f"- {rel_path(f, ws)}" for f in files]
        if not files:
            out.append("- " + missing(pattern, stage))
            n_missing += 1

    out += ["", "## videos", ""]
    rows = stage_io.read_csv(met / "videos.csv")
    paths = [v for r in rows for v in r.values() if v and str(v).endswith(".mp4")]
    on_disk = sorted((run / "videos").glob("*.mp4")) if (run / "videos").is_dir() else []
    paths += [rel_path(p, ws) for p in on_disk]
    for p in dict.fromkeys(paths):
        exists = (ws / p).exists() or Path(p).exists() or (run / "videos" / Path(p).name).exists()
        out.append(f"- {p}" if exists else "- " + missing(p, "videos"))
    if not paths:
        out.append("- " + missing("videos/*.mp4", "videos"))
        n_missing += 1

    report = met / "report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(out) + "\n", encoding="utf-8")
    stage_io.log_metrics(log, {"report_missing_items": n_missing})
    log.info("report written to %s", report)

    return report

if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "results" / "x"
        met = Path(tmp) / "artifacts" / "x"
        stage_io.upsert_csv(met / "sfm_metrics.csv", {"n_images": 30, "reproj_px": 0.81234567})
        for i in range(5):
            stage_io.upsert_csv(met / "cleanup.csv", {"source": f"s{i}", "n_points": 100 * i}, "source")
        stage_io.upsert_csv(met / "videos.csv", {"name": "nav", "path": "results/x/videos/navigation_sparse_45s.mp4"})
        (run_dir / "videos").mkdir(parents=True)
        (run_dir / "videos" / "navigation_sparse_45s.mp4").write_bytes(b"0")
        (run_dir / "figures").mkdir(parents=True)
        (run_dir / "figures" / "floorplan.png").write_bytes(b"0")
        (run_dir / "exports" / "colmap").mkdir(parents=True)
        (run_dir / "exports" / "colmap" / "project.ini").write_text("")
        (run_dir / "exports" / "meshlab").mkdir(parents=True)
        (run_dir / "exports" / "meshlab" / "poisson_pre.ply").write_bytes(b"0")
        stage_io.upsert_csv(met / "poisson.csv", {"variant": "pre", "n_faces": 10}, "variant")

        text = write_report(str(met), str(run_dir), max_rows=3, max_figures=2).read_text()
        assert "| n_images | reproj_px |" in text and "| 30 | 0.8123 |" in text
        assert "... (5 rows, full file: artifacts/x/cleanup.csv)" in text and "| s3 |" not in text
        assert "FALTANDO: extraction_metrics.csv (estágio extraction desligado ou falhou)" in text
        assert "FALTANDO: ablation.csv (estágio ablation desligado ou falhou)" in text
        assert "- results/x/figures/floorplan.png" in text and "- results/x/exports/colmap/project.ini" in text
        assert "FALTANDO: exports/web/mesh.glb (estágio mesh desligado ou falhou)" in text
        assert "- results/x/videos/navigation_sparse_45s.mp4" in text
        assert "- results/x/exports/meshlab/poisson_pre.ply" in text and "| pre | 10 |" in text
        assert "FALTANDO: exports/meshlab/close_holes_*.ply (estágio close_holes desligado ou falhou)" in text
        assert "FALTANDO: close_holes.csv (estágio close_holes desligado ou falhou)" in text
        assert text.index("dense_compare.csv") < text.index("poisson.csv") < text.index("close_holes.csv") < text.index("mesh.csv")
        assert text.count("FALTANDO") == 18 + 4 + 7 + 1 + 1, text.count("FALTANDO")
    print("OK")

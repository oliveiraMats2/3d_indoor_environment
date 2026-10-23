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
import cv2
import numpy as np
from tqdm import tqdm

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

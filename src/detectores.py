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

import functools, logging, os, time
import cv2
import numpy as np
from tqdm import tqdm

log = logging.getLogger(__name__)

# name: (constructor, matcher norm, descriptor size and dtype)
DETECTORS = {
    "sift": (lambda n: cv2.SIFT_create(nfeatures=n), cv2.NORM_L2, 128, np.float32),
    "orb": (lambda n: cv2.ORB_create(nfeatures=n), cv2.NORM_HAMMING, 32, np.uint8),
    "akaze": (lambda n: cv2.xfeatures2d.AKAZE_create(max_points=n), cv2.NORM_HAMMING, 61, np.uint8),
}

# matches as rows of (queryIdx, trainIdx, distance)
to_match_array = lambda ms: np.float32([[m.queryIdx, m.trainIdx, m.distance] for m in ms]).reshape(-1, 3)

def time_it(f, reps, warmup):

    t = []

    for i in range(warmup + reps):

        t0 = time.perf_counter()
        r = f()

        if i >= warmup:

            t1 = time.perf_counter()
            elapsed = (t1 - t0) * 1e3
            t.append(elapsed)

    times = np.array(t, np.float64)

    return r, times

def save(folder, **arrs):

    os.makedirs(folder, exist_ok=True)

    for k, v in arrs.items():

        path = os.path.join(folder, k + ".npy")
        np.save(path, v)

def match(bf, da, db, ratio):

    if da is None or db is None or len(da) < 2 or len(db) < 2:
        return [], []

    knn = bf.knnMatch(da, db, k=2)
    pairs = [p for p in knn if p]
    raw = [p[0] for p in pairs]
    lowe = [p[0] for p in pairs if len(p) == 2 and p[0].distance < ratio * p[1].distance]

    return raw, lowe

def intrinsics_matrix(intr, w, h):

    # ponytail: approximate intrinsics (f = max(w, h), principal point at center), replace with calibration if available
    default_f = max(w, h)
    fx = intr.get("fx") or default_f
    fy = intr.get("fy") or default_f
    cx = intr.get("cx") or w / 2
    cy = intr.get("cy") or h / 2
    rows = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    K = np.array(rows, np.float64)

    return K

def run(cfg: dict, frame_paths: list[str], out_dir: str, lowe_ratio: float, intrinsics: dict) -> None:

    names = []

    for p in frame_paths:

        base = os.path.basename(p)
        stem = os.path.splitext(base)[0]
        names.append(stem)

    imgs = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in frame_paths]
    h, w = imgs[0].shape
    K = intrinsics_matrix(intrinsics, w, h)
    reps, warmup = cfg["reps"], cfg["warmup"]
    save(out_dir, K=K)
    log.info(f"{len(imgs)} frames {w}x{h}, fx={K[0, 0]:.1f}px -> {out_dir}")

    for name in cfg["methods"]:

        create, norm, dim, dtype = DETECTORS[name]
        det = create(cfg["nfeatures"])
        bf = cv2.BFMatcher(norm)
        res, tdet = [], []
        named_imgs = zip(names, imgs)
        progress = tqdm(named_imgs, total=len(imgs), desc=f"{name} detection", unit="frame")
        for q, g in progress:

            detect = functools.partial(det.detectAndCompute, g, None)
            (kp, d), t = time_it(detect, reps, warmup)
            kp_rows = [[k.pt[0], k.pt[1], k.size, k.angle, k.response, k.octave] for k in kp]
            kps_flat = np.float32(kp_rows)
            kps = kps_flat.reshape(-1, 6)

            if d is None:
                d = np.zeros((0, dim), dtype)

            grid_range = [[0, g.shape[0]], [0, g.shape[1]]]
            hist = np.histogram2d(kps[:, 1], kps[:, 0], bins=8, range=grid_range)
            grid = hist[0]
            grid_counts = grid.astype(np.int64)
            det_dir = os.path.join(out_dir, "detection", name, q)
            save(det_dir, keypoints=kps, descriptors=d, times_ms=t, grid_counts=grid_counts)
            res.append((kps, d))
            t_mean = t.mean()
            tdet.append(t_mean)

        tmat, ninl = [], []
        pair_indices = range(len(res) - 1)
        progress = tqdm(pair_indices, desc=f"{name} matching", unit="pair")
        for i in progress:

            (ka, da), (kb, db) = res[i], res[i + 1]
            match_pair = functools.partial(match, bf, da, db, lowe_ratio)
            (raw, lowe), t = time_it(match_pair, reps, warmup)
            n_lowe = len(lowe)
            mask = np.zeros(n_lowe, bool)
            E = np.full((3, 3), np.nan)

            if n_lowe >= 5:

                query_idx = [m.queryIdx for m in lowe]
                train_idx = [m.trainIdx for m in lowe]
                pa = ka[query_idx, :2]
                pb = kb[train_idx, :2]
                Ee, mk = cv2.findEssentialMat(pa, pb, K, cv2.RANSAC, 0.999, 1.0)
                if Ee is not None and mk is not None:

                    E = Ee[:3].astype(np.float64)
                    mask_flat = mk.ravel()
                    mask = mask_flat.astype(bool)

            pair_dir = os.path.join(out_dir, "matching", name, f"{names[i]}__{names[i + 1]}")
            matches_raw = to_match_array(raw)
            matches_lowe = to_match_array(lowe)
            save(pair_dir, matches_raw=matches_raw, matches_lowe=matches_lowe, inlier_mask=mask, E=E, times_ms=t)
            t_mean = t.mean()
            tmat.append(t_mean)
            n_inliers = mask.sum()
            ninl.append(n_inliers)

        det_ms = np.mean(tdet)
        kps_counts = [len(k) for k, _ in res]
        kps_per_frame = np.mean(kps_counts)
        match_ms = np.mean(tmat) if tmat else 0
        inliers_per_pair = np.mean(ninl) if ninl else 0
        log.info(f"{name}: detection {det_ms:.1f} ms, {kps_per_frame:.0f} kps/frame | "
                 f"matching {match_ms:.1f} ms, {inliers_per_pair:.0f} inliers/pair")

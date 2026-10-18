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

# name: (constructor, descriptor size and dtype)
DETECTORS = {
    "sift": (lambda n: cv2.SIFT_create(nfeatures=n), 128, np.float32),
    "orb": (lambda n: cv2.ORB_create(nfeatures=n), 32, np.uint8),
    "akaze": (lambda n: cv2.xfeatures2d.AKAZE_create(max_points=n), 61, np.uint8),
}

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

def run(cfg: dict, frame_paths: list[str], out_dir: str) -> None:

    if not frame_paths:
        raise ValueError("detection requires at least one quality-approved frame")

    names = []

    for p in frame_paths:

        base = os.path.basename(p)
        stem = os.path.splitext(base)[0]
        names.append(stem)

    imgs = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in frame_paths]
    h, w = imgs[0].shape
    reps, warmup = cfg["reps"], cfg["warmup"]
    log.info(f"{len(imgs)} frames {w}x{h} -> {out_dir}")

    for name in cfg["methods"]:

        create, dim, dtype = DETECTORS[name]
        det = create(cfg["nfeatures"])
        tdet = []
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
            t_mean = t.mean()
            tdet.append(t_mean)

        det_ms = np.mean(tdet)
        log.info(f"{name}: detection {det_ms:.1f} ms")

    save(out_dir, detection_complete=np.array(True))

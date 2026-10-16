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

import argparse, glob, os, time
import cv2
import numpy as np

# nome: (construtor, norma do matcher, dimensao e dtype do descritor)
DETECTORES = {
    "sift": (lambda n: cv2.SIFT_create(nfeatures=n), cv2.NORM_L2, 128, np.float32),
    "orb": (lambda n: cv2.ORB_create(nfeatures=n), cv2.NORM_HAMMING, 32, np.uint8),
    "akaze": (lambda n: cv2.xfeatures2d.AKAZE_create(max_points=n), cv2.NORM_HAMMING, 61, np.uint8),
}


def cronometrar(f, reps, warmup):
    t = []
    for i in range(warmup + reps):
        t0 = time.perf_counter()
        r = f()
        if i >= warmup:
            t.append((time.perf_counter() - t0) * 1e3)
    return r, np.array(t, np.float64)


def salvar(pasta, **arrs):
    os.makedirs(pasta, exist_ok=True)
    for k, v in arrs.items():
        np.save(os.path.join(pasta, k + ".npy"), v)


def casar(bf, da, db, ratio):
    if da is None or db is None or len(da) < 2 or len(db) < 2:
        return [], []
    pares = [p for p in bf.knnMatch(da, db, k=2) if p]
    return [p[0] for p in pares], [p[0] for p in pares if len(p) == 2 and p[0].distance < ratio * p[1].distance]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pasta_quadros")
    ap.add_argument("--saida", default="artifacts")
    ap.add_argument("--run-id", default=time.strftime("%Y%m%d_%H%M%S"))
    ap.add_argument("--nfeatures", type=int, default=5000, help="mesmo limite para os 3 detectores")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--ratio", type=float, default=0.75)
    ap.add_argument("--focal", type=float, default=None, help="em pixels (padrao: max(w, h))")
    a = ap.parse_args()

    arqs = sorted(glob.glob(os.path.join(a.pasta_quadros, "*.jpg")) + glob.glob(os.path.join(a.pasta_quadros, "*.png")))
    nomes = [os.path.splitext(os.path.basename(p))[0] for p in arqs]
    imgs = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in arqs]
    h, w = imgs[0].shape
    f = a.focal or max(w, h)
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float64)  # ponytail: intrinsecos aproximados, trocar pela calibracao se houver
    raiz = os.path.join(a.saida, a.run_id)
    salvar(raiz, K=K)
    print(f"{len(imgs)} quadros {w}x{h}, f={f:.1f}px -> {raiz}")

    for nome, (criar, norma, dim, tipo) in DETECTORES.items():
        det, bf = criar(a.nfeatures), cv2.BFMatcher(norma)
        res, tdet = [], []
        for q, g in zip(nomes, imgs):
            (kp, d), t = cronometrar(lambda: det.detectAndCompute(g, None), a.reps, a.warmup)
            kps = np.float32([[k.pt[0], k.pt[1], k.size, k.angle, k.response, k.octave] for k in kp]).reshape(-1, 6)
            d = np.zeros((0, dim), tipo) if d is None else d
            grade = np.histogram2d(kps[:, 1], kps[:, 0], bins=8, range=[[0, g.shape[0]], [0, g.shape[1]]])[0]
            salvar(os.path.join(raiz, "detection", nome, q), keypoints=kps, descriptors=d,
                   times_ms=t, grid_counts=grade.astype(np.int64))
            res.append((kps, d))
            tdet.append(t.mean())

        tmat, ninl = [], []
        for i in range(len(res) - 1):
            (ka, da), (kb, db) = res[i], res[i + 1]
            (brutos, lowe), t = cronometrar(lambda: casar(bf, da, db, a.ratio), a.reps, a.warmup)
            mascara, E = np.zeros(len(lowe), bool), np.full((3, 3), np.nan)
            if len(lowe) >= 5:
                pa = ka[[m.queryIdx for m in lowe], :2]
                pb = kb[[m.trainIdx for m in lowe], :2]
                Ee, mk = cv2.findEssentialMat(pa, pb, K, cv2.RANSAC, 0.999, 1.0)
                if Ee is not None and mk is not None:
                    E, mascara = Ee[:3].astype(np.float64), mk.ravel().astype(bool)
            arr = lambda ms: np.float32([[m.queryIdx, m.trainIdx, m.distance] for m in ms]).reshape(-1, 3)
            salvar(os.path.join(raiz, "matching", nome, f"{nomes[i]}__{nomes[i + 1]}"), matches_raw=arr(brutos),
                   matches_lowe=arr(lowe), inlier_mask=mascara, E=E, times_ms=t)
            tmat.append(t.mean())
            ninl.append(mascara.sum())
        print(f"{nome}: deteccao {np.mean(tdet):.1f} ms, {np.mean([len(k) for k, _ in res]):.0f} kps/quadro | "
              f"matching {np.mean(tmat) if tmat else 0:.1f} ms, {np.mean(ninl) if ninl else 0:.0f} inliers/par")


if __name__ == "__main__":
    main()

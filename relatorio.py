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

"""Tabelas e figuras SIFT x ORB x AKAZE a partir dos .npy salvos por detectores.py."""
import argparse, csv, os
import cv2
import numpy as np

COLS_DET = ["n_keypoints", "tempo_ms_media", "tempo_ms_desvio", "tempo_por_kp_ms", "cobertura_8x8",
            "entropia_8x8", "escala_p5", "escala_p95", "bytes_descritor"]
COLS_MAT = ["n_matches_raw", "n_matches_lowe", "inliers", "razao_inliers", "matching_score", "sucesso",
            "sampson_px", "tempo_matching_ms_media", "tempo_matching_ms_desvio"]


def ls(p):
    return sorted(d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))) if os.path.isdir(p) else []


def div(a, b):
    return a / b if b else np.nan


def imagem(pasta, q):
    for ext in (".jpg", ".png"):
        if os.path.exists(os.path.join(pasta, q + ext)):
            return cv2.imread(os.path.join(pasta, q + ext))
    return None


def salvar(path, img, lado=1600):
    s = lado / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 90])


def kps(kp):
    return [cv2.KeyPoint(float(x), float(y), float(s), float(a), float(r), int(o)) for x, y, s, a, r, o in kp]


def metricas_det(d):
    kp, desc = np.load(os.path.join(d, "keypoints.npy")), np.load(os.path.join(d, "descriptors.npy"))
    t, g = np.load(os.path.join(d, "times_ms.npy")), np.load(os.path.join(d, "grid_counts.npy")).ravel()
    n, p = len(kp), g[g > 0] / max(g.sum(), 1)
    return kp, [n, t.mean(), t.std(), div(t.mean(), n), (g > 0).mean(), float(-(p * np.log(p)).sum() / np.log(64)),
                *(np.percentile(kp[:, 2], [5, 95]) if n else [np.nan, np.nan]), div(desc.nbytes, n)]


def sampson(K, E, p1, p2):
    if len(p1) == 0 or np.isnan(E).any():
        return np.nan
    Ki = np.linalg.inv(K)
    F = Ki.T @ E @ Ki
    x1, x2 = np.c_[p1, np.ones(len(p1))], np.c_[p2, np.ones(len(p2))]
    Fx1, Ftx2 = x1 @ F.T, x2 @ F
    d = (x2 * Fx1).sum(1) ** 2 / (Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2)
    return float(np.sqrt(d).mean())


def escrever(path, cab, linhas):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cab)
        w.writerows(linhas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("pasta_quadros")
    ap.add_argument("--max-figuras", type=int, default=5, help="quadros por detector com figura de keypoints")
    a = ap.parse_args()
    K = np.load(os.path.join(a.run_dir, "K.npy"))
    met, fig = os.path.join(a.run_dir, "metrics"), os.path.join(a.run_dir, "figures")
    os.makedirs(met, exist_ok=True)
    det_rows, mat_rows, resumo = [], [], []

    for det in ls(os.path.join(a.run_dir, "detection")):
        base = os.path.join(a.run_dir, "detection", det)
        quadros = ls(base)
        com_fig = {quadros[int(i)] for i in np.linspace(0, len(quadros) - 1, min(a.max_figuras, len(quadros)))}
        pts, dr = {}, []
        for q in quadros:
            kp, m = metricas_det(os.path.join(base, q))
            pts[q] = kp
            dr.append(m)
            det_rows.append([det, q, *m])
            img = imagem(a.pasta_quadros, q) if q in com_fig else None
            if img is not None:
                img = cv2.drawKeypoints(img, kps(kp), None, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                cv2.putText(img, f"{det.upper()} {len(kp)} keypoints", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 3,
                            (0, 0, 255), 6)
                salvar(os.path.join(fig, "keypoints", det, q + ".jpg"), img)

        mr = []
        mbase = os.path.join(a.run_dir, "matching", det)
        for par in ls(mbase):
            qa, qb = par.split("__")
            L = lambda f: np.load(os.path.join(mbase, par, f))
            raw, lowe, mask, E, t = L("matches_raw.npy"), L("matches_lowe.npy"), L("inlier_mask.npy").astype(bool), L("E.npy"), L("times_ms.npy")
            inl = lowe[mask] if len(lowe) else lowe
            p1, p2 = pts[qa][inl[:, 0].astype(int), :2], pts[qb][inl[:, 1].astype(int), :2]
            ni = len(inl)
            m = [len(raw), len(lowe), ni, div(ni, len(lowe)), div(ni, min(len(pts[qa]), len(pts[qb]))), int(ni >= 8),
                 sampson(K, E, p1, p2), t.mean(), t.std()]
            mr.append(m)
            mat_rows.append([det, par, *m])
            if ni >= 8:
                ia, ib = imagem(a.pasta_quadros, qa), imagem(a.pasta_quadros, qb)
                if ia is not None and ib is not None:
                    dm = [cv2.DMatch(int(i), int(j), float(d)) for i, j, d in inl]
                    img = cv2.drawMatches(ia, kps(pts[qa]), ib, kps(pts[qb]), dm, None, (0, 255, 0), (0, 0, 255),
                                          flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
                    salvar(os.path.join(fig, "registro", det, par + ".jpg"), img)

        mr = np.array(mr, float).reshape(-1, len(COLS_MAT))
        resumo.append([det, len(quadros), len(mr), *np.nanmean(np.array(dr, float), 0),
                       *(np.nanmean(mr[:, [0, 1, 2, 3, 4, 6, 7]], 0) if len(mr) else [np.nan] * 7),
                       mr[:, 5].mean() if len(mr) else np.nan])

    escrever(os.path.join(met, "detection_metrics.csv"), ["detector", "quadro", *COLS_DET], det_rows)
    escrever(os.path.join(met, "matching_metrics.csv"), ["detector", "par", *COLS_MAT], mat_rows)
    cab = ["detector", "n_quadros", "n_pares", *COLS_DET, *[c for c in COLS_MAT if c not in ("sucesso", "tempo_matching_ms_desvio")],
           "taxa_sucesso"]
    escrever(os.path.join(met, "resumo.csv"), cab, resumo)

    # tabela transposta: uma linha por metrica, uma coluna por detector
    print(f"{'metrica':<26}" + "".join(f"{r[0]:>14}" for r in resumo))
    for i, c in enumerate(cab[1:], 1):
        print(f"{c:<26}" + "".join(f"{r[i]:>14.4g}" for r in resumo))
    print(f"CSVs em {met}, figuras em {fig}")


if __name__ == "__main__":
    main()

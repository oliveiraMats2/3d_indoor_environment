import argparse, csv, os
import cv2
import numpy as np


def extrair_features(video, stride, larg=640, nfeat=1000):
    cap = cv2.VideoCapture(video)
    orb = cv2.ORB_create(nfeat) # <<<< aqui
    idx, feats, nit = [], [], []
    i, g = 0, None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % stride == 0:
            s = larg / f.shape[1]
            g = cv2.cvtColor(cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_AREA),
                             cv2.COLOR_BGR2GRAY)
            kp, d = orb.detectAndCompute(g, None) # <<<< aqui
            pts = np.float32([k.pt for k in kp]) if kp else np.zeros((0, 2), np.float32)
            idx.append(i)
            feats.append((pts, d))
            nit.append(cv2.Laplacian(g, cv2.CV_64F).var())
        i += 1
    h, w = g.shape
    return np.array(idx), feats, np.array(nit), (w, h), cap.get(cv2.CAP_PROP_FPS)


class Sobreposicao: # <<<< aqui
    def __init__(self, feats, wh):
        self.feats, self.cache = feats, {}
        w, h = wh
        self.rect = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        self.area = float(w * h)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    def __call__(self, a, b):
        if (a, b) in self.cache:
            return self.cache[(a, b)]
        self.cache[(a, b)] = v = self._calc(a, b)
        return v

    def _calc(self, a, b):
        (pa, da), (pb, db) = self.feats[a], self.feats[b]
        if da is None or db is None or len(pa) < 20 or len(pb) < 20:
            return 0.0
        pares = [p for p in self.bf.knnMatch(db, da, k=2) if len(p) == 2]
        bons = [m for m, n in pares if m.distance < 0.75 * n.distance] # <<<< aqui distance: 0.75 parametro .yaml
        if len(bons) < 15:
            return 0.0
        src = np.float32([pb[m.queryIdx] for m in bons])
        dst = np.float32([pa[m.trainIdx] for m in bons])
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if H is None or mask.sum() < 12:
            return 0.0
        quad = cv2.perspectiveTransform(self.rect.reshape(-1, 1, 2), H).astype(np.float32)
        if not cv2.isContourConvex(quad):
            return 0.0
        area, _ = cv2.intersectConvexConvex(self.rect, quad.reshape(-1, 2))
        return float(min(area / self.area, 1.0))


def selecionar(n, ov, alvo):
    kept, last, prev = [0], 0, 0
    for j in range(1, n):
        if ov(last, j) < alvo:
            novo = prev if prev != last else j  
            kept.append(novo)
            last = novo
        prev = j
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--saida", default="imagens")
    ap.add_argument("--min", type=int, default=100, dest="nmin")
    ap.add_argument("--max", type=int, default=300, dest="nmax")
    ap.add_argument("--overlap", type=float, default=0.70, help="alvo inicial (0.60 a 0.80)")
    ap.add_argument("--stride", type=int, default=2, help="analisa 1 a cada N quadros")
    ap.add_argument("--largura", type=int, default=1920, help="largura maxima dos quadros salvos")
    a = ap.parse_args()

    print("Analisando o video...")
    idx, feats, nit, wh, fps = extrair_features(a.video, a.stride)
    ok = nit >= np.percentile(nit, 25)                    
    idx = idx[ok]
    feats = [f for f, o in zip(feats, ok) if o]
    print(f"{len(idx)} candidatos nitidos de {len(ok)} analisados ({fps:.1f} fps)")

    ov = Sobreposicao(feats, wh)
    alvo, passo = a.overlap, 0.025
    kept = selecionar(len(idx), ov, alvo)
    for _ in range(12):                                
        n = len(kept)
        if a.nmin <= n <= a.nmax:
            break
        novo = alvo - passo if n > a.nmax else alvo + passo
        if not 0.60 <= novo <= 0.80:
            break
        alvo = novo
        kept = selecionar(len(idx), ov, alvo)
    print(f"alvo final = {alvo:.3f} -> {len(kept)} quadros")

    if len(kept) > a.nmax:                                # ultimo recurso: subamostra uniforme
        kept = [kept[int(i)] for i in np.linspace(0, len(kept) - 1, a.nmax)]
        print(f"Subamostrado para {a.nmax} (alvo minimo 0.60 ainda gerou quadros demais).")
    if len(kept) < a.nmin:
        print(f"AVISO: so {len(kept)} quadros. Grave um video mais longo ou mais lento.")
        
    quadros = {int(idx[k]): n for n, k in enumerate(kept)}
    os.makedirs(a.saida, exist_ok=True)
    cap = cv2.VideoCapture(a.video)
    i = 0
    while quadros:
        okk, f = cap.read()
        if not okk:
            break
        if i in quadros:
            if f.shape[1] > a.largura:
                s = a.largura / f.shape[1] # <<<< aqui (imagem quadrada)
                f = cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(a.saida, f"img_{quadros.pop(i):04d}.jpg"), f,
                        [cv2.IMWRITE_JPEG_QUALITY, 95]) # <<<< aqui save com lossless
        i += 1

    #criacao do csv para registro dos quadros selecionados
    sobre = [None] + [ov(kept[k - 1], kept[k]) for k in range(1, len(kept))]
    with open(os.path.join(a.saida, "selecao.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["imagem", "frame_video", "tempo_s", "sobreposicao_com_anterior"])
        for n, k in enumerate(kept):
            w.writerow([f"img_{n:04d}.jpg", int(idx[k]), round(idx[k] / fps, 2),
                        "" if sobre[n] is None else round(sobre[n], 3)])
    v = [s for s in sobre if s is not None]
    if v:
        print(f"Sobreposicao entre consecutivos: min={min(v):.2f} media={np.mean(v):.2f} max={max(v):.2f}")


if __name__ == "__main__":
    main()
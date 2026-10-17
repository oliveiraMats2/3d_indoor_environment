"""Extrai do video os quadros com sobreposicao alvo (bloco `extraction` do YAML)."""
import logging
import os

import cv2
import numpy as np
from tqdm import tqdm

from src.extraction import features, video, writer
from src.extraction.overlap import Sobreposicao
from src.extraction.selection import adjust, subsample
from src.frame_quality import laplacian_var

log = logging.getLogger(__name__)


def run(cfg: dict) -> list[str]:
    log.info("Analisando o video...")
    orb = cv2.ORB_create(cfg["nfeatures"])
    idx, feats, nitidez = [], [], []
    total = -(-video.count(cfg["video"]) // cfg["stride"])  # ceil
    for i, f in tqdm(video.sample(cfg["video"], cfg["stride"]), total=total, desc="analisando video", unit="quadro"):
        gray = features.to_gray(f, cfg["analysis_width"])
        idx.append(i)
        feats.append(features.detect(orb, gray))
        nitidez.append(laplacian_var(gray))
        
    h, w = gray.shape
    fps = video.fps(cfg["video"])

    ok = np.array(nitidez) >= np.percentile(nitidez, cfg["sharp_percentile"])
    idx = [i for i, o in zip(idx, ok) if o]
    feats = [f for f, o in zip(feats, ok) if o]
    log.info("%d candidatos nitidos de %d analisados (%.1f fps)", len(idx), len(ok), fps)

    kept, alvo = adjust(len(idx), Sobreposicao(feats, (w, h), cfg), cfg)
    log.info("alvo final = %.3f -> %d quadros", alvo, len(kept))

    if len(kept) > cfg["max_frames"]:
        kept = subsample(kept, cfg["max_frames"])
        log.info("Subamostrado para %d (alvo minimo ainda gerou quadros demais).", cfg["max_frames"])

    if len(kept) < cfg["min_frames"]:
        log.warning("So %d quadros. Grave um video mais longo ou mais lento.", len(kept))

    os.makedirs(cfg["output_dir"], exist_ok=True)
    nomes = {idx[k]: f"img_{n:04d}.{cfg['image_format']}" for n, k in enumerate(kept)}
    for i, f in tqdm(video.read(cfg["video"], nomes), total=len(nomes), desc="salvando quadros", unit="quadro"):
        writer.save_frame(os.path.join(cfg["output_dir"], nomes[i]), f,
                          cfg["max_width"], cfg["jpeg_quality"])
        
    writer.save_csv(os.path.join(cfg["output_dir"], "selecao.csv"),
                    [[nome, i, round(i / fps, 2)] for i, nome in nomes.items()])
    
    return sorted(os.path.join(cfg["output_dir"], n) for n in nomes.values())

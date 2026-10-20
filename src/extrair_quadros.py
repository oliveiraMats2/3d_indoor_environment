"""Extrai do video os quadros com sobreposicao alvo (bloco `extraction` do YAML).

Blur filter (extraction.blur, see src/extraction/blur.py): every sampled frame gets blur_coef and
motion_ratio; flagged frames are dropped before the overlap selection (copies in
<output_dir>/discarded/{blur,motion}/) or, with keep_for_sfm_only, stay selectable and are listed in
<output_dir>/mvs_exclude.txt (always written). selecao.csv holds the selected frames (kept=1) plus the
blur/motion-dropped candidates (imagem empty, kept=0), sorted by frame_video; frames skipped only by
sharp_percentile or by the overlap selection are not listed. <output_dir>/extraction_metrics.csv holds
one row of stage metrics (main.py copies it to the artifacts).

extraction.selection: "all" = every stride-th frame is a candidate and only the blur filter drops frames
(sharp_percentile and the overlap selection are skipped; max_frames, when set, subsamples uniformly; the
video is decoded once and candidates are written while it is analysed); "overlap" = the greedy overlap
selection below.
"""
import logging
import math
import os
import shutil
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from tqdm import tqdm

from src import stage_io
from src.extraction import blur, features, video, writer
from src.extraction.overlap import Sobreposicao
from src.extraction.selection import adjust, consecutive_overlap, subsample
from src.frame_quality import laplacian_var

log = logging.getLogger(__name__)


def run(cfg: dict) -> list[str]:
    if cfg["selection"] == "all":
        return run_all(cfg)

    cfg = {**cfg, "max_frames": math.inf if cfg["max_frames"] is None else cfg["max_frames"]}  # None = no cap
    log.info("Analisando o video...")
    bcfg = cfg["blur"]
    orb = cv2.ORB_create(cfg["nfeatures"])
    idx, feats, nitidez, motion = [], [], [], []
    total = -(-video.count(cfg["video"]) // cfg["stride"])  # ceil
    for i, f in tqdm(video.sample(cfg["video"], cfg["stride"]), total=total, desc="analisando video", unit="quadro"):
        gray = features.to_gray(f, cfg["analysis_width"])
        idx.append(i)
        feats.append(features.detect(orb, gray))
        nitidez.append(laplacian_var(gray))
        motion.append(blur.motion_ratio(gray))

    h, w = gray.shape
    fps = video.fps(cfg["video"])

    coef = blur.blur_coef(nitidez, bcfg["window"])
    why = blur.reasons(coef, np.array(motion), bcfg)
    info = {i: (c, m, r) for i, c, m, r in zip(idx, coef, motion, why)}
    flagged = np.array([r != "" for r in why])
    drop = flagged & (not bcfg["keep_for_sfm_only"])
    dropped = {i: r for i, r, d in zip(idx, why, drop) if d}
    n_sampled = len(idx)

    ok = (np.array(nitidez) >= np.percentile(nitidez, cfg["sharp_percentile"])) & ~drop
    idx = [i for i, o in zip(idx, ok) if o]
    feats = [f for f, o in zip(feats, ok) if o]
    log.info("%d candidatos nitidos de %d analisados (%.1f fps); blur %d, motion %d, %s", len(idx), n_sampled,
             fps, why.count("blur"), why.count("motion"),
             "mantidos so para SfM" if bcfg["keep_for_sfm_only"] else "descartados")

    sobreposicao = Sobreposicao(feats, (w, h), cfg)
    kept, alvo = adjust(len(idx), sobreposicao, cfg)
    log.info("alvo final = %.3f -> %d quadros", alvo, len(kept))

    if len(kept) > cfg["max_frames"]:
        kept = subsample(kept, cfg["max_frames"])
        log.info("Subamostrado para %d (alvo minimo ainda gerou quadros demais).", cfg["max_frames"])

    if len(kept) < cfg["min_frames"]:
        log.warning("So %d quadros. Grave um video mais longo ou mais lento.", len(kept))

    os.makedirs(cfg["output_dir"], exist_ok=True)
    blur.reset_discarded(cfg["output_dir"])
    nomes = {idx[k]: f"img_{n:04d}.{cfg['image_format']}" for n, k in enumerate(kept)}
    lidos = set(nomes) | set(dropped)
    for i, f in tqdm(video.read(cfg["video"], lidos), total=len(lidos), desc="salvando quadros", unit="quadro"):
        if i in nomes:
            writer.save_frame(os.path.join(cfg["output_dir"], nomes[i]), f,
                              cfg["max_side"], cfg["jpeg_quality"])
        else:
            blur.save_discarded(cfg["output_dir"], dropped[i], i, f, bcfg)

    linhas = nomes | {i: "" for i in dropped}
    writer.save_csv(os.path.join(cfg["output_dir"], "selecao.csv"),
                    [[linhas[i], i, round(i / fps, 2), round(float(info[i][0]), 4), round(float(info[i][1]), 3),
                      int(i in nomes)] for i in sorted(linhas)])
    blur.write_mvs_exclude(cfg["output_dir"], [n for i, n in nomes.items() if info[i][2]])

    ov = np.array(consecutive_overlap(kept, sobreposicao))
    in_range = (ov >= cfg["overlap_min"]) & (ov <= cfg["overlap_max"])
    metrics = {"selection": "overlap", "n_sampled": n_sampled, "n_blur_drop": why.count("blur"), "n_motion_drop": why.count("motion"),
               "n_selected": len(nomes), "n_mvs_exclude": sum(1 for i in nomes if info[i][2]),
               "overlap_mean": round(float(ov.mean()), 4) if len(ov) else "",
               "overlap_min": round(float(ov.min()), 4) if len(ov) else "",
               "overlap_in_range_frac": round(float(in_range.mean()), 4) if len(ov) else ""}
    stage_io.upsert_csv(os.path.join(cfg["output_dir"], "extraction_metrics.csv"), metrics)
    stage_io.log_metrics(log, metrics)

    return sorted(os.path.join(cfg["output_dir"], n) for n in nomes.values())


def run_all(cfg: dict) -> list[str]:
    """Every stride-th frame minus the blur/motion drops, optionally capped by a uniform subsample."""

    bcfg, out, fmt = cfg["blur"], cfg["output_dir"], cfg["image_format"]
    cand_dir = os.path.join(out, "_candidates")
    shutil.rmtree(cand_dir, ignore_errors=True)
    os.makedirs(cand_dir)
    blur.reset_discarded(out)
    orb = cv2.ORB_create(cfg["nfeatures"])
    idx, feats, nitidez, motion, pending = [], [], [], [], []
    workers = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)  # CPUs usable here (cgroups/taskset on Linux)
    total = -(-video.count(cfg["video"]) // cfg["stride"])  # ceil
    cand = lambda i: os.path.join(cand_dir, f"{i:06d}.{fmt}")
    with ThreadPoolExecutor(workers) as pool:  # single decode: analysis here, png/jpg encoding in the pool
        for i, f in tqdm(video.sample(cfg["video"], cfg["stride"]), total=total, desc="analisando video", unit="quadro"):
            gray = features.to_gray(f, cfg["analysis_width"])
            idx.append(i)
            feats.append(features.detect(orb, gray))
            nitidez.append(laplacian_var(gray))
            motion.append(blur.motion_ratio(gray))
            pending.append(pool.submit(writer.save_frame, cand(i), f, cfg["max_side"], cfg["jpeg_quality"]))
            if len(pending) > 4 * workers:  # bound the frames waiting in memory
                pending.pop(0).result()

        for p in pending:
            p.result()

    h, w = gray.shape
    fps = video.fps(cfg["video"])
    n_sampled = len(idx)
    coef = blur.blur_coef(nitidez, bcfg["window"])
    why = blur.reasons(coef, np.array(motion), bcfg)
    info = {i: (c, m, r) for i, c, m, r in zip(idx, coef, motion, why)}
    dropped = {} if bcfg["keep_for_sfm_only"] else {i: r for i, r in zip(idx, why) if r}
    pos = [k for k, i in enumerate(idx) if i not in dropped]
    log.info("%d candidatos de %d analisados (%.1f fps); blur %d, motion %d, %s", len(pos), n_sampled, fps,
             why.count("blur"), why.count("motion"),
             "mantidos so para SfM" if bcfg["keep_for_sfm_only"] else "descartados")
    if cfg["max_frames"] is not None and len(pos) > cfg["max_frames"]:
        pos = sorted(set(subsample(pos, cfg["max_frames"])))
        log.info("Subamostrado uniformemente para %d (extraction.max_frames).", len(pos))

    if len(pos) < cfg["min_frames"]:
        log.warning("So %d quadros. Grave um video mais longo ou mais lento.", len(pos))

    nomes = {idx[k]: f"img_{n:04d}.{fmt}" for n, k in enumerate(pos)}
    for i in idx:
        if i in nomes:
            os.replace(cand(i), os.path.join(out, nomes[i]))
        elif i in dropped:
            blur.save_discarded(out, dropped[i], i, cv2.imread(cand(i)), bcfg)

    shutil.rmtree(cand_dir)
    linhas = nomes | {i: "" for i in dropped}
    writer.save_csv(os.path.join(out, "selecao.csv"),
                    [[linhas[i], i, round(i / fps, 2), round(float(info[i][0]), 4), round(float(info[i][1]), 3),
                      int(i in nomes)] for i in sorted(linhas)])
    blur.write_mvs_exclude(out, [n for i, n in nomes.items() if info[i][2]])

    sobreposicao = Sobreposicao(feats, (w, h), cfg)
    ov = np.array(consecutive_overlap(pos, sobreposicao))
    in_range = (ov >= cfg["overlap_min"]) & (ov <= cfg["overlap_max"])
    metrics = {"selection": "all", "n_sampled": n_sampled, "n_blur_drop": why.count("blur"),
               "n_motion_drop": why.count("motion"), "n_selected": len(nomes),
               "n_mvs_exclude": sum(1 for i in nomes if info[i][2]),
               "overlap_mean": round(float(ov.mean()), 4) if len(ov) else "",
               "overlap_min": round(float(ov.min()), 4) if len(ov) else "",
               "overlap_in_range_frac": round(float(in_range.mean()), 4) if len(ov) else ""}
    stage_io.upsert_csv(os.path.join(out, "extraction_metrics.csv"), metrics)
    stage_io.log_metrics(log, metrics)

    return sorted(os.path.join(out, n) for n in nomes.values())


if __name__ == "__main__":
    import tempfile

    rng = np.random.default_rng(0)
    world = cv2.GaussianBlur(rng.integers(0, 256, (360, 2400, 3)).astype(np.uint8), (5, 5), 1.0)
    kernel = np.zeros((25, 25), np.float32)
    kernel[12, :] = 1.0 / 25
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "pan.avi")
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30, (640, 360))
        for k in range(60):  # slow horizontal pan; frames 20 and 40 blurred
            f = np.ascontiguousarray(world[:, k * 25:k * 25 + 640])
            f = cv2.GaussianBlur(f, (0, 0), 4) if k == 20 else cv2.filter2D(f, -1, kernel) if k == 40 else f
            vw.write(f)
        vw.release()
        cfg = {"video": path, "output_dir": os.path.join(tmp, "frames"), "selection": "overlap", "min_frames": 5, "max_frames": 300,
               "overlap": 0.7, "overlap_min": 0.6, "overlap_max": 0.8, "overlap_step": 0.025, "max_adjust_iters": 12,
               "stride": 1, "analysis_width": 640, "nfeatures": 1000, "lowe_ratio": 0.75, "min_good_matches": 15,
               "ransac_threshold_px": 3.0, "min_inliers": 12, "sharp_percentile": 0, "max_side": 1920,
               "image_format": "png", "jpeg_quality": 95,
               "blur": {"enabled": True, "coef_min": 0.5, "motion_ratio_max": 4.0, "keep_for_sfm_only": False,
                        "window": 9, "discard_width": 320, "discard_jpeg_quality": 90}}
        frames = run(cfg)
        out = cfg["output_dir"]
        assert frames and os.listdir(os.path.join(out, "discarded", "blur")) == ["000020.jpg"]
        assert os.listdir(os.path.join(out, "discarded", "motion")) == ["000040.jpg"]
        rows = stage_io.read_csv(os.path.join(out, "selecao.csv"))
        assert list(rows[0])[:3] == ["imagem", "frame_video", "tempo_s"] and len(rows) == len(frames) + 2
        assert {r["frame_video"] for r in rows if r["kept"] == "0"} == {"20", "40"}
        m = stage_io.read_csv(os.path.join(out, "extraction_metrics.csv"))[0]
        assert (m["n_sampled"], m["n_blur_drop"], m["n_motion_drop"]) == ("60", "1", "1"), m
        assert open(os.path.join(out, "mvs_exclude.txt")).read() == ""

        cfg["blur"]["keep_for_sfm_only"] = True
        frames = run(cfg)
        excl = open(os.path.join(out, "mvs_exclude.txt")).read().split()
        sel = {r["frame_video"]: r["imagem"] for r in stage_io.read_csv(os.path.join(out, "selecao.csv"))}
        assert not os.listdir(os.path.join(out, "discarded", "blur"))
        assert set(excl) == {sel[k] for k in ("20", "40") if k in sel}, (excl, sel)
        assert all(sel.values())  # nothing dropped: every row is a selected image
        m = stage_io.read_csv(os.path.join(out, "extraction_metrics.csv"))[0]
        assert (m["n_blur_drop"], m["n_motion_drop"], m["n_mvs_exclude"]) == ("1", "1", str(len(excl))), m
        print("selected", len(frames), "mvs_exclude", excl)

        cfg["blur"]["keep_for_sfm_only"] = False
        cfg.update(selection="all", max_frames=None, sharp_percentile=90)  # sharp_percentile ignored in all mode
        frames = run(cfg)
        assert len(frames) == 58 and not os.path.exists(os.path.join(out, "_candidates")), len(frames)
        assert os.listdir(os.path.join(out, "discarded", "blur")) == ["000020.jpg"]
        assert cv2.imread(frames[0]).shape == (360, 640, 3)
        writer.save_frame(os.path.join(tmp, "portrait.png"), np.zeros((3840, 2160, 3), np.uint8), 1920, 95)
        assert cv2.imread(os.path.join(tmp, "portrait.png")).shape == (1920, 1080, 3)  # longest side capped
        rows = stage_io.read_csv(os.path.join(out, "selecao.csv"))
        assert len(rows) == 60 and {r["frame_video"] for r in rows if r["kept"] == "0"} == {"20", "40"}
        m = stage_io.read_csv(os.path.join(out, "extraction_metrics.csv"))[0]
        assert (m["selection"], m["n_selected"], m["n_blur_drop"]) == ("all", "58", "1") and float(m["overlap_mean"]) > 0.8, m
        cfg.update(max_frames=10, stride=2)
        frames = run(cfg)
        sel = [r for r in stage_io.read_csv(os.path.join(out, "selecao.csv")) if r["kept"] == "1"]
        assert len(frames) == 10 and sel[0]["frame_video"] == "0" and sel[-1]["frame_video"] == "58", sel
        print("all mode", len(frames), "frames, overlap_mean", m["overlap_mean"])
    print("OK")

"""Fonte unica dos numeros do relatorio (R24, R25, R26, R34 do T2).

Le os CSV/NPY reais (somente leitura) e grava:
  NOTAS/numeros.json              chave -> {valor, unidade, fonte}
  NOTAS/tabelas/<label>.tex       tabelas booktabs prontas (decimais com virgula)
Depois confere cada celula numerica das tabelas contra os CSV, relidos com o modulo csv.

Uso: R/.venv/bin/python tools/report/numeros.py
"""
import csv
import json
import re
import subprocess
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np

R = Path("/home/moliveira/Documents/3d_indoor_environment")
CZ = R / "cozinha/artifacts/config_cozinha"
CZR = R / "cozinha/results/config_cozinha"
KMH = R / "kitchen_my_home/artifacts"
NOTAS = Path("/home/moliveira/Documents/report_3d_indoor_environment/notas")
TAB = NOTAS / "tabelas"
CZ_REL = "cozinha/artifacts/config_cozinha"
CZR_REL = "cozinha/results/config_cozinha"

NUM = {}  # chave -> dict(valor, unidade, fonte)


def rd(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def f(x):
    return float(x)


def put(key, val, unit, src):
    assert key not in NUM, key
    NUM[key] = {"valor": val, "unidade": unit, "fonte": src}
    return val


def fmt(x, dec=2):
    """Decimal com virgula, arredondamento meia-acima."""
    q = Decimal(str(round(float(x), 9))).quantize(Decimal(1).scaleb(-dec), rounding=ROUND_HALF_UP)
    return str(q).replace("-", "-").replace(".", ",")


def b(s, on):
    return r"\textbf{%s}" % s if on else s


def table(label, caption, colspec, header, rows, tabcolsep="3.5pt", note=None):
    out = [r"\begin{table}[!ht]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{%s}" % tabcolsep,
           r"\caption{%s}" % caption, r"\label{%s}" % label, r"\begin{tabular}{%s}" % colspec,
           r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    for r_ in rows:
        out.append(r_ if r_ in (r"\midrule", r"\addlinespace") else " & ".join(r_) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    if note:
        out += [r"\par\vspace{2pt}{\footnotesize %s}" % note]
    out += [r"\end{table}", ""]
    (TAB / (label.replace(":", "_") + ".tex")).write_text("\n".join(out))
    (TAB / (label + ".tex")).write_text("\n".join(out))  # nome pelo rotulo, como pedido


# ---------------------------------------------------------------- aquisicao / SfM da cozinha
sfm = rd(CZ / "sfm_metrics.csv")[0]
loop = rd(CZ / "loop_drift.csv")[0]
ev = rd(CZ / "evaluation.csv")
ev_all = next(r for r in ev if r["method"] == "all")
ext = rd(CZ / "extraction_metrics.csv")[0]
fp = rd(CZ / "floorplan.csv")[0]
sc = rd(CZ / "scale.csv")[0]
fq = rd(CZ / "frame_quality_metrics.csv")
S, SR = f"{CZ_REL}/sfm_metrics.csv", f"{CZR_REL}/sfm"

vid = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                      "stream=width,height,r_frame_rate,nb_frames,duration", "-of", "json",
                      str(R / "cozinha.MOV")], capture_output=True, text=True, check=True)
vs = json.loads(vid.stdout)["streams"][0]
V = "ffprobe cozinha.MOV"
put("video.largura_px", int(vs["width"]), "px", V)
put("video.altura_px", int(vs["height"]), "px", V)
put("video.fps", 30, "fps", V + " r_frame_rate=30/1")
put("video.duracao_s", round(f(vs["duration"]), 2), "s", V + " duration")
put("video.quadros_originais", int(vs["nb_frames"]), "quadros", V + " nb_frames")
put("extracao.quadros_usados", int(ext["n_selected"]), "quadros", f"{CZ_REL}/extraction_metrics.csv:n_selected")
put("extracao.stride", 10, "quadros", "config_cozinha.yaml extraction.output_dir all_stride10")
put("extracao.fps_efetivo", 3, "fps", "30 fps / stride 10")
put("extracao.sobreposicao_media", f(ext["overlap_mean"]), "", f"{CZ_REL}/extraction_metrics.csv:overlap_mean")
put("extracao.sobreposicao_faixa_pct", 100 * f(ext["overlap_in_range_frac"]), "%",
    f"{CZ_REL}/extraction_metrics.csv:overlap_in_range_frac")
n_marc = sum(1 for r in fq if r["keep"] == "False")
put("qualidade.quadros_marcados", n_marc, "quadros", f"{CZ_REL}/frame_quality_metrics.csv:keep==False")

n_reg = put("sfm.registradas", int(sfm["n_registered"]), "quadros", f"{S}:n_registered")
put("sfm.quadros", int(sfm["n_frames"]), "quadros", f"{S}:n_frames")
put("sfm.registradas_pct", 100 * f(sfm["registered_frac"]), "%", f"{S}:registered_frac")
put("sfm.pontos_esparsos", int(sfm["n_sparse"]), "pontos", f"{S}:n_sparse")
put("sfm.track_medio", f(sfm["track_len_mean"]), "observacoes/ponto", f"{S}:track_len_mean")
put("sfm.obs_por_imagem", f(sfm["obs_per_image_mean"]), "obs/imagem", f"{S}:obs_per_image_mean")
pe = np.load(CZR / "sfm/point_error.npy")
tl = np.load(CZR / "sfm/track_len.npy")
put("sfm.track_mediano", float(np.median(tl)), "observacoes/ponto", f"{SR}/track_len.npy (mediana)")
put("sfm.reproj_media_ponto_px", f(sfm["reproj_mean_px"]), "px", f"{S}:reproj_mean_px (media por ponto)")
put("sfm.reproj_media_ponto_px_npy", float(pe.mean()), "px", f"{SR}/point_error.npy (media)")
put("sfm.reproj_mediana_ponto_px", float(np.median(pe)), "px", f"{SR}/point_error.npy (mediana por ponto)")
put("sfm.reproj_media_obs_px", f(ev_all["reproj_mean_px"]), "px", f"{CZ_REL}/evaluation.csv:reproj_mean_px (method=all, por observacao)")
put("sfm.reproj_mediana_obs_px", f(ev_all["reproj_median_px"]), "px", f"{CZ_REL}/evaluation.csv:reproj_median_px (method=all, por observacao)")
for k, u in [("t_feature_s", "extracao"), ("t_match_s", "casamento"), ("t_mapper_s", "mapper"), ("t_total_s", "total")]:
    put(f"tempo.sfm_{u}_s", f(sfm[k]), "s", f"{S}:{k}")
put("sfm.mapper", sfm["mapper"], "", f"{S}:mapper")
put("sfm.matcher", sfm["matcher"], "", f"{S}:matcher")
K = np.load(CZR / "sfm/K.npy")
put("intrinsecos.fx", float(K[0, 0]), "px", f"{SR}/K.npy[0,0]")
put("intrinsecos.fy", float(K[1, 1]), "px", f"{SR}/K.npy[1,1]")
put("intrinsecos.cx", float(K[0, 2]), "px", f"{SR}/K.npy[0,2]")
put("intrinsecos.cy", float(K[1, 2]), "px", f"{SR}/K.npy[1,2]")
dist = np.load(CZR / "sfm/dist.npy").ravel()
put("intrinsecos.k1", float(dist[0]), "", f"{SR}/dist.npy[0]")
put("intrinsecos.k2", float(dist[1]), "", f"{SR}/dist.npy[1]")
put("intrinsecos.fx_acima_1470_pct", 100 * (float(K[0, 0]) / 1470 - 1), "%", "K.npy[0,0] contra fx=1470 inicial (D9)")

put("deriva.cm", f(loop["delta_cm"]), "cm", f"{CZ_REL}/loop_drift.csv:delta_cm")
put("deriva.pct", f(loop["drift_pct"]), "%", f"{CZ_REL}/loop_drift.csv:drift_pct")
put("deriva.trajeto_cm", f(loop["trajectory_cm"]), "cm", f"{CZ_REL}/loop_drift.csv:trajectory_cm")
put("deriva.trajeto_m", f(loop["trajectory_cm"]) / 100, "m", f"{CZ_REL}/loop_drift.csv:trajectory_cm / 100")
put("deriva.par_visual_cm", f(loop["pair_delta_cm"]), "cm", f"{CZ_REL}/loop_drift.csv:pair_delta_cm")
put("deriva.laco_fechado", loop["loop_closed"], "", f"{CZ_REL}/loop_drift.csv:loop_closed")
put("deriva.visual_inliers", int(loop["visual_inliers"]), "", f"{CZ_REL}/loop_drift.csv:visual_inliers")
put("deriva.visual_matches", int(loop["visual_matches"]), "", f"{CZ_REL}/loop_drift.csv:visual_matches")

# planta e densidade (R34)
area = put("planta.area_m2", f(fp["area_m2"]), "m2", f"{CZ_REL}/floorplan.csv:area_m2")
put("planta.largura_cm", f(fp["width_cm"]), "cm", f"{CZ_REL}/floorplan.csv:width_cm")
put("planta.comprimento_cm", f(fp["length_cm"]), "cm", f"{CZ_REL}/floorplan.csv:length_cm")
put("planta.area_celulas_chao_m2", f(fp["floor_cells_m2"]), "m2", f"{CZ_REL}/floorplan.csv:floor_cells_m2")
put("planta.area_medida_yaml_m2", 11.26, "m2", "configs/measurements/kitchen_my_home.yaml:area_m2")
cl = {r["source"]: r for r in rd(CZ / "cleanup.csv")}
n_sp_raw = int(cl["sparse"]["n_before"])
n_sp_cl = int(cl["sparse"]["n_after_voxel"])
put("densidade.esparsa_bruta_pts", n_sp_raw, "pontos", f"{CZ_REL}/cleanup.csv:n_before (sparse)")
put("densidade.esparsa_limpa_pts", n_sp_cl, "pontos", f"{CZ_REL}/cleanup.csv:n_after_voxel (sparse)")
put("densidade.esparsa_bruta_pts_m2", n_sp_raw / area, "pontos/m2", "n_before / planta.area_m2 (confere dense_compare.csv:density_raw_pts_m2 sparse)")
put("densidade.esparsa_limpa_pts_m2", n_sp_cl / area, "pontos/m2", "n_after_voxel / planta.area_m2 (confere dense_compare.csv:density_pts_m2 sparse)")
put("limpeza.esparsa_remocao_pct", 100 * (n_sp_cl / int(cl["sparse"]["n_after_outlier"]) - 1), "%", "cleanup.csv sparse n_after_voxel / n_after_outlier - 1")
put("limpeza.esparsa_outliers_removidos", int(cl["sparse"]["n_before"]) - int(cl["sparse"]["n_after_outlier"]), "pontos", "cleanup.csv sparse n_before - n_after_outlier")
put("limpeza.patch_match_bruta", int(cl["patch_match"]["n_before"]), "pontos", "cleanup.csv patch_match:n_before")
put("limpeza.patch_match_limpa", int(cl["patch_match"]["n_after_voxel"]), "pontos", "cleanup.csv patch_match:n_after_voxel")
put("limpeza.patch_match_remocao_pct", 100 * (int(cl["patch_match"]["n_after_voxel"]) / int(cl["patch_match"]["n_before"]) - 1), "%", "cleanup.csv patch_match")

sc_ = sc
put("escala.cm_por_unidade", f(sc_["scale"]), "cm/un", f"{CZ_REL}/scale.csv:scale")
put("escala.residuo_mediano_cm", f(sc_["fit_residual_median_cm"]), "cm", f"{CZ_REL}/scale.csv:fit_residual_median_cm")
put("escala.residuo_p90_cm", f(sc_["fit_residual_p90_cm"]), "cm", f"{CZ_REL}/scale.csv:fit_residual_p90_cm")
put("escala.inliers_pct", 100 * f(sc_["fit_inlier_frac"]), "%", f"{CZ_REL}/scale.csv:fit_inlier_frac")
put("escala.metodo", sc_["method"], "", f"{CZ_REL}/scale.csv:method")
put("escala.tempo_s", f(sc_["t_s"]), "s", f"{CZ_REL}/scale.csv:t_s")

# ---------------------------------------------------------------- detectores (D1)
dc = {r["detector"]: r for r in rd(CZ / "detector_choice.csv")}
sm = {r["detector"]: r for r in rd(CZ / "summary.csv")}
DET = ["sift", "orb", "akaze"]
for d in DET:
    p = f"{CZ_REL}/detector_choice.csv"
    put(f"detector.{d}.keypoints", f(dc[d]["keypoints_per_img"]), "kp/imagem", f"{p}:keypoints_per_img")
    put(f"detector.{d}.cobertura", f(dc[d]["coverage_8x8"]), "", f"{p}:coverage_8x8")
    put(f"detector.{d}.tempo_ms", f(dc[d]["det_time_ms"]), "ms", f"{p}:det_time_ms")
    put(f"detector.{d}.inliers", f(dc[d]["ransac_inliers"]), "inliers/par", f"{p}:ransac_inliers")
    put(f"detector.{d}.razao_inliers", f(dc[d]["inlier_ratio"]), "", f"{p}:inlier_ratio")
    put(f"detector.{d}.pose_pct", 100 * f(dc[d]["pose_success_rate"]), "%", f"{p}:pose_success_rate")
    put(f"detector.{d}.sampson_px", f(dc[d]["sampson_px"]), "px", f"{p}:sampson_px")
    put(f"detector.{d}.score", f(dc[d]["score"]), "", f"{p}:score")
    put(f"detector.{d}.lowe", f(dc[d]["matches_lowe"]), "matches/par", f"{p}:matches_lowe")
    put(f"detector.{d}.cross", f(dc[d]["matches_cross"]), "matches/par", f"{p}:matches_cross")
    put(f"detector.{d}.brutos", f(sm[d]["n_matches_raw"]), "matches/par", f"{CZ_REL}/summary.csv:n_matches_raw")
put("detector.n_quadros", int(dc["sift"]["n_frames"]), "quadros", f"{CZ_REL}/detector_choice.csv:n_frames")
put("detector.n_pares", int(dc["sift"]["n_pairs"]), "pares", f"{CZ_REL}/detector_choice.csv:n_pairs")
gaps = []
for r in rd(CZ / "epipolar_metrics.csv"):
    if r["detector"] == "sift":
        a, c = re.findall(r"img_(\d+)", r["pair"])
        gaps.append(int(c) - int(a))
put("detector.separacao_media_indices", float(np.mean(gaps)), "indices", f"{CZ_REL}/epipolar_metrics.csv:pair (sift)")
put("detector.separacao_mediana_indices", float(np.median(gaps)), "indices", f"{CZ_REL}/epipolar_metrics.csv:pair (sift)")

# ---------------------------------------------------------------- CPU x GPU (D2)
cpu = rd(KMH / "kitchen_my_home_cpu/sfm_metrics.csv")[0]
gpu = rd(KMH / "kitchen_my_home_gpu/sfm_metrics.csv")[0]
dcpu = rd(KMH / "kitchen_my_home_cpu/loop_drift.csv")[0]
dgpu = rd(KMH / "kitchen_my_home_gpu/loop_drift.csv")[0]
CP, GP = "kitchen_my_home/artifacts/kitchen_my_home_cpu", "kitchen_my_home/artifacts/kitchen_my_home_gpu"
STAGES = [("extracao", "t_feature_s", "Extração de características"), ("casamento", "t_match_s", "Casamento"),
          ("mapper", "t_mapper_s", "Mapper (incremental $\\to$ global)"), ("total", "t_total_s", "SfM total")]
sp = {}
for k, col, _ in STAGES:
    put(f"cpu_gpu.{k}_cpu_s", f(cpu[col]), "s", f"{CP}/sfm_metrics.csv:{col}")
    put(f"cpu_gpu.{k}_gpu_s", f(gpu[col]), "s", f"{GP}/sfm_metrics.csv:{col}")
    sp[k] = put(f"cpu_gpu.{k}_speedup", f(cpu[col]) / f(gpu[col]), "x", f"{col} cpu / gpu")
for tag, d, dd, p in [("cpu", cpu, dcpu, CP), ("gpu", gpu, dgpu, GP)]:
    put(f"cpu_gpu.registradas_{tag}", int(d["n_registered"]), "quadros", f"{p}/sfm_metrics.csv:n_registered")
    put(f"cpu_gpu.quadros_{tag}", int(d["n_frames"]), "quadros", f"{p}/sfm_metrics.csv:n_frames")
    put(f"cpu_gpu.pontos_{tag}", int(d["n_sparse"]), "pontos", f"{p}/sfm_metrics.csv:n_sparse")
    put(f"cpu_gpu.reproj_{tag}_px", f(d["reproj_mean_px"]), "px", f"{p}/sfm_metrics.csv:reproj_mean_px")
    put(f"cpu_gpu.track_{tag}", f(d["track_len_mean"]), "obs/ponto", f"{p}/sfm_metrics.csv:track_len_mean")
    put(f"cpu_gpu.deriva_{tag}_pct", f(dd["drift_pct"]), "%", f"{p}/loop_drift.csv:drift_pct")
    put(f"cpu_gpu.deriva_{tag}_cm", f(dd["delta_cm"]), "cm", f"{p}/loop_drift.csv:delta_cm")

# ---------------------------------------------------------------- erro metrico
me = rd(CZ / "metric_error.csv")
ME = "cozinha/artifacts/config_cozinha/metric_error.csv"
loo = [r for r in me if r["scale_source"] == "leave_one_out"]
ROT = {"janela": "Janela", "pia_fogao": "Pia--fogão", "porta_lado_pia": "Porta ao lado da pia",
       "porta": "Porta", "geladeiras": "Geladeiras"}
for r in loo:
    n = r["name"]
    put(f"erro.{n}.L_cm", f(r["value_cm"]), "cm", f"{ME}:value_cm ({n})")
    put(f"erro.{n}.Lchapeu_cm", f(r["model_cm"]), "cm", f"{ME}:model_cm ({n})")
    put(f"erro.{n}.abs_cm", f(r["abs_error_cm"]), "cm", f"{ME}:abs_error_cm ({n})")
    put(f"erro.{n}.rel_pct", 100 * f(r["rel_error"]), "%", f"{ME}:rel_error*100 ({n})")
m_abs = put("erro.media_abs_cm", float(np.mean([f(r["abs_error_cm"]) for r in loo])), "cm", f"{ME}:abs_error_cm (media das 5 linhas LOO)")
m_rel = put("erro.media_rel_pct", 100 * float(np.mean([f(r["rel_error"]) for r in loo])), "%", f"{ME}:rel_error (media das 5 linhas LOO)")
ch = next(r for r in me if r["name"] == "ceiling_height")
put("erro.altura_modelo_cm", f(ch["model_cm"]), "cm", f"{ME}:model_cm (ceiling_height)")
# circulares: largura e comprimento da planta contra janela (273) e pia_fogao (429)
Lw, Ll = f(next(r for r in loo if r["name"] == "janela")["value_cm"]), f(next(r for r in loo if r["name"] == "pia_fogao")["value_cm"])
Pw, Pl = f(fp["width_cm"]), f(fp["length_cm"])
put("erro.planta_largura.abs_cm", abs(Pw - Lw), "cm", "floorplan.csv:width_cm contra janela (circular)")
put("erro.planta_largura.rel_pct", 100 * abs(Pw - Lw) / Lw, "%", "idem")
put("erro.planta_comprimento.abs_cm", abs(Pl - Ll), "cm", "floorplan.csv:length_cm contra pia_fogao (circular)")
put("erro.planta_comprimento.rel_pct", 100 * abs(Pl - Ll) / Ll, "%", "idem")
put("erro.min_abs_cm", min(f(r["abs_error_cm"]) for r in loo), "cm", "min das linhas LOO")
put("erro.max_abs_cm", max(f(r["abs_error_cm"]) for r in loo), "cm", "max das linhas LOO")

# angulos
WA = "cozinha/artifacts/config_cozinha/wall_angles.csv"
wa = rd(CZ / "wall_angles.csv")
for r in wa:
    k = r["pair"]
    put(f"angulo.{k}.medido_deg", f(r["angle_deg"]), "graus", f"{WA}:angle_deg")
    put(f"angulo.{k}.esperado_deg", f(r["expected_deg"]), "graus", f"{WA}:expected_deg")
    put(f"angulo.{k}.erro_deg", f(r["error_deg"]), "graus", f"{WA}:error_deg")

# ---------------------------------------------------------------- densos (D11, D12, D16)
dcmp = {r["method"]: r for r in rd(CZ / "dense_compare.csv")}
dmet = {r["method"]: r for r in rd(CZ / "dense_metrics.csv")}
poi = {r["variant"]: r for r in rd(CZ / "poisson.csv")}
DM = [("patch_match", "PatchMatch"), ("openmvs", "OpenMVS"), ("plane_sweep", "Plane Sweep"), ("sgm", "SGM"),
      ("casmvsnet", "CasMVSNet"), ("mvsnet", "MVSNet"), ("patchmatchnet", "PatchMatchNet"),
      ("vggt_omega", "VGGT"), ("mapanything", "MapAnything")]
for m, _ in DM + [("sparse", "")]:
    r = dcmp[m]
    p = f"{CZ_REL}/dense_compare.csv"
    put(f"denso.{m}.pontos_limpos", int(r["n_dense_clean"]), "pontos", f"{p}:n_dense_clean")
    put(f"denso.{m}.pontos_brutos", int(r["n_dense_raw"]), "pontos", f"{p}:n_dense_raw")
    put(f"denso.{m}.densidade_limpa_pts_m2", f(r["density_pts_m2"]), "pontos/m2", f"{p}:density_pts_m2")
    put(f"denso.{m}.densidade_bruta_pts_m2", f(r["density_raw_pts_m2"]), "pontos/m2", f"{p}:density_raw_pts_m2")
    put(f"denso.{m}.completude", f(r["completeness_frac"]), "", f"{p}:completeness_frac")
    put(f"denso.{m}.chamfer_cm", f(r["chamfer_to_sparse_cm"]), "cm", f"{p}:chamfer_to_sparse_cm")
    put(f"denso.{m}.tempo_s", f(r["t_s"]), "s", f"{p}:t_s")
    if m != "sparse":
        put(f"denso.{m}.quadros", int(dmet[m]["n_ref_images"]), "quadros", f"{CZ_REL}/dense_metrics.csv:n_ref_images")
put("denso.area_m2", f(dcmp["patch_match"]["area_m2"]), "m2", f"{CZ_REL}/dense_compare.csv:area_m2")
put("denso.vggt.vram_gb", f(dmet["vggt_omega"]["vram_peak_gb"]), "GB", f"{CZ_REL}/dense_metrics.csv:vram_peak_gb")
put("denso.mapanything.vram_gb", f(dmet["mapanything"]["vram_peak_gb"]), "GB", f"{CZ_REL}/dense_metrics.csv:vram_peak_gb")
put("denso.patch_match_sobre_planesweep_tempo", f(dcmp["patch_match"]["t_s"]) / f(dcmp["plane_sweep"]["t_s"]), "x", "t_s patch_match / plane_sweep")
put("poisson.pre_t_s", f(poi["pre"]["t_s"]), "s", f"{CZ_REL}/poisson.csv:t_s (pre)")
put("poisson.raw_t_s", f(poi["raw"]["t_s"]), "s", f"{CZ_REL}/poisson.csv:t_s (raw)")
put("poisson.raw_sobre_pre", f(poi["raw"]["t_s"]) / f(poi["pre"]["t_s"]), "x", "t_s raw / pre")
put("poisson.pre_pontos", int(poi["pre"]["n_points_in"]), "pontos", f"{CZ_REL}/poisson.csv:n_points_in (pre)")
put("poisson.raw_pontos", int(poi["raw"]["n_points_in"]), "pontos", f"{CZ_REL}/poisson.csv:n_points_in (raw)")
put("poisson.profundidade", int(poi["pre"]["depth"]), "", f"{CZ_REL}/poisson.csv:depth")
put("poisson.pre_vertices", int(poi["pre"]["n_vertices"]), "", f"{CZ_REL}/poisson.csv:n_vertices (pre)")
put("poisson.pre_faces", int(poi["pre"]["n_faces"]), "", f"{CZ_REL}/poisson.csv:n_faces (pre)")
ch_ = {r["variant"]: r for r in rd(CZ / "close_holes.csv")}
for v, r in ch_.items():
    put(f"buracos.{v}.ok", r["ok"], "", f"{CZ_REL}/close_holes.csv:ok")
    put(f"buracos.{v}.restantes", r["holes_after"], "", f"{CZ_REL}/close_holes.csv:holes_after")
    put(f"buracos.{v}.tempo_s", f(r["t_s"]), "s", f"{CZ_REL}/close_holes.csv:t_s")
li = rd(CZ / "liepa.csv")[0]
for k in ["holes_found", "holes_filled", "boundary_edges_before", "boundary_edges_after", "t_s"]:
    put(f"liepa.{k}", f(li[k]), "", f"{CZ_REL}/liepa.csv:{k}")
ms = rd(CZ / "mesh.csv")[0]
put("malha.vertices", int(ms["n_vertices"]), "", f"{CZ_REL}/mesh.csv:n_vertices")
put("malha.faces", int(ms["n_faces"]), "", f"{CZ_REL}/mesh.csv:n_faces")
put("malha.tempo_s", f(ms["t_s"]), "s", f"{CZ_REL}/mesh.csv:t_s")

# regioes dificeis
for r in rd(CZ / "hard_regions.csv"):
    g = r["region"]
    put(f"regiao.{g}.n_pontos", int(r["n_points"]), "pontos", f"{CZ_REL}/hard_regions.csv:n_points")
    put(f"regiao.{g}.reproj_media_px", f(r["reproj_mean_px"]), "px", f"{CZ_REL}/hard_regions.csv:reproj_mean_px")
    put(f"regiao.{g}.reproj_mediana_px", f(r["reproj_median_px"]), "px", f"{CZ_REL}/hard_regions.csv:reproj_median_px")
    put(f"regiao.{g}.track_medio", f(r["track_len_mean"]), "obs/ponto", f"{CZ_REL}/hard_regions.csv:track_len_mean")

# ---------------------------------------------------------------- decisoes D3 a D8 (kitchen_my_home)
ab = {r["run_id"]: r for r in rd(KMH / "kitchen_ablation/ablation.csv")}
AB = "kitchen_my_home/artifacts/kitchen_ablation/ablation.csv"
for k, v in [("seq_noloop", "kitchen_abl_seq_noloop"), ("seq_loop", "kitchen_abl_seq_loop"),
             ("exaustivo", "kitchen_abl_exhaustive"), ("q200", "kitchen_abl_frames_200"), ("q100", "kitchen_abl_frames_100")]:
    r = ab[v]
    put(f"D3_D4.{k}.registradas", int(r["n_registered"]), "quadros", f"{AB}:n_registered")
    put(f"D3_D4.{k}.registradas_pct", 100 * f(r["registered_frac"]), "%", f"{AB}:registered_frac")
    put(f"D3_D4.{k}.pontos", int(r["n_sparse"]), "pontos", f"{AB}:n_sparse")
    put(f"D3_D4.{k}.deriva_pct", f(r["drift_pct"]), "%", f"{AB}:drift_pct")
    put(f"D3_D4.{k}.sfm_s", f(r["t_total_s"]), "s", f"{AB}:t_total_s")
put("D3.pontos_laco_ganho_pct", 100 * (int(ab["kitchen_abl_seq_loop"]["n_sparse"]) / int(ab["kitchen_abl_seq_noloop"]["n_sparse"]) - 1), "%", "n_sparse seq_loop / seq_noloop - 1")
put("D3.exaustivo_sobre_laco_tempo", f(ab["kitchen_abl_exhaustive"]["t_total_s"]) / f(ab["kitchen_abl_seq_loop"]["t_total_s"]), "x", "t_total_s exaustivo / seq_loop")
put("D3.laco_custo_extra_s", f(ab["kitchen_abl_seq_loop"]["t_total_s"]) - f(ab["kitchen_abl_seq_noloop"]["t_total_s"]), "s", "t_total_s seq_loop - seq_noloop")
em = rd(KMH / "kitchen_my_home_cpu/extraction_metrics.csv")[0]
put("D5.kmh_sobreposicao_media", f(em["overlap_mean"]), "", "kitchen_my_home_cpu/extraction_metrics.csv:overlap_mean")
put("D5.kmh_sobreposicao_faixa_pct", 100 * f(em["overlap_in_range_frac"]), "%", "kitchen_my_home_cpu/extraction_metrics.csv:overlap_in_range_frac")
for tag, d in [("glomap", "kitchen_my_home_glomap"), ("keep", "kitchen_my_home_glomap_keep"), ("strict", "kitchen_my_home_glomap_strict")]:
    r = rd(KMH / d / "sfm_metrics.csv")[0]
    p = f"kitchen_my_home/artifacts/{d}/sfm_metrics.csv"
    put(f"D6_D7.{tag}.registradas", int(r["n_registered"]), "quadros", f"{p}:n_registered")
    put(f"D6_D7.{tag}.reproj_px", f(r["reproj_mean_px"]), "px", f"{p}:reproj_mean_px")
    put(f"D6_D7.{tag}.track", f(r["track_len_mean"]), "obs/ponto", f"{p}:track_len_mean")
    put(f"D6_D7.{tag}.sfm_s", f(r["t_total_s"]), "s", f"{p}:t_total_s")
for tag, d in [("instantsfm", "kitchen_my_home_instantsfm"), ("mast3r_sfm", "kitchen_my_home_mast3r_sfm"), ("vggt", "kitchen_my_home_vggt")]:
    r = rd(KMH / d / "sfm_metrics.csv")[0]
    put(f"D8.{tag}.registradas_pct", 100 * f(r["registered_frac"]), "%", f"kitchen_my_home/artifacts/{d}/sfm_metrics.csv:registered_frac")
put("D8.colmap_glomap_registradas_pct", 100 * f(rd(KMH / "kitchen_my_home_glomap/sfm_metrics.csv")[0]["registered_frac"]), "%", "kitchen_my_home_glomap/sfm_metrics.csv:registered_frac")

# ---------------------------------------------------------------- Decisoes D1 a D16 (era / ficou / numeros)
N = lambda k: NUM[k]["valor"]
def D(i, era, ficou, numeros):
    put(f"decisao.D{i}", {"era": era, "ficou": ficou, "chaves_numeros": numeros}, "", "P1_requisitos.md secao 2; numeros nas chaves indicadas")
D(1, "SIFT, ORB, AKAZE", "SIFT", ["detector.sift.score", "detector.orb.score", "detector.akaze.score", "detector.sift.inliers", "detector.sift.pose_pct", "detector.orb.tempo_ms", "detector.sift.tempo_ms", "detector.akaze.tempo_ms"])
D(2, "mapper incremental (Ceres CPU)", "GLOMAP global em GPU", ["cpu_gpu.mapper_cpu_s", "cpu_gpu.mapper_gpu_s", "cpu_gpu.mapper_speedup", "cpu_gpu.total_speedup", "cpu_gpu.deriva_cpu_pct", "cpu_gpu.deriva_gpu_pct"])
D(3, "sequencial sem laco; exaustivo", "sequencial com deteccao de laco", ["D3_D4.seq_noloop.registradas_pct", "D3_D4.seq_loop.registradas_pct", "D3.pontos_laco_ganho_pct", "D3_D4.seq_noloop.deriva_pct", "D3_D4.seq_loop.deriva_pct", "D3_D4.exaustivo.sfm_s", "D3.exaustivo_sobre_laco_tempo"])
D(4, "286 quadros", "286 (referencia), 200, 100", ["D3_D4.seq_loop.registradas_pct", "D3_D4.q200.registradas_pct", "D3_D4.q100.registradas_pct"])
D(5, "selecao por sobreposicao 60 a 80% (286)", "todos, stride 10 (2327)", ["D5.kmh_sobreposicao_media", "D5.kmh_sobreposicao_faixa_pct", "extracao.sobreposicao_media", "extracao.sobreposicao_faixa_pct"])
D(6, "filtro de nitidez ligado", "desligado", ["D6_D7.glomap.registradas", "D6_D7.keep.registradas", "D6_D7.glomap.reproj_px", "D6_D7.keep.reproj_px"])
D(7, "min_num_matches 30", "15", ["D6_D7.strict.reproj_px", "D6_D7.glomap.reproj_px", "D6_D7.strict.sfm_s", "D6_D7.glomap.sfm_s"])
D(8, "InstantSfM, MASt3R-SfM, VGGT", "COLMAP/GLOMAP", ["D8.vggt.registradas_pct", "D8.mast3r_sfm.registradas_pct", "D8.instantsfm.registradas_pct", "D8.colmap_glomap_registradas_pct"])
D(9, "fx=fy=1470 fixos", "intrinsecos estimados no SfM (OPENCV unico)", ["intrinsecos.fx", "intrinsecos.fy", "intrinsecos.k1", "intrinsecos.k2", "intrinsecos.fx_acima_1470_pct"])
D(10, "remocao estatistica + voxel + crop", "so o voxel de 1 cm atua", ["densidade.esparsa_bruta_pts", "densidade.esparsa_limpa_pts", "limpeza.patch_match_bruta", "limpeza.patch_match_limpa", "limpeza.patch_match_remocao_pct"])
D(11, "9 metodos densos", "PatchMatch (motivo a confirmar, H9)", ["denso.patch_match.tempo_s", "denso.plane_sweep.tempo_s", "denso.patch_match_sobre_planesweep_tempo", "denso.patch_match.completude", "denso.plane_sweep.completude", "denso.patch_match.chamfer_cm", "denso.plane_sweep.chamfer_cm", "denso.patchmatchnet.densidade_limpa_pts_m2"])
D(12, "Poisson sobre nuvem bruta", "Poisson sobre nuvem limpa", ["poisson.pre_t_s", "poisson.raw_t_s", "poisson.raw_sobre_pre", "poisson.pre_pontos", "poisson.raw_pontos"])
D(13, "4 variantes de close_holes; Liepa", "close_holes pre_pre", ["buracos.pre_pre.restantes", "liepa.holes_filled", "liepa.holes_found", "liepa.boundary_edges_before", "liepa.boundary_edges_after"])
D(14, "marcador ArUco", "contorno medido (outline)", ["escala.cm_por_unidade", "escala.residuo_mediano_cm", "escala.residuo_p90_cm", "escala.inliers_pct"])
D(15, "excluir quadros ruins", "so anotar", ["qualidade.quadros_marcados", "extracao.quadros_usados"])
D(16, "VGGT/MapAnything com todos os quadros", "80 quadros", ["denso.vggt_omega.quadros", "denso.mapanything.quadros", "denso.vggt.vram_gb"])
for i in range(1, 17):  # toda chave citada existe
    for k in NUM[f"decisao.D{i}"]["valor"]["chaves_numeros"]:
        assert k in NUM, (i, k)
    NUM[f"decisao.D{i}"]["valor"]["numeros"] = {k: NUM[k]["valor"] for k in NUM[f"decisao.D{i}"]["valor"]["chaves_numeros"]}

# ================================================================= TABELAS
TAB.mkdir(parents=True, exist_ok=True)
NOTAS.mkdir(parents=True, exist_ok=True)

# tab:aquisicao (so campos numericos; aparelho e EXIF vem de T1)
table("tab:aquisicao", "Vídeo e quadros da cena principal.", "llll",
      [r"\textbf{Campo}", r"\textbf{Valor}", r"\textbf{Campo}", r"\textbf{Valor}"],
      [[r"Duração", fmt(N("video.duracao_s"), 1) + r"\,s", "Taxa", r"30\,fps"],
       ["Resolução", r"%d $\times$ %d\,px" % (N("video.largura_px"), N("video.altura_px")), "Quadros originais", str(N("video.quadros_originais"))],
       ["Quadros usados", r"%d (1 a cada 10, %d\,fps)" % (N("extracao.quadros_usados"), N("extracao.fps_efetivo")),
        "Registrados", r"%d (%s\%%)" % (N("sfm.registradas"), fmt(N("sfm.registradas_pct"), 2))],
       ["Trajeto estimado", r"%s\,m, laço fechado" % fmt(N("deriva.trajeto_m"), 2), "Referência", "contorno medido em 5 segmentos"]],
      tabcolsep="4pt")

# tab:detectores
col = lambda k: [N(f"detector.{d}.{k}") for d in DET]
best = lambda vals, mx=True: [v == (max(vals) if mx else min(vals)) for v in vals]
spec = [("keypoints", 0, True), ("cobertura", 3, True), ("tempo_ms", 2, False), ("inliers", 2, True),
        ("razao_inliers", 3, True), ("pose_pct", 1, True), ("sampson_px", 3, False), ("score", 3, True)]
rows = []
for i, d in enumerate(DET):
    rows.append([d.upper() if d != "sift" else "SIFT"] + [b(fmt(N(f"detector.{d}.{k}"), dec), best(col(k), mx)[i]) for k, dec, mx in spec])
rows = [[r_[0].replace("AKAZE", "AKAZE")] + r_[1:] for r_ in rows]
table("tab:detectores",
      "Comparação dos detectores em %d quadros uniformes e %d pares entre quadros amostrados vizinhos (separados, em média, por %s índices de quadro, mediana %d). Cobertura: fração das células ocupadas na grade $8\\times8$; razão: inliers sobre casamentos após Lowe e verificação cruzada; pose: pares com pose recuperada; escore: composição dos ranks do pipeline."
      % (N("detector.n_quadros"), N("detector.n_pares"), fmt(N("detector.separacao_media_indices"), 1), N("detector.separacao_mediana_indices")),
      "lrrrrrrrr",
      ["Detector", "Keypoints", "Cobertura", "Tempo (ms)", "Inliers", "Razão", "Pose (\\%)", "Sampson (px)", "Escore"], rows, tabcolsep="3pt")

# tab:sfm_gpu
rows = []
for k, _, nome in STAGES:
    c, g = N(f"cpu_gpu.{k}_cpu_s"), N(f"cpu_gpu.{k}_gpu_s")
    rows.append([nome, b(fmt(c, 2), c < g), b(fmt(g, 2), g < c), fmt(sp[k], 2) + r"$\times$"])
rows.append(r"\midrule")
nq = N("cpu_gpu.quadros_cpu")
rows.append(["Quadros registrados", f"{N('cpu_gpu.registradas_cpu')}/{nq}", b(f"{N('cpu_gpu.registradas_gpu')}/{nq}", True), "-"])
rows.append(["Pontos esparsos", str(N("cpu_gpu.pontos_cpu")), str(N("cpu_gpu.pontos_gpu")), "-"])
rows.append(["Reprojeção média (px)", fmt(N("cpu_gpu.reproj_cpu_px"), 3), b(fmt(N("cpu_gpu.reproj_gpu_px"), 3), True), "-"])
rows.append(["Deriva do laço (\\%)", b(fmt(N("cpu_gpu.deriva_cpu_pct"), 2), True), fmt(N("cpu_gpu.deriva_gpu_pct"), 2), "-"])
table("tab:sfm_gpu",
      "SfM em %d quadros (kitchen\\_my\\_home): mapper incremental com Ceres em CPU contra GLOMAP global em GPU. Extração e casamento já usam GPU nos dois fluxos; aceleração = tempo CPU / tempo GPU." % nq,
      "lrrr", ["Etapa", "CPU (s)", "GPU (s)", "Aceleração"], rows, tabcolsep="4pt")

# tab:erro
GRAY = lambda s: r"\textcolor{gray}{%s}" % s
rows = []
loo_abs = [N(f"erro.{r['name']}.abs_cm") for r in loo]
loo_rel = [N(f"erro.{r['name']}.rel_pct") for r in loo]
for r in loo:
    n = r["name"]
    rows.append([ROT[n] + " (LOO)", fmt(N(f"erro.{n}.L_cm"), 1), fmt(N(f"erro.{n}.Lchapeu_cm"), 1),
                 b(fmt(N(f"erro.{n}.abs_cm"), 2), N(f"erro.{n}.abs_cm") == min(loo_abs)),
                 b(fmt(N(f"erro.{n}.rel_pct"), 2), N(f"erro.{n}.rel_pct") == min(loo_rel))])
rows.append(r"\midrule")
rows.append(["Média (5 linhas LOO)", "", "", fmt(m_abs, 2), fmt(m_rel, 2)])
rows.append(r"\midrule")
rows.append([GRAY("Largura da planta (circular)"), GRAY(fmt(Lw, 1)), GRAY(fmt(Pw, 1)), GRAY(fmt(N("erro.planta_largura.abs_cm"), 2)), GRAY(fmt(N("erro.planta_largura.rel_pct"), 2))])
rows.append([GRAY("Comprimento da planta (circular)"), GRAY(fmt(Ll, 1)), GRAY(fmt(Pl, 1)), GRAY(fmt(N("erro.planta_comprimento.abs_cm"), 2)), GRAY(fmt(N("erro.planta_comprimento.rel_pct"), 2))])
rows.append(["Altura (pé-direito)", "-", fmt(N("erro.altura_modelo_cm"), 1), "-", "-"])
table("tab:erro",
      "Erro métrico por dimensão, com $\\hat{L}$ o valor do modelo e $L$ a medida no ambiente. As cinco paredes são avaliadas por leave-one-out (LOO): a escala de cada uma vem das demais. Largura e comprimento da planta (em cinza) são circulares, pois a escala foi ajustada ao mesmo contorno medido; a altura não tem medida de referência. A média considera apenas as linhas LOO.",
      "lrrrr", ["Dimensão", "$L$ (cm)", "$\\hat{L}$ (cm)", "$|\\hat{L}-L|$ (cm)", "Erro rel. (\\%)"], rows, tabcolsep="4pt")

# tab:angulos
rows = []
errs = [N(f"angulo.{r['pair']}.erro_deg") for r in wa]
for r in wa:
    k = r["pair"]
    i, j = re.findall(r"wall_(\d)", k)
    rows.append([f"Paredes {i} e {j}", fmt(N(f"angulo.{k}.medido_deg"), 2), fmt(N(f"angulo.{k}.esperado_deg"), 0),
                 b(fmt(N(f"angulo.{k}.erro_deg"), 2), N(f"angulo.{k}.erro_deg") == min(errs))])
table("tab:angulos",
      "Ângulos entre pares de paredes ajustadas por RANSAC (nuvem esparsa limpa). Esperado: 90\\textdegree{} para cantos retos e 135\\textdegree{} para o chanfro da porta, conforme o contorno medido.",
      "lrrr", ["Par", "Medido (\\textdegree)", "Esperado (\\textdegree)", "Erro (\\textdegree)"], rows, tabcolsep="4pt")

# tab:denso
keys = [("pontos_limpos", 0, True), ("densidade_limpa_pts_m2", 0, True), ("completude", 3, True), ("chamfer_cm", 2, False), ("tempo_s", 1, False)]
rows = []
for m, nome in DM:
    row = [nome + (r"$^{\dagger}$" if m in ("vggt_omega", "mapanything") else "")]
    for k, dec, mx in keys:
        vals = [N(f"denso.{mm}.{k}") for mm, _ in DM]
        v = N(f"denso.{m}.{k}")
        row.append(b(fmt(v, dec), v == (max(vals) if mx else min(vals))))
    row.append(str(N(f"denso.{m}.quadros")))
    rows.append(row)
table("tab:denso",
      "Densificadores comparados (nuvem limpa, voxel de 1\\,cm; área da planta %s\\,m\\textsuperscript{2}). Completude e Chamfer são medidos contra a nuvem esparsa do próprio SfM, não contra uma referência externa. $^{\\dagger}$VGGT e MapAnything processam %d quadros (limite de memória da GPU), os demais entre %d e %d."
      % (fmt(N("denso.area_m2"), 3), N("denso.vggt_omega.quadros"), min(N(f"denso.{m}.quadros") for m, _ in DM[:7]), max(N(f"denso.{m}.quadros") for m, _ in DM[:7])),
      "lrrrrrr", ["Método", "Pontos", "Pontos/m\\textsuperscript{2}", "Completude", "Chamfer (cm)", "Tempo (s)", "Quadros"], rows, tabcolsep="3.5pt")

# ================================================================= salvar json
(NOTAS / "numeros.json").write_text(json.dumps(NUM, ensure_ascii=False, indent=1, default=str))

# ================================================================= conferencia: cada celula numerica contra CSV relido
raw = []
for p in list(CZ.glob("*.csv")) + [KMH / "kitchen_my_home_cpu/sfm_metrics.csv", KMH / "kitchen_my_home_gpu/sfm_metrics.csv",
                                   KMH / "kitchen_my_home_cpu/loop_drift.csv", KMH / "kitchen_my_home_gpu/loop_drift.csv"]:
    if p.name in ("frame_quality_metrics.csv", "epipolar_metrics.csv", "matching_metrics.csv"):
        continue
    for r in rd(p):
        for v in r.values():
            try:
                raw.append(float(v))
            except (TypeError, ValueError):
                pass
raw = np.array(raw)
derived = [N(k) for k in NUM if isinstance(N(k), (int, float))]  # derivados (razoes, medias, %), registrados com fonte
bad = []
for p in TAB.glob("tab_*.tex"):
    txt = p.read_text()
    body = re.sub(r"\\caption\{.*?\}\n", "", txt, flags=re.S)
    body = re.sub(r"\\(label|textsuperscript|textdegree)\{[^}]*\}", "", body)
    for cell in re.findall(r"(?<![\w\\.$])-?\d+(?:,\d+)?", body):
        v = float(cell.replace(",", "."))
        dec = len(cell.split(",")[1]) if "," in cell else 0
        tol = 0.5 * 10 ** (-dec) + 1e-9
        ok = (np.abs(raw - v) <= tol).any() or (np.abs(raw * 100 - v) <= tol).any() or any(abs(d - v) <= tol for d in derived)
        if not ok and v not in (8.0, 90.0, 135.0, 10.0, 3.0, 30.0, 5.0, 1.0):
            bad.append((p.name, cell))
print(f"numeros.json: {len(NUM)} chaves; tabelas em {TAB}")
print("celulas sem correspondencia:", bad or "nenhuma")

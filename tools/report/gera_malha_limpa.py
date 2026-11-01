"""Gera a nuvem densa e a malha 3D limpas da rodada config_cozinha (figuras G8, G9 e G13 do relatorio).

Uso (na raiz do repo): R/.venv/bin/python tools/report/gera_malha_limpa.py [run_dir] [out_dir]
Entradas (cm): clean/patch_match/{points,colors}.npy, mesh/room_fill/{T_room,footprint_cm}.npy, planes/floor.npy,
scale/centers_cm.npy. Nada existente e sobrescrito; saida em <run>/malha_limpa/.
Teto removido: h > 278 cm na nuvem e na malha (a versao com teto usava 300).
Etapas: (1) espurios da nuvem (criterio de images/report/scripts/densa.py); (2) Poisson profundidade 10
(normais como src/mesh/poisson.py); (3) corte por densidade q=0,02 e componentes < 1% da maior;
(4) recorte da malha pelo contorno da sala + 20 cm e altura [-10, 278] cm, e nova remocao de componentes pequenas;
(5) exporta ply/obj/glb e render de conferencia.
"""
import json, sys, time
from pathlib import Path
import numpy as np, open3d as o3d
from scipy.spatial import cKDTree
from matplotlib.path import Path as MPath

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.mesh import poisson  # noqa: E402

RUN = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/moliveira/Documents/3d_indoor_environment/cozinha/results/config_cozinha")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else RUN / "malha_limpa"); OUT.mkdir(parents=True, exist_ok=True)
MARGEM, H_PISO, H_TETO, K_VIZ, K_STD = 20.0, -10.0, 278.0, 20, 2.0  # H_TETO: teto removido acima disto (teto em 280-292 cm; a versao anterior usava 300)
DEPTH, KNN, QDENS, MIN_FRAC, THREADS = 10, 30, 0.02, 0.01, 8
T = {}

def ext(v): return (np.ptp(v, axis=0)).round(1).tolist()

def dist_poly(xy, fp):
    dentro = MPath(np.vstack([fp, fp[:1]])).contains_points(xy)
    d = np.full(len(xy), np.inf)
    for a, b in zip(fp, np.roll(fp, -1, 0)):
        ab = b - a; t = np.clip(((xy - a) @ ab) / (ab @ ab), 0, 1)
        d = np.minimum(d, np.linalg.norm(xy - (a + t[:, None] * ab), axis=1))
    return np.where(dentro, 0.0, d)

def poda(m):  # componentes < MIN_FRAC * maior
    lab, n, _ = m.cluster_connected_triangles(); lab, n = np.asarray(lab), np.asarray(n)
    m.remove_triangles_by_mask(n[lab] < MIN_FRAC * n.max()); m.remove_unreferenced_vertices()

# --- 1. nuvem
t0 = time.perf_counter()
p = np.load(RUN / "clean/patch_match/points.npy").astype(np.float64).reshape(-1, 3)
c = np.load(RUN / "clean/patch_match/colors.npy").reshape(-1, 3)
c8 = (c if c.max() > 1.5 else c * 255).astype(np.uint8)
Ti = np.linalg.inv(np.load(RUN / "mesh/room_fill/T_room.npy"))
fp = np.load(RUN / "mesh/room_fill/footprint_cm.npy")
tr = lambda v: v @ Ti[:3, :3].T + Ti[:3, 3]
q = tr(p)
c_fora = dist_poly(q[:, :2], fp) > MARGEM
c_alt = (q[:, 2] < H_PISO) | (q[:, 2] > H_TETO)
base = ~(c_fora | c_alt)
dk = cKDTree(q).query(q, k=K_VIZ + 1, workers=-1)[0][:, 1:].mean(1)
lim = dk[base].mean() + K_STD * dk[base].std()
c_flut = base & (dk > lim)
esp = c_fora | c_alt | c_flut
cont = dict(total=len(p), fora_contorno=int(c_fora.sum()), fora_altura=int((c_alt & ~c_fora).sum()),
            flutuante_ruido=int(c_flut.sum()), espurios=int(esp.sum()), limiar_dk_cm=round(float(lim), 3))
print(cont, flush=True)
keep = ~esp
cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p[keep]))
cloud.colors = o3d.utility.Vector3dVector(c8[keep] / 255.0)
o3d.io.write_point_cloud(str(OUT / "nuvem_limpa.ply"), cloud)
T["nuvem_s"] = time.perf_counter() - t0

# --- 2. Poisson
t0 = time.perf_counter()
cen = poisson.load_centers(RUN)
poisson.estimate_normals(cloud, p[keep], cen, {"normal_knn": KNN, "device": "cpu"})
mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud, depth=DEPTH, n_threads=THREADS)
dens = np.asarray(dens); T["poisson_s"] = time.perf_counter() - t0
n_pois = (len(mesh.vertices), len(mesh.triangles)); e_pois = ext(np.asarray(mesh.vertices))

# --- 3. densidade + componentes
mesh.remove_vertices_by_mask(dens < np.quantile(dens, QDENS)); mesh.remove_unreferenced_vertices(); poda(mesh)
n_dens = (len(mesh.vertices), len(mesh.triangles)); e_dens = ext(np.asarray(mesh.vertices))

# --- 4. recorte pela sala (aplicado)
v = tr(np.asarray(mesh.vertices))
fora = (dist_poly(v[:, :2], fp) > MARGEM) | (v[:, 2] < H_PISO) | (v[:, 2] > H_TETO)
mesh.remove_vertices_by_mask(fora); mesh.remove_unreferenced_vertices(); poda(mesh); mesh.compute_vertex_normals()
n_fin = (len(mesh.vertices), len(mesh.triangles)); vf = np.asarray(mesh.vertices)
e_fin = ext(vf); e_sala = ext(tr(vf))

# --- 5. exporta
o3d.io.write_triangle_mesh(str(OUT / "malha_limpa.ply"), mesh)
o3d.io.write_triangle_mesh(str(OUT / "malha_limpa.obj"), mesh)
o3d.io.write_triangle_mesh(str(OUT / "malha_limpa.glb"), mesh)
res = dict(nuvem=cont, nuvem_antes=len(p), nuvem_depois=int(keep.sum()), ext_nuvem_antes=ext(p), ext_nuvem_depois=ext(p[keep]),
           ext_nuvem_depois_sala=ext(q[keep]), poisson=n_pois, ext_poisson=e_pois, apos_densidade=n_dens, ext_apos_densidade=e_dens,
           final=n_fin, ext_final=e_fin, ext_final_sala=e_sala, recorte_vertices_removidos=int(fora.sum()),
           tempos_s={k: round(x, 1) for k, x in T.items()})
(OUT / "resumo.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))

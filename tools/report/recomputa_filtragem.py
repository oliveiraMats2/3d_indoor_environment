"""Recomputes an explicit outlier filtering for figures G8, G9, G10 of the report.

Sparse cloud (cm): Open3D statistical outlier (nb_neighbors=20, std_ratio 100/2.0/1.0), masks aligned to the rows of
sfm/points3d.npy. Mesh: Poisson depth 10 on clean/patch_match, density-quantile trim (0.02/0.05/0.10) + small
component removal. Nothing existing is overwritten; output goes to <run>/report_filtragem/.
Usage: python tools/report/recomputa_filtragem.py <run_dir> [out_dir]   (run from the repo root)
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.mesh import poisson  # noqa: E402  (reuses normals orientation of the pipeline)

NB, DEPTH, KNN, MIN_FRAC = 20, 10, 30, 0.01
run = Path(sys.argv[1])
out = Path(sys.argv[2]) if len(sys.argv) > 2 else run / "report_filtragem"
out.mkdir(parents=True, exist_ok=True)
rows = []

# --- sparse outliers (G8, G10)
pts = np.load(run / "scale" / "points3d_cm.npy").astype(np.float64)
pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
for std in (100.0, 2.0, 1.0):
    t0 = time.perf_counter()
    _, idx = pcd.remove_statistical_outlier(nb_neighbors=NB, std_ratio=std)
    keep = np.zeros(len(pts), bool)
    keep[np.asarray(idx)] = True
    np.save(out / f"outlier_mask_std{std}.npy", ~keep)  # True = outlier
    rows.append(dict(etapa="esparsa_outlier", parametros=f"nb={NB} std={std}", antes=len(pts), mantidos=int(keep.sum()),
                     removidos=int((~keep).sum()), pct_removidos=round(100 * (~keep).mean(), 3),
                     t_s=round(time.perf_counter() - t0, 2)))
    print(rows[-1], flush=True)

# --- Poisson before/after density trim (G9)
t0 = time.perf_counter()
p = np.load(run / "clean/patch_match/points.npy").astype(np.float64).reshape(-1, 3)
c = np.load(run / "clean/patch_match/colors.npy").astype(np.uint8).reshape(-1, 3)
cen = poisson.load_centers(run)
cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p))
cloud.colors = o3d.utility.Vector3dVector(c / 255.0)
poisson.estimate_normals(cloud, p, cen, {"normal_knn": KNN, "device": "cpu"})
mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud, depth=DEPTH, n_threads=-1)
dens = np.asarray(dens)
t_pois = time.perf_counter() - t0
nv0, nf0 = len(mesh.vertices), len(mesh.triangles)
o3d.io.write_triangle_mesh(str(out / "poisson_q0.00.ply"), mesh)
np.save(out / "poisson_densities.npy", dens.astype(np.float32))
rows.append(dict(etapa="poisson", parametros=f"depth={DEPTH} q=0", antes=nv0, mantidos=nv0, removidos=0,
                 pct_removidos=0.0, faces_antes=nf0, faces_depois=nf0, t_s=round(t_pois, 1)))
for q in (0.02, 0.05, 0.10):
    m = o3d.geometry.TriangleMesh(mesh)
    m.remove_vertices_by_mask(dens < np.quantile(dens, q))
    m.remove_unreferenced_vertices()
    lab, n, area = m.cluster_connected_triangles()
    lab, n = np.asarray(lab), np.asarray(n)
    m.remove_triangles_by_mask(n[lab] < MIN_FRAC * n.max())
    m.remove_unreferenced_vertices()
    m.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(out / f"poisson_q{q:.2f}.ply"), m)
    nv, nf = len(m.vertices), len(m.triangles)
    rows.append(dict(etapa="poisson_trim", parametros=f"q={q} comp_min={MIN_FRAC}*maior", antes=nv0, mantidos=nv,
                     removidos=nv0 - nv, pct_removidos=round(100 * (nv0 - nv) / nv0, 2), faces_antes=nf0,
                     faces_depois=nf, t_s=0))
    print(rows[-1], flush=True)

keys = ["etapa", "parametros", "antes", "mantidos", "removidos", "pct_removidos", "faces_antes", "faces_depois", "t_s"]
with open(out / "filtragem.csv", "w", newline="") as f:
    w = csv.DictWriter(f, keys, restval="")
    w.writeheader()
    w.writerows(rows)

"""Render de conferencia da malha limpa, mesma vista de images/report/malha_antes_depois.py (sem corte adicional; malha ja sem teto).
Uso: R/.venv/bin/python tools/report/render_malha_limpa.py [run_dir] [saida.jpg]"""
import sys, time, numpy as np, open3d as o3d
from pathlib import Path
from PIL import Image
from open3d.visualization import rendering as rd
RUN = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/moliveira/Documents/3d_indoor_environment/cozinha/results/config_cozinha")
SAIDA = sys.argv[2] if len(sys.argv) > 2 else str(RUN / "malha_limpa/render_malha_limpa.jpg")
CORTE, W, H, AZ, EL, DIST = 1000.0, 1000, 820, 0.0, 55.0, 640.0
m = o3d.io.read_triangle_mesh(str(RUN / "malha_limpa/malha_limpa.ply"))
v, f = np.asarray(m.vertices), np.asarray(m.triangles)
pl = np.load(RUN / "planes/floor.npy"); up = pl[:3] / np.linalg.norm(pl[:3])
alt = v @ pl[:3] + pl[3]
sel = alt[f].mean(1) <= CORTE
m.remove_triangles_by_mask(~sel); m.remove_unreferenced_vertices(); m.compute_vertex_normals()
v = np.asarray(m.vertices); ctr = v.mean(0)
base = np.cross(up, [1, 0, 0]); base /= np.linalg.norm(base); side = np.cross(base, up)
a, e = np.radians(AZ), np.radians(EL)
eye = ctr + DIST * (np.cos(e) * (np.cos(a) * base + np.sin(a) * side) + np.sin(e) * up)
r = rd.OffscreenRenderer(W, H)
mat = rd.MaterialRecord(); mat.shader = "defaultLit"
r.scene.add_geometry("m", m, mat); r.scene.show_skybox(False)
r.scene.set_lighting(r.scene.LightingProfile.NO_SHADOWS, [0.3, -0.5, -0.8])
r.setup_camera(55, ctr, eye, up)
def quadro(bg):
    r.scene.set_background(bg); ant = None
    for _ in range(60):
        im = np.array(r.render_to_image()).astype(int); time.sleep(1.0)
        if ant is not None and np.abs(im - ant).mean() < 3 and im.sum() > 0: break
        ant = im
    return im
img = quadro([1, 1, 1, 1])
Image.fromarray(img.astype(np.uint8)[..., :3]).convert("RGB").save(SAIDA, quality=85, optimize=True)
print(SAIDA)

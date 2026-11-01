"""T5: dimensoes da cozinha (paredes e pe-direito por picos de histograma) no LiDAR e no modelo alinhado."""
import sys, json
from pathlib import Path
import numpy as np
W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W))
from src.evaluation import lidar as L

out = W / "outputs/t5_lidar"
rng = np.random.default_rng(0)
raw = L.load_lidar_points("/home/moliveira/Documents/3d_indoor_environment/cozinha_lidar/fused/lidar_fused.ply", "m", 1.0)
li = L.clean_cloud(raw, 1.0)
fr = L.manhattan_frame(li, 8.0, np.array([0, 1.0, 0]), rng)   # eixos x,y da sala, z para cima
cc = li @ fr.T

def peak(v, a, b):
    h, e = np.histogram(v, bins=np.arange(a, b, 1.0)); i = h.argmax()
    sel = (v > e[i] - 3) & (v < e[i] + 4)
    return float(np.median(v[sel]))

def dims(q, s=1.0):
    inner = (q[:, 0] > -480) & (q[:, 0] < -110) & (q[:, 1] > 20) & (q[:, 1] < 260)
    fl, ce = peak(q[inner, 2], -115, -85), peak(q[inner, 2], 185, 225)
    band = (q[:, 2] > fl + 30) & (q[:, 2] < fl + 250)
    xs = q[band & (q[:, 1] > 20) & (q[:, 1] < 260), 0]; ys = q[band & (q[:, 0] > -480) & (q[:, 0] < -110), 1]
    x0, x1 = peak(xs, -540, -490), peak(xs, -100, -50); y0, y1 = peak(ys, -20, 20), peak(ys, 260, 300)
    d = {"x_parede_a_parede_cm": (x1 - x0), "y_parede_a_parede_cm": (y1 - y0), "pe_direito_cm": ce - fl,
         "piso_z": fl, "teto_z": ce, "paredes_x": [x0, x1], "paredes_y": [y0, y1]}
    return {k: ([vv / s for vv in v] if isinstance(v, list) else v / s) if k.endswith("_cm") or k.startswith("paredes") else v for k, v in d.items()}

res = {"lidar": dims(cc)}
for name, rel in [("clean_patch_match", "run/clean/patch_match/points.npy"), ("sparse", "run/scale/points3d_cm.npy")]:
    T = np.load(out / f"run/lidar/T_{name}.npy"); s = L.decompose(T)[0]
    p = np.load(out / rel).reshape(-1, 3)
    rec = L.voxel(L.apply(T, L.voxel(p, 0.5 / s)), 1.0) @ fr.T
    res[name + "_alinhado"] = dims(rec)
    res[name + "_nativo"] = dims(rec, s)   # dividido pela escala = unidades do modelo (cm)
json.dump(res, open(out / "dims.json", "w"), indent=1)
for k, v in res.items(): print(k, {a: (round(b, 1) if isinstance(b, float) else b) for a, b in v.items() if "cm" in a})

"""Video de navegacao da cozinha (E2): malha limpa sem teto + trajetoria das 2326 cameras + frustums de uma amostra.

Uso (na raiz do repo report):
  R/.venv/bin/python tools/report/video_navegacao.py [--run RUN] [--saida MP4] [--cache DIR] [--previa t1,t2,...]
--previa salva PNGs (1920x1080) dos instantes pedidos em DIR/previa (ou na pasta da saida) e nao gera o video.
--cache DIR guarda/le a malha simplificada (evita repetir a simplificacao, ~40 s).

Entradas (run): malha_limpa/malha_limpa.ply (cm, cor por vertice), mesh/room_fill/T_room.npy (sala -> cm),
scale/centers_cm.npy, exports/colmap/sparse/0/{images,cameras}.txt (poses; escala scale/scale.npy cm/unidade).
Tudo e desenhado no referencial da sala (x, y no piso, z = altura em cm): p_sala = inv(T_room) p_cm.
Render: Open3D OffscreenRenderer (Filament) em 3840x2160, reduzido para 1920x1080 (antialiasing), texto com PIL,
quadros enviados ao ffmpeg (libx264, yuv420p, crf 20, preset slow, +faststart). Nao sobrescreve arquivo existente.
Roteiro (50 s, 30 fps): titulo e vista de cima; trajetoria desenhada progressivamente; orbita obliqua; descida
para o nivel dos olhos (150 cm) com passeio por geladeira, bancada e pia, fogao; subida e titulo final.
"""
import argparse, subprocess, sys, time
from pathlib import Path
import numpy as np, open3d as o3d
from open3d.visualization import rendering as rd
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as Rot

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="/home/moliveira/Documents/3d_indoor_environment/cozinha/results/config_cozinha")
ap.add_argument("--saida", default=None)
ap.add_argument("--cache", default=None)
ap.add_argument("--previa", default=None)
ap.add_argument("--faces", type=int, default=1_800_000)
A = ap.parse_args()
RUN = Path(A.run)
SAIDA = Path(A.saida) if A.saida else RUN / "videos/navegacao_cozinha_1080p.mp4"
W, H, SS, FPS, DUR = 1920, 1080, 2, 30, 50.0
N = int(DUR * FPS)
FUNDO = (0.955, 0.950, 0.935)
TITULO = "Reconstrução 3D da cozinha"
FONTE = "/usr/share/fonts/truetype/noto/NotoSans-%s.ttf"
FRUST_N, FRUST_D = 70, 16.0          # frustums da amostra e profundidade (cm)
EYE_H = 150.0                        # altura do olho no passeio (cm)

# ---------------------------------------------------------------- dados
Ti = np.linalg.inv(np.load(RUN / "mesh/room_fill/T_room.npy"))
sala = lambda p: p @ Ti[:3, :3].T + Ti[:3, 3]

def malha():
    cache = Path(A.cache) / f"malha_sala_{A.faces}.ply" if A.cache else None
    if cache is not None and cache.exists():
        m = o3d.io.read_triangle_mesh(str(cache))
    else:
        m = o3d.io.read_triangle_mesh(str(RUN / "malha_limpa/malha_limpa.ply"))
        print("malha:", len(m.triangles), "faces", flush=True)
        m.transform(Ti)
        if len(m.triangles) > A.faces:
            m = m.simplify_quadric_decimation(A.faces)
        m.compute_vertex_normals()
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True); o3d.io.write_triangle_mesh(str(cache), m)
    print("malha usada:", len(m.triangles), "faces", flush=True)
    return m

def poses_colmap():
    sp = RUN / "exports/colmap/sparse/0"
    s = float(np.load(RUN / "scale/scale.npy").ravel()[0])
    cam = [l.split() for l in open(sp / "cameras.txt") if not l.startswith("#")][0]
    w, h, fx, fy, cx, cy = int(cam[2]), int(cam[3]), *map(float, cam[4:8])
    linhas = [l for l in open(sp / "images.txt") if not l.startswith("#")][0::2]
    d = {}
    for l in linhas:
        t = l.split(); q = list(map(float, t[1:5])); tr = np.array(list(map(float, t[5:8])))
        d[t[9]] = (Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix(), tr)
    nomes = sorted(d)
    Rs = np.array([d[n][0] for n in nomes]); C = np.array([-d[n][0].T @ d[n][1] for n in nomes]) * s
    return Rs, C, (w, h, fx, fy, cx, cy), s

Rs, Ccm, (IW, IH, FX, FY, CX, CY), ESC = poses_colmap()
cen_ref = np.load(RUN / "scale/centers_cm.npy")
err = np.abs(Ccm - cen_ref).max()
print(f"poses COLMAP x {ESC:.4f} cm vs centers_cm.npy: diferenca maxima {err:.2e} cm", flush=True)
assert err < 1e-3, "poses do COLMAP nao coincidem com scale/centers_cm.npy"
NCAM = len(Ccm)
TRAJ = sala(Ccm)

def frustum(i, d):
    """5 pontos (centro + 4 cantos a profundidade d) no referencial da sala."""
    uv = np.array([[0, 0], [IW, 0], [IW, IH], [0, IH]], float)
    raios = np.c_[(uv[:, 0] - CX) / FX, (uv[:, 1] - CY) / FY, np.ones(4)] * d
    return sala(np.vstack([Ccm[i], Ccm[i] + raios @ Rs[i]]))   # R^T r = r @ R

AMOSTRA = np.linspace(0, NCAM - 1, FRUST_N).round().astype(int)
FR = {i: frustum(i, FRUST_D) for i in AMOSTRA}
ARESTAS = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]

def cor_tempo(u):  # gradiente azul -> laranja ao longo da captura
    a, b = np.array([0.02, 0.22, 0.70]), np.array([0.85, 0.22, 0.00])
    u = np.asarray(u)[..., None]; return a * (1 - u) + b * u

# ---------------------------------------------------------------- roteiro de camera
CTR = np.array([-10.0, -22.0, 70.0])   # centro da sala (referencial da sala)

def esf(az, el, dist, alvo=CTR):
    a, e = np.radians(az), np.radians(el)
    return alvo + dist * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])

# (t, olho, alvo, fov vertical)
K = []
def kf(t, olho, alvo, fov): K.append((t, np.asarray(olho, float), np.asarray(alvo, float), float(fov)))
# 0-14 s: de cima (eixo longo da sala na horizontal), aproximacao lenta enquanto a trajetoria e desenhada
INI = CTR + [45, 0, 0]   # sala mais baixa na tela no inicio (espaco para o titulo)
for t, az, el, dist, alvo in [(0, 180, 84, 900, INI), (4, 182, 82, 860, INI), (9, 188, 78, 760, CTR), (14, 196, 72, 740, CTR)]:
    kf(t, esf(az, el, dist, alvo), alvo, 34)
# 14-27 s: orbita obliqua lenta (desce de 72 para 58 graus; abaixo disso as paredes escondem o interior)
for j, t in enumerate(np.arange(15.5, 26.6, 1.5)):
    u = (t - 14) / 12.5
    kf(t, esf(196 + 214 * u, 72 - 14 * u, 740 - 40 * u), CTR, 34 + 6 * u)
# az final ~ 410 = 50 graus: olho a nordeste. Entra pela abertura do teto em direcao a porta norte
kf(28.6, [10, 170, 380], [-5, -60, 90], 50)
kf(30.4, [0, 115, 225], [-8, -150, 95], 62)
kf(32.0, [-2, 95, EYE_H], [-10, -200, 95], 66)
# passeio no nivel dos olhos
# ordem geladeira -> pia -> fogao: o olhar gira sempre no mesmo sentido (sudeste, sul, oeste), sem meia-volta
kf(34.6, [-25, 30, EYE_H], [100, -140, 105], 64)      # geladeira (canto sudeste, x 70-130 cm)
kf(37.2, [-15, -40, EYE_H], [-15, -215, 92], 66)      # bancada e pia (parede sul)
kf(39.8, [15, -75, EYE_H], [-150, -85, 80], 64)       # fogao (parede oeste)
kf(41.9, [25, -55, 185], [-70, -110, 85], 62)
# subida e vista final de cima
kf(44.0, [5, -10, 330], [-5, -40, 60], 52)
FIM = CTR + [28, 0, 0]   # desloca a sala para baixo na tela (espaco para o titulo)
kf(46.5, esf(186, 80, 860, FIM), FIM, 36)
kf(50.0, esf(180, 84, 900, FIM), FIM, 34)

tk = np.array([k[0] for k in K])
sp_olho = CubicSpline(tk, np.array([k[1] for k in K]), bc_type="clamped")
sp_alvo = CubicSpline(tk, np.array([k[2] for k in K]), bc_type="clamped")
sp_fov = CubicSpline(tk, np.array([k[3] for k in K]), bc_type="clamped")
TT = np.arange(N) / FPS
OLHO = gaussian_filter1d(sp_olho(TT), 6, axis=0, mode="nearest")
ALVO = gaussian_filter1d(sp_alvo(TT), 6, axis=0, mode="nearest")
FOV = gaussian_filter1d(sp_fov(TT), 6, mode="nearest")

# trajetoria desenhada entre 4 e 13 s (ease in-out)
def suave(x): x = np.clip(x, 0, 1); return x * x * (3 - 2 * x)
def n_traj(t): return int(round(NCAM * suave((t - 4.0) / 9.0)))

def legenda(t):
    if t < 4.0: return "vista geral"
    if t < 14.0: return "trajetória da câmera: 2326 poses"
    if t < 27.5: return "vista geral"
    if t < 33.3: return "interior da cozinha"
    if t < 35.9: return "interior da cozinha: geladeira"
    if t < 38.5: return "interior da cozinha: bancada e pia"
    if t < 41.6: return "interior da cozinha: fogão"
    if t < 45.0: return "vista geral"
    return None

def titulo_alfa(t):
    ini = suave((t - 0.3) / 0.8) * (1 - suave((t - 3.2) / 0.8))
    fim = suave((t - 46.3) / 1.0)
    return max(ini, fim)

# ---------------------------------------------------------------- render
r = rd.OffscreenRenderer(W * SS, H * SS)
cena = r.scene
cena.show_skybox(False); cena.set_background([*FUNDO, 1.0])
cena.set_lighting(cena.LightingProfile.NO_SHADOWS, [0.3, -0.5, -0.8])
cena.scene.set_sun_light([0.35, 0.45, -0.82], [1.0, 0.98, 0.94], 60000)
cena.scene.set_indirect_light_intensity(18000)
mm = rd.MaterialRecord(); mm.shader = "defaultLit"
cena.add_geometry("malha", malha(), mm)
ml = rd.MaterialRecord(); ml.shader = "unlitLine"; ml.line_width = 2.2 * SS
mf = rd.MaterialRecord(); mf.shader = "unlitLine"; mf.line_width = 1.6 * SS
mc = rd.MaterialRecord(); mc.shader = "unlitLine"; mc.line_width = 3.0 * SS

def lineset(pts, lin, cores):
    ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(pts), o3d.utility.Vector2iVector(lin))
    ls.colors = o3d.utility.Vector3dVector(cores); return ls

def atualiza_linhas(t, olho):
    for nome in ("traj", "frust", "atual"):
        if cena.has_geometry(nome): cena.remove_geometry(nome)
    n = n_traj(t)
    if n < 2: return
    dentro = olho[2] < 265                       # no passeio: so frustums (a linha da trajetoria poluiria a vista)
    perto = 30.0 if dentro else 0.0              # e esconde o que esta a menos de 30 cm do olho
    p = TRAJ[:n]
    seg = np.c_[np.arange(n - 1), np.arange(1, n)]
    if perto:
        seg = seg[np.linalg.norm(p[seg[:, 0]] - olho, axis=1) > perto]
    if len(seg) and not dentro:
        cena.add_geometry("traj", lineset(p, seg, cor_tempo(seg[:, 0] / (NCAM - 1))), ml)
    pts, lin, cor = [], [], []
    for i in AMOSTRA[AMOSTRA < n]:
        f = FR[i]
        if perto and np.linalg.norm(f[0] - olho) < perto + FRUST_D: continue
        b = len(pts) * 5; pts.append(f); lin += [[b + a, b + c] for a, c in ARESTAS]
        cor += [cor_tempo(i / (NCAM - 1)) * 0.85] * len(ARESTAS)
    if pts:
        cena.add_geometry("frust", lineset(np.vstack(pts), lin, np.array(cor)), mf)
    if n < NCAM:                                  # camera "atual" em destaque enquanto a trajetoria e desenhada
        f = frustum(n - 1, FRUST_D * 2.2)
        cena.add_geometry("atual", lineset(f, ARESTAS, np.tile([0.85, 0.12, 0.12], (8, 1))), mc)

F_REG, F_SEMI = ImageFont.truetype(FONTE % "Medium", 30), ImageFont.truetype(FONTE % "SemiBold", 64)
F_SUB = ImageFont.truetype(FONTE % "Regular", 30)

def sobrepoe(img, t):
    base = img.convert("RGBA"); ov = Image.new("RGBA", base.size, (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
    # legenda com transicao suave
    txt = legenda(t); a = 1.0
    for tb in (4.0, 14.0, 27.5, 33.3, 35.9, 38.5, 41.6, 45.0):
        a = min(a, suave(abs(t - tb) / 0.35))
    if t > 45.0: txt = None
    if txt:
        x0, y0 = 48, H - 48
        bb = d.textbbox((0, 0), txt, font=F_REG); tw, th = bb[2] - bb[0], bb[3] - bb[1]
        d.rounded_rectangle([x0 - 18, y0 - th - 22, x0 + tw + 18, y0 + 4], 12, fill=(255, 255, 255, int(200 * a)))
        d.text((x0, y0 - th - 9 - bb[1]), txt, font=F_REG, fill=(40, 44, 52, int(255 * a)))
    ta = titulo_alfa(t)
    if ta > 0.01:
        linhas = [(TITULO, F_SEMI, (30, 34, 42))]
        if t > 40: linhas.append(("malha limpa e trajetória de 2326 poses de câmera", F_SUB, (70, 74, 82)))
        y = 70
        for s, fnt, c in linhas:
            bb = d.textbbox((0, 0), s, font=fnt); tw = bb[2] - bb[0]
            d.text(((W - tw) // 2, y - bb[1]), s, font=fnt, fill=(*c, int(255 * ta)))
            y += bb[3] - bb[1] + 22
    return Image.alpha_composite(base, ov).convert("RGB")

def quadro(k):
    t = TT[k]; olho, alvo = OLHO[k], ALVO[k]
    atualiza_linhas(t, olho)
    r.setup_camera(float(FOV[k]), alvo, olho, [0, 0, 1], 4.0, 6000.0)
    for tent in range(12):   # o Filament as vezes devolve o quadro preto ou com faixa preta: renderiza de novo
        im = np.asarray(r.render_to_image())
        if im.mean() > 30 and (im.mean(axis=(1, 2)) < 5).sum() == 0 and (im.mean(axis=(0, 2)) < 5).sum() == 0: break
        print(f"quadro {k}: render incompleto, repetindo", flush=True)
    else:
        raise RuntimeError(f"quadro {k} preto")
    img = Image.fromarray(im[..., :3]).resize((W, H), Image.LANCZOS)
    return sobrepoe(img, t)

# aquecimento: o primeiro render sai preto
r.setup_camera(34, INI, esf(180, 84, 900, INI), [0, 0, 1], 4.0, 6000.0)
for _ in range(3): r.render_to_image()

if A.previa:
    pasta = (Path(A.cache) if A.cache else SAIDA.parent) / "previa"; pasta.mkdir(parents=True, exist_ok=True)
    for s in A.previa.split(","):
        k = min(int(float(s) * FPS), N - 1); quadro(k).save(pasta / f"t{float(s):05.1f}.png"); print(pasta / f"t{float(s):05.1f}.png")
    sys.exit(0)

if SAIDA.exists():
    sys.exit(f"{SAIDA} ja existe; nao sobrescrevo")
SAIDA.parent.mkdir(parents=True, exist_ok=True)
tmp = SAIDA.with_name(SAIDA.stem + ".parcial.mp4")
cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
       "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)]
ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
t0 = time.time()
for k in range(N):
    ff.stdin.write(np.asarray(quadro(k)).tobytes())
    if k % 60 == 0: print(f"quadro {k}/{N}  {time.time() - t0:.0f} s", flush=True)
ff.stdin.close(); assert ff.wait() == 0
tmp.rename(SAIDA)
print("ok", SAIDA, f"{SAIDA.stat().st_size / 1e6:.1f} MB, {time.time() - t0:.0f} s", flush=True)

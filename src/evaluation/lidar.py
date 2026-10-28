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

# PROMPT COMPLEMENT
# Stage: lidar (evaluation, after mesh): every reconstruction against the real LiDAR map (ground truth).
# Goal: load the LiDAR in cm, align each reconstruction to it (similarity: scale + rotation + translation) and
#       measure accuracy / completeness / F-score (Tanks and Temples) and chamfer; a Poisson mesh of the LiDAR is
#       compared the same way; room width / length / height from the LiDAR planes vs the tape measurements.
# In:  lidar.path (.ply/.pcd/.xyz/.pts point cloud or mesh, .obj/.glb/.gltf/.stl mesh, .las/.laz via laspy,
#      .e57 via pye57 when importable); <run_dir>/dense/<m>/points.npy (model units), clean/<m>/points.npy,
#      scale/points3d_cm.npy, mesh/poisson/<variant>/mesh.ply, mesh/{vertices,faces}.npy (working units);
#      scale/scale.npy, artifacts room_fill.csv (cm_per_model_unit), planes/T_floor.npy, sfm/{cam_R,centers}.npy;
#      <ws>/medidas/medidas.yaml (contorno_cm, pe_direito_cm).
# Out: <run_dir>/lidar/{T_<method>.npy (4x4 similarity native units -> LiDAR cm), lidar_mesh.ply};
#      artifacts lidar_eval.csv (one row per method, key method), lidar_dims.csv (key dimension).
# YAML: lidar.{path: null, unit: m|cm|mm, voxel_cm: 1.0, taus_cm: [2, 5, 10], icp_voxels_cm: [8, 4, 2],
#      methods: auto | [names], up: auto|x|y|z|-x|-y|-z, crop_margin_cm: 10, mesh: true, poisson_depth: 10,
#      measurements: null (-> <ws>/medidas/medidas.yaml), min_fitness: 0.5}.
# Rules:
# - path null -> INFO "no LiDAR", return None; unreadable file / missing optional package -> WARNING, None.
# - Methods (auto): dense_<m> for every dense/<m>/points.npy, clean_<m> for every clean/<m>, sparse, poisson_<v>,
#   mesh_final, plus lidar_poisson (LiDAR mesh, identity transform). Meshes are sampled uniformly (~2 pts/voxel).
# - Initial scale (cm per model unit): scale.npy, else room_fill cm_per_model_unit, else ratio of the RMS radii.
# - Gravity: recon up = floor normal of planes/T_floor (fallback camera up); LiDAR up = lidar.up (auto = none).
#   Each cloud gets a Manhattan frame (up + dominant wall direction, or 3 dominant normal directions); the
#   candidates are the cube rotations between both frames (4 yaws 0/90/180/270 when both ups are known, else 24)
#   plus FPFH RANSAC; each is refined by a coarse scaled ICP (Umeyama) and the highest fitness wins. The global
#   search runs once on the reference (first dense method); the other methods start from it.
# - Refinement: scaled point-to-point ICP (Umeyama, trimmed) over icp_voxels_cm, then point-to-plane
#   open3d.t multi_scale_icp on CUDA (CPU + WARNING without CUDA) with the scale fixed.
# - Metrics in cm after voxelising both clouds at voxel_cm: accuracy = recon -> full LiDAR distances
#   (mean/median/RMSE/p90); the LiDAR is cropped to the recon box (room frame, + crop_margin_cm) for
#   completeness = LiDAR -> recon; precision/recall/F at each tau; chamfer = 0.5 (acc_mean + comp_mean) (same
#   definition as src/dense/compare.chamfer).
# - Dims: LiDAR in the room frame, planes peeled by RANSAC; height = highest - lowest horizontal plane, width /
#   length = smaller / larger distance between the outermost walls along the two Manhattan axes, vs the bbox of
#   contorno_cm and pe_direito_cm (empty when not measured).

# >>> BEGIN IMPLEMENTATION >>>
"""Evaluation against the LiDAR ground truth: loading, similarity alignment, T&T metrics and room dimensions."""
import itertools
import logging
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.core as o3c
import yaml
from scipy.spatial import cKDTree

from src import stage_io

log = logging.getLogger(__name__)

DEFAULTS = {"path": None, "unit": "m", "voxel_cm": 1.0, "taus_cm": [2, 5, 10], "icp_voxels_cm": [8, 4, 2],
            "methods": "auto", "up": "auto", "crop_margin_cm": 10.0, "mesh": True, "poisson_depth": 10,
            "measurements": None, "min_fitness": 0.5}
UNIT_CM = {"m": 100.0, "cm": 1.0, "mm": 0.1}
POINT_EXT = {".ply", ".pcd", ".xyz", ".xyzn", ".xyzrgb", ".pts"}
MESH_EXT = {".obj", ".glb", ".gltf", ".stl", ".off"}
MAX_SAMPLES = 4_000_000

def lidar_cfg(cfg: dict) -> dict:

    return {**DEFAULTS, **(cfg.get("lidar") or {})}

def device() -> o3c.Device:

    if o3c.cuda.is_available():
        return o3c.Device("CUDA:0")
    log.warning("lidar: open3d without CUDA, ICP and nearest neighbours on the CPU")

    return o3c.Device("CPU:0")

# ---------------------------------------------------------------- loading

def sample_mesh(mesh: o3d.geometry.TriangleMesh, spacing: float) -> np.ndarray:
    """Uniform samples, ~2 per spacing^2 of surface (capped at MAX_SAMPLES)."""

    n = int(np.clip(2.0 * mesh.get_surface_area() / spacing ** 2, 1000, MAX_SAMPLES))

    return np.asarray(mesh.sample_points_uniformly(n).points)

def read_las(path: Path) -> np.ndarray | None:

    try:
        import laspy
    except ImportError:
        log.warning("lidar: laspy missing (uv pip install 'laspy[lazrs]'), cannot read %s", path)
        return None

    return np.asarray(laspy.read(str(path)).xyz, dtype=np.float64)

def read_e57(path: Path) -> np.ndarray | None:

    try:
        import pye57
    except ImportError:
        log.warning("lidar: pye57 not importable, cannot read %s (convert it to .ply/.las)", path)
        return None

    e57 = pye57.E57(str(path))
    scans = []
    for i in range(e57.scan_count):
        d = e57.read_scan(i, ignore_missing_fields=True, transform=True)
        scans.append(np.stack([d["cartesianX"], d["cartesianY"], d["cartesianZ"]], 1))

    return np.vstack(scans).astype(np.float64)

def load_lidar_points(path, unit: str, voxel_cm: float) -> np.ndarray | None:
    """Points in cm (meshes sampled at voxel_cm); None (WARNING) when unreadable."""

    path = Path(path)
    if not path.exists():
        log.warning("lidar: %s not found", path)
        return None

    to_cm = UNIT_CM[unit]
    ext = path.suffix.lower()
    pts = None
    if ext in {".las", ".laz"}:
        pts = read_las(path)
    elif ext == ".e57":
        pts = read_e57(path)
    elif ext in MESH_EXT or ext == ".ply":
        mesh = o3d.io.read_triangle_mesh(str(path))
        if len(mesh.triangles):
            pts = sample_mesh(mesh, voxel_cm / to_cm)
        elif ext == ".ply":
            pts = np.asarray(o3d.io.read_point_cloud(str(path)).points)
    elif ext in POINT_EXT:
        pts = np.asarray(o3d.io.read_point_cloud(str(path)).points)
    else:
        log.warning("lidar: unsupported format %s", path)
        return None

    if pts is None or len(pts) == 0:
        log.warning("lidar: no points read from %s", path)
        return None

    return np.asarray(pts, dtype=np.float64) * to_cm

def voxel(points: np.ndarray, size: float) -> np.ndarray:

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(points, np.float64)))

    return np.asarray(pcd.voxel_down_sample(size).points)

def clean_cloud(points: np.ndarray, size: float, nb: int = 20, std: float = 2.5) -> np.ndarray:
    """Voxel grid + statistical outlier removal."""

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)).voxel_down_sample(size)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb, std_ratio=std)

    return np.asarray(pcd.points)

# ---------------------------------------------------------------- reconstruction sources

def source_list(run_dir: Path, methods) -> dict[str, tuple[Path, str, bool]]:
    """name -> (path, kind cloud|mesh|mesh_npy, in_model_units)."""

    found = {}
    for d in sorted((run_dir / "dense").glob("*/points.npy")):
        found[f"dense_{d.parent.name}"] = (d, "cloud", True)
    for d in sorted((run_dir / "clean").glob("*/points.npy")):
        found[f"clean_{d.parent.name}"] = (d, "cloud", False)
    if (run_dir / "scale" / "points3d_cm.npy").exists():
        found["sparse"] = (run_dir / "scale" / "points3d_cm.npy", "cloud", False)
    for d in sorted((run_dir / "mesh" / "poisson").glob("*/mesh.ply")):
        found[f"poisson_{d.parent.name}"] = (d, "mesh", False)
    if (run_dir / "mesh" / "vertices.npy").exists() and (run_dir / "mesh" / "faces.npy").exists():
        found["mesh_final"] = (run_dir / "mesh" / "vertices.npy", "mesh_npy", False)

    if methods == "auto" or methods is None:
        return found
    missing = [m for m in methods if m not in found and m != "lidar_poisson"]
    if missing:
        log.warning("lidar: methods without output skipped: %s (found %s)", missing, list(found))

    return {m: found[m] for m in methods if m in found}

def load_source(path: Path, kind: str, spacing: float) -> np.ndarray:
    """Native-unit points; meshes sampled with ~spacing (native units)."""

    if kind == "cloud":
        return np.load(path).astype(np.float64).reshape(-1, 3)
    if kind == "mesh":
        mesh = o3d.io.read_triangle_mesh(str(path))
    else:
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.load(path).astype(np.float64)),
                                         o3d.utility.Vector3iVector(np.load(path.parent / "faces.npy").astype(np.int32)))

    return sample_mesh(mesh, spacing)

def model_scale(run_dir: Path, artifacts_dir: Path) -> tuple[float, str, str]:
    """(cm per model unit, origin, working unit cm|model); nan when unknown."""

    s, unit = stage_io.scale_info(run_dir)
    if unit == "cm":
        return s, "scale.npy", unit

    rows = stage_io.read_csv(artifacts_dir / "room_fill.csv")
    try:
        s = float(rows[-1]["cm_per_model_unit"]) if rows else float("nan")
    except (KeyError, ValueError):
        s = float("nan")

    return (s, "room_fill", unit) if np.isfinite(s) and s > 0 else (float("nan"), "extent", unit)

def recon_up(run_dir: Path) -> np.ndarray | None:
    """Floor normal (up) in the reconstruction frame: planes/T_floor, else the mean camera up."""

    path = run_dir / "planes" / "T_floor.npy"
    if path.exists():
        T = np.load(path)
        if not np.allclose(T[:3, :3], np.eye(3)):
            return T[2, :3] / np.linalg.norm(T[2, :3])

    path = run_dir / "sfm" / "cam_R.npy"
    if path.exists():
        up = -np.load(path)[:, 1, :].mean(0)
        return up / np.linalg.norm(up)

    return None

# ---------------------------------------------------------------- geometry helpers

def apply(T: np.ndarray, p: np.ndarray) -> np.ndarray:

    return p @ T[:3, :3].T + T[:3, 3]

def similarity(s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:

    T = np.eye(4)
    T[:3, :3], T[:3, 3] = s * R, t

    return T

def decompose(T: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:

    s = float(np.cbrt(np.linalg.det(T[:3, :3])))

    return s, T[:3, :3] / s, T[:3, 3]

def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> np.ndarray:
    """Least-squares similarity dst ~ s R src + t (Umeyama 1991)."""

    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd(b.T @ a / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / (a ** 2).sum(1).mean()) if with_scale else 1.0

    return similarity(s, R, mu_d - s * R @ mu_s)

def rot_to(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation taking unit a to unit b."""

    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-12:
        if c > 0:
            return np.eye(3)
        axis = np.cross(a, [1.0, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1.0, 0])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    return np.eye(3) + vx + vx @ vx / (1 + c)

def rot_z(a: float) -> np.ndarray:

    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])

def normals(points: np.ndarray, radius: float) -> np.ndarray:

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))

    return np.asarray(pcd.normals)

def dominant(n: np.ndarray, cos_tol: float, rng: np.random.Generator) -> np.ndarray:
    """Axis (sign-free) supported by the most normals, refined by the mean of its inliers."""

    cand = n[rng.choice(len(n), min(400, len(n)), replace=False)]
    best = cand[np.argmax((np.abs(n @ cand.T) > cos_tol).sum(0))]
    inl = n[np.abs(n @ best) > cos_tol]
    inl = inl * np.sign(inl @ best)[:, None]
    m = inl.mean(0)

    return m / np.linalg.norm(m)

def manhattan_frame(points: np.ndarray, spacing: float, up: np.ndarray | None,
                    rng: np.random.Generator) -> np.ndarray:
    """R (rows = frame axes in world) with z = up (when known) and x along the dominant wall direction."""

    pts = points[rng.choice(len(points), min(len(points), 60000), replace=False)]
    n = normals(pts, 3 * spacing)
    if up is None:
        up = dominant(n, np.cos(np.radians(10)), rng)
        n_perp = n[np.abs(n @ up) < np.sin(np.radians(10))]
        x = dominant(n_perp, np.cos(np.radians(10)), rng) if len(n_perp) > 10 else np.cross(up, [1.0, 0, 0])
        x = x - (x @ up) * up
        x /= np.linalg.norm(x)
        return np.stack([x, np.cross(up, x), up])

    R = rot_to(up / np.linalg.norm(up), np.array([0, 0, 1.0]))
    h = (n @ R.T)
    h = h[np.abs(h[:, 2]) < 0.3]
    theta = np.arctan2(h[:, 1], h[:, 0])
    w = np.hypot(h[:, 0], h[:, 1])
    yaw = np.angle((w * np.exp(4j * theta)).sum()) / 4 if len(h) else 0.0

    return rot_z(-yaw) @ R

def cube_rotations(keep_z: bool) -> list[np.ndarray]:
    """Proper signed permutation matrices (24); keep_z -> the 4 yaws about z."""

    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            C = np.zeros((3, 3))
            C[np.arange(3), perm] = signs
            if np.linalg.det(C) > 0 and (not keep_z or C[2, 2] == 1):
                out.append(C)

    return out

# ---------------------------------------------------------------- nearest neighbours

def nn_dist(query: np.ndarray, ref: np.ndarray, dev: o3c.Device | None = None) -> np.ndarray:
    """Distance from each query point to its nearest ref point (CUDA knn, scipy cKDTree fallback)."""

    if len(query) == 0 or len(ref) == 0:
        return np.full(len(query), np.inf)
    if dev is not None and dev.get_type() == o3c.Device.DeviceType.CUDA:
        try:
            nns = o3c.nns.NearestNeighborSearch(o3c.Tensor(ref.astype(np.float32), device=dev))
            nns.knn_index()
            out = []
            for i in range(0, len(query), 500_000):
                q = o3c.Tensor(query[i:i + 500_000].astype(np.float32), device=dev)
                _, d2 = nns.knn_search(q, 1)
                out.append(np.sqrt(d2.cpu().numpy().ravel().astype(np.float64)))
            return np.concatenate(out)
        except Exception as exc:  # noqa: BLE001 - fall back to the CPU tree
            log.warning("lidar: CUDA knn failed (%s), using cKDTree", exc)

    return cKDTree(ref).query(query, workers=-1)[0]

# ---------------------------------------------------------------- alignment

def scaled_icp(src: np.ndarray, T: np.ndarray, target: np.ndarray, voxels: list[float],
               iters: int = 40, trim: float = 0.9) -> tuple[np.ndarray, float]:
    """Trimmed point-to-point ICP estimating a similarity (Umeyama) per iteration, coarse to fine.
    src native units, target cm. Returns (T, fitness = frac of src within 2 voxel at the last level)."""

    fitness = 0.0
    for k, v in enumerate(voxels):
        s = decompose(T)[0]
        sd = voxel(src, v / s)
        td = voxel(target, v)
        tree = cKDTree(td)
        max_d = (6.0 if k == 0 else 3.0) * v
        for _ in range(iters):
            p = apply(T, sd)
            d, idx = tree.query(p, distance_upper_bound=max_d, workers=-1)
            ok = np.isfinite(d)
            if ok.sum() < 20:
                break
            keep = np.where(ok)[0]
            keep = keep[d[keep] <= np.quantile(d[keep], trim)]
            T_new = umeyama(sd[keep], td[idx[keep]])
            moved = np.abs(apply(T_new, sd[keep]) - p[keep]).max(1).mean()
            T = T_new
            if moved < 0.01 * v:
                break
        d = tree.query(apply(T, sd), workers=-1)[0]
        fitness = float((d < 2 * v).mean())

    return T, fitness

def fpfh_candidate(src_cm: np.ndarray, tgt_cm: np.ndarray, v: float) -> np.ndarray | None:
    """Rigid FPFH + RANSAC (legacy pipeline) between two clouds already in cm."""

    reg = o3d.pipelines.registration
    feats = []
    for pts in (src_cm, tgt_cm):
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(voxel(pts, v)))
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2 * v, max_nn=30))
        feats.append((pcd, reg.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=5 * v, max_nn=100))))
    (sp, sf), (tp, tf) = feats
    try:
        res = reg.registration_ransac_based_on_feature_matching(
            sp, tp, sf, tf, True, 1.5 * v, reg.TransformationEstimationPointToPoint(False), 3,
            [reg.CorrespondenceCheckerBasedOnEdgeLength(0.9), reg.CorrespondenceCheckerBasedOnDistance(1.5 * v)],
            reg.RANSACConvergenceCriteria(200_000, 0.999))
    except RuntimeError as exc:
        log.warning("lidar: FPFH RANSAC failed (%s)", exc)
        return None

    return np.asarray(res.transformation)

def global_align(src: np.ndarray, s0: float, up_r: np.ndarray | None, lidar: np.ndarray,
                 frame_l: np.ndarray, up_l_known: bool, voxels: list[float],
                 rng: np.random.Generator) -> tuple[np.ndarray, float, str]:
    """Best similarity (native -> LiDAR cm) over cube-rotation candidates between the Manhattan frames and
    FPFH RANSAC, each refined by the coarse scaled ICP. Returns (T, fitness, candidate name)."""

    v = voxels[0]
    src_cm = voxel(src * s0, v)
    frame_r = manhattan_frame(src_cm, v, up_r, rng)
    keep_z = up_r is not None and up_l_known
    l_f = voxel(lidar, v) @ frame_l.T
    mid_l = 0.5 * (np.percentile(l_f, 2, 0) + np.percentile(l_f, 98, 0))

    cands = []
    for k, C in enumerate(cube_rotations(keep_z)):
        R = frame_l.T @ C @ frame_r
        q = (src_cm @ R.T) @ frame_l.T
        t_f = mid_l - 0.5 * (np.percentile(q, 2, 0) + np.percentile(q, 98, 0))
        cands.append((f"cube{k}", similarity(s0, R, frame_l.T @ t_f)))
    T_f = fpfh_candidate(src_cm, lidar, v)
    if T_f is not None:
        cands.append(("fpfh", T_f @ similarity(s0, np.eye(3), np.zeros(3))))

    best = (None, -1.0, "")
    for name, T in cands:
        T_c, fit = scaled_icp(src, T, lidar, [2 * v, v], iters=25)
        s_c = decompose(T_c)[0]
        if not 0.5 < s_c / s0 < 2.0:
            continue
        log.debug("lidar candidate %s fitness %.3f scale %.4g", name, fit, s_c)
        if fit > best[1]:
            best = (T_c, fit, name)
    log.info("lidar: global alignment %s, fitness %.3f over %d candidates", best[2], best[1], len(cands))

    return best

def p2plane_icp(src: np.ndarray, T: np.ndarray, lidar: np.ndarray, voxels: list[float],
                dev: o3c.Device) -> tuple[np.ndarray, float, float]:
    """Point-to-plane open3d.t multi_scale_icp with the scale of T fixed. Returns (T, fitness, rmse cm)."""

    reg = o3d.t.pipelines.registration
    s, R, t = decompose(T)
    sp = o3d.t.geometry.PointCloud(o3c.Tensor(voxel(src * s, voxels[-1]).astype(np.float32), device=dev))
    tp = o3d.t.geometry.PointCloud(o3c.Tensor(voxel(lidar, voxels[-1]).astype(np.float32), device=dev))
    tp.estimate_normals(max_nn=30, radius=3.0 * voxels[-1])
    init = np.eye(4)
    init[:3, :3], init[:3, 3] = R, t
    crit = [reg.ICPConvergenceCriteria(1e-6, 1e-6, 50) for _ in voxels]
    res = reg.multi_scale_icp(sp, tp, o3d.utility.DoubleVector([float(v) for v in voxels]), crit,
                              o3d.utility.DoubleVector([2.0 * v for v in voxels]),
                              o3c.Tensor(init, dtype=o3c.float64),
                              reg.TransformationEstimationPointToPlane(reg.robust_kernel.RobustKernel(
                                  reg.robust_kernel.RobustKernelMethod.TukeyLoss, 2.0 * voxels[-1])))
    Tr = res.transformation.cpu().numpy()

    return Tr @ similarity(s, np.eye(3), np.zeros(3)), float(res.fitness), float(res.inlier_rmse)

# ---------------------------------------------------------------- metrics

def crop_box(points: np.ndarray, frame: np.ndarray, margin: float) -> tuple[np.ndarray, np.ndarray]:

    q = points @ frame.T

    return np.percentile(q, 0.5, 0) - margin, np.percentile(q, 99.5, 0) + margin

def in_box(points: np.ndarray, frame: np.ndarray, box) -> np.ndarray:

    q = points @ frame.T

    return np.all((q >= box[0]) & (q <= box[1]), axis=1)

def tnt_metrics(recon: np.ndarray, lidar: np.ndarray, lidar_crop: np.ndarray, taus: list[float],
                dev: o3c.Device) -> dict:
    """Accuracy (recon -> LiDAR), completeness (cropped LiDAR -> recon), P/R/F per tau, chamfer (cm)."""

    d_acc = nn_dist(recon, lidar, dev) if len(recon) else np.full(1, np.inf)
    d_comp = nn_dist(lidar_crop, recon, dev) if len(lidar_crop) else np.full(1, np.inf)
    row = {"n_recon_vox": len(recon), "n_lidar_crop_vox": len(lidar_crop),
           "acc_mean_cm": d_acc.mean(), "acc_median_cm": np.median(d_acc), "acc_rmse_cm": np.sqrt((d_acc ** 2).mean()),
           "acc_p90_cm": np.percentile(d_acc, 90), "comp_mean_cm": d_comp.mean(), "comp_median_cm": np.median(d_comp),
           "chamfer_cm": 0.5 * (d_acc.mean() + d_comp.mean())}
    for tau in taus:
        p, r = float((d_acc < tau).mean()), float((d_comp < tau).mean())
        tag = f"{tau:g}cm"
        row.update({f"precision_{tag}": p, f"recall_{tag}": r, f"fscore_{tag}": 2 * p * r / (p + r) if p + r else 0.0})

    return {k: round(float(v), 4) if isinstance(v, (float, np.floating)) else v for k, v in row.items()}

# ---------------------------------------------------------------- dims

def lidar_dims(lidar: np.ndarray, frame: np.ndarray, rng_seed: int = 0) -> dict:
    """Width / length / height (cm) from RANSAC planes of the LiDAR in the room frame (z up)."""

    q = voxel(lidar @ frame.T, 4.0)
    o3d.utility.random.seed(rng_seed)
    remaining, planes = np.arange(len(q)), []
    for _ in range(14):
        if len(remaining) < 0.01 * len(q):
            break
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(q[remaining]))
        model, inl = pcd.segment_plane(2.0, 3, 2000)
        if len(inl) < 0.02 * len(q):
            break
        model = np.asarray(model) / np.linalg.norm(model[:3])
        planes.append((model, len(inl)))
        remaining = np.delete(remaining, inl)

    def spread(axis: int) -> float:
        offs = [-p[3] / p[axis] for p, _ in planes if abs(p[axis]) > np.cos(np.radians(10))]
        return float(max(offs) - min(offs)) if len(offs) >= 2 else float("nan")

    wx, wy, h = spread(0), spread(1), spread(2)
    log.info("lidar dims: %d planes, x %.1f y %.1f z %.1f cm", len(planes), wx, wy, h)

    return {"width": float(np.nanmin([wx, wy])) if np.isfinite([wx, wy]).any() else float("nan"),
            "length": float(np.nanmax([wx, wy])) if np.isfinite([wx, wy]).any() else float("nan"), "height": h}

def measured_dims(path: Path) -> dict:

    if not path.exists():
        log.warning("lidar: measurements %s missing", path)
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    out = {}
    if data.get("contorno_cm"):
        ext = np.ptp(np.asarray(data["contorno_cm"], float), axis=0)
        out.update(width=float(ext.min()), length=float(ext.max()))
    if data.get("pe_direito_cm") is not None:
        out["height"] = float(data["pe_direito_cm"])

    return out

# ---------------------------------------------------------------- stage

def run(cfg: dict, run_dir: str, artifacts_dir: str) -> Path | None:

    lc = lidar_cfg(cfg)
    if not lc["path"]:
        log.info("lidar: no LiDAR (lidar.path null), evaluation skipped")
        return None
    if lc["unit"] not in UNIT_CM:
        log.warning("lidar: unit %s not in %s", lc["unit"], list(UNIT_CM))
        return None

    t_all = time.perf_counter()
    run_dir, artifacts_dir = Path(run_dir), Path(artifacts_dir)
    out = run_dir / "lidar"
    out.mkdir(parents=True, exist_ok=True)
    dev = device()
    rng = np.random.default_rng(0)
    v_cm, voxels = float(lc["voxel_cm"]), sorted((float(v) for v in lc["icp_voxels_cm"]), reverse=True)
    taus = [float(t) for t in lc["taus_cm"]]

    raw = load_lidar_points(lc["path"], lc["unit"], v_cm)
    if raw is None:
        return None
    lidar = clean_cloud(raw, v_cm)
    log.info("lidar: %s -> %d points, %d after voxel %.1f cm + outliers", lc["path"], len(raw), len(lidar), v_cm)
    up_l = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}.get(str(lc["up"]).lstrip("-"))
    up_l = None if up_l is None else np.asarray(up_l, float) * (-1 if str(lc["up"]).startswith("-") else 1)
    frame_l = manhattan_frame(lidar, voxels[0], up_l, rng)

    sources = source_list(run_dir, lc["methods"])
    s_model, s_from, unit = model_scale(run_dir, artifacts_dir)
    if not sources:
        log.warning("lidar: no reconstruction to evaluate in %s", run_dir)
    up_r = recon_up(run_dir)

    T_ref, rows, boxes = None, [], []
    for name, (path, kind, model_units) in sources.items():
        t0 = time.perf_counter()
        to_model = 1.0 if (model_units or unit != "cm") else 1.0 / s_model  # native -> model units
        s0_model = s_model
        if not np.isfinite(s0_model):
            probe = load_source(path, kind, 1.0) if kind == "cloud" else np.asarray(o3d.io.read_triangle_mesh(str(path)).vertices) if kind == "mesh" else np.load(path)
            rms = lambda p: np.sqrt(((p - np.median(p, 0)) ** 2).sum(1).mean())
            s0_model = rms(voxel(lidar, voxels[0])) / rms(probe * to_model)
            s_from = "extent"
        s0 = s0_model * to_model
        pts_native = load_source(path, kind, v_cm / s0)
        pts_native = voxel(pts_native, 0.5 * v_cm / s0)
        if len(pts_native) < 100:
            log.warning("lidar: %s has %d points, skipped", name, len(pts_native))
            continue

        cand = "reference"
        if T_ref is not None:
            T, fit = scaled_icp(pts_native, T_ref @ similarity(to_model, np.eye(3), np.zeros(3)), lidar, voxels)
        if T_ref is None or fit < lc["min_fitness"]:
            T, fit, cand = global_align(pts_native, s0, up_r, lidar, frame_l, up_l is not None, voxels, rng)
            if T is None:
                log.warning("lidar: %s could not be aligned", name)
                continue
            T, fit = scaled_icp(pts_native, T, lidar, voxels)
            if T_ref is None:
                T_ref = T @ similarity(1.0 / to_model, np.eye(3), np.zeros(3))  # model units -> cm
        T, fit_pl, rmse = p2plane_icp(pts_native, T, lidar, voxels, dev)
        np.save(out / f"T_{name}.npy", T)

        recon = voxel(apply(T, pts_native), v_cm)
        box = crop_box(recon, frame_l, lc["crop_margin_cm"])
        boxes.append(box)
        crop = lidar[in_box(lidar, frame_l, box)]
        s_est = decompose(T)[0]
        row = {"method": name, "source": str(path.relative_to(run_dir)), "init_scale_from": s_from,
               "init_cm_per_unit": round(s0, 6), "cm_per_unit": round(s_est, 6), "align": cand,
               "fitness_scaled_icp": round(fit, 4), "fitness_p2plane": round(fit_pl, 4), "icp_rmse_cm": round(rmse, 4),
               **tnt_metrics(recon, lidar, crop, taus, dev), "t_s": round(time.perf_counter() - t0, 3)}
        stage_io.log_metrics(log, row)
        stage_io.upsert_csv(artifacts_dir / "lidar_eval.csv", row, key="method")
        rows.append(row)

    # up of the LiDAR for the room frame: lidar.up, else the reconstruction floor normal carried by T_ref
    if up_l is None and T_ref is not None and up_r is not None:
        up_l = decompose(T_ref)[1] @ up_r
        frame_room = manhattan_frame(lidar, voxels[0], up_l, rng)
    else:
        frame_room = frame_l

    if lc["mesh"]:
        t0 = time.perf_counter()
        from src.mesh import poisson
        pcfg = {"normal_knn": 20, "depth": int(lc["poisson_depth"]), "density_quantile": 0.02, "threads": 8}
        centers = None
        if T_ref is not None and (run_dir / "sfm" / "centers.npy").exists():
            centers = apply(T_ref, np.load(run_dir / "sfm" / "centers.npy").reshape(-1, 3))
        mesh = poisson.reconstruct(voxel(lidar, 2 * v_cm), None, centers, pcfg)
        if len(mesh.triangles):
            o3d.io.write_triangle_mesh(str(out / "lidar_mesh.ply"), mesh)
            pts = voxel(sample_mesh(mesh, v_cm), v_cm)
            if boxes:
                lo = np.min([b[0] for b in boxes], 0); hi = np.max([b[1] for b in boxes], 0)
                pts = pts[in_box(pts, frame_l, (lo, hi))]  # same frame as the per-method boxes
                crop = lidar[in_box(lidar, frame_l, (lo, hi))]
            else:
                crop = lidar
            np.save(out / "T_lidar_poisson.npy", np.eye(4))
            row = {"method": "lidar_poisson", "source": "lidar/lidar_mesh.ply", "init_scale_from": "",
                   "init_cm_per_unit": 1.0, "cm_per_unit": 1.0, "align": "identity",
                   **tnt_metrics(pts, lidar, crop, taus, dev), "t_s": round(time.perf_counter() - t0, 3)}
            stage_io.log_metrics(log, row)
            stage_io.upsert_csv(artifacts_dir / "lidar_eval.csv", row, key="method")
            rows.append(row)
        else:
            log.warning("lidar: Poisson mesh of the LiDAR has no faces")

    meas_path = Path(lc["measurements"]) if lc["measurements"] else stage_io.workspace(cfg) / "medidas" / "medidas.yaml"
    meas = measured_dims(meas_path) if cfg.get("extraction") or lc["measurements"] else {}
    for dim, value in lidar_dims(lidar, frame_room).items():
        m = meas.get(dim)
        err = abs(value - m) if m is not None and np.isfinite(value) else None
        drow = {"dimension": dim, "lidar_cm": round(value, 2) if np.isfinite(value) else None,
                "measured_cm": m, "abs_err_cm": None if err is None else round(err, 2),
                "rel_err": None if err is None else round(err / m, 4)}
        stage_io.log_metrics(log, drow)
        stage_io.upsert_csv(artifacts_dir / "lidar_dims.csv", drow, key="dimension")

    log.info("lidar: %d rows in %.1f s", len(rows), time.perf_counter() - t_all)

    return artifacts_dir / "lidar_eval.csv" if rows else None
# <<< END IMPLEMENTATION <<<

def box_room(rng: np.random.Generator, n: int = 60000) -> np.ndarray:
    """Points (cm) on the inner surfaces of a 270 x 430 x 280 room with an L-shaped counter and a box."""

    a = rng.uniform(size=(n, 2))
    k = rng.integers(0, 8, n)
    p = np.zeros((n, 3))
    W, L, H = 270.0, 430.0, 280.0
    for i, (ax, val, u, v, su, sv, ou, ov) in enumerate([
            (2, 0, 0, 1, W, L, 0, 0), (2, H, 0, 1, W, L, 0, 0), (0, 0, 1, 2, L, H, 0, 0), (0, W, 1, 2, L, H, 0, 0),
            (1, 0, 0, 2, W, H, 0, 0), (1, L, 0, 2, W, H, 0, 0), (2, 90, 0, 1, 60, 200, 210, 20),
            (0, 150, 1, 2, 80, 120, 300, 0)]):
        m = k == i
        p[m, ax] = val
        p[m, u] = ou + a[m, 0] * su
        p[m, v] = ov + a[m, 1] * sv

    return p

if __name__ == "__main__":
    import tempfile
    from scipy.spatial.transform import Rotation
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    rng = np.random.default_rng(1)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        room = box_room(rng)
        s_true = 27.35  # cm per model unit
        R_true = Rotation.from_euler("zyx", [37, -20, 64], degrees=True).as_matrix()
        t_true = np.array([120.0, -340.0, 55.0])
        recon = room[room[:, 1] < 330]  # partial coverage
        R_m = Rotation.random(random_state=3).as_matrix()  # model frame
        model = (recon @ R_m.T) / s_true
        lidar_cm = (room @ R_m.T) @ R_true.T + t_true + rng.normal(0, 0.3, room.shape)
        T_true = similarity(s_true, R_true, t_true)
        rd, ad = tmp / "run", tmp / "art"
        (rd / "dense" / "toy").mkdir(parents=True)
        np.save(rd / "dense" / "toy" / "points.npy", model)
        stage_io.write_ply(tmp / "lidar.ply", lidar_cm / 100.0)
        stage_io.upsert_csv(ad / "room_fill.csv", {"cm_per_model_unit": s_true * 1.03})  # 3 % off on purpose
        cfg = {"lidar": {"path": str(tmp / "lidar.ply"), "unit": "m", "mesh": True, "poisson_depth": 8,
                         "measurements": str(tmp / "m.yaml")}}
        (tmp / "m.yaml").write_text("contorno_cm: [[0, 0], [270, 0], [270, 430], [0, 430]]\npe_direito_cm: 280\n")
        assert run({"lidar": {"path": None}}, str(rd), str(ad)) is None
        assert run(cfg, str(rd), str(ad)) is not None
        T = np.load(rd / "lidar" / "T_dense_toy.npy")
        s, R, t = decompose(T)
        rot_err = np.degrees(np.arccos(np.clip((np.trace(R.T @ R_true) - 1) / 2, -1, 1)))
        t_err = np.linalg.norm(apply(T, model) - apply(T_true, model), axis=1).mean()
        print(f"scale {s:.4f} (true {s_true}) rot err {rot_err:.3f} deg mean point err {t_err:.3f} cm")
        assert abs(s / s_true - 1) < 0.005 and rot_err < 0.5 and t_err < 1.0
        rows = {r["method"]: r for r in stage_io.read_csv(ad / "lidar_eval.csv")}
        print(rows["dense_toy"])
        assert float(rows["dense_toy"]["fscore_5cm"]) > 0.6 and float(rows["dense_toy"]["acc_median_cm"]) < 1.5
        assert "lidar_poisson" in rows and (rd / "lidar" / "lidar_mesh.ply").exists()
        dims = {r["dimension"]: r for r in stage_io.read_csv(ad / "lidar_dims.csv")}
        print(dims)
        assert abs(float(dims["width"]["lidar_cm"]) - 270) < 3 and abs(float(dims["length"]["lidar_cm"]) - 430) < 3
        assert abs(float(dims["height"]["lidar_cm"]) - 280) < 3
    print("OK")

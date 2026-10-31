# PROMPT COMPLEMENT
# Tool: lidar_fuse (standalone, before the pipeline): one coherent LiDAR map out of the per-frame clouds of a
#   phone LiDAR capture (HEGE export: PLY/<0000000>.ply, one xyz+rgb cloud per frame, metres).
# Facts: the frames are already in one world frame (ARKit poses), so consecutive frames agree to a few mm,
#   but the tracking drifts: a place seen again minutes later lands centimetres off, and a plain merge of
#   every frame doubles walls and blurs edges.
# Goal: globally consistent alignment, then a clean cloud and a mesh with the holes closed:
#   1. fragments: runs of --fragment consecutive frames (drift inside one is ~mm) are merged on a voxel grid,
#      keeping voxels seen by >= --min-frames-fragment frames (drops LiDAR flying pixels and spikes);
#   2. pairs: every pair of fragments that overlap (at the ARKit poses) is registered with coarse-to-fine
#      robust point-to-plane ICP (Tukey loss) from the identity; pairs with low fitness / high RMSE dropped;
#   3. pose graph: consecutive pairs = odometry edges, the others = loop closures (line process prunes the
#      wrong ones); Levenberg-Marquardt spreads the drift over the whole trajectory;
#   4. merge: each fragment's voxels are moved by its optimised pose and re-voxelised, a voxel needs
#      >= --min-frames frames in total; statistical outlier removal;
#   5. mesh: Poisson (low-density trim) -> MeshLab repair + Close Holes -> Liepa (src/mesh/liepa.py) for the
#      holes MeshLab leaves.
# In:  <capture>/PLY/*.ply. Truncated/empty PLYs (interrupted upload) are skipped and counted.
# Out: <out>/{lidar_fused.ply (cloud, metres), fragment_poses.npy (F x 4 x 4 corrections), pairs.csv,
#      mesh_poisson.ply, mesh_closed.ply, lidar_fuse.csv}. lidar_fused.ply is what lidar.path points to.
#      Alignment quality: median |point-to-plane| distance between overlapping fragments before (ARKit
#      poses as delivered) and after the optimisation (misalign_before_mm / misalign_after_mm).
# Run: python -m src.cloud.lidar_fuse cozinha_lidar [--out cozinha_lidar/fused] [--fragment 100] [--voxel-cm 1]
# >>> BEGIN IMPLEMENTATION >>>
"""Globally consistent fusion of a phone LiDAR capture (fragments + robust ICP + pose graph) and hole-closed mesh."""
import argparse, csv, logging, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import open3d as o3d

from src.mesh import liepa

log = logging.getLogger(__name__)
REG = o3d.pipelines.registration
DT = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgba", "u1", 4)])
FRAME_BYTES = 8734958
OFF = 1 << 20  # 21 bits per axis


def read(path: Path) -> np.ndarray | None:
    """xyz (float64) + rgb (0-1) of one frame as an (N, 6) array; None when truncated."""

    if path.stat().st_size != FRAME_BYTES:
        return None
    raw = path.read_bytes()
    d = np.frombuffer(raw, DT, offset=raw.index(b"end_header\n") + 11)
    p = np.stack([d["x"], d["y"], d["z"]], 1).astype(np.float64)
    ok = np.isfinite(p).all(1) & (np.abs(p).sum(1) > 0)
    return np.hstack([p[ok], d["rgba"][ok, :3] / 255.0])


def keys_of(p: np.ndarray, voxel: float) -> np.ndarray:
    q = np.floor(p / voxel).astype(np.int64) + OFF
    return (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]


def accumulate(groups: list[tuple[np.ndarray, int]], voxel: float) -> tuple[np.ndarray, np.ndarray]:
    """Voxel means of [x y z r g b] and the number of frames that saw each voxel.

    groups: (points (N, 6), frame weight) pairs; a frame counts once per voxel."""

    ks, ss = [], []
    for pts, w in groups:
        k = keys_of(pts[:, :3], voxel)
        u, inv = np.unique(k, return_inverse=True)
        s = np.zeros((len(u), 8))
        s[:, 0] = np.bincount(inv, minlength=len(u))  # points
        for c in range(6):
            s[:, 1 + c] = np.bincount(inv, pts[:, c], len(u))
        s[:, 7] = w  # frames
        ks.append(u)
        ss.append(s)
    k = np.concatenate(ks)
    s = np.concatenate(ss)
    u, inv = np.unique(k, return_inverse=True)
    tot = np.stack([np.bincount(inv, s[:, c], len(u)) for c in range(8)], 1)
    return tot[:, 1:7] / tot[:, :1], tot[:, 7]


def to_pcd(pts: np.ndarray) -> o3d.geometry.PointCloud:
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts[:, :3]))
    pc.colors = o3d.utility.Vector3dVector(np.clip(pts[:, 3:6], 0, 1))
    return pc


def build_fragment(files: list[Path], args) -> tuple[np.ndarray | None, int, int]:
    """Merge a run of frames; returns (voxel means (M, 6) or None, frame weights, frames skipped)."""

    frames = [read(f) for f in files[::args.frame_step]]
    good = [f for f in frames if f is not None]
    if len(good) < 2:
        return None, 0, len(frames) - len(good)
    means, nf = accumulate([(f, 1) for f in good], args.voxel_cm / 100)
    keep = nf >= min(args.min_frames_fragment, len(good))
    out = np.hstack([means[keep], nf[keep, None] * args.frame_step])
    return out, len(good), len(frames) - len(good)


def icp(src, dst, voxels: list[float], init=np.eye(4)) -> REG.RegistrationResult:
    T, res = init, None
    for v in voxels:
        s = src.voxel_down_sample(v)
        d = dst.voxel_down_sample(v)
        for pc in (s, d):
            pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=v * 3, max_nn=30))
        est = REG.TransformationEstimationPointToPlane(REG.TukeyLoss(k=v * 1.5))
        res = REG.registration_icp(s, d, v * 3, T, est, REG.ICPConvergenceCriteria(max_iteration=50))
        T = res.transformation
    return res


def misalign_mm(src, dst, T: np.ndarray, voxel: float) -> float:
    """Median |point-to-plane| distance (mm) of the overlapping part of src (moved by T) to dst."""

    s = o3d.geometry.PointCloud(src).transform(T).voxel_down_sample(voxel)
    d = dst.voxel_down_sample(voxel)
    d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 3, max_nn=30))
    tree = o3d.geometry.KDTreeFlann(d)
    P, Q, N = np.asarray(s.points), np.asarray(d.points), np.asarray(d.normals)
    r = []
    for p in P[:: max(1, len(P) // 20000)]:
        k, idx, dist = tree.search_knn_vector_3d(p, 1)
        if dist[0] < (5 * voxel) ** 2:
            r.append(abs(np.dot(p - Q[idx[0]], N[idx[0]])))
    return float(np.median(r) * 1000) if r else float("nan")


def register(frags: list[o3d.geometry.PointCloud], args) -> tuple[list[np.ndarray], list[dict]]:
    """Pairwise robust ICP between overlapping fragments + pose graph optimisation."""

    v = args.voxel_cm / 100
    vox = [v * 5, v * 2.5, v]
    small = [f.voxel_down_sample(v * 5) for f in frags]
    boxes = [(np.asarray(f.points).min(0), np.asarray(f.points).max(0)) for f in small]
    cand = []
    for i in range(len(frags)):
        for j in range(i + 1, len(frags)):
            lo = np.maximum(boxes[i][0], boxes[j][0])
            hi = np.minimum(boxes[i][1], boxes[j][1])
            if (hi > lo).all():
                cand.append((i, j))
    log.info("%d fragments, %d candidate pairs (bounding boxes overlap)", len(frags), len(cand))

    def overlap(ij):
        i, j = ij
        d = np.asarray(small[j].compute_point_cloud_distance(small[i]))
        return (d < v * 10).mean()

    with ThreadPoolExecutor(args.threads) as ex:
        ov = list(ex.map(overlap, cand))
    cand = [c for c, o in zip(cand, ov) if o >= args.min_overlap or c[1] == c[0] + 1]
    log.info("%d pairs overlap >= %.0f%%", len(cand), args.min_overlap * 100)

    def run(ij):
        i, j = ij
        res = icp(frags[j], frags[i], vox)
        info = REG.get_information_matrix_from_point_clouds(frags[j], frags[i], v * 3, res.transformation)
        return i, j, res, info

    with ThreadPoolExecutor(args.threads) as ex:
        results = list(ex.map(run, cand))

    graph = REG.PoseGraph()
    for _ in frags:
        graph.nodes.append(REG.PoseGraphNode(np.eye(4)))
    rows = []
    for i, j, res, info in results:
        odo = j == i + 1
        ok = res.fitness >= args.min_fitness and res.inlier_rmse <= args.max_rmse_cm / 100
        rows.append({"i": i, "j": j, "odometry": odo, "fitness": round(res.fitness, 3),
                     "rmse_mm": round(res.inlier_rmse * 1000, 2), "used": bool(ok or odo),
                     "shift_mm": round(np.linalg.norm(res.transformation[:3, 3]) * 1000, 1)})
        if ok or odo:
            T = res.transformation if ok else np.eye(4)  # failed odometry: trust the ARKit poses
            graph.edges.append(REG.PoseGraphEdge(j, i, T, info, uncertain=not odo))
    REG.global_optimization(graph, REG.GlobalOptimizationLevenbergMarquardt(),
                            REG.GlobalOptimizationConvergenceCriteria(),
                            REG.GlobalOptimizationOption(max_correspondence_distance=v * 3, edge_prune_threshold=0.25,
                                                         preference_loop_closure=args.loop_preference, reference_node=0))
    # nodes hold fragment -> world corrections; T_ij above maps j onto i, i.e. pose_i^-1 pose_j
    return [np.asarray(n.pose) for n in graph.nodes], rows


def mesh_and_close(cloud: o3d.geometry.PointCloud, out: Path, args) -> dict:
    """Poisson mesh of the fused cloud, low-density trim, MeshLab repair + Close Holes, then Liepa."""

    import pymeshlab

    v = args.voxel_cm / 100
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=v * 4, max_nn=30))
    cloud.orient_normals_consistent_tangent_plane(30)
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud, depth=args.poisson_depth, n_threads=-1)
    dens = np.asarray(dens)
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, args.density_quantile))
    o3d.io.write_triangle_mesh(str(out / "mesh_poisson.ply"), mesh)

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(out / "mesh_poisson.ply"))
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_null_faces()
    ms.meshing_repair_non_manifold_edges(method="Remove Faces")
    ms.meshing_repair_non_manifold_vertices(vertdispratio=0)
    ms.meshing_remove_connected_component_by_diameter(mincomponentdiag=pymeshlab.PercentageValue(5))
    ms.meshing_remove_unreferenced_vertices()
    holes_before = ms.get_topological_measures().get("number_holes")
    ms.meshing_close_holes(maxholesize=args.max_hole_size, selfintersection=False)
    m = ms.current_mesh()
    V, F = m.vertex_matrix(), m.face_matrix()
    C = m.vertex_color_matrix()[:, :3] if m.has_vertex_color() else np.full((len(V), 3), 0.7)

    V, F, C, lm = liepa.fill(V, F, C, {"max_hole_edges": args.max_hole_size * 3, "density": 2 ** 0.5,
                                       "fairing": 2, "max_refine_iters": 30})
    closed = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F))
    closed.vertex_colors = o3d.utility.Vector3dVector(np.clip(C, 0, 1))
    closed.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(out / "mesh_closed.ply"), closed)
    return {"mesh_vertices": len(V), "mesh_faces": len(F), "holes_poisson": holes_before,
            "holes_filled_liepa": lm.get("holes_filled"), "holes_left": lm.get("holes_too_large"),
            "boundary_edges_after": lm.get("boundary_edges_after")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("capture", type=Path, help="folder with PLY/*.ply")
    ap.add_argument("--out", type=Path, default=None, help="default <capture>/fused")
    ap.add_argument("--fragment", type=int, default=100, help="consecutive frames per fragment")
    ap.add_argument("--frame-step", type=int, default=2, help="1 of every N frames inside a fragment")
    ap.add_argument("--voxel-cm", type=float, default=1.0)
    ap.add_argument("--min-frames-fragment", type=int, default=3, help="frames that must see a voxel inside a fragment")
    ap.add_argument("--min-frames", type=int, default=6, help="frames that must see a voxel in the final map")
    ap.add_argument("--min-overlap", type=float, default=0.15)
    ap.add_argument("--min-fitness", type=float, default=0.3)
    ap.add_argument("--max-rmse-cm", type=float, default=1.5)
    ap.add_argument("--loop-preference", type=float, default=2.0)
    ap.add_argument("--poisson-depth", type=int, default=11)
    ap.add_argument("--density-quantile", type=float, default=0.02)
    ap.add_argument("--max-hole-size", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=32)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    out = args.out or args.capture / "fused"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    files = sorted((args.capture / "PLY").glob("*.ply"))
    runs = [files[i:i + args.fragment] for i in range(0, len(files), args.fragment)]
    with ThreadPoolExecutor(min(args.threads, 8)) as ex:  # 8 readers: IO + numpy, bounded RAM
        built = list(ex.map(lambda r: build_fragment(r, args), runs))
    frag_pts = [b[0] for b in built if b[0] is not None]
    used, skipped = sum(b[1] for b in built), sum(b[2] for b in built)
    log.info("%d fragments from %d frames (%d truncated/empty skipped)", len(frag_pts), used, skipped)
    frags = [to_pcd(p) for p in frag_pts]

    poses, rows = register(frags, args)
    v = args.voxel_cm / 100
    before, after = [], []
    for r in rows:
        if r["used"] and not r["odometry"]:
            i, j = r["i"], r["j"]
            before.append(misalign_mm(frags[j], frags[i], np.eye(4), v))
            after.append(misalign_mm(frags[j], frags[i], np.linalg.inv(poses[i]) @ poses[j], v))
    with open(out / "pairs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    np.save(out / "fragment_poses.npy", np.stack(poses))

    moved = []
    for P, pts in zip(poses, frag_pts):
        xyz = pts[:, :3] @ P[:3, :3].T + P[:3, 3]
        moved.append(np.hstack([xyz, pts[:, 3:6], pts[:, 6:7]]))
    allp = np.concatenate(moved)
    k = keys_of(allp[:, :3], v)
    u, inv = np.unique(k, return_inverse=True)
    w = allp[:, 6]
    cnt = np.bincount(inv, w, len(u))
    mean = np.stack([np.bincount(inv, allp[:, c] * w, len(u)) for c in range(6)], 1) / cnt[:, None]
    fused = to_pcd(mean[cnt >= args.min_frames])
    fused, _ = fused.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    o3d.io.write_point_cloud(str(out / "lidar_fused.ply"), fused)
    ext = fused.get_max_bound() - fused.get_min_bound()
    log.info("fused cloud: %d points, extent %.2f x %.2f x %.2f m", len(fused.points), *ext)

    metrics = {"frames_used": used, "frames_skipped": skipped, "fragments": len(frags),
               "pairs": len(rows), "loop_pairs_used": sum(r["used"] and not r["odometry"] for r in rows),
               "misalign_before_mm": round(float(np.nanmedian(before)), 2) if before else None,
               "misalign_after_mm": round(float(np.nanmedian(after)), 2) if after else None,
               "max_correction_mm": round(max(np.linalg.norm(P[:3, 3]) for P in poses) * 1000, 1),
               "fused_points": len(fused.points), "extent_m": "x".join(f"{e:.2f}" for e in ext)}
    metrics.update(mesh_and_close(fused, out, args))
    metrics["t_s"] = round(time.time() - t0, 1)
    with open(out / "lidar_fuse.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(metrics))
        wr.writeheader()
        wr.writerow(metrics)
    for key, val in metrics.items():
        log.info("metric %s=%s", key, val)


if __name__ == "__main__":
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    main()

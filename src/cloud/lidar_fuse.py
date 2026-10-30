# PROMPT COMPLEMENT
# Tool: lidar_fuse (standalone, before the pipeline): one coherent LiDAR map out of the per-frame clouds of a
#   phone LiDAR capture (HEGE export: PLY/<0000000>.ply, one xyz+rgb cloud per frame).
# Goal: the frames overlap, so each one is registered against the growing map (point-to-plane ICP,
#   coarse-to-fine), loop closures are added between frames far apart in time that see the same place, a pose
#   graph spreads the drift, the registered frames are merged into one cloud, and a watertight-ish mesh is
#   built from it with the holes closed (Poisson -> MeshLab Close Holes -> Liepa from src/mesh/liepa.py).
# In:  <capture>/PLY/*.ply (metres; works whether the frames are already in a world frame or in the sensor
#      frame: the ICP starts from the previous pose).
# Out: <out>/{lidar_fused.ply (cloud, metres), poses.npy (N x 4 x 4), frames.txt, mesh_poisson.ply,
#      mesh_closed.ply, lidar_fuse.csv (metrics)}. lidar_fused.ply is what lidar.path of a config points to.
# Run: python -m src.cloud.lidar_fuse cozinha_lidar [--stride 5] [--voxel-cm 1] [--out cozinha_lidar/fused]
# Rules: truncated/empty PLYs (interrupted upload) are skipped and counted; a frame whose ICP fitness is below
#   --min-fitness keeps the previous pose (counted as lost); everything runs on all CPU cores.
# >>> BEGIN IMPLEMENTATION >>>
"""Fuse the per-frame clouds of a phone LiDAR capture into one registered cloud and a hole-closed mesh."""
import argparse, csv, logging, os, time
from pathlib import Path

import numpy as np
import open3d as o3d

from src.mesh import liepa

log = logging.getLogger(__name__)
REG = o3d.pipelines.registration


def load(path: Path, voxel: float) -> o3d.geometry.PointCloud | None:
    """One frame, NaN-free and voxelised; None when the file is truncated or empty."""

    try:
        pc = o3d.io.read_point_cloud(str(path))
    except Exception:
        return None
    if not pc.has_points():
        return None
    pc = pc.remove_non_finite_points().voxel_down_sample(voxel)
    return pc if len(pc.points) > 1000 else None


def normals(pc: o3d.geometry.PointCloud, voxel: float) -> o3d.geometry.PointCloud:
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 3, max_nn=30))
    return pc


def icp(src, dst, init: np.ndarray, voxels: list[float]) -> REG.RegistrationResult:
    """Coarse-to-fine point-to-plane ICP of src onto dst starting at init."""

    T, res = init, None
    for v in voxels:
        s = normals(src.voxel_down_sample(v), v)
        d = normals(dst.voxel_down_sample(v), v)
        res = REG.registration_icp(s, d, v * 2, T, REG.TransformationEstimationPointToPlane(),
                                   REG.ICPConvergenceCriteria(max_iteration=40))
        T = res.transformation
    return res


def register(frames: list, args) -> tuple[list[np.ndarray], REG.PoseGraph, dict]:
    """Sequential frame-to-frame odometry plus loop closures, as an Open3D pose graph."""

    vox = [args.voxel_cm * 4 / 100, args.voxel_cm * 2 / 100, args.voxel_cm / 100]
    graph = REG.PoseGraph()
    poses = [np.eye(4)]
    graph.nodes.append(REG.PoseGraphNode(np.eye(4)))
    lost = 0
    for i in range(1, len(frames)):
        res = icp(frames[i], frames[i - 1], np.eye(4), vox)
        T = res.transformation if res.fitness >= args.min_fitness else np.eye(4)
        lost += res.fitness < args.min_fitness
        info = REG.get_information_matrix_from_point_clouds(frames[i], frames[i - 1], vox[-1] * 2, T)
        poses.append(poses[-1] @ T)
        graph.nodes.append(REG.PoseGraphNode(poses[-1]))
        graph.edges.append(REG.PoseGraphEdge(i, i - 1, T, info, uncertain=False))
        if i % 100 == 0:
            log.info("odometry %d/%d (fitness %.2f, lost %d)", i, len(frames), res.fitness, lost)

    # loop closures: keyframes far apart in time whose centres are close
    centres = np.array([(P @ np.r_[np.asarray(f.get_center()), 1])[:3] for P, f in zip(poses, frames)])
    keys = list(range(0, len(frames), args.loop_every))
    loops = 0
    for a in keys:
        for b in keys:
            if b - a < args.loop_gap or np.linalg.norm(centres[a] - centres[b]) > args.loop_dist_m:
                continue
            init = np.linalg.inv(poses[a]) @ poses[b]
            res = icp(frames[b], frames[a], init, vox)
            if res.fitness < args.loop_fitness:
                continue
            info = REG.get_information_matrix_from_point_clouds(frames[b], frames[a], vox[-1] * 2, res.transformation)
            graph.edges.append(REG.PoseGraphEdge(b, a, res.transformation, info, uncertain=True))
            loops += 1
    log.info("loop closures: %d", loops)
    REG.global_optimization(graph, REG.GlobalOptimizationLevenbergMarquardt(), REG.GlobalOptimizationConvergenceCriteria(),
                            REG.GlobalOptimizationOption(max_correspondence_distance=vox[-1] * 2, edge_prune_threshold=0.25,
                                                         reference_node=0))
    return [n.pose for n in graph.nodes], graph, {"lost": int(lost), "loop_closures": loops}


def mesh_and_close(cloud: o3d.geometry.PointCloud, out: Path, args) -> dict:
    """Poisson mesh of the fused cloud, low-density trim, MeshLab Close Holes, then Liepa on what is left."""

    import pymeshlab

    normals(cloud, args.voxel_cm * 3 / 100)
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
    return {"mesh_vertices": len(V), "mesh_faces": len(F), "holes_filled_liepa": lm.get("holes_filled"),
            "holes_left_liepa": lm.get("holes_too_large"), "boundary_edges_after": lm.get("boundary_edges_after")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("capture", type=Path, help="folder with PLY/*.ply")
    ap.add_argument("--out", type=Path, default=None, help="default <capture>/fused")
    ap.add_argument("--stride", type=int, default=5, help="1 of every N frames (they overlap heavily)")
    ap.add_argument("--voxel-cm", type=float, default=1.0)
    ap.add_argument("--min-fitness", type=float, default=0.3)
    ap.add_argument("--loop-every", type=int, default=10, help="keyframe step (in used frames) for loop closures")
    ap.add_argument("--loop-gap", type=int, default=50, help="min used-frame gap of a loop closure")
    ap.add_argument("--loop-dist-m", type=float, default=0.75)
    ap.add_argument("--loop-fitness", type=float, default=0.5)
    ap.add_argument("--poisson-depth", type=int, default=11)
    ap.add_argument("--density-quantile", type=float, default=0.02)
    ap.add_argument("--max-hole-size", type=int, default=1000)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    out = args.out or args.capture / "fused"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    files = sorted((args.capture / "PLY").glob("*.ply"))[::args.stride]
    frames, names, bad = [], [], 0
    for f in files:
        pc = load(f, args.voxel_cm / 100)
        if pc is None:
            bad += 1
        else:
            frames.append(pc)
            names.append(f.name)
    log.info("%d frames used, %d truncated/empty skipped (stride %d)", len(frames), bad, args.stride)

    poses, _, reg = register(frames, args)
    fused = o3d.geometry.PointCloud()
    for P, f in zip(poses, frames):
        fused += o3d.geometry.PointCloud(f).transform(P)
    fused = fused.voxel_down_sample(args.voxel_cm / 100)
    fused, _ = fused.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    o3d.io.write_point_cloud(str(out / "lidar_fused.ply"), fused)
    np.save(out / "poses.npy", np.stack(poses))
    (out / "frames.txt").write_text("\n".join(names) + "\n")
    ext = fused.get_max_bound() - fused.get_min_bound()
    log.info("fused cloud: %d points, extent %.2f x %.2f x %.2f m", len(fused.points), *ext)

    metrics = {"frames_used": len(frames), "frames_skipped": bad, "stride": args.stride, **reg,
               "fused_points": len(fused.points), "extent_m": "x".join(f"{e:.2f}" for e in ext)}
    metrics.update(mesh_and_close(fused, out, args))
    metrics["t_s"] = round(time.time() - t0, 1)
    with open(out / "lidar_fuse.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(metrics))
        w.writeheader()
        w.writerow(metrics)
    for k, v in metrics.items():
        log.info("metric %s=%s", k, v)


if __name__ == "__main__":
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    main()

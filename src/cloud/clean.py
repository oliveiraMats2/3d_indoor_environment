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
# Stage: cleanup (after scale; once per source: sparse, patch_match, openmvs).
# Goal: remove outliers, even out density and crop the far tails of a point cloud before planes/mesh.
# In:  sparse -> <run_dir>/scale/points3d_cm.npy + <run_dir>/sfm/colors.npy;
#      patch_match|openmvs -> <run_dir>/dense/<source>/{points,colors}.npy (model units, multiplied by s
#      from stage_io.scale_info when unit=cm, raw otherwise).
# Out: <run_dir>/clean/<source>/{points,colors}.npy (working units, same frame as scale/points3d_cm.npy),
#      <run_dir>/clean/<source>/cloud_clean.ply; row keyed by source in <artifacts_dir>/cleanup.csv
#      (source, n_before, n_after_outlier, n_after_voxel, n_after_crop).
# YAML: cleanup.{enabled, voxel_cm, nb_neighbors, std_ratio, crop_percentiles: [lo, hi], device: auto|CPU:0|CUDA:0};
#       scale.provisional_extent_cm (via stage_io.units_per_cm when unit=model).
# Rules: statistical outlier removal -> voxel_down_sample(voxel_cm * units_per_cm) -> axis-aligned crop at
#   the per-axis crop_percentiles. Missing source -> WARNING and return None. device CUDA:0 uses the
#   open3d.t API; CUDA unavailable or failing -> WARNING and legacy CPU. voxel_cm <= 0 skips the voxel step.

# >>> BEGIN IMPLEMENTATION >>>
"""Point-cloud cleanup: statistical outliers, voxel downsampling and percentile crop."""
import logging
from pathlib import Path

import numpy as np
import open3d as o3d

from src import stage_io

log = logging.getLogger(__name__)

SOURCES = ("sparse", "patch_match", "openmvs")

def load_source(run_dir: Path, source: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Points in working units and RGB uint8 colors of a source, None when missing."""

    if source == "sparse":
        p_path, c_path = run_dir / "scale" / "points3d_cm.npy", run_dir / "sfm" / "colors.npy"
    else:
        p_path, c_path = run_dir / "dense" / source / "points.npy", run_dir / "dense" / source / "colors.npy"

    if not p_path.exists():
        log.warning("cleanup: source %s missing (%s), skipped", source, p_path)
        return None

    points = np.load(p_path).astype(np.float64).reshape(-1, 3)
    colors = np.load(c_path).astype(np.uint8).reshape(-1, 3) if c_path.exists() else None
    if colors is None or len(colors) != len(points):
        log.warning("cleanup: colors of %s missing or misaligned, using gray", source)
        colors = np.full((len(points), 3), 128, np.uint8)

    if source != "sparse":
        s, unit = stage_io.scale_info(run_dir)
        if unit == "cm":
            points = points * s

    return points, colors

def filter_legacy(points, colors, nb_neighbors: int, std_ratio: float, voxel: float):
    """Legacy CPU pipeline; returns (points, colors) after outliers and after voxel."""

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    after_outlier = (np.asarray(pcd.points), np.asarray(pcd.colors))
    if voxel > 0:
        pcd = pcd.voxel_down_sample(voxel)

    return after_outlier, (np.asarray(pcd.points), np.asarray(pcd.colors))

def filter_tensor(points, colors, nb_neighbors: int, std_ratio: float, voxel: float, device: str):
    """Same pipeline with the open3d.t API on the given device."""

    dev = o3d.core.Device(device)
    pcd = o3d.t.geometry.PointCloud(dev)
    pcd.point.positions = o3d.core.Tensor(points.astype(np.float32), device=dev)
    pcd.point.colors = o3d.core.Tensor((colors / 255.0).astype(np.float32), device=dev)
    pcd, _ = pcd.remove_statistical_outliers(nb_neighbors, std_ratio)
    after_outlier = (pcd.point.positions.cpu().numpy(), pcd.point.colors.cpu().numpy())
    if voxel > 0:
        pcd = pcd.voxel_down_sample(voxel)

    return after_outlier, (pcd.point.positions.cpu().numpy(), pcd.point.colors.cpu().numpy())

def crop_percentiles(points: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Boolean mask of the points inside the per-axis [lo, hi] percentile box."""

    if len(points) == 0:
        return np.zeros(0, bool)

    p_lo, p_hi = np.percentile(points, [lo, hi], axis=0)

    return np.all((points >= p_lo) & (points <= p_hi), axis=1)

def run(cfg: dict, run_dir: str, artifacts_dir: str, source: str) -> Path | None:

    c = cfg["cleanup"]
    if not c["enabled"]:
        log.info("cleanup disabled")
        return None

    run_dir = Path(run_dir)
    loaded = load_source(run_dir, source)
    if loaded is None:
        return None

    points, colors = loaded
    n_before = len(points)
    if n_before <= c["nb_neighbors"]:
        log.warning("cleanup: %s has only %d points, skipped", source, n_before)
        return None

    voxel = c["voxel_cm"] * stage_io.units_per_cm(run_dir, points, room_extent_cm=cfg["scale"]["provisional_extent_cm"])
    device = c.get("device", "CPU:0")
    if device == "auto":  # CUDA:0 when this Open3D build sees a CUDA GPU (Linux CUDA wheels), else CPU
        device = "CUDA:0" if o3d.core.cuda.is_available() else "CPU:0"
        log.info("cleanup: device auto -> %s", device)
    args = (points, colors, c["nb_neighbors"], c["std_ratio"], voxel)

    result = None
    if device.upper().startswith("CUDA"):
        if o3d.core.cuda.is_available():
            try:
                result = filter_tensor(*args, device=device)
            except Exception as exc:  # noqa: BLE001 - any device failure falls back to CPU
                log.warning("cleanup: %s failed (%s), falling back to legacy CPU", device, exc)
        else:
            log.warning("cleanup: CUDA not available in Open3D, falling back to legacy CPU")
    if result is None:
        result = filter_legacy(*args)

    (pts_o, _), (pts_v, col_v) = result
    keep = crop_percentiles(pts_v, *c["crop_percentiles"])
    pts_c = pts_v[keep].astype(np.float64)
    col_c = np.clip(np.round(col_v[keep] * 255.0), 0, 255).astype(np.uint8)

    out_dir = run_dir / "clean" / source
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "points.npy", pts_c)
    np.save(out_dir / "colors.npy", col_c)
    stage_io.write_ply(out_dir / "cloud_clean.ply", pts_c, col_c)

    row = {"source": source, "n_before": n_before, "n_after_outlier": len(pts_o),
           "n_after_voxel": len(pts_v), "n_after_crop": len(pts_c)}
    stage_io.log_metrics(log, row)
    stage_io.upsert_csv(Path(artifacts_dir) / "cleanup.csv", row, key="source")

    return out_dir

# <<< END IMPLEMENTATION <<<

if __name__ == "__main__":
    import tempfile

    rng = np.random.default_rng(0)
    # dense 4 m x 3 m floor plane (cm), 40k points, plus 200 gaussian outliers 50 m away
    plane = np.column_stack([rng.uniform(0, 400, 40000), rng.uniform(0, 300, 40000), rng.normal(0, 0.2, 40000)])
    outliers = rng.normal(5000, 300, (200, 3))
    pts = np.vstack([plane, outliers])
    cols = rng.integers(0, 256, (len(pts), 3)).astype(np.uint8)
    cfg = {"cleanup": {"enabled": True, "voxel_cm": 5.0, "nb_neighbors": 20, "std_ratio": 2.0,
                       "crop_percentiles": [1, 99], "device": "CPU:0"},
           "scale": {"provisional_extent_cm": 500}}

    with tempfile.TemporaryDirectory() as tmp:
        run_dir, art = Path(tmp) / "run", Path(tmp) / "art"
        (run_dir / "scale").mkdir(parents=True)
        (run_dir / "sfm").mkdir()
        (run_dir / "dense" / "patch_match").mkdir(parents=True)
        np.save(run_dir / "scale" / "scale.npy", np.array([2.0]))
        np.save(run_dir / "scale" / "points3d_cm.npy", pts)
        np.save(run_dir / "sfm" / "colors.npy", cols)
        np.save(run_dir / "dense" / "patch_match" / "points.npy", pts / 2.0)  # model units, s = 2 cm/unit
        np.save(run_dir / "dense" / "patch_match" / "colors.npy", cols)

        assert run(cfg, run_dir, art, "openmvs") is None  # missing source
        out = run(cfg, run_dir, art, "sparse")
        p = np.load(out / "points.npy")
        assert np.abs(p).max() < 1000, "far outliers survived"
        assert len(np.load(out / "colors.npy")) == len(p) and (out / "cloud_clean.ply").exists()
        cropped = crop_percentiles(p, 1, 99)
        assert cropped.mean() > 0.9

        out2 = run(cfg, run_dir, art, "patch_match")
        p2 = np.load(out2 / "points.npy")
        assert p2[:, 0].max() > 300, "dense not scaled to cm"

        out3 = run({**cfg, "cleanup": {**cfg["cleanup"], "device": "auto"}}, run_dir, art, "sparse")  # CUDA:0 or CPU:0
        assert np.abs(np.load(out3 / "points.npy")).max() < 1000

        rows = stage_io.read_csv(art / "cleanup.csv")
        assert [r["source"] for r in rows] == ["patch_match", "sparse"]
        for r in rows:
            n = [int(r[k]) for k in ("n_before", "n_after_outlier", "n_after_voxel", "n_after_crop")]
            assert n[0] - n[1] >= 200 and n[2] < n[1] and n[3] < n[2], n
        print(rows)
    print("OK")

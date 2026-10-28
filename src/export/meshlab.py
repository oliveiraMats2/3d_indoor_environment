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
# Stage: export (MeshLab bundle), after mesh.
# Goal: gather every cloud/mesh of the run in one folder MeshLab opens as a project, plus a filter script
#   (normals -> Screened Poisson -> remove isolated pieces -> remove unreferenced vertices).
# In:  exports/colmap/sparse.ply, dense/<method>/fused.ply, clean/<source>/cloud_clean.ply,
#      exports/meshlab/{poisson_*.ply, close_holes_*.ply, room_*.ply} (poisson / close_holes / room_fill stages),
#      exports/meshlab/mesh.ply (mesh stage), scale/scale.npy (stage_io.scale_info).
# Out: exports/meshlab/{sparse.ply, fused_<method>.ply, cloud_clean_<source>.ply, project.mlp, process.mlx};
#      artifacts_dir/export.csv, one row per project layer (layer, from, unit, scale_applied, n_vertices,
#      extent_x/y/z in that unit). Missing inputs are skipped with INFO.
#      Poisson threads in process.mlx = poisson.threads (fallback PORTABLE_THREADS when <= 0 or absent),
#      never the machine core count, so the script is the same on every machine.
# Units: every layer of the project is in CENTIMETRES when scale/scale.npy is finite (unit cm), the working unit
#      of clean/, mesh/ and the poisson/close_holes/room_fill/mesh copies. The two model-unit inputs
#      (exports/colmap/sparse.ply, dense/<method>/fused.ply) are written scaled by s = cm per model unit (xyz * s;
#      normals and colors kept, Open3D PLY). exports/colmap/ itself stays in model units (the COLMAP workspace
#      the densifiers read). Unknown scale (nan) -> plain copies, unit model. process.mlx diameters follow.
# YAML: export: {enabled, normals_k, normals_smooth_iter, poisson_depth, poisson_samples_per_node,
#      poisson_point_weight, isolated_diameter_frac}
# Rules: relative filenames in project.mlp; XML format and filter names/Param types match pymeshlab
#   2025.7 (MeshLab 2025.07) save_project / save_filter_script output.
# >>> BEGIN IMPLEMENTATION >>>
"""MeshLab export: copy the run's clouds/mesh, write project.mlp and process.mlx."""
import logging, os, shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from src import stage_io

log = logging.getLogger(__name__)

IDENTITY = "\n1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 \n"
MESH_LAYERS = ("poisson_*.ply", "close_holes_*.ply", "room_*.ply")  # copies written by poisson / close_holes / room_fill
PORTABLE_THREADS = 8  # process.mlx Poisson threads when poisson.threads is absent or <= 0 (-1 = all cores)

def write_scaled(src: Path, dst: Path, s: float) -> None:
    """dst = src with xyz * s (normals, colors kept); s == 1 -> plain copy."""

    if s == 1.0:
        shutil.copy2(src, dst)
        return

    import open3d as o3d
    pcd = o3d.io.read_point_cloud(str(src))
    pcd.scale(s, center=np.zeros(3))
    o3d.io.write_point_cloud(str(dst), pcd)

def collect_layers(cfg: dict, run_dir: Path, out_dir: Path) -> list[tuple[Path, str, float]]:
    """Write the available inputs into out_dir in working units; return (layer, from, scale applied) in display
    order. Model-unit inputs (colmap sparse, dense fused) are scaled by cm per model unit when the scale is metric."""

    s, unit = stage_io.scale_info(run_dir)
    s = s if unit == "cm" else 1.0
    methods = list(cfg.get("dense", {}).get("methods", []))
    sources = ["sparse"] + methods
    wanted = [(run_dir / "exports" / "colmap" / "sparse.ply", "sparse.ply", s)]
    wanted += [(run_dir / "dense" / m / "fused.ply", f"fused_{m}.ply", s) for m in methods]
    wanted += [(run_dir / "clean" / c / "cloud_clean.ply", f"cloud_clean_{c}.ply", 1.0) for c in sources]

    layers = []
    for src, name, factor in wanted:
        if not src.exists():
            log.info("export: %s missing, skipped", src.relative_to(run_dir))
            continue
        write_scaled(src, out_dir / name, factor)
        layers.append((out_dir / name, src.relative_to(run_dir).as_posix(), factor))

    for pattern in MESH_LAYERS:
        found = sorted(out_dir.glob(pattern))
        if not found:
            log.info("export: no %s in exports/meshlab, not in the project", pattern)
        layers += [(p, p.relative_to(run_dir).as_posix(), 1.0) for p in found]

    mesh = out_dir / "mesh.ply"
    if mesh.exists():
        layers.append((mesh, mesh.relative_to(run_dir).as_posix(), 1.0))
    else:
        log.info("export: exports/meshlab/mesh.ply missing (mesh stage off?), not in the project")

    return layers

def layer_bounds(path: Path) -> tuple[int, np.ndarray, np.ndarray]:
    """(n vertices, min xyz, max xyz) of a PLY cloud or mesh (n = 0, nan bounds when empty)."""

    import open3d as o3d
    pts = np.asarray(o3d.io.read_point_cloud(str(path)).points)
    if not len(pts):
        return 0, np.full(3, np.nan), np.full(3, np.nan)

    return len(pts), pts.min(0), pts.max(0)

def bbox_diag(bounds: list[tuple[int, np.ndarray, np.ndarray]]) -> float:
    """Diagonal of the joint bounding box of the layers' vertices (1.0 if nothing readable)."""

    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for n, b_lo, b_hi in bounds:
        if n:
            lo, hi = np.minimum(lo, b_lo), np.maximum(hi, b_hi)

    diag = float(np.linalg.norm(hi - lo)) if np.isfinite(lo).all() else 0.0

    return diag or 1.0

def write_project(path: Path, layers: list[Path]) -> None:

    root = ET.Element("MeshLabProject")
    group = ET.SubElement(root, "MeshGroup")
    for layer in layers:
        mesh = ET.SubElement(group, "MLMesh", label=layer.stem, idInFile="-1", visible="1",
                             filename=os.path.relpath(layer, path.parent))
        ET.SubElement(mesh, "MLMatrix44").text = IDENTITY
    ET.SubElement(root, "RasterGroup")
    ET.indent(root, space=" ")
    path.write_text("<!DOCTYPE MeshLabDocument>\n" + ET.tostring(root, encoding="unicode") + "\n")

def param(f: ET.Element, kind: str, name: str, value, **extra) -> None:

    if isinstance(value, bool):
        value = str(value).lower()
    ET.SubElement(f, "Param", type=kind, name=name, value=str(value), **{k: str(v) for k, v in extra.items()})

def poisson_threads(cfg: dict) -> int:
    """Fixed thread count for process.mlx: poisson.threads (legacy mesh.poisson_threads), else PORTABLE_THREADS."""

    threads = cfg.get("poisson", {}).get("threads", cfg.get("mesh", {}).get("poisson_threads", PORTABLE_THREADS))

    return int(threads) if threads and threads > 0 else PORTABLE_THREADS

def write_script(path: Path, ecfg: dict, diag: float, threads: int) -> None:
    """Filter names / Param names and types as written by pymeshlab 2025.7 save_filter_script."""

    root = ET.Element("FilterScript")
    f = ET.SubElement(root, "filter", name="Compute normals for point sets")
    param(f, "RichInt", "K", ecfg["normals_k"])
    param(f, "RichInt", "smoothIter", ecfg["normals_smooth_iter"])
    param(f, "RichBool", "flipFlag", False)
    ET.SubElement(f, "Param", type="RichPosition", name="viewPos", x="0", y="0", z="0")

    f = ET.SubElement(root, "filter", name="Surface Reconstruction: Screened Poisson")
    param(f, "RichBool", "visibleLayer", False)
    param(f, "RichInt", "depth", ecfg["poisson_depth"])
    param(f, "RichInt", "fullDepth", 5)
    param(f, "RichInt", "cgDepth", 0)
    param(f, "RichFloat", "scale", 1.1)
    param(f, "RichFloat", "samplesPerNode", ecfg["poisson_samples_per_node"])
    param(f, "RichFloat", "pointWeight", ecfg["poisson_point_weight"])
    param(f, "RichInt", "iters", 8)
    param(f, "RichBool", "confidence", False)
    param(f, "RichBool", "preClean", True)
    param(f, "RichInt", "threads", threads)

    f = ET.SubElement(root, "filter", name="Remove Isolated pieces (wrt Diameter)")
    param(f, "RichAbsPerc", "MinComponentDiag", round(ecfg["isolated_diameter_frac"] * diag, 6), min=0, max=round(diag, 6))
    param(f, "RichBool", "removeUnref", True)
    ET.SubElement(root, "filter", name="Remove Unreferenced Vertices")

    ET.indent(root, space=" ")
    path.write_text("<!DOCTYPE FilterScript>\n" + ET.tostring(root, encoding="unicode") + "\n")

def run(cfg: dict, run_dir: str, artifacts_dir: str) -> Path | None:

    ecfg = cfg["export"]
    if not ecfg["enabled"]:
        log.info("export disabled")
        return None

    run_dir = Path(run_dir)
    out_dir = run_dir / "exports" / "meshlab"
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = collect_layers(cfg, run_dir, out_dir)
    if not entries:
        log.warning("export: no cloud or mesh found under %s, nothing to export", run_dir)
        return None

    layers = [p for p, _, _ in entries]
    unit = stage_io.scale_info(run_dir)[1]
    project, script = out_dir / "project.mlp", out_dir / "process.mlx"
    write_project(project, layers)
    bounds = {p: layer_bounds(p) for p in layers}
    for i, (path, src, factor) in enumerate(entries):
        n, lo, hi = bounds[path]
        ext = np.round(hi - lo, 3)
        row = {"layer": path.name, "from": src, "unit": unit, "scale_applied": round(factor, 6), "n_vertices": n,
               "extent_x": ext[0], "extent_y": ext[1], "extent_z": ext[2]}
        stage_io.upsert_csv(Path(artifacts_dir) / "export.csv", row, None if i == 0 else "layer")  # i == 0 resets
        log.info("export: %s (%s, x%.6g) extent %s %s", path.name, src, factor, ext.tolist(), unit)
    meshes = {p for pattern in MESH_LAYERS for p in out_dir.glob(pattern)} | {out_dir / "mesh.ply"}
    clouds = [bounds[p] for p in layers if p not in meshes]
    write_script(script, ecfg, bbox_diag(clouds or list(bounds.values())), poisson_threads(cfg))
    stage_layers = ", ".join(p.name for p in layers)
    log.info("export: %d layers in %s -> %s (%s)", len(layers), unit, project, stage_layers)
    log.info("MeshLab: File > Open Project... %s", project)
    log.info("MeshLab: select a point-cloud layer, then Filters > Show current filter script > Load Script "
             "(%s) > Apply Script; Poisson writes a new layer 'Poisson mesh' (File > Export Mesh As...)", script)

    return project
# <<< END IMPLEMENTATION <<<

if __name__ == "__main__":
    import tempfile
    logging.basicConfig(level=logging.INFO)
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        rd = Path(tmp) / "results" / "t"
        pts, cols = rng.random((50, 3)), rng.integers(0, 255, (50, 3))
        stage_io.write_ply(rd / "exports/colmap/sparse.ply", pts, cols)
        stage_io.write_ply(rd / "dense/patch_match/fused.ply", pts * 2, cols)
        stage_io.write_ply(rd / "clean/sparse/cloud_clean.ply", pts, cols)
        (rd / "scale").mkdir()
        np.save(rd / "scale/scale.npy", np.array([3.0]))  # cm per model unit
        stage_io.write_ply(rd / "exports/meshlab/mesh.ply", pts)
        for name in ("close_holes_pre_raw.ply", "close_holes_pre_pre.ply", "poisson_pre.ply"):
            stage_io.write_ply(rd / "exports/meshlab" / name, pts)
        cfg = {"dense": {"methods": ["patch_match", "openmvs"]}, "poisson": {"threads": 6},
               "export": {"enabled": True, "normals_k": 10, "normals_smooth_iter": 0, "poisson_depth": 8,
                          "poisson_samples_per_node": 1.5, "poisson_point_weight": 4.0,
                          "isolated_diameter_frac": 0.1}}
        art = Path(tmp) / "artifacts"
        project = run(cfg, str(rd), str(art))
        out = rd / "exports" / "meshlab"
        import open3d as o3d
        fused = np.asarray(o3d.io.read_point_cloud(str(out / "fused_patch_match.ply")).points)
        assert np.allclose(fused, pts * 2 * 3, atol=1e-4)  # model -> cm
        assert np.allclose(np.asarray(o3d.io.read_point_cloud(str(out / "sparse.ply")).points), pts * 3, atol=1e-4)
        assert np.allclose(np.asarray(o3d.io.read_point_cloud(str(out / "cloud_clean_sparse.ply")).points), pts,
                           atol=1e-6)  # already cm: untouched
        rows = {r["layer"]: r for r in stage_io.read_csv(art / "export.csv")}
        assert rows["fused_patch_match.ply"]["scale_applied"] == "3.0" and rows["mesh.ply"]["unit"] == "cm", rows
        assert len(rows) == 7 and rows["cloud_clean_sparse.ply"]["from"] == "clean/sparse/cloud_clean.ply", rows
        expected = ["sparse.ply", "fused_patch_match.ply", "cloud_clean_sparse.ply", "poisson_pre.ply",
                    "close_holes_pre_pre.ply", "close_holes_pre_raw.ply", "mesh.ply"]
        assert all((out / n).exists() for n in expected) and not (out / "fused_openmvs.ply").exists()
        assert project.read_text().startswith("<!DOCTYPE MeshLabDocument>")
        meshes = ET.parse(project).getroot().findall("MeshGroup/MLMesh")
        assert [m.get("filename") for m in meshes] == expected
        assert all(m.findtext("MLMatrix44").split() == IDENTITY.split() for m in meshes)
        filters = [f.get("name") for f in ET.parse(out / "process.mlx").getroot().findall("filter")]
        assert filters == ["Compute normals for point sets", "Surface Reconstruction: Screened Poisson",
                           "Remove Isolated pieces (wrt Diameter)", "Remove Unreferenced Vertices"]
        threads = ET.parse(out / "process.mlx").getroot().find("filter[2]/Param[@name='threads']").get("value")
        assert threads == "6" and poisson_threads({"poisson": {"threads": -1}}) == PORTABLE_THREADS
        assert poisson_threads({"mesh": {"poisson_threads": 4}}) == 4 and poisson_threads({}) == PORTABLE_THREADS
        assert run({**cfg, "export": {**cfg["export"], "enabled": False}}, str(rd), tmp) is None
    print("OK")

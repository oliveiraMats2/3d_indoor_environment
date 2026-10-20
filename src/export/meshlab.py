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
#      exports/meshlab/mesh.ply (mesh stage). Out: exports/meshlab/{sparse.ply, fused_<method>.ply,
#      cloud_clean_<source>.ply, project.mlp, process.mlx}. Missing inputs are skipped with INFO.
# YAML: export: {enabled, normals_k, normals_smooth_iter, poisson_depth, poisson_samples_per_node,
#      poisson_point_weight, isolated_diameter_frac}
# Rules: relative filenames in project.mlp; XML format and filter names/Param types match pymeshlab
#   2025.7 (MeshLab 2025.07) save_project / save_filter_script output. Log only, no CSV.
# >>> BEGIN IMPLEMENTATION >>>
"""MeshLab export: copy the run's clouds/mesh, write project.mlp and process.mlx."""
import logging, os, shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

IDENTITY = "\n1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 \n"

def collect_layers(cfg: dict, run_dir: Path, out_dir: Path) -> list[Path]:
    """Copy the available inputs into out_dir; return the layer files (out_dir paths) in display order."""

    methods = list(cfg.get("dense", {}).get("methods", []))
    sources = ["sparse"] + methods
    wanted = [(run_dir / "exports" / "colmap" / "sparse.ply", "sparse.ply")]
    wanted += [(run_dir / "dense" / m / "fused.ply", f"fused_{m}.ply") for m in methods]
    wanted += [(run_dir / "clean" / s / "cloud_clean.ply", f"cloud_clean_{s}.ply") for s in sources]

    layers = []
    for src, name in wanted:
        if not src.exists():
            log.info("export: %s missing, skipped", src.relative_to(run_dir))
            continue
        shutil.copy2(src, out_dir / name)
        layers.append(out_dir / name)

    mesh = out_dir / "mesh.ply"
    if mesh.exists():
        layers.append(mesh)
    else:
        log.info("export: exports/meshlab/mesh.ply missing (mesh stage off?), not in the project")

    return layers

def bbox_diag(paths: list[Path]) -> float:
    """Diagonal of the joint bounding box of the layers' vertices (1.0 if nothing readable)."""

    import open3d as o3d
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for path in paths:
        pts = np.asarray(o3d.io.read_point_cloud(str(path)).points)
        if len(pts):
            lo, hi = np.minimum(lo, pts.min(0)), np.maximum(hi, pts.max(0))

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

def write_script(path: Path, ecfg: dict, diag: float) -> None:
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
    param(f, "RichInt", "threads", os.cpu_count() or 1)

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
    layers = collect_layers(cfg, run_dir, out_dir)
    if not layers:
        log.warning("export: no cloud or mesh found under %s, nothing to export", run_dir)
        return None

    project, script = out_dir / "project.mlp", out_dir / "process.mlx"
    write_project(project, layers)
    clouds = [p for p in layers if p.name != "mesh.ply"]
    write_script(script, ecfg, bbox_diag(clouds or layers))
    stage_layers = ", ".join(p.name for p in layers)
    log.info("export: %d layers -> %s (%s)", len(layers), project, stage_layers)
    log.info("MeshLab: File > Open Project... %s", project)
    log.info("MeshLab: select a point-cloud layer, then Filters > Show current filter script > Load Script "
             "(%s) > Apply Script; Poisson writes a new layer 'Poisson mesh' (File > Export Mesh As...)", script)

    return project
# <<< END IMPLEMENTATION <<<

if __name__ == "__main__":
    import tempfile
    from src import stage_io
    logging.basicConfig(level=logging.INFO)
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        rd = Path(tmp) / "results" / "t"
        pts, cols = rng.random((50, 3)), rng.integers(0, 255, (50, 3))
        stage_io.write_ply(rd / "exports/colmap/sparse.ply", pts, cols)
        stage_io.write_ply(rd / "dense/patch_match/fused.ply", pts * 2, cols)
        stage_io.write_ply(rd / "clean/sparse/cloud_clean.ply", pts, cols)
        stage_io.write_ply(rd / "exports/meshlab/mesh.ply", pts)
        cfg = {"dense": {"methods": ["patch_match", "openmvs"]},
               "export": {"enabled": True, "normals_k": 10, "normals_smooth_iter": 0, "poisson_depth": 8,
                          "poisson_samples_per_node": 1.5, "poisson_point_weight": 4.0,
                          "isolated_diameter_frac": 0.1}}
        project = run(cfg, str(rd), str(Path(tmp) / "artifacts"))
        out = rd / "exports" / "meshlab"
        expected = ["sparse.ply", "fused_patch_match.ply", "cloud_clean_sparse.ply", "mesh.ply"]
        assert all((out / n).exists() for n in expected) and not (out / "fused_openmvs.ply").exists()
        assert project.read_text().startswith("<!DOCTYPE MeshLabDocument>")
        meshes = ET.parse(project).getroot().findall("MeshGroup/MLMesh")
        assert [m.get("filename") for m in meshes] == expected
        assert all(m.findtext("MLMatrix44").split() == IDENTITY.split() for m in meshes)
        filters = [f.get("name") for f in ET.parse(out / "process.mlx").getroot().findall("filter")]
        assert filters == ["Compute normals for point sets", "Surface Reconstruction: Screened Poisson",
                           "Remove Isolated pieces (wrt Diameter)", "Remove Unreferenced Vertices"]
        assert run({**cfg, "export": {**cfg["export"], "enabled": False}}, str(rd), tmp) is None
    print("OK")

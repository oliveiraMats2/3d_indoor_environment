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
# Stage: close_holes (MeshLab "Close Holes" via pymeshlab), after poisson, before mesh.
# Goal: fill the small holes left in every Poisson mesh; variant pre = repair first (duplicate vertices/faces,
#   null faces, unreferenced vertices, non-manifold edges/vertices, isolated pieces < min_component_frac of the
#   bbox diagonal) then close; raw = close directly (MeshLab refusal -> WARNING, row ok=false, no crash).
# In:  mesh/poisson/<pv>/mesh.ply (pv in poisson.variants). Out: mesh/close_holes/<pv>_<cv>/{mesh.ply,
#   vertices.npy, faces.npy, colors.npy} (vertex colours kept), copy exports/meshlab/close_holes_<pv>_<cv>.ply,
#   artifacts close_holes.csv (one row per <pv>_<cv>).
# YAML: close_holes: {enabled, variants: [pre, raw], max_hole_size, min_component_frac, self_intersection}
# Rules: pymeshlab missing (no wheel) -> WARNING, return None; missing Poisson meshes -> WARNING, skipped.
#   Topology from MeshSet.get_topological_measures(); every metric logged and in the CSV.
# >>> BEGIN IMPLEMENTATION >>>
"""Close Holes stage: MeshLab hole filling (with or without repair) on every Poisson mesh variant."""
import logging, shutil, time
from pathlib import Path

import numpy as np

from src import stage_io

log = logging.getLogger(__name__)

def boundary_loops(faces: np.ndarray) -> int:
    """Connected components of the boundary-edge graph (edges used by exactly one face)."""

    e = np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    uniq, count = np.unique(e, axis=0, return_counts=True)
    parent = {}

    def find(a):
        while parent.setdefault(a, a) != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in uniq[count == 1].tolist():
        parent[find(a)] = find(b)

    return len({find(a) for a in parent})

def topology(ms) -> dict:
    """MeshLab topological measures; number_holes is -1 (undefined) on non-manifold meshes, then the
    boundary loops are counted directly."""

    t = ms.get_topological_measures()
    holes = t["number_holes"]
    if holes < 0:
        holes = boundary_loops(ms.current_mesh().face_matrix())
        log.info("close_holes: mesh not 2-manifold, MeshLab number_holes=-1; %d boundary loops counted instead", holes)

    return {"holes": holes, "boundary_edges": t["boundary_edges"],
            "non_manifold_edges": t["non_two_manifold_edges"]}

def repair(ms, pm, min_component_frac: float) -> None:
    """MeshLab cleaning filters (pymeshlab 2025.7 names), in the usual order."""

    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_null_faces()
    ms.meshing_remove_unreferenced_vertices()
    ms.meshing_repair_non_manifold_edges(method="Remove Faces")
    ms.meshing_repair_non_manifold_vertices(vertdispratio=0)
    ms.meshing_remove_connected_component_by_diameter(
        mincomponentdiag=pm.PercentageValue(100.0 * min_component_frac), removeunref=True)
    ms.meshing_remove_unreferenced_vertices()

def save(ms, out_dir: Path, export: Path) -> None:

    out_dir.mkdir(parents=True, exist_ok=True)
    m = ms.current_mesh()
    has_color = m.has_vertex_color()
    ms.save_current_mesh(str(out_dir / "mesh.ply"), binary=True, save_vertex_color=has_color,
                         save_face_color=False)
    np.save(out_dir / "vertices.npy", m.vertex_matrix())
    np.save(out_dir / "faces.npy", m.face_matrix().astype(np.int32))
    colors = m.vertex_color_matrix()[:, :3] if has_color else np.zeros((m.vertex_number(), 3))
    np.save(out_dir / "colors.npy", np.clip(np.round(colors * 255), 0, 255).astype(np.uint8))
    export.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out_dir / "mesh.ply", export)

def process(pm, src: Path, cv: str, ccfg: dict, out_dir: Path, export: Path) -> dict:
    """One <pv>_<cv> variant; returns the CSV row fields (except variant/input)."""

    t0 = time.perf_counter()
    ms = pm.MeshSet()
    ms.load_new_mesh(str(src))
    before = topology(ms)
    row = {"holes_before": before["holes"], "boundary_edges_before": before["boundary_edges"],
           "non_manifold_edges_before": before["non_manifold_edges"],
           "holes_after": "", "boundary_edges_after": "", "n_vertices": "", "n_faces": "", "ok": False}
    try:
        if cv == "pre":
            repair(ms, pm, ccfg["min_component_frac"])
        ms.meshing_close_holes(maxholesize=int(ccfg["max_hole_size"]),
                               selfintersection=bool(ccfg["self_intersection"]))
    except pm.PyMeshLabException as exc:
        log.warning("close_holes %s: MeshLab refused (%s); no output", out_dir.name, str(exc).strip())
        shutil.rmtree(out_dir, ignore_errors=True)  # never leave a stale mesh of an older run
        export.unlink(missing_ok=True)
        row["t_s"] = round(time.perf_counter() - t0, 3)
        return row

    after = topology(ms)
    m = ms.current_mesh()
    save(ms, out_dir, export)
    row.update(holes_after=after["holes"], boundary_edges_after=after["boundary_edges"],
               n_vertices=m.vertex_number(), n_faces=m.face_number(), ok=True,
               t_s=round(time.perf_counter() - t0, 3))

    return row

def run(cfg: dict, run_dir: str, artifacts_dir: str) -> dict | None:

    ccfg = cfg["close_holes"]
    if not ccfg["enabled"]:
        log.info("close_holes disabled")
        return None

    try:
        import pymeshlab as pm
    except ImportError as exc:
        log.warning("close_holes: pymeshlab not importable (%s); no wheel for this platform? stage skipped", exc)
        return None

    run_dir, artifacts_dir = Path(run_dir), Path(artifacts_dir)
    pvs = list(cfg.get("poisson", {}).get("variants", ["pre", "raw"]))
    inputs = {pv: run_dir / "mesh" / "poisson" / pv / "mesh.ply" for pv in pvs}
    missing = [pv for pv, p in inputs.items() if not p.exists()]
    if missing:
        log.warning("close_holes: Poisson mesh missing for variant(s) %s, skipped", missing)
    inputs = {pv: p for pv, p in inputs.items() if pv not in missing}
    if not inputs:
        log.warning("close_holes: no mesh/poisson/<variant>/mesh.ply under %s, stage skipped", run_dir)
        return None

    rows = {}
    for pv, src in inputs.items():
        for cv in ccfg["variants"]:
            name = f"{pv}_{cv}"
            out_dir = run_dir / "mesh" / "close_holes" / name
            export = run_dir / "exports" / "meshlab" / f"close_holes_{name}.ply"
            row = {"variant": name, "input": str(src.relative_to(run_dir)),
                   **process(pm, src, cv, ccfg, out_dir, export)}
            row = {k: row[k] for k in ("variant", "input", "holes_before", "holes_after", "boundary_edges_before",
                                       "boundary_edges_after", "non_manifold_edges_before", "n_vertices",
                                       "n_faces", "ok", "t_s")}
            stage_io.upsert_csv(artifacts_dir / "close_holes.csv", row, key="variant")
            stage_io.log_metrics(log, row)
            log.info("close_holes %s: holes %s -> %s, ok=%s", name, row["holes_before"], row["holes_after"], row["ok"])
            rows[name] = row

    return rows
# <<< END IMPLEMENTATION <<<

if __name__ == "__main__":
    import tempfile
    import open3d as o3d
    logging.basicConfig(level=logging.INFO)
    with tempfile.TemporaryDirectory() as tmp:
        rd, ad = Path(tmp) / "results" / "t", Path(tmp) / "artifacts" / "t"
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
        v, f = np.asarray(sphere.vertices), np.asarray(sphere.triangles)
        f = np.delete(f, [50, 300, 600], axis=0)                          # 3 small holes (3 edges each)
        piece = np.array([[3.0, 3, 3], [3.01, 3, 3], [3, 3.01, 3]])       # tiny isolated triangle
        v, f = np.vstack([v, piece]), np.vstack([f, [[len(v), len(v) + 1, len(v) + 2]]])
        # poisson "pre" mesh gets a duplicated face (non-manifold edges -> raw close refused), "raw" does not
        for pv, faces in (("pre", np.vstack([f, f[10:11]])), ("raw", f)):
            mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(faces))
            mesh.vertex_colors = o3d.utility.Vector3dVector(np.tile([[0.8, 0.2, 0.1]], (len(v), 1)))
            (rd / "mesh/poisson" / pv).mkdir(parents=True)
            o3d.io.write_triangle_mesh(str(rd / "mesh/poisson" / pv / "mesh.ply"), mesh)
        cfg = {"poisson": {"variants": ["pre", "raw"]},
               "close_holes": {"enabled": True, "variants": ["pre", "raw"], "max_hole_size": 30,
                               "min_component_frac": 0.01, "self_intersection": True}}
        rows = run(cfg, str(rd), str(ad))
        csv_rows = stage_io.read_csv(ad / "close_holes.csv")
        assert [r["variant"] for r in csv_rows] == ["pre_pre", "pre_raw", "raw_pre", "raw_raw"]
        pp = rows["pre_pre"]
        assert pp["ok"] and pp["holes_after"] < pp["holes_before"] and pp["boundary_edges_after"] == 0, pp
        assert not rows["pre_raw"]["ok"] and rows["raw_raw"]["ok"] and rows["raw_raw"]["holes_after"] == 1
        assert rows["raw_raw"]["n_faces"] == rows["raw_pre"]["n_faces"] + 1  # lone triangle kept, not capped
        for name, r in rows.items():
            if not r["ok"]:
                assert not (rd / "mesh/close_holes" / name).exists()
                continue
            out = rd / "mesh/close_holes" / name
            m = o3d.io.read_triangle_mesh(str(out / "mesh.ply"))
            assert len(m.triangles) == r["n_faces"] == len(np.load(out / "faces.npy"))
            assert m.has_vertex_colors() and np.allclose(np.asarray(m.vertex_colors)[0], [0.8, 0.2, 0.1], atol=0.01)
            assert (np.load(out / "colors.npy")[0] == [204, 51, 26]).all()
            assert (rd / "exports/meshlab" / f"close_holes_{name}.ply").exists()
        assert run({**cfg, "close_holes": {**cfg["close_holes"], "enabled": False}}, str(rd), str(ad)) is None
        assert run(cfg, str(Path(tmp) / "empty"), str(ad)) is None
    print("OK")

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
# Stage: 4, dense MVS, alternative method (OpenMVS).
# Goal: densify the COLMAP SfM model with OpenMVS (InterfaceCOLMAP -> DensifyPointCloud) and store
#   the cloud in the same layout as the COLMAP patch-match method, so dense_compare/cleanup/mesh can use it.
# In:  <run_dir>/dense/patch_match/ws/ (undistorted COLMAP workspace, reused when present), else
#      <run_dir>/exports/colmap/{images,sparse/0} undistorted here with `colmap image_undistorter`;
#      <frames_dir>/mvs_exclude.txt (one frame name per line, with or without extension).
# Out: <run_dir>/dense/openmvs/{fused.ply,points.npy,colors.npy} (model units, COLMAP frame),
#      workspace <run_dir>/dense/openmvs/{undistorted/,mvs_in/,scene.mvs,scene_dense.*,openmvs.log};
#      row method=openmvs in <artifacts_dir>/dense_metrics.csv (n_dense, t_s, max_image_size).
# YAML (block dense): enabled, methods (must contain openmvs), max_image_size,
#      openmvs: {bin_dir: null (PATH), resolution_level: 1, number_views: 5, max_threads: 0 (all cores)}.
#      COLMAP binary from sfm.colmap_bin.
# Version assumptions: OpenMVS 2.x (tested option names of 2.0-2.3): InterfaceCOLMAP -i/-o/--image-folder,
#      DensifyPointCloud -i/-o/-w/--resolution-level/--number-views/--max-threads. The model is handed to
#      InterfaceCOLMAP as TXT (sparse/*.txt), which every 1.x/2.x release reads; COLMAP 3.x/4.x CLI.
# Rules: missing binaries or a failing tool -> WARNING and return None (no CSV row, never raise).
#      Excluded frames are removed from the model with `colmap image_deleter` before OpenMVS.

# >>> BEGIN IMPLEMENTATION >>>

import logging, os, shutil, subprocess, time
from pathlib import Path

import numpy as np

from src import stage_io

log = logging.getLogger(__name__)

OPENMVS_TOOLS = ("InterfaceCOLMAP", "DensifyPointCloud")
INSTALL_HINT = ("install OpenMVS 2.x (conda: `micromamba install -c conda-forge openmvs`, or build from "
                "https://github.com/cdcseacave/openMVS/wiki/Building) and set dense.openmvs.bin_dir or PATH")

PLY_TYPES = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1", "short": "i2", "int16": "i2",
             "ushort": "u2", "uint16": "u2", "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
             "float": "f4", "float32": "f4", "double": "f8", "float64": "f8"}

def find_tool(name: str, bin_dir: str | None) -> str | None:

    if bin_dir:
        path = Path(bin_dir).expanduser() / name
        return str(path) if path.is_file() and os.access(path, os.X_OK) else None

    return shutil.which(name)

def build_commands(tools: dict, mvs_in: Path, work: Path, ocfg: dict) -> dict:
    """OpenMVS command lines (lists) for a COLMAP undistorted folder mvs_in = {images/, sparse/*.txt}."""

    scene, dense = work / "scene.mvs", work / "scene_dense.mvs"
    return {
        "interface": [tools["InterfaceCOLMAP"], "-i", str(mvs_in), "-o", str(scene),
                      "--image-folder", str(mvs_in / "images"), "-w", str(work)],
        "densify": [tools["DensifyPointCloud"], "-i", str(scene), "-o", str(dense), "-w", str(work),
                    "--resolution-level", str(int(ocfg["resolution_level"])),
                    "--number-views", str(int(ocfg["number_views"])),
                    "--max-threads", str(int(ocfg["max_threads"]))],
    }

def undistort_command(colmap: str, image_dir: Path, sparse_dir: Path, out: Path, max_size: int) -> list:

    return [colmap, "image_undistorter", "--image_path", str(image_dir), "--input_path", str(sparse_dir),
            "--output_path", str(out), "--output_type", "COLMAP", "--max_image_size", str(int(max_size))]

def call(cmd: list, log_path: Path, cwd: Path) -> bool:

    log.info("run %s", " ".join(cmd))
    with open(log_path, "a") as fh:
        fh.write("$ " + " ".join(cmd) + "\n")
        fh.flush()
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=cwd).returncode

    if rc != 0:
        log.warning("%s failed (exit %d), see %s", Path(cmd[0]).name, rc, log_path)

    return rc == 0

def read_ply(path) -> tuple[np.ndarray, np.ndarray | None]:
    """Vertex xyz (float64) and rgb (uint8 or None) of an ascii/binary PLY whose first element is vertex."""

    with open(path, "rb") as fh:
        if fh.readline().strip() != b"ply":
            raise ValueError(f"{path}: not a PLY file")
        fmt, n, props, element = None, 0, [], None
        while True:
            line = fh.readline()
            if not line:
                raise ValueError(f"{path}: truncated header")
            words = line.decode("ascii", "replace").split()
            if not words or words[0] in ("comment", "obj_info"):
                continue
            if words[0] == "end_header":
                break
            if words[0] == "format":
                fmt = words[1]
            elif words[0] == "element":
                element = words[1]
                if element == "vertex":
                    n = int(words[2])
                elif n == 0:
                    raise ValueError(f"{path}: element {element} before vertex is not supported")
            elif words[0] == "property" and element == "vertex":
                if words[1] == "list":
                    raise ValueError(f"{path}: list property in vertex is not supported")
                props.append((words[2], PLY_TYPES[words[1]]))

        if fmt == "ascii":
            table = np.loadtxt(fh, max_rows=n, ndmin=2)
            data = {name: table[:, i] for i, (name, _) in enumerate(props)}
        else:
            order = "<" if fmt == "binary_little_endian" else ">"
            dtype = np.dtype([(name, order + t) for name, t in props])
            data = np.frombuffer(fh.read(n * dtype.itemsize), dtype=dtype, count=n)

    points = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)
    names = [name for name, _ in props]
    rgb = next((c for c in (("red", "green", "blue"), ("r", "g", "b"), ("diffuse_red", "diffuse_green",
                "diffuse_blue")) if all(k in names for k in c)), None)
    colors = None if rgb is None else np.stack([data[k] for k in rgb], axis=1).astype(np.uint8)

    return points, colors

def excluded_names(frames_dir: Path, model_names: list[str]) -> list[str]:
    """Model image names whose stem is listed in mvs_exclude.txt."""

    path = frames_dir / "mvs_exclude.txt"
    if not path.exists():
        return []

    stems = {Path(line.strip()).stem for line in path.read_text().splitlines() if line.strip()}
    return [name for name in model_names if Path(name).stem in stems]

def model_image_names(images_txt: Path) -> list[str]:

    lines = [l for l in images_txt.read_text().splitlines() if not l.startswith("#")]  # POINTS2D may be empty

    return [line.split()[9] for line in lines[::2] if line.strip()]

def prepare_input(colmap: str, src_ws: Path, mvs_in: Path, frames_dir: Path, log_path: Path) -> bool:
    """mvs_in = {images -> src_ws/images, sparse/*.txt without the excluded frames}."""

    src_ws, mvs_in = src_ws.resolve(), mvs_in.resolve()

    if mvs_in.exists():
        shutil.rmtree(mvs_in)
    sparse = mvs_in / "sparse"
    sparse.mkdir(parents=True)
    (mvs_in / "images").symlink_to(os.path.relpath(src_ws / "images", mvs_in), target_is_directory=True)  # relative: movable

    convert = [colmap, "model_converter", "--input_path", str(src_ws / "sparse"), "--output_path", str(sparse),
               "--output_type", "TXT"]
    if not call(convert, log_path, mvs_in):
        return False

    drop = excluded_names(frames_dir, model_image_names(sparse / "images.txt"))
    if not (frames_dir / "mvs_exclude.txt").exists():
        log.warning("no mvs_exclude.txt in %s: OpenMVS uses every registered image", frames_dir)
    if not drop:
        return True

    log.info("excluding %d images from OpenMVS (mvs_exclude.txt)", len(drop))
    names_file, deleted = mvs_in / "exclude_names.txt", mvs_in / "sparse_deleted"
    names_file.write_text("\n".join(drop) + "\n")
    deleted.mkdir()
    delete = [colmap, "image_deleter", "--input_path", str(sparse), "--output_path", str(deleted),
              "--image_names_path", str(names_file)]
    if not call(delete, log_path, mvs_in):
        return False

    shutil.rmtree(sparse)
    sparse.mkdir()
    convert = [colmap, "model_converter", "--input_path", str(deleted), "--output_path", str(sparse),
               "--output_type", "TXT"]
    return call(convert, log_path, mvs_in)

def run(cfg: dict, run_dir: str, artifacts_dir: str) -> Path | None:

    dcfg = cfg["dense"]
    if not dcfg["enabled"] or "openmvs" not in dcfg["methods"]:
        log.info("dense openmvs disabled")
        return None

    ocfg = dcfg["openmvs"]
    tools = {name: find_tool(name, ocfg["bin_dir"]) for name in OPENMVS_TOOLS}
    missing = [name for name, path in tools.items() if path is None]
    if missing:
        log.warning("OpenMVS not found (%s, bin_dir=%s): skipping openmvs; %s",
                    ", ".join(missing), ocfg["bin_dir"], INSTALL_HINT)
        return None

    colmap = shutil.which(cfg["sfm"]["colmap_bin"])
    if colmap is None:
        log.warning("COLMAP binary %s not found: skipping openmvs", cfg["sfm"]["colmap_bin"])
        return None

    run_path = Path(run_dir).resolve()
    work = run_path / "dense" / "openmvs"
    work.mkdir(parents=True, exist_ok=True)
    log_path = work / "openmvs.log"
    log_path.write_text("")
    t0 = time.perf_counter()

    src_ws = run_path / "dense" / "patch_match" / "ws"
    if (src_ws / "images").is_dir() and (src_ws / "sparse").is_dir():
        log.info("reusing undistorted workspace %s", src_ws)
    else:
        colmap_ws = run_path / "exports" / "colmap"
        if not (colmap_ws / "sparse" / "0").is_dir():
            log.warning("no SfM model in %s: skipping openmvs", colmap_ws / "sparse" / "0")
            return None
        src_ws = work / "undistorted"
        cmd = undistort_command(colmap, colmap_ws / "images", colmap_ws / "sparse" / "0", src_ws,
                                dcfg["max_image_size"])
        if not call(cmd, log_path, work):
            return None

    frames_dir = Path(cfg["extraction"]["output_dir"]).resolve()
    mvs_in = work / "mvs_in"
    if not prepare_input(colmap, src_ws, mvs_in, frames_dir, log_path):
        return None

    cmds = build_commands(tools, mvs_in, work, ocfg)
    if not call(cmds["interface"], log_path, work) or not call(cmds["densify"], log_path, work):
        return None

    dense_ply = work / "scene_dense.ply"
    if not dense_ply.exists():
        log.warning("OpenMVS produced no %s, see %s", dense_ply, log_path)
        return None

    points, colors = read_ply(dense_ply)
    if colors is None:
        log.warning("OpenMVS cloud has no colors: writing gray")
        colors = np.full((len(points), 3), 128, np.uint8)

    fused = work / "fused.ply"
    shutil.copyfile(dense_ply, fused)
    np.save(work / "points.npy", points)
    np.save(work / "colors.npy", colors)

    row = {"method": "openmvs", "n_dense": len(points), "t_s": round(time.perf_counter() - t0, 2),
           "max_image_size": dcfg["max_image_size"]}
    stage_io.log_metrics(log, row)
    stage_io.upsert_csv(Path(artifacts_dir) / "dense_metrics.csv", row, key="method")

    return fused

if __name__ == "__main__":
    import tempfile
    logging.basicConfig(level=logging.INFO)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ocfg = {"bin_dir": str(tmp / "nobin"), "resolution_level": 1, "number_views": 5, "max_threads": 0}
        cfg = {"dense": {"enabled": True, "methods": ["patch_match", "openmvs"], "max_image_size": 1600,
                         "openmvs": ocfg},
               "sfm": {"colmap_bin": "colmap"}, "extraction": {"output_dir": str(tmp / "frames")}}

        # command building
        tools = {"InterfaceCOLMAP": "/b/InterfaceCOLMAP", "DensifyPointCloud": "/b/DensifyPointCloud"}
        cmds = build_commands(tools, tmp / "in", tmp / "w", ocfg)
        assert cmds["interface"][:5] == ["/b/InterfaceCOLMAP", "-i", str(tmp / "in"), "-o", str(tmp / "w/scene.mvs")]
        assert "--image-folder" in cmds["interface"] and str(tmp / "in/images") in cmds["interface"]
        d = cmds["densify"]
        assert d[0] == "/b/DensifyPointCloud" and d[d.index("-i") + 1] == str(tmp / "w/scene.mvs")
        assert d[d.index("--resolution-level") + 1] == "1" and d[d.index("--number-views") + 1] == "5"
        assert d[d.index("--max-threads") + 1] == "0"
        u = undistort_command("colmap", tmp / "img", tmp / "sp", tmp / "out", 1600)
        assert u[1] == "image_undistorter" and u[u.index("--max_image_size") + 1] == "1600"

        # missing-binary skip: None, no row, no workspace
        art = tmp / "artifacts"
        assert run(cfg, str(tmp / "run"), str(art)) is None
        assert not (art / "dense_metrics.csv").exists() and not (tmp / "run").exists()
        ocfg["bin_dir"] = None
        if shutil.which("InterfaceCOLMAP") is None:
            assert run(cfg, str(tmp / "run"), str(art)) is None and not (art / "dense_metrics.csv").exists()
        cfg["dense"]["enabled"] = False
        assert run(cfg, str(tmp / "run"), str(art)) is None
        fake = tmp / "bin"
        fake.mkdir()
        (fake / "InterfaceCOLMAP").write_text("#!/bin/sh\n")
        (fake / "InterfaceCOLMAP").chmod(0o755)
        assert find_tool("InterfaceCOLMAP", str(fake)) and find_tool("DensifyPointCloud", str(fake)) is None

        # ply -> npy (binary with normals, like OpenMVS, and ascii)
        rng = np.random.default_rng(0)
        pts, cols = rng.normal(size=(50, 3)), rng.integers(0, 256, (50, 3)).astype(np.uint8)
        stage_io.write_ply(tmp / "a.ply", pts, cols)
        p, c = read_ply(tmp / "a.ply")
        assert np.allclose(p, pts.astype(np.float32)) and (c == cols).all() and p.dtype == np.float64
        dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
                       ("red", "u1"), ("green", "u1"), ("blue", "u1")])
        data = np.zeros(50, dt)
        data["x"], data["y"], data["z"] = pts.T
        data["red"], data["green"], data["blue"] = cols.T
        props = "".join(f"property {'float' if dt[n].kind == 'f' else 'uchar'} {n}\n" for n in dt.names)
        with open(tmp / "b.ply", "wb") as fh:
            fh.write(f"ply\nformat binary_little_endian 1.0\ncomment OpenMVS\nelement vertex 50\n{props}"
                     "end_header\n".encode())
            fh.write(data.tobytes())
        p, c = read_ply(tmp / "b.ply")
        assert np.allclose(p, pts.astype(np.float32)) and (c == cols).all()
        (tmp / "c.ply").write_text("ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\nproperty float y\n"
                                   "property float z\nend_header\n0 1 2\n3 4 5\n")
        p, c = read_ply(tmp / "c.ply")
        assert p.shape == (2, 3) and p[1, 2] == 5 and c is None

        # mvs_exclude parsing
        (tmp / "frames").mkdir()
        (tmp / "frames" / "mvs_exclude.txt").write_text("img_0001.png\nimg_0003\n\n")
        assert excluded_names(tmp / "frames", ["img_0000.png", "img_0001.png", "img_0003.jpg"]) == \
            ["img_0001.png", "img_0003.jpg"]
        (tmp / "images.txt").write_text("# c\n1 1 0 0 0 0 0 0 1 img_0000.png\n10 20 -1\n"
                                        "2 1 0 0 0 0 0 0 1 img_0001.png\n\n3 1 0 0 0 0 0 0 1 img_0002.png\n1 2 3\n")
        assert model_image_names(tmp / "images.txt") == ["img_0000.png", "img_0001.png", "img_0002.png"]
    print("OK")

# <<< END IMPLEMENTATION <<<

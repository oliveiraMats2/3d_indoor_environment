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
# Stage: gaussian_splatting - 3D Gaussian Splatting (Kerbl et al. 2023, Inria code) on the GPU, optional
#      novel-view synthesis benchmark next to the COLMAP sparse/dense reconstruction.
# Goal: undistort the SfM model into an Inria dataset, train with the --eval split (every 8th registered image,
#      sorted by name, is a test view: llffhold 8), render the test views and score them (PSNR, SSIM, LPIPS).
# In:  <run_dir>/exports/colmap/{images,sparse/0} (sfm stage).
# Out: <run_dir>/gaussian_splatting/{data/{images,sparse/0}, model/ (Inria: cfg_args, point_cloud/iteration_<N>/
#      point_cloud.ply, test/ours_<N>/{renders,gt}, results.json, per_view.json), gs.log};
#      <run_dir>/exports/gaussian_splatting/point_cloud.ply (SuperSplat / SIBR); row run_id in
#      <artifacts_dir>/gaussian_splatting.csv.
# YAML: gaussian_splatting.{enabled, rerun, repo_dir, python, iterations, eval, resolution, max_image_size,
#      data_device, sh_degree, gpu_index, timeout_s}; sfm.colmap_bin (image_undistorter).
# Rules: interpreter or repo missing, no NVIDIA GPU (nvidia-smi), no colmap binary or no sparse model -> WARNING
#      and return None (never crash main.py); reuse data/ and the trained model when rerun is false (only the
#      missing render/metrics steps run); every command is timed and appended to gs.log; the Inria scripts run
#      with cwd = repo_dir and CUDA_VISIBLE_DEVICES = gpu_index.

# >>> BEGIN IMPLEMENTATION >>>
"""3D Gaussian Splatting (Inria): undistorted COLMAP dataset, train.py, render.py and metrics.py as subprocesses."""
import json
import logging
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

from src import stage_io
from src.sfm.colmap import colmap_info

log = logging.getLogger(__name__)

STAGE = "gaussian_splatting"
LLFFHOLD = 8  # Inria --eval: sorted image names, index % 8 == 0 -> test view
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")

def gpu_available() -> bool:
    """`nvidia-smi -L` lists a GPU (the Inria rasterizer is CUDA only)."""

    if not shutil.which("nvidia-smi"):
        return False
    try:
        return subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False

def tool_paths(gcfg: dict) -> tuple[Path, Path]:
    """(interpreter, repo_dir) with ~ expanded."""

    return Path(gcfg["python"]).expanduser(), Path(gcfg["repo_dir"]).expanduser()

def build_commands(python, repo, colmap: str, image_path, sparse_path, data, model,
                   gcfg: dict) -> dict[str, list[str]]:
    """argv of undistort (COLMAP), train, render and metrics (Inria scripts, absolute paths: cwd = repo)."""

    py, repo, data, model = str(python), Path(repo), Path(data).absolute(), Path(model).absolute()
    it = str(gcfg["iterations"])
    undistort = [colmap, "image_undistorter", "--image_path", str(image_path), "--input_path", str(sparse_path),
                 "--output_path", str(data), "--output_type", "COLMAP",
                 "--max_image_size", str(gcfg["max_image_size"])]
    train = [py, str(repo / "train.py"), "-s", str(data), "-m", str(model),
             "--iterations", it, "-r", str(gcfg["resolution"]), "--data_device", gcfg["data_device"],
             "--sh_degree", str(gcfg["sh_degree"]), "--test_iterations", it, "--save_iterations", it,
             "--quiet", "--disable_viewer"]
    if gcfg["eval"]:
        train.insert(6, "--eval")
    render = [py, str(repo / "render.py"), "-m", str(model), "--iteration", it, "--skip_train", "--quiet"]
    metrics = [py, str(repo / "metrics.py"), "-m", str(model)]

    return {"undistort": undistort, "train": train, "render": render, "metrics": metrics}

def move_sparse(data) -> int:
    """image_undistorter writes data/sparse/*.bin; Inria reads data/sparse/0/. Moves the files, returns how many."""

    sparse = Path(data) / "sparse"
    dst = sparse / "0"
    dst.mkdir(parents=True, exist_ok=True)
    files = [p for p in sparse.iterdir() if p.is_file()]
    for p in files:
        shutil.move(str(p), str(dst / p.name))

    return len(files)

def list_images(images_dir) -> list[Path]:

    images_dir = Path(images_dir)
    if not images_dir.is_dir():
        return []

    return sorted(p for p in images_dir.iterdir() if p.suffix in IMAGE_EXT)

def data_ready(data) -> bool:
    """Prepared Inria dataset: data/sparse/0/cameras.{bin,txt} and at least one image."""

    sp = Path(data) / "sparse" / "0"
    has_model = (sp / "cameras.bin").exists() or (sp / "cameras.txt").exists()

    return has_model and bool(list_images(Path(data) / "images"))

def split_counts(n_images: int, use_eval: bool) -> tuple[int, int]:
    """(n_train, n_test) of the Inria split: --eval holds out index % LLFFHOLD == 0, else every image trains."""

    n_test = math.ceil(n_images / LLFFHOLD) if use_eval else 0

    return n_images - n_test, n_test

def read_results(path, iterations: int) -> dict:
    """{psnr, ssim, lpips} of ours_<iterations> in results.json (else the highest ours_*); {} if missing."""

    path = Path(path)
    if not path.exists():
        return {}

    res = json.loads(path.read_text())
    key = f"ours_{iterations}"
    if key not in res:
        its = sorted((int(k.split("_")[-1]), k) for k in res if k.startswith("ours_") and k.split("_")[-1].isdigit())
        if not its:
            return {}
        key = its[-1][1]

    m = res[key]

    return {"psnr": round(float(m["PSNR"]), 4), "ssim": round(float(m["SSIM"]), 4),
            "lpips": round(float(m["LPIPS"]), 4)}

def ply_vertex_count(path) -> int:
    """`element vertex N` of a PLY header (= number of Gaussians of an Inria point_cloud.ply)."""

    with open(path, "rb") as fh:
        for raw in fh:
            line = raw.decode("ascii", errors="replace").strip()
            if line.startswith("element vertex"):
                return int(line.split()[-1])
            if line == "end_header":
                break

    return 0

def run_cmd(cmd: list[str], log_path: Path, cwd=None, env=None, timeout=None) -> float:

    name = Path(cmd[1]).name if cmd[1].endswith(".py") else cmd[1]
    log.info("running %s", name)
    t0 = time.perf_counter()
    with open(log_path, "a") as fh:
        fh.write(f"\n$ {' '.join(cmd)}\n")
        fh.flush()
        try:
            proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=cwd, env=env, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"{name} timed out after {timeout} s (gaussian_splatting.timeout_s), see {log_path}")
    dt = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"{name} failed (code {proc.returncode}), see {log_path}")
    log.info("%s took %.1f s", name, dt)

    return dt

def export_ply(ply: Path, run_dir) -> Path:

    out = Path(run_dir) / "exports" / STAGE / "point_cloud.ply"
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ply, out)

    return out

def finish(row: dict, ply: Path, results: Path, data: Path, gcfg: dict, run_dir, csv_path) -> Path:
    """Counts, metrics and export of a trained model; upserts the run_id row and returns the exported ply."""

    n_images = len(list_images(data / "images")) or row.get("n_images") or ""
    if n_images != "":
        row["n_train"], row["n_test"] = split_counts(int(n_images), gcfg["eval"])
    row["n_images"] = n_images
    row["n_gaussians"] = ply_vertex_count(ply)
    row.update(read_results(results, gcfg["iterations"]))
    times = [row[k] for k in ("t_undistort_s", "t_train_s", "t_render_s", "t_metrics_s") if row.get(k) not in ("", None)]
    row["t_s"] = round(sum(float(t) for t in times), 2) if times else ""
    out = export_ply(ply, run_dir)
    stage_io.upsert_csv(csv_path, row, key="run_id")
    stage_io.log_metrics(log, row)
    log.info("gaussian_splatting: %s (open in SuperSplat or SIBR)", out)

    return out

def run(cfg: dict, run_dir: str, artifacts_dir: str) -> Path | None:
    """Returns <run_dir>/exports/gaussian_splatting/point_cloud.ply or None when skipped/failed."""

    gcfg = cfg[STAGE]
    if not gcfg["enabled"]:
        log.info("gaussian_splatting disabled")
        return None

    out_dir = Path(run_dir) / STAGE
    data, model, log_path = out_dir / "data", out_dir / "model", out_dir / "gs.log"
    ply = model / "point_cloud" / f"iteration_{gcfg['iterations']}" / "point_cloud.ply"
    results = model / "results.json"
    csv_path = Path(artifacts_dir) / f"{STAGE}.csv"
    colmap_dir = Path(run_dir) / "exports" / "colmap"
    run_id = cfg["run"]["id"]
    old = next((r for r in stage_io.read_csv(csv_path) if r.get("run_id") == str(run_id)), {})
    row = {"run_id": run_id, "n_images": "", "n_train": "", "n_test": "", "iterations": gcfg["iterations"],
           "resolution": gcfg["resolution"], "max_image_size": gcfg["max_image_size"], "n_gaussians": "",
           "psnr": "", "ssim": "", "lpips": "", "t_undistort_s": "", "t_train_s": "", "t_render_s": "",
           "t_metrics_s": "", "t_s": ""}

    if ply.exists() and results.exists() and not gcfg["rerun"]:
        log.info("reusing %s (gaussian_splatting.rerun=false)", model)
        row = {**row, **{k: v for k, v in old.items() if k in row and k.startswith("t_")}}
        return finish(row, ply, results, data, gcfg, run_dir, csv_path)

    python, repo = tool_paths(gcfg)
    colmap = cfg.get("sfm", {}).get("colmap_bin", "colmap")
    if not python.is_file():
        log.warning("gaussian_splatting skipped: interpreter %s not found (gaussian_splatting.python)", python)
        return None
    if not (repo / "train.py").is_file():
        log.warning("gaussian_splatting skipped: no train.py in %s (gaussian_splatting.repo_dir)", repo)
        return None
    if not gpu_available():
        log.warning("gaussian_splatting skipped: no NVIDIA GPU (nvidia-smi missing or lists no GPU)")
        return None
    if not (colmap_dir / "sparse" / "0").is_dir():
        log.warning("gaussian_splatting skipped: no sparse model in %s", colmap_dir / "sparse" / "0")
        return None

    reuse_data = data_ready(data) and not gcfg["rerun"]
    if not reuse_data and colmap_info(colmap)["path"] is None:
        log.warning("gaussian_splatting skipped: COLMAP binary '%s' not found (image_undistorter)", colmap)
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    if gcfg["rerun"] or not log_path.exists():
        log_path.write_text("")
    cmds = build_commands(python, repo, colmap, colmap_dir / "images", colmap_dir / "sparse" / "0", data, model, gcfg)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gcfg["gpu_index"])}
    timeout = gcfg["timeout_s"]

    try:
        if reuse_data:
            log.info("reusing the undistorted dataset %s", data)
            row["t_undistort_s"] = old.get("t_undistort_s", "")
        else:
            shutil.rmtree(data, ignore_errors=True)
            row["t_undistort_s"] = round(run_cmd(cmds["undistort"], log_path, timeout=timeout), 2)
            log.info("moved %d model files to %s", move_sparse(data), data / "sparse" / "0")

        if ply.exists() and not gcfg["rerun"]:
            log.info("reusing the trained model %s (render and metrics only)", ply)
            row["t_train_s"] = old.get("t_train_s", "")
        else:
            shutil.rmtree(model, ignore_errors=True)
            row["t_train_s"] = round(run_cmd(cmds["train"], log_path, cwd=repo, env=env, timeout=timeout), 2)

        if gcfg["eval"]:
            row["t_render_s"] = round(run_cmd(cmds["render"], log_path, cwd=repo, env=env, timeout=timeout), 2)
            row["t_metrics_s"] = round(run_cmd(cmds["metrics"], log_path, cwd=repo, env=env, timeout=timeout), 2)
    except RuntimeError as exc:
        log.warning("gaussian_splatting failed: %s", exc)
        return None

    if not ply.exists():
        log.warning("gaussian_splatting failed: %s not written, see %s", ply, log_path)
        return None
    if gcfg["eval"] and not results.exists():
        log.warning("gaussian_splatting: metrics.py wrote no %s (see %s): psnr/ssim/lpips left empty", results, log_path)

    return finish(row, ply, results, data, gcfg, run_dir, csv_path)

# <<< END IMPLEMENTATION <<<

if __name__ == "__main__":
    import tempfile

    import numpy as np

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    gs_cfg = {"enabled": True, "rerun": False, "repo_dir": "~/gaussian-splatting",
              "python": "~/gaussian-splatting/.venv/bin/python", "iterations": 300, "eval": True, "resolution": -1,
              "max_image_size": 1600, "data_device": "cpu", "sh_degree": 3, "gpu_index": "0", "timeout_s": None}

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cmds = build_commands("/py", "/repo", "colmap", "i", "s", tmp / "d", tmp / "m", gs_cfg)
        u, tr, re_, me = cmds["undistort"], cmds["train"], cmds["render"], cmds["metrics"]
        assert u[1] == "image_undistorter" and u[u.index("--max_image_size") + 1] == "1600"
        assert u[u.index("--output_type") + 1] == "COLMAP" and u[u.index("--output_path") + 1] == str(tmp / "d")
        assert tr[:2] == ["/py", "/repo/train.py"] and "--eval" in tr and "--quiet" in tr and "--disable_viewer" in tr
        assert tr[tr.index("-s") + 1] == str(tmp / "d") and tr[tr.index("-m") + 1] == str(tmp / "m")
        assert tr[tr.index("--iterations") + 1] == "300" and tr[tr.index("-r") + 1] == "-1"
        assert tr[tr.index("--data_device") + 1] == "cpu" and tr[tr.index("--sh_degree") + 1] == "3"
        assert tr[tr.index("--test_iterations") + 1] == "300" and tr[tr.index("--save_iterations") + 1] == "300"
        assert "--eval" not in build_commands("p", "r", "c", "i", "s", "d", "m", {**gs_cfg, "eval": False})["train"]
        assert re_[1] == "/repo/render.py" and "--skip_train" in re_ and re_[re_.index("--iteration") + 1] == "300"
        assert me[1:] == ["/repo/metrics.py", "-m", str(tmp / "m")]

        # sparse/*.bin -> sparse/0/ (image_undistorter layout -> Inria layout)
        fake = tmp / "fake"
        (fake / "sparse").mkdir(parents=True)
        (fake / "images").mkdir()
        for n in ("cameras.bin", "images.bin", "points3D.bin"):
            (fake / "sparse" / n).write_bytes(b"x")
        assert not data_ready(fake)
        (fake / "images" / "img_0000.png").write_bytes(b"x")
        assert move_sparse(fake) == 3 and sorted(p.name for p in (fake / "sparse" / "0").iterdir()) == \
            ["cameras.bin", "images.bin", "points3D.bin"]
        assert not any(p.is_file() for p in (fake / "sparse").iterdir()) and data_ready(fake)
        assert move_sparse(fake) == 0
        assert split_counts(17, True) == (14, 3) and split_counts(16, True) == (14, 2) and split_counts(5, False) == (5, 0)

        # results.json: exact iteration, fallback to the highest ours_*, missing
        rj = tmp / "results.json"
        rj.write_text(json.dumps({"ours_7000": {"SSIM": 0.5, "PSNR": 20.0, "LPIPS": 0.4},
                                  "ours_300": {"SSIM": 0.81234567, "PSNR": 25.123456, "LPIPS": 0.2}}))
        assert read_results(rj, 300) == {"psnr": 25.1235, "ssim": 0.8123, "lpips": 0.2}
        assert read_results(rj, 999)["psnr"] == 20.0 and read_results(tmp / "nope.json", 300) == {}

        # tiny PLY: vertex count from the header
        stage_io.write_ply(tmp / "t.ply", np.zeros((7, 3)))
        assert ply_vertex_count(tmp / "t.ply") == 7

        cfg = {STAGE: {**gs_cfg, "enabled": False}, "run": {"id": "r1"}, "sfm": {"colmap_bin": "colmap"}}
        assert run(cfg, str(tmp / "run"), str(tmp / "art")) is None
        cfg[STAGE] = {**gs_cfg, "python": str(tmp / "no" / "python")}
        assert run(cfg, str(tmp / "run"), str(tmp / "art")) is None
        cfg[STAGE] = {**gs_cfg, "python": str(tmp / "t.ply"), "repo_dir": str(tmp / "norepo")}  # interpreter "exists"
        assert run(cfg, str(tmp / "run"), str(tmp / "art")) is None

        # reuse path: synthetic data/ + model/, no Inria and no COLMAP call
        cfg[STAGE] = {**gs_cfg, "python": str(tmp / "no" / "python")}
        gsd = tmp / "run" / STAGE
        (gsd / "data" / "images").mkdir(parents=True)
        for i in range(17):
            (gsd / "data" / "images" / f"img_{i:04d}.png").write_bytes(b"x")
        stage_io.write_ply(gsd / "model" / "point_cloud" / "iteration_300" / "point_cloud.ply", np.zeros((42, 3)))
        shutil.copy(rj, gsd / "model" / "results.json")
        stage_io.upsert_csv(tmp / "art" / f"{STAGE}.csv", {"run_id": "r1", "t_train_s": 12.5, "t_render_s": 1.5},
                            key="run_id")
        out = run(cfg, str(tmp / "run"), str(tmp / "art"))
        assert out == tmp / "run" / "exports" / STAGE / "point_cloud.ply" and ply_vertex_count(out) == 42
        rows = stage_io.read_csv(tmp / "art" / f"{STAGE}.csv")
        assert len(rows) == 1 and rows[0]["run_id"] == "r1"
        r = rows[0]
        assert (r["n_images"], r["n_train"], r["n_test"], r["n_gaussians"]) == ("17", "14", "3", "42")
        assert (r["psnr"], r["ssim"], r["lpips"], r["iterations"]) == ("25.1235", "0.8123", "0.2", "300")
        assert r["t_train_s"] == "12.5" and r["t_s"] == "14.0"
        assert list(r) == ["run_id", "n_images", "n_train", "n_test", "iterations", "resolution", "max_image_size",
                           "n_gaussians", "psnr", "ssim", "lpips", "t_undistort_s", "t_train_s", "t_render_s",
                           "t_metrics_s", "t_s"]

    print("OK")

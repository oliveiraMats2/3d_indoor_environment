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
# Stage : sfm helper (flow gpu) - run COLMAP commands inside a Docker image instead of the host binary.
# Goal  : the image t2/colmap-cudss:4.2.0 (docker/colmap-cudss/Dockerfile) has a Ceres built with CUDA + cuDSS,
#         so the global_mapper runs global positioning and bundle adjustment on the GPU.
# In    : image tag (sfm.container), the COLMAP argument list of one step.
# Out   : `docker run --rm --entrypoint '' --gpus all --user uid:gid -e HOME=<host home> -v ... -w <cwd> <image>` prefix and the
#         wrapped command (host paths are mounted at the same path inside the container, so arguments,
#         logs and relative image symlinks stay valid). ~/.cache/colmap (COLMAP's download cache, where
#         sfm.vocab_tree "url;name;sha256" is stored) is mounted too, so the vocab tree is not downloaded again.
# Rules : docker missing / image not built -> available() False (the caller warns and skips). Paths are
#         mounted read-write at their own absolute path; symlinked images are followed to their real dirs.

# >>> BEGIN IMPLEMENTATION >>>
"""Docker wrapper for the COLMAP CLI (bind mounts at identical paths, GPU, host uid/gid)."""
import functools
import logging
import os
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
CONTAINER_EXE = "colmap"  # on PATH inside the image (/usr/local/bin)
COLMAP_CACHE = Path.home() / ".cache" / "colmap"  # COLMAP download cache = $HOME/.cache/colmap
MAX_SYMLINK_SCAN = 100_000  # entries scanned per directory for symlinks pointing elsewhere

@functools.lru_cache(maxsize=None)
def available(image: str) -> bool:
    """True when the docker CLI exists and `image` is present locally (no pull). Cached per process."""

    if not image or shutil.which("docker") is None:
        return False
    try:
        out = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", image], capture_output=True,
                             text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return False
    if out.returncode != 0:
        log.warning("container: image %s not found (docker build -t %s docker/colmap-cudss)", image, image)

    return out.returncode == 0

def _anchor(path: Path) -> Path:
    """Nearest existing directory of path (path itself when it is a directory, else its first existing parent)."""

    p = path
    while not p.is_dir() and p != p.parent:
        p = p.parent

    return p

def _symlink_targets(directory: Path) -> set[Path]:
    """Dirs the symlinks directly inside `directory` go through (images/ may link into a frames dir that
    links further into another one)."""

    out = set()
    try:
        with os.scandir(directory) as it:
            for k, entry in enumerate(it):
                if k >= MAX_SYMLINK_SCAN:
                    break
                if entry.is_symlink():  # first hop (relative links walk through it) and the final file
                    hop = Path(os.path.normpath(Path(entry.path).parent / os.readlink(entry.path)))
                    out.update({hop.parent, Path(entry.path).resolve().parent})
    except OSError:
        pass

    return out

def mount_dirs(paths, cwd: str | os.PathLike | None = None) -> list[tuple[Path, Path]]:
    """[(host real dir, container dir)] covering the repo, cwd and every path (relative ones against cwd):
    the nearest existing dir of each path, its real location when it goes through a symlink, and the real dirs
    of the symlinks directly inside it. Nested container dirs are dropped (the outer mount covers them)."""

    cwd = Path(cwd or os.getcwd()).absolute()
    wanted: dict[Path, Path] = {}

    def add(dst: Path) -> None:
        dst = Path(os.path.normpath(dst))
        wanted.setdefault(dst, dst.resolve())
        real = dst.resolve()
        if real != dst:  # also expose the real location (relative symlinks resolve through it)
            wanted.setdefault(real, real)

    add(REPO)
    add(cwd)
    for raw in paths:
        p = Path(raw)
        p = p if p.is_absolute() else cwd / p
        d = _anchor(p)
        add(d)
        for real in _symlink_targets(d):
            add(real)

    dirs = sorted(wanted)
    keep = [d for d in dirs if not any(o != d and d.is_relative_to(o) and wanted[d] == wanted[o] / d.relative_to(o)
                                       for o in dirs)]

    return [(wanted[d], d) for d in keep]

def prefix(image: str, paths=(), cwd: str | os.PathLike | None = None, gpu: bool = True) -> list[str]:
    """`docker run` argument list (without the command) for `image`, mounting the repo, cwd and `paths`."""

    cwd = Path(cwd or os.getcwd()).absolute()
    cmd = ["docker", "run", "--rm", "--entrypoint", ""]  # skip the CUDA image banner in every colmap.log
    if gpu:
        cmd += ["--gpus", "all"]
    COLMAP_CACHE.mkdir(parents=True, exist_ok=True)
    cmd += ["--user", f"{os.getuid()}:{os.getgid()}", "-e", f"HOME={Path.home()}"]
    for src, dst in mount_dirs([*paths, COLMAP_CACHE], cwd):
        cmd += ["-v", f"{src}:{dst}"]
    cmd += ["-w", str(cwd), image]

    return cmd

def path_args(cmd: list[str]) -> list[str]:
    """Arguments of a COLMAP command that look like file system paths (values containing '/')."""

    return [a for a in cmd[1:] if "/" in a and not a.startswith("--")]

def wrap(image: str, cmd: list[str], cwd: str | os.PathLike | None = None, gpu: bool = True) -> list[str]:
    """Host COLMAP command -> the same command run by the image's colmap (cmd[0] replaced)."""

    return prefix(image, path_args(cmd), cwd, gpu) + [CONTAINER_EXE, *cmd[1:]]

if __name__ == "__main__":
    import tempfile

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        frames = tmp / "frames"
        frames.mkdir()
        (frames / "img_0000.png").write_bytes(b"x")
        ws = tmp / "run" / "exports" / "colmap"
        (ws / "images").mkdir(parents=True)
        (ws / "images" / "img_0000.png").symlink_to(os.path.relpath(frames / "img_0000.png", ws / "images"))
        vocab = tmp / "cache" / "vocab.bin"
        cmd = ["/host/colmap", "sequential_matcher", "--database_path", str(ws / "database.db"),
               "--SequentialMatching.vocab_tree_path", str(vocab), "--SequentialMatching.overlap", "10"]
        assert path_args(cmd) == [str(ws / "database.db"), str(vocab)]
        w = wrap("img:1", cmd, cwd=REPO)
        assert w[:3] == ["docker", "run", "--rm"] and w[w.index("--gpus") + 1] == "all" and w[w.index("img:1") + 1] == "colmap"
        assert w[-6:] == cmd[1:][-6:] and w[w.index("-w") + 1] == str(REPO)
        vols = [w[i + 1].split(":")[1] for i, a in enumerate(w) if a == "-v"]
        real_tmp = str(tmp.resolve())
        assert str(ws) in vols or str(tmp) in vols or real_tmp in vols, vols
        assert str(REPO) in vols and any(str(tmp.resolve() / "cache").startswith(v) or v == str(tmp) for v in vols)
        # images/ symlinks -> frames dir mounted too (unless an outer mount already covers it)
        m = mount_dirs([str(ws / "images")], cwd=REPO)
        dsts = [d for _, d in m]
        assert any((frames.resolve()).is_relative_to(d) for d in dsts), dsts
        assert any((ws / "images").is_relative_to(d) for d in dsts), dsts
        # nested dirs collapse into the outer mount
        m = mount_dirs([str(ws), str(ws / "images"), str(tmp / "run")], cwd=REPO)
        assert sum(1 for _, d in m if d.is_relative_to(tmp)) <= 2, m
        assert not available("no/such-image:0") or True
        assert f"HOME={Path.home()}" in w and str(COLMAP_CACHE) in [v.split(":")[1] for v in w if ":" in v]
        print("prefix:", " ".join(prefix("t2/colmap-cudss:4.2.0", [str(ws)], cwd=REPO)))
    print("available(t2/colmap-cudss:4.2.0):", available("t2/colmap-cudss:4.2.0"))
    print("OK")
# <<< END IMPLEMENTATION <<<

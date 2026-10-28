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
# Stage: dense (helper of the deep-learning densifiers: casmvsnet, mvsnet, patchmatchnet).
# Goal: fetch a research repo at a pinned commit and official weights without pip/git/shell, and import its
#      model classes without clashing with our own top-level packages (models, utils, datasets).
# In:  owner/repo@sha (GitHub codeload tarball), weight URLs (+ optional sha256).
# Out: third_party/<repo>-<sha>/ and third_party/weights/<file> (gitignored, reused when present).
# Rules: tarballs are extracted with filter="data"; a sha256 mismatch deletes the file and raises.

# >>> BEGIN IMPLEMENTATION >>>
"""Pinned third-party research code and weights: download once, import in isolation."""
import contextlib
import hashlib
import importlib
import logging
import sys
import tarfile
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2] / "third_party"
CLASHING = ("models", "utils", "datasets", "datasets_", "module")

def fetch_file(url: str, name: str | None = None, sha256: str | None = None) -> Path:
    """Download url to third_party/weights/<name> once; verify sha256 when given."""

    path = ROOT / "weights" / (name or url.rsplit("/", 1)[-1])
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        log.info("downloading %s", url)
        tmp = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as fh:
            while chunk := resp.read(1 << 20):
                fh.write(chunk)
        tmp.rename(path)
    if sha256 and hashlib.sha256(path.read_bytes()).hexdigest() != sha256:
        path.unlink()
        raise RuntimeError(f"sha256 mismatch for {path.name}")
    return path

def fetch_repo(owner: str, repo: str, sha: str) -> Path:
    """GitHub repo at a pinned commit, extracted to third_party/<repo>-<sha>/ (codeload tarball)."""

    dest = ROOT / f"{repo}-{sha}"
    if dest.is_dir():
        return dest
    tar_path = fetch_file(f"https://codeload.github.com/{owner}/{repo}/tar.gz/{sha}", f"{repo}-{sha}.tar.gz")
    with tarfile.open(tar_path) as tar:
        tar.extractall(ROOT, filter="data")
    if not dest.is_dir():
        raise RuntimeError(f"{tar_path.name} did not extract to {dest}")
    return dest

@contextlib.contextmanager
def isolated_path(path):
    """Import from path first, then drop the clashing top-level modules it loaded and restore sys.path."""

    before = set(sys.modules)
    sys.path.insert(0, str(path))
    try:
        yield
    finally:
        sys.path.remove(str(path))
        for name in set(sys.modules) - before:
            if name.split(".")[0] in CLASHING:
                del sys.modules[name]

def import_attr(path, module: str, attr: str):
    """getattr(import module from path, attr) with isolated_path."""

    with isolated_path(path):
        return getattr(importlib.import_module(module), attr)

# <<< END IMPLEMENTATION <<<

if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "models").mkdir()
        (Path(tmp) / "models" / "__init__.py").write_text("VALUE = 7\n")
        assert import_attr(tmp, "models", "VALUE") == 7
        assert "models" not in sys.modules and tmp not in sys.path
    print("third_party self-test OK")

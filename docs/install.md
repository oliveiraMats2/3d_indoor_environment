# Installation

[← README](../README.md)

## Install: CPU flow (default)

Requirements:

| | Linux | macOS |
|---|---|---|
| OS | glibc >= 2.35 (Ubuntu 22.04+), x86_64 or aarch64 | 13+ on Apple silicon (arm64), 14+ on Intel |
| Python | 3.10 to 3.12 (installed by uv) | 3.10 to 3.12 (Intel: open3d 0.19, the last release with Intel wheels) |
| COLMAP >= 4.0 | built from source (https://colmap.github.io/install.html), with CUDA when there is an NVIDIA GPU | `brew install colmap` (CPU only) |
| pycolmap 4.x (optional) | only without the `colmap` CLI: extra `cuda` (NVIDIA, x86_64) or `cpu` | extra `cpu` (arm64, macOS 14+) |
| OpenMVS (optional) | build from source (https://github.com/cdcseacave/openMVS/wiki/Building) | same |

Python environment with uv (`.venv`):

```bash
uv python install 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r pyproject.toml
uv pip install -r pyproject.toml --extra cpu    # optional: pycolmap, only when there is no `colmap` CLI
                                                # (NVIDIA + Linux x86_64: --extra cuda instead; never both)
```

The platform-specific pins (open3d 0.19 vs 0.20, pymeshlab wheels) are markers in `pyproject.toml`, so the
same commands work on Linux and macOS. `uv sync` (from the pinned `uv.lock`) gives the same `.venv`.

COLMAP must be on `PATH` (or set `sfm.colmap_bin`). Missing or older than 4.0 -> the SfM stage falls back to
pycolmap when it is installed (`sfm.backend: auto`; same outputs and CSV columns, column `backend` says which
ran), else WARNING and the stage is skipped. The `cpu` and `cuda` extras are mutually exclusive (both provide
the `pycolmap` module): `uv pip uninstall pycolmap` before switching to `cuda`.
OpenMVS is looked up on `PATH` (or `dense.openmvs.bin_dir`); missing -> WARNING, `openmvs` skipped.

Check: `.venv/bin/python -m src.flows` prints `flows self-test OK`; `colmap -h` prints the version.

## Install: GPU flow (alternative, `flow: gpu`)

Tested on Linux x86_64, Python 3.12, RTX 5080 (sm_120), driver 580. The GPU environment `.venv-gpu` is separate
from `.venv` and holds the whole pipeline plus torch (CUDA 12.8 wheels), so one interpreter runs `main.py` end to
end. Needs an NVIDIA driver, Docker with the NVIDIA container toolkit and, only for the compiled extensions, a
CUDA toolkit >= 12.8 (the system nvcc 12.0 cannot target sm_120).

```bash
# 1. venv + pipeline + torch cu128 + pycolmap-cuda12 + runtime deps of the GPU techniques
uv venv .venv-gpu --python 3.12
uv pip install --python .venv-gpu/bin/python -r requirements.txt

# 2. InstantSfM, VGGT, VGGT-Omega, MapAnything without their pins (numpy 1.26 / numpy<2 / opencv 4.10)
uv pip install --python .venv-gpu/bin/python --no-deps -r requirements-nodeps.txt

# 3. bae (InstantSfM's sparse solver), compiled for sm_120; CUDA_HOME = a CUDA toolkit >= 12.8
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
TORCH_CUDA_ARCH_LIST=12.0 USE_CUDSS=0 MAX_JOBS=8 \
  uv pip install --python .venv-gpu/bin/python --no-build-isolation --no-deps -r requirements-build.txt

# 4. COLMAP 4.2.0 + Ceres 2.3 with CUDA + cuDSS (GPU global positioning and BA of flow: gpu), once (~1 h)
docker build -t t2/colmap-cudss:4.2.0 docker/colmap-cudss
```

Steps 3 and later are needed only by the techniques that use them: bae by `sfm.technique: instantsfm`, the
container by `flow: gpu` (GLOMAP). Optional extras (MASt3R-SfM, OpenMVS container, Gaussian Splatting) are in
[GPU SfM techniques](gpu_sfm.md) and [Gaussian Splatting](gaussian_splatting.md).

Check: `.venv-gpu/bin/python -c "import torch; print(torch.cuda.is_available())"` prints `True`, and
`.venv-gpu/bin/python -m src.sfm.gpu` prints `OK`.

## What changes on macOS (no CUDA)

- SfM runs on the CPU (`sfm.use_gpu: auto` resolves to false). Slower, same results.
- COLMAP patch-match stereo needs CUDA, so the `patch_match` dense method is skipped with a WARNING; without
  OpenMVS there is no dense cloud, and Poisson/close_holes/mesh fall back to the sparse cloud
  (`poisson.source: dense` -> sparse with a WARNING). Set `dense.enabled: false` to skip the attempt.
- Recording the viewer orbit (`viewer.enabled`) needs a GUI session (logged-in desktop). Over SSH the
  recording fails in its child process, the video is skipped with a WARNING and the run continues.
  `python3 view.py` also needs the desktop.

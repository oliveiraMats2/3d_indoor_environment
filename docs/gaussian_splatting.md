# Gaussian Splatting (optional)

[← README](../README.md)

`gaussian_splatting.enabled: true` runs the Inria 3DGS code (https://github.com/graphdeco-inria/gaussian-splatting)
as subprocesses with its OWN interpreter (`gaussian_splatting.python`), so torch and its CUDA extensions never
enter this repo's `.venv`. Missing interpreter, repo or NVIDIA GPU -> WARNING, stage skipped. Install recipe
(uv venv at `~/gaussian-splatting/.venv`, torch for CUDA 12.8; the submodules are compiled with
`TORCH_CUDA_ARCH_LIST=12.0` and `CUDA_HOME` pointing to a CUDA 12.8 toolkit, because the system nvcc 12.0 cannot
target the RTX 5080, sm_120):

```bash
git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting ~/gaussian-splatting
cd ~/gaussian-splatting
# gcc 13 needs it: add `#include <cstdint>` after `#include <vector>` in
# submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python plyfile tqdm opencv-python-headless joblib pillow numpy setuptools wheel ninja
export TORCH_CUDA_ARCH_LIST="12.0" CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH MAX_JOBS=8
for s in simple-knn fused-ssim diff-gaussian-rasterization; do
  uv pip install --python .venv/bin/python --no-build-isolation ./submodules/$s
done
```

The first `metrics.py` run downloads the LPIPS VGG weights (~530 MB) to `~/.cache/torch/hub/checkpoints/`.
Smoke test (194 frames, 2000 iterations, RTX 5080): train 117 s, ~3.5 GB VRAM peak with `data_device: cpu`.

Check: `~/gaussian-splatting/.venv/bin/python -c "import torch, diff_gaussian_rasterization, simple_knn;
print(torch.cuda.is_available())"` prints `True`.

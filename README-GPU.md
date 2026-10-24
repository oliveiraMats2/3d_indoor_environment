# GPU SfM techniques

`sfm.technique` picks how the sparse model is built. Every technique writes the same COLMAP workspace,
`<run_dir>/exports/colmap/{images/, database.db, sparse/0, sparse.ply, project.ini, colmap.log}`, so densification,
mesh, export and evaluation run unchanged.

| `sfm.technique` | module | what runs | interpreter |
|---|---|---|---|
| `colmap` (default) | `src/sfm/colmap.py` | COLMAP incremental mapper (CLI, or pycolmap) | `.venv` or `.venv-gpu` |
| `glomap` | `src/sfm/colmap.py` | COLMAP 4 `global_mapper` (GLOMAP), same as `sfm.mapper: global` | `.venv` or `.venv-gpu` |
| `instantsfm` | `src/sfm/instantsfm.py` | pycolmap SIFT + matching, InstantSfM global mapper in PyTorch | `.venv-gpu` |
| `mast3r_sfm` | `src/sfm/mast3r_sfm.py` | MASt3R pairs (retrieval) + sparse global alignment | `.venv-gpu` |
| `vggt` | `src/sfm/vggt.py` | VGGT feed-forward cameras (one pass) | `.venv-gpu` |

```yaml
sfm:
  technique: vggt            # colmap | glomap | instantsfm | mast3r_sfm | vggt
  ba_global_frames_ratio: 1.3  # colmap incremental only: fewer global BAs (null = COLMAP 1.1)
```

## How each technique ends in COLMAP format

- `instantsfm`: SIFT and matches go into `database.db` through pycolmap (same options as the pycolmap backend:
  `sfm.matcher`, `sfm.use_gpu`, `sfm.intrinsics: fixed` becomes the focal prior). InstantSfM solves rotation
  averaging, global positioning and its GPU bundle adjustment; the cameras of its largest cluster are then written
  into a pycolmap `Reconstruction` and the points are triangulated from the same database. Its own writer is not
  used: it leaves points2D pointing at dropped tracks and pycolmap refuses the model.
- `mast3r_sfm` and `vggt`: the network gives poses and intrinsics at 512 / 518 px; they are mapped back to the
  original frame size (resize + crop of each loader), written into a `Reconstruction` built from the pycolmap
  database (SIFT + matching on the used frames), and `pycolmap.triangulate_points` builds the tracks at full
  resolution. `ba: true` adds a pycolmap bundle adjustment (poses, focal and principal point when
  `refine_intrinsics: true`), then drops points over 4 px or under 1.5 deg.
- `mast3r_sfm` and `vggt` use a uniform subsample of `max_frames` frames (`images/` holds only those), so the
  dense stage sees fewer, wider-baseline views. `instantsfm` uses every frame by default.

Each GPU technique runs in a spawned child process (as the pycolmap backend): a CUDA error or abort leaves
`main.py` running, the VRAM is released at the end, and everything it prints goes to `colmap.log`. No shell
command is run from the code. `sfm_metrics.csv` gets `technique`, `t_<step>_s`, `vram_peak_gb` (torch peak),
`registered`, `points`, `mean_reproj_px`. A model is reused (`sfm.rerun: false`) only when
`exports/colmap/technique.txt` names the same technique.

## Licenses

| technique | code | weights | commercial use |
|---|---|---|---|
| colmap / glomap | BSD | none | yes |
| instantsfm | CC BY-NC 4.0 (cre185/InstantSfM) | none | no |
| mast3r_sfm | CC BY-NC-SA 4.0 (naver/mast3r) | MASt3R checkpoint + retrieval: CC BY-NC-SA 4.0 (naver) | no |
| vggt | VGGT License (facebookresearch/vggt, commercial allowed, no military use) | `facebook/VGGT-1B`: CC BY-NC 4.0 (default, official). `facebook/VGGT-1B-Commercial`: VGGT License, gated (Hugging Face approval + `hf auth login`) | only with the Commercial weights |

Only the official weights published by the authors are used: `facebook/VGGT-1B` and
`naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric` from the Hugging Face hub, and the MASt3R retrieval
`.pth` + codebook from `https://download.europe.naverlabs.com/ComputerVision/MASt3R/`.

## Install (`.venv-gpu`, uv, no conda for Python packages)

Tested on Linux x86_64, Python 3.12, RTX 5080 (sm_120), driver 580. The GPU venv is separate from `.venv`
(which `uv sync` manages for the CPU pipeline) and holds the whole pipeline plus torch.

```bash
# 1. venv + pipeline + torch cu128 + pycolmap-cuda12 + runtime deps of the three techniques
uv venv .venv-gpu --python 3.12
uv pip install --python .venv-gpu/bin/python -r requirements.txt

# 2. InstantSfM and VGGT without their pins (numpy 1.26 / numpy<2 / opencv 4.10)
uv pip install --python .venv-gpu/bin/python --no-deps -r requirements-nodeps.txt

# 3. bae (InstantSfM's sparse solver), compiled for sm_120 with a CUDA toolkit >= 12.8
#    (the system nvcc 12.0 cannot target sm_120; the toolkit env of the Gaussian Splatting recipe in README.md works)
export CUDA_HOME=$HOME/micromamba/envs/gs PATH=$HOME/micromamba/envs/gs/bin:$PATH
TORCH_CUDA_ARCH_LIST=12.0 USE_CUDSS=0 MAX_JOBS=8 \
  uv pip install --python .venv-gpu/bin/python --no-build-isolation --no-deps -r requirements-build.txt

# 4. MASt3R: official clone (not a pip package), put on sys.path from sfm.mast3r_sfm.repo_dir
git clone --recursive https://github.com/naver/mast3r ~/mast3r
git -C ~/mast3r checkout f5209afc300cec36239a7ac992263f36847bbba0 && git -C ~/mast3r submodule update --init --recursive

# 5. asmk (MASt3R retrieval): its committed hamming.c is Cython 0.29 and does not build on Python 3.12
git clone https://github.com/jenicek/asmk ~/asmk && git -C ~/asmk checkout 2a96d9c03a841dffdfddabc699a20512dcd09363
.venv-gpu/bin/cython -3 ~/asmk/cython/hamming.pyx -o ~/asmk/cython/hamming.c
uv pip install --python .venv-gpu/bin/python --no-build-isolation ~/asmk
```

Without asmk set `sfm.mast3r_sfm.scene_graph: swin` (sliding window pairs instead of retrieval).
`scikit-sparse` (an InstantSfM dependency that needs the SuiteSparse headers) is not installed:
`src/sfm/instantsfm.py` registers a scipy `splu` shim for the single call InstantSfM makes.

Weights download on the first run (or ahead of time, ~7.5 GB in total):

```bash
.venv-gpu/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('facebook/VGGT-1B', allow_patterns=['config.json', 'model.safetensors'])
snapshot_download('naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric')
from src.sfm import mast3r_sfm; mast3r_sfm.download_retrieval(mast3r_sfm.DEFAULTS)"
```

Check: every module has a self-test, `.venv-gpu/bin/python -m src.sfm.vggt` (and `gpu`, `mast3r_sfm`,
`instantsfm`, `colmap`, `pycolmap_backend`) prints `OK`.

## Run

```bash
# one YAML per technique, all equal to parede_sala.yaml except sfm.technique; outputs in parede_sala_GPU/
for t in vggt mast3r_sfm instantsfm glomap; do .venv-gpu/bin/python main.py parede_sala_GPU_$t.yaml; done
```

## Key reference

| key | default | meaning |
|---|---|---|
| `sfm.technique` | `colmap` | `colmap` \| `glomap` \| `instantsfm` \| `mast3r_sfm` \| `vggt` |
| `sfm.ba_global_frames_ratio` | `null` | incremental mapper: global BA when the model grew by this ratio (COLMAP 1.1); 1.3 = fewer global BAs |
| `sfm.instantsfm.max_frames` | `null` | uniform subsample; null = every frame |
| `sfm.instantsfm.num_iteration_bundle_adjustment` / `ba_max_num_iterations` | 3 / 100 | its GPU BA rounds and iterations per round |
| `sfm.instantsfm.skip_view_graph_calibration` / `skip_retriangulation` / `skip_pruning` | false / true / true | InstantSfM stages |
| `sfm.instantsfm.ba` | false | extra pycolmap BA after the retriangulation |
| `sfm.mast3r_sfm.repo_dir` | `~/mast3r` | recursive clone of naver/mast3r |
| `sfm.mast3r_sfm.max_frames` | 200 | uniform subsample; pairs, time and VRAM grow with it |
| `sfm.mast3r_sfm.scene_graph` | `retrieval` | `retrieval` (retrieval-`retrieval_na`-`retrieval_k`) \| `swin` (`window`) |
| `sfm.mast3r_sfm.optim_level` | `refine+depth` | `coarse` (no 2D refinement) \| `refine` \| `refine+depth` |
| `sfm.mast3r_sfm.lr1/niter1/lr2/niter2` | 0.07/300/0.01/300 | sparse global alignment schedule (demo values) |
| `sfm.mast3r_sfm.ba` / `refine_intrinsics` | true / true | pycolmap BA on the triangulated SIFT tracks |
| `sfm.vggt.weights` | `facebook/VGGT-1B` | official weights (or the gated `facebook/VGGT-1B-Commercial`) |
| `sfm.vggt.max_frames` | 80 | one forward pass: 80 frames ~9.9 GB peak on the RTX 5080, 150 runs out of 16 GB |
| `sfm.vggt.image_size` | 518 | long side; the short side keeps the aspect (multiple of 14), no crop or pad |
| `sfm.vggt.dtype` | `auto` | bf16 on sm_80+ |
| `sfm.vggt.ba` / `refine_intrinsics` | true / true | pycolmap BA on the triangulated SIFT tracks |

## Smoke tests (20 frames of parede_sala, every 8th of the first 160, RTX 5080)

| technique | time (s) | VRAM peak (GB) | registered | points | mean reproj (px) | next stages open it |
|---|---|---|---|---|---|---|
| colmap (incremental) | 12.5 | n/a (COLMAP CLI) | 20/20 | 10 615 | 0.71 | yes |
| colmap, `ba_global_frames_ratio: 1.3` | 10.7 | n/a (COLMAP CLI) | 20/20 | 10 632 | 0.71 | yes |
| glomap | 10.8 | n/a (COLMAP CLI) | 20/20 | 10 764 | 0.68 | yes |
| instantsfm | 20.2 | 0.86 | 20/20 | 15 180 | 1.15 | yes |
| mast3r_sfm | 71.1 | 3.58 | 20/20 | 11 489 | 0.69 | yes |
| vggt | 22.2 (forward 1.0) | 6.88 | 20/20 | 15 906 | 0.56 | yes |
| vggt, 80 frames of the whole video | 33.0 (forward 6.2) | 9.86 | 80/80 | 16 425 | 0.55 | yes |

"Next stages open it" = `sfm_convert` reads `sparse/0` (TXT) and writes the numpy arrays. Times include model
loading (VGGT 5 GB, MASt3R 2.7 GB). On 20 frames the speed gap does not show; it grows with the frame count
(the incremental mapper is superlinear).

## GPU bundle adjustment in COLMAP itself (Ceres 2.3 + cuDSS), not done

Rebuilding COLMAP with a CUDA Ceres takes 1-2 h and can break the `colmap` micromamba env. Steps, when wanted:

1. CUDA toolkit >= 12.8 and cuDSS (NVIDIA, `libcudss` 0.5+) installed under a prefix, e.g. `~/opt/cuda-12.8`.
2. Ceres Solver 2.3 from source: `cmake -DUSE_CUDA=ON -DCUDSS_DIR=... -DCMAKE_CUDA_ARCHITECTURES=120`, install to `~/opt/ceres`.
3. COLMAP 4.x from source against that Ceres: `cmake -DCUDA_ENABLED=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DCeres_DIR=~/opt/ceres/lib/cmake/Ceres`.
4. Point `sfm.colmap_bin` to the new binary and set `sfm.ba_use_gpu: true` (Mapper.ba_use_gpu / GlobalMapper.ba_ceres_use_gpu).
5. Expected gain ~1.5-3x in the BA share of the mapper time.

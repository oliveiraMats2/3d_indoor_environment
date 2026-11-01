# 3D Indoor Environment

3D reconstruction of an indoor environment from frames extracted from a phone video: frames -> features ->
two-view geometry -> SfM -> metric scale -> cleanup -> planes and floor plan -> dense MVS (and optional
Gaussian Splatting) -> Poisson mesh -> hole closing -> exports (COLMAP, MeshLab, web) -> evaluation -> videos ->
report. One YAML drives the whole run.

Two flows share the same code, chosen by the top-level key `flow` of the config (`src/flows.py`):

| | `flow: cpu` (default) | `flow: gpu` (alternative) |
|---|---|---|
| SfM | COLMAP SIFT + incremental mapper, Ceres BA on the CPU | COLMAP SIFT + GLOMAP `global_mapper`, global positioning + BA on the GPU (`t2/colmap-cudss:4.2.0` container) |
| ablations | yes: Stage 5 builds `ablation.csv` from `evaluation.ablation_runs` | no: not run and not required (`ablation_runs` is ignored) |
| environment | `.venv` (`pyproject.toml`) | `.venv-gpu` (`requirements*.txt`, torch cu128) + Docker with the NVIDIA container toolkit |
| example configs | `config.yaml`, `configs/default.yaml`, `configs/kitchen_my_home_cpu.yaml`, `configs/experiments/*.yaml` | `config_gpu.yaml`, `config_gpu_keep.yaml`, `configs/kitchen_my_home_gpu.yaml` |

Detection, matching, two-view geometry, densification, mesh, scale and the evaluation metrics are the same in
both flows. Keys written in the YAML always win over the flow preset (the preset only fills missing `sfm` keys).

## Choosing the flow

```yaml
# config.yaml (read by `python3 main.py` without arguments)
flow: cpu        # cpu (default, also when the key is missing or null) | gpu
```

`flow: cpu` fills `sfm.technique: colmap`, `sfm.mapper: incremental`, `sfm.container: null`.
`flow: gpu` fills `sfm.technique: glomap`, `sfm.mapper: global`, `sfm.container: t2/colmap-cudss:4.2.0`,
`sfm.container_steps: [mapper]`, `sfm.container_ba_use_gpu: true`.

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
[GPU SfM techniques](#gpu-sfm-techniques) and [Gaussian Splatting](#gaussian-splatting-optional).

Check: `.venv-gpu/bin/python -c "import torch; print(torch.cuda.is_available())"` prints `True`, and
`.venv-gpu/bin/python -m src.sfm.gpu` prints `OK`.

## Execução rápida

```bash
./run.sh                       # main.py + view.py com config_cozinha.yaml
./run.sh <config>.yaml         # idem, com outro config
./run.sh -h                    # uso
```

O `run.sh` roda o pipeline (`main.py`) e em seguida o visualizador (`view.py`) com o mesmo config.

## Run

CPU flow (default):

```bash
source .venv/bin/activate
python3 main.py                        # config.yaml (flow: cpu)
python3 main.py <config>.yaml          # any other config; same as --config <path>.yaml
python3 view.py <config>.yaml          # interactive Open3D viewer of a finished run of that config
```

CPU ablations: each variant is one full run, then `kitchen_ablation.yaml` (`evaluation.ablation_runs`) reads
their CSVs and writes `ablation.csv`. Stop at the first failure:

```bash
for c in kitchen_abl_seq_loop kitchen_abl_seq_noloop kitchen_abl_exhaustive kitchen_abl_frames_200 \
         kitchen_abl_frames_100 kitchen_ablation; do
  python3 main.py $c.yaml || break
done
```

GPU flow (alternative): set `flow: gpu` in the config (or use a config that already has it) and run with
`.venv-gpu`:

```bash
.venv-gpu/bin/python main.py config_gpu.yaml    # kitchen, GLOMAP + cuDSS container, every densifier
.venv-gpu/bin/python main.py config.yaml        # after changing flow: cpu -> gpu in config.yaml
```

The config is a path or a bare name, looked up in the current directory, then in the repo root, `configs/`
and `configs/experiments/` (`.yaml` may be omitted). So `python3 main.py default.yaml`,
`python3 main.py blur_keep.yaml` and `python3 main.py configs/video_image.yaml` all work, from any directory.
`main.py` then changes to the repo root, so every relative path inside the YAML starts there.

## Pipeline

`main.py` is the only entry point. Stages, in the order they run (each one is skipped when its
`<block>.enabled` is false; optional tools missing -> WARNING, never a crash):

1. **extraction** (`src/extrair_quadros.py`, `src/extraction/`): one decode of the video; every `stride`-th
   frame, or a greedy 60-80% overlap selection; blur and motion-blur drop; longest side capped at `max_side`.
2. **frame_quality** (`src/frame_quality.py`): blur, exposure, contrast and texture checks; then a uniform
   subsample of `detection.max_frames` frames for the 2D stages (COLMAP still uses every frame).
3. **detection** (`src/detectores.py`): SIFT, ORB and AKAZE keypoints and descriptors.
4. **matching** (`src/matching`): BF/FLANN with the Lowe filter, metrics and figures.
5. **geometry** (`src/epipolar`): F/E with RANSAC and inlier masks.
6. **two_view** (`src/epipolar/two_view.py`): relative pose, triangulation and epipolar lines per pair.
7. **sfm** (`src/sfm/colmap.py`, `convert.py`): COLMAP features, matching and mapper; the model becomes the
   COLMAP export and numpy arrays.
8. **scale** (`src/sfm/scale.py`): cm per model unit from ArUco markers or manual pairs (unknown -> model units).
9. **cleanup** of the sparse cloud (`src/cloud/clean.py`): outlier removal, voxel, crop.
10. **planes** (`src/cloud/planes.py`) and **floorplan** (`src/cloud/floorplan.py`): floor, ceiling, walls,
    wall angles, top-view floor plan and room dimensions.
11. **dense** (`src/dense`): COLMAP patch-match stereo and/or OpenMVS, then cleanup of each dense cloud.
12. **dense_compare** (`src/dense/compare.py`): density, completeness and chamfer between methods.
13. **gaussian_splatting** (`src/gs/gaussian_splatting.py`, optional, `gaussian_splatting.enabled`): Inria 3D
    Gaussian Splatting on the undistorted SfM model; every 8th image is held out and the test views are scored
    (PSNR, SSIM, LPIPS), a novel-view synthesis benchmark next to the sparse/dense reconstruction.
14. **poisson** (`src/mesh/poisson.py`): Poisson surface, variants `pre` (cleaned cloud) and `raw` (no cleanup).
15. **close_holes** (`src/mesh/close_holes.py`): MeshLab Close Holes on every Poisson variant, with (`pre`)
    and without (`raw`) mesh repair first.
16. **liepa** (`src/mesh/liepa.py`, optional, `liepa.enabled`): Liepa (2003) hole filling — minimum-weight
    triangulation (dihedral angle, area), density refinement and thin-plate fairing of each hole up to
    `max_hole_edges`; writes `exports/meshlab/liepa_scene.ply`. Smooth patches ("bubble" on big holes).
17. **mesh** (`src/mesh/mesh.py`): the chosen mesh in PLY/OBJ/STL/OFF/GLB plus a scene with cameras and path.
18. **export** (`src/export/meshlab.py`): MeshLab project with every cloud and mesh as layers, filter script.
19. **evaluation** (`src/evaluation`): reprojection error, metric error against tape measures, loop drift,
    smooth vs textured walls; ablation table across runs only in `flow: cpu` (`src/evaluation/__init__.py`
    orchestrates Stage 5 there; `flow: gpu` runs the same metrics stage by stage, without ablation).
20. **viewer** (`src/viewer.py`): orbit video of the chosen cloud, recorded in a child process.
21. **videos** (`src/videos.py`): top-view trajectory, kept vs discarded frames, `videos.csv`.
22. **report** (`src/relatorio.py`): `report.md` built only from the saved files.

### Stage 5: evaluation

When evaluation is enabled, Stage 5 writes numerical arrays under
`results/<id>/evaluation/` and the following tables when their inputs are available:
`artifacts/<id>/evaluation.csv`, `metric_error.csv`, `loop_drift.csv` and `ablation.csv`.
The last pipeline stage always consolidates the available tables and figure paths in
`artifacts/<id>/report.md`.

| Key | Meaning |
|---|---|
| `report` | `max_figures` and `max_rows` for the generated `artifacts/<id>/report.md` |
| `evaluation.enabled` | run Stage 5 before the report; uses global SfM when present and otherwise evaluates the current two-view clouds |
| `evaluation.measured_cm` | measured `room_width`, `room_length` and `ceiling_height` values used by metric error |
| `evaluation.loop_closed` | whether the capture really returned to its starting point |
| `evaluation.ablation_runs` | run ids whose evaluation CSVs are compared in `ablation.csv` (`flow: cpu` only) |

## Adding a new video

1. Put the video at the repo root as `video_<name>.MOV` (the `/video_*` workspaces are git-ignored).
2. Copy the default config: `cp configs/default.yaml configs/<config>.yaml` (e.g. `video_image.MOV` ->
   `configs/video_image.yaml`), set `extraction.video: video_<name>.MOV` and
   `geometry.intrinsics: {fx: null, fy: null, cx: null, cy: null}` (the default intrinsics belong to
   `video_preview.MOV`), and `evaluation.loop_closed: null` unless you know the path is a closed loop.
3. `python3 main.py <config>.yaml`. Outputs go to `video_<name>/`.

`configs/video_image.yaml` is such a copy for `video_image.MOV` (1250 frames, 41.7 s, 30 fps, portrait
2160x3840 -> frames of 1080x1920). It lives in `configs/`, not in `configs/experiments/`: it is another
dataset, not an ablation axis.

## Outputs

`<ws>` = `extraction.video` without the extension (e.g. `video_preview`), `<id>` = `run.id` (null = the
config file name), `<tag>` = first 8 hex chars of the SHA-1 of the `extraction` block (without `rerun`,
`tag`, `output_dir`): runs with the same extraction parameters share the same frames.

```
video_<name>.MOV
video_<name>/
  frames/<tag>/            img_0000.png ..., selecao.csv, extraction_metrics.csv, mvs_exclude.txt,
                           discarded/{blur,motion}/ (shared by every config with the same tag)
  results/<id>/            run.log, config.resolved.yaml, K.npy
    detection/ matching/ epipolar/ figures/       2D stages, epipolar lines, floorplan.png
    sfm/ scale/ clean/<source>/ planes/ floorplan/ dense/<method>/ mesh/{poisson,close_holes,liepa}/ evaluation/
    gaussian_splatting/    data/{images,sparse/0} (undistorted), model/ (Inria: point_cloud/iteration_<N>/,
                           test/ours_<N>/{renders,gt}, results.json, per_view.json), gs.log
    exports/
      colmap/              COLMAP workspace: images/, database.db, sparse/0/, sparse.ply, project.ini, colmap.log
      meshlab/             project.mlp, process.mlx, sparse.ply, fused_<method>.ply, cloud_clean_<source>.ply,
                           poisson_<v>.ply, close_holes_<pv>_<cv>.ply, liepa_scene.ply,
                           mesh.{ply,obj,stl,off}, scene.obj
      web/                 scene.glb, mesh.glb
      gaussian_splatting/  point_cloud.ply (trained Gaussians)
    videos/                navigation_<cloud>_<s>s.mp4, trajectory_topdown.mp4, frames_kept_vs_discarded.mp4
  artifacts/<id>/          one CSV per stage (extraction_metrics, frame_quality_metrics, sfm_metrics, scale,
                           cleanup, planes, floorplan, dense_metrics, gaussian_splatting, poisson, close_holes,
                           liepa, mesh, evaluation, videos, ablation, ...) and report.md
```

- `extraction.rerun: false` reuses the frames of `<tag>` when `selecao.csv` exists; `true` extracts again.
- `run.cache: true` reuses detection and the stage outputs that support it; `run.overwrite: true` deletes
  `results/<id>/` first; with both false an existing run is never touched.

## Opening the outputs

- **COLMAP**: `cd <ws>/results/<id>/exports/colmap && colmap gui --project_path project.ini` (paths in
  `project.ini` are relative, images are hardlinks or copies, so the folder can be moved or zipped). Or open
  `colmap gui` and `File > Import model` on `sparse/0`.
- **MeshLab 2025.07**: `File > Open Project...` -> `exports/meshlab/project.mlp`. Layers: sparse cloud, fused
  dense clouds, cleaned clouds, Poisson `pre`/`raw`, every Close Holes variant and the final mesh. To redo the
  MeshLab processing on a cloud: `Filters > Show current filter script > Load Script` -> `process.mlx`.
- **Web / viewers**: `exports/web/scene.glb` (cloud, camera frustums and trajectory) and `mesh.glb`; any glTF
  viewer (browser glTF viewers, Blender, three.js).
- **Gaussian Splatting**: `exports/gaussian_splatting/point_cloud.ply` in SuperSplat (https://superspl.at/editor,
  drag and drop) or SIBR (`SIBR_gaussianViewer_app -m <ws>/results/<id>/gaussian_splatting/model`). The scores are
  in `artifacts/<id>/gaussian_splatting.csv` (one row per run id): `n_images, n_train, n_test, iterations,
  resolution, max_image_size, n_gaussians, psnr, ssim, lpips, t_undistort_s, t_train_s, t_render_s, t_metrics_s, t_s`.

| | OBJ (`exports/meshlab/*.obj`) | GLB (`exports/web/*.glb`) |
|---|---|---|
| format | text, vertices/faces (+ per-vertex colour) | binary glTF 2.0, one self-contained file |
| size / load | large, slow to parse | compact, fast |
| point clouds | not a standard OBJ element | yes (points primitive), used by `scene.glb` |
| best for | MeshLab, CAD and mesh tools | web and 3D viewers, sharing |

## Configs

- One YAML = one experiment. `configs/default.yaml` holds every key with its default and a comment; unknown
  keys are rejected (pydantic `extra="forbid"`), so typos fail before anything runs.
- `configs/experiments/*.yaml` are FULL copies of `default.yaml` with exactly one line changed, marked
  `# axis of this copy` (no inheritance: each file reads on its own). After changing `default.yaml`,
  regenerate the copies so they stay one line apart:

| file | axis | variant |
|---|---|---|
| `lowe_070.yaml` | `filters.lowe_ratio` | 0.70 |
| `sfm_exhaustive.yaml` | `sfm.matcher` | exhaustive |
| `sfm_sequential_loop.yaml` | `sfm.loop_detection` | true (vocab tree) |
| `frames_50.yaml` | `extraction.max_frames` | 50 |
| `blur_keep.yaml` | `extraction.blur.enabled` | false (no blur drop) |
| `dense_openmvs.yaml` | `dense.methods` | [openmvs] |
| `ablation.yaml` | `evaluation.ablation_runs` | every run above + default: builds `ablation.csv` |

- Default intrinsics (`geometry.intrinsics: {fx: 1476.32, fy: 1471.86, cx: 960, cy: 540}`) are the COLMAP
  OPENCV calibration of `video_preview.MOV` at 1920x1080 landscape (the model in `colmap_gpu/`). They are used
  only by the two-view stages and by `sfm.intrinsics: fixed`; COLMAP estimates its own by default. `main.py`
  warns when `cx, cy` do not match the frame size. For another video set them to null.

### Blur decision

Blurred frames have fewer and less precise keypoints: in SfM they raise the reprojection error and in MVS
they add noisy depth. The extraction therefore drops them by default (`extraction.blur.enabled: true`):
`coef_min` drops a frame whose Laplacian variance is below that fraction of the rolling median of its
neighbours, `motion_ratio_max` drops directional (motion) blur; copies go to `discarded/{blur,motion}/`.
`keep_for_sfm_only: true` keeps the flagged frames for SfM (more views and overlap) but lists them in
`mvs_exclude.txt`, so patch-match and OpenMVS ignore them. `blur_keep.yaml` turns the filter off to measure
the effect (compare its reprojection and dense metrics with `default` in the ablation).

### GPU

| what | GPU when | keys |
|---|---|---|
| COLMAP SIFT extraction and matching | COLMAP built with CUDA and `nvidia-smi` sees a GPU | `sfm.use_gpu: auto\|true\|false`, `sfm.gpu_index` |
| COLMAP bundle adjustment (Ceres CUDA) | also `sfm.ba_use_gpu: true` | `sfm.ba_use_gpu` |
| COLMAP patch-match stereo | always CUDA (else skipped) | `dense.gpu_index`, `dense.max_image_size` |
| Open3D cleanup | Open3D CUDA build | `cleanup.device: auto\|CPU:0\|CUDA:0` |
| viewer orbit rendering | OpenGL / EGL | `viewer.offscreen` |
| Gaussian Splatting train/render/metrics | always CUDA (else skipped) | `gaussian_splatting.gpu_index`, `gaussian_splatting.data_device` |
| SfM with InstantSfM, MASt3R-SfM or VGGT | always CUDA, `.venv-gpu` (see [GPU SfM techniques](#gpu-sfm-techniques)) | `sfm.technique`, `sfm.instantsfm`, `sfm.mast3r_sfm`, `sfm.vggt` |

Poisson (`poisson.threads`), Close Holes and OpenMVS (`dense.openmvs.max_threads`) run on the CPU.

### YAML reference

Generated from `src/constants.py` and the comments of `configs/default.yaml`; blocks in YAML order.

| key | default | meaning |
|---|---|---|
| `flow` | `cpu` | cpu (COLMAP incremental, ablations) \| gpu (GLOMAP + cuDSS container, no ablations); src/flows.py |
| **`run`** | | id and output folders of this experiment |
| `run.id` | `null` | null = uses THIS FILE'S NAME; outputs go to `<ws>/results/<id>/` |
| `run.output_root` | `null` | null = `<ws>/results` |
| `run.artifacts_root` | `null` | null = `<ws>/artifacts` (CSVs and report.md) |
| `run.seed` | `42` | seed of python, numpy and OpenCV RNGs |
| `run.log_level` | `INFO` | DEBUG \| INFO \| WARNING |
| `run.overwrite` | `false` | true = delete `<ws>/results/<id>/` before running |
| `run.cache` | `true` | true = reuse detection (and stage outputs that support it) of an existing run |
| **`extraction`** | | src/extrair_quadros.py |
| `extraction.rerun` | `false` | true = extract again (old img_* and selecao.csv in output_dir are deleted); false = reuse frames if selecao.csv exists |
| `extraction.video` | `video_preview.MOV` | input video at the repo root; `<ws>` = its name without extension |
| `extraction.output_dir` | `null` | null = `<ws>/frames/<tag>`; shared by every experiment with the same tag |
| `extraction.tag` | `null` | null = SHA-1[:8] of this block without rerun/tag/output_dir (same params -> same frames) |
| `extraction.selection` | `all` | all = every stride-th frame, dropped only by extraction.blur (overlap/sharp_percentile ignored) \| overlap = greedy overlap selection |
| `extraction.min_frames` | `100` | all: only warns below it \| overlap: lower bound of the target adjustment |
| `extraction.max_frames` | `null` | null = no cap; all: uniform subsample down to it \| overlap: upper bound of the adjustment |
| `extraction.overlap` | `0.7` | initial overlap target (0.60 to 0.80); overlap mode only |
| `extraction.overlap_min` | `0.6` | lower bound of the overlap target (overlap mode) |
| `extraction.overlap_max` | `0.8` | upper bound of the overlap target (overlap mode) |
| `extraction.overlap_step` | `0.025` | target step per adjustment iteration (overlap mode) |
| `extraction.max_adjust_iters` | `12` | overlap mode only |
| `extraction.stride` | `1` | analyze 1 of every N video frames (1 = every frame) |
| `extraction.analysis_width` | `640` | width of the grey copy used for blur/overlap analysis |
| `extraction.nfeatures` | `1000` | ORB features used only to estimate overlap |
| `extraction.lowe_ratio` | `0.75` | Lowe ratio of the overlap estimate |
| `extraction.min_good_matches` | `15` | min ORB matches to estimate overlap |
| `extraction.ransac_threshold_px` | `3.0` | homography RANSAC threshold of the overlap estimate |
| `extraction.min_inliers` | `12` | min homography inliers of the overlap estimate |
| `extraction.sharp_percentile` | `25` | overlap mode only: drop the blurriest 25% of sampled frames before selection |
| `extraction.max_side` | `1920` | longest side of saved frames (landscape or portrait: 2160x3840 -> 1080x1920) |
| `extraction.image_format` | `png` | png = lossless; jpg allowed |
| `extraction.jpeg_quality` | `95` | used only with image_format: jpg |
| **`extraction.blur`** | | src/extraction/blur.py |
| `extraction.blur.enabled` | `true` | false = no blur/motion drop (blur_keep.yaml) |
| `extraction.blur.coef_min` | `0.5` | drop if lapvar / rolling median < coef_min |
| `extraction.blur.motion_ratio_max` | `4.0` | drop if structure-tensor lmax/lmin > this (motion blur) |
| `extraction.blur.keep_for_sfm_only` | `false` | true = keep flagged frames for SfM and list them in mvs_exclude.txt |
| `extraction.blur.window` | `9` | neighbouring sampled frames in the rolling median |
| `extraction.blur.discard_width` | `640` | width of the jpg copies in discarded/{blur,motion}/ |
| `extraction.blur.discard_jpeg_quality` | `90` | quality of the discarded/ copies |
| **`frame_quality`** | | src/frame_quality.py |
| `frame_quality.resize_width` | `640` | width of the analysis copy |
| **`frame_quality.blur`** | | Laplacian variance checks |
| `frame_quality.blur.laplacian_var_min` | `100.0` | drop below this absolute Laplacian variance |
| `frame_quality.blur.relative_to_median` | `0.5` | drop below this fraction of the median Laplacian variance |
| **`frame_quality.exposure`** | | mean brightness and clipping checks |
| `frame_quality.exposure.mean_min` | `40` | drop darker frames (mean grey) |
| `frame_quality.exposure.mean_max` | `215` | drop brighter frames (mean grey) |
| `frame_quality.exposure.clipped_max_frac` | `0.05` | fraction of pixels <= 5 or >= 250 |
| **`frame_quality.contrast`** | | grey standard deviation check |
| `frame_quality.contrast.std_min` | `20.0` | drop flatter frames |
| **`frame_quality.texture`** | | FAST keypoint count check |
| `frame_quality.texture.min_keypoints` | `300` | FAST threshold 20 |
| `frame_quality.texture.fast_threshold` | `20` | FAST detector threshold |
| **`detection`** | | src/detectores.py |
| `detection.methods` | `[sift, orb, akaze]` | detectors compared (sift, orb, akaze) |
| `detection.nfeatures` | `5000` | max keypoints per frame |
| `detection.reps` | `30` | timed runs per frame, after warmup |
| `detection.warmup` | `2` | untimed runs discarded before measurement |
| `detection.max_frames` | `100` | uniform subsample of the quality-kept frames fed to detection/matching/geometry/two_view; null = all (COLMAP always uses every frame) |
| **`report`** | | src/relatorio.py |
| `report.max_figures` | `5` | keypoint/registration figures per detector |
| `report.max_rows` | `20` | rows per table in report.md |
| **`filters`** | | Lowe filter of the 2D matching stage |
| `filters.lowe_ratio` | `0.75` | lower = cleaner and fewer matches; 0.7 to 0.8 is usual (also used by detection) |
| `filters.min_matches` | `10` | pairs below it are success=False |
| **`matching`** | | src/matching |
| `matching.enabled` | `true` | false = skip the 2D matching study |
| `matching.name` | `brute_force` | brute_force \| flann; exact, FLANN mixes index error with detector quality |
| `matching.strategy` | `sequential` | sequential (consecutive frames) \| all_pairs (ablation) |
| `matching.cross_check` | `false` | must stay false: knnMatch(k=2) for Lowe excludes crossCheck |
| `matching.reps` | `30` | timed runs per pair, after warmup |
| `matching.warmup` | `2` | untimed runs discarded before measurement |
| **`matching.flann`** | | FLANN parameters (name: flann) |
| `matching.flann.checks` | `50` | FLANN search checks |
| `matching.flann.float_index` | `{algorithm: 1, trees: 5}` | KDTREE |
| `matching.flann.binary_index` | `{algorithm: 6, key_size: 12, multi_probe_le...` | LSH |
| **`geometry`** | | src/epipolar (F/E and RANSAC) |
| `geometry.enabled` | `true` | false = skip F/E and the two-view stage |
| **`geometry.ransac`** | | OpenCV robust estimator of F/E |
| `geometry.ransac.method` | `USAC_MAGSAC` | less sensitive to the hard threshold than RANSAC |
| `geometry.ransac.threshold_px` | `1.0` | inlier threshold (px) |
| `geometry.ransac.confidence` | `0.999` | RANSAC confidence |
| `geometry.ransac.max_iters` | `5000` | RANSAC iterations |
| `geometry.intrinsics` | `{cx: 960, cy: 540, fx: 1476.32, fy: 1471.86}` | fx, fy, cx, cy of the two-view stages (and sfm.intrinsics: fixed); default = COLMAP OPENCV calibration of video_preview (colmap_gpu/, 1920x1080); null fx/fy -> max(w, h), null cx/cy -> w/2, h/2 |
| **`visualize`** | | figures of the 2D stages |
| **`visualize.matches`** | | match figures |
| `visualize.matches.max_lines` | `100` | matches drawn per pair |
| `visualize.matches.dpi` | `150` | figure dpi |
| `visualize.matches.format` | `png` | image format of the figures |
| **`visualize.epilines`** | | epipolar line figures (two_view) |
| `visualize.epilines.n_lines` | `20` | epipolar lines drawn per pair |
| `visualize.epilines.dpi` | `150` | figure dpi |
| **`sfm`** | | src/sfm/colmap.py + convert.py (COLMAP >= 4.0 CLI; older -> WARNING, stage skipped) |
| `sfm.enabled` | `true` | false = no COLMAP (scale and every 3D stage then lack input) |
| `sfm.rerun` | `false` | false + existing sparse/0/images.bin -> reuse the model |
| `sfm.colmap_bin` | `colmap` | not found -> WARNING, stage skipped |
| `sfm.camera_model` | `OPENCV` | fx,fy,cx,cy,k1,k2,p1,p2 |
| `sfm.single_camera` | `true` | one shared camera (same phone/zoom) |
| `sfm.intrinsics` | `estimate` | estimate: COLMAP estimates \| fixed: geometry.intrinsics + fixed_dist, not refined in BA |
| `sfm.fixed_dist` | `[0, 0, 0, 0]` | k1,k2,p1,p2 used only with intrinsics: fixed |
| `sfm.matcher` | `sequential` | sequential \| exhaustive |
| `sfm.sequential_overlap` | `10` | neighbours matched per frame (sequential) |
| `sfm.loop_detection` | `false` | true needs vocab_tree |
| `sfm.vocab_tree` | `https://github.com/colmap/colmap/releases/d...` | path or COLMAP url;name;sha256 (downloaded and cached by COLMAP), used only with loop_detection |
| `sfm.mapper` | `incremental` | incremental (mapper) \| global (global_mapper, GLOMAP) |
| `sfm.use_gpu` | `auto` | auto = GPU when COLMAP is built with CUDA and nvidia-smi sees a GPU \| true \| false (SIFT + matching) |
| `sfm.gpu_index` | `'0'` | CUDA device(s) of SIFT/matching |
| `sfm.ba_use_gpu` | `false` | BA on Ceres CUDA |
| `sfm.max_image_size` | `1920` | max side for SIFT extraction |
| `sfm.max_num_features` | `8192` | max SIFT features per image |
| `sfm.min_num_matches` | `15` | min matches per pair for the mapper |
| `sfm.multiple_models` | `true` | allow several submodels; the largest goes to sparse/0 |
| `sfm.ba_global_max_num_iterations` | `null` | null = COLMAP default (incremental only) |
| `sfm.ba_global_max_refinements` | `null` | null = COLMAP default (incremental only) |
| `sfm.project_quality` | `high` | project_generator --quality: low\|medium\|high\|extreme |
| `sfm.link_images` | `auto` | exports/colmap/images/: auto = hardlink, copy if refused \| hardlink \| symlink (relative) \| copy |
| `sfm.backend` | `auto` | auto = COLMAP CLI when colmap_bin is found (primary), else pycolmap (uv sync --extra cpu\|cuda) \| cli \| pycolmap |
| **`scale`** | | src/sfm/scale.py |
| `scale.enabled` | `true` | metric scale from ArUco markers or manual pairs |
| `scale.method` | `aruco` | aruco \| manual |
| `scale.aruco_dict` | `DICT_4X4_50` | cv2.aruco.DICT_* name |
| `scale.marker_cm` | `null` | real marker side in cm; null -> scale unknown (unit=model) |
| `scale.min_parallax_deg` | `1.0` | frame pairs with less parallax are ignored when triangulating the marker |
| `scale.manual_pairs` | `[]` | [{length_cm, p: [{image,x,y},{image,x,y}], q: [{image,x,y},{image,x,y}]}] |
| `scale.provisional_extent_cm` | `500` | only used to turn *_cm thresholds into model units when unit=model |
| **`cleanup`** | | src/cloud/clean.py (sparse and each dense method) |
| `cleanup.enabled` | `true` | outlier removal, voxel and crop of each cloud |
| `cleanup.voxel_cm` | `1.0` | voxel downsampling side |
| `cleanup.nb_neighbors` | `20` | statistical outlier removal neighbours |
| `cleanup.std_ratio` | `2.0` | drop points with mean distance > mean + std_ratio * std |
| `cleanup.crop_percentiles` | `[1, 99]` | axis-aligned box removing extreme tails (floaters) |
| `cleanup.device` | `auto` | auto = CUDA:0 when Open3D sees a CUDA GPU, else CPU:0 \| CPU:0 \| CUDA:0 |
| **`planes`** | | src/cloud/planes.py |
| `planes.enabled` | `true` | floor, ceiling and walls by RANSAC |
| `planes.source` | `sparse` | sparse \| patch_match \| openmvs -> `clean/<source>/points.npy` |
| `planes.distance_cm` | `2.0` | RANSAC inlier threshold |
| `planes.ransac_n` | `3` | points per RANSAC sample |
| `planes.iterations` | `2000` | RANSAC iterations per plane |
| `planes.max_walls` | `4` | max vertical planes kept as walls |
| `planes.min_inlier_frac` | `0.02` | min plane inliers / total points (also stop criterion) |
| `planes.angle_tol_deg` | `15` | tolerance for horizontal / vertical planes |
| `planes.max_planes` | `10` | max RANSAC rounds |
| **`floorplan`** | | src/cloud/floorplan.py |
| `floorplan.enabled` | `true` | top-view occupancy figure and room dimensions |
| `floorplan.cell_cm` | `5.0` | grid cell side |
| `floorplan.dpi` | `150` | figure dpi |
| `floorplan.fig_inches` | `8.0` | longest side of the figure in inches |
| `floorplan.extent_percentiles` | `[1, 99]` | robust percentiles for the room rectangle sides |
| **`dense`** | | src/dense (patch_match.py, openmvs.py) |
| `dense.enabled` | `true` | dense MVS (patch_match needs COLMAP with CUDA; openmvs needs OpenMVS) |
| `dense.rerun` | `false` | false: reuse `dense/<method>/fused.ply` |
| `dense.methods` | `[patch_match, openmvs]` | openmvs is skipped with a WARNING when not installed |
| `dense.max_image_size` | `1600` | longest side (px); VRAM and time grow ~side^2; 640 = quick test |
| `dense.gpu_index` | `'0'` | "-1" = all GPUs |
| **`dense.patch_match`** | | COLMAP patch_match_stereo |
| `dense.patch_match.num_iterations` | `5` | propagation iterations |
| `dense.patch_match.window_radius` | `5` | NCC window (2r+1)^2 |
| `dense.patch_match.num_samples` | `15` | Monte Carlo source-view samples per pixel |
| `dense.patch_match.num_src_images` | `20` | source images per reference |
| `dense.patch_match.geom_consistency` | `true` | photometric + geometric pass (~2x time, much less noise) |
| `dense.patch_match.cache_size` | `32` | GB of CPU RAM cache |
| **`dense.fusion`** | | COLMAP stereo_fusion |
| `dense.fusion.min_num_pixels` | `5` | min consistent pixels per point |
| `dense.fusion.max_reproj_error` | `2.0` | px |
| `dense.fusion.max_depth_error` | `0.01` | relative |
| `dense.fusion.cache_size` | `32` | GB |
| **`dense.openmvs`** | | OpenMVS DensifyPointCloud |
| `dense.openmvs.bin_dir` | `null` | folder with InterfaceCOLMAP/DensifyPointCloud; null = PATH |
| `dense.openmvs.resolution_level` | `1` | 0 = full resolution, 1 = half, ... |
| `dense.openmvs.number_views` | `5` | neighbour views per depth map (0 = all) |
| `dense.openmvs.max_threads` | `0` | 0 = all cores |
| **`dense_compare`** | | src/dense/compare.py |
| `dense_compare.enabled` | `true` | density, completeness and chamfer between dense methods |
| **`gaussian_splatting`** | | src/gs/gaussian_splatting.py (Inria 3DGS, novel-view synthesis benchmark; optional, CUDA only) |
| `gaussian_splatting.enabled` | `false` | true -> exports/gaussian_splatting/point_cloud.ply + PSNR/SSIM/LPIPS; missing tool/GPU -> WARNING, skipped |
| `gaussian_splatting.rerun` | `false` | false: reuse gaussian_splatting/data and model/point_cloud/iteration_<iterations> |
| `gaussian_splatting.repo_dir` | `~/gaussian-splatting` | clone of graphdeco-inria/gaussian-splatting (train.py, render.py, metrics.py) |
| `gaussian_splatting.python` | `~/gaussian-splatting/.venv/bin/python` | interpreter with torch + the built submodules |
| `gaussian_splatting.iterations` | `30000` | training iterations (Inria default; 7000 = quick test) |
| `gaussian_splatting.eval` | `true` | hold out every 8th image (llffhold) as test views for PSNR/SSIM/LPIPS |
| `gaussian_splatting.resolution` | `-1` | Inria -r: -1 = auto (at most 1600 px wide) \| 1, 2, 4, 8 = downscale factor \| else target width |
| `gaussian_splatting.max_image_size` | `1600` | image_undistorter --max_image_size (longest side of the training images) |
| `gaussian_splatting.data_device` | `cpu` | cpu \| cuda; cpu: thousands of frames do not fit in 16 GB VRAM |
| `gaussian_splatting.sh_degree` | `3` | spherical harmonics degree of the Gaussian colours |
| `gaussian_splatting.gpu_index` | `'0'` | CUDA_VISIBLE_DEVICES of train/render/metrics |
| `gaussian_splatting.timeout_s` | `null` | null = no limit; else kill each command after this many seconds |
| **`poisson`** | | src/mesh/poisson.py (Open3D) |
| `poisson.enabled` | `true` | Poisson surface reconstruction |
| `poisson.source` | `dense` | dense -> first dense.methods with output, else sparse (WARNING) \| sparse |
| `poisson.variants` | `[pre, raw]` | pre = cleaned cloud (cleanup stage) \| raw = same cloud without cleanup |
| `poisson.depth` | `10` | octree depth (higher = finer, slower) |
| `poisson.density_quantile` | `0.05` | drop vertices with Poisson density below this quantile |
| `poisson.normal_knn` | `30` | neighbours for normal estimation |
| `poisson.threads` | `8` | Poisson threads (-1 = all cores; much slower on many-core machines) |
| **`close_holes`** | | src/mesh/close_holes.py (pymeshlab, MeshLab "Close Holes"; no pymeshlab -> WARNING, skipped) |
| `close_holes.enabled` | `true` | MeshLab Close Holes on every Poisson variant |
| `close_holes.variants` | `[pre, raw]` | pre = repair (dedupe, non-manifold, small pieces) then close \| raw = close directly |
| `close_holes.max_hole_size` | `30` | max hole size in boundary edges (MeshLab maxholesize) |
| `close_holes.min_component_frac` | `0.01` | pre: remove isolated pieces smaller than this fraction of the bbox diagonal |
| `close_holes.self_intersection` | `true` | forbid patches that self-intersect (MeshLab selfintersection) |
| **`liepa`** | | src/mesh/liepa.py (Liepa 2003 hole filling: triangulate + refine + fair; optional) |
| `liepa.enabled` | `false` | true -> exports/meshlab/liepa_scene.ply (+ mesh/liepa/); false -> stage skipped |
| `liepa.input` | `poisson` | poisson (pre) \| close_holes (pre_pre): mesh whose holes are filled |
| `liepa.max_hole_edges` | `200` | holes with more boundary edges are left open (keeps the scene border open) |
| `liepa.density` | `1.414...` | refinement alpha (Liepa sqrt(2)); higher = denser patch |
| `liepa.fairing` | `2` | 2 = thin plate (bi-Laplacian, paper) \| 1 = membrane (Laplacian) \| 0 = flat, no fairing |
| `liepa.max_refine_iters` | `30` | cap on the split/relax rounds per hole |
| **`mesh`** | | src/mesh/mesh.py (formats + scene of ONE chosen mesh) |
| `mesh.enabled` | `true` | export the chosen mesh and the scene (cameras + trajectory) |
| `mesh.input` | `close_holes` | close_holes (pre_pre) \| poisson (pre) \| liepa: mesh exported to PLY/OBJ/STL/OFF/GLB |
| `mesh.frustum_cm` | `20` | camera pyramid depth |
| `mesh.frustum_every` | `5` | draw 1 camera every N |
| `mesh.trajectory_cm` | `1.0` | trajectory tube diameter |
| **`export`** | | src/export/meshlab.py (MeshLab project + filter script) |
| `export.enabled` | `true` | MeshLab project.mlp + process.mlx |
| `export.normals_k` | `10` | neighbours for normals |
| `export.normals_smooth_iter` | `0` | normal smoothing iterations (process.mlx) |
| `export.poisson_depth` | `10` | Screened Poisson octree depth |
| `export.poisson_samples_per_node` | `1.5` | 1-5 clean cloud, 15-20 noisy cloud |
| `export.poisson_point_weight` | `4.0` | 0 = classic Poisson |
| `export.isolated_diameter_frac` | `0.1` | remove pieces with diameter < frac * bbox diagonal |
| **`viewer`** | | src/viewer.py: main.py records the orbit; python3 view.py <this file> opens the window |
| `viewer.enabled` | `true` | main.py records `<run_dir>/videos/navigation_<cloud>_<seconds>s.mp4` (child process: a renderer crash only skips the video) |
| `viewer.cloud` | `sparse` | sparse \| patch_match \| openmvs \| mesh |
| `viewer.seconds` | `45` | length of the orbit video |
| `viewer.fps` | `30` | orbit video fps |
| `viewer.offscreen` | `auto` | auto = hidden Open3D window with a display, else OffscreenRenderer (EGL) \| true \| false |
| `viewer.timeout_s` | `1800` | kill the recording process after this many seconds |
| `viewer.width` | `1280` | render width (px) |
| `viewer.height` | `720` | render height (px) |
| `viewer.frustum_cm` | `20` | camera pyramid depth in the viewer |
| `viewer.point_size` | `2.0` | rendered point size (px) |
| `viewer.fov_deg` | `60.0` | vertical FOV of the orbit camera |
| `viewer.orbit_radius_factor` | `1.5` | orbit radius = factor * robust half-diagonal of the scene |
| `viewer.orbit_elevation_deg` | `30.0` | orbit camera elevation |
| `viewer.background` | `[1.0, 1.0, 1.0]` | RGB background in [0, 1] |
| **`videos`** | | src/videos.py |
| `videos.enabled` | `true` | trajectory, kept-vs-discarded videos and videos.csv |
| `videos.topdown_fps` | `10` | fps of trajectory_topdown.mp4 (1 frame per camera) |
| `videos.kept_vs_discarded` | `true` | writes frames_kept_vs_discarded.mp4 |
| `videos.width` | `960` | output width (height proportional, even) |
| `videos.codec` | `auto` | auto = avc1 (H.264) if the OpenCV build can encode it, else mp4v \| any cv2.VideoWriter fourcc |
| **`evaluation`** | | src/evaluation |
| `evaluation.enabled` | `true` | reprojection, metric error, loop drift, hard regions, ablation |
| `evaluation.measured_cm` | `[]` | [{name: room_width\|room_length\|ceiling_height, value_cm}] measured by hand |
| `evaluation.ablation_runs` | `[]` | run ids to compare (configs/experiments/ablation.yaml) |
| `evaluation.loop_closed` | `false` | video_preview.MOV: first/last frames overlap but the path is an arc, not a full loop \| null = unknown \| true |
| `evaluation.hard_regions` | `true` | smooth vs textured wall reprojection (needs planes) |
| `evaluation.texture_radius_px` | `30` | radius of local keypoint density |
| `evaluation.texture_split_quantile` | `0.5` | texture quantile splitting smooth_wall / textured_wall |
| `evaluation.inlier_px` | `2.0` | point is inlier if its mean reprojection error < inlier_px |

## GPU SfM techniques

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

### How each technique ends in COLMAP format

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

### Licenses

| technique | code | weights | commercial use |
|---|---|---|---|
| colmap / glomap | BSD | none | yes |
| instantsfm | CC BY-NC 4.0 (cre185/InstantSfM) | none | no |
| mast3r_sfm | CC BY-NC-SA 4.0 (naver/mast3r) | MASt3R checkpoint + retrieval: CC BY-NC-SA 4.0 (naver) | no |
| vggt | VGGT License (facebookresearch/vggt, commercial allowed, no military use) | `facebook/VGGT-1B`: CC BY-NC 4.0 (default, official). `facebook/VGGT-1B-Commercial`: VGGT License, gated (Hugging Face approval + `hf auth login`) | only with the Commercial weights |

Only the official weights published by the authors are used: `facebook/VGGT-1B` and
`naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric` from the Hugging Face hub, and the MASt3R retrieval
`.pth` + codebook from `https://download.europe.naverlabs.com/ComputerVision/MASt3R/`.

### Extra installs (`.venv-gpu`)

Base GPU install: [Install: GPU flow](#install-gpu-flow-alternative-flow-gpu). MASt3R-SfM also needs:

```bash
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

### Run

```bash
# one YAML per technique, all equal to parede_sala.yaml except sfm.technique; outputs in parede_sala_GPU/
for t in vggt mast3r_sfm instantsfm glomap; do .venv-gpu/bin/python main.py parede_sala_GPU_$t.yaml || break; done
```

### Key reference

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

### Smoke tests (20 frames of parede_sala, every 8th of the first 160, RTX 5080)

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

### The two T2 flows (`flow: cpu | gpu`)

The top-level key `flow` (src/flows.py) picks the SfM path; densification, mesh, scale and the evaluation metrics
are shared, the ablations belong to `flow: cpu`.
Keys written in the YAML win over the preset.

| | `flow: cpu` | `flow: gpu` |
|---|---|---|
| config | `configs/kitchen_my_home_cpu.yaml` | `config_gpu.yaml` (= `configs/kitchen_my_home_gpu.yaml`, same run id) |
| run | `python3 main.py kitchen_my_home_cpu.yaml` (`.venv`) | `.venv-gpu/bin/python main.py config_gpu.yaml` |
| SfM | COLMAP SIFT (GPU) + incremental mapper, Ceres BA on the CPU | COLMAP SIFT (GPU) + GLOMAP `global_mapper`, global positioning + BA on the GPU in the `t2/colmap-cudss:4.2.0` container |
| SIFT | twice: OpenCV SIFT in the 2D comparison (detection/matching/epipolar, with ORB and AKAZE) and COLMAP SIFT for SfM | same |
| dense | `patch_match` only (light) | all `dense.methods` (patch_match, openmvs, plane_sweep, sgm, casmvsnet, mvsnet, patchmatchnet, vggt_omega, mapanything) |
| ablation | `configs/experiments/kitchen_abl_*.yaml` (exhaustive, sequential with/without loop, 100/200 frames) + `kitchen_ablation.yaml` | - (not run, not required) |
| Gaussian Splatting | off | on |

What stays on the CPU in both flows (CPU-only algorithms in COLMAP/Open3D): rotation averaging, track building and
retriangulation of GLOMAP, `stereo_fusion`, Poisson, `close_holes`, `liepa`, `room_fill`, evaluation. Open3D
cleanup uses `cleanup.device: CPU:0` on purpose (all cleanups take ~4 s; the VRAM is kept for Gaussian Splatting).

### GPU bundle adjustment in COLMAP itself (Ceres 2.3 + cuDSS): `docker/colmap-cudss`

A COLMAP build without cuDSS in its Ceres (e.g. the prebuilt packages), so `GlobalMapper.gp_use_gpu` / `ba_ceres_use_gpu` fall back to the
CPU. `docker/colmap-cudss/Dockerfile` builds COLMAP 4.2.0 against Ceres 2.3 with CUDA + cuDSS for sm_120:

```bash
docker build -t t2/colmap-cudss:4.2.0 docker/colmap-cudss   # once (~1 h); needs the NVIDIA container toolkit
```

`flow: gpu` sets `sfm.container: t2/colmap-cudss:4.2.0`, `container_steps: [mapper]` and
`container_ba_use_gpu: true` (src/sfm/container.py runs the step with `--gpus all`, the workspace bind-mounted).
Check in the run log: `colmap_info docker:t2/colmap-cudss:4.2.0: version 4.2, CUDA build True, GPU visible True`,
and no "falling back to CPU" in `exports/colmap/colmap.log`. Kitchen (270 frames): global positioning 10.4 s,
iterative BA 4.3 s.

## Gaussian Splatting (optional)

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

What changes on macOS (no CUDA):

- SfM runs on the CPU (`sfm.use_gpu: auto` resolves to false). Slower, same results.
- COLMAP patch-match stereo needs CUDA, so the `patch_match` dense method is skipped with a WARNING; without
  OpenMVS there is no dense cloud, and Poisson/close_holes/mesh fall back to the sparse cloud
  (`poisson.source: dense` -> sparse with a WARNING). Set `dense.enabled: false` to skip the attempt.
- Recording the viewer orbit (`viewer.enabled`) needs a GUI session (logged-in desktop). Over SSH the
  recording fails in its child process, the video is skipped with a WARNING and the run continues.
  `python3 view.py` also needs the desktop.

## History: first milestone (2D feature study)

Stage list and layout of the first milestone, kept as written. Stages 3 to 6 still run as described;
pose, triangulation and evaluation have since been implemented (see Pipeline).

1. **extraction** (`src/extrair_quadros.py`): sharp frames with 60 to 80% overlap from the video.
2. **frame quality** (`src/frame_quality.py`): blur, exposure, contrast and texture checks.
3. **detection** (`src/detectores.py`): SIFT x ORB x AKAZE keypoints and descriptors as `.npy`.
4. **matching** (`src/matching`): BF/FLANN, Lowe filtering and runtime logs for the configured frame pairs.
5. **geometry** (`src/epipolar`): fundamental/essential matrices and RANSAC inlier masks.
6. **report** (`src/relatorio.py`) and matching metrics/figures: artifacts built only from saved `.npy` files.
7. **pose**, **triangulation** and **evaluation**: pending stages.

Stage 3 part 2 (`config.yaml` partial at the time): stages, in order: extraction, frame_quality, detection,
matching, geometry, two_view, optional evaluation, report.

```
src/
  matching/             matcher, Lowe filter, metrics and figures
  epipolar/             F/E with RANSAC; pose, epipolar lines and triangulation are pending
```

Requires `opencv-contrib-python` (in OpenCV 5, AKAZE lives in `cv2.xfeatures2d`).
Keypoint figures (`figures/keypoints/<detector>/`) show scale as the circle and orientation as
the line; `figures/registration/<detector>/` holds only pairs with >= 8 RANSAC inliers of `E`.

## Chosen detector: AKAZE

Preliminary test (6 frames, 1 s apart, 3 reps):

| | SIFT | ORB | AKAZE |
|---|---|---|---|
| detection (ms) | 644 | 86 | 828 |
| inliers / pair | 49 | 116 | 156 |
| matching (ms) | 29 | 30 | 19 |

AKAZE gives the most inliers per pair, the best spatial coverage and a compact binary
descriptor (61 B vs 512 B for SIFT). Its slower detection does not matter offline,
because reconstruction quality depends on inliers and coverage. ORB is the choice
when time is the constraint. Confirm with `summary.csv` on the full frame set.

## Frame quality analyzer (`analyze_frames.py`)

Standalone diagnosis of a folder of already extracted frames, run before spending hours on SfM/MVS. It does not
run COLMAP, does not reconstruct anything and does not re-extract frames; it only reads the images, measures
them and checks two-view geometry between temporally close pairs. Code in `frame_analyzer/` (independent of
`src/`), OpenCV SIFT required (`opencv-contrib-python`; missing SIFT -> clear error, no fallback detector).

```bash
python3 analyze_frames.py --input video_image/frames/<tag> --output quality_report
python3 analyze_frames.py --input <frames> --output <out> --stride 2 --offsets 1,2,5 --grid 8x8
python3 analyze_frames.py --help        # every threshold, with its default
```

Per frame, at the analysis resolution (`--analysis-max-side`, default 1280): Laplacian variance (sharpness,
judged relative to the sequence median), SIFT keypoint count and density per MP, occupancy of a `ROWSxCOLS`
grid, brightness, dark and saturated fractions. Per pair `(i, i + d)` for each `d` in `--offsets`: Lowe ratio
test, fundamental matrix with RANSAC/MAGSAC, inliers and inlier ratio (homography as a side diagnostic only).
A pair becomes a graph edge only when F is verified and both `--edge-min-inliers` and
`--edge-min-inlier-ratio` hold; consecutive frames never get an edge for free. Each frame ends up `good`,
`review` or `poor` through explicit rules listed in its `flags` column (no combined score).

Outputs (`--output`, default `quality_report/`):

| File | Content |
|---|---|
| `quality_report.html` | self-contained report (inline CSS, figures embedded, opens offline), in Portuguese: executive summary, frames, resolution, sharpness, SIFT, spatial heatmaps, matching, connectivity graph, components, frames to review, visual reference, limitations, acquisition recommendations, full config and timings |
| `README.md` | what each file is, how to read it, and the exact command of that run |
| `frame_quality.csv` | one row per frame: metrics, degree, component, flags, quality |
| `pair_quality.csv` | one row per analyzed pair: matches, inliers, ratio, status, edge accepted |
| `connectivity_edges.csv` | accepted edges and the criterion applied |
| `redundant_pairs.csv` | consecutive pairs with almost no motion (suggestions, nothing is removed) |
| `summary.json` | aggregated numbers, thresholds, warnings, timings, config |
| `connectivity_graph.html` | interactive graph (green = connected, orange = isolated); skipped with `--no-interactive-graph` |
| `grid_counts.npy` | (N, rows, cols) keypoints per grid cell for every frame |
| `*.png` | the charts, heatmaps and static graph (`connectivity_graph.png`) embedded in the report (data only, never photos) |

Reading it: several connected components mean the chain of verified pairs breaks, and the recommendations
list the frame index ranges where it breaks; indices are positions in the analyzed list (after `--limit` and
`--stride`), so use `filename` to find the frame. `good` means no rule fired, not that the reconstruction
will work: Laplacian variance has no universal threshold, many keypoints can come from repetitive texture,
and grid occupancy is coverage of the image plane, not of the room.

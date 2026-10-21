# 3D Indoor Environment

3D reconstruction of an indoor environment from frames extracted from a phone video: frames -> features ->
two-view geometry -> COLMAP SfM -> metric scale -> cleanup -> planes and floor plan -> dense MVS -> Poisson
mesh -> hole closing -> exports (COLMAP, MeshLab, web) -> evaluation -> videos -> report. One YAML drives
the whole run.

## Setup

Requirements:

| | Linux | macOS |
|---|---|---|
| OS | glibc >= 2.35 (Ubuntu 22.04+), x86_64 or aarch64 | 13+ on Apple silicon (arm64), 14+ on Intel |
| Python | 3.10 to 3.12 | 3.10 to 3.12 (Intel: open3d 0.19, the last release with Intel wheels) |
| COLMAP >= 4.0 | with CUDA: conda-forge (`micromamba create -n colmap -c conda-forge colmap`) or built from source | `brew install colmap` (CPU only) |
| pycolmap 4.x (optional) | only without the `colmap` CLI: `uv sync --extra cuda` (NVIDIA, x86_64) or `--extra cpu` | `uv sync --extra cpu` (arm64, macOS 14+) |
| OpenMVS (optional) | build from source (https://github.com/cdcseacave/openMVS/wiki/Building) | same; there is no brew or conda package |
| NVIDIA GPU + driver | optional: GPU SIFT/matching and patch-match stereo | not available |

Create a local Python 3.12 environment and install the dependencies with uv:

```bash
uv python install 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r pyproject.toml
```

Equivalent alternatives: `uv sync` (creates `.venv` from the pinned `uv.lock`), or `uv pip install -e .`
inside the activated `.venv`. The platform-specific pins (open3d 0.19 vs 0.20, pymeshlab wheels) are markers
in `pyproject.toml`, so the same commands work on Linux and macOS. Activate `.venv` before running anything:
`main.py`, `view.py` and `run.sh` call the `python3` on `PATH`.

COLMAP must be on `PATH` (or set `sfm.colmap_bin`). Missing or older than 4.0 -> the SfM stage falls back to
pycolmap when it is installed (`sfm.backend: auto`; same outputs and CSV columns, column `backend` says which
ran), else WARNING and the stage is skipped. pycolmap is optional and not in the base install. Pick ONE extra,
they are mutually exclusive (both provide the `pycolmap` module): `uv sync --extra cuda` (pycolmap-cuda12,
Linux x86_64 + NVIDIA) or `uv sync --extra cpu` (pycolmap: Linux x86_64, macOS arm64 14+, Windows); with
`uv pip install -r pyproject.toml` add `--extra cuda` or `--extra cpu`; with pip install only one of
`"pycolmap-cuda12>=4.2,<5"` or `"pycolmap>=4.2,<5"` (`pip uninstall pycolmap` first when switching to cuda).
OpenMVS is looked up on `PATH` (or `dense.openmvs.bin_dir`); missing -> WARNING, `openmvs` skipped.

What changes on macOS (no CUDA):

- SfM runs on the CPU (`sfm.use_gpu: auto` resolves to false). Slower, same results.
- COLMAP patch-match stereo needs CUDA, so the `patch_match` dense method is skipped with a WARNING; without
  OpenMVS there is no dense cloud, and Poisson/close_holes/mesh fall back to the sparse cloud
  (`poisson.source: dense` -> sparse with a WARNING). Set `dense.enabled: false` to skip the attempt.
- Recording the viewer orbit (`viewer.enabled`) needs a GUI session (logged-in desktop). Over SSH the
  recording fails in its child process, the video is skipped with a WARNING and the run continues.
  `python3 view.py` also needs the desktop.

## Run

```bash
python3 main.py <config>.yaml          # whole pipeline; no argument = configs/default.yaml
python3 main.py --config <path>.yaml   # same thing
./run.sh <config>.yaml                 # POSIX one-liner that forwards to main.py
python3 view.py <config>.yaml          # interactive Open3D viewer of a finished run of that config
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
13. **poisson** (`src/mesh/poisson.py`): Poisson surface, variants `pre` (cleaned cloud) and `raw` (no cleanup).
14. **close_holes** (`src/mesh/close_holes.py`): MeshLab Close Holes on every Poisson variant, with (`pre`)
    and without (`raw`) mesh repair first.
15. **mesh** (`src/mesh/mesh.py`): the chosen mesh in PLY/OBJ/STL/OFF/GLB plus a scene with cameras and path.
16. **export** (`src/export/meshlab.py`): MeshLab project with every cloud and mesh as layers, filter script.
17. **evaluation** (`src/evaluation`): reprojection error, metric error against tape measures, loop drift,
    smooth vs textured walls, ablation table across runs.
18. **viewer** (`src/viewer.py`): orbit video of the chosen cloud, recorded in a child process.
19. **videos** (`src/videos.py`): top-view trajectory, kept vs discarded frames, `videos.csv`.
20. **report** (`src/relatorio.py`): `report.md` built only from the saved files.

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
    sfm/ scale/ clean/<source>/ planes/ floorplan/ dense/<method>/ mesh/{poisson,close_holes}/ evaluation/
    exports/
      colmap/              COLMAP workspace: images/, database.db, sparse/0/, sparse.ply, project.ini, colmap.log
      meshlab/             project.mlp, process.mlx, sparse.ply, fused_<method>.ply, cloud_clean_<source>.ply,
                           poisson_<v>.ply, close_holes_<pv>_<cv>.ply, mesh.{ply,obj,stl,off}, scene.obj
      web/                 scene.glb, mesh.glb
    videos/                navigation_<cloud>_<s>s.mp4, trajectory_topdown.mp4, frames_kept_vs_discarded.mp4
  artifacts/<id>/          one CSV per stage (extraction_metrics, frame_quality_metrics, sfm_metrics, scale,
                           cleanup, planes, floorplan, dense_metrics, poisson, close_holes, mesh, evaluation,
                           videos, ablation, ...) and report.md
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

Poisson (`poisson.threads`), Close Holes and OpenMVS (`dense.openmvs.max_threads`) run on the CPU.

### YAML reference

Generated from `src/constants.py` and the comments of `configs/default.yaml`; blocks in YAML order.

| key | default | meaning |
|---|---|---|
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
| **`mesh`** | | src/mesh/mesh.py (formats + scene of ONE chosen mesh) |
| `mesh.enabled` | `true` | export the chosen mesh and the scene (cameras + trajectory) |
| `mesh.input` | `close_holes` | close_holes (pre_pre) \| poisson (pre): mesh exported to PLY/OBJ/STL/OFF/GLB |
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

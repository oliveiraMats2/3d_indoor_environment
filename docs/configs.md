# Configs

[← README](../README.md)

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
| SfM with InstantSfM, MASt3R-SfM or VGGT | always CUDA, `.venv-gpu` (see [GPU SfM techniques](gpu_sfm.md)) | `sfm.technique`, `sfm.instantsfm`, `sfm.mast3r_sfm`, `sfm.vggt` |

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

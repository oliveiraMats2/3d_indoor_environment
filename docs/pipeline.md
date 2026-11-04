# Pipeline

[← README](../README.md)

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

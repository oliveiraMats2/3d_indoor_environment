# Outputs

[← README](../README.md)

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

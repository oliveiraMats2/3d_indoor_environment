# 3D Indoor Environment: Stage 3, part 2 (epipolar geometry)

Two-view geometry for every detector (SIFT, ORB, AKAZE) and every consecutive frame pair, built on the
Lowe-filtered matches of Stage 3 part 1 (code in `src/epipolar/`):

- **geometry** stage (`src/epipolar/__init__.py`, `estimate.py`): estimate the fundamental matrix `F` and the
  essential matrix `E` with a robust estimator (USAC_MAGSAC by default); inliers = accepted by both.
- **two_view** stage (`src/epipolar/two_view.py`): recover `R, t` from `E` with the cheirality check (`pose.py`),
  draw epipolar lines on both frames (`visualize.py`) and triangulate the initial two-view 3D cloud (`triangulate.py`).

Full write-up (method, results, discussion): [docs/etapa3_parte2.md](docs/etapa3_parte2.md).

## Run

Python 3.12 virtual environment with the dependencies of `pyproject.toml`:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install opencv-contrib-python numpy pyyaml "pydantic>=2" tqdm   # or: uv pip install -r pyproject.toml
```

Put the video next to `main.py` (file name = `extraction.video`, default `3d_image_video.MOV`), then:

```bash
python main.py                       # uses config.yaml
python main.py --config <file>.yaml  # any other config (run id = file name)
```

Stages, in order: extraction, frame_quality, detection, matching, geometry, two_view, report.
`opencv-contrib-python` is required (AKAZE lives in `cv2.xfeatures2d` on OpenCV 5).

## config.yaml

`config.yaml` is partial: only the stages implemented so far (extraction to two-view geometry); later stages are
not included yet. Unknown keys are rejected (pydantic `extra="forbid"`).

| Key | Meaning |
|---|---|
| `run` | `id` (null = config file name), output/artifact roots, `seed`, `log_level`, `overwrite` (delete `results/<id>/` first), `cache` (reuse detection) |
| `extraction` | video and frame selection (overlap 60 to 80%, sharpness); `rerun: false` reuses `frames/` when `selecao.csv` exists |
| `frame_quality` | blur, exposure, contrast and texture thresholds for discarding frames |
| `detection` | `methods` (sift, orb, akaze), `nfeatures`, timing `reps`/`warmup` |
| `filters` | `lowe_ratio`, `min_matches` (pairs below it are unsuccessful) |
| `matching` | `name` (brute_force, flann), `strategy` (sequential, all_pairs), timing reps, FLANN index params |
| `report` | `max_figures` for the detection/matching report |
| `geometry.enabled` | run the geometry and two_view stages |
| `geometry.ransac.method` | OpenCV robust method name, resolved as `cv2.<method>` (default `USAC_MAGSAC`) |
| `geometry.ransac.threshold_px` | max distance (px) from a point to its epipolar line to count as inlier |
| `geometry.ransac.confidence` | RANSAC confidence (0.999) |
| `geometry.ransac.max_iters` | RANSAC iteration cap (5000) |
| `geometry.intrinsics.{fx,fy,cx,cy}` | camera intrinsics; null fx/fy -> `max(w, h)`, null cx/cy -> `w/2, h/2` |
| `visualize.matches` | `max_lines`, `dpi`, `format` of the match figures |
| `visualize.epilines.n_lines` | number of inliers sampled (seeded by `run.seed`) whose epipolar lines are drawn |
| `visualize.epilines.dpi` | figure resolution; the side-by-side canvas is 12 in wide (150 dpi -> 1800 px) |

## Outputs

`results/<id>/epipolar/<det>/<A>__<B>/` (one folder per detector and frame pair):

| File | Content |
|---|---|
| `F.npy`, `E.npy`, `K.npy` | fundamental, essential (NaN if estimation failed) and intrinsics, 3x3 |
| `inlier_mask.npy`, `sampson_errors.npy` | inliers of both F and E over the Lowe matches; Sampson errors |
| `R.npy`, `t.npy`, `pose_mask.npy` | rotation, unit-norm translation of camera 2 w.r.t. camera 1; inliers that pass cheirality |
| `points3d.npy`, `colors.npy`, `cloud.ply` | triangulated points (N, 3), RGB colors from frame A, cloud for viewing (Open3D) |

`results/<id>/figures/epilines/<det>/<A>__<B>.png`: epipolar lines of the sampled inliers drawn on both frames
(lines from frame A on frame B and vice versa, same color per pair). `results/<id>/K.npy` keeps the K used.

`artifacts/<id>/epipolar_metrics.csv` (one row per detector and pair, rewritten every run):

| Column | Meaning |
|---|---|
| `id`, `detector`, `pair` | `<det>/<A>__<B>`, detector name, pair name |
| `n_matches`, `n_inliers` | Lowe matches, inliers of F and E |
| `rot_deg` | rotation angle of `R` in degrees (NaN if no pose) |
| `t_x`, `t_y`, `t_z` | unit translation direction |
| `n_pose_inliers` | inliers that pass the cheirality check in `recoverPose` |
| `n_points3d` | triangulated points kept (in front of both cameras) |
| `parallax_med_deg` | median triangulation angle between the two rays |
| `cheirality_frac` | kept / triangulated points |

Per-detector medians of these metrics go to `results/<id>/run.log`.

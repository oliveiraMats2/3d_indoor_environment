# History and side tools

[← README](../README.md)

## History: first milestone (2D feature study)

Stage list and layout of the first milestone, kept as written. Stages 3 to 6 still run as described;
pose, triangulation and evaluation have since been implemented (see [pipeline.md](pipeline.md)).

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

## First-milestone detector: AKAZE (superseded by SIFT)

The final pipeline uses **SIFT**: in the full benchmark (100 frames, 99 pairs) SIFT scored 0.936 against 0.328 for ORB and 0.061 for AKAZE, and COLMAP SfM uses SIFT (`detector_choice.csv`). The text below is the first-milestone record, kept for history.

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

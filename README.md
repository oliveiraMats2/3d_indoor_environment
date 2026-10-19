# 3D Indoor Environment

3D reconstruction of an indoor environment from frames extracted from a phone video.

## Setup

Create a local Python 3.12 environment and install the dependencies with uv:

```bash
uv python install 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r pyproject.toml
```

## Run

```bash
python main.py --config configs/default.yaml
```

`main.py` is the only entry point. Stages, in order:

1. **extraction** (`src/extrair_quadros.py`): sharp frames with 60 to 80% overlap from the video.
2. **frame quality** (`src/frame_quality.py`): blur, exposure, contrast and texture checks.
3. **detection** (`src/detectores.py`): SIFT x ORB x AKAZE keypoints and descriptors as `.npy`.
4. **matching** (`src/matching`): BF/FLANN, Lowe filtering and runtime logs for the configured frame pairs.
5. **geometry** (`src/epipolar`): fundamental/essential matrices and RANSAC inlier masks.
6. **report** (`src/relatorio.py`) and matching metrics/figures: artifacts built only from saved `.npy` files.
7. **pose**, **triangulation** and **evaluation**: pending stages.

## Configs

- One YAML = one experiment. `configs/default.yaml` holds every key with its default;
  each file in `configs/experiments/` is a full copy with one line changed (e.g. `lowe_070.yaml`).
- `run.id: null` uses the YAML file name as the run id.
- Unknown keys are rejected (pydantic `extra="forbid"`), so typos fail before anything runs.
- `extraction.rerun: false` reuses the frames in `frames/` (when `selecao.csv` exists); `true` deletes them and extracts again.
- `run.cache: true` reuses detection
  (when `results/<id>/K.npy` exists). `run.overwrite: true` deletes `results/<id>/` first;
  with both false an existing run is never touched.

## Outputs

- `frames/`: extracted frames plus `selecao.csv`, shared by all experiments.
- `results/<id>/`: `run.log`, `config.resolved.yaml`, `K.npy`, `detection/`, `matching/`, `figures/`.
- `artifacts/<id>/`: `frame_quality_metrics.csv`, `detection_metrics.csv`, `matching_metrics.csv`, `summary.csv`.

## Layout

```
main.py                 entry point (config validation, stage order, logging, cache)
configs/                default.yaml, experiments/*.yaml
src/
  constants.py          YAML schema (pydantic, unknown keys rejected)
  extrair_quadros.py    stage 1, helpers in extraction/
  frame_quality.py      stage 2
  detectores.py         stage 3
  relatorio.py          stage 4
  matching/             matcher, Lowe filter, metrics and figures
  epipolar/             F/E with RANSAC; pose, epipolar lines and triangulation are pending
  evaluation/           reprojection, metric error, loop drift, ablation (stub)
```

Requires `opencv-contrib-python` (in OpenCV 5, AKAZE lives in `cv2.xfeatures2d`).
Keypoint figures (`figures/keypoints/<detector>/`) show scale as the circle and orientation as
the line; `figures/registration/<detector>/` holds only pairs with >= 8 RANSAC inliers of `E`.
Intrinsics are approximate (`f = max(w, h)`) unless `geometry.intrinsics` is set.

### Chosen detector: AKAZE

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

# 3D Indoor Environment

Extracts sharp, well-overlapping frames from a video for 3D reconstruction (e.g. COLMAP).

## Setup

Install dependencies into the `shopin` conda env with uv:

```bash
conda activate shopin
uv pip install --python "$(which python)" -r pyproject.toml
```

## Run

```bash
python extrair_quadros.py 3d_image_video.MOV --saida frames
```

Output: `frames/img_XXXX.jpg` plus `frames/selecao.csv` (source frame, timestamp, overlap with previous image).

## Options

| Flag | Default | Meaning |
|---|---|---|
| `--min` / `--max` | 100 / 300 | target number of frames |
| `--overlap` | 0.70 | initial overlap target (0.60 to 0.80) |
| `--stride` | 2 | analyze 1 of every N frames |
| `--largura` | 1920 | max width of saved frames |

## How it works

1. Samples frames, detects ORB features, and drops the blurriest 25% (Laplacian variance).
2. Estimates the overlap between frames with a RANSAC homography.
3. Keeps the most spaced-out frames that still meet the overlap target, adjusting the target until the frame count fits the min/max range.

## Detectors (SIFT x ORB x AKAZE)

Requires `opencv-contrib-python` (in OpenCV 5, AKAZE lives in `cv2.xfeatures2d`).

```bash
python detectores.py frames --run-id t1                 # 5000 features, 30 reps, 2 warmup -> artifacts/t1/*.npy
python relatorio.py artifacts/t1 frames                 # CSVs + figures only from the .npy
```

Output: `artifacts/t1/metrics/{detection_metrics,matching_metrics,resumo}.csv`,
`figures/keypoints/<detector>/` (scale = circle, orientation = line) and
`figures/registro/<detector>/` (only pairs with >= 8 RANSAC inliers of `E`).
Intrinsics are approximate (`f = max(w, h)`); pass `--focal` if calibrated.

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
when time is the constraint. Confirm with `resumo.csv` on the full frame set.

# Running

[← README](../README.md)

## Choosing the flow

```yaml
# config.yaml (read by `python3 main.py` without arguments)
flow: cpu        # cpu (default, also when the key is missing or null) | gpu
```

`flow: cpu` fills `sfm.technique: colmap`, `sfm.mapper: incremental`, `sfm.container: null`.
`flow: gpu` fills `sfm.technique: glomap`, `sfm.mapper: global`, `sfm.container: t2/colmap-cudss:4.2.0`,
`sfm.container_steps: [mapper]`, `sfm.container_ba_use_gpu: true`.

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

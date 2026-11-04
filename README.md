# 3D Indoor Environment

3D reconstruction of an indoor environment from a phone video: frames -> features -> two-view geometry -> SfM ->
metric scale -> cleanup -> planes and floor plan -> dense MVS (and optional Gaussian Splatting) -> Poisson mesh
-> hole closing -> exports (COLMAP, MeshLab, web) -> evaluation -> videos -> report. One YAML drives the whole run.

## Installation

CPU flow (default). Needs COLMAP >= 4.0 on `PATH` and [uv](https://docs.astral.sh/uv/):

```bash
uv python install 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r pyproject.toml
```

GPU flow (`flow: gpu`, NVIDIA + Docker): separate `.venv-gpu` plus the `t2/colmap-cudss:4.2.0` container.
Requirements per OS, the GPU steps and the checks are in [docs/install.md](docs/install.md).

## Running: `run.sh`

```bash
./run.sh                       # main.py + view.py with config_cozinha.yaml
./run.sh <config>.yaml         # same, with another config
./run.sh -h                    # usage
```

`run.sh` runs the pipeline (`main.py`) and then the interactive viewer (`view.py`) with the same config. It uses
`.venv/bin/python` when it exists, else `uv run python`. To run the stages directly:

```bash
python3 main.py <config>.yaml                    # CPU flow
.venv-gpu/bin/python main.py config_gpu.yaml     # GPU flow
python3 view.py <config>.yaml                    # viewer of a finished run
```

Config lookup, ablations and how to add a new video: [docs/running.md](docs/running.md).

## Flows

The top-level key `flow` of the config picks the SfM path (`src/flows.py`); everything else is shared.

| | `flow: cpu` (default) | `flow: gpu` |
|---|---|---|
| SfM | COLMAP SIFT + incremental mapper, Ceres BA on the CPU | COLMAP SIFT + GLOMAP `global_mapper`, BA on the GPU (`t2/colmap-cudss:4.2.0`) |
| ablations | `ablation.csv` from `evaluation.ablation_runs` | not run |
| environment | `.venv` | `.venv-gpu` + Docker with the NVIDIA container toolkit |
| example configs | `config.yaml`, `configs/kitchen_my_home_cpu.yaml`, `configs/experiments/*.yaml` | `config_gpu.yaml`, `config_gpu_keep.yaml`, `configs/kitchen_my_home_gpu.yaml` |

Keys written in the YAML always win over the flow preset.

## Documentation

| Doc | Content |
|---|---|
| [docs/install.md](docs/install.md) | CPU and GPU installs, requirements, macOS notes |
| [docs/running.md](docs/running.md) | `main.py`/`view.py`, config lookup, ablations, adding a new video |
| [docs/pipeline.md](docs/pipeline.md) | the stages in order and Stage 5 (evaluation) |
| [docs/outputs.md](docs/outputs.md) | output layout and how to open it (COLMAP, MeshLab, glTF, Gaussian Splatting) |
| [docs/configs.md](docs/configs.md) | config files, blur decision, GPU usage, full YAML reference |
| [docs/gpu_sfm.md](docs/gpu_sfm.md) | InstantSfM, MASt3R-SfM, VGGT, GLOMAP, the cuDSS container |
| [docs/gaussian_splatting.md](docs/gaussian_splatting.md) | optional 3DGS stage and its install |
| [docs/history.md](docs/history.md) | first milestone (AKAZE, later replaced by SIFT), frame quality analyzer (`analyze_frames.py`) |
| [docs/etapa3_parte2.md](docs/etapa3_parte2.md) | Stage 3 part 2 write-up |

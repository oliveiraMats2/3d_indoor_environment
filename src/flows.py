# ╔══════════════════════════════════════════════════════════════════════════════════════╗
# ║  ⠀⠀⠀⠀⣠⠶⡒⠒⢬⡲⣮⠂⣆⣀⠀⠀⠀⠀⠀⠀⢀⣤⣴⣦⣤⡀⠀⠀⠀⠀   MATEUS OLIVEIRA                        ║
# ║  ⠀⠀⠀⣀⣥⠠⣿⠆⠐⣻⣾⣿⣿⢷⡄⠀⠀⠀⠀⢠⡿⠋⠉⠉⠙⢿⡄⠀⠀⠀   m203656@dac.unicamp.edu.br             ║
# ║  ⠀⠀⢘⡵⢋⠄⡙⠒⣤⣄⣉⠙⣿⣗⠑⡄⠀⠀⠀⠘⡇⠀⠀⠀⠀⠈⡇⠀⠀⠀   UNICAMP - Universidade Estadual de     ║
# ║  ⠀⣴⢿⡜⢡⡞⢀⢼⣿⣿⣿⣿⣿⣿⠟⣂⠀⠀⢀⣀⠱⡀⠀⠀⠀⢰⠁⠀⠀⠀               Campinas                     ║
# ║  ⠰⢫⢟⡇⢸⡇⢸⢾⣿⣿⣿⣿⣿⣿⡷⠰⠀⢰⡏⠀⠀⢡⠀⠀⢠⠃⠀⠀⠀⠀   IC - Institute of Computing            ║
# ║  ⢰⠁⣿⢣⣿⠇⢀⣿⣿⡿⠿⠤⣭⣥⣶⡆⠀⠸⣷⣤⣠⡾⠀⢀⡇⠀⠀⠀⠀⠀   Computer Science Department              ║
# ║  ⡞⣰⣧⠟⡝⢸⢸⣿⣥⠖⣴⡆⣤⣬⠉⠀⠀⠀⠈⠉⠉⠀⠀⢸⣇⠀⠀⠀⠀⠀   github.com/oliveiraMats2              ║
# ║  ⠀⡿⡟⢸⡇⠸⡄⢹⣿⢸⣿⣇⡏⠟⣰⣄⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠁⠀⠀⠀   linkedin.com/in/mateus-eng            ║
# ║  ⠀⠇⣧⠘⡇⠦⣹⣸⣿⡇⡿⡿⣡⣼⣿⣿⣷⣦⣄⡀⠀⠀⣸⣿⣿⠄⠻⢷⣦⠀                                            ║
# ║  ⠀⢀⠘⣇⢹⡸⣿⣿⣿⢹⢃⣠⣿⣿⣿⣿⣿⣿⣿⣿⣆⠀⠑⠋⠉⠀⠀⠈⣿⣧   UNICAMP · IC · 2026                    ║
# ║  ⠀⢸⣿⡌⠘⢷⣿⣿⡏⢀⣾⣿⣿⣿⣿⣿⣿⢻⣿⣿⣿⡆⠀⠀⠀⠀⠀⠀⣿⡿                                            ║
# ║  ⠀⠈⣿⣿⣦⡌⢿⠏⣰⣿⣿⣿⣿⣿⣿⡿⡏⣼⣿⣿⣿⡇⣄⠀⠀⠀⢀⣼⣿⠇                                            ║
# ║  ⠀⠀⠹⣿⣿⢻⡀⣼⣿⣿⢻⣿⣿⣿⣿⡇⡇⢻⣿⣿⣿⡇⣿⣿⣶⣿⣿⠟⠁⠀                                            ║
# ║  ⠀⠀⠀⢻⣿⣦⡓⢿⣿⣿⡆⣿⣿⣿⣿⢃⣶⡸⣿⣿⣿⡇⠀⠉⠉⠁⠀⠀⠀⠀                                            ║
# ║  ⠀⠀⠀⠈⣿⣿⣿⡆⠀⠀⠀⣿⣿⣿⡟⣼⡿⠁⢹⣿⣿⣷⠀⠀⠀⠀⠀⠀⠀⠀                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════════════╝

# PROMPT COMPLEMENT
# Stage: config (before validation) - the two pipeline flows chosen by the top-level YAML key `flow`.
# Goal: one switch for the SfM path: cpu = COLMAP SIFT + incremental mapper (Ceres on the CPU, light ablation
#      configs), gpu = GLOMAP global mapper with global positioning + bundle adjustment on the GPU through the
#      COLMAP + Ceres/cuDSS container (docker/colmap-cudss). Densification, mesh and evaluation are shared.
# In:  the raw YAML dict.
# Out: the same dict with the flow preset filled in under `sfm`.
# Rules: keys written in the YAML always win (the preset only fills missing keys); flow null = no preset.

# >>> BEGIN IMPLEMENTATION >>>
"""Flow presets (flow: cpu | gpu) applied to the raw config before validation."""
import copy
import logging

log = logging.getLogger(__name__)

GPU_IMAGE = "t2/colmap-cudss:4.2.0"
FLOWS = {
    "cpu": {"sfm": {"technique": "colmap", "mapper": "incremental", "container": None}},
    "gpu": {"sfm": {"technique": "glomap", "mapper": "global", "container": GPU_IMAGE,
                    "container_steps": ["mapper"], "container_ba_use_gpu": True}},
}

def apply(raw: dict) -> dict:
    """Fill the preset of raw['flow'] into raw (explicit YAML keys are kept)."""

    flow = raw.get("flow")
    if flow is None:
        return raw
    if flow not in FLOWS:
        raise ValueError(f"flow must be one of {sorted(FLOWS)} or null, got {flow!r}")
    out = copy.deepcopy(raw)
    for block, preset in FLOWS[flow].items():
        section = out.setdefault(block, {}) or {}
        out[block] = {**preset, **section}
    log.info("flow %s: %s", flow, {k: out["sfm"].get(k) for k in FLOWS[flow]["sfm"]})
    return out

# <<< END IMPLEMENTATION <<<

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.constants import Config

    assert apply({}) == {}
    cpu = apply({"flow": "cpu"})
    assert cpu["sfm"]["technique"] == "colmap" and cpu["sfm"]["container"] is None
    gpu = apply({"flow": "gpu", "sfm": {"matcher": "sequential", "container": None}})
    assert gpu["sfm"]["technique"] == "glomap" and gpu["sfm"]["mapper"] == "global"
    assert gpu["sfm"]["container"] is None and gpu["sfm"]["matcher"] == "sequential"  # YAML wins
    for raw in (cpu, gpu):
        Config.model_validate(raw)
    try:
        apply({"flow": "tpu"})
        raise AssertionError("bad flow accepted")
    except ValueError:
        pass
    print("flows self-test OK")

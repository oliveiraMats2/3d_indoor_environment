"""Roda so o estagio lidar sobre a reconstrucao da cozinha (symlinks somente leitura para R)."""
import logging, sys, yaml
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation import lidar

logging.basicConfig(level=logging.INFO)
W = Path(__file__).resolve().parents[1]
src = Path("/home/moliveira/Documents/3d_indoor_environment/cozinha/results/config_cozinha")
out = W / "outputs" / "t5_lidar"
run = out / "run"; art = out / "artifacts"
run.mkdir(parents=True, exist_ok=True); art.mkdir(exist_ok=True)
for name in ["clean", "scale", "planes", "sfm"]:
    link = run / name
    if not link.exists():
        link.symlink_to(src / name)
cfg = yaml.safe_load((W / "configs/report/cozinha_lidar_eval.yaml").read_text())
print(lidar.run(cfg, str(run), str(art)))

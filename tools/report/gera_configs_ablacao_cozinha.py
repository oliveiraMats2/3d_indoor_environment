"""Gera configs/report/ablacao_cozinha_*.yaml a partir de config_cozinha.yaml (rodada final, flow gpu).

Uso: <R>/.venv-gpu/bin/python tools/report/gera_configs_ablacao_cozinha.py
Mantem a rodada final (GLOMAP em GPU, escala por contorno, leave-one-out) e muda so o eixo da ablacao
(matcher/laco e stride) e os estagios: densificacao, malha, exportacao, visualizador, videos, geometria 2D e
casamento 2D ficam desligados. Saidas em cozinha_ablacao/<variante>/ (relativo a R), sem tocar em cozinha/results.
"""
import sys
from pathlib import Path

import yaml

W = Path(__file__).resolve().parents[2]
BASE = W / "config_cozinha.yaml"
OUT = W / "configs" / "report"
OFF = ["matching", "geometry", "dense", "dense_compare", "gaussian_splatting", "poisson", "close_holes",
       "liepa", "mesh", "export", "viewer", "videos", "photo_vs_render"]
# variante: (stride, matcher, loop_detection)
VARIANTES = {
    "seq_sem_laco": (80, "sequential", False),
    "seq_com_laco": (80, "sequential", True),
    "exaustivo": (80, "exhaustive", False),
    "seq_com_laco_146": (160, "sequential", True),
}

base = yaml.safe_load(BASE.read_text())
OUT.mkdir(parents=True, exist_ok=True)
for nome, (stride, matcher, laco) in VARIANTES.items():
    c = yaml.safe_load(yaml.safe_dump(base))
    c["run"].update(id=nome, output_root="cozinha_ablacao", artifacts_root="cozinha_ablacao/artifacts")
    e = c["extraction"]
    e.update(stride=stride, output_dir=f"cozinha_ablacao/frames/stride{stride}", tag=None, selection="all",
             min_frames=0, max_frames=None)
    e["blur"]["enabled"] = False
    c["detection"].update(reps=3, warmup=1, max_frames=20)
    c["visualize"]["matches"]["enabled"] = False
    c["sfm"].update(matcher=matcher, loop_detection=laco)
    for k in OFF:
        if k in c:
            c[k]["enabled"] = False
    cab = (f"# Ablacao na cozinha (T8, R28), variante {nome}: stride {stride}, sfm.matcher {matcher}, "
           f"loop_detection {laco}.\n# Gerado por tools/report/gera_configs_ablacao_cozinha.py; base config_cozinha.yaml.\n"
           f"# Estagios ate SfM + escala + loop_drift + metric_error; saida: cozinha_ablacao/{nome}/\n")
    (OUT / f"ablacao_cozinha_{nome}.yaml").write_text(cab + yaml.safe_dump(c, sort_keys=False, allow_unicode=True))
    print(nome, stride, matcher, laco)

"""Recalcula a ablacao do kitchen_my_home (R28): erro metrico medio so com linhas leave-one-out.

O ablation.csv original faz a media de todas as linhas de metric_error.csv, inclusive room_width e
room_length (circulares: a escala sai do mesmo contorno). Aqui so entram as linhas com
scale_source == leave_one_out e rel_error finito. Le os CSV originais, nao altera nada neles.

Uso: R/.venv/bin/python tools/report/ablacao_loo.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.evaluation.ablation import _to_float  # noqa: E402
from src import stage_io  # noqa: E402

ART = Path("/home/moliveira/Documents/3d_indoor_environment/kitchen_my_home/artifacts")
OUT = Path("/home/moliveira/Documents/report_3d_indoor_environment/notas")
VARIANTS = [  # (rotulo, run_id)
    ("seq+laco", "kitchen_abl_seq_loop"),
    ("seq sem laco", "kitchen_abl_seq_noloop"),
    ("exaustivo", "kitchen_abl_exhaustive"),
    ("200 quadros", "kitchen_abl_frames_200"),
    ("100 quadros", "kitchen_abl_frames_100"),
]
COLS = ["variante", "run_id", "quadros", "registradas", "pontos", "reproj_media_px", "reproj_mediana_px",
        "n_loo", "erro_loo_medio_cm", "erro_loo_medio_pct", "deriva_cm", "deriva_pct", "sfm_s",
        "erro_antigo_ablation_csv", "reproj_media_colmap_px"]


def first(path):
    rows = stage_io.read_csv(path)
    return rows[0] if rows else {}


def main():
    old = {r["run_id"]: r for r in stage_io.read_csv(ART / "kitchen_ablation" / "ablation.csv")}
    out = []
    for label, rid in VARIANTS:
        d = ART / rid
        sfm, ev, ld = first(d / "sfm_metrics.csv"), first(d / "evaluation.csv"), first(d / "loop_drift.csv")
        me = stage_io.read_csv(d / "metric_error.csv")
        loo = [r for r in me if r.get("scale_source") == "leave_one_out"
               and np.isfinite(_to_float(r.get("rel_error")))]
        rel = [_to_float(r["rel_error"]) for r in loo]
        ab = [_to_float(r["abs_error_cm"]) for r in loo]
        out.append({
            "variante": label, "run_id": rid, "quadros": sfm["n_frames"], "registradas": sfm["n_registered"],
            "pontos": ev["n_sparse"], "reproj_media_px": f"{float(ev['reproj_mean_px']):.4f}",
            "reproj_mediana_px": f"{float(ev['reproj_median_px']):.4f}", "n_loo": len(loo),
            "erro_loo_medio_cm": f"{np.mean(ab):.2f}" if loo else "-",
            "erro_loo_medio_pct": f"{100 * np.mean(rel):.2f}" if loo else "-",
            "deriva_cm": f"{float(ld['delta_cm']):.2f}", "deriva_pct": f"{float(ld['drift_pct']):.2f}",
            "sfm_s": f"{float(sfm['t_total_s']):.2f}",
            "erro_antigo_ablation_csv": old[rid]["metric_error_mean_rel"],
            "reproj_media_colmap_px": f"{float(sfm['reproj_mean_px']):.4f}",
        })
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "t3_ablacao_kitchen.csv", "w") as fh:
        fh.write(",".join(COLS) + "\n")
        for r in out:
            fh.write(",".join(str(r[c]) for c in COLS) + "\n")
    print((OUT / "t3_ablacao_kitchen.csv").read_text())


if __name__ == "__main__":
    main()

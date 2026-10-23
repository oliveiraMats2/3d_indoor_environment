"""Self-contained HTML quality report and output README of the frame quality analyzer."""
from __future__ import annotations

import base64
import logging
import math
import os
import shlex
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Environment, FileSystemLoader

log = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).parent / "templates"
TEMPLATE_NAME = "report.html.j2"
REPORT_NAME = "quality_report.html"
README_NAME = "README.md"
GRAPH_HTML_NAME = "connectivity_graph.html"

# (anchor id, heading) in the order required by the contract
SECTIONS: list[tuple[str, str]] = [
    ("resumo", "1. Resumo executivo"),
    ("frames", "2. Frames analisados"),
    ("resolucao", "3. Estatísticas de resolução"),
    ("nitidez", "4. Nitidez (variância do Laplaciano)"),
    ("sift", "5. SIFT: contagem e densidade de keypoints"),
    ("espacial", "6. Distribuição espacial (heatmaps)"),
    ("matching", "7. Correspondências e inliers"),
    ("grafo", "8. Grafo de conectividade"),
    ("componentes", "9. Componentes conexas"),
    ("revisao", "10. Frames que precisam de revisão"),
    ("referencia", "11. Connectivity Graph: Visual Reference"),
    ("limitacoes", "12. Limitações metodológicas"),
    ("recomendacoes", "13. Recomendações para a aquisição"),
]
APPENDIX: list[tuple[str, str]] = [
    ("execucao", "Apêndice A. Execução, avisos e falhas"),
    ("configuracao", "Apêndice B. Configuração e limiares aplicados"),
]

# figure key -> section anchor; keys may also carry a "plot_" prefix
FIGURE_SECTIONS: dict[str, str] = {
    "sharpness": "nitidez",
    "sharpness_distribution": "nitidez",
    "sift_keypoints": "sift",
    "sift_density": "sift",
    "sift_distribution": "sift",
    "grid_occupancy": "espacial",
    "keypoint_heatmap": "espacial",
    "grid_examples": "espacial",
    "matching_inliers": "matching",
    "matching_inlier_ratio": "matching",
    "matching_scatter": "matching",
    "temporal_redundancy": "revisao",
    "graph": "grafo",
    "connectivity_graph": "grafo",
    "degree_distribution": "componentes",
}
FIGURE_CAPTIONS: dict[str, str] = {
    "sharpness": "Variância do Laplaciano por frame, com os limiares relativos de revisão e de rejeição.",
    "sharpness_distribution": "Distribuição da variância do Laplaciano nos frames legíveis.",
    "sift_keypoints": "Número de keypoints SIFT por frame.",
    "sift_density": "Densidade de keypoints SIFT por megapixel (na resolução de análise).",
    "sift_distribution": "Distribuição do número de keypoints SIFT.",
    "grid_occupancy": "Fração de células da grade com pelo menos um keypoint, por frame.",
    "keypoint_heatmap": "Média de keypoints por célula da grade sobre todos os frames legíveis.",
    "grid_examples": "Contagem de keypoints por célula em frames reais selecionados (sem a imagem de fundo).",
    "matching_inliers": "Inliers da matriz fundamental por par, uma série por distância temporal.",
    "matching_inlier_ratio": "Razão de inliers (inliers / correspondências após Lowe) por par.",
    "matching_scatter": "Inliers versus razão de inliers; a região sombreada é a de aceitação de arestas.",
    "temporal_redundancy": "Pares consecutivos: razão de inliers e deslocamento mediano ao longo da sequência.",
    "graph": "Grafo de conectividade: verde = frames conectados, laranja = frames isolados.",
    "connectivity_graph": "Grafo de conectividade: verde = frames conectados, laranja = frames isolados.",
    "degree_distribution": "Distribuição do grau (número de arestas aceitas) por frame.",
}

STAT_LABELS: dict[str, str] = {
    "laplacian_var": "Variância do Laplaciano",
    "sift_keypoints": "Keypoints SIFT",
    "sift_density_per_mp": "Densidade SIFT (kp/MP)",
    "occupancy_ratio": "Ocupação da grade",
    "inliers": "Inliers por par",
    "inlier_ratio": "Razão de inliers",
    "brightness_mean": "Brilho médio",
    "dark_frac": "Fração muito escura",
    "saturated_frac": "Fração saturada",
}
STAT_COLUMNS = ("mean", "std", "min", "p05", "median", "p95", "max")

# AnalyzerConfig field -> CLI flag (used to rebuild the exact command)
CLI_FLAGS: dict[str, str] = {
    "limit": "--limit", "stride": "--stride", "analysis_max_side": "--analysis-max-side", "seed": "--seed",
    "sift_nfeatures": "--sift-nfeatures", "sift_contrast_threshold": "--sift-contrast-threshold",
    "sift_edge_threshold": "--sift-edge-threshold", "lowe_ratio": "--lowe-ratio",
    "min_lowe_matches": "--min-lowe-matches", "ransac_threshold_px": "--ransac-threshold",
    "ransac_confidence": "--ransac-confidence", "ransac_max_iters": "--ransac-max-iters",
    "edge_min_inliers": "--edge-min-inliers", "edge_min_inlier_ratio": "--edge-min-inlier-ratio",
    "sharpness_poor_rel": "--sharpness-poor-rel", "sharpness_review_rel": "--sharpness-review-rel",
    "sharpness_abs_min": "--sharpness-abs-min", "keypoints_poor_min": "--keypoints-poor-min",
    "keypoints_review_min": "--keypoints-review-min", "occupancy_review_min": "--occupancy-review-min",
    "degree_review_min": "--degree-review-min", "redundancy_min_inlier_ratio": "--redundancy-min-inlier-ratio",
    "redundancy_max_median_disp_px": "--redundancy-max-median-disp", "reference_url": "--reference-url",
}
NEGATED_FLAGS: dict[str, str] = {
    "compute_homography": "--no-homography",
    "redundancy": "--no-redundancy",
    "interactive_graph": "--no-interactive-graph",
}

MISSING = "—"


# ---------------------------------------------------------------- formatting


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def fmt_num(value: Any, digits: int = 2) -> str:
    """Format a number for the report; None/NaN become a dash."""
    value = _py(value)
    if _is_missing(value):
        return MISSING
    if isinstance(value, bool):
        return "sim" if value else "não"
    if isinstance(value, int) or (isinstance(value, float) and value.is_integer() and abs(value) >= 1000):
        return str(int(value))
    if isinstance(value, float):
        return f"{value:.1f}" if abs(value) >= 100 else f"{value:.{digits}f}"
    try:
        return fmt_num(float(value), digits) if not isinstance(value, str) else value
    except (TypeError, ValueError):
        return str(value)


def fmt_pct(value: Any, digits: int = 1) -> str:
    """Format a 0..100 percentage."""
    return MISSING if _is_missing(value) else f"{float(value):.{digits}f}%"


def _ratio_pct(value: Any) -> str:
    """Format a 0..1 fraction as a percentage."""
    return MISSING if _is_missing(value) else fmt_pct(100.0 * float(value), 0)


def _frac_pct(part: Any, total: Any) -> str:
    if _is_missing(part) or _is_missing(total) or not total:
        return MISSING
    return fmt_pct(100.0 * float(part) / float(total))


def _cell(value: Any) -> str:
    if isinstance(value, float):
        return fmt_num(value, 3)
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)
    return fmt_num(value) if not isinstance(value, str) else value


def _table(columns: list[str], rows: list[list[Any]], numeric: set[int] | None = None) -> dict:
    """Plain table structure rendered by the template (numeric columns right-aligned)."""
    if numeric is None:
        numeric = {k for k in range(len(columns))
                   if rows and all(isinstance(r[k], (int, float)) and not isinstance(r[k], bool)
                                   or _is_missing(r[k]) for r in rows)}
    return {"columns": columns, "numeric": sorted(numeric),
            "rows": [[_cell(v) for v in r] for r in rows]}


def _df_table(df: pd.DataFrame | None, columns: list[str], labels: list[str] | None = None,
              max_rows: int | None = None) -> dict | None:
    if df is None or df.empty:
        return None
    cols = [c for c in columns if c in df.columns]
    if not cols:
        return None
    labels = labels or cols
    names = [labels[columns.index(c)] for c in cols]
    sub = df[cols] if max_rows is None else df[cols].head(max_rows)
    rows = [[_py(v) for v in r] for r in sub.itertuples(index=False, name=None)]
    numeric = {k for k, c in enumerate(cols) if pd.api.types.is_numeric_dtype(df[c]) and df[c].dtype != bool}
    return _table(names, rows, numeric)


def _py(value: Any) -> Any:
    """numpy scalar -> python scalar."""
    return value.item() if hasattr(value, "item") else value


# ---------------------------------------------------------------- figures


def _figure_base(key: str) -> str:
    return key[5:] if key.startswith("plot_") else key


def embed_png(path: Path) -> str | None:
    """Read a PNG and return it as a data URI; None when unavailable."""
    path = Path(path)
    if path.suffix.lower() != ".png":
        log.warning("figure %s is not a PNG; not embedded", path)
        return None
    try:
        data = path.read_bytes()
    except OSError as exc:
        log.warning("figure %s not embedded: %s", path, exc)
        return None
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def _figures_by_section(figures: dict[str, Path] | None) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for key, path in (figures or {}).items():
        if path is None:
            continue
        uri = embed_png(Path(path))
        if uri is None:
            continue
        base = _figure_base(key)
        section = FIGURE_SECTIONS.get(base, "outras")
        caption = FIGURE_CAPTIONS.get(base, base.replace("_", " "))
        out.setdefault(section, []).append({"key": key, "uri": uri, "caption": caption,
                                            "name": Path(path).name})
    return out


# ---------------------------------------------------------------- summary helpers


def _cfg_get(cfg: Any, summary: dict, name: str, default: Any = None) -> Any:
    if cfg is not None and hasattr(cfg, name):
        return getattr(cfg, name)
    return (summary.get("config") or {}).get(name, default)


def _cfg_dict(cfg: Any, summary: dict) -> dict:
    if cfg is not None and hasattr(cfg, "to_dict"):
        try:
            return cfg.to_dict()
        except Exception as exc:  # noqa: BLE001 - report must not fail on a config quirk
            log.warning("cfg.to_dict() failed: %s", exc)
    return dict(summary.get("config") or {})


def _stat(summary: dict, metric: str, key: str = "median") -> Any:
    return ((summary.get("stats") or {}).get(metric) or {}).get(key)


def _stats_table(summary: dict, metrics: list[str]) -> dict | None:
    stats = summary.get("stats") or {}
    rows = []
    for m in metrics:
        s = stats.get(m)
        if not s:
            continue
        rows.append([STAT_LABELS.get(m, m)] + [_py(s.get(c)) for c in STAT_COLUMNS])
    if not rows:
        return None
    return _table(["Métrica", "média", "desvio", "mín", "p05", "mediana", "p95", "máx"], rows,
                  set(range(1, 8)))


def _readable(frames_df: pd.DataFrame | None) -> pd.DataFrame:
    if frames_df is None or frames_df.empty:
        return pd.DataFrame()
    if "readable" in frames_df.columns:
        return frames_df[frames_df["readable"].astype(bool)]
    return frames_df


def _count_below(df: pd.DataFrame, column: str, threshold: Any) -> int | None:
    if df.empty or column not in df.columns or _is_missing(threshold):
        return None
    return int((pd.to_numeric(df[column], errors="coerce") < float(threshold)).sum())


def _count_above(df: pd.DataFrame, column: str, threshold: Any) -> int | None:
    if df.empty or column not in df.columns or _is_missing(threshold):
        return None
    return int((pd.to_numeric(df[column], errors="coerce") > float(threshold)).sum())


def _flag_total(summary: dict, *needles: str) -> int:
    counts = summary.get("flag_counts") or {}
    return int(sum(n for f, n in counts.items() if any(s in f for s in needles)))


def _filename_of(frames_df: pd.DataFrame | None, index: Any) -> str:
    if frames_df is None or frames_df.empty or "filename" not in frames_df.columns:
        return ""
    if "index" in frames_df.columns:
        hit = frames_df.loc[frames_df["index"] == index, "filename"]
    else:
        hit = frames_df["filename"].iloc[[index]] if 0 <= int(index) < len(frames_df) else pd.Series(dtype=str)
    return str(hit.iloc[0]) if len(hit) else ""


def _frame_label(frames_df: pd.DataFrame | None, index: Any) -> str:
    name = _filename_of(frames_df, index)
    return f"{index} ({name})" if name else str(index)


def _sorted_ranges(summary: dict) -> list[tuple[int, int, int]]:
    ranges = (summary.get("graph") or {}).get("component_ranges") or []
    out = []
    for r in ranges:
        try:
            out.append((int(r[0]), int(r[1]), int(r[2])))
        except (TypeError, ValueError, IndexError):
            continue
    return sorted(out)


def component_gaps(summary: dict) -> list[tuple[int, int]]:
    """Index pairs (last frame of a component, first frame of the next) sorted along the sequence."""
    ranges = [r for r in _sorted_ranges(summary) if r[2] > 1]
    return [(a[1], b[0]) for a, b in zip(ranges, ranges[1:]) if b[0] > a[1]]


# ---------------------------------------------------------------- content builders


def _key_numbers(summary: dict) -> list[dict]:
    g = summary.get("graph") or {}
    q = summary.get("quality_counts") or {}
    items = [
        ("Arquivos listados", fmt_num(summary.get("n_files"))),
        ("Ilegíveis", fmt_num(summary.get("n_unreadable"))),
        ("Bons / revisar / ruins", " / ".join(fmt_num(q.get(k)) for k in ("good", "review", "poor"))),
        ("Variância do Laplaciano (mediana)", fmt_num(_stat(summary, "laplacian_var"))),
        ("Keypoints SIFT (mediana)", fmt_num(_stat(summary, "sift_keypoints"))),
        ("Ocupação da grade (mediana)", fmt_num(_stat(summary, "occupancy_ratio"))),
        ("Pares verificados / analisados",
         f"{fmt_num(summary.get('n_pairs_verified'))} / {fmt_num(summary.get('n_pairs'))}"),
        ("Arestas aceitas", fmt_num(summary.get("n_edges_accepted", g.get("n_edges")))),
        ("Componentes conexas", fmt_num(g.get("n_components"))),
        ("Frames isolados", fmt_num(g.get("n_isolated"))),
        ("Frames na maior componente", fmt_pct(g.get("largest_component_pct"))),
    ]
    return [{"label": a, "value": b} for a, b in items]


def build_verdict(summary: dict) -> list[str]:
    """Short data-derived diagnosis; never claims that reconstruction will succeed."""
    g = summary.get("graph") or {}
    q = summary.get("quality_counts") or {}
    n_files = summary.get("n_files") or 0
    n_readable = summary.get("n_readable")
    if not n_files:
        return ["Diagnóstico: nenhum frame foi analisado; não há dados para avaliar a sequência."]
    lines: list[str] = []
    poor, review = q.get("poor") or 0, q.get("review") or 0
    lines.append(f"Diagnóstico 2D: {fmt_num(poor)} de {fmt_num(n_files)} frames ({_frac_pct(poor, n_files)}) "
                 f"foram classificados como ruins e {fmt_num(review)} ({_frac_pct(review, n_files)}) "
                 "precisam de revisão.")
    if summary.get("n_unreadable"):
        lines.append(f"{fmt_num(summary['n_unreadable'])} arquivo(s) não puderam ser lidos.")
    n_comp = g.get("n_components")
    if n_comp is not None:
        pct = g.get("largest_component_pct")
        if n_comp == 0:
            lines.append("O grafo de conectividade não tem nós: nenhum frame legível ficou conectado.")
        elif n_comp == 1:
            lines.append(f"O grafo de conectividade forma uma única componente ({fmt_pct(pct)} dos frames legíveis), "
                         "sem cortes na cadeia de pares verificados.")
        else:
            lines.append(f"O grafo de conectividade está fragmentado em {fmt_num(n_comp)} componentes; a maior "
                         f"contém {fmt_pct(pct)} dos frames legíveis. Os cortes indicam trechos em que a "
                         "geometria de dois frames não foi verificada com as distâncias temporais testadas.")
        if g.get("n_isolated"):
            lines.append(f"{fmt_num(g['n_isolated'])} frame(s) não têm nenhuma aresta aceita (isolados).")
    if n_readable == 0:
        lines.append("Nenhum frame legível: as métricas não puderam ser calculadas.")
    severe = (poor / n_files > 0.2) or (n_comp is not None and n_comp > 1) or bool(g.get("n_isolated"))
    if severe:
        lines.append("Há indícios concretos de problemas na aquisição; veja as recomendações na seção 13.")
    else:
        lines.append("Os indicadores 2D não apontam problemas graves nesta sequência.")
    lines.append("Este é um diagnóstico das imagens e da geometria de pares; ele não garante que uma "
                 "reconstrução 3D terá sucesso.")
    return lines


def build_recommendations(summary: dict, frames_df: pd.DataFrame | None, cfg: Any) -> list[str]:
    """Acquisition advice derived from the measured values."""
    recs: list[str] = []
    readable = _readable(frames_df)
    n_read = len(readable) if not readable.empty else (summary.get("n_readable") or 0)
    g = summary.get("graph") or {}

    blurry = _count_below(readable, "sharpness_rel", _cfg_get(cfg, summary, "sharpness_review_rel"))
    if blurry is None:
        blurry = _flag_total(summary, "blur", "sharp")
    if blurry and n_read and blurry / n_read >= 0.10:
        recs.append(f"{fmt_num(blurry)} frames ({_frac_pct(blurry, n_read)} dos legíveis) estão abaixo de "
                    f"{fmt_num(_cfg_get(cfg, summary, 'sharpness_review_rel'))} x a mediana de nitidez: mova a "
                    "câmera mais devagar, aumente a iluminação ou use um tempo de exposição (shutter) menor "
                    "para reduzir o borrão de movimento.")

    low_occ = _count_below(readable, "occupancy_ratio", _cfg_get(cfg, summary, "occupancy_review_min"))
    if low_occ is None:
        low_occ = _flag_total(summary, "occupancy")
    if low_occ and n_read and low_occ / n_read >= 0.10:
        recs.append(f"{fmt_num(low_occ)} frames ({_frac_pct(low_occ, n_read)}) ocupam menos de "
                    f"{_ratio_pct(_cfg_get(cfg, summary, 'occupancy_review_min'))} das células da grade: aponte a "
                    "câmera para regiões com textura (móveis, objetos, quinas) em vez de paredes lisas, teto "
                    "ou piso uniforme.")

    low_kp = _count_below(readable, "sift_keypoints", _cfg_get(cfg, summary, "keypoints_review_min"))
    if low_kp and n_read and low_kp / n_read >= 0.10:
        recs.append(f"{fmt_num(low_kp)} frames têm menos de {fmt_num(_cfg_get(cfg, summary, 'keypoints_review_min'))} "
                    "keypoints SIFT: enquadre mais conteúdo texturizado e evite aproximar demais de superfícies lisas.")

    dark = _count_above(readable, "dark_frac", _cfg_get(cfg, summary, "dark_frac_review"))
    if dark is None:
        dark = _flag_total(summary, "dark")
    sat = _count_above(readable, "saturated_frac", _cfg_get(cfg, summary, "saturated_frac_review"))
    if sat is None:
        sat = _flag_total(summary, "saturat")
    if dark:
        recs.append(f"{fmt_num(dark)} frames têm mais de {_ratio_pct(_cfg_get(cfg, summary, 'dark_frac_review'))} "
                    "dos pixels muito escuros: aumente a iluminação do ambiente ou ajuste a exposição.")
    if sat:
        recs.append(f"{fmt_num(sat)} frames têm mais de {_ratio_pct(_cfg_get(cfg, summary, 'saturated_frac_review'))} "
                    "dos pixels saturados: reduza a exposição, evite apontar para janelas e lâmpadas, ou trave "
                    "a exposição em uma região média da cena.")

    n_comp = g.get("n_components") or 0
    if n_comp > 1:
        gaps = component_gaps(summary)
        if gaps:
            desc = "; ".join(f"entre os frames {_frame_label(frames_df, a)} e {_frame_label(frames_df, b)}"
                             for a, b in gaps[:20])
            more = f" (e mais {len(gaps) - 20})" if len(gaps) > 20 else ""
            recs.append(f"O grafo tem {fmt_num(n_comp)} componentes. A cadeia de pares verificados se rompe {desc}"
                        f"{more}. Regrave esses trechos com movimento mais lento e mais sobreposição entre "
                        "quadros, ou extraia mais frames desses intervalos.")
        elif sum(1 for r in _sorted_ranges(summary) if r[2] > 1) > 1:
            recs.append(f"O grafo tem {fmt_num(n_comp)} componentes intercaladas ao longo da sequência: revise as "
                        "faixas de índices da seção 9 e regrave os trechos com mais sobreposição.")
    isolated = g.get("isolated_nodes") or []
    if isolated:
        shown = ", ".join(_frame_label(frames_df, i) for i in isolated[:15])
        more = f" e mais {len(isolated) - 15}" if len(isolated) > 15 else ""
        recs.append(f"Frames isolados (sem aresta aceita): {shown}{more}. Verifique borrão, falta de textura "
                    "ou mudança brusca de ponto de vista nesses pontos.")

    red = summary.get("redundancy") or {}
    n_red = red.get("n_candidates") or 0
    n_consec = None
    if frames_df is not None and n_read:
        n_consec = max(n_read - 1, 0)
    if n_red and n_consec and n_red / n_consec >= 0.30:
        recs.append(f"{fmt_num(n_red)} de {fmt_num(n_consec)} pares consecutivos ({_frac_pct(n_red, n_consec)}) são "
                    "candidatos a redundância (quase sem deslocamento). Um stride maior reduziria o custo, mas "
                    "confirme rodando esta análise de novo com o novo stride que o grafo continua com uma "
                    "única componente.")

    if summary.get("n_unreadable"):
        recs.append(f"{fmt_num(summary['n_unreadable'])} arquivo(s) ilegíveis: refaça a extração desses frames ou "
                    "remova-os da pasta.")
    if not recs:
        recs.append("Nenhum problema dominante foi detectado pelos limiares atuais; revise individualmente os "
                    "frames listados na seção 10, se houver.")
    return recs


def _limitations(summary: dict, cfg: Any) -> list[dict]:
    hom = "ativado" if _cfg_get(cfg, summary, "compute_homography", True) else "desativado nesta execução"
    return [
        {"title": "Variância do Laplaciano não tem limiar universal",
         "text": "O valor depende da resolução de análise, da textura e do conteúdo da cena: uma parede lisa nítida "
                 "pode ter variância menor que uma estante borrada. Por isso a classificação usa limiares "
                 "relativos à mediana da própria sequência; um limiar absoluto só faz sentido se calibrado para a "
                 "mesma câmera, resolução e ambiente."},
        {"title": "Muitos keypoints não significam boa reconstrução",
         "text": "Texturas repetitivas (azulejos, grades, tecidos estampados) geram muitos keypoints SIFT, mas com "
                 "descritores ambíguos que produzem correspondências erradas. A contagem é um indicador "
                 "necessário, não suficiente."},
        {"title": "Ocupação da grade é distribuição no plano da imagem",
         "text": "A ocupação mede como os keypoints se espalham pelas células da imagem 2D. Ela não mede "
                 "cobertura 3D do ambiente: uma sequência inteira com boa ocupação pode olhar sempre para a "
                 "mesma parede."},
        {"title": "Razão de inliers alta com poucas correspondências não basta",
         "text": "Com poucas correspondências, o RANSAC encontra facilmente um modelo consistente por acaso. Por "
                 "isso uma aresta exige ao mesmo tempo um número mínimo de inliers e uma razão mínima de inliers."},
        {"title": "Frames semelhantes podem ser redundantes, mas úteis",
         "text": "Pares consecutivos quase idênticos acrescentam pouca baseline, mas mantêm a cadeia de "
                 "conectividade. Os candidatos a redundância são apenas sugestões; removê-los pode partir o grafo."},
        {"title": "\"Bom\" não garante reconstrução",
         "text": "A classificação good/review/poor aplica regras transparentes a métricas 2D e à geometria de "
                 "pares. Ela não avalia baseline, escala, oclusões, superfícies refletivas nem a convergência de "
                 "um SfM; nenhuma reconstrução é executada por esta ferramenta."},
        {"title": "Matriz fundamental versus homografia (diagnóstico)",
         "text": "A verificação geométrica usa a matriz fundamental (geometria epipolar geral). A homografia, "
                 f"{hom}, é apenas complementar: quando ela explica quase tantos inliers quanto F, o par pode ser "
                 "dominado por um plano ou por rotação pura, situação pouco informativa para triangulação. Ela não "
                 "entra na aceitação de arestas."},
        {"title": "Grafo limitado às distâncias temporais testadas",
         "text": "Só são comparados pares com as distâncias temporais configuradas; um frame pode casar com outro "
                 "distante (revisita) sem que isso apareça aqui. Uma componente desconectada não prova que o trecho "
                 "é irrecuperável, apenas que não foi conectado pelos pares testados."},
    ]


def _matching_by_offset(pairs_df: pd.DataFrame | None) -> dict | None:
    if pairs_df is None or pairs_df.empty or "offset" not in pairs_df.columns:
        return None
    rows = []
    for off, grp in pairs_df.groupby("offset"):
        verified = int(grp["verified"].astype(bool).sum()) if "verified" in grp else None
        accepted = int(grp["edge_accepted"].astype(bool).sum()) if "edge_accepted" in grp else None
        med = (lambda c: _py(pd.to_numeric(grp[c], errors="coerce").median()) if c in grp else None)
        rows.append([_py(off), len(grp), verified, accepted, med("lowe_matches"), med("inliers"),
                     med("inlier_ratio"), med("median_disp_px")])
    return _table(["Distância", "Pares", "Verificados", "Arestas aceitas", "Lowe (mediana)",
                   "Inliers (mediana)", "Razão (mediana)", "Desloc. px (mediana)"], rows, set(range(8)))


def _status_table(pairs_df: pd.DataFrame | None) -> dict | None:
    if pairs_df is None or pairs_df.empty or "status" not in pairs_df.columns:
        return None
    counts = pairs_df["status"].value_counts()
    return _table(["Status", "Pares"], [[str(k), int(v)] for k, v in counts.items()], {1})


def _homography_note(pairs_df: pd.DataFrame | None) -> str | None:
    if pairs_df is None or pairs_df.empty or "homography_inliers" not in pairs_df.columns:
        return None
    sub = pairs_df[(pairs_df["homography_inliers"] >= 0)]
    if "inliers" not in sub.columns or sub.empty:
        return None
    sub = sub[sub["inliers"] > 0]
    if sub.empty:
        return None
    ratio = (sub["homography_inliers"] / sub["inliers"]).median()
    return (f"Diagnóstico complementar: em {fmt_num(len(sub))} pares com homografia calculada, a mediana de "
            f"inliers da homografia dividida pelos inliers da matriz fundamental é {fmt_num(_py(ratio))}.")


def _component_table(summary: dict, frames_df: pd.DataFrame | None) -> dict | None:
    ranges = (summary.get("graph") or {}).get("component_ranges") or []
    if not ranges:
        return None
    rows = []
    for k, r in enumerate(ranges):
        try:
            first, last, size = int(r[0]), int(r[1]), int(r[2])
        except (TypeError, ValueError, IndexError):
            continue
        rows.append([k, size, first, last, _filename_of(frames_df, first), _filename_of(frames_df, last)])
    return _table(["Componente", "Frames", "Primeiro índice", "Último índice", "Primeiro arquivo",
                   "Último arquivo"], rows, {0, 1, 2, 3})


def _degree_table(summary: dict) -> dict | None:
    hist = (summary.get("graph") or {}).get("degree_histogram") or {}
    if not hist:
        return None
    rows = sorted(([int(k), int(v)] for k, v in hist.items()), key=lambda r: r[0])
    return _table(["Grau", "Frames"], rows, {0, 1})


def _review_frames(frames_df: pd.DataFrame | None, max_rows: int) -> tuple[dict | None, int]:
    if frames_df is None or frames_df.empty or "quality" not in frames_df.columns:
        return None, 0
    sub = frames_df[frames_df["quality"] != "good"].copy()
    if sub.empty:
        return None, 0
    sub["_order"] = sub["quality"].map({"poor": 0, "review": 1}).fillna(2)
    sort_cols = ["_order"] + (["index"] if "index" in sub.columns else [])
    sub = sub.sort_values(sort_cols)
    cols = ["index", "filename", "quality", "flags", "sharpness_rel", "sift_keypoints", "occupancy_ratio",
            "degree", "component_id", "error"]
    labels = ["Índice", "Arquivo", "Qualidade", "Flags", "Nitidez rel.", "Keypoints", "Ocupação", "Grau",
              "Componente", "Erro"]
    return _df_table(sub, cols, labels, max_rows), len(sub)


def _unreadable_table(summary: dict, frames_df: pd.DataFrame | None) -> dict | None:
    items = summary.get("unreadable") or []
    if items:
        return _table(["Arquivo", "Erro"], [[str(u.get("filename", "")), str(u.get("error", ""))] for u in items],
                      set())
    if frames_df is not None and not frames_df.empty and "readable" in frames_df.columns:
        bad = frames_df[~frames_df["readable"].astype(bool)]
        return _df_table(bad, ["filename", "error"], ["Arquivo", "Erro"])
    return None


def _rules_table() -> dict | None:
    try:
        from frame_analyzer.classify import CLASSIFICATION_RULES
    except ImportError:
        log.info("classify.CLASSIFICATION_RULES not available; rules table skipped")
        return None
    rows = [[str(r.get("flag", "")), str(r.get("level", "")), str(r.get("description", ""))]
            for r in CLASSIFICATION_RULES or []]
    return _table(["Flag", "Nível", "Regra"], rows, set()) if rows else None


def _dict_table(d: dict, key_label: str, value_label: str) -> dict | None:
    if not d:
        return None
    rows = [[str(k), v if isinstance(v, (int, float, str)) and not isinstance(v, bool) else _cell(v)]
            for k, v in d.items()]
    return _table([key_label, value_label], rows, set())


def _graph_link(out_dir: Path, graph_html: Path | None) -> str | None:
    if graph_html is None:
        return None
    graph_html = Path(graph_html)
    if not graph_html.exists():
        log.warning("interactive graph %s not found; link omitted", graph_html)
        return None
    try:
        return Path(os.path.relpath(graph_html, out_dir)).as_posix()
    except ValueError:
        return graph_html.name


def _resolution(summary: dict, frames_df: pd.DataFrame | None) -> dict:
    res = summary.get("resolution") or {}
    unique = res.get("unique") or []
    table = _table(["Largura", "Altura", "Frames"],
                   [[_py(u.get("width")), _py(u.get("height")), _py(u.get("count"))] for u in unique],
                   {0, 1, 2}) if unique else None
    analysis = None
    readable = _readable(frames_df)
    if not readable.empty and {"analysis_width", "analysis_height"} <= set(readable.columns):
        grp = readable.groupby(["analysis_width", "analysis_height"]).size().reset_index(name="count")
        analysis = _table(["Largura", "Altura", "Frames"],
                          [[_py(w), _py(h), _py(c)] for w, h, c in grp.itertuples(index=False, name=None)],
                          {0, 1, 2})
    return {"consistent": res.get("consistent"), "table": table, "analysis": analysis,
            "width_min": res.get("width_min"), "width_max": res.get("width_max"),
            "height_min": res.get("height_min"), "height_max": res.get("height_max")}


# ---------------------------------------------------------------- command


def build_command(cfg: Any) -> str:
    """Reconstruct the analyze_frames.py command line that reproduces cfg (non-default options only)."""
    from frame_analyzer.config import AnalyzerConfig

    default = AnalyzerConfig(input_dir=Path("."))
    parts = ["python", "analyze_frames.py", "--input", shlex.quote(str(cfg.input_dir)),
             "--output", shlex.quote(str(cfg.output_dir))]
    if tuple(cfg.extensions) != tuple(default.extensions):
        parts += ["--extensions", ",".join(e.lstrip(".") for e in cfg.extensions)]
    if (cfg.grid_rows, cfg.grid_cols) != (default.grid_rows, default.grid_cols):
        parts += ["--grid", f"{cfg.grid_rows}x{cfg.grid_cols}"]
    if tuple(cfg.temporal_offsets) != tuple(default.temporal_offsets):
        parts += ["--offsets", ",".join(str(o) for o in cfg.temporal_offsets)]
    for name, flag in CLI_FLAGS.items():
        value = getattr(cfg, name, None)
        if value is not None and value != getattr(default, name):
            parts += [flag, shlex.quote(str(value))]
    for name, flag in NEGATED_FLAGS.items():
        if not getattr(cfg, name, True):
            parts.append(flag)
    return " ".join(parts)


def _safe_command(cfg: Any) -> str | None:
    try:
        return build_command(cfg)
    except (AttributeError, TypeError) as exc:
        log.warning("could not rebuild the command line: %s", exc)
        return None


# ---------------------------------------------------------------- public API


def build_context(out_dir: Path, summary: dict, frames_df: pd.DataFrame | None, pairs_df: pd.DataFrame | None,
                  edges_df: pd.DataFrame | None, redundant_df: pd.DataFrame | None,
                  figures: dict[str, Path] | None, cfg: Any, graph_html: Path | None) -> dict:
    """All values the template needs, derived from the summary and the dataframes."""
    summary = summary or {}
    g = summary.get("graph") or {}
    max_rows = int(_cfg_get(cfg, summary, "review_table_max_rows", 200) or 200)
    review_table, n_review = _review_frames(frames_df, max_rows)
    red_max = max_rows
    redundant_table = _df_table(redundant_df, ["i", "j", "filename_i", "filename_j", "inliers", "inlier_ratio",
                                               "median_disp_px", "reason"],
                                ["i", "j", "Arquivo i", "Arquivo j", "Inliers", "Razão", "Desloc. px", "Motivo"],
                                red_max)
    n_redundant = 0 if redundant_df is None else len(redundant_df)
    edges_table = _df_table(edges_df, ["source", "target", "offset", "inliers", "inlier_ratio"],
                            ["Origem", "Destino", "Distância", "Inliers", "Razão"], 30)
    criteria = None
    if edges_df is not None and not edges_df.empty and "criteria" in edges_df.columns:
        criteria = str(edges_df["criteria"].iloc[0])
    if criteria is None:
        criteria = (f"verified=fundamental; inliers>={_cfg_get(cfg, summary, 'edge_min_inliers')}; "
                    f"inlier_ratio>={_cfg_get(cfg, summary, 'edge_min_inlier_ratio')}")
    flag_counts = summary.get("flag_counts") or {}
    flags_table = _table(["Flag", "Frames"], [[str(k), int(v)] for k, v in
                                              sorted(flag_counts.items(), key=lambda kv: -kv[1])], {1}) \
        if flag_counts else None
    timings = summary.get("timings") or {}
    timings_table = _table(["Etapa", "Tempo (s)"], [[str(k), _py(v)] for k, v in timings.items()], {1}) \
        if timings else None
    thresholds = summary.get("thresholds") or {}
    reference_url = _cfg_get(cfg, summary, "reference_url")
    return {
        "title": "Relatório de qualidade dos frames",
        "sections": SECTIONS,
        "appendix": APPENDIX,
        "summary": summary,
        "g": g,
        "key_numbers": _key_numbers(summary),
        "verdict": build_verdict(summary),
        "input_dir": summary.get("input_dir", str(_cfg_get(cfg, summary, "input_dir", ""))),
        "output_dir": summary.get("output_dir", str(out_dir)),
        "n_files": summary.get("n_files"),
        "n_readable": summary.get("n_readable"),
        "n_unreadable": summary.get("n_unreadable"),
        "quality_counts": summary.get("quality_counts") or {},
        "flags_table": flags_table,
        "rules_table": _rules_table(),
        "unreadable_table": _unreadable_table(summary, frames_df),
        "stride": _cfg_get(cfg, summary, "stride"),
        "limit": _cfg_get(cfg, summary, "limit"),
        "analysis_max_side": _cfg_get(cfg, summary, "analysis_max_side"),
        "grid": f"{_cfg_get(cfg, summary, 'grid_rows')}x{_cfg_get(cfg, summary, 'grid_cols')}",
        "resolution": _resolution(summary, frames_df),
        "sharpness_table": _stats_table(summary, ["laplacian_var"]),
        "sharpness_median": thresholds.get("sharpness_median", _stat(summary, "laplacian_var")),
        "sharpness_poor_abs": thresholds.get("sharpness_poor_abs"),
        "sharpness_review_abs": thresholds.get("sharpness_review_abs"),
        "sharpness_poor_rel": _cfg_get(cfg, summary, "sharpness_poor_rel"),
        "sharpness_review_rel": _cfg_get(cfg, summary, "sharpness_review_rel"),
        "sharpness_abs_min": _cfg_get(cfg, summary, "sharpness_abs_min"),
        "sift_table": _stats_table(summary, ["sift_keypoints", "sift_density_per_mp"]),
        "exposure_table": _stats_table(summary, ["brightness_mean", "dark_frac", "saturated_frac"]),
        "occupancy_table": _stats_table(summary, ["occupancy_ratio"]),
        "matching_table": _stats_table(summary, ["inliers", "inlier_ratio"]),
        "offset_table": _matching_by_offset(pairs_df),
        "status_table": _status_table(pairs_df),
        "homography_note": _homography_note(pairs_df),
        "offsets_used": summary.get("offsets_used") or list(_cfg_get(cfg, summary, "temporal_offsets", []) or []),
        "n_pairs": summary.get("n_pairs"),
        "n_pairs_verified": summary.get("n_pairs_verified"),
        "n_edges_accepted": summary.get("n_edges_accepted", g.get("n_edges")),
        "edge_criteria": criteria,
        "edges_table": edges_table,
        "n_edges_rows": 0 if edges_df is None else len(edges_df),
        "graph_link": _graph_link(out_dir, graph_html),
        "component_table": _component_table(summary, frames_df),
        "degree_table": _degree_table(summary),
        "isolated": [_frame_label(frames_df, i) for i in (g.get("isolated_nodes") or [])],
        "review_table": review_table,
        "n_review": n_review,
        "max_rows": max_rows,
        "redundant_table": redundant_table,
        "n_redundant": n_redundant,
        "redundancy": summary.get("redundancy") or {},
        "reference_url": reference_url,
        "limitations": _limitations(summary, cfg),
        "recommendations": build_recommendations(summary, frames_df, cfg),
        "figures": _figures_by_section(figures),
        "timings_table": timings_table,
        "warnings": list(summary.get("warnings") or []),
        "config_table": _dict_table(_cfg_dict(cfg, summary), "Parâmetro", "Valor"),
        "thresholds_table": _dict_table(thresholds, "Limiar", "Valor"),
        "command": _safe_command(cfg),
        "fmt": fmt_num,
        "pct": fmt_pct,
        "frac_pct": _frac_pct,
    }


def render_report(context: dict) -> str:
    """Render the Jinja2 template (autoescape on)."""
    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)), autoescape=True,
                      trim_blocks=True, lstrip_blocks=True)
    return env.get_template(TEMPLATE_NAME).render(**context)


def write_report(out_dir: Path, summary: dict, frames_df: pd.DataFrame | None, pairs_df: pd.DataFrame | None,
                 edges_df: pd.DataFrame | None, redundant_df: pd.DataFrame | None,
                 figures: dict[str, Path] | None, cfg: Any, graph_html: Path | None) -> Path:
    """Write quality_report.html (self-contained: inline CSS, figures as base64 PNG data URIs)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    context = build_context(out_dir, summary, frames_df, pairs_df, edges_df, redundant_df, figures, cfg, graph_html)
    path = out_dir / REPORT_NAME
    path.write_text(render_report(context), encoding="utf-8")
    log.info("report written to %s", path)
    return path


def write_output_readme(out_dir: Path, summary: dict, cfg: Any) -> Path:
    """Write README.md in the output directory describing every generated file."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summary or {}
    g = summary.get("graph") or {}
    q = summary.get("quality_counts") or {}
    command = _safe_command(cfg) or "python analyze_frames.py --input <frames> --output <saida>"
    lines = [
        "# Diagnóstico de qualidade dos frames",
        "",
        "Saída do `analyze_frames.py`: diagnóstico 2D dos frames antes de uma reconstrução 3D. Nenhuma "
        "reconstrução, COLMAP ou reextração de frames foi executada.",
        "",
        "## Comando",
        "",
        "```bash",
        command,
        "```",
        "",
        "## Resumo desta execução",
        "",
        f"- Entrada: `{summary.get('input_dir', _cfg_get(cfg, summary, 'input_dir', ''))}`",
        f"- Início: {summary.get('started_at', MISSING)}; fim: {summary.get('finished_at', MISSING)}; "
        f"duração: {fmt_num(summary.get('elapsed_s'))} s",
        f"- Frames: {fmt_num(summary.get('n_files'))} listados, {fmt_num(summary.get('n_unreadable'))} ilegíveis; "
        f"bons/revisar/ruins = {fmt_num(q.get('good'))}/{fmt_num(q.get('review'))}/{fmt_num(q.get('poor'))}",
        f"- Pares: {fmt_num(summary.get('n_pairs'))} analisados, {fmt_num(summary.get('n_pairs_verified'))} "
        f"verificados, {fmt_num(summary.get('n_edges_accepted', g.get('n_edges')))} arestas aceitas",
        f"- Grafo: {fmt_num(g.get('n_components'))} componentes, {fmt_num(g.get('n_isolated'))} isolados, "
        f"maior componente com {fmt_pct(g.get('largest_component_pct'))} dos frames legíveis",
        f"- Avisos: {len(summary.get('warnings') or [])}",
        "",
        "## Arquivos",
        "",
        "| Arquivo | Conteúdo |",
        "|---|---|",
        f"| `{REPORT_NAME}` | relatório autocontido (abre offline): resumo, métricas, figuras, grafo, frames "
        "a revisar, limitações e recomendações |",
        "| `frame_quality.csv` | uma linha por arquivo listado: resolução, nitidez (variância do Laplaciano e "
        "relativa à mediana), keypoints SIFT, densidade, ocupação da grade, brilho, grau no grafo, componente, "
        "flags e qualidade (good/review/poor) |",
        "| `pair_quality.csv` | uma linha por par analisado (i, j = i + distância): correspondências brutas, após "
        "Lowe, inliers da matriz fundamental, razão de inliers, status, homografia (diagnóstico), deslocamento "
        "mediano e se virou aresta |",
        "| `connectivity_edges.csv` | arestas aceitas do grafo e o critério aplicado |",
        "| `redundant_pairs.csv` | pares consecutivos candidatos a redundância (sugestão, não remoção) |",
        "| `summary.json` | todos os números agregados, limiares, tempos por etapa, avisos e a configuração |",
        f"| `{GRAPH_HTML_NAME}` | grafo interativo (passe o mouse sobre um nó para ver índice, arquivo, grau e "
        "componente); só existe se o grafo interativo estiver ativado |",
        "| `grid_counts.npy` | array (N, linhas, colunas) com keypoints por célula da grade, por frame |",
        "| `*.png` | gráficos, heatmaps e o grafo estático (`connectivity_graph.png`) embutidos no relatório; "
        "só dados, nunca fotos |",
        "",
        "## Como interpretar",
        "",
        "- `quality`: `poor` = ao menos uma regra de rejeição (ilegível, muito borrado, poucos keypoints); "
        "`review` = ao menos uma regra de atenção (nitidez abaixo do limiar de revisão, ocupação baixa, "
        "exposição, grau baixo, isolado); `good` = nenhuma regra disparou. As regras exatas estão em `flags` "
        "e na seção 2 do relatório. `good` não garante que a reconstrução vá funcionar.",
        "- Nitidez é relativa à mediana da própria sequência: a variância do Laplaciano depende da resolução e da "
        "textura da cena, não existe limiar universal.",
        "- Ocupação da grade mede a distribuição dos keypoints no plano da imagem, não a cobertura 3D.",
        "- Uma aresta exige matriz fundamental estimada com inliers e razão de inliers acima dos limiares "
        f"(`edge_min_inliers` = {_cfg_get(cfg, summary, 'edge_min_inliers')}, `edge_min_inlier_ratio` = "
        f"{_cfg_get(cfg, summary, 'edge_min_inlier_ratio')}). Frames consecutivos não ganham aresta automaticamente.",
        "- Mais de uma componente conexa: a cadeia de pares verificados se rompe entre elas; a seção 13 do "
        "relatório lista os intervalos de índices.",
        "- Índices (`index`, `i`, `j`) são posições na lista analisada (após `--limit` e `--stride`), não números "
        "de frame do vídeo; use `filename` para localizar o arquivo.",
        "",
    ]
    path = out_dir / README_NAME
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("output README written to %s", path)
    return path

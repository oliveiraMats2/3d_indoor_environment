"""Connectivity graph of frames built only from geometrically verified, accepted pairs."""
from __future__ import annotations

import logging
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from frame_analyzer.config import AnalyzerConfig  # noqa: E402

log = logging.getLogger(__name__)

# Green family for connected components (cycled by component id; labels carry identity too).
COMPONENT_GREENS = ("#1f7a1f", "#2fa84f", "#0b5d3b", "#6aa84f")
ISOLATED_ORANGE = "#eb6834"
ISOLATED_EDGE = "#8a3a14"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
BREAK_COLOR = "#898781"

SPRING_MAX_NODES = 40          # below this, a force layout is readable
EDGE_LABEL_MAX_EDGES = 60      # inlier labels on edges only for small graphs
EDGE_COLUMNS = ["source", "target", "offset", "inliers", "inlier_ratio", "criteria"]


def build_graph(frame_indices: list[int], pairs_df: pd.DataFrame) -> nx.Graph:
    """Nodes = readable frame indices; edges ONLY for rows with edge_accepted True."""
    G = nx.Graph()
    G.add_nodes_from(int(i) for i in frame_indices)
    if pairs_df is None or pairs_df.empty or "edge_accepted" not in pairs_df.columns:
        return G
    accepted = pairs_df[pairs_df["edge_accepted"].fillna(False).astype(bool)]
    for row in accepted.itertuples(index=False):
        i, j = int(row.i), int(row.j)
        if i not in G or j not in G:
            log.warning("accepted pair (%d, %d) references a frame outside the graph; skipped", i, j)
            continue
        offset = int(getattr(row, "offset", j - i))
        G.add_edge(i, j, inliers=int(row.inliers), inlier_ratio=float(row.inlier_ratio), offset=offset)
    return G


def edges_dataframe(G: nx.Graph, cfg: AnalyzerConfig) -> pd.DataFrame:
    """Accepted edges as a table with the acceptance criteria spelled out."""
    criteria = (f"verified=fundamental; inliers>={cfg.edge_min_inliers}; "
                f"inlier_ratio>={cfg.edge_min_inlier_ratio}")
    rows = []
    for u, v, d in G.edges(data=True):
        s, t = (u, v) if u < v else (v, u)
        rows.append({"source": int(s), "target": int(t), "offset": int(d.get("offset", t - s)),
                     "inliers": int(d.get("inliers", 0)), "inlier_ratio": float(d.get("inlier_ratio", float("nan"))),
                     "criteria": criteria})
    df = pd.DataFrame(rows, columns=EDGE_COLUMNS)
    return df.sort_values(["source", "target"]).reset_index(drop=True) if not df.empty else df


def graph_stats(G: nx.Graph) -> dict:
    """Component, isolation and degree statistics of the connectivity graph."""
    comps = [sorted(int(n) for n in c) for c in nx.connected_components(G)]
    comps.sort(key=lambda c: (-len(c), c[0]))
    degree = {int(n): int(d) for n, d in G.degree()}
    hist: dict[int, int] = {}
    for d in degree.values():
        hist[d] = hist.get(d, 0) + 1
    n_nodes = G.number_of_nodes()
    largest = len(comps[0]) if comps else 0
    return {
        "n_nodes": n_nodes,
        "n_edges": G.number_of_edges(),
        "n_components": len(comps),
        "n_isolated": sum(1 for d in degree.values() if d == 0),
        "isolated_nodes": sorted(n for n, d in degree.items() if d == 0),
        "largest_component_size": largest,
        "largest_component_pct": 100.0 * largest / n_nodes if n_nodes else 0.0,
        "components": comps,
        "degree": degree,
        "degree_histogram": dict(sorted(hist.items())),
        "component_of": {n: cid for cid, c in enumerate(comps) for n in c},
    }


# ----------------------------------------------------------------------------- layout

def _serpentine_cols(n_nodes: int) -> int:
    return max(10, min(60, int(math.ceil(math.sqrt(n_nodes * 2.0)))))


def compute_layout(G: nx.Graph, seed: int) -> tuple[dict[int, tuple[float, float]], str]:
    """Deterministic layout: spring(seed) for small graphs, sequence serpentine otherwise."""
    nodes = sorted(G.nodes())
    if not nodes:
        return {}, "empty"
    if len(nodes) <= SPRING_MAX_NODES:
        return _packed_spring_layout(G, seed), "spring"
    first = nodes[0]
    span = nodes[-1] - first + 1
    cols = _serpentine_cols(span)
    pos = {}
    for n in nodes:
        k = n - first
        row, col = divmod(k, cols)
        x = col if row % 2 == 0 else cols - 1 - col
        pos[int(n)] = (float(x), -float(row) * 1.6)
    return pos, "serpentine"


def _packed_spring_layout(G: nx.Graph, seed: int, per_row: int = 3) -> dict[int, tuple[float, float]]:
    """spring_layout(seed) per component, components packed in a grid; isolated frames in a bottom row."""
    comps = sorted((sorted(c) for c in nx.connected_components(G) if len(c) > 1), key=lambda c: (-len(c), c[0]))
    isolated = sorted(n for n in G.nodes() if G.degree(n) == 0)
    pos: dict[int, tuple[float, float]] = {}
    cell = 2.6
    for k, comp in enumerate(comps):
        sub = nx.spring_layout(G.subgraph(comp), seed=seed, k=1.2 / math.sqrt(len(comp)), scale=1.0)
        r, c = divmod(k, per_row)
        for n, (x, y) in sub.items():
            pos[int(n)] = (float(x) + c * cell, float(y) - r * cell)
    n_rows = -(-len(comps) // per_row) if comps else 0
    y0 = -n_rows * cell + (cell / 2 if comps else 0.0)
    for k, n in enumerate(isolated):
        pos[int(n)] = (-1.0 + (k % 10) * 0.6, y0 - (k // 10) * 0.6)
    return pos


def _edge_curve(p0: tuple[float, float], p1: tuple[float, float], offset: int, layout: str,
                samples: int = 12) -> np.ndarray:
    """Straight segment for offset 1 (or spring layout), quadratic arc for longer temporal jumps."""
    a, b = np.asarray(p0), np.asarray(p1)
    if layout != "serpentine" or offset <= 1:
        return np.vstack([a, b])
    mid = (a + b) / 2.0
    d = b - a
    length = float(np.hypot(*d)) or 1.0
    normal = np.array([-d[1], d[0]]) / length
    height = min(0.15 + 0.08 * offset, 0.7)
    ctrl = mid + normal * height
    t = np.linspace(0.0, 1.0, samples)[:, None]
    return (1 - t) ** 2 * a + 2 * (1 - t) * t * ctrl + t ** 2 * b


def _label_step(n_nodes: int) -> int:
    if n_nodes <= 100:
        return 1
    if n_nodes <= 400:
        return 5
    return 10


def _component_color(cid: int) -> str:
    return COMPONENT_GREENS[cid % len(COMPONENT_GREENS)]


def _break_points(stats: dict) -> list[tuple[int, int]]:
    """Consecutive readable frames that fall in different components (sequence gaps)."""
    comp_of = stats.get("component_of", {})
    nodes = sorted(comp_of)
    return [(a, b) for a, b in zip(nodes, nodes[1:]) if comp_of[a] != comp_of[b]]


# ----------------------------------------------------------------------------- PNG

def _placeholder_png(path: Path, message: str) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=SURFACE)
    ax.set_axis_off()
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=14, color=INK_SECONDARY, transform=ax.transAxes)
    ax.set_title("Grafo de conectividade", color=INK, loc="left")
    fig.savefig(path, dpi=120, facecolor=SURFACE)
    plt.close(fig)
    return path


def _figure_size(pos: dict, layout: str) -> tuple[float, float]:
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    if layout == "spring":
        w = min(max(7.0, (max(xs) - min(xs) + 1.0) * 2.4), 18.0)
        return w, min(max(4.5, (max(ys) - min(ys) + 1.0) * 2.4 + 1.2), 16.0)
    w = min(max(10.0, (max(xs) - min(xs) + 2) * 0.32), 30.0)
    h = min(max(5.0, (max(ys) - min(ys) + 2) * 0.32 + 2.0), 40.0)
    return w, h


def draw_graph_png(G: nx.Graph, stats: dict, path: Path, seed: int) -> Path:
    """Static connectivity graph: green connected frames/edges, orange isolated frames."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pos, layout = compute_layout(G, seed)
    if not pos:
        return _placeholder_png(path, "sem dados: nenhum frame legível")
    comp_of = stats["component_of"]
    degree = stats["degree"]
    n = len(pos)
    fig, ax = plt.subplots(figsize=_figure_size(pos, layout), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    segs, colors = [], []
    for u, v, d in G.edges(data=True):
        segs.append(_edge_curve(pos[u], pos[v], int(d.get("offset", abs(v - u))), layout))
        colors.append(_component_color(comp_of[u]))
    lw = 1.6 if n <= 200 else 0.9
    if segs:
        ax.add_collection(LineCollection(segs, colors=colors, linewidths=lw, alpha=0.85, zorder=1))

    if layout == "serpentine":
        for a, b in _break_points(stats):
            (xa, ya), (xb, yb) = pos[a], pos[b]
            ax.plot([xa, xb], [ya, yb], linestyle=(0, (2, 2)), color=BREAK_COLOR, linewidth=0.8, zorder=0)

    size = 160 if n <= 60 else (60 if n <= 300 else 22)
    connected = [x for x in pos if degree.get(x, 0) > 0]
    isolated = [x for x in pos if degree.get(x, 0) == 0]
    if connected:
        ax.scatter([pos[x][0] for x in connected], [pos[x][1] for x in connected], s=size,
                   c=[_component_color(comp_of[x]) for x in connected], edgecolors=SURFACE, linewidths=0.8, zorder=2)
    if isolated:
        ax.scatter([pos[x][0] for x in isolated], [pos[x][1] for x in isolated], s=size * 1.3,
                   c=ISOLATED_ORANGE, edgecolors=ISOLATED_EDGE, linewidths=0.9, zorder=3)

    _draw_node_labels(ax, pos, stats, layout, n)
    if G.number_of_edges() <= EDGE_LABEL_MAX_EDGES and n <= 60:
        for u, v, d in G.edges(data=True):
            mx, my = (pos[u][0] + pos[v][0]) / 2, (pos[u][1] + pos[v][1]) / 2
            ax.text(mx, my, str(d.get("inliers", "")), fontsize=7, color=INK_SECONDARY, ha="center", va="center",
                    zorder=4, bbox={"boxstyle": "round,pad=0.15", "fc": SURFACE, "ec": "none", "alpha": 0.85})
    _draw_component_tags(ax, pos, stats, layout)
    _draw_legend(ax, stats, layout)

    ax.set_title(
        f"Grafo de conectividade — {stats['n_nodes']} frames, {stats['n_edges']} arestas aceitas, "
        f"{stats['n_components']} componentes, {stats['n_isolated']} isolados "
        f"(maior componente: {stats['largest_component_pct']:.1f}%)",
        loc="left", color=INK, fontsize=11)
    ax.set_axis_off()
    ax.autoscale_view()
    ax.margins(0.03 if layout == "serpentine" else 0.1)
    if layout == "serpentine":
        ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor=SURFACE)
    plt.close(fig)
    log.info("connectivity graph PNG written to %s (%s layout)", path, layout)
    return path


def _draw_node_labels(ax, pos: dict, stats: dict, layout: str, n: int) -> None:
    step = _label_step(n)
    starts = {c[0] for c in stats["components"] if len(c) > 1}
    isolated = set(stats["isolated_nodes"]) if len(stats["isolated_nodes"]) <= 200 else set()
    fs = 8 if n <= 100 else 6
    dy = 0.0 if layout == "spring" else -0.42
    for node, (x, y) in pos.items():
        if node % step and node not in starts and node not in isolated:
            continue
        inside = layout == "spring" and n <= 60
        ax.text(x, y + dy, str(node), fontsize=fs, ha="center", va="center" if inside else "top",
                color="#ffffff" if inside and node not in isolated else INK, zorder=5)


def _draw_component_tags(ax, pos: dict, stats: dict, layout: str) -> None:
    """Label each multi-frame component (C0 = largest) near its first frame."""
    multi = [(cid, c) for cid, c in enumerate(stats["components"]) if len(c) > 1]
    if len(multi) > 40:
        multi = multi[:40]
    for cid, comp in multi:
        if layout == "serpentine":
            x, y = pos[comp[0]][0], pos[comp[0]][1] + 0.45
        else:
            x, y = float(np.mean([pos[n][0] for n in comp])), max(pos[n][1] for n in comp) + 0.12
        ax.text(x, y, f"C{cid} ({len(comp)})", fontsize=7,
                fontweight="bold", color=_component_color(cid), ha="center", va="bottom", zorder=6,
                bbox={"boxstyle": "round,pad=0.2", "fc": SURFACE, "ec": _component_color(cid), "lw": 0.8})


def _draw_legend(ax, stats: dict, layout: str) -> None:
    handles = [
        Line2D([], [], marker="o", linestyle="-", color=COMPONENT_GREENS[0], markersize=8,
               label="frame conectado / aresta aceita (verde)"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor=ISOLATED_ORANGE, markeredgecolor=ISOLATED_EDGE,
               markersize=9, label="frame isolado, sem aresta aceita (laranja)"),
    ]
    if stats["n_components"] > 1:
        handles.append(Line2D([], [], marker="s", linestyle="", color=COMPONENT_GREENS[1], markersize=8,
                              label="tons de verde + rótulo Ck (n) = componentes distintos (C0 = maior)"))
    if layout == "serpentine":
        handles.append(Line2D([], [], linestyle=(0, (2, 2)), color=BREAK_COLOR,
                              label="quebra: frames vizinhos em componentes diferentes"))
        handles.append(Line2D([], [], linestyle="", label="ordem temporal em serpentina; arcos = saltos > 1"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.01), ncol=2, frameon=False, fontsize=8)


# ----------------------------------------------------------------------------- HTML

def write_graph_html(G: nx.Graph, stats: dict, frames_df: pd.DataFrame, path: Path, seed: int) -> Path:
    """Self-contained interactive plotly graph (plotly.js inlined) with the same layout and colors."""
    import plotly.graph_objects as go

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pos, layout = compute_layout(G, seed)
    comp_of = stats.get("component_of", {})
    degree = stats.get("degree", {})
    names = _filenames(frames_df)
    fig = go.Figure()

    by_color: dict[str, tuple[list, list, list]] = {}
    for u, v, d in G.edges(data=True):
        curve = _edge_curve(pos[u], pos[v], int(d.get("offset", abs(v - u))), layout)
        xs, ys, _ = by_color.setdefault(_component_color(comp_of[u]), ([], [], []))
        xs.extend(curve[:, 0].tolist() + [None])
        ys.extend(curve[:, 1].tolist() + [None])
    for i, (color, (xs, ys, _)) in enumerate(by_color.items()):
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line={"color": color, "width": 1.2}, hoverinfo="skip",
                                 name="aresta aceita", legendgroup="edges", showlegend=i == 0, opacity=0.85))

    if layout == "serpentine":
        bx, by = [], []
        for a, b in _break_points(stats):
            bx += [pos[a][0], pos[b][0], None]
            by += [pos[a][1], pos[b][1], None]
        if bx:
            fig.add_trace(go.Scatter(x=bx, y=by, mode="lines", line={"color": BREAK_COLOR, "width": 1, "dash": "dot"},
                                     hoverinfo="skip", name="quebra entre componentes"))

    edge_info = _edge_hover(G)
    for label, sel, style in (
        ("frame conectado (verde; tom = componente)", [x for x in pos if degree.get(x, 0) > 0], "connected"),
        ("frame isolado (laranja)", [x for x in pos if degree.get(x, 0) == 0], "isolated"),
    ):
        if not sel:
            continue
        colors = ([_component_color(comp_of[x]) for x in sel] if style == "connected" else ISOLATED_ORANGE)
        text = [f"frame {x}<br>{names.get(x, '')}<br>grau: {degree.get(x, 0)}<br>"
                f"componente: C{comp_of.get(x, -1)} ({len(stats['components'][comp_of[x]])} frames)"
                f"{edge_info.get(x, '')}" for x in sel]
        fig.add_trace(go.Scatter(
            x=[pos[x][0] for x in sel], y=[pos[x][1] for x in sel], mode="markers", name=label,
            marker={"color": colors, "size": 9 if len(pos) > 300 else 13,
                    "line": {"color": ISOLATED_EDGE if style == "isolated" else SURFACE, "width": 1}},
            text=text, hovertemplate="%{text}<extra></extra>"))

    fig.update_layout(
        title=(f"Grafo de conectividade — {stats.get('n_nodes', 0)} frames, {stats.get('n_edges', 0)} arestas, "
               f"{stats.get('n_components', 0)} componentes, {stats.get('n_isolated', 0)} isolados"),
        plot_bgcolor=SURFACE, paper_bgcolor=SURFACE, font={"color": INK},
        xaxis={"visible": False}, yaxis={"visible": False, "scaleanchor": "x" if layout == "serpentine" else None},
        legend={"orientation": "h", "y": -0.05}, hovermode="closest", height=max(600, min(2400, 30 * _rows(pos))),
    )
    if not pos:
        fig.add_annotation(text="sem dados: nenhum frame legível", showarrow=False, x=0.5, y=0.5,
                           xref="paper", yref="paper")
    fig.write_html(str(path), include_plotlyjs=True, full_html=True)
    log.info("interactive connectivity graph written to %s", path)
    return path


def _rows(pos: dict) -> int:
    return len({round(p[1], 3) for p in pos.values()}) or 1


def _filenames(frames_df: pd.DataFrame | None) -> dict[int, str]:
    if frames_df is None or frames_df.empty or "filename" not in frames_df.columns:
        return {}
    idx = frames_df["index"] if "index" in frames_df.columns else frames_df.index
    return {int(i): str(f) for i, f in zip(idx, frames_df["filename"])}


def _edge_hover(G: nx.Graph, max_edges: int = 8) -> dict[int, str]:
    out = {}
    for n in G.nodes():
        items = sorted(G[n].items())[:max_edges]
        if items:
            out[int(n)] = "<br>arestas: " + ", ".join(f"{m} ({d.get('inliers', '?')} inl.)" for m, d in items)
    return out

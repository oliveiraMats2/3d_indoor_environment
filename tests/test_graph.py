"""Tests of the connectivity graph built from synthetic pair tables."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from frame_analyzer.config import AnalyzerConfig
from frame_analyzer.graph import build_graph, draw_graph_png, edges_dataframe, graph_stats, write_graph_html


def _pairs(rows: list[tuple[int, int, int, float, bool]]) -> pd.DataFrame:
    return pd.DataFrame([{"i": i, "j": j, "offset": j - i, "inliers": inl, "inlier_ratio": r, "edge_accepted": acc}
                         for i, j, inl, r, acc in rows])


def _sequence_pairs(n: int, breaks: set[int] = frozenset()) -> pd.DataFrame:
    rows = []
    for i in range(n):
        for d in (1, 2, 5):
            j = i + d
            if j >= n:
                continue
            acc = not any(i < b <= j for b in breaks)
            rows.append((i, j, 120 if acc else 5, 0.6 if acc else 0.05, acc))
    return _pairs(rows)


def test_only_accepted_pairs_become_edges() -> None:
    pairs = _pairs([(0, 1, 100, 0.5, True), (1, 2, 3, 0.1, False), (2, 3, 80, 0.4, True), (3, 4, 0, float("nan"), False)])
    G = build_graph([0, 1, 2, 3, 4], pairs)
    assert set(G.nodes()) == {0, 1, 2, 3, 4}
    assert {tuple(sorted(e)) for e in G.edges()} == {(0, 1), (2, 3)}
    assert not G.has_edge(1, 2) and not G.has_edge(3, 4)
    assert G[0][1]["inliers"] == 100 and G[0][1]["offset"] == 1


def test_isolated_and_components() -> None:
    pairs = _pairs([(0, 1, 100, 0.5, True), (1, 2, 90, 0.5, True), (3, 4, 60, 0.4, True), (4, 5, 2, 0.1, False)])
    stats = graph_stats(build_graph(list(range(7)), pairs))
    assert stats["n_nodes"] == 7 and stats["n_edges"] == 3
    assert stats["isolated_nodes"] == [5, 6] and stats["n_isolated"] == 2
    assert stats["n_components"] == 4
    assert stats["components"][0] == [0, 1, 2] and stats["components"][1] == [3, 4]
    assert stats["largest_component_size"] == 3
    assert abs(stats["largest_component_pct"] - 100 * 3 / 7) < 1e-9
    assert stats["component_of"][0] == 0 and stats["component_of"][4] == 1
    assert stats["degree"][1] == 2 and stats["degree"][6] == 0
    assert stats["degree_histogram"] == {0: 2, 1: 4, 2: 1}


def test_empty_graph_stats() -> None:
    stats = graph_stats(build_graph([], pd.DataFrame()))
    assert stats["n_nodes"] == 0 and stats["n_components"] == 0 and stats["largest_component_pct"] == 0.0


def test_unreadable_frames_are_not_nodes() -> None:
    G = build_graph([0, 2], _pairs([(0, 2, 50, 0.5, True)]))
    assert set(G.nodes()) == {0, 2} and G.has_edge(0, 2)


def test_edges_dataframe_columns() -> None:
    cfg = AnalyzerConfig(input_dir=Path("."))
    G = build_graph([0, 1, 2], _pairs([(1, 2, 70, 0.7, True), (0, 1, 40, 0.3, True)]))
    df = edges_dataframe(G, cfg)
    assert list(df.columns) == ["source", "target", "offset", "inliers", "inlier_ratio", "criteria"]
    assert df[["source", "target"]].values.tolist() == [[0, 1], [1, 2]]
    assert "inliers>=30" in df["criteria"].iloc[0] and "inlier_ratio>=0.5" in df["criteria"].iloc[0]
    empty = edges_dataframe(build_graph([0], pd.DataFrame()), cfg)
    assert list(empty.columns) == ["source", "target", "offset", "inliers", "inlier_ratio", "criteria"] and empty.empty


def _frames(n: int) -> pd.DataFrame:
    return pd.DataFrame({"index": range(n), "filename": [f"frame_{i:05d}.png" for i in range(n)]})


def test_draw_small_graph(tmp_path: Path) -> None:
    G = build_graph(list(range(8)), _sequence_pairs(8, breaks={4}))
    stats = graph_stats(G)
    png = draw_graph_png(G, stats, tmp_path / "g.png", seed=42)
    html = write_graph_html(G, stats, _frames(8), tmp_path / "g.html", seed=42)
    assert png.stat().st_size > 1000
    text = html.read_text(encoding="utf-8")
    assert "frame_00003.png" in text and 'src="http' not in text and len(text) > 1_000_000


def test_draw_large_graph(tmp_path: Path) -> None:
    n = 300
    pairs = _sequence_pairs(n, breaks={100, 220})
    pairs.loc[(pairs["i"] >= 150) & (pairs["i"] <= 152) | (pairs["j"] >= 150) & (pairs["j"] <= 152), "edge_accepted"] = False
    G = build_graph(list(range(n)), pairs)
    stats = graph_stats(G)
    assert stats["n_isolated"] >= 1 and stats["n_components"] >= 3
    png = draw_graph_png(G, stats, tmp_path / "big.png", seed=1)
    html = write_graph_html(G, stats, _frames(n), tmp_path / "big.html", seed=1)
    assert png.stat().st_size > 1000 and html.stat().st_size > 1000


def test_draw_is_deterministic_and_handles_empty(tmp_path: Path) -> None:
    G = build_graph(list(range(10)), _sequence_pairs(10))
    stats = graph_stats(G)
    a = draw_graph_png(G, stats, tmp_path / "a.png", seed=7).read_bytes()
    b = draw_graph_png(G, stats, tmp_path / "b.png", seed=7).read_bytes()
    assert a == b
    E = build_graph([], pd.DataFrame())
    es = graph_stats(E)
    assert draw_graph_png(E, es, tmp_path / "e.png", seed=0).stat().st_size > 0
    assert write_graph_html(E, es, pd.DataFrame(), tmp_path / "e.html", seed=0).stat().st_size > 0

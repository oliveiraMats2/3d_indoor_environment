"""Diagnostic charts (matplotlib PNG) computed only from the analyzer dataframes."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.ticker  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from frame_analyzer.config import AnalyzerConfig  # noqa: E402

log = logging.getLogger(__name__)

DPI = 120
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
PRIMARY = SERIES[0]
POOR = "#d03b3b"         # status critical
REVIEW = "#c98500"       # darker warning step, legible on the light surface
GOOD = "#0ca30c"
ISOLATED = "#eb6834"
CONNECTED = "#1f7a1f"
BLUE_RAMP = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])

STYLE = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_SECONDARY, "axes.titlecolor": INK,
    "axes.titlelocation": "left", "axes.titlesize": 12, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK_SECONDARY, "ytick.labelcolor": INK_SECONDARY,
    "legend.frameon": False, "legend.fontsize": 8, "font.size": 9, "lines.linewidth": 1.2,
}


# ----------------------------------------------------------------------------- helpers

def _save(fig: plt.Figure, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def _placeholder(path: Path, title: str, reason: str = "") -> Path:
    """Figure stating that there is no data, so the report keeps its layout."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.set_axis_off()
        ax.set_title(title)
        msg = "sem dados" + (f"\n{reason}" if reason else "")
        ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=13, color=INK_SECONDARY, transform=ax.transAxes)
        log.info("no data for '%s'; placeholder written to %s", title, path)
        return _save(fig, path)


def _values(df: pd.DataFrame | None, col: str) -> pd.Series:
    if df is None or df.empty or col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _has(df: pd.DataFrame | None, *cols: str) -> bool:
    return all(_values(df, c).notna().any() for c in cols)


def _xindex(df: pd.DataFrame) -> pd.Series:
    return df["index"] if "index" in df.columns else pd.Series(np.arange(len(df)), index=df.index)


def _seq_width(n: int) -> float:
    return float(np.clip(8 + n / 120, 10, 18))


def _hline(ax, y: float, color: str, label: str, style: str = "--") -> None:
    ax.axhline(y, color=color, linestyle=style, linewidth=1.2, zorder=3)
    ax.annotate(label, xy=(1.0, y), xycoords=("axes fraction", "data"), xytext=(4, 0), textcoords="offset points",
                va="center", ha="left", fontsize=8, color=INK_SECONDARY)


def _vline(ax, x: float, color: str, label: str) -> None:
    ax.axvline(x, color=color, linestyle="--", linewidth=1.2, zorder=3)
    ax.annotate(label, xy=(x, 1.0), xycoords=("data", "axes fraction"), xytext=(3, -2), textcoords="offset points",
                va="top", ha="left", fontsize=8, color=INK_SECONDARY, rotation=90)


def _wide_range(v: pd.Series) -> bool:
    pos = v[v > 0]
    return len(pos) > 1 and pos.max() / pos.min() > 100


def _offset_colors(offsets: list[int]) -> dict[int, str]:
    return {o: SERIES[k % len(SERIES)] for k, o in enumerate(sorted(offsets))}


def _readable(frames_df: pd.DataFrame | None) -> pd.DataFrame:
    if frames_df is None or frames_df.empty:
        return pd.DataFrame()
    if "readable" in frames_df.columns:
        return frames_df[frames_df["readable"].fillna(False).astype(bool)]
    return frames_df


def _short(name: str, width: int = 28) -> str:
    return name if len(name) <= width else "…" + name[-(width - 1):]


# ----------------------------------------------------------------------------- sharpness

def _sharpness_thresholds(frames_df: pd.DataFrame, cfg: AnalyzerConfig) -> dict[str, float]:
    lv = _values(_readable(frames_df), "laplacian_var").dropna()
    med = float(lv.median()) if len(lv) else float("nan")
    th = {"median": med, "poor": med * cfg.sharpness_poor_rel, "review": med * cfg.sharpness_review_rel}
    if cfg.sharpness_abs_min is not None:
        th["abs_min"] = float(cfg.sharpness_abs_min)
    return th


def plot_sharpness(frames_df: pd.DataFrame, cfg: AnalyzerConfig, path: Path) -> Path:
    """Laplacian variance per frame with the relative poor/review thresholds."""
    title = "Nitidez por frame (variância do Laplaciano)"
    if not _has(frames_df, "laplacian_var"):
        return _placeholder(path, title)
    x, y = _xindex(frames_df), _values(frames_df, "laplacian_var")
    th = _sharpness_thresholds(frames_df, cfg)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(_seq_width(len(y)), 4.2))
        ax.plot(x, y, color=PRIMARY, linewidth=1.0, label="variância do Laplaciano", zorder=2)
        poor = y < th["poor"]
        review = (y >= th["poor"]) & (y < th["review"])
        ax.scatter(x[review], y[review], s=18, color=REVIEW, marker="o", zorder=4,
                   label=f"revisar (< {cfg.sharpness_review_rel:g}× mediana): {int(review.sum())}")
        ax.scatter(x[poor], y[poor], s=22, color=POOR, marker="x", zorder=5,
                   label=f"ruim (< {cfg.sharpness_poor_rel:g}× mediana): {int(poor.sum())}")
        _hline(ax, th["median"], MUTED, f"mediana {th['median']:.1f}", ":")
        _hline(ax, th["review"], REVIEW, f"revisar {th['review']:.1f}")
        _hline(ax, th["poor"], POOR, f"ruim {th['poor']:.1f}")
        if "abs_min" in th:
            _hline(ax, th["abs_min"], POOR, f"mínimo abs. {th['abs_min']:.1f}", "-.")
        if _wide_range(y):
            ax.set_yscale("log")
        ax.set_xlabel("índice do frame")
        ax.set_ylabel("variância do Laplaciano")
        ax.set_title(title)
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.15), ncol=3)
        return _save(fig, path)


def plot_sharpness_distribution(frames_df: pd.DataFrame, cfg: AnalyzerConfig, path: Path) -> Path:
    """Histogram of the Laplacian variance with the threshold positions."""
    title = "Distribuição da nitidez (variância do Laplaciano)"
    y = _values(frames_df, "laplacian_var").dropna()
    if y.empty:
        return _placeholder(path, title)
    th = _sharpness_thresholds(frames_df, cfg)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 4))
        log_scale = _wide_range(y) and (y > 0).all()
        bins = np.geomspace(y.min(), y.max(), 40) if log_scale else 40
        ax.hist(y, bins=bins, color=PRIMARY, edgecolor=SURFACE, linewidth=0.8)
        if log_scale:
            ax.set_xscale("log")
        _vline(ax, th["poor"], POOR, f"ruim {th['poor']:.1f}")
        _vline(ax, th["review"], REVIEW, f"revisar {th['review']:.1f}")
        _vline(ax, th["median"], MUTED, f"mediana {th['median']:.1f}")
        ax.set_xlabel("variância do Laplaciano")
        ax.set_ylabel("número de frames")
        ax.set_title(title)
        return _save(fig, path)


# ----------------------------------------------------------------------------- SIFT

def plot_sift_keypoints(frames_df: pd.DataFrame, cfg: AnalyzerConfig, path: Path) -> Path:
    """SIFT keypoints per frame with poor/review minimums."""
    title = "Keypoints SIFT por frame"
    if not _has(frames_df, "sift_keypoints"):
        return _placeholder(path, title)
    x, y = _xindex(frames_df), _values(frames_df, "sift_keypoints")
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(_seq_width(len(y)), 4.2))
        ax.plot(x, y, color=PRIMARY, linewidth=1.0, label="keypoints SIFT", zorder=2)
        poor = y < cfg.keypoints_poor_min
        review = (y >= cfg.keypoints_poor_min) & (y < cfg.keypoints_review_min)
        ax.scatter(x[review], y[review], s=18, color=REVIEW, zorder=4, label=f"revisar: {int(review.sum())}")
        ax.scatter(x[poor], y[poor], s=22, color=POOR, marker="x", zorder=5, label=f"ruim: {int(poor.sum())}")
        _hline(ax, cfg.keypoints_review_min, REVIEW, f"revisar < {cfg.keypoints_review_min}")
        _hline(ax, cfg.keypoints_poor_min, POOR, f"ruim < {cfg.keypoints_poor_min}")
        ax.set_ylim(bottom=0)
        ax.set_xlabel("índice do frame")
        ax.set_ylabel("número de keypoints")
        ax.set_title(title)
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.15), ncol=3)
        return _save(fig, path)


def plot_sift_density(frames_df: pd.DataFrame, path: Path) -> Path:
    """SIFT keypoint density (per megapixel of the analysis image) per frame."""
    title = "Densidade de keypoints SIFT por frame (por megapixel)"
    if not _has(frames_df, "sift_density_per_mp"):
        return _placeholder(path, title)
    x, y = _xindex(frames_df), _values(frames_df, "sift_density_per_mp")
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(_seq_width(len(y)), 4))
        ax.plot(x, y, color=PRIMARY, linewidth=1.0)
        med = float(y.median())
        _hline(ax, med, MUTED, f"mediana {med:.0f}", ":")
        ax.set_ylim(bottom=0)
        ax.set_xlabel("índice do frame")
        ax.set_ylabel("keypoints / MP (resolução de análise)")
        ax.set_title(title)
        return _save(fig, path)


def plot_sift_distribution(frames_df: pd.DataFrame, path: Path) -> Path:
    """Histograms of keypoint count and density side by side (separate axes)."""
    title = "Distribuição de keypoints SIFT"
    kp, dens = _values(frames_df, "sift_keypoints").dropna(), _values(frames_df, "sift_density_per_mp").dropna()
    if kp.empty and dens.empty:
        return _placeholder(path, title)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, data, label in ((axes[0], kp, "keypoints por frame"), (axes[1], dens, "keypoints / MP")):
            if data.empty:
                ax.set_axis_off()
                ax.text(0.5, 0.5, "sem dados", ha="center", va="center", transform=ax.transAxes, color=INK_SECONDARY)
                continue
            ax.hist(data, bins=40, color=PRIMARY, edgecolor=SURFACE, linewidth=0.8)
            _vline(ax, float(data.median()), MUTED, f"mediana {data.median():.0f}")
            ax.set_xlabel(label)
            ax.set_ylabel("número de frames")
        fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=12)
        fig.tight_layout()
        return _save(fig, path)


# ----------------------------------------------------------------------------- grid occupancy

def plot_grid_occupancy(frames_df: pd.DataFrame, cfg: AnalyzerConfig, path: Path) -> Path:
    """Fraction of grid cells with at least one keypoint, per frame."""
    title = f"Ocupação da grade {cfg.grid_rows}×{cfg.grid_cols} por keypoints (plano da imagem)"
    if not _has(frames_df, "occupancy_ratio"):
        return _placeholder(path, title)
    x, y = _xindex(frames_df), _values(frames_df, "occupancy_ratio")
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(_seq_width(len(y)), 4))
        ax.plot(x, y, color=PRIMARY, linewidth=1.0, label="fração de células ocupadas")
        low = y < cfg.occupancy_review_min
        ax.scatter(x[low], y[low], s=18, color=REVIEW, zorder=4, label=f"revisar: {int(low.sum())}")
        _hline(ax, cfg.occupancy_review_min, REVIEW, f"revisar < {cfg.occupancy_review_min:.0%}")
        ax.set_ylim(0, 1.02)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_xlabel("índice do frame")
        ax.set_ylabel("células ocupadas")
        ax.set_title(title)
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.15), ncol=2)
        return _save(fig, path)


def _readable_counts(grid_counts: np.ndarray | None, frames_df: pd.DataFrame | None) -> np.ndarray:
    if grid_counts is None or np.asarray(grid_counts).ndim != 3 or len(grid_counts) == 0:
        return np.zeros((0, 0, 0))
    gc = np.asarray(grid_counts)
    if frames_df is not None and not frames_df.empty and "readable" in frames_df.columns and len(frames_df) == len(gc):
        gc = gc[frames_df["readable"].fillna(False).astype(bool).to_numpy()]
    return gc


def _annotate_cells(ax, data: np.ndarray, vmax: float, fmt: str, fontsize: float) -> None:
    for (r, c), v in np.ndenumerate(data):
        ax.text(c, r, format(v, fmt), ha="center", va="center", fontsize=fontsize,
                color="#ffffff" if vmax and v > 0.55 * vmax else INK)


def plot_keypoint_heatmap(grid_counts: np.ndarray, frames_df: pd.DataFrame, path: Path) -> Path:
    """Mean keypoints per grid cell over readable frames."""
    title = "Média de keypoints por célula da grade (frames legíveis)"
    gc = _readable_counts(grid_counts, frames_df)
    if gc.size == 0:
        return _placeholder(path, title)
    mean = gc.mean(axis=0)
    rows, cols = mean.shape
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(min(1.0 + 0.9 * cols, 14), min(1.0 + 0.8 * rows, 12)))
        im = ax.imshow(mean, cmap=BLUE_RAMP, vmin=0, aspect="auto")
        ax.grid(False)
        if rows * cols <= 400:
            _annotate_cells(ax, mean, float(mean.max()), ".0f" if mean.max() >= 10 else ".1f", 8)
        ax.set_xticks(range(cols))
        ax.set_yticks(range(rows))
        ax.set_xlabel("coluna da grade")
        ax.set_ylabel("linha da grade")
        ax.set_title(f"{title}\n{len(gc)} frames")
        fig.colorbar(im, ax=ax, label="keypoints médios por célula")
        return _save(fig, path)


def _example_frames(frames_df: pd.DataFrame, n: int) -> list[int]:
    """Row labels of real frames spread over the occupancy ranking (worst ... best)."""
    rd = _readable(frames_df)
    occ = _values(rd, "occupancy_ratio")
    occ = occ[occ.notna()]
    if occ.empty or n <= 0:
        return []
    order = occ.sort_values(kind="stable").index
    picks = np.unique(np.linspace(0, len(order) - 1, min(n, len(order))).round().astype(int))
    return [order[p] for p in picks]


def plot_grid_examples(grid_counts: np.ndarray, frames_df: pd.DataFrame, path: Path, n: int = 6) -> Path:
    """Keypoint grids of real frames from worst to best occupancy (no photo is drawn)."""
    title = "Exemplos de distribuição de keypoints na grade (pior → melhor ocupação)"
    if grid_counts is None or frames_df is None or frames_df.empty or len(grid_counts) != len(frames_df):
        return _placeholder(path, title)
    labels = _example_frames(frames_df, n)
    if not labels:
        return _placeholder(path, title)
    gc = np.asarray(grid_counts)
    pos = [frames_df.index.get_loc(lbl) for lbl in labels]
    vmax = float(max(gc[p].max() for p in pos)) or 1.0
    k = len(pos)
    ncols = min(3, k)
    nrows = int(np.ceil(k / ncols))
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows), squeeze=False)
        for ax in axes.flat[k:]:
            ax.set_axis_off()
        im = None
        for ax, p, lbl in zip(axes.flat, pos, labels):
            row = frames_df.loc[lbl]
            im = ax.imshow(gc[p], cmap=BLUE_RAMP, vmin=0, vmax=vmax, aspect="auto")
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            if gc[p].size <= 144:
                _annotate_cells(ax, gc[p], vmax, "d", 7)
            idx = int(row["index"]) if "index" in row else p
            ax.set_title(f"#{idx} {_short(str(row.get('filename', '')))}\n"
                         f"ocupação {float(row['occupancy_ratio']):.0%}, {int(gc[p].sum())} kp", fontsize=9)
        fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=12)
        fig.tight_layout()
        if im is not None:
            fig.colorbar(im, ax=axes.ravel().tolist(), label="keypoints por célula", shrink=0.8)
        return _save(fig, path)


# ----------------------------------------------------------------------------- matching

def _pairs_by_offset(pairs_df: pd.DataFrame | None, col: str) -> dict[int, pd.DataFrame]:
    if pairs_df is None or pairs_df.empty or not _has(pairs_df, col) or "offset" not in pairs_df.columns:
        return {}
    return {int(o): g.sort_values("i") for o, g in pairs_df.groupby("offset")}


def _accepted(df: pd.DataFrame) -> pd.Series:
    if "edge_accepted" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["edge_accepted"].fillna(False).astype(bool)


def _per_offset_panels(pairs_df: pd.DataFrame, col: str, threshold: float, thr_label: str, ylabel: str,
                       title: str, path: Path, percent: bool = False) -> Path:
    groups = _pairs_by_offset(pairs_df, col)
    if not groups:
        return _placeholder(path, title)
    colors = _offset_colors(list(groups))
    n_max = max(len(g) for g in groups.values())
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(len(groups), 1, figsize=(_seq_width(n_max), 2.6 * len(groups) + 0.8),
                                 sharex=True, sharey=True, squeeze=False)
        for ax, (off, g) in zip(axes[:, 0], groups.items()):
            y = pd.to_numeric(g[col], errors="coerce")
            ax.plot(g["i"], y, color=colors[off], linewidth=1.0, label=f"par (i, i+{off})")
            rej = ~_accepted(g)
            ax.scatter(g["i"][rej], y[rej].fillna(0), s=10, color=POOR, marker="x", zorder=4,
                       label=f"aresta rejeitada: {int(rej.sum())} de {len(g)}")
            _hline(ax, threshold, MUTED, thr_label)
            ax.set_ylabel(ylabel)
            ax.set_title(f"distância temporal {off}", fontsize=10)
            ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2)
            if percent:
                ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        axes[-1, 0].set_xlabel("índice i do primeiro frame do par")
        axes[0, 0].set_ylim(bottom=0)
        fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=12)
        fig.tight_layout()
        return _save(fig, path)


def plot_matching_inliers(pairs_df: pd.DataFrame, cfg: AnalyzerConfig, path: Path) -> Path:
    """Fundamental-matrix inliers per pair along the sequence, one panel per temporal offset."""
    return _per_offset_panels(pairs_df, "inliers", cfg.edge_min_inliers, f"mínimo {cfg.edge_min_inliers}",
                              "inliers (matriz F)", "Inliers da verificação geométrica por par", path)


def plot_matching_inlier_ratio(pairs_df: pd.DataFrame, cfg: AnalyzerConfig, path: Path) -> Path:
    """Inlier ratio (inliers / Lowe matches) per pair, one panel per temporal offset."""
    return _per_offset_panels(pairs_df, "inlier_ratio", cfg.edge_min_inlier_ratio,
                              f"mínimo {cfg.edge_min_inlier_ratio:.0%}", "razão de inliers",
                              "Razão de inliers (inliers / matches de Lowe) por par", path, percent=True)


def plot_matching_scatter(pairs_df: pd.DataFrame, cfg: AnalyzerConfig, path: Path) -> Path:
    """Inliers vs inlier ratio per pair, colored by offset, acceptance region shaded."""
    title = "Inliers × razão de inliers por par (região de aceitação sombreada)"
    if pairs_df is None or pairs_df.empty or not _has(pairs_df, "inliers", "inlier_ratio"):
        return _placeholder(path, title)
    df = pairs_df.assign(_x=_values(pairs_df, "inlier_ratio"), _y=_values(pairs_df, "inliers")).dropna(
        subset=["_x", "_y"])
    offsets = sorted(int(o) for o in df["offset"].unique()) if "offset" in df.columns else [0]
    colors = _offset_colors(offsets)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 6))
        ymax = max(float(df["_y"].max()) * 1.05, cfg.edge_min_inliers * 1.5)
        ax.fill_between([cfg.edge_min_inlier_ratio, 1.0], cfg.edge_min_inliers, ymax, color=GOOD, alpha=0.08,
                        zorder=0, label="região de aceitação da aresta")
        ax.axvline(cfg.edge_min_inlier_ratio, color=MUTED, linestyle="--", linewidth=1)
        ax.axhline(cfg.edge_min_inliers, color=MUTED, linestyle="--", linewidth=1)
        size = 14 if len(df) > 1000 else 24
        for off in offsets:
            g = df[df["offset"] == off] if "offset" in df.columns else df
            ax.scatter(g["_x"], g["_y"], s=size, color=colors[off], alpha=0.6, edgecolors=SURFACE, linewidths=0.4,
                       label=f"distância {off} (n={len(g)})")
        acc = int(_accepted(df).sum())
        ax.text(0.99, 0.98, f"arestas aceitas: {acc} de {len(df)} pares", transform=ax.transAxes, ha="right",
                va="top", color=INK_SECONDARY, fontsize=9)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, ymax)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_xlabel(f"razão de inliers (mínimo {cfg.edge_min_inlier_ratio:.0%})")
        ax.set_ylabel(f"inliers (mínimo {cfg.edge_min_inliers})")
        ax.set_title(title)
        ax.legend(loc="upper left")
        return _save(fig, path)


def plot_temporal_redundancy(pairs_df: pd.DataFrame, redundant_df: pd.DataFrame | None, cfg: AnalyzerConfig,
                             path: Path) -> Path:
    """Consecutive pairs: inlier ratio and median displacement, redundancy candidates marked."""
    title = "Redundância temporal (pares consecutivos i, i+1)"
    if pairs_df is None or pairs_df.empty or "offset" not in pairs_df.columns:
        return _placeholder(path, title)
    g = pairs_df[pairs_df["offset"] == 1].sort_values("i")
    if g.empty or not (_has(g, "inlier_ratio") or _has(g, "median_disp_px")):
        return _placeholder(path, title)
    red_i: set[int] = set()
    if redundant_df is not None and not redundant_df.empty and "i" in redundant_df.columns:
        red_i = {int(v) for v in redundant_df["i"]}
    mask = g["i"].astype(int).isin(red_i)
    with plt.rc_context(STYLE):
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(_seq_width(len(g)), 6.2), sharex=True)
        ratio, disp = _values(g, "inlier_ratio"), _values(g, "median_disp_px")
        a1.plot(g["i"], ratio, color=PRIMARY, linewidth=1.0)
        a1.scatter(g["i"][mask], ratio[mask], s=20, color=REVIEW, zorder=4,
                   label=f"candidato a redundante: {int(mask.sum())}")
        _hline(a1, cfg.redundancy_min_inlier_ratio, MUTED, f"≥ {cfg.redundancy_min_inlier_ratio:.0%}")
        a1.set_ylim(0, 1.02)
        a1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        a1.set_ylabel("razão de inliers")
        a1.legend(loc="lower left")
        a2.plot(g["i"], disp, color=PRIMARY, linewidth=1.0)
        a2.scatter(g["i"][mask], disp[mask], s=20, color=REVIEW, zorder=4)
        _hline(a2, cfg.redundancy_max_median_disp_px, MUTED, f"≤ {cfg.redundancy_max_median_disp_px:g} px")
        if _wide_range(disp.dropna()):
            a2.set_yscale("log")
        else:
            a2.set_ylim(bottom=0)
        a2.set_ylabel("deslocamento mediano (px)")
        a2.set_xlabel("índice i do frame")
        fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=12)
        fig.tight_layout()
        return _save(fig, path)


# ----------------------------------------------------------------------------- graph degree

def plot_degree_distribution(stats: dict, path: Path) -> Path:
    """Histogram of node degree in the connectivity graph (degree 0 = isolated, orange)."""
    title = "Distribuição do grau dos frames no grafo de conectividade"
    hist = (stats or {}).get("degree_histogram") or {}
    if not hist:
        return _placeholder(path, title)
    degs = sorted(int(d) for d in hist)
    counts = [int(hist[d]) for d in degs]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(degs) + 3), 4))
        colors = [ISOLATED if d == 0 else CONNECTED for d in degs]
        bars = ax.bar(degs, counts, color=colors, edgecolor=SURFACE, linewidth=2, width=0.8)
        if len(degs) <= 30:
            for b, c in zip(bars, counts):
                ax.annotate(str(c), (b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 2),
                            textcoords="offset points", ha="center", va="bottom", fontsize=8, color=INK_SECONDARY)
        ax.set_xticks(degs)
        ax.set_xlabel("grau (número de arestas aceitas)")
        ax.set_ylabel("número de frames")
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
        handles = [plt.Rectangle((0, 0), 1, 1, color=CONNECTED, label="conectado")]
        if 0 in degs:
            handles.append(plt.Rectangle((0, 0), 1, 1, color=ISOLATED, label="isolado (grau 0)"))
        ax.legend(handles=handles, loc="upper right")
        return _save(fig, path)

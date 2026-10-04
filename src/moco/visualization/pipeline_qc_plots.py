"""Original-repo-style figures comparing pipeline QC across sources: QC-FC, distance dependence, Q, FC edges."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: SLURM nodes and the login node have no display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from nilearn import plotting
from omegaconf import DictConfig
from scipy import ndimage, stats

from moco.evaluation.qc_metrics import upper_triangle
from moco.visualization.test_visualization import bootstrap_median_ci


def load_source(source_dir: Path) -> dict:
    """Everything `scripts/pipeline_qc.py` wrote for one source; matrices are (R, R)."""
    return {
        "summary": json.loads((source_dir / "summary.json").read_text()),
        "subjects": pd.read_csv(source_dir / "per_subject.csv"),
        **{name: np.load(source_dir / f"{name}.npy")
           for name in ("qc_fc_r", "qc_fc_fdr_significant", "distance_matrix", "fc_mean_z", "fc_significant")},
    }


def load_node_coords(centroids_csv: str) -> np.ndarray:
    """(R, 3) MNI centroids in Schaefer label order, the ROI order of every QC matrix."""
    return pd.read_csv(centroids_csv)[["R", "A", "S"]].to_numpy(float)


def top_edges(values: np.ndarray, significant: np.ndarray, top_percent: float) -> tuple[np.ndarray, float]:
    """Keep the significant edges whose |value| is in the top `top_percent` % of the significant ones.

    Args:
        values: (R, R) edge values, e.g. mean Fisher-z FC.
        significant: (R, R) bool mask of FDR-significant edges.
        top_percent: Share of significant edges to keep, e.g. 5.

    Returns:
        (mask (R, R), |value| threshold).
    """
    magnitude = np.abs(upper_triangle(values))[upper_triangle(significant)]
    threshold = float(np.percentile(magnitude, 100 - top_percent))
    return significant & (np.abs(values) >= threshold), threshold


def plot_qcfc_distributions(data: dict, labels: dict, colors: dict, cfg: DictConfig) -> Figure:
    """Ciric-style QC-FC r densities, one panel per source on a shared x-axis."""
    style = cfg.distribution
    r_all = np.concatenate([upper_triangle(s["qc_fc_r"]) for s in data.values()])
    xlim = (float(np.quantile(r_all, 0.0005)) - 0.05, float(np.quantile(r_all, 0.9995)) + 0.05)
    grid = np.linspace(*xlim, style.kde_points)
    fig, axes = plt.subplots(1, len(data), figsize=(style.panel_width * len(data), style.panel_height),
                             sharex=True, sharey=True, squeeze=False, constrained_layout=True)
    for ax, (name, source) in zip(axes[0], data.items(), strict=True):
        r = upper_triangle(source["qc_fc_r"])
        density = stats.gaussian_kde(r)(grid)
        density /= density.max()  # heights are not compared, only shapes and medians
        ax.fill_between(grid, 0, density, color=colors[name], alpha=style.fill_alpha, linewidth=0, zorder=2)
        ax.plot(grid, density, color=colors[name], linewidth=1.5, zorder=3)
        ax.axvline(0, color="black", linewidth=2, zorder=4)
        ax.text(0.97, 0.95, rf"Median $|QC\mathrm{{-}}FC|$ = {np.median(np.abs(r)):.3f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=style.text_size, fontweight="bold")
        ax.set_title(labels[name], fontsize=style.title_size, fontweight="bold", color=colors[name])
        ax.set_xlabel("QC–FC correlation (r)", fontsize=style.text_size)
        ax.set_xlim(xlim)
        ax.set_ylim(0, 1.12)
        ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)
    fig.suptitle("QC-FC distribution")
    return fig


def plot_qcfc_matrices(data: dict, labels: dict, cfg: DictConfig) -> Figure:
    """QC-FC r matrices side by side on one shared colour scale."""
    fig, axes = plt.subplots(1, len(data), figsize=(cfg.matrix.panel_size * len(data), cfg.matrix.panel_size),
                             squeeze=False, constrained_layout=True)
    for ax, (name, source) in zip(axes[0], data.items(), strict=True):
        image = ax.imshow(source["qc_fc_r"], cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        ax.set_title(labels[name])
        ax.set_xlabel("ROI")
    axes[0, 0].set_ylabel("ROI")
    fig.colorbar(image, ax=axes[0].tolist(), label="QC-FC (r)", shrink=cfg.matrix.colorbar_shrink)
    fig.suptitle("QC-FC matrix")
    return fig


def _density_contours(ax: plt.Axes, x: np.ndarray, y: np.ndarray, cfg: DictConfig) -> None:
    """Filled iso-density contours of (x, y): a smoothed 2-D histogram, the fast stand-in for a KDE."""
    style = cfg.distance
    counts, x_edges, y_edges = np.histogram2d(x, y, bins=style.bins)
    density = ndimage.gaussian_filter(counts.T, sigma=style.smooth_sigma)
    density /= density.max()
    levels = np.linspace(style.density_floor, 1, style.levels + 1)
    centres_x, centres_y = (x_edges[:-1] + x_edges[1:]) / 2, (y_edges[:-1] + y_edges[1:]) / 2
    ax.contourf(centres_x, centres_y, density, levels=levels, cmap="Blues")


def plot_distance_dependence(data: dict, labels: dict, cfg: DictConfig) -> Figure:
    """QC-FC r against inter-ROI distance per source: density contours, zero line and linear fit."""
    style = cfg.distance
    fig, axes = plt.subplots(1, len(data), figsize=(style.panel_width * len(data), style.panel_height),
                             sharey=True, squeeze=False, constrained_layout=True)
    for ax, (name, source) in zip(axes[0], data.items(), strict=True):
        distance, r = upper_triangle(source["distance_matrix"]), upper_triangle(source["qc_fc_r"])
        _density_contours(ax, distance, r, cfg)
        ax.axhline(0, color="black", linewidth=2.5, zorder=4)
        fit = stats.linregress(distance, r)
        x = np.linspace(distance.min(), distance.max(), 2)
        ax.plot(x, fit.intercept + fit.slope * x, color="red", linewidth=3, zorder=5)
        ax.text(0.97, 0.98, f"{r.max():.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=11)
        ax.text(0.97, 0.02, f"{r.min():.2f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=11)
        ax.text(0.03, 0.03, f"QC–FC–DD r = {source['summary']['qcfc_dd_r']:.3f}\n"
                f"Slope = {fit.slope:.4f} per mm\nIntercept = {fit.intercept:.3f}", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=9,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
        ax.set_title(labels[name])
        ax.set_xlabel("ROI-pair distance (mm)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("QC–FC correlation (r)")
    fig.suptitle("QC-FC distance dependence")
    return fig


def plot_modularity(data: dict, labels: dict, colors: dict, cfg: DictConfig) -> Figure:
    """Box plot of per-subject modularity Q per source, medians joined with their bootstrap CI."""
    style = cfg.modularity
    names = list(data)
    q = [data[n]["subjects"]["Q"].to_numpy() for n in names]
    x = np.arange(1, len(names) + 1)
    fig, ax = plt.subplots(figsize=(style.box_width_inches * len(names), style.height))
    box = ax.boxplot(q, positions=x, widths=0.5, patch_artist=True, showfliers=False)
    for patch, name in zip(box["boxes"], names, strict=True):
        patch.set(facecolor=colors[name], edgecolor=style.edge, alpha=style.box_alpha, linewidth=1.2)
    for part in ("whiskers", "caps"):
        for line in box[part]:
            line.set(color=style.edge, linewidth=1.2)
    for line in box["medians"]:
        line.set(color=style.edge, linewidth=2)

    rng = np.random.default_rng(cfg.seed)
    medians = np.array([np.median(values) for values in q])
    bounds = np.array([bootstrap_median_ci(values, cfg.n_boot, cfg.ci, rng) for values in q])  # (sources, 2)
    # asymmetric error bars: the bootstrap CI of the median need not be centred on it
    yerr = np.vstack([medians - bounds[:, 0], bounds[:, 1] - medians])
    ax.errorbar(x, medians, yerr=yerr, color=style.trend, linewidth=2, marker="o", markersize=6, capsize=5,
                zorder=4, label=f"Median, {cfg.ci:.0%} bootstrap CI")
    ax.set_xticks(x, [f"{labels[n]}\n(n={len(values)})" for n, values in zip(names, q, strict=True)])
    ax.set_ylabel("Modularity Q")
    ax.set_title("Modularity Q")
    # headroom so the legend never sits on a whisker
    ax.set_ylim(top=ax.get_ylim()[1] + style.legend_headroom * np.ptp(ax.get_ylim()))
    ax.legend(frameon=False, loc="upper left")
    ax.yaxis.grid(True, color="#e0e0e0", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_connectomes(matrices: dict, titles: dict, coords: np.ndarray, cmap: str, vmin: float, vmax: float,
                     colorbar_label: str, edge_width: float | None, style: DictConfig, cfg: DictConfig) -> Figure:
    """Glass-brain edges per source, stacked in rows or side by side, on a shared colour scale.

    Args:
        matrices: source name -> (R, R) edge weights, 0 where an edge is not drawn.
        titles: source name -> panel title.
        coords: (R, 3) MNI node positions.
        cmap: Edge colormap.
        vmin: Lower colour limit.
        vmax: Upper colour limit.
        colorbar_label: Colour bar label.
        edge_width: Fixed edge width, or None to let nilearn scale width by |weight|.
        style: Layout block, cfg.connectome.qcfc or cfg.connectome.fc.
        cfg: Composed `configs/plot_pipeline_qc.yaml`.
    """
    n = len(matrices)
    n_rows, n_cols = (1, n) if style.side_by_side else (n, 1)
    width, height = style.panel_size
    fig = plt.figure(figsize=(width * n_cols, height * n_rows))
    edge_kwargs = {"alpha": 1.0} if edge_width is None else {"alpha": 1.0, "linewidth": edge_width}
    # panels fill [0, 0.92] in x, leaving room for the colour bar; each keeps a strip on top for its title
    panel_w, panel_h = 0.92 / n_cols, 1 / n_rows
    title_h = style.title_space * panel_h
    for i, (name, matrix) in enumerate(matrices.items()):
        left, bottom = (i % n_cols) * panel_w, 1 - (i // n_cols + 1) * panel_h
        plotting.plot_connectome(
            matrix, coords, node_color=cfg.connectome.node_color, node_size=cfg.connectome.node_size,
            edge_cmap=cmap, edge_vmin=vmin, edge_vmax=vmax, edge_kwargs=edge_kwargs,
            display_mode=style.display_mode, figure=fig, axes=(left, bottom, panel_w, panel_h - title_h),
            annotate=False, colorbar=False,
        )
        fig.text(left + panel_w / 2, bottom + panel_h - title_h / 2, titles[name], ha="center", va="center",
                 fontsize=style.title_size)
    colorbar_ax = fig.add_axes((0.935, 0.15, 0.015, 0.7))
    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
    fig.colorbar(mappable, cax=colorbar_ax).set_label(colorbar_label, fontsize=12)
    return fig


def qcfc_connectomes(data: dict, labels: dict, coords: np.ndarray, cfg: DictConfig) -> Figure:
    """All FDR-significant QC-FC edges per source, coloured green -> red by |QC-FC r|."""
    matrices = {n: np.where(s["qc_fc_fdr_significant"], np.abs(s["qc_fc_r"]), 0.0) for n, s in data.items()}
    titles = {n: f"{labels[n]}: QC-FC significant edges (n={s['summary']['qcfc_n_significant']:,})"
              for n, s in data.items()}
    vmax = max(float(m.max()) for m in matrices.values())
    return plot_connectomes(matrices, titles, coords, "RdYlGn_r", 0.0, vmax, "|QC-FC r|",
                            cfg.connectome.qcfc_edge_width, cfg.connectome.qcfc, cfg)


def fc_connectomes(data: dict, labels: dict, coords: np.ndarray, cfg: DictConfig) -> Figure:
    """Strongest FDR-significant FC edges per source (top cfg.fc_top_percent %), signed mean Fisher-z."""
    matrices, titles = {}, {}
    for name, source in data.items():
        keep, threshold = top_edges(source["fc_mean_z"], source["fc_significant"], cfg.fc_top_percent)
        matrices[name] = np.where(keep, source["fc_mean_z"], 0.0)
        titles[name] = (f"{labels[name]}: FC significant edges\ntop {cfg.fc_top_percent:g}% by |z| "
                        f"(|z| ≥ {threshold:.3f}, n={int(keep.sum()) // 2:,})")
    vmax = max(float(np.abs(m).max()) for m in matrices.values())
    return plot_connectomes(matrices, titles, coords, "RdBu_r", -vmax, vmax, "Mean Fisher-z FC", None,
                            cfg.connectome.fc, cfg)


def plot_pipeline_qc(cfg: DictConfig) -> dict[str, Figure]:
    """Build every comparison figure for cfg.sources.

    Args:
        cfg: Composed `configs/plot_pipeline_qc.yaml`.

    Returns:
        File stem -> figure.

    Raises:
        ValueError: If labels or colours do not match sources, or the sources were computed on different subjects.
    """
    if not len(cfg.labels) == len(cfg.colors) == len(cfg.sources):
        raise ValueError(f"{len(cfg.sources)} sources but {len(cfg.labels)} labels and {len(cfg.colors)} colours")
    data = {name: load_source(Path(cfg.qc_root) / name) for name in cfg.sources}
    if len({tuple(source["subjects"]["subject_id"]) for source in data.values()}) != 1:
        raise ValueError("sources were computed on different subjects; paired comparison is not valid")
    labels = dict(zip(cfg.sources, cfg.labels, strict=True))
    colors = dict(zip(cfg.sources, cfg.colors, strict=True))
    coords = load_node_coords(cfg.centroids_csv)
    return {
        "qc_fc_distribution": plot_qcfc_distributions(data, labels, colors, cfg),
        "qc_fc_matrix": plot_qcfc_matrices(data, labels, cfg),
        "qc_fc_distance_dependence": plot_distance_dependence(data, labels, cfg),
        "qc_fc_significant_edges": qcfc_connectomes(data, labels, coords, cfg),
        "modularity_q": plot_modularity(data, labels, colors, cfg),
        "fc_significant_edges": fc_connectomes(data, labels, coords, cfg),
    }

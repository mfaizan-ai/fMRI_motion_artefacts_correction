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
        ax.plot([0, 0], [0, np.interp(0, grid, density)], color="black", linewidth=2, zorder=4)  # stop at the curve
        ax.text(0.97, 0.95, rf"Median $|QC\mathrm{{-}}FC|$ = {np.median(np.abs(r)):.3f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=style.text_size, fontweight="bold")
        ax.set_title(labels[name], fontsize=style.title_size, fontweight="bold", color=colors[name])
        ax.set_xlim(xlim)
        ax.set_ylim(0, 1.12)
        ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)
    fig.supxlabel("QC–FC correlation (r)", fontsize=style.text_size + 1)
    fig.suptitle("QC-FC distribution")
    return fig


def _matched_colorbar(fig: Figure, ax: plt.Axes, mappable, label: str, cfg: DictConfig) -> None:
    """Colour bar beside `ax`, exactly as tall as its drawn box."""
    fig.canvas.draw()  # applies equal aspect / glass-brain extents, so the box is final
    box = ax.get_position()
    cax = fig.add_axes((box.x1 + cfg.colorbar.pad, box.y0, cfg.colorbar.width, box.height))
    fig.colorbar(mappable, cax=cax).set_label(label, fontsize=cfg.colorbar.label_size)


def _matrix_row(matrices: dict, labels: dict, cmap: str, vmin: float, vmax: float, colorbar_label: str,
                title: str, cfg: DictConfig) -> Figure:
    """(R, R) matrices side by side on one colour scale, one shared ROI label per axis."""
    style = cfg.matrix
    fig, axes = plt.subplots(1, len(matrices), figsize=(style.panel_size * len(matrices), style.panel_size),
                             squeeze=False)
    fig.subplots_adjust(left=0.05, right=0.93, bottom=0.12, top=0.86, wspace=style.wspace)
    for ax, (name, matrix) in zip(axes[0], matrices.items(), strict=True):
        image = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(labels[name], fontsize=style.title_size)
    axes[0, 0].set_ylabel("ROI")
    fig.supxlabel("ROI", y=0.02)
    _matched_colorbar(fig, axes[0, -1], image, colorbar_label, cfg)
    fig.suptitle(title, fontsize=style.title_size + 2)
    return fig


def plot_qcfc_matrices(data: dict, labels: dict, cfg: DictConfig) -> Figure:
    """QC-FC r matrices on a shared ±1 scale."""
    return _matrix_row({n: s["qc_fc_r"] for n, s in data.items()}, labels, "RdBu_r", -1.0, 1.0, "QC-FC (r)",
                       "QC-FC matrix", cfg)


def plot_fc_matrices(data: dict, labels: dict, cfg: DictConfig) -> Figure:
    """Group-mean Fisher-z FC matrices on a shared symmetric scale (diagonal is NaN, drawn blank)."""
    vmax = max(float(np.nanmax(np.abs(s["fc_mean_z"]))) for s in data.values())
    return _matrix_row({n: s["fc_mean_z"] for n, s in data.items()}, labels, "RdBu_r", -vmax, vmax,
                       "Mean Fisher-z FC", "FC matrix", cfg)


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
    """Box plot of per-subject modularity Q, one coloured box per source, no outlier points."""
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
    ax.set_xticks(x, [f"{labels[n]}\n(n={len(values)})" for n, values in zip(names, q, strict=True)])
    ax.set_ylabel("Modularity Q")
    ax.set_title("Modularity Q")
    ax.yaxis.grid(True, color="#e0e0e0", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_connectomes(matrices: dict, titles: dict, coords: np.ndarray, cmap: str, vmin: float, vmax: float,
                     colorbar_label: str, edge_width: float | None, title: str, cfg: DictConfig) -> Figure:
    """One glass brain per source (cfg.connectome.display_mode), side by side on a shared colour scale.

    Args:
        matrices: source name -> (R, R) edge weights, 0 where an edge is not drawn.
        titles: source name -> panel title.
        coords: (R, 3) MNI node positions.
        cmap: Edge colormap.
        vmin: Lower colour limit.
        vmax: Upper colour limit.
        colorbar_label: Colour bar label.
        edge_width: Fixed edge width, or None to let nilearn scale width by |weight|.
        title: Figure title.
        cfg: Composed `configs/plot_pipeline_qc.yaml`.
    """
    style = cfg.connectome
    n = len(matrices)
    width, height = style.panel_size
    fig = plt.figure(figsize=(width * n, height))
    edge_kwargs = {"alpha": 1.0} if edge_width is None else {"alpha": 1.0, "linewidth": edge_width}
    # panels fill [0, 0.92] in x, leaving room for the colour bar; the top strip holds titles
    panel_w, top = 0.92 / n, 1 - style.title_space
    for i, (name, matrix) in enumerate(matrices.items()):
        display = plotting.plot_connectome(
            matrix, coords, node_color=style.node_color, node_size=style.node_size, edge_cmap=cmap,
            edge_vmin=vmin, edge_vmax=vmax, edge_kwargs=edge_kwargs, display_mode=style.display_mode,
            figure=fig, axes=(i * panel_w, 0.0, panel_w, top), annotate=False, colorbar=False,
        )
        fig.text((i + 0.5) * panel_w, top, titles[name], ha="center", va="bottom", fontsize=style.title_size)
    brain_ax = next(iter(display.axes.values())).ax
    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
    _matched_colorbar(fig, brain_ax, mappable, colorbar_label, cfg)
    fig.suptitle(title, y=1.02, fontsize=style.title_size + 2)
    return fig


def qcfc_connectomes(data: dict, labels: dict, coords: np.ndarray, cfg: DictConfig) -> Figure:
    """All FDR-significant QC-FC edges per source, coloured green -> red by |QC-FC r|."""
    matrices = {n: np.where(s["qc_fc_fdr_significant"], np.abs(s["qc_fc_r"]), 0.0) for n, s in data.items()}
    titles = {n: f"{labels[n]} (n = {s['summary']['qcfc_n_significant']:,})" for n, s in data.items()}
    vmax = max(float(m.max()) for m in matrices.values())
    return plot_connectomes(matrices, titles, coords, "RdYlGn_r", 0.0, vmax, "|QC-FC r|",
                            cfg.connectome.qcfc_edge_width, "QC-FC significant edges (FDR)", cfg)


def fc_connectomes(data: dict, labels: dict, coords: np.ndarray, cfg: DictConfig) -> Figure:
    """Strongest FDR-significant FC edges per source (top cfg.fc_top_percent %), signed mean Fisher-z."""
    matrices, titles = {}, {}
    for name, source in data.items():
        keep, threshold = top_edges(source["fc_mean_z"], source["fc_significant"], cfg.fc_top_percent)
        matrices[name] = np.where(keep, source["fc_mean_z"], 0.0)
        titles[name] = f"{labels[name]} (n = {int(keep.sum()) // 2:,}, |z| ≥ {threshold:.2f})"
    vmax = max(float(np.abs(m).max()) for m in matrices.values())
    return plot_connectomes(matrices, titles, coords, "RdBu_r", -vmax, vmax, "Mean Fisher-z FC", None,
                            f"FC significant edges, top {cfg.fc_top_percent:g}% by |z|", cfg)


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
        "fc_matrix": plot_fc_matrices(data, labels, cfg),
        "qc_fc_distance_dependence": plot_distance_dependence(data, labels, cfg),
        "qc_fc_significant_edges": qcfc_connectomes(data, labels, coords, cfg),
        "modularity_q": plot_modularity(data, labels, colors, cfg),
        "fc_significant_edges": fc_connectomes(data, labels, coords, cfg),
    }

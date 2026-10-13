"""Figures for one order's voxel-wise ISC, drawn from the maps and tables saved in its order folder."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: SLURM nodes and the login node have no display
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import plotting
from omegaconf import DictConfig

from moco.evaluation.isc_voxel_wise import DATASETS, DIFF


def load_map(out_dir: Path, name: str, key: str) -> nib.Nifti1Image:
    """A saved group map with NaN (outside the common mask) set to 0, which nilearn leaves undrawn."""
    img = nib.load(out_dir / "group_maps" / f"{name}_{key}.nii.gz")
    return nib.Nifti1Image(np.nan_to_num(img.get_fdata(dtype=np.float32)), img.affine)


def colour_limits(out_dir: Path) -> dict[str, float]:
    """99th-percentile |z|: shared by raw and corrected so they compare directly, separate for the difference."""
    def top(name: str) -> float:
        values = load_map(out_dir, name, "mean_z").get_fdata()
        return float(np.percentile(np.abs(values[values != 0]), 99))
    shared = max(top(name) for name in DATASETS)
    return {"raw": shared, "corrected": shared, DIFF: top(DIFF)}


def _style(ax: plt.Axes, cfg: DictConfig) -> None:
    ax.tick_params(length=0, labelsize=9, labelcolor=cfg.plot.muted)
    ax.grid(axis="y", color=cfg.plot.grid, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(cfg.plot.grid)


def plot_slices(out_dir: Path, order: str, key: str, label: str, cfg: DictConfig) -> plt.Figure:
    """Axial slices of one group map, one row per map (raw, corrected, corrected - raw), same cuts everywhere."""
    limits = colour_limits(out_dir)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    for ax, name in zip(axes, (*DATASETS, DIFF), strict=True):
        plotting.plot_stat_map(load_map(out_dir, name, key), bg_img=cfg.bg_img, display_mode="z",
                               cut_coords=list(cfg.plot.cuts), cmap="RdBu_r", vmax=limits[name],
                               symmetric_cbar=True, axes=ax, title=f"order {order} {name}: {label}")
    return fig


def plot_peak_views(out_dir: Path, order: str, cfg: DictConfig) -> plt.Figure:
    """Sagittal, coronal and axial views through the voxel with the highest raw group ISC."""
    raw = load_map(out_dir, "raw", "mean_z")
    xyz = nib.affines.apply_affine(raw.affine, np.unravel_index(np.argmax(raw.get_fdata()), raw.shape))
    limits = colour_limits(out_dir)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    for ax, name in zip(axes, (*DATASETS, DIFF), strict=True):
        plotting.plot_stat_map(load_map(out_dir, name, "mean_z"), bg_img=cfg.bg_img, display_mode="ortho",
                               cut_coords=xyz, cmap="RdBu_r", vmax=limits[name], symmetric_cbar=True, axes=ax,
                               title=f"order {order} {name}: group mean Fisher z")
    fig.suptitle(f"Through the raw ISC peak at {np.round(xyz).astype(int).tolist()} mm", x=0.01, ha="left",
                 fontsize=11, fontweight="semibold", color=cfg.plot.ink)
    return fig


def plot_distributions(out_dir: Path, order: str, cfg: DictConfig) -> plt.Figure:
    """Group ISC across voxels (raw vs corrected) and each infant's median ISC before and after correction."""
    common = nib.load(out_dir / "common_mask.nii.gz").get_fdata() > 0
    subjects = pd.read_csv(out_dir / "tables" / "subject_summary.csv")
    colors = cfg.plot.colors
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
    for name in DATASETS:
        values = load_map(out_dir, name, "mean_z").get_fdata()[common]
        left.hist(values, bins=120, histtype="step", linewidth=2, color=colors[name],
                  label=f"{name}: median {np.median(values):+.3f}")
    left.axvline(0, color=cfg.plot.muted, linewidth=0.8, linestyle=(0, (4, 3)))
    left.set(xlabel="group mean Fisher z", ylabel="voxels")
    left.set_title("Group ISC across voxels", loc="left", fontsize=11, fontweight="semibold", color=cfg.plot.ink)
    left.legend(frameon=False, fontsize=9)

    for row in subjects.itertuples():
        right.plot([0, 1], [row.median_z_raw, row.median_z_corrected], color=cfg.plot.muted, alpha=0.35,
                   linewidth=1)
    for x, name in enumerate(DATASETS):
        right.scatter(np.full(len(subjects), x), subjects[f"median_z_{name}"], s=18, color=colors[name], zorder=3)
    n_up = int((subjects["median_z_corrected"] > subjects["median_z_raw"]).sum())
    right.set_xticks([0, 1], list(DATASETS))
    right.set(ylabel="median Fisher z over voxels", xlim=(-0.4, 1.4))
    right.set_title(f"Each infant: {n_up} of {len(subjects)} higher after correction", loc="left", fontsize=11,
                    fontweight="semibold", color=cfg.plot.ink)
    for ax in (left, right):
        _style(ax, cfg)
    fig.suptitle(f"order {order}", x=0.01, ha="left", fontsize=12, fontweight="semibold", color=cfg.plot.ink)
    fig.tight_layout()
    return fig


def plot_networks(table: pd.DataFrame, order: str, cfg: DictConfig) -> plt.Figure:
    """Network ISC, raw vs corrected (left) and the paired difference (right), with bootstrap CIs.

    Args:
        table: `tables/network_isc.csv`: network, map, mean_z, ci_low_z, ci_high_z.
        order: Order label for the title.
        cfg: Composed `configs/isc_voxel_wise.yaml`.
    """
    networks = list(cfg.networks)
    y = np.arange(len(networks))[::-1]
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    def draw(ax: plt.Axes, name: str, offset: float, color: str, label: str | None) -> None:
        t = table[table["map"] == name].set_index("network").loc[networks]
        ax.errorbar(t["mean_z"], y + offset, xerr=[t["mean_z"] - t["ci_low_z"], t["ci_high_z"] - t["mean_z"]],
                    fmt="o", color=color, markersize=7, elinewidth=2, capsize=0, label=label)

    for offset, name in zip((0.15, -0.15), DATASETS, strict=True):
        draw(left, name, offset, cfg.plot.colors[name], name)
    draw(right, DIFF, 0.0, cfg.plot.ink, None)
    for ax, title in ((left, "Network ISC"), (right, "Corrected - raw (paired)")):
        ax.axvline(0, color=cfg.plot.muted, linewidth=0.8, linestyle=(0, (4, 3)))
        ax.set_title(title, loc="left", fontsize=11, fontweight="semibold", color=cfg.plot.ink)
        ax.set_xlabel(f"mean Fisher z ({cfg.bootstrap.ci:.0%} bootstrap CI)", fontsize=9, color=cfg.plot.muted)
        _style(ax, cfg)
        ax.grid(axis="x", color=cfg.plot.grid, linewidth=0.8)
    left.set_yticks(y, networks)
    left.legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle(f"order {order}: ISC per Yeo-7 network", x=0.01, ha="left", fontsize=12, fontweight="semibold",
                 color=cfg.plot.ink)
    fig.tight_layout()
    return fig


def save_interactive(out_dir: Path, order: str, cfg: DictConfig) -> None:
    """Clickable three-view HTML viewers: raw and corrected group ISC, and the FDR-significant difference."""
    html_dir = out_dir / "figures" / "interactive"
    html_dir.mkdir(parents=True, exist_ok=True)
    limits = colour_limits(out_dir)
    for name, key in (("raw", "mean_z"), ("corrected", "mean_z"), (DIFF, "mean_z_fdr")):
        view = plotting.view_img(load_map(out_dir, name, key), bg_img=cfg.bg_img, cmap="RdBu_r",
                                 symmetric_cmap=True, vmax=limits[name], threshold=None,
                                 title=f"order {order} {name} {key}")
        view.save_as_html(html_dir / f"{name}_{key}.html")


def make_order_figures(out_dir: Path, order: str, cfg: DictConfig) -> None:
    """Every figure of one order into `out_dir/figures`."""
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    figures = {
        "group_maps": plot_slices(out_dir, order, "mean_z", "group mean Fisher z (all voxels)", cfg),
        "group_maps_fdr": plot_slices(out_dir, order, "mean_z_fdr",
                                      f"group mean Fisher z, FDR q < {cfg.fdr_alpha}", cfg),
        "peak_views": plot_peak_views(out_dir, order, cfg),
        "isc_distributions": plot_distributions(out_dir, order, cfg),
        "network_isc": plot_networks(pd.read_csv(out_dir / "tables" / "network_isc.csv"), order, cfg),
    }
    for stem, fig in figures.items():
        fig.savefig(fig_dir / f"{stem}.png", dpi=cfg.plot.dpi, bbox_inches="tight")
        plt.close(fig)
    save_interactive(out_dir, order, cfg)

"""Per-order ISC figures: network group ISC with bootstrap CIs, and the group network ISFC matrix."""
import matplotlib

matplotlib.use("Agg")  # headless: SLURM nodes and the login node have no display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from omegaconf import DictConfig


def plot_network_isc(table: pd.DataFrame, title: str, cfg: DictConfig) -> Figure:
    """Group ISC per network as bars with their bootstrap CI; FDR-significant networks are starred.

    Args:
        table: `network_isc_group.csv` rows: name, group_isc, ci_low, ci_high, significant.
        title: Figure title.
        cfg: Composed `configs/isc_network.yaml`.
    """
    style = cfg.plot
    y = np.arange(len(table))[::-1]  # first network on top
    fig, ax = plt.subplots(figsize=tuple(style.isc_figsize))
    xerr = np.vstack([table["group_isc"] - table["ci_low"], table["ci_high"] - table["group_isc"]])
    ax.barh(y, table["group_isc"], height=0.6, color=style.bar_color, zorder=2)
    ax.errorbar(table["group_isc"], y, xerr=xerr, fmt="none", ecolor=style.text_primary, capsize=4,
                linewidth=1.2, zorder=3)
    for yi, high, significant in zip(y, table["ci_high"], table["significant"], strict=True):
        if significant:
            ax.text(high, yi, "  *", va="center", ha="left", fontsize=style.label_size + 2,
                    color=style.text_primary)
    ax.axvline(0, color=style.text_secondary, linewidth=0.8)
    ax.set_yticks(y, table["name"])
    ax.set_xlabel(f"Group ISC (r), {cfg.ci:.0%} bootstrap CI", fontsize=style.label_size,
                  color=style.text_secondary)
    ax.set_title(title, loc="left", fontsize=style.title_size, color=style.text_primary)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.grid(True, color=style.grid, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.text(0.01, 0.005, f"* FDR q < {cfg.alpha:g} ({cfg.n_boot:,} subject-bootstrap resamples)",
             fontsize=style.label_size - 1, color=style.text_secondary)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def plot_network_isfc(matrix: np.ndarray, names: list[str], title: str, cfg: DictConfig) -> Figure:
    """Annotated (N, N) group network ISFC on a symmetric scale; the diagonal is the network ISC."""
    style = cfg.plot
    vmax = float(np.abs(matrix).max())
    fig, ax = plt.subplots(figsize=tuple(style.isfc_figsize))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    for i, j in np.ndindex(matrix.shape):
        color = "white" if abs(matrix[i, j]) > 0.6 * vmax else style.text_primary
        ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=style.label_size - 2, color=color)
    ax.set_title(title, loc="left", fontsize=style.title_size, color=style.text_primary)
    fig.colorbar(image, ax=ax, label="ISFC (r)", fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig

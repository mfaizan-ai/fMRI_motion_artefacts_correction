"""Training curves: every logged generator, discriminator and validation term, one panel each."""
import math

import matplotlib

matplotlib.use("Agg")  # headless: SLURM nodes and the login node have no display
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from omegaconf import DictConfig


def plot_panels(df: pd.DataFrame, panels: dict, title: str, y_label: str, smooth: bool,
                cfg: DictConfig) -> Figure | None:
    """One panel per logged column, each in its own colour: the raw curve faint, its rolling mean on top.

    Args:
        df: Per-epoch (train) or per-validation (val) log with an "epoch" column.
        panels: Column -> panel title; columns absent from df are skipped.
        title: Figure title.
        y_label: Label of every panel's y-axis.
        smooth: Draw the rolling mean; validation points are sparse, so they are drawn as markers instead.
        cfg: Composed `configs/plot_training.yaml`.

    Returns:
        The figure, or None if df has none of the columns.

    Raises:
        ValueError: If the group has more panels than cfg.colors.series has colours.
    """
    style = cfg.style
    if len(panels) > len(cfg.colors.series):
        raise ValueError(f"{len(panels)} panels but only {len(cfg.colors.series)} colours in cfg.colors.series")
    # colour by the column's slot in the config, not among the logged ones, so a loss keeps its colour across runs
    colors = dict(zip(panels, cfg.colors.series, strict=False))
    columns = [c for c in panels if c in df and df[c].notna().any()]
    if not columns:
        return None
    ncols = min(style.ncols, len(columns))
    nrows = math.ceil(len(columns) / ncols)
    width, height = style.panel_size
    fig, axes = plt.subplots(nrows, ncols, figsize=(width * ncols, height * nrows), squeeze=False,
                             constrained_layout=True)
    for ax, column in zip(axes.flat, columns, strict=False):
        values, color = df[column].astype(float), colors[column]
        if smooth:
            ax.plot(df["epoch"], values, color=color, alpha=style.raw_alpha, linewidth=1)
            trend = values.rolling(cfg.smooth_window, center=True, min_periods=1).mean()
            ax.plot(df["epoch"], trend, color=color, linewidth=style.line_width)
        else:
            ax.plot(df["epoch"], values, color=color, linewidth=style.line_width, marker="o",
                    markersize=style.marker_size)
        if column in cfg.log_scale:
            ax.set_yscale("log")
        ax.set_title(panels[column], loc="left", fontsize=style.title_size, color=cfg.colors.text_primary)
        ax.set_xlabel("Epoch", fontsize=style.label_size, color=cfg.colors.text_secondary)
        ax.set_ylabel(y_label, fontsize=style.label_size, color=cfg.colors.text_secondary)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(cfg.colors.axis)
        ax.tick_params(colors=cfg.colors.text_secondary, labelsize=style.tick_size)
        ax.grid(True, color=cfg.colors.grid, linewidth=0.6)
        ax.set_axisbelow(True)
    for ax in axes.flat[len(columns):]:
        ax.set_visible(False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=style.title_size + 2, fontweight="bold",
                 color=cfg.colors.text_primary)
    return fig


def plot_training(train: pd.DataFrame, val: pd.DataFrame, name: str, cfg: DictConfig) -> dict[str, Figure]:
    """Generator, discriminator and validation figures for one run.

    Args:
        train: train_losses.csv, one row per epoch.
        val: val_metrics.csv, one row per validation epoch.
        name: Run name for the figure titles.
        cfg: Composed `configs/plot_training.yaml`.

    Returns:
        File stem -> figure, for the groups that have at least one logged column.
    """
    figures = {
        "generator_losses": plot_panels(train, cfg.generator, f"{name}: generator losses", "Loss", True, cfg),
        "discriminator_losses": plot_panels(train, cfg.discriminator, f"{name}: discriminator losses", "Loss",
                                            True, cfg),
        "validation_metrics": plot_panels(val, cfg.validation, f"{name}: validation", "Value", False, cfg),
    }
    return {stem: fig for stem, fig in figures.items() if fig is not None}

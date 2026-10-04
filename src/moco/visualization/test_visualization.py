"""Box plots of the per-chunk test metrics by motion grade: raw vs corrected, and the improvement."""
import matplotlib

matplotlib.use("Agg")  # headless: SLURM nodes and the login node have no display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from omegaconf import DictConfig


def bootstrap_median_ci(values: np.ndarray, n_boot: int, ci: float, rng: np.random.Generator) -> tuple[float, float]:
    """Percentile bootstrap confidence interval of the median.

    Args:
        values: 1-D sample.
        n_boot: Number of bootstrap resamples.
        ci: Coverage, e.g. 0.95.
        rng: Random generator, so the interval is reproducible.

    Returns:
        (lower, upper) bounds of the interval.
    """
    resamples = rng.choice(values, size=(n_boot, len(values)), replace=True)
    medians = np.median(resamples, axis=1)
    alpha = (1 - ci) / 2
    return float(np.quantile(medians, alpha)), float(np.quantile(medians, 1 - alpha))


def median_ci_by_grade(df: pd.DataFrame, column: str, grades: list[str], n_boot: int, ci: float,
                       rng: np.random.Generator) -> pd.DataFrame:
    """Median and bootstrap CI of one column for each grade.

    Args:
        df: Per-chunk metrics with a "grade" column.
        column: Metric column to summarise.
        grades: Grades in plotting order.
        n_boot: Number of bootstrap resamples.
        ci: Coverage, e.g. 0.95.
        rng: Random generator.

    Returns:
        One row per grade with columns median, lower, upper.
    """
    rows = []
    for grade in grades:
        values = df.loc[df["grade"] == grade, column].to_numpy()
        lower, upper = bootstrap_median_ci(values, n_boot, ci, rng)
        rows.append({"grade": grade, "median": float(np.median(values)), "lower": lower, "upper": upper})
    return pd.DataFrame(rows).set_index("grade")


def _style_box(box: dict, faces: list[str], edge: str, alpha: float, cfg: DictConfig) -> None:
    for patch, face in zip(box["boxes"], faces, strict=True):
        patch.set(facecolor=face, edgecolor=edge, linewidth=cfg.style.line_width, alpha=alpha)
    for part in ("whiskers", "caps"):
        for line in box[part]:
            line.set(color=edge, linewidth=cfg.style.line_width)
    for line in box["medians"]:
        line.set(color=edge, linewidth=cfg.style.median_width)
    for flier in box["fliers"]:
        flier.set(marker="o", markersize=cfg.style.flier_size, markerfacecolor=edge, markeredgecolor="none",
                  alpha=cfg.style.flier_alpha)


def _median_line(ax: plt.Axes, x: np.ndarray, summary: pd.DataFrame, color: str, label: str | None,
                 cfg: DictConfig) -> None:
    # asymmetric error bars: the bootstrap CI of the median need not be centred on it
    yerr = np.vstack([summary["median"] - summary["lower"], summary["upper"] - summary["median"]])
    ax.errorbar(x, summary["median"], yerr=yerr, color=color, linewidth=cfg.style.trend_width,
                marker="o", markersize=cfg.style.marker_size, capsize=cfg.style.cap_size, zorder=4, label=label)


def _style_axes(ax: plt.Axes, cfg: DictConfig) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(cfg.colors.axis)
    ax.tick_params(colors=cfg.colors.text_secondary, labelsize=cfg.style.tick_size)
    ax.yaxis.grid(True, color=cfg.colors.grid, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_test_metrics(df: pd.DataFrame, cfg: DictConfig) -> Figure:
    """Six panels: each metric raw vs corrected (top row) and its improvement (bottom row), by grade.

    Args:
        df: `test_metrics_per_chunk.csv` from `scripts/evaluate.py`; needs grade, <metric>_input,
            <metric>_corrected and <metric>_improvement for every metric in cfg.metrics.
        cfg: Composed `configs/plot_test.yaml`.

    Returns:
        The figure; medians are joined across grades with their bootstrap CI.
    """
    rng = np.random.default_rng(cfg.seed)
    grades = sorted(df["grade"].unique(), key=lambda g: int(g.split()[-1]))  # "Grade 10" after "Grade 9"
    counts = df["grade"].value_counts()
    x = np.arange(len(grades))
    offset, width = cfg.style.pair_offset, cfg.style.box_width
    grade_colors = list(cfg.colors.grades)
    if len(grade_colors) < len(grades):
        raise ValueError(f"{len(grades)} grades but only {len(grade_colors)} colours in cfg.colors.grades")

    fig, axes = plt.subplots(2, len(cfg.metrics), figsize=tuple(cfg.style.figsize), squeeze=False)
    tick_labels = [f"{g}\n(n={counts[g]})" for g in grades]
    for col, (metric, labels) in enumerate(cfg.metrics.items()):
        ax_pair, ax_gain = axes[0, col], axes[1, col]

        for side, sign, color in (("input", -1, cfg.colors.raw), ("corrected", 1, cfg.colors.corrected)):
            column = f"{metric}_{side}"
            data = [df.loc[df["grade"] == g, column].to_numpy() for g in grades]
            box = ax_pair.boxplot(data, positions=x + sign * offset, widths=width, patch_artist=True)
            _style_box(box, [color] * len(grades), color, cfg.style.box_alpha, cfg)
            summary = median_ci_by_grade(df, column, grades, cfg.n_boot, cfg.ci, rng)
            _median_line(ax_pair, x + sign * offset, summary, color, cfg.legend[side], cfg)
        ax_pair.set_title(labels.name, color=cfg.colors.text_primary, fontsize=cfg.style.title_size, loc="left")
        ax_pair.set_ylabel(labels.name, color=cfg.colors.text_secondary, fontsize=cfg.style.label_size)

        column = f"{metric}_improvement"
        data = [df.loc[df["grade"] == g, column].to_numpy() for g in grades]
        box = ax_gain.boxplot(data, positions=x, widths=width, patch_artist=True)
        _style_box(box, grade_colors[: len(grades)], cfg.colors.text_secondary, cfg.style.gain_box_alpha, cfg)
        summary = median_ci_by_grade(df, column, grades, cfg.n_boot, cfg.ci, rng)
        _median_line(ax_gain, x, summary, cfg.colors.text_primary, None, cfg)
        ax_gain.axhline(0, color=cfg.colors.text_secondary, linewidth=0.8, linestyle="--", zorder=1)
        ax_gain.set_title(labels.improvement, color=cfg.colors.text_primary, fontsize=cfg.style.title_size,
                          loc="left")
        ax_gain.set_ylabel(labels.improvement_axis, color=cfg.colors.text_secondary, fontsize=cfg.style.label_size)

        for ax in (ax_pair, ax_gain):
            ax.set_xticks(x, tick_labels)
            _style_axes(ax, cfg)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=len(handles), frameon=False,
               fontsize=cfg.style.label_size, labelcolor=cfg.colors.text_secondary)
    fig.suptitle(cfg.title, x=0.01, ha="left", color=cfg.colors.text_primary, fontsize=cfg.style.suptitle_size)
    fig.text(0.01, 0.005, f"Boxes: quartiles, whiskers 1.5 IQR. Lines: median with {cfg.ci:.0%} bootstrap CI "
             f"({cfg.n_boot} resamples).", color=cfg.colors.text_secondary, fontsize=cfg.style.tick_size)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    return fig

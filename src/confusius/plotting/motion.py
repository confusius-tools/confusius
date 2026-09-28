"""Motion diagnostics plotting utilities."""

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from confusius._utils.motion_plotting import (
    FD_LABELS,
    OPTIMIZER_ITERATION_COLOR,
    OPTIMIZER_METRIC_COLOR,
    get_motion_diagnostic_label,
    get_motion_diagnostic_linewidth,
)

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.figure import Figure


def plot_motion_diagnostics(
    motion_df: "pd.DataFrame",
    *,
    figsize: tuple[float, float] | None = None,
) -> tuple["Figure", np.ndarray]:
    """Plot motion diagnostics from `create_motion_dataframe`.

    Parameters
    ----------
    motion_df : pandas.DataFrame
        Motion summary table, typically `result.attrs["motion_params"]` from
        [`register_volumewise`][confusius.registration.register_volumewise]. The
        function plots whichever standard columns are present.
    figsize : tuple of float, optional
        Figure size passed to Matplotlib. If not provided, a height is chosen from the
        number of panels.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the motion plots.
    axes : numpy.ndarray
        Array of Matplotlib axes, one per panel.

    Raises
    ------
    ValueError
        If `motion_df` contains none of the supported motion-diagnostics columns.
    """
    time = np.asarray(motion_df.index, dtype=float)
    panels: list[str] = []

    rotation_cols = [
        col for col in ["rotation", "rot_x", "rot_y", "rot_z"] if col in motion_df
    ]
    translation_cols = [
        col for col in ["trans_x", "trans_y", "trans_z"] if col in motion_df
    ]
    displacement_cols = [
        col for col in ["mean_fd", "max_fd", "rms_fd"] if col in motion_df
    ]
    has_metric = "final_metric_value" in motion_df
    has_iterations = "n_iterations" in motion_df

    if rotation_cols:
        panels.append("rotation")
    if translation_cols:
        panels.append("translation")
    if displacement_cols:
        panels.append("displacement")
    if has_metric or has_iterations:
        panels.append("optimizer")

    if not panels:
        raise ValueError(
            "motion_df does not contain any supported diagnostics columns."
        )

    if figsize is None:
        figsize = (9, 2 + 1.8 * len(panels))
    fig, axes = plt.subplots(
        len(panels), 1, figsize=figsize, sharex=True, constrained_layout=True
    )
    axes = np.atleast_1d(axes)

    panel_index = 0

    if rotation_cols:
        ax = axes[panel_index]
        for col in rotation_cols:
            label = get_motion_diagnostic_label(col)
            ax.plot(
                time,
                np.rad2deg(motion_df[col]),
                lw=get_motion_diagnostic_linewidth(col),
                label=label,
            )
        ax.set_ylabel("Rotation (deg)")
        ax.set_title("Motion estimates")
        if any(col != "rotation" for col in rotation_cols):
            ax.legend(frameon=False, ncol=len(rotation_cols))
        panel_index += 1

    if translation_cols:
        ax = axes[panel_index]
        for col in translation_cols:
            ax.plot(
                time,
                motion_df[col],
                lw=get_motion_diagnostic_linewidth(col),
                label=get_motion_diagnostic_label(col),
            )
        ax.set_ylabel("Translation (mm)")
        ax.legend(frameon=False, ncol=len(translation_cols))
        panel_index += 1

    if displacement_cols:
        ax = axes[panel_index]
        for col in displacement_cols:
            ax.plot(
                time,
                motion_df[col],
                lw=get_motion_diagnostic_linewidth(col),
                label=FD_LABELS[col],
            )
        ax.set_ylabel("Displacement (mm)")
        ax.legend(frameon=False, ncol=len(displacement_cols))
        panel_index += 1

    if has_metric or has_iterations:
        metric_color = OPTIMIZER_METRIC_COLOR
        iteration_color = OPTIMIZER_ITERATION_COLOR
        ax = axes[panel_index]
        ax.set_title("Optimizer summary")
        ax.tick_params(axis="x", colors="white")
        if has_metric:
            ax.plot(
                time,
                motion_df["final_metric_value"],
                color=metric_color,
                lw=1.8,
            )
            ax.set_ylabel("Final metric", color=metric_color)
            ax.tick_params(axis="y", colors=metric_color)
            ax.spines["left"].set_color(metric_color)
            ax.spines["right"].set_visible(False)
        if has_iterations:
            iter_ax = ax.twinx() if has_metric else ax
            iter_ax.plot(
                time,
                motion_df["n_iterations"],
                color=iteration_color,
                lw=1.2,
                alpha=0.9,
            )
            iter_ax.set_ylabel("Iterations", color=iteration_color)
            iter_ax.tick_params(axis="x", colors="white")
            iter_ax.tick_params(axis="y", colors=iteration_color)
            iter_ax.spines["right" if has_metric else "left"].set_color(iteration_color)
            if has_metric:
                iter_ax.spines["left"].set_visible(False)

    if motion_df.index.name == "time":
        time_units = motion_df.attrs.get("time_units", "s")
        xlabel = f"Time ({time_units})"
    else:
        xlabel = "Frame"
    axes[-1].set_xlabel(xlabel)
    return fig, axes

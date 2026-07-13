"""Motion diagnostics plotting utilities."""

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

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
            label = col.replace("rot_", "") if col != "rotation" else None
            ax.plot(time, np.rad2deg(motion_df[col]), lw=1.6, label=label)
        ax.set_ylabel("Rotation (deg)")
        ax.set_title("Motion estimates")
        if any(col != "rotation" for col in rotation_cols):
            ax.legend(frameon=False, ncol=len(rotation_cols))
        panel_index += 1

    if translation_cols:
        ax = axes[panel_index]
        for col in translation_cols:
            ax.plot(time, motion_df[col], lw=1.6, label=col.removeprefix("trans_"))
        ax.set_ylabel("Translation (mm)")
        ax.legend(frameon=False, ncol=len(translation_cols))
        panel_index += 1

    if displacement_cols:
        ax = axes[panel_index]
        for col in displacement_cols:
            label = {
                "mean_fd": "Mean FD",
                "max_fd": "Max FD",
                "rms_fd": "RMS FD",
            }[col]
            lw = 1.8 if col == "mean_fd" else 1.2
            ax.plot(time, motion_df[col], lw=lw, label=label)
        ax.set_ylabel("Displacement (mm)")
        ax.legend(frameon=False, ncol=len(displacement_cols))
        panel_index += 1

    if has_metric or has_iterations:
        metric_color = "#d93a54"
        iteration_color = "#3ad9a4"
        ax = axes[panel_index]
        ax.set_title("Optimizer summary")
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
            iter_ax.tick_params(axis="y", colors=iteration_color)
            iter_ax.spines["right" if has_metric else "left"].set_color(iteration_color)

    axes[-1].set_xlabel("Time (s)" if motion_df.index.name == "time" else "Frame")
    return fig, axes

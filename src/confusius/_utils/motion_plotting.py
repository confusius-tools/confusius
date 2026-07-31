"""Shared motion-diagnostics plotting constants."""

from typing import Any

from confusius._utils.colors import RED_DARK, TURQUOISE

OPTIMIZER_METRIC_COLOR = RED_DARK
"""Line and axis color for final optimizer metric values."""

OPTIMIZER_ITERATION_COLOR = TURQUOISE
"""Line and axis color for optimizer iteration counts."""

FD_LABELS = {
    "mean_fd": "Mean FD",
    "max_fd": "Max FD",
    "rms_fd": "RMS FD",
}
"""Display labels for framewise-displacement columns."""


def get_dark_motion_color_cycle() -> Any:
    """Return Matplotlib's dark-background color cycle for motion traces.

    Returns
    -------
    cycler.Cycler
        Color cycle used by Matplotlib's built-in `dark_background` style.
    """
    from matplotlib import style as mpl_style

    return mpl_style.library["dark_background"]["axes.prop_cycle"]


def get_motion_diagnostic_label(column: str) -> str | None:
    """Return the legend label for a motion-diagnostics column.

    Parameters
    ----------
    column : str
        Motion diagnostics column name.

    Returns
    -------
    str or None
        Legend label. Returns None for scalar in-plane 2D rotation, matching
        [`plot_motion_diagnostics`][confusius.plotting.plot_motion_diagnostics].
    """
    if column == "rotation":
        return None
    if column.startswith(("rot_", "trans_")):
        return column.rsplit("_", maxsplit=1)[-1]
    return FD_LABELS.get(column, column)


def get_motion_diagnostic_linewidth(column: str) -> float:
    """Return the line width for a motion-diagnostics column.

    Parameters
    ----------
    column : str
        Motion diagnostics column name.

    Returns
    -------
    float
        Matplotlib line width.
    """
    if column == "mean_fd":
        return 1.8
    if column in FD_LABELS:
        return 1.2
    return 1.6

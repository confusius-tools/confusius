"""Progress reporting protocol and plotters for `register_volumewise`."""

from __future__ import annotations

import warnings
from threading import Lock
from typing import TYPE_CHECKING, Protocol

import numpy as np
import numpy.typing as npt

from confusius._utils.motion_plotting import (
    OPTIMIZER_ITERATION_COLOR,
    OPTIMIZER_METRIC_COLOR,
    get_dark_motion_color_cycle,
    get_motion_diagnostic_label,
    get_motion_diagnostic_linewidth,
)
from confusius._utils.stack import find_stack_level
from confusius.registration.motion import (
    compute_framewise_displacement,
    extract_motion_parameters,
)

if TYPE_CHECKING:
    import xarray as xr
    from matplotlib.figure import Figure

    from confusius.registration import RegistrationDiagnostics


class VolumewiseRegistrationProgress(Protocol):
    """Duck-typed contract for `register_volumewise` progress reporting."""

    def frame_completed(
        self,
        frame_index: int,
        registered_frame: xr.DataArray,
        affine_matrix: npt.NDArray[np.floating],
        diagnostics: RegistrationDiagnostics,
    ) -> None:
        """Report that one frame finished and provide its registered output.

        Parameters
        ----------
        frame_index : int
            Index of the completed frame.
        registered_frame : xarray.DataArray
            Registered frame output.
        affine_matrix : (N+1, N+1) numpy.ndarray
            Affine transform estimated for the completed frame.
        diagnostics : confusius.registration.RegistrationDiagnostics
            Diagnostics collected for the completed frame.
        """
        ...

    def close(self) -> None:
        """Report that the full volumewise run has ended."""
        ...


class MatplotlibVolumewiseRegistrationProgressPlotter:
    """Plot volume-wise registration diagnostics in real time.

    Displays motion estimates, framewise displacement, final optimizer metric, and
    iteration count for each completed frame. Frames may finish out of order;
    diagnostics are always written at the corresponding frame index.

    Parameters
    ----------
    n_frames : int
        Number of frames that will be registered.
    reference : xarray.DataArray
        Spatial reference DataArray used to name motion parameters and compute
        framewise displacement.
    time_coords : array-like, optional
        Time coordinate values to show on the x-axis. If not provided, frame indices
        are used.
    time_units : str, optional
        Unit label for `time_coords`.
    redraw_every : int, default: 25
        Redraw the figure after every `redraw_every` completed volumes.
    """

    def __init__(
        self,
        n_frames: int,
        *,
        reference: xr.DataArray,
        time_coords: npt.ArrayLike | None = None,
        time_units: str | None = None,
        redraw_every: int = 25,
    ) -> None:
        import matplotlib
        import matplotlib.pyplot as plt

        self._n_frames = n_frames
        self._reference = reference
        self._affines: list[npt.NDArray[np.floating] | None] = [None] * n_frames
        self._motion_values: dict[str, npt.NDArray[np.floating]] = {}
        self._fd_values = {
            "mean_fd": np.full(n_frames, np.nan, dtype=float),
            "max_fd": np.full(n_frames, np.nan, dtype=float),
            "rms_fd": np.full(n_frames, np.nan, dtype=float),
        }
        self._metric_values = np.full(n_frames, np.nan, dtype=float)
        self._n_iterations = np.full(n_frames, np.nan, dtype=float)
        self._completed = np.zeros(n_frames, dtype=bool)
        self._redraw_every = max(1, redraw_every)
        self._lock = Lock()

        if time_coords is None:
            self._x = np.arange(n_frames)
            self._xlabel = "Frame"
        else:
            self._x = np.asarray(time_coords)
            self._xlabel = f"Time ({time_units})" if time_units else "Time"

        try:
            from IPython.core.getipython import get_ipython

            _ip = get_ipython()
            self._notebook = (
                _ip is not None and type(_ip).__name__ == "ZMQInteractiveShell"
            )
        except ImportError:
            self._notebook = False

        if not self._notebook:
            _non_interactive = {"agg", "pdf", "ps", "svg"}
            if matplotlib.get_backend().lower() in _non_interactive:
                warnings.warn(
                    f"The active matplotlib backend '{matplotlib.get_backend()}' is "
                    "non-interactive; the volumewise registration progress window "
                    "will not be visible. Set an interactive backend before calling "
                    "register_volumewise, e.g.: import matplotlib; "
                    "matplotlib.use('Qt5Agg')",
                    stacklevel=find_stack_level(),
                )
            plt.ion()

        self._fig, axes = plt.subplots(
            4,
            1,
            figsize=(9, 10),
            sharex=True,
            facecolor="black",
            constrained_layout=True,
        )
        (
            self._rotation_ax,
            self._translation_ax,
            self._fd_ax,
            self._optimizer_ax,
        ) = axes
        self._iteration_ax = self._optimizer_ax.twinx()
        self._rotation_lines = self._setup_motion_axis(
            self._rotation_ax,
            ["rotation"] if reference.ndim == 2 else ["rot_x", "rot_y", "rot_z"],
            ylabel="Rotation (deg)",
            title="Motion estimates",
        )
        self._translation_lines = self._setup_motion_axis(
            self._translation_ax,
            [f"trans_{dim}" for dim in ("x", "y", "z") if dim in reference.dims],
            ylabel="Translation (mm)",
        )
        self._fd_lines = self._setup_motion_axis(
            self._fd_ax,
            ["mean_fd", "max_fd", "rms_fd"],
            ylabel="Displacement (mm)",
        )
        (self._metric_line,) = self._optimizer_ax.plot(
            [], [], color=OPTIMIZER_METRIC_COLOR, lw=1.8, label="Final metric"
        )
        (self._iteration_line,) = self._iteration_ax.plot(
            [], [], color=OPTIMIZER_ITERATION_COLOR, lw=1.2, label="Iterations"
        )
        self._style_optimizer_axes()
        self._optimizer_ax.set_title("Optimizer summary", color="white", fontsize=10)
        self._optimizer_ax.set_xlabel(self._xlabel, color="white", fontsize=9)

    def _style_axis(self, ax, *, ylabel: str) -> None:
        """Apply compact progress-plot style to one axis.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis to style.
        ylabel : str
            Y-axis label.
        """
        ax.set_facecolor("black")
        ax.set_ylabel(ylabel, color="white", fontsize=9)
        ax.tick_params(colors="white", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("white")

    def _style_optimizer_axes(self) -> None:
        """Style the optimizer panel like `plot_motion_diagnostics`.

        Returns
        -------
        None
            The optimizer axes are updated in place.
        """
        metric_color = OPTIMIZER_METRIC_COLOR
        iteration_color = OPTIMIZER_ITERATION_COLOR
        self._optimizer_ax.set_facecolor("black")
        self._optimizer_ax.set_ylabel("Final metric", color=metric_color, fontsize=9)
        self._optimizer_ax.tick_params(axis="x", colors="white", labelsize=8)
        self._optimizer_ax.tick_params(axis="y", colors=metric_color, labelsize=8)
        self._optimizer_ax.spines["left"].set_color(metric_color)
        self._optimizer_ax.spines["bottom"].set_color("white")
        self._optimizer_ax.spines["top"].set_color("white")
        self._optimizer_ax.spines["right"].set_visible(False)

        self._iteration_ax.set_facecolor("none")
        self._iteration_ax.set_ylabel("Iterations", color=iteration_color, fontsize=9)
        self._iteration_ax.tick_params(axis="x", colors="white", labelsize=8)
        self._iteration_ax.tick_params(axis="y", colors=iteration_color, labelsize=8)
        self._iteration_ax.spines["right"].set_color(iteration_color)
        self._iteration_ax.spines["left"].set_visible(False)
        self._iteration_ax.spines["bottom"].set_color("white")
        self._iteration_ax.spines["top"].set_color("white")

    def _setup_motion_axis(
        self, ax, columns: list[str], *, ylabel: str, title: str = ""
    ):
        """Create line artists for one motion-diagnostics panel.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis to populate.
        columns : list[str]
            Data columns plotted on this axis.
        ylabel : str
            Y-axis label.
        title : str, default: ""
            Axis title.

        Returns
        -------
        dict
            Mapping from column name to matplotlib line artist.
        """
        self._style_axis(ax, ylabel=ylabel)
        ax.set_prop_cycle(get_dark_motion_color_cycle())
        if title:
            ax.set_title(title, color="white", fontsize=10)
        lines = {}
        for col in columns:
            (line,) = ax.plot(
                [],
                [],
                lw=get_motion_diagnostic_linewidth(col),
                label=get_motion_diagnostic_label(col),
            )
            lines[col] = line
        if any(not line.get_label().startswith("_") for line in lines.values()):
            ax.legend(
                frameon=False,
                ncols=max(1, len(lines)),
                labelcolor="white",
                fontsize=8,
            )
        return lines

    def frame_completed(
        self,
        frame_index: int,
        registered_frame: xr.DataArray,
        affine_matrix: npt.NDArray[np.floating],
        diagnostics: RegistrationDiagnostics,
    ) -> None:
        """Update the plot after one frame finishes.

        Parameters
        ----------
        frame_index : int
            Index of the completed frame.
        registered_frame : xarray.DataArray
            Registered frame output. Not plotted.
        affine_matrix : (N+1, N+1) numpy.ndarray
            Affine transform estimated for the completed frame.
        diagnostics : confusius.registration.RegistrationDiagnostics
            Diagnostics collected for the completed frame.
        """
        del registered_frame
        with self._lock:
            self._affines[frame_index] = affine_matrix
            self._completed[frame_index] = True
            self._metric_values[frame_index] = diagnostics.final_metric_value
            self._n_iterations[frame_index] = diagnostics.n_iterations
            self._update_motion_values(frame_index)
            completed_count = int(self._completed.sum())
            should_render = (
                completed_count == self._n_frames
                or completed_count % self._redraw_every == 0
            )
            if should_render:
                completed = self._completed.copy()
                motion_values = {
                    key: value.copy() for key, value in self._motion_values.items()
                }
                fd_values = {
                    key: value.copy() for key, value in self._fd_values.items()
                }
                metric_values = self._metric_values.copy()
                n_iterations = self._n_iterations.copy()

        if not should_render:
            return

        for col, line in self._rotation_lines.items():
            line.set_data(self._x, np.rad2deg(motion_values[col]))
        for col, line in self._translation_lines.items():
            line.set_data(self._x, motion_values[col])
        for col, line in self._fd_lines.items():
            line.set_data(self._x, fd_values[col])
        self._metric_line.set_data(self._x[completed], metric_values[completed])
        self._iteration_line.set_data(self._x[completed], n_iterations[completed])
        for ax in (
            self._rotation_ax,
            self._translation_ax,
            self._fd_ax,
            self._optimizer_ax,
            self._iteration_ax,
        ):
            ax.relim()
            ax.autoscale_view()
        self._render()

    def _update_motion_values(self, frame_index: int) -> None:
        """Update motion and FD buffers for one completed frame.

        Parameters
        ----------
        frame_index : int
            Index of the completed frame.
        """
        affine = self._affines[frame_index]
        if affine is None:
            return
        params = extract_motion_parameters([affine])[0]
        if affine.shape[0] == 3:
            values = {"rotation": params[0]}
            for dim in ("x", "y", "z"):
                if dim in self._reference.dims:
                    values[f"trans_{dim}"] = params[1 + self._reference.dims.index(dim)]
        else:
            values = {f"rot_{dim}": params[i] for i, dim in enumerate(("x", "y", "z"))}
            values.update(
                {f"trans_{dim}": params[3 + i] for i, dim in enumerate(("x", "y", "z"))}
            )
        for key, value in values.items():
            self._motion_values.setdefault(
                key, np.full(self._n_frames, np.nan, dtype=float)
            )[frame_index] = value

        for left in (frame_index - 1, frame_index):
            right = left + 1
            if left < 0 or right >= self._n_frames:
                continue
            left_affine = self._affines[left]
            right_affine = self._affines[right]
            if left_affine is None or right_affine is None:
                continue
            fd = compute_framewise_displacement(
                [left_affine, right_affine], self._reference
            )
            for key, values_arr in self._fd_values.items():
                values_arr[left] = fd[key][0]

    def close(self) -> None:
        """Finalize the plot when volume-wise registration ends."""
        self._render()
        if self._notebook:
            import matplotlib.pyplot as plt

            plt.close(self._fig)

    def _render(self) -> None:
        """Push pending drawing commands to the screen or notebook output."""
        if self._notebook:
            from IPython.display import display

            display(self._fig, clear=True)
        else:
            self._fig.canvas.draw_idle()
            self._fig.canvas.flush_events()

    @property
    def metric_values(self) -> npt.NDArray[np.floating]:
        """Final optimizer metric value recorded for each frame.

        Returns
        -------
        numpy.ndarray
            Copy of the internal metric value buffer.
        """
        return self._metric_values.copy()

    @property
    def n_iterations(self) -> npt.NDArray[np.floating]:
        """Number of optimizer iterations recorded for each frame.

        Returns
        -------
        numpy.ndarray
            Copy of the internal iteration-count buffer.
        """
        return self._n_iterations.copy()

    @property
    def figure(self) -> Figure:
        """The matplotlib figure used for plotting.

        Returns
        -------
        matplotlib.figure.Figure
            The figure instance owned by this plotter.
        """
        return self._fig

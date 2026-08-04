"""Napari widget for live volumewise registration diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from qtpy.QtCore import QSize, QTimer
from qtpy.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from confusius._napari._theme import get_napari_colors, style_plot_toolbar
from confusius._utils.motion_plotting import get_dark_motion_color_cycle
from confusius.registration.motion import (
    compute_framewise_displacement,
    extract_motion_parameters,
)

if TYPE_CHECKING:
    import xarray as xr
    from napari import Viewer

    from confusius.registration import RegistrationDiagnostics


class VolumewiseRegistrationDiagnosticsPlotter(QWidget):
    """Undocked widget for live volumewise registration diagnostics.

    Parameters
    ----------
    viewer : napari.Viewer
        Active napari viewer, used to follow theme changes.
    n_frames : int
        Number of frames expected in the registration run.
    reference : xarray.DataArray
        Spatial reference DataArray used to name motion parameters and compute FD.
    time_coords : array-like, optional
        Time coordinate values. If not provided, frame indices are used.
    time_units : str, optional
        Time coordinate unit label.
    redraw_every : int, default: 25
        Redraw the figure after every `redraw_every` completed volumes.
    """

    def __init__(
        self,
        viewer: Viewer,
        *,
        n_frames: int,
        reference: xr.DataArray,
        time_coords: npt.ArrayLike | None = None,
        time_units: str | None = None,
        redraw_every: int = 25,
    ) -> None:
        super().__init__()
        self._viewer = viewer
        self._n_frames = n_frames
        self._reference = reference
        self._redraw_every = max(1, redraw_every)
        self._affines: list[npt.NDArray[np.floating] | None] = [None] * n_frames
        self._completed = np.zeros(n_frames, dtype=bool)
        self._motion_values: dict[str, npt.NDArray[np.floating]] = {}
        self._fd_values = {
            "mean_fd": np.full(n_frames, np.nan, dtype=float),
            "max_fd": np.full(n_frames, np.nan, dtype=float),
            "rms_fd": np.full(n_frames, np.nan, dtype=float),
        }
        self._metric_values = np.full(n_frames, np.nan, dtype=float)
        self._n_iterations = np.full(n_frames, np.nan, dtype=float)
        if time_coords is None:
            self._x = np.arange(n_frames)
            self._xlabel = "Frame"
        else:
            self._x = np.asarray(time_coords)
            self._xlabel = f"Time ({time_units})" if time_units else "Time"

        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(16)
        self._render_timer.timeout.connect(self._render)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(QSize(850, 760))
        self._setup_ui()
        self._apply_theme()
        self._viewer.events.theme.connect(lambda *_: self._apply_theme())

    def sizeHint(self) -> QSize:
        """Return preferred initial widget size.

        Returns
        -------
        QSize
            Preferred widget size.
        """
        return QSize(900, 820)

    def _setup_ui(self) -> None:
        """Build the matplotlib canvas and axes."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(0)
        self._figure = Figure(constrained_layout=True)
        self._canvas = FigureCanvas(self._figure)
        self._toolbar = NavigationToolbar(self._canvas, self)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas)

        axes = self._figure.subplots(4, 1, sharex=True)
        (
            self._rotation_ax,
            self._translation_ax,
            self._fd_ax,
            self._optimizer_ax,
        ) = axes
        self._iteration_ax = self._optimizer_ax.twinx()
        self._rotation_lines = self._setup_motion_axis(
            self._rotation_ax,
            ["rotation"] if self._reference.ndim == 2 else ["rot_x", "rot_y", "rot_z"],
            ylabel="Rotation (deg)",
            title="Motion estimates",
        )
        self._translation_lines = self._setup_motion_axis(
            self._translation_ax,
            [f"trans_{dim}" for dim in ("x", "y", "z") if dim in self._reference.dims],
            ylabel="Translation (mm)",
        )
        self._fd_lines = self._setup_motion_axis(
            self._fd_ax,
            ["mean_fd", "max_fd", "rms_fd"],
            ylabel="Displacement (mm)",
        )
        (self._metric_line,) = self._optimizer_ax.plot([], [], color="#d93a54", lw=1.8)
        (self._iteration_line,) = self._iteration_ax.plot(
            [], [], color="#3ad9a4", lw=1.2, alpha=0.9
        )
        self._optimizer_ax.set_title("Optimizer summary")
        self._optimizer_ax.set_ylabel("Final metric")
        self._iteration_ax.set_ylabel("Iterations")
        self._optimizer_ax.set_xlabel(self._xlabel)

    def _setup_motion_axis(
        self, ax, columns: list[str], *, ylabel: str, title: str = ""
    ):
        """Create line artists for one panel.

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
            Mapping from column name to line artist.
        """
        ax.set_ylabel(ylabel)
        ax.set_prop_cycle(get_dark_motion_color_cycle())
        if title:
            ax.set_title(title)
        labels = {"mean_fd": "Mean FD", "max_fd": "Max FD", "rms_fd": "RMS FD"}
        lines = {}
        for col in columns:
            if col == "rotation":
                label = None
            elif col.startswith(("rot_", "trans_")):
                label = col.rsplit("_", maxsplit=1)[-1]
            else:
                label = labels.get(col, col)
            lw = 1.8 if col == "mean_fd" else 1.2 if col in labels else 1.6
            (line,) = ax.plot([], [], lw=lw, label=label)
            lines[col] = line
        if any(not line.get_label().startswith("_") for line in lines.values()):
            ax.legend(frameon=False, ncols=max(1, len(lines)), fontsize=8)
        return lines

    def _apply_theme(self) -> None:
        """Style the plot using the current napari theme."""
        colors = get_napari_colors(self._viewer.theme)
        self._figure.patch.set_facecolor(colors["bg"])
        axes = [
            self._rotation_ax,
            self._translation_ax,
            self._fd_ax,
            self._optimizer_ax,
            self._iteration_ax,
        ]
        for ax in axes:
            ax.set_facecolor(colors["bg"])
            ax.tick_params(colors=colors["fg"], labelsize=8)
            ax.xaxis.label.set_color(colors["fg"])
            ax.yaxis.label.set_color(colors["fg"])
            ax.title.set_color(colors["fg"])
            for spine in ax.spines.values():
                spine.set_edgecolor(colors["fg"])
            legend = ax.get_legend()
            if legend is not None:
                for text in legend.get_texts():
                    text.set_color(colors["fg"])
        self._optimizer_ax.yaxis.label.set_color("#d93a54")
        self._optimizer_ax.tick_params(axis="y", colors="#d93a54")
        self._optimizer_ax.spines["left"].set_color("#d93a54")
        self._optimizer_ax.spines["right"].set_visible(False)
        self._iteration_ax.set_facecolor("none")
        self._iteration_ax.yaxis.label.set_color("#3ad9a4")
        self._iteration_ax.tick_params(axis="y", colors="#3ad9a4")
        self._iteration_ax.spines["right"].set_color("#3ad9a4")
        self._iteration_ax.spines["left"].set_visible(False)
        style_plot_toolbar(self._toolbar, colors)
        self._canvas.draw_idle()

    def add_frame(
        self,
        frame_index: int,
        affine_matrix: npt.NDArray[np.floating],
        diagnostics: RegistrationDiagnostics,
    ) -> None:
        """Add diagnostics for one completed frame.

        Parameters
        ----------
        frame_index : int
            Completed frame index.
        affine_matrix : (N+1, N+1) numpy.ndarray
            Affine transform estimated for the completed frame.
        diagnostics : confusius.registration.RegistrationDiagnostics
            Registration diagnostics for the completed frame.
        """
        self._affines[frame_index] = affine_matrix
        self._completed[frame_index] = True
        self._metric_values[frame_index] = diagnostics.final_metric_value
        self._n_iterations[frame_index] = diagnostics.n_iterations
        self._update_motion_values(frame_index)
        completed_count = int(self._completed.sum())
        if (
            completed_count == self._n_frames
            or completed_count % self._redraw_every == 0
        ):
            self._schedule_render()

    def reset(self) -> None:
        """Clear all plotted diagnostics."""
        self._render_timer.stop()
        self._affines = [None] * self._n_frames
        self._completed[:] = False
        self._motion_values.clear()
        for values in self._fd_values.values():
            values[:] = np.nan
        self._metric_values[:] = np.nan
        self._n_iterations[:] = np.nan
        self._render()

    def _update_motion_values(self, frame_index: int) -> None:
        """Update cached motion values for one frame.

        Parameters
        ----------
        frame_index : int
            Completed frame index.
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

    def _schedule_render(self) -> None:
        """Schedule one coalesced redraw."""
        if not self._render_timer.isActive():
            self._render_timer.start()

    def _render(self) -> None:
        """Render all buffered diagnostics."""
        for col, line in self._rotation_lines.items():
            line.set_data(
                self._x,
                np.rad2deg(
                    self._motion_values.get(col, np.full(self._n_frames, np.nan))
                ),
            )
        for col, line in self._translation_lines.items():
            line.set_data(
                self._x, self._motion_values.get(col, np.full(self._n_frames, np.nan))
            )
        for col, line in self._fd_lines.items():
            line.set_data(self._x, self._fd_values[col])
        completed = self._completed.copy()
        self._metric_line.set_data(self._x[completed], self._metric_values[completed])
        self._iteration_line.set_data(self._x[completed], self._n_iterations[completed])
        for ax in [
            self._rotation_ax,
            self._translation_ax,
            self._fd_ax,
            self._optimizer_ax,
            self._iteration_ax,
        ]:
            ax.relim()
            ax.autoscale_view()
        self._canvas.draw_idle()

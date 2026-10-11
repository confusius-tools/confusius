"""Signals plotter widget for the bottom dock."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import napari
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from napari.utils.notifications import show_error, show_info
from qtpy.QtCore import QSize, QTimer, Signal
from qtpy.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from confusius._dims import TIME_DIM
from confusius._napari._export import (
    ExportSignal,
    prepare_export_signal,
    prompt_delimited_export_path,
    write_delimited_signals,
)
from confusius._napari._signals._store import SignalStore
from confusius._napari._theme import (
    create_export_button,
    get_napari_colors,
    style_export_button,
    style_plot_toolbar,
)

if TYPE_CHECKING:
    from qtpy.QtGui import QShowEvent

    from confusius._napari._events._store import EventStore


class LivePlotSignal(NamedTuple):
    """Raw live snapshot available for optional pinning.

    Attributes
    ----------
    origin : str
        Stable snapshot origin, including recording and mouse configuration.
    name : str
        Display name.
    x : numpy.ndarray
        Axis coordinates.
    y : numpy.ndarray
        Raw values, independent of display standardisation.
    color : str
        Signal colour.
    coordinate_dim : str
        Axis represented by the snapshot; defaults to time.
    """

    origin: str
    name: str
    x: np.ndarray
    y: np.ndarray
    color: str
    coordinate_dim: str = TIME_DIM


class SignalPlotter(QWidget):
    """Render shared live traces and stored signals without owning sources.

    Parameters
    ----------
    viewer : napari.Viewer
        Active viewer.
    store : SignalStore, optional
        Shared signal store. If not provided, create a viewer-local store.
    event_store : EventStore, optional
        Temporal events shaded on the plot.
    """

    frame_clicked = Signal(float)
    """Emitted with the clicked x-axis coordinate."""

    def __init__(
        self,
        viewer: napari.Viewer,
        store: SignalStore | None = None,
        event_store: EventStore | None = None,
    ) -> None:
        super().__init__()
        self._viewer = viewer
        self._signals_store = store if store is not None else SignalStore(self)
        self._live_sources = self._signals_store.live_sources(viewer)
        self._event_store = event_store
        self._closed = False
        self._event_spans = []
        self._xaxis_dim: str | None = None
        self._autoscale = True
        self._ylim_min: float | None = None
        self._ylim_max: float | None = None
        self._show_grid = True
        self._zscore = False
        self._show_cursor = False
        self._cursor_world = 0.0
        self._vline = None
        self._bg = None
        self._export_signals: list[ExportSignal] = []
        self._live_plot_signals: list[LivePlotSignal] = []
        self._signature: tuple = ()
        self._lines = []
        self._auto_xlim: tuple[float, float] | None = None
        self._axis_reference = None
        self._axis_coordinates: np.ndarray | None = None
        self._cursor_timer = QTimer(self)
        self._cursor_timer.setSingleShot(True)
        self._cursor_timer.setInterval(16)
        self._cursor_timer.timeout.connect(self._flush_cursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(200)
        self._setup_ui()
        self._signals_store.changed.connect(self._refresh_plot)
        self._live_sources.traces_changed.connect(self._refresh_plot)
        if event_store is not None:
            event_store.changed.connect(self._on_events_changed)
        self._apply_theme()
        self._refresh_plot()

    def sizeHint(self) -> QSize:
        """Return the preferred dock size.

        Returns
        -------
        qtpy.QtCore.QSize
            Preferred plot size.
        """
        return QSize(800, 300)

    def _setup_ui(self) -> None:
        """Construct the existing matplotlib canvas and toolbar."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(0)
        self._figure = Figure(tight_layout=True)
        self._canvas = FigureCanvas(self._figure)
        self._canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._toolbar = NavigationToolbar(self._canvas, self)
        self._export_button = create_export_button(
            self._toolbar, "signalExportButton", self._save_current_plot
        )
        self._toolbar.addWidget(self._export_button)
        self._pin_button = create_export_button(
            self._toolbar,
            "signalPinButton",
            self._pin_current_signals,
            text="Pin",
            tooltip="Snapshot the currently plotted live traces.",
        )
        self._toolbar.addWidget(self._pin_button)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas)
        self._axes = self._figure.add_subplot(111)
        self._canvas.mpl_connect("draw_event", self._on_draw)
        self._canvas.mpl_connect("button_press_event", self._on_click)

    def _apply_theme(self) -> None:
        """Apply viewer colours without changing source colours."""
        colors = get_napari_colors(self._viewer.theme)
        self._figure.set_facecolor(colors["bg"])
        self._axes.set_facecolor(colors["bg"])
        style_plot_toolbar(self._toolbar, colors)
        style_export_button(self._export_button, colors)
        style_export_button(self._pin_button, colors)
        self._signature = ()

    def on_theme_changed(self) -> None:
        """Refresh plot styling for the current viewer theme."""
        self._apply_theme()
        self._refresh_plot()

    def set_ylim(self, min_val: float | None, max_val: float | None) -> None:
        """Set manual ordinate bounds.

        Parameters
        ----------
        min_val : float, optional
            Lower bound.
        max_val : float, optional
            Upper bound.
        """
        self._ylim_min, self._ylim_max = min_val, max_val
        self._refresh_plot()

    def get_ylim(self) -> tuple[float, float]:
        """Return current ordinate bounds.

        Returns
        -------
        tuple[float, float]
            Lower and upper bounds.
        """
        return self._axes.get_ylim()

    def set_autoscale(self, enabled: bool) -> None:
        """Set ordinate autoscaling.

        Parameters
        ----------
        enabled : bool
            Whether to autoscale.
        """
        self._autoscale = enabled
        self._refresh_plot()

    def set_show_grid(self, enabled: bool) -> None:
        """Set grid visibility.

        Parameters
        ----------
        enabled : bool
            Whether to show the grid.
        """
        self._show_grid = enabled
        self._refresh_plot()

    def set_zscore(self, enabled: bool) -> None:
        """Set display-only standardisation.

        Parameters
        ----------
        enabled : bool
            Whether to standardise plotted values.
        """
        self._zscore = enabled
        self._refresh_plot()

    def set_show_cursor(self, enabled: bool) -> None:
        """Set x-axis cursor visibility.

        Parameters
        ----------
        enabled : bool
            Whether to show the cursor.
        """
        self._show_cursor = enabled
        self._signature = ()
        self._refresh_plot()

    def set_xaxis_dim(self, dim: str | None) -> None:
        """Set the display axis without changing FC's time extraction.

        Parameters
        ----------
        dim : str, optional
            Axis name. If not provided, use time.
        """
        if dim != self._xaxis_dim:
            self._xaxis_dim = dim
            self._live_sources.set_display_axis(dim or TIME_DIM)
            self._auto_xlim = None
            self._signature = ()
            self._refresh_plot()

    def set_xaxis_cursor(self, world_value: float) -> None:
        """Update the cursor in coordinate units.

        Parameters
        ----------
        world_value : float
            Current viewer axis coordinate.
        """
        self._cursor_world = world_value
        if not self._cursor_timer.isActive():
            self._cursor_timer.start()

    @staticmethod
    def _apply_zscore(ts: np.ndarray) -> np.ndarray:
        """Standardise finite values while preserving missing samples.

        Parameters
        ----------
        ts : numpy.ndarray
            Raw values.

        Returns
        -------
        numpy.ndarray
            Standardised values; constants become zero.
        """
        values = np.asarray(ts, dtype=float)
        finite = np.isfinite(values)
        result = values.copy()
        if finite.any():
            std = values[finite].std()
            result[finite] = (
                (values[finite] - values[finite].mean()) / std if std > 0 else 0
            )
        return result

    def _refresh_plot(self) -> None:
        """Update existing lines in place when only live values change."""
        colors = get_napari_colors(self._viewer.theme)
        plotted = []
        snapshots = []
        axis = self._xaxis_dim or TIME_DIM
        xlabel = axis.title()
        self._axis_reference = None
        self._axis_coordinates = None
        for live in self._signals_store.visible_live_signals():
            try:
                x, y = self._live_sources.get_trace(live.id, axis)
            except ValueError:
                continue
            if self._axis_reference is None:
                self._axis_reference = live.reference_layer
                self._axis_coordinates = x
                assert live.reference_layer is not None
                coord = live.reference_layer.metadata["xarray"][axis]
                xlabel = coord.attrs.get("long_name", axis.title())
                unit = coord.attrs.get("units")
                if unit:
                    xlabel += f" ({unit})"
            plotted.append((live.id, live.display_name, live.color, x, y))
            snapshots.append(
                LivePlotSignal(
                    self._live_sources.pin_origin(live.id, axis),
                    live.display_name,
                    x,
                    y,
                    live.color,
                    axis,
                )
            )
        for stored in self._signals_store.visible_stored_signals():
            plotted.append((stored.id, stored.name, stored.color, stored.x, stored.y))
        self._set_live_plot_signals(snapshots)
        self._set_export_signals(
            [
                ExportSignal(name, x, self._apply_zscore(y) if self._zscore else y)
                for _, name, _, x, y in plotted
            ]
        )
        signature = tuple((sid, name, color) for sid, name, color, _, _ in plotted)
        old_xlim = self._axes.get_xlim()
        user_xlim = self._auto_xlim is not None and not np.allclose(
            old_xlim, self._auto_xlim
        )
        rebuild = signature != self._signature or not self._lines
        if rebuild:
            self._axes.clear()
            self._event_spans = []
            self._vline = None
            self._lines = []
            for _, name, color, x, y in plotted:
                values = self._apply_zscore(y) if self._zscore else y
                self._lines.extend(
                    self._axes.plot(x, values, color=color, label=name, linewidth=1.5)
                )
            self._signature = signature
            if plotted:
                legend = self._axes.legend(
                    fontsize=8, facecolor=colors["bg"], edgecolor=colors["fg"]
                )
                for text in legend.get_texts():
                    text.set_color(colors["fg"])
            else:
                self._axes.text(
                    0.5,
                    0.5,
                    "Enable a source and hold Shift over a recording,\nor add a Points/Labels source.",
                    transform=self._axes.transAxes,
                    ha="center",
                    va="center",
                    color=colors["fg"],
                )
            self._draw_event_spans()
        else:
            for line, (_, _, _, x, y) in zip(self._lines, plotted):
                line.set_data(x, self._apply_zscore(y) if self._zscore else y)
        self._axes.set_facecolor(colors["bg"])
        self._axes.tick_params(colors=colors["fg"])
        for spine in self._axes.spines.values():
            spine.set_color(colors["fg"])
        self._axes.set_xlabel(xlabel, color=colors["fg"])
        self._axes.set_ylabel(
            "Z-score" if self._zscore else "Intensity", color=colors["fg"]
        )
        self._axes.grid(self._show_grid, alpha=0.2)
        if plotted:
            self._axes.relim()
            self._axes.autoscale_view(scalex=False, scaley=self._autoscale)
            if not self._autoscale:
                self._axes.set_ylim(self._ylim_min, self._ylim_max)
            numeric = [
                np.asarray(x)
                for _, _, _, x, _ in plotted
                if np.issubdtype(np.asarray(x).dtype, np.number) and len(x)
            ]
            if numeric:
                extent = (
                    float(min(np.nanmin(x) for x in numeric)),
                    float(max(np.nanmax(x) for x in numeric)),
                )
                if extent[0] != extent[1]:
                    self._axes.set_xlim(old_xlim if user_xlim else extent)
                    self._auto_xlim = extent
        else:
            self._auto_xlim = None
        if self._show_cursor and plotted and self._vline is None:
            self._vline = self._axes.axvline(
                self._cursor_plot_value(), color=colors["fg"], alpha=0.6, animated=True
            )
        self._bg = None
        self._canvas.draw_idle()

    def _set_live_plot_signals(self, signals: list[LivePlotSignal]) -> None:
        """Set optional pinning snapshots.

        Parameters
        ----------
        signals : list[LivePlotSignal]
            Currently plotted raw traces.
        """
        self._live_plot_signals = signals
        self._pin_button.setEnabled(bool(signals))

    def _pin_current_signals(self) -> None:
        """Snapshot plotted raw traces, independent of display standardisation."""
        for signal in list(self._live_plot_signals):
            self._signals_store.pin_signal(
                signal.origin,
                f"{signal.name} (pinned)",
                signal.x,
                signal.y,
                signal.color,
                f"Pinned from {signal.name}",
                coordinate_dim=signal.coordinate_dim,
            )

    def _set_export_signals(self, signals: list[ExportSignal]) -> None:
        """Prepare currently displayed traces for export.

        Parameters
        ----------
        signals : list[ExportSignal]
            Displayed traces.
        """
        self._export_signals = prepare_export_signal(signals)
        self._export_button.setEnabled(bool(self._export_signals))

    def _write_current_plot_delimited(self, path: Path, delimiter: str) -> None:
        """Write the current displayed values.

        Parameters
        ----------
        path : pathlib.Path
            Destination.
        delimiter : str
            Column delimiter.
        """
        write_delimited_signals(path, self._export_signals, delimiter=delimiter)

    def _save_current_plot(self) -> None:
        """Prompt for a delimited export path and write current traces."""
        selection = prompt_delimited_export_path(
            self, "Export Signals", str(Path.home() / "signals.tsv")
        )
        if selection is None:
            return
        path, delimiter = selection
        try:
            self._write_current_plot_delimited(path, delimiter)
        except (OSError, ValueError) as exc:
            show_error(str(exc))
        else:
            show_info(f"Exported signals to {path}")

    def _on_click(self, event) -> None:
        """Navigate from a left click in the plot.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            Canvas mouse event.
        """
        if (
            not self._toolbar.mode
            and event.button == 1
            and event.inaxes is self._axes
            and event.xdata is not None
        ):
            value = float(event.xdata)
            coordinates = self._axis_coordinates
            if (
                self._axis_reference is not None
                and coordinates is not None
                and not np.issubdtype(coordinates.dtype, np.number)
            ):
                axis = self._axis_reference.metadata["xarray"].get_axis_num(
                    self._xaxis_dim or TIME_DIM
                )
                index = int(np.clip(round(value), 0, len(coordinates) - 1))
                value = (
                    index * self._axis_reference.scale[axis]
                    + self._axis_reference.translate[axis]
                )
            self.frame_clicked.emit(float(value))

    def _on_draw(self, event) -> None:
        """Capture the cursor blit background.

        Parameters
        ----------
        event : matplotlib.backend_bases.DrawEvent
            Canvas draw event.
        """
        if self._vline is not None:
            self._bg = self._canvas.copy_from_bbox(self._axes.bbox)
            self._flush_cursor()

    def _flush_cursor(self) -> None:
        """Blit the time cursor without redrawing all signal lines."""
        if self._vline is None:
            return
        value = self._cursor_plot_value()
        self._vline.set_xdata([value, value])
        if self._bg is not None:
            self._canvas.restore_region(self._bg)
            self._axes.draw_artist(self._vline)
            self._canvas.blit(self._axes.bbox)

    def _cursor_plot_value(self) -> float:
        """Map the world cursor onto a categorical display axis when needed.

        Returns
        -------
        float
            Numeric plot position for the cursor.
        """
        coordinates = self._axis_coordinates
        layer = self._axis_reference
        if (
            layer is None
            or coordinates is None
            or np.issubdtype(coordinates.dtype, np.number)
        ):
            return self._cursor_world
        axis = layer.metadata["xarray"].get_axis_num(self._xaxis_dim or TIME_DIM)
        index = int(
            np.clip(
                round((self._cursor_world - layer.translate[axis]) / layer.scale[axis]),
                0,
                len(coordinates) - 1,
            )
        )
        return float(self._axes.convert_xunits(coordinates[index]))

    def _on_events_changed(self) -> None:
        """Refresh temporal event shading."""
        self._draw_event_spans()
        self._bg = None
        self._canvas.draw_idle()

    def _draw_event_spans(self) -> None:
        """Shade stored temporal events when the display axis is time."""
        for span in self._event_spans:
            span.remove()
        self._event_spans = []
        store = self._event_store
        if (
            store is None
            or not store.shade_signals
            or self._xaxis_dim not in (None, TIME_DIM)
            or not self._lines
        ):
            return
        for onset, duration, trial_type in store.iter_events():
            color = store.color_for(trial_type)
            span = (
                self._axes.axvspan(
                    onset,
                    onset + duration,
                    color=color,
                    alpha=0.15,
                    linewidth=0,
                    zorder=0,
                )
                if duration > 0
                else self._axes.axvline(
                    onset, color=color, alpha=0.6, linewidth=1, zorder=0
                )
            )
            self._event_spans.append(span)

    def closeEvent(self, a0) -> None:
        """Detach rendering callbacks; shared sources remain available to FC.

        Parameters
        ----------
        a0 : qtpy.QtGui.QCloseEvent
            Close event.
        """
        if not self._closed:
            self._cursor_timer.stop()
            self._signals_store.changed.disconnect(self._refresh_plot)
            self._live_sources.traces_changed.disconnect(self._refresh_plot)
            if self._event_store is not None:
                self._event_store.changed.disconnect(self._on_events_changed)
            self._closed = True
        super().closeEvent(a0)

    def showEvent(self, a0: QShowEvent | None) -> None:
        """Reconnect rendering when a closed plot dock is shown again.

        Parameters
        ----------
        a0 : qtpy.QtGui.QShowEvent, optional
            Qt show event.
        """
        if self._closed:
            self._signals_store.changed.connect(self._refresh_plot)
            self._live_sources.traces_changed.connect(self._refresh_plot)
            if self._event_store is not None:
                self._event_store.changed.connect(self._on_events_changed)
            self._closed = False
            self._refresh_plot()
        super().showEvent(a0)

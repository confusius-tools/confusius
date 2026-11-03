"""Signals plotter widget for the bottom dock."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import napari
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.patches import Patch
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
    from matplotlib.artist import Artist
    from qtpy.QtGui import QShowEvent

    from confusius._napari._events._store import EventStore


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

    axis_range_changed = Signal(object)
    """Emitted when the finite numeric trace extent changes."""

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
        self._baseline_spans = []
        self._baseline_warning = None
        self._extraction_warning = None
        self._axis_range: tuple[float, float] | None = None
        self._xaxis_dim: str | None = None
        self._autoscale = True
        self._ylim_min: float | None = None
        self._ylim_max: float | None = None
        self._show_grid = True
        self._standardization = "raw"
        self._baseline_intervals: list[tuple[float, float]] = []
        self._baseline_source: Literal["manual", "events"] = "manual"
        self._baseline_outside = False
        self._show_cursor = True
        self._cursor_world = 0.0
        self._vline = None
        self._bg = None
        self._export_signals: list[ExportSignal] = []
        self._signature: tuple | None = None
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
        self._signature = None

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
        self.set_standardization("zscore" if enabled else "raw")

    def get_axis_range(self) -> tuple[float, float] | None:
        """Return finite numeric trace bounds, independent of zoom.

        Returns
        -------
        tuple[float, float] or None
            Numeric extent. No numeric trace coordinates yields no bounds.
        """
        return self._axis_range

    def set_standardization(
        self,
        method: Literal["raw", "zscore", "psc"],
        baseline_intervals: list[tuple[float, float]] | None = None,
        *,
        baseline_source: Literal["manual", "events"] = "manual",
        outside: bool = False,
    ) -> None:
        """Set display/export standardization without changing raw traces.

        Parameters
        ----------
        method : {"raw", "zscore", "psc"}
            Display transformation.
        baseline_intervals : list[tuple[float, float]], optional
            Inclusive PSC baseline intervals in numeric x-axis units. If not
            provided, use all finite samples in manual mode. Overlapping
            intervals count once. Event intervals are half-open and in seconds.
        baseline_source : {"manual", "events"}, default: "manual"
            Manual intervals or an explicitly selected event union. Events require
            a numeric time axis in seconds; an empty selection is undefined.
        outside : bool, default: False
            Whether to use the complement of the event union within each trace.
            Ignored for manual intervals.

        Raises
        ------
        ValueError
            If the method or interval bounds are invalid.
        """
        if method not in {"raw", "zscore", "psc"}:
            raise ValueError("Unknown standardization method.")
        if baseline_source not in {"manual", "events"}:
            raise ValueError("Unknown baseline source.")
        intervals = list(baseline_intervals or [])
        if any(
            not np.isfinite([start, end]).all() or start > end
            for start, end in intervals
        ):
            raise ValueError("Baseline intervals must have finite ordered bounds.")
        if baseline_source == "events":
            intervals = [(start, end) for start, end in intervals if start < end]
        self._standardization = method
        self._baseline_intervals = intervals
        self._baseline_source = baseline_source
        self._baseline_outside = outside
        self._refresh_plot()

    def _standardize(self, x: np.ndarray, y: np.ndarray, signal_id: str) -> np.ndarray:
        """Transform one trace for display and export.

        Parameters
        ----------
        x : numpy.ndarray
            Axis coordinates.
        y : numpy.ndarray
            Raw samples.
        signal_id : str
            Signal identity used to validate event-based time coordinates.

        Returns
        -------
        numpy.ndarray
            Display values. Undefined PSC baselines produce missing values.
        """
        if self._standardization == "raw":
            return y
        if self._standardization == "zscore":
            return self._apply_zscore(y)
        values = np.asarray(y, dtype=float)
        baseline = np.isfinite(values)
        events = self._baseline_source == "events"
        if events and (
            not self._baseline_intervals or not self._has_seconds_axis(signal_id)
        ):
            return np.full_like(values, np.nan)
        if self._baseline_intervals:
            coordinates = np.asarray(x)
            if not np.issubdtype(coordinates.dtype, np.number):
                return np.full_like(values, np.nan)
            selected = np.zeros(len(values), dtype=bool)
            for start, end in self._baseline_intervals:
                # Events follow the Events tab's half-open interval semantics.
                upper = coordinates < end if events else coordinates <= end
                selected |= (coordinates >= start) & upper
            baseline &= ~selected if events and self._baseline_outside else selected
            baseline &= np.isfinite(coordinates)
        if not baseline.any():
            return np.full_like(values, np.nan)
        mean = values[baseline].mean()
        if abs(mean) < np.finfo(np.float64).eps:
            return np.full_like(values, np.nan)
        result = (values - mean) / abs(mean) * 100
        result[~np.isfinite(values)] = np.nan
        return result

    def _has_seconds_axis(self, signal_id: str) -> bool:
        """Check event-baseline compatibility without assuming non-time units.

        Parameters
        ----------
        signal_id : str
            Live or stored signal identity.

        Returns
        -------
        bool
            Whether the signal represents time in seconds. Imported time columns
            without unit metadata are treated as seconds, like BIDS events.
        """
        if self._xaxis_dim not in (None, TIME_DIM):
            return False
        live = self._signals_store.get_live_signal(signal_id)
        if live is not None and live.reference_layer is not None:
            data = live.reference_layer.metadata["xarray"]
            if TIME_DIM not in data.dims:
                return False
            return data[TIME_DIM].attrs.get("units", "s") in {"s", "second", "seconds"}
        stored = next(
            (s for s in self._signals_store.stored_signals() if s.id == signal_id), None
        )
        return stored is not None and stored.coordinate_dim == TIME_DIM

    def set_show_cursor(self, enabled: bool) -> None:
        """Set x-axis cursor visibility.

        Parameters
        ----------
        enabled : bool
            Whether to show the cursor.
        """
        self._show_cursor = enabled
        self._signature = None
        self._refresh_plot()

    def set_xaxis_dim(self, dim: str | None) -> None:
        """Set the display axis without changing FC's time extraction.

        Parameters
        ----------
        dim : str, optional
            Axis name. If not provided, use time. An empty name suspends
            plotting when the active image has no plottable slider axis.
        """
        if dim != self._xaxis_dim:
            self._xaxis_dim = dim
            if dim != "":
                self._live_sources.set_display_axis(dim or TIME_DIM)
            self._auto_xlim = None
            self._signature = None
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
        errors = []
        axis = self._xaxis_dim or TIME_DIM
        xlabel = axis.title() if self._xaxis_dim != "" else "No plottable axis"
        self._axis_reference = None
        self._axis_coordinates = None
        for live in (
            self._signals_store.visible_live_signals() if self._xaxis_dim != "" else []
        ):
            try:
                x, y = self._live_sources.get_trace(live.id, axis)
            except ValueError as exc:
                if live.source_type != "mouse":
                    errors.append(f"{live.display_name}: {exc}")
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
            plotted.append(
                (live.id, live.display_name, live.color, live.line_style, x, y)
            )
        for stored in (
            self._signals_store.visible_stored_signals()
            if self._xaxis_dim != ""
            else []
        ):
            plotted.append(
                (
                    stored.id,
                    stored.name,
                    stored.color,
                    stored.line_style,
                    stored.x,
                    stored.y,
                )
            )
        self._set_export_signals(
            [
                ExportSignal(name, x, self._standardize(x, y, sid))
                for sid, name, _, _, x, y in plotted
            ]
        )
        signature = (
            self._standardization,
            tuple(
                (sid, name, color, style) for sid, name, color, style, _, _ in plotted
            ),
        )
        old_xlim = self._axes.get_xlim()
        user_xlim = self._auto_xlim is not None and not np.allclose(
            old_xlim, self._auto_xlim
        )
        rebuild = signature != self._signature or not self._lines
        if rebuild:
            self._axes.clear()
            self._event_spans = []
            self._baseline_spans = []
            self._baseline_warning = None
            self._extraction_warning = None
            self._vline = None
            self._lines = []
            for sid, name, color, style, x, y in plotted:
                values = self._standardize(x, y, sid)
                self._lines.extend(
                    self._axes.plot(
                        x,
                        values,
                        color=color,
                        linestyle=style,
                        label=name,
                        linewidth=1.5,
                    )
                )
            self._signature = signature
            if plotted:
                handles: list[Artist] = list(self._lines)
                if self._standardization == "psc":
                    handles.append(
                        Patch(
                            facecolor=colors["accent"], alpha=0.15, label="PSC baseline"
                        )
                    )
                legend = self._axes.legend(
                    handles=handles,
                    fontsize=8,
                    facecolor=colors["bg"],
                    edgecolor=colors["fg"],
                )
                for text in legend.get_texts():
                    text.set_color(colors["fg"])
            else:
                self._axes.text(
                    0.5,
                    0.5,
                    "Select an image with a slider axis."
                    if self._xaxis_dim == ""
                    else "Hold Shift over a recording,\nor create a Points/Labels layer.",
                    transform=self._axes.transAxes,
                    ha="center",
                    va="center",
                    color=colors["fg"],
                )
            self._draw_event_spans()
        else:
            for line, (sid, _, _, _, x, y) in zip(self._lines, plotted):
                line.set_data(x, self._standardize(x, y, sid))
        self._axes.set_facecolor(colors["bg"])
        self._axes.tick_params(colors=colors["fg"])
        for spine in self._axes.spines.values():
            spine.set_color(colors["fg"])
        self._axes.set_xlabel(xlabel, color=colors["fg"])
        self._axes.set_ylabel(
            {"raw": "Intensity", "zscore": "Z-score", "psc": "Signal change (%)"}[
                self._standardization
            ],
            color=colors["fg"],
        )
        # Supplying line properties with False silently enables matplotlib's grid.
        if self._show_grid:
            self._axes.grid(True, alpha=0.2)
        else:
            self._axes.grid(False)
        numeric = [
            np.asarray(x)[np.isfinite(x)]
            for _, _, _, _, x, _ in plotted
            if np.issubdtype(np.asarray(x).dtype, np.number)
        ]
        numeric = [x for x in numeric if len(x)]
        extent = (
            (float(min(x.min() for x in numeric)), float(max(x.max() for x in numeric)))
            if numeric
            else None
        )
        if plotted:
            self._axes.relim()
            self._axes.autoscale_view(scalex=False, scaley=self._autoscale)
            if not self._autoscale:
                self._axes.set_ylim(self._ylim_min, self._ylim_max)
            if extent is not None and extent[0] != extent[1]:
                self._axes.set_xlim(old_xlim if user_xlim else extent)
                self._auto_xlim = extent
        else:
            self._auto_xlim = None
        if self._show_cursor and plotted and self._vline is None:
            self._vline = self._axes.axvline(
                self._cursor_plot_value(), color=colors["fg"], alpha=0.6, animated=True
            )
        self._draw_baseline_spans(extent)
        if self._extraction_warning is not None:
            self._extraction_warning.remove()
            self._extraction_warning = None
        if errors:
            self._extraction_warning = self._axes.text(
                0.01,
                0.99,
                "Unavailable traces:\n" + "\n".join(errors),
                transform=self._axes.transAxes,
                va="top",
                fontsize=8,
                color=colors["fg"],
            )
        self._bg = None
        self._canvas.draw_idle()
        if extent != self._axis_range:
            self._axis_range = extent
            self.axis_range_changed.emit(extent)

    def _draw_baseline_spans(self, extent: tuple[float, float] | None) -> None:
        """Shade PSC baseline intervals and explain undefined percent changes.

        Parameters
        ----------
        extent : tuple[float, float], optional
            Full numeric axis range, used when no custom baseline is selected.
        """
        for span in self._baseline_spans:
            span.remove()
        self._baseline_spans = []
        if self._baseline_warning is not None:
            self._baseline_warning.remove()
            self._baseline_warning = None
        if self._standardization != "psc":
            return
        colors = get_napari_colors(self._viewer.theme)
        intervals = self._baseline_intervals
        if self._baseline_source == "manual" and not intervals:
            intervals = [extent] if extent is not None else []
        event_axis_valid = (
            self._xaxis_dim in (None, TIME_DIM)
            and extent is not None
            and (
                self._axis_reference is None
                or self._axis_reference.metadata["xarray"][TIME_DIM].attrs.get(
                    "units", "s"
                )
                in {"s", "second", "seconds"}
            )
        )
        if self._baseline_source == "events":
            if not event_axis_valid:
                intervals = []
            elif extent is not None:
                intervals = [
                    (max(start, extent[0]), min(end, extent[1]))
                    for start, end in intervals
                    if start < extent[1] and end > extent[0]
                ]
        # Merge overlapping intervals so their shading does not become darker.
        merged: list[tuple[float, float]] = []
        for start, end in sorted(intervals):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        if (
            self._baseline_source == "events"
            and self._baseline_outside
            and self._baseline_intervals
            and extent is not None
            and event_axis_valid
        ):
            complement = []
            cursor = extent[0]
            for start, end in merged:
                if cursor < start:
                    complement.append((cursor, start))
                cursor = max(cursor, end)
            if cursor < extent[1]:
                complement.append((cursor, extent[1]))
            merged = complement
        for start, end in merged:
            self._baseline_spans.append(
                self._axes.axvspan(
                    start,
                    end,
                    color=colors["accent"],
                    alpha=0.15,
                    label="PSC baseline" if not self._baseline_spans else "_baseline",
                )
            )
        undefined = [
            str(line.get_label())
            for line in self._lines
            if not np.isfinite(line.get_ydata()).any()
        ]
        if undefined:
            self._baseline_warning = self._axes.text(
                0.01,
                0.01,
                "PSC undefined: baseline has no finite samples or a zero mean.\n"
                + (
                    "Event baselines require selected periods and a seconds time axis.\n"
                    if self._baseline_source == "events"
                    else ""
                )
                + "\n".join(undefined),
                transform=self._axes.transAxes,
                va="bottom",
                fontsize=8,
                color=colors["fg"],
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

"""Shared helpers for resolving the time axis of napari layers.

These functions centralise the logic for finding the viewer's time axis and for
reading the true time coordinate value (from xarray metadata) at the current
viewer position. They are used by both the timestamp overlay and the temporal
event annotation feature so the two stay consistent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from confusius._dims import TIME_DIM

if TYPE_CHECKING:
    import napari
    from napari.layers import Layer


def find_time_dim_index(viewer: napari.Viewer) -> int | None:
    """Return the viewer axis index for "time", or `None` if absent.

    napari does not propagate `axis_labels` from layers to `viewer.dims`, so we
    inspect each layer's labels directly and map the layer-local index to the
    viewer axis (layers are right-aligned in the viewer dims).

    Parameters
    ----------
    viewer : napari.Viewer
        The active napari viewer instance.

    Returns
    -------
    int | None
        Viewer axis index of the time dimension, or `None` when no layer has a
        time axis.
    """
    for layer in viewer.layers:
        labels = layer.axis_labels
        if TIME_DIM in labels:
            layer_idx = list(labels).index(TIME_DIM)
            offset = viewer.dims.ndim - layer.ndim
            return offset + layer_idx
    return None


def time_is_sliced(viewer: napari.Viewer) -> bool:
    """Return whether the time axis exists and is currently a slider.

    Parameters
    ----------
    viewer : napari.Viewer
        The active napari viewer instance.

    Returns
    -------
    bool
        Whether a time axis is present and not part of the displayed dimensions.
    """
    time_idx = find_time_dim_index(viewer)
    return time_idx is not None and time_idx not in viewer.dims.displayed


def read_time_units(layer: Layer | None) -> str | None:
    """Read time units from a layer's metadata.

    Parameters
    ----------
    layer : napari.layers.Layer, optional
        Reference layer to read units from. If not provided, returns `None`.

    Returns
    -------
    str | None
        The time units string, or `None` when no layer is given.
    """
    if layer is None:
        return None
    da = layer.metadata.get("xarray")
    if da is not None and TIME_DIM in da.coords:
        return da.coords[TIME_DIM].attrs.get("units", "s")
    # Fallback for non-xarray layers (e.g., video).
    return layer.metadata.get("time_units")


def read_time_value(layer: Layer | None, viewer: napari.Viewer) -> float | None:
    """Read the true time coordinate from a layer at the current viewer position.

    Maps the viewer's world coordinate to the layer's data index via
    `world_to_data` so that layers with different time origins or scales are
    resolved correctly. The data index is then used to look up the true xarray
    coordinate, avoiding napari's linear scale/translate approximation for
    non-uniform spacing.

    Parameters
    ----------
    layer : napari.layers.Layer, optional
        Reference layer whose time coordinate to read. If not provided, returns
        `None`.
    viewer : napari.Viewer
        The active napari viewer instance.

    Returns
    -------
    float | None
        The time coordinate value, or `None` when the layer lacks xarray time
        metadata or the resolved index is out of range.
    """
    if layer is None:
        return None
    da = layer.metadata.get("xarray")
    if da is None or TIME_DIM not in da.coords:
        return None

    world_point = np.array(viewer.dims.point)
    offset = viewer.dims.ndim - layer.ndim
    layer_world_point = world_point[offset:]
    data_point = layer.world_to_data(layer_world_point)

    time_local_idx = list(da.dims).index(TIME_DIM)
    step = int(np.round(data_point[time_local_idx]))

    coords = da.coords[TIME_DIM].values
    if 0 <= step < len(coords):
        return float(coords[step])
    return None


def resolve_reference_layer(viewer: napari.Viewer) -> Layer | None:
    """Return the layer to read the time coordinate from.

    Prefers a single selected time-aware layer so the user can pick which
    recording drives the time readout. When the selection is empty or ambiguous
    (zero or several time-aware layers), falls back to the first time-aware
    layer in the viewer.

    Parameters
    ----------
    viewer : napari.Viewer
        The active napari viewer instance.

    Returns
    -------
    napari.layers.Layer | None
        The resolved reference layer, or `None` when no layer has a time axis.
    """
    selected = [
        layer for layer in viewer.layers.selection if TIME_DIM in layer.axis_labels
    ]
    if len(selected) == 1:
        return selected[0]
    for layer in viewer.layers:
        if TIME_DIM in layer.axis_labels:
            return layer
    return None


def read_current_time(viewer: napari.Viewer) -> tuple[float | None, str | None]:
    """Return the current time value and units from the resolved reference layer.

    Falls back to napari's linear `dims.point` approximation when the reference
    layer has no xarray time coordinate (e.g. a video layer), so a usable time
    is still returned.

    Parameters
    ----------
    viewer : napari.Viewer
        The active napari viewer instance.

    Returns
    -------
    value : float | None
        The current time value in the reference layer's units, or `None` when no
        time axis is available.
    units : str | None
        The time units, or `None` when no reference layer is found.
    """
    layer = resolve_reference_layer(viewer)
    if layer is None:
        return None, None

    value = read_time_value(layer, viewer)
    if value is None:
        time_idx = find_time_dim_index(viewer)
        point = viewer.dims.point
        if time_idx is not None and time_idx < len(point):
            value = float(point[time_idx])
    return value, read_time_units(layer)

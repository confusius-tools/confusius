"""Shared helpers for the ConfUSIus napari plugin."""

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from napari.layers import Layer

CATEGORICAL_COLORS = [
    "#4e79a7",
    "#f28e2b",
    "#e15759",
    "#76b7b2",
    "#59a14f",
    "#edc948",
    "#b07aa1",
    "#ff9da7",
    "#9c755f",
    "#bab0ab",
]
"""Qualitative palette cycled to assign distinct colors to categories.

Shared by the signal store (per imported signal) and the event store (per trial
type) so the plugin uses one consistent set of distinguishable colors.
"""


def extract_voxel_trace(
    layer: "Layer", world_position: npt.NDArray[np.floating], xaxis_index: int
) -> npt.NDArray[np.floating] | None:
    """Extract the nearest-voxel trace along one axis at a world position.

    Parameters
    ----------
    layer : napari.layers.Layer
        Image layer to extract from.
    world_position : numpy.ndarray
        World-coordinate position, in `layer`'s number of dimensions.
    xaxis_index : int
        Data-axis index kept as a full slice (e.g. the `time` axis); all other axes
        are collapsed to their nearest-voxel index.

    Returns
    -------
    numpy.ndarray or None
        1D trace along `xaxis_index`, or `None` if the rounded voxel position falls
        outside the layer's data bounds.
    """
    data = layer.data
    ind = [int(round(v)) for v in layer.world_to_data(world_position)]
    ind[xaxis_index] = slice(None)  # type: ignore[call-overload]

    if not all(
        0 <= i < max_i for i, max_i in zip(ind, data.shape) if isinstance(i, int)
    ):
        return None

    return np.asarray(data[tuple(ind)], dtype=float)


def extract_label_mean_trace(
    layer: "Layer",
    labels_data: npt.NDArray[np.integer],
    label_id: int,
    xaxis_index: int,
) -> npt.NDArray[np.floating] | None:
    """Extract the mean trace over voxels matching one label, along one axis.

    Parameters
    ----------
    layer : napari.layers.Layer
        Image layer to extract from.
    labels_data : numpy.ndarray
        Labels array, either the same shape as `layer.data` or its spatial shape
        (i.e. `layer.data`'s shape with `xaxis_index` removed).
    label_id : int
        Label value to average over.
    xaxis_index : int
        Data-axis index kept as a full slice (e.g. the `time` axis); the mean is
        taken over all other (spatial) axes.

    Returns
    -------
    numpy.ndarray or None
        1D mean trace along `xaxis_index`, or `None` if `labels_data`'s shape does
        not match `layer.data` (directly or spatially) or no voxel matches `label_id`.
    """
    img_data = np.asarray(layer.data)
    img_spatial = tuple(s for i, s in enumerate(img_data.shape) if i != xaxis_index)

    if labels_data.shape == img_data.shape:
        labels_spatial = np.max(labels_data, axis=xaxis_index)
    elif labels_data.shape == img_spatial:
        labels_spatial = labels_data
    else:
        return None

    mask = labels_spatial == label_id
    if not mask.any():
        return None

    img_arr = np.moveaxis(img_data, xaxis_index, -1)
    return np.asarray(img_arr[mask].mean(axis=0), dtype=float)

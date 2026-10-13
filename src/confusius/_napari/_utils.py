"""Shared helpers for the ConfUSIus napari plugin."""

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from scipy.ndimage import affine_transform

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

Shared by the signal store (per stored signal) and the event store (per trial
type) so the plugin uses one consistent set of distinguishable colors.
"""


def get_layer_affine(layer: "Layer", axes: list[int]) -> npt.NDArray[np.float64]:
    """Return the current data-to-world affine restricted to selected layer axes.

    Parameters
    ----------
    layer : napari.layers.Layer
        Layer including its current scale, translation and manual transform.
    axes : list[int]
        Data/world axes to retain, in order.

    Returns
    -------
    numpy.ndarray
        Homogeneous affine with shape `(len(axes) + 1, len(axes) + 1)`.
    """
    origin = np.asarray(layer.data_to_world(np.zeros(layer.ndim)))[axes]
    affine = np.eye(len(axes) + 1)
    affine[:-1, -1] = origin
    for column, axis in enumerate(axes):
        position = np.zeros(layer.ndim)
        position[axis] = 1
        affine[:-1, column] = np.asarray(layer.data_to_world(position))[axes] - origin
    return affine


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
    ind = [round(v) for v in layer.world_to_data(world_position)]
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
    labels_layer: "Layer | None" = None,
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
    labels_layer : napari.layers.Layer, optional
        Source geometry for nearest-neighbor mapping into the image's current
        world-space grid. If not provided, the label array must already match.

    Returns
    -------
    numpy.ndarray or None
        1D mean trace along `xaxis_index`, or `None` if `labels_data`'s shape does
        not match `layer.data` (directly or spatially) or no voxel matches `label_id`.
    """
    img_data = np.asarray(layer.data)
    img_spatial = tuple(s for i, s in enumerate(img_data.shape) if i != xaxis_index)

    if labels_layer is not None:
        image_axes = [axis for axis in range(layer.ndim) if axis != xaxis_index]
        label_axes = (
            image_axes
            if labels_layer.ndim == layer.ndim
            else list(range(labels_layer.ndim))
        )
        if len(label_axes) != len(image_axes):
            return None
        labels_spatial = (
            np.max(labels_data, axis=xaxis_index)
            if labels_layer.ndim == layer.ndim
            else labels_data
        )
        mapping = np.linalg.solve(
            get_layer_affine(labels_layer, label_axes),
            get_layer_affine(layer, image_axes),
        )
        # Labels remain fixed in world space when the recording is transformed.
        if labels_spatial.shape != img_spatial or not np.allclose(
            mapping, np.eye(len(image_axes) + 1)
        ):
            labels_spatial = affine_transform(
                labels_spatial,
                mapping[:-1, :-1],
                mapping[:-1, -1],
                output_shape=img_spatial,
                order=0,
                mode="constant",
                cval=0,
                prefilter=False,
            )
    elif labels_data.shape == img_data.shape:
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

"""Transform-related helpers for the napari registration panel."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import numpy as np

from confusius._dims import SPATIAL_DIMS
from confusius._napari._registration._transforms import (
    AffineTransformPayload,
    affine_transform_from_payload,
)
from confusius._napari._registration._panel_utils import (
    _get_source_dataarray,
    _prepare_between_scan_data,
)

if TYPE_CHECKING:
    import numpy.typing as npt
    from napari.layers import Layer


def _affine_payload_from_layer(layer: "Layer") -> AffineTransformPayload | None:
    """Return the stored affine transform payload for a napari layer.

    Parameters
    ----------
    layer : napari.layers.Layer
        Layer whose metadata should be inspected.

    Returns
    -------
    AffineTransformPayload or None
        Stored payload when present and affine, otherwise `None`.
    """
    payload = layer.metadata.get("confusius_transform")
    if not isinstance(payload, dict) or payload.get("kind") != "affine":
        return None
    affine_transform_from_payload(payload)
    return cast("AffineTransformPayload", payload)


def _spatial_manual_affine_from_layer(
    layer: "Layer", *, spatial_dims: Sequence[str]
) -> "npt.NDArray[np.float64]":
    """Return the spatial sub-affine from a napari layer's manual transform.

    Parameters
    ----------
    layer : napari.layers.Layer
        Layer whose manual napari affine should be extracted.
    spatial_dims : sequence of str
        Spatial dimension names, in the exact order expected by registration.

    Returns
    -------
    (N+1, N+1) numpy.ndarray
        Spatial homogeneous affine in world coordinates.

    Raises
    ------
    ValueError
        If the layer does not contain the requested spatial dimensions.
    ValueError
        If the layer affine has an unexpected shape.
    ValueError
        If the manual affine mixes selected spatial axes with ignored axes.
    """
    data = _get_source_dataarray(layer)
    layer_dims = [str(dim) for dim in data.dims]
    missing_dims = [dim for dim in spatial_dims if dim not in layer_dims]
    if missing_dims:
        raise ValueError(
            "Selected manual napari transform does not contain spatial dims "
            f"{missing_dims}."
        )

    affine = np.asarray(layer.affine.affine_matrix, dtype=float)
    expected_shape = (len(layer_dims) + 1, len(layer_dims) + 1)
    if affine.shape != expected_shape:
        raise ValueError(
            f"Selected manual napari transform has shape {affine.shape}, "
            f"but layer '{layer.name}' expects {expected_shape}."
        )

    spatial_indices = [layer_dims.index(dim) for dim in spatial_dims]
    ignored_indices = [i for i in range(len(layer_dims)) if i not in spatial_indices]
    linear = affine[:-1, :-1]

    if ignored_indices:
        spatial_to_ignored = linear[np.ix_(spatial_indices, ignored_indices)]
        ignored_to_spatial = linear[np.ix_(ignored_indices, spatial_indices)]
        if not np.allclose(spatial_to_ignored, 0.0) or not np.allclose(
            ignored_to_spatial, 0.0
        ):
            raise ValueError(
                "Selected manual napari transform mixes spatial axes with ignored "
                "non-spatial axes, so it cannot be used as a registration "
                "initialization."
            )

    spatial_affine = np.eye(len(spatial_dims) + 1, dtype=float)
    spatial_affine[:-1, :-1] = linear[np.ix_(spatial_indices, spatial_indices)]
    spatial_affine[:-1, -1] = affine[np.ix_(spatial_indices, [-1])].ravel()
    return spatial_affine


def _make_manual_transform_payload(layer: "Layer") -> AffineTransformPayload:
    """Build an affine payload from a layer's manual napari transform.

    Parameters
    ----------
    layer : napari.layers.Layer
        Layer whose current manual napari transform should be serialized.

    Returns
    -------
    AffineTransformPayload
        JSON-serializable affine payload representing the visible manual layer
        transform on the layer's own spatial output grid.
    """
    data = _get_source_dataarray(layer)
    spatial_data = _prepare_between_scan_data(data)
    spatial_dims = [str(dim) for dim in spatial_data.dims if dim in SPATIAL_DIMS]
    manual_affine = _spatial_manual_affine_from_layer(layer, spatial_dims=spatial_dims)
    pull_affine = np.linalg.inv(manual_affine)
    return {
        "kind": "affine",
        "name": f"{layer.name} (manual)",
        "affine": pull_affine.tolist(),
        "source_layer_name": layer.name,
        "target_layer_name": layer.name,
        "operation": "manual_napari_transform",
        "transform_model": "affine",
        "metric": "manual",
        "output_grid": {
            "dims": [str(dim) for dim in spatial_data.dims],
            "shape": [int(spatial_data.sizes[dim]) for dim in spatial_data.dims],
            "spacing": [
                float(spatial_data.fusi.spacing[dim]) for dim in spatial_data.dims
            ],
            "origin": [
                float(spatial_data.fusi.origin[dim]) for dim in spatial_data.dims
            ],
            "units": [
                cast("str | None", spatial_data.coords[dim].attrs.get("units"))
                if dim in spatial_data.coords
                else None
                for dim in spatial_data.dims
            ],
        },
        "diagnostics": {
            "metric": "manual",
            "final_metric_value": 0.0,
            "n_iterations": 0,
            "stop_condition": "Saved from manual napari layer transform.",
            "status": "completed",
        },
    }

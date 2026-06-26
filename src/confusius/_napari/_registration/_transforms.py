"""Affine transform helpers for the napari registration panel."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Literal, SupportsFloat, SupportsIndex, TypedDict, cast

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Mapping

    import xarray as xr

    from confusius.registration import RegistrationDiagnostics


class TransformDiagnosticsPayload(TypedDict):
    """JSON-serializable registration diagnostics summary."""

    metric: str
    final_metric_value: float
    n_iterations: int
    stop_condition: str
    status: str


class OutputGridPayload(TypedDict):
    """JSON-serializable resampling grid description."""

    dims: list[str]
    shape: list[int]
    spacing: list[float]
    origin: list[float]
    units: list[str | None]


class AffineTransformPayload(TypedDict):
    """JSON-serializable affine transform payload used by the napari plugin."""

    kind: Literal["affine"]
    name: str
    affine: list[list[float]]
    source_layer_name: str
    target_layer_name: str
    operation: str
    transform_model: str
    metric: str
    output_grid: OutputGridPayload
    diagnostics: TransformDiagnosticsPayload


def make_output_grid_payload(reference: "xr.DataArray") -> OutputGridPayload:
    """Return the resampling grid defined by a reference DataArray.

    Parameters
    ----------
    reference : xarray.DataArray
        Spatial DataArray defining the output grid.

    Returns
    -------
    OutputGridPayload
        JSON-serializable output-grid description.
    """
    dims = [str(dim) for dim in reference.dims]
    return {
        "dims": dims,
        "shape": [int(reference.sizes[dim]) for dim in dims],
        "spacing": [float(reference.fusi.spacing[dim]) for dim in dims],
        "origin": [float(reference.fusi.origin[dim]) for dim in dims],
        "units": [
            cast("str | None", reference.coords[dim].attrs.get("units"))
            if dim in reference.coords
            else None
            for dim in dims
        ],
    }


def make_affine_transform_payload(
    affine: npt.NDArray[np.floating],
    *,
    reference: "xr.DataArray",
    source_layer_name: str,
    target_layer_name: str,
    operation: str,
    transform_model: str,
    metric: str,
    diagnostics: "RegistrationDiagnostics",
    name: str | None = None,
) -> AffineTransformPayload:
    """Build a JSON-serializable payload for a registered affine transform.

    Parameters
    ----------
    affine : (N+1, N+1) numpy.ndarray
        Affine transform in homogeneous coordinates.
    reference : xarray.DataArray
        Fixed/reference DataArray defining the output resampling grid.
    source_layer_name : str
        Name of the moving/source layer used when estimating the transform.
    target_layer_name : str
        Name of the fixed/target layer used when estimating the transform.
    operation : str
        Registration operation that produced the transform.
    transform_model : str
        Transform model used during registration.
    metric : str
        Similarity metric used during registration.
    diagnostics : confusius.registration.RegistrationDiagnostics
        Per-call registration diagnostics.
    name : str, optional
        Human-friendly transform name. If not provided, a default name is generated.

    Returns
    -------
    AffineTransformPayload
        JSON-serializable transform payload.
    """
    affine = np.asarray(affine, dtype=float)
    payload_name = (
        name or f"{source_layer_name} → {target_layer_name} ({transform_model})"
    )
    return {
        "kind": "affine",
        "name": payload_name,
        "affine": affine.tolist(),
        "source_layer_name": source_layer_name,
        "target_layer_name": target_layer_name,
        "operation": operation,
        "transform_model": transform_model,
        "metric": metric,
        "output_grid": make_output_grid_payload(reference),
        "diagnostics": {
            "metric": diagnostics.metric,
            "final_metric_value": float(diagnostics.final_metric_value),
            "n_iterations": int(diagnostics.n_iterations),
            "stop_condition": diagnostics.stop_condition,
            "status": diagnostics.status,
        },
    }


def affine_transform_from_payload(
    payload: "Mapping[str, object]",
) -> npt.NDArray[np.float64]:
    """Return the affine matrix stored in a payload.

    Parameters
    ----------
    payload : mapping
        Transform payload loaded from metadata or JSON.

    Returns
    -------
    (N+1, N+1) numpy.ndarray
        Affine matrix.

    Raises
    ------
    ValueError
        If the payload is not an affine transform payload.
    """
    if payload.get("kind") != "affine":
        raise ValueError("Transform payload is not an affine transform.")

    affine = np.asarray(payload.get("affine"), dtype=float)
    if affine.ndim != 2 or affine.shape[0] != affine.shape[1] or affine.shape[0] < 3:
        raise ValueError(
            "Affine payload must contain a square homogeneous matrix of shape "
            "(N+1, N+1)."
        )
    return affine


def output_grid_from_payload(payload: "Mapping[str, object]") -> OutputGridPayload:
    """Return the output grid stored in a transform payload.

    Parameters
    ----------
    payload : mapping
        Transform payload loaded from metadata or JSON.

    Returns
    -------
    OutputGridPayload
        Output-grid description stored in the payload.

    Raises
    ------
    ValueError
        If the payload does not carry a valid output grid.
    """
    grid = payload.get("output_grid")
    if not isinstance(grid, dict):
        raise ValueError("Transform payload does not contain an output grid.")

    grid_dict = cast("dict[str, object]", grid)
    dims = grid_dict.get("dims")
    shape = grid_dict.get("shape")
    spacing = grid_dict.get("spacing")
    origin = grid_dict.get("origin")
    units = grid_dict.get("units")
    if not all(isinstance(v, list) for v in (dims, shape, spacing, origin, units)):
        raise ValueError("Transform payload output grid is malformed.")

    dims_list = cast("list[object]", dims)
    shape_list = cast("list[SupportsIndex]", shape)
    spacing_list = cast("list[SupportsFloat]", spacing)
    origin_list = cast("list[SupportsFloat]", origin)
    units_list = cast("list[object]", units)

    return {
        "dims": [str(v) for v in dims_list],
        "shape": [int(v) for v in shape_list],
        "spacing": [float(v) for v in spacing_list],
        "origin": [float(v) for v in origin_list],
        "units": [None if v is None else str(v) for v in units_list],
    }


def save_affine_transform_payload(
    path: str | Path, payload: AffineTransformPayload
) -> None:
    """Save an affine transform payload as JSON.

    Parameters
    ----------
    path : str or pathlib.Path
        Output JSON path.
    payload : AffineTransformPayload
        Transform payload to save.
    """
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def load_affine_transform_payload(path: str | Path) -> AffineTransformPayload:
    """Load an affine transform payload from JSON.

    Parameters
    ----------
    path : str or pathlib.Path
        Input JSON path.

    Returns
    -------
    AffineTransformPayload
        Loaded payload.

    Raises
    ------
    ValueError
        If the file does not contain an affine transform payload.
    """
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("Transform file must contain a JSON object.")
    affine_transform_from_payload(payload)
    output_grid_from_payload(payload)
    return cast("AffineTransformPayload", payload)

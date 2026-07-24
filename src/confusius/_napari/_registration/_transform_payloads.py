"""Transform payload types and serialization helpers for napari registration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Literal,
    NotRequired,
    SupportsFloat,
    SupportsIndex,
    TypedDict,
    cast,
)

import numpy as np
import numpy.typing as npt
import xarray as xr

from confusius.io import load as load_dataarray
from confusius.io import save as save_dataarray
from confusius.registration.bspline import validate_bspline_dataarray

if TYPE_CHECKING:
    from collections.abc import Mapping

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


class BSplineDataArrayPayload(TypedDict):
    """JSON-serializable B-spline control-point DataArray."""

    dims: list[str]
    data: list[object]
    coords: dict[str, list[object]]
    attrs: dict[str, object]


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
    input_grid: NotRequired[OutputGridPayload]
    diagnostics: TransformDiagnosticsPayload


class BSplineTransformPayload(TypedDict):
    """B-spline transform payload used by the napari plugin."""

    kind: Literal["bspline"]
    name: str
    bspline: BSplineDataArrayPayload
    source_layer_name: str
    target_layer_name: str
    operation: str
    transform_model: str
    metric: str
    output_grid: OutputGridPayload
    input_grid: NotRequired[OutputGridPayload]
    diagnostics: TransformDiagnosticsPayload


TransformPayload = AffineTransformPayload | BSplineTransformPayload
"""Union of affine and B-spline transform payloads."""


def _make_json_serializable(value: object) -> object:
    """Return a JSON-serializable copy of a nested value."""
    if isinstance(value, np.ndarray):
        return (
            _make_json_serializable(np.asarray(value).astype(object).reshape(-1)[0])
            if value.ndim == 0
            else [_make_json_serializable(v) for v in list(value)]
        )
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _make_json_serializable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_make_json_serializable(v) for v in value]
    return value


def make_output_grid_payload(reference: xr.DataArray) -> OutputGridPayload:
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


def _make_diagnostics_payload(
    diagnostics: RegistrationDiagnostics,
) -> TransformDiagnosticsPayload:
    """Return a JSON-serializable diagnostics summary.

    Parameters
    ----------
    diagnostics : confusius.registration.RegistrationDiagnostics
        Per-call registration diagnostics to serialize.

    Returns
    -------
    TransformDiagnosticsPayload
        JSON-serializable diagnostics summary.
    """
    return {
        "metric": diagnostics.metric,
        "final_metric_value": float(diagnostics.final_metric_value),
        "n_iterations": int(diagnostics.n_iterations),
        "stop_condition": diagnostics.stop_condition,
        "status": diagnostics.status,
    }


def make_affine_transform_payload(
    affine: npt.NDArray[np.floating],
    *,
    reference: xr.DataArray,
    source: xr.DataArray | None = None,
    source_layer_name: str,
    target_layer_name: str,
    operation: str,
    transform_model: str,
    metric: str,
    diagnostics: RegistrationDiagnostics,
    name: str | None = None,
) -> AffineTransformPayload:
    """Build a JSON-serializable payload for a registered affine transform.

    Parameters
    ----------
    affine : (N+1, N+1) numpy.ndarray
        Affine transform in homogeneous coordinates.
    reference : xarray.DataArray
        Fixed/reference DataArray defining the output resampling grid.
    source : xarray.DataArray, optional
        Original moving/source DataArray defining the inverse-apply resampling grid. If
        not provided, `input_grid` is omitted from the payload.
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
        JSON-serializable affine transform payload.
    """
    affine = np.asarray(affine, dtype=float)
    payload_name = (
        name or f"{source_layer_name} → {target_layer_name} ({transform_model})"
    )
    payload: AffineTransformPayload = {
        "kind": "affine",
        "name": payload_name,
        "affine": affine.tolist(),
        "source_layer_name": source_layer_name,
        "target_layer_name": target_layer_name,
        "operation": operation,
        "transform_model": transform_model,
        "metric": metric,
        "output_grid": make_output_grid_payload(reference),
        "diagnostics": _make_diagnostics_payload(diagnostics),
    }
    if source is not None:
        payload["input_grid"] = make_output_grid_payload(source)
    return payload


def _serialize_bspline_dataarray(transform: xr.DataArray) -> BSplineDataArrayPayload:
    """Return a JSON-serializable B-spline DataArray payload.

    Parameters
    ----------
    transform : xarray.DataArray
        B-spline control-point grid to serialize.

    Returns
    -------
    BSplineDataArrayPayload
        JSON-serializable B-spline DataArray payload.
    """
    validate_bspline_dataarray(transform)
    return {
        "dims": [str(dim) for dim in transform.dims],
        "data": np.asarray(transform, dtype=float).tolist(),
        "coords": {
            str(dim): (
                np.asarray(transform.coords[dim], dtype=np.str_).tolist()
                if str(dim) == "component"
                else np.asarray(transform.coords[dim], dtype=float).tolist()
            )
            for dim in transform.dims
            if dim in transform.coords
        },
        "attrs": json.loads(json.dumps(_make_json_serializable(transform.attrs))),
    }


def _deserialize_bspline_dataarray(payload: BSplineDataArrayPayload) -> xr.DataArray:
    """Reconstruct a B-spline DataArray from its JSON payload.

    Parameters
    ----------
    payload : BSplineDataArrayPayload
        JSON payload describing a B-spline control-point grid.

    Returns
    -------
    xarray.DataArray
        Reconstructed B-spline control-point grid.
    """
    dims = [str(dim) for dim in payload["dims"]]
    coords = {
        str(dim): xr.DataArray(
            (
                np.asarray(values, dtype=np.str_)
                if str(dim) == "component"
                else np.asarray(values, dtype=float)
            ),
            dims=[str(dim)],
        )
        for dim, values in payload["coords"].items()
    }
    transform = xr.DataArray(
        np.asarray(payload["data"], dtype=float),
        dims=dims,
        coords=coords,
        attrs=dict(payload["attrs"]),
    )
    validate_bspline_dataarray(transform)
    return transform


def _normalize_loaded_bspline_transform(transform: xr.DataArray) -> xr.DataArray:
    """Drop NIfTI-added metadata and synthetic singleton spatial dims."""
    normalized = transform.copy(deep=False)
    normalized.attrs = normalized.attrs.copy()
    for key in (
        "confusius_transform_kind",
        "confusius_transform_metadata_json",
        "name",
        "source_layer_name",
        "target_layer_name",
        "operation",
        "transform_model",
        "metric",
        "output_grid",
        "input_grid",
        "diagnostics",
        "qform_code",
    ):
        normalized.attrs.pop(key, None)

    affines = normalized.attrs.get("affines")
    if isinstance(affines, dict):
        normalized_affines = dict(affines)
        normalized_affines.pop("physical_to_qform", None)
        if normalized_affines:
            normalized.attrs["affines"] = normalized_affines
        else:
            normalized.attrs.pop("affines", None)

    component_values = [str(v) for v in normalized.coords["component"].values]
    for dim in list(normalized.dims[1:]):
        if dim not in component_values and normalized.sizes[dim] == 1:
            normalized = normalized.squeeze(dim, drop=True)

    return normalized


def make_bspline_transform_payload(
    transform: xr.DataArray,
    *,
    reference: xr.DataArray,
    source: xr.DataArray | None = None,
    source_layer_name: str,
    target_layer_name: str,
    operation: str,
    transform_model: str,
    metric: str,
    diagnostics: RegistrationDiagnostics,
    name: str | None = None,
) -> BSplineTransformPayload:
    """Build a JSON-serializable payload for a registered B-spline transform.

    Parameters
    ----------
    transform : xarray.DataArray
        B-spline control-point grid.
    reference : xarray.DataArray
        Fixed/reference DataArray defining the output resampling grid.
    source : xarray.DataArray, optional
        Original moving/source DataArray defining the inverse-apply resampling grid. If
        not provided, `input_grid` is omitted from the payload.
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
    BSplineTransformPayload
        JSON-serializable B-spline transform payload.
    """
    payload_name = (
        name or f"{source_layer_name} → {target_layer_name} ({transform_model})"
    )
    payload: BSplineTransformPayload = {
        "kind": "bspline",
        "name": payload_name,
        "bspline": _serialize_bspline_dataarray(transform),
        "source_layer_name": source_layer_name,
        "target_layer_name": target_layer_name,
        "operation": operation,
        "transform_model": transform_model,
        "metric": metric,
        "output_grid": make_output_grid_payload(reference),
        "diagnostics": _make_diagnostics_payload(diagnostics),
    }
    if source is not None:
        payload["input_grid"] = make_output_grid_payload(source)
    return payload


def get_affine_transform_from_payload(
    payload: Mapping[str, object],
) -> npt.NDArray[np.float64]:
    """Return the affine matrix stored in a payload.

    Parameters
    ----------
    payload : mapping
        Transform payload loaded from metadata or disk.

    Returns
    -------
    (N+1, N+1) numpy.ndarray
        Affine matrix.
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


def get_bspline_transform_from_payload(payload: Mapping[str, object]) -> xr.DataArray:
    """Return the B-spline transform stored in a payload.

    Parameters
    ----------
    payload : mapping
        Transform payload loaded from metadata or disk.

    Returns
    -------
    xarray.DataArray
        B-spline control-point grid.
    """
    if payload.get("kind") != "bspline":
        raise ValueError("Transform payload is not a B-spline transform.")

    bspline = payload.get("bspline")
    if not isinstance(bspline, dict):
        raise TypeError("B-spline payload must contain a serialized DataArray.")
    return _deserialize_bspline_dataarray(cast("BSplineDataArrayPayload", bspline))


def _coerce_grid_payload(
    grid: object, *, field_name: str, missing_message: str
) -> OutputGridPayload:
    """Return a validated grid payload from a raw mapping field."""
    if not isinstance(grid, dict):
        raise TypeError(missing_message)

    grid_dict = cast("dict[str, object]", grid)
    dims = grid_dict.get("dims")
    shape = grid_dict.get("shape")
    spacing = grid_dict.get("spacing")
    origin = grid_dict.get("origin")
    units = grid_dict.get("units")
    if not all(isinstance(v, list) for v in (dims, shape, spacing, origin, units)):
        raise ValueError(f"Transform payload {field_name} is malformed.")

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


def get_output_grid_from_payload(payload: Mapping[str, object]) -> OutputGridPayload:
    """Return the output grid stored in a transform payload.

    Parameters
    ----------
    payload : mapping
        Transform payload loaded from metadata or disk.

    Returns
    -------
    OutputGridPayload
        Output-grid description stored in the payload.
    """
    return _coerce_grid_payload(
        payload.get("output_grid"),
        field_name="output grid",
        missing_message="Transform payload does not contain an output grid.",
    )


def get_input_grid_from_payload(
    payload: Mapping[str, object],
) -> OutputGridPayload | None:
    """Return the input grid stored in a transform payload, if present.

    Parameters
    ----------
    payload : mapping
        Transform payload loaded from metadata or disk.

    Returns
    -------
    OutputGridPayload or None
        Input-grid description stored in the payload, or `None` when the payload does
        not carry one.
    """
    if "input_grid" not in payload:
        return None
    return _coerce_grid_payload(
        payload.get("input_grid"),
        field_name="input grid",
        missing_message="Transform payload does not contain an input grid.",
    )


def _save_bspline_transform_payload(
    path: str | Path, payload: BSplineTransformPayload
) -> None:
    """Save a B-spline transform payload as NIfTI plus sidecar.

    Parameters
    ----------
    path : str or pathlib.Path
        Output NIfTI path.
    payload : BSplineTransformPayload
        Transform payload to save.

    Raises
    ------
    ValueError
        If `path` does not have a `.nii` or `.nii.gz` extension.
    """
    path = Path(path)
    if not path.name.endswith(".nii") and not path.name.endswith(".nii.gz"):
        raise ValueError(
            "B-spline transform files must have .nii or .nii.gz extension."
        )

    transform = get_bspline_transform_from_payload(payload).copy(deep=False)
    transform.attrs = transform.attrs.copy()
    transform.attrs["confusius_transform_kind"] = "bspline"
    for key in (
        "name",
        "source_layer_name",
        "target_layer_name",
        "operation",
        "transform_model",
        "metric",
        "output_grid",
        "diagnostics",
    ):
        transform.attrs[key] = _make_json_serializable(payload[key])
    if "input_grid" in payload:
        transform.attrs["input_grid"] = _make_json_serializable(payload["input_grid"])
    save_dataarray(transform, path)


def _load_bspline_transform_payload(path: str | Path) -> BSplineTransformPayload:
    """Load a B-spline transform payload from NIfTI or legacy Zarr.

    Parameters
    ----------
    path : str or pathlib.Path
        Input transform path.

    Returns
    -------
    BSplineTransformPayload
        Loaded B-spline transform payload.
    """
    path = Path(path)
    if path.suffix == ".zarr":
        ds = xr.open_zarr(path)
        try:
            if ds.attrs.get("confusius_transform_kind") != "bspline":
                raise ValueError(
                    "Zarr transform store does not contain a ConfUSIus B-spline transform."
                )
            payload_metadata = json.loads(
                cast("str", ds.attrs["confusius_transform_payload_json"])
            )
            if not isinstance(payload_metadata, dict):
                raise TypeError("Stored transform payload metadata is malformed.")
            transform = ds["bspline_transform"].load()
        finally:
            ds.close()

        if "bspline" in payload_metadata:
            payload_with_kind = {"kind": "bspline", **payload_metadata}
            return cast("BSplineTransformPayload", payload_with_kind)
    else:
        transform = load_dataarray(path)
        if transform.attrs.get("confusius_transform_kind") != "bspline":
            raise ValueError(
                "NIfTI transform file does not contain a ConfUSIus B-spline transform."
            )
        if "confusius_transform_metadata_json" in transform.attrs:
            payload_metadata = json.loads(
                cast("str", transform.attrs["confusius_transform_metadata_json"])
            )
            if not isinstance(payload_metadata, dict):
                raise TypeError("Stored transform payload metadata is malformed.")
        else:
            payload_metadata = {
                key: transform.attrs[key]
                for key in (
                    "name",
                    "source_layer_name",
                    "target_layer_name",
                    "operation",
                    "transform_model",
                    "metric",
                    "output_grid",
                    "diagnostics",
                )
                if key in transform.attrs
            }
            if "input_grid" in transform.attrs:
                payload_metadata["input_grid"] = transform.attrs["input_grid"]

    transform = _normalize_loaded_bspline_transform(transform)
    validate_bspline_dataarray(transform)
    payload: BSplineTransformPayload = {
        "kind": "bspline",
        "bspline": _serialize_bspline_dataarray(transform),
        "name": str(payload_metadata.get("name", path.stem)),
        "source_layer_name": str(payload_metadata.get("source_layer_name", "loaded")),
        "target_layer_name": str(payload_metadata.get("target_layer_name", "loaded")),
        "operation": str(payload_metadata.get("operation", "loaded_transform")),
        "transform_model": str(payload_metadata.get("transform_model", "bspline")),
        "metric": str(payload_metadata.get("metric", "unknown")),
        "output_grid": get_output_grid_from_payload(payload_metadata),
        "diagnostics": cast(
            "TransformDiagnosticsPayload",
            payload_metadata.get(
                "diagnostics",
                {
                    "metric": str(payload_metadata.get("metric", "unknown")),
                    "final_metric_value": 0.0,
                    "n_iterations": 0,
                    "stop_condition": "Loaded from disk.",
                    "status": "completed",
                },
            ),
        ),
    }
    input_grid = get_input_grid_from_payload(payload_metadata)
    if input_grid is not None:
        payload["input_grid"] = input_grid
    return payload


def save_transform_payload(path: str | Path, payload: TransformPayload) -> None:
    """Save a transform payload to disk as JSON for affine payloads or NIfTI for B-spline payloads.

    Parameters
    ----------
    path : str or pathlib.Path
        Output path.
    payload : TransformPayload
        Transform payload to save.

    Notes
    -----
    Affine payloads are saved as JSON. B-spline payloads are saved as NIfTI.
    """
    if payload["kind"] == "affine":
        Path(path).write_text(json.dumps(payload, indent=2) + "\n")
        return
    _save_bspline_transform_payload(path, payload)


def load_transform_payload(path: str | Path) -> TransformPayload:
    """Load an affine or B-spline transform payload from disk.

    Parameters
    ----------
    path : str or pathlib.Path
        Input path.

    Returns
    -------
    TransformPayload
        Loaded transform payload.
    """
    path = Path(path)
    if (
        path.suffix == ".zarr"
        or path.name.endswith(".nii")
        or path.name.endswith(".nii.gz")
    ):
        return _load_bspline_transform_payload(path)

    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise TypeError("Transform file must contain a JSON object.")

    kind = payload.get("kind")
    if kind != "affine":
        raise ValueError(
            "JSON transform files currently support affine payloads only. "
            "Use .nii or .nii.gz for B-spline transforms."
        )
    get_affine_transform_from_payload(payload)
    get_output_grid_from_payload(payload)
    return cast("TransformPayload", payload)

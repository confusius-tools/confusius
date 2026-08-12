"""File writers for ConfUSIus data formats (npe2).

These are called by napari when layers are saved via File → Save, File → Save As, or the
CLI.

Each public function is a `write` command: it receives the output path, the layer data
array, and the layer metadata dict, and writes the data to disk.

When a layer was loaded via the ConfUSIus reader, the original DataArray is stored in
`meta["metadata"]["xarray"]` and used directly. Otherwise—for example, labels layers
drawn by the user—a DataArray is reconstructed from the napari layer properties
(`scale`, `translate`, `axis_labels`, `units`).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import numpy as np

from confusius._dims import SPATIAL_DIMS, TIME_DIM

_NAPARI_GENERIC_AXIS = re.compile(r"^axis -?\d+$")
"""Matches napari's default generic axis labels (e.g. 'axis -4', 'axis -3', ...)."""

_NAPARI_NOPHYSICAL_UNITS = frozenset({"pixel"})
"""Napari units that indicate the absence of physical units."""

if TYPE_CHECKING:
    import xarray as xr


def _compute_dataarray_from_layer(data: Any, meta: dict[str, Any]) -> xr.DataArray:
    """Reconstruct a ConfUSIus DataArray from raw napari layer properties.

    Builds uniform coordinates for each dimension using `scale` (spacing) and
    `translate` (origin) from the napari layer state. `axis_labels` and `units` are used
    when present; sensible defaults are applied otherwise.

    Parameters
    ----------
    data : array-like
        Layer data array (numpy or dask).
    meta : dict
        Napari layer metadata dict as passed to a writer. Expected keys:
        `axis_labels`, `scale`, `translate`, `units`.

    Returns
    -------
    xarray.DataArray
        DataArray with physical coordinates derived from the layer state.
    """
    import xarray as xr

    ndim = np.asarray(data).ndim

    _default_dims: dict[int, tuple[str, ...]] = {
        1: SPATIAL_DIMS[-1:],  # ("x",)
        2: SPATIAL_DIMS[-2:],  # ("y", "x")
        3: SPATIAL_DIMS,  # ("z", "y", "x")
        4: (TIME_DIM, *SPATIAL_DIMS),  # ("time", "z", "y", "x")
    }
    raw_labels = meta.get("axis_labels")
    if raw_labels and not all(
        _NAPARI_GENERIC_AXIS.match(str(lbl)) for lbl in raw_labels
    ):
        axis_labels: tuple[str, ...] = tuple(str(lbl) for lbl in raw_labels)
    else:
        axis_labels = _default_dims.get(ndim, tuple(f"dim{i}" for i in range(ndim)))

    scale: list[float] = list(meta.get("scale") or ([1.0] * ndim))
    translate: list[float] = list(meta.get("translate") or ([0.0] * ndim))

    # Napari may pass pint Unit objects rather than strings. Convert to str and treat
    # non-physical units ('pixel') as absent.
    raw_units = meta.get("units") or ([None] * ndim)
    units: list[str | None] = [
        None if u is None or str(u) in _NAPARI_NOPHYSICAL_UNITS else str(u)
        for u in raw_units
    ]

    data_array = np.asarray(data)
    voxel_to_world_name = {"k": "z", "j": "y", "i": "x"}
    physical_to_voxel_name = {v: k for k, v in voxel_to_world_name.items()}

    result_dims: list[str] = []
    coords: dict[str, xr.DataArray] = {}
    voxel_dims: list[str] = []
    physical_names: list[str] = []
    affine_scales: list[float] = []
    affine_translates: list[float] = []
    physical_attrs: dict[str, dict[str, Any]] = {}

    for axis, dim in enumerate(axis_labels):
        n = data_array.shape[axis]
        if dim in physical_to_voxel_name:
            result_dim = physical_to_voxel_name[dim]
            physical_name = dim
        elif dim in voxel_to_world_name:
            result_dim = dim
            physical_name = voxel_to_world_name[dim]
        else:
            result_dim = dim
            physical_name = ""

        result_dims.append(result_dim)
        attrs: dict[str, Any] = {"voxdim": abs(float(scale[axis]))}
        if units[axis] is not None:
            attrs["units"] = units[axis]

        if physical_name:
            coords[result_dim] = xr.DataArray(np.arange(n), dims=[result_dim])
            voxel_dims.append(result_dim)
            physical_names.append(physical_name)
            affine_scales.append(float(scale[axis]))
            affine_translates.append(float(translate[axis]))
            physical_attrs[physical_name] = attrs
        else:
            coord_values = translate[axis] + np.arange(n) * scale[axis]
            coords[result_dim] = xr.DataArray(
                coord_values, dims=[result_dim], attrs=attrs
            )

    result = xr.DataArray(data_array, dims=result_dims, coords=coords)
    if not voxel_dims:
        return result

    from confusius._utils.geometry import add_world_coords_from_voxel_affine

    affine = np.eye(len(voxel_dims) + 1, dtype=float)
    affine[:-1, :-1] = np.diag(affine_scales)
    affine[:-1, -1] = affine_translates
    return add_world_coords_from_voxel_affine(
        result,
        affine,
        voxel_dims=tuple(voxel_dims),
        world_coord_names=tuple(physical_names),
        world_coord_attrs=physical_attrs,
    )


def _get_or_compute_dataarray_from_layer(
    data: Any, meta: dict[str, Any]
) -> xr.DataArray:
    """Return a ConfUSIus DataArray for the given napari layer.

    Uses the original DataArray stored in `meta["metadata"]["xarray"]` when available
    (layers loaded via the ConfUSIus reader). Falls back to
    [`_compute_dataarray_from_layer`][confusius._writers._compute_dataarray_from_layer]
    otherwise (e.g. user-drawn labels layers).

    Parameters
    ----------
    data : array-like
        Layer data array as passed by napari to the writer.
    meta : dict
        Layer metadata dict as passed by napari to the writer.

    Returns
    -------
    xarray.DataArray
        DataArray ready for saving.
    """
    da: xr.DataArray | None = meta.get("metadata", {}).get("source_xarray")
    if da is None:
        da = meta.get("metadata", {}).get("xarray")
    if da is not None:
        return da
    return _compute_dataarray_from_layer(data, meta)


def write_nifti(path: str, data: Any, meta: dict[str, Any]) -> list[str]:
    """Write an image or labels layer to a NIfTI file (`.nii` / `.nii.gz`).

    If the layer was loaded via the ConfUSIus reader, the original DataArray (stored in
    `meta["metadata"]["xarray"]`) is used directly, preserving all coordinates and
    attributes. Otherwise the DataArray is reconstructed from the napari layer state
    (`scale`, `translate`, `axis_labels`, `units`), which is the typical case for
    user-drawn labels layers.

    A BIDS-style JSON sidecar is written alongside the NIfTI file by
    [`confusius.io.save_nifti`][confusius.io.save_nifti].

    Parameters
    ----------
    path : str
        Output file path (e.g. `/data/volume.nii.gz`).
    data : Any
        Layer data array as passed by napari.
    meta : dict
        Layer metadata dict as passed by napari.

    Returns
    -------
    list[str]
        List of paths written (the NIfTI file path).
    """
    from confusius.io import save

    da = _get_or_compute_dataarray_from_layer(data, meta)
    save(da, path)
    return [path]


def write_zarr(path: str, data: Any, meta: dict[str, Any]) -> list[str]:
    """Write an image or labels layer to a Zarr store (`.zarr`).

    If the layer was loaded via the ConfUSIus reader, the original DataArray (stored in
    `meta["metadata"]["xarray"]`) is used directly, preserving all coordinates and
    attributes. Otherwise the DataArray is reconstructed from the napari layer state
    (`scale`, `translate`, `axis_labels`, `units`), which is the typical case for
    user-drawn labels layers.

    Parameters
    ----------
    path : str
        Output path for the Zarr store directory (e.g. `/data/labels.zarr`).
    data : Any
        Layer data array as passed by napari.
    meta : dict
        Layer metadata dict as passed by napari.

    Returns
    -------
    list[str]
        List of paths written (the Zarr store path).
    """
    from confusius.io import save

    da = _get_or_compute_dataarray_from_layer(data, meta)
    save(da, path)
    return [path]

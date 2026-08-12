"""Generic file loading and saving dispatcher."""

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

import confusius.io.nifti as _nifti
import confusius.io.scan as _scan
from confusius._utils.atlas import restore_atlas_cmap_and_norm
from confusius._utils.geometry import (
    add_world_coords_from_voxel_affine,
    has_voxel_world_geometry,
)
from confusius.io._utils import (
    ZARR_V3_CONSOLIDATED_METADATA_WARNING,
    make_attrs_zarr_safe,
    restore_affines_in_attrs,
)
from confusius.io.utils import check_path
from confusius.validation import ensure_fusi


def _restore_voxel_world_index_from_coords(data: xr.DataArray) -> xr.DataArray:
    """Rebuild voxel-to-world geometry from serialized affine-like world coordinates."""
    if has_voxel_world_geometry(data):
        return data
    voxel_dims = tuple(dim for dim in ("k", "j", "i") if dim in data.dims)
    world_names = ("z", "y", "x")[-len(voxel_dims) :]
    if len(voxel_dims) < 2 or any(name not in data.coords for name in world_names):
        return data

    voxel_mesh = np.meshgrid(
        *(np.asarray(data.coords[dim].values, dtype=np.float64) for dim in voxel_dims),
        indexing="ij",
    )
    design = np.stack(
        [mesh.ravel() for mesh in voxel_mesh] + [np.ones(voxel_mesh[0].size)], axis=1
    )
    affine = np.eye(len(voxel_dims) + 1, dtype=np.float64)
    try:
        broadcast_world = xr.broadcast(*(data.coords[name] for name in world_names))
    except ValueError:
        return data
    for row, coord in enumerate(broadcast_world):
        if set(coord.dims) != set(voxel_dims):
            return data
        coeffs, *_ = np.linalg.lstsq(
            design,
            np.asarray(coord.transpose(*voxel_dims).values, dtype=np.float64).ravel(),
            rcond=None,
        )
        affine[row, :] = coeffs
    affine[np.isclose(affine, 0.0, atol=1e-12)] = 0.0

    attrs = {name: data.coords[name].attrs for name in world_names}
    return add_world_coords_from_voxel_affine(
        data,
        affine,
        voxel_dims=voxel_dims,
        world_coord_names=world_names,
        world_coord_attrs=attrs,
    )


def load(path: str | Path, variable: str | None = None, **kwargs: Any) -> xr.DataArray:
    """Load a fUSI DataArray from file, dispatching by extension.

    Supported formats:

    - **NIfTI** (`.nii`, `.nii.gz`): loaded via [`load_nifti`][confusius.io.load_nifti].
    - **SCAN** (`.scan`): loaded via [`load_scan`][confusius.io.load_scan].
    - **Zarr** (`.zarr`): opened via [`xarray.open_zarr`][xarray.open_zarr] and a single
      variable is extracted. For loading the full dataset, use
      [`xarray.open_zarr`][xarray.open_zarr] directly.

    If `attrs["rgb_lookup"]` is present but `attrs["cmap"]`/`attrs["norm"]` are missing
    (as happens after a save/load round-trip, since matplotlib colormap/norm objects are
    not JSON-serializable and are dropped on save), `cmap`/`norm` are rebuilt via
    [`build_atlas_cmap_and_norm`][confusius._utils.atlas.build_atlas_cmap_and_norm] so
    atlas-derived masks and annotations keep their canonical colors after reload.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the file to load.
    variable : str, optional
        Zarr only. Name of the variable to extract as a DataArray. If not provided, the
        first variable in the dataset is returned.
    **kwargs
        Additional keyword arguments forwarded to the underlying loader.

    Returns
    -------
    xarray.DataArray
        The loaded data.

    Raises
    ------
    ValueError
        If the file extension is not supported.
    """
    path = check_path(path)
    name = path.name

    if name.endswith((".nii", ".nii.gz")):
        data_array = _nifti.load_nifti(path, **kwargs)
    elif name.endswith(".scan"):
        data_array = _scan.load_scan(path, **kwargs)
    elif name.endswith(".zarr"):
        ds = xr.open_zarr(path, **kwargs)
        data_array = (
            ds[variable] if variable is not None else ds[next(iter(ds.data_vars))]
        )
        data_array = _restore_voxel_world_index_from_coords(data_array)
    else:
        raise ValueError(
            f"Unsupported file extension in {name!r}. Supported"
            " extensions are: .nii, .nii.gz, .scan, .zarr."
        )

    restore_atlas_cmap_and_norm(data_array)
    restore_affines_in_attrs(data_array.attrs)
    return data_array


def save(data_array: xr.DataArray, path: str | Path, **kwargs: Any) -> None:
    """Save a fUSI DataArray to file, dispatching by extension.

    Supported formats:

    - **NIfTI** (`.nii`, `.nii.gz`): saved via
      [`save_nifti`][confusius.io.save_nifti].
    - **Zarr** (`.zarr`): saved via
      [`xarray.DataArray.to_zarr`][xarray.DataArray.to_zarr].

    Parameters
    ----------
    data_array : xarray.DataArray
        DataArray to save.
    path : str or pathlib.Path
        Output path. The extension determines the format.
    **kwargs
        Additional keyword arguments forwarded to the underlying saver.

    Raises
    ------
    ValueError
        If the file extension is not supported.
    """
    path = check_path(path)
    name = path.name

    if name.endswith((".nii", ".nii.gz")):
        _nifti.save_nifti(data_array, path, **kwargs)
        return
    if name.endswith(".zarr"):
        data_array = ensure_fusi(data_array)
        data_array = data_array.copy(deep=False)
        data_array.attrs = make_attrs_zarr_safe(data_array.attrs)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=ZARR_V3_CONSOLIDATED_METADATA_WARNING,
            )
            data_array.to_zarr(path, **kwargs)
        return

    raise ValueError(
        f"Unsupported file extension in {name!r}. Supported"
        " extensions are: .nii, .nii.gz, .zarr."
    )

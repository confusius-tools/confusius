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
    attach_voxel_to_world_index,
    get_voxel_to_world_affine,
    get_voxel_to_world_coord_names,
)
from confusius.io._utils import (
    ZARR_V3_CONSOLIDATED_METADATA_WARNING,
    make_attrs_zarr_safe,
    restore_affines_in_attrs,
)
from confusius.io.utils import check_path
from confusius.validation import ensure_fusi


def load(path: str | Path, variable: str | None = None, **kwargs: Any) -> xr.DataArray:
    """Load VoxelData from file, dispatching by extension.

    Supported formats:

    - **NIfTI** (`.nii`, `.nii.gz`): loaded via [`load_nifti`][confusius.io.load_nifti].
    - **SCAN** (`.scan`): loaded via [`load_scan`][confusius.io.load_scan].
    - **Zarr** (`.zarr`): opened via [`xarray.open_zarr`][xarray.open_zarr] and a single
      variable is extracted. Must be a store previously written by
      [`save`][confusius.io.save] (identified by `attrs["voxel_to_world"]`); for an
      arbitrary/foreign Zarr store, use [`xarray.open_zarr`][xarray.open_zarr] directly
      and build VoxelData yourself (e.g. via
      [`create_fusi_dataarray`][confusius.xarray.create_fusi_dataarray]).

    If `attrs["rgb_lookup"]` is present but `attrs["cmap"]`/`attrs["norm"]` are missing
    (as happens after a save/load round-trip, since matplotlib colormap/norm objects are
    not JSON-serializable and are dropped on save), `cmap`/`norm` are rebuilt via
    [`build_atlas_cmap_and_norm`][confusius._utils.atlas.build_atlas_cmap_and_norm] so
    atlas-derived masks and annotations keep their canonical colors after reload.

    A Zarr-saved VoxelData DataArray stores `attrs["voxel_to_world"]` instead of
    dense `z`/`y`/`x` coordinate arrays (see
    [`save`][confusius.io.save]); this rebuilds the world coordinates and
    `VoxelToWorldIndex` from that affine.

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
        If the file extension is not supported, or the Zarr store at `path` wasn't
        written by [`save`][confusius.io.save] (no `attrs["voxel_to_world"]`).
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
        if "voxel_to_world" not in data_array.attrs:
            raise ValueError(
                f"{path} was not written by confusius.io.save() (no "
                "attrs['voxel_to_world']). Use xarray.open_zarr directly to load an "
                "arbitrary Zarr store."
            )
        voxel_to_world = np.asarray(
            data_array.attrs.pop("voxel_to_world"), dtype=np.float64
        )
        world_coord_attrs = data_array.attrs.pop("world_coord_attrs", None)
        data_array = attach_voxel_to_world_index(
            data_array, voxel_to_world, world_coord_attrs=world_coord_attrs
        )
    else:
        raise ValueError(
            f"Unsupported file extension in {name!r}. Supported"
            " extensions are: .nii, .nii.gz, .scan, .zarr."
        )

    restore_atlas_cmap_and_norm(data_array)
    restore_affines_in_attrs(data_array.attrs)
    return data_array


def save(data_array: xr.DataArray, path: str | Path, **kwargs: Any) -> None:
    """Save VoxelData to file, dispatching by extension.

    Supported formats:

    - **NIfTI** (`.nii`, `.nii.gz`): saved via
      [`save_nifti`][confusius.io.save_nifti].
    - **Zarr** (`.zarr`): saved via
      [`xarray.DataArray.to_zarr`][xarray.DataArray.to_zarr]. Voxel-to-world geometry is
      stored as `attrs["voxel_to_world"]` rather than dense `z`/`y`/`x` coordinate
      arrays, since those are cheaply derived from the affine on load and, for
      oblique geometry, would otherwise duplicate a full dense array per axis.

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
        # ensure_fusi always returns a voxel-to-world-indexed DataArray (validate_fusi
        # unconditionally requires the index), so this always stores geometry as
        # attrs["voxel_to_world"] rather than dense z/y/x coordinate arrays.
        data_array = ensure_fusi(data_array)
        data_array = data_array.copy(deep=False)
        voxel_to_world = get_voxel_to_world_affine(data_array)
        world_coord_names = get_voxel_to_world_coord_names(data_array)
        world_coord_attrs = {
            coord_name: dict(data_array.coords[coord_name].attrs)
            for coord_name in world_coord_names
            if coord_name in data_array.coords
        }
        data_array = data_array.drop_vars(world_coord_names)
        data_array.attrs = {
            **data_array.attrs,
            "voxel_to_world": voxel_to_world,
            "world_coord_attrs": world_coord_attrs,
        }
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

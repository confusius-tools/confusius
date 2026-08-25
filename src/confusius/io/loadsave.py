"""Generic file loading and saving dispatcher."""

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

import confusius.io.echoframe as _echoframe
import confusius.io.nifti as _nifti
import confusius.io.scan as _scan
from confusius._dims import WORLD_DIMS
from confusius._utils.atlas import restore_atlas_cmap_and_norm
from confusius._utils.geometry import (
    attach_voxel_to_world_index,
    get_voxel_to_world_affine,
    get_voxel_to_world_units,
)
from confusius.io._utils import (
    ZARR_V3_CONSOLIDATED_METADATA_WARNING,
    make_attrs_zarr_safe,
    restore_affines_in_attrs,
)
from confusius.io.utils import check_path
from confusius.validation import ensure_voxeldata


def load(path: str | Path, variable: str | None = None, **kwargs: Any) -> xr.DataArray:
    """Load a VoxelData array from file, dispatching by extension.

    Supported formats:

    - **NIfTI** (`.nii`, `.nii.gz`): loaded via [`load_nifti`][confusius.io.load_nifti].
    - **SCAN** (`.scan`): loaded via [`load_scan`][confusius.io.load_scan].
    - **EchoFrame DAT** (`.dat`): loaded via
      [`load_echoframe_dat`][confusius.io.load_echoframe_dat]. If no metadata path is
      provided, [`load_echoframe_dat`][confusius.io.load_echoframe_dat] looks for
      `ScanParameters.mat` next to the DAT file.
    - **Zarr** (`.zarr`): opened via [`xarray.open_zarr`][xarray.open_zarr] and a single
      variable is extracted. Must be a store previously written by
      [`save`][confusius.io.save] (identified by `attrs["voxel_to_world"]`); for an
      arbitrary/foreign Zarr store, use [`xarray.open_zarr`][xarray.open_zarr] directly
      and build a VoxelData array yourself (e.g. via
      [`create_voxeldata`][confusius.xarray.create_voxeldata]).

    If `attrs["rgb_lookup"]` is present but `attrs["cmap"]`/`attrs["norm"]` are missing
    (as happens after a save/load round-trip, since matplotlib colormap/norm objects are
    not JSON-serializable and are dropped on save), `cmap`/`norm` are rebuilt via
    [`build_atlas_cmap_and_norm`][confusius._utils.atlas.build_atlas_cmap_and_norm] so
    atlas-derived masks and annotations keep their canonical colors after reload.

    A Zarr-saved VoxelData array stores `attrs["voxel_to_world"]`
    instead of dense `z`/`y`/`x` coordinate arrays (see
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
        Loaded VoxelData array.

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
    elif name.endswith(".dat"):
        data_array = _echoframe.load_echoframe_dat(path, **kwargs)
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
        units = data_array.attrs.pop("voxel_to_world_units", "mm")
        data_array = attach_voxel_to_world_index(
            data_array, voxel_to_world, units=units
        )
    else:
        raise ValueError(
            f"Unsupported file extension in {name!r}. Supported"
            " extensions are: .nii, .nii.gz, .scan, .dat, .zarr."
        )

    restore_atlas_cmap_and_norm(data_array)
    restore_affines_in_attrs(data_array.attrs)
    return data_array


def save(data_array: xr.DataArray, path: str | Path, **kwargs: Any) -> None:
    """Save a VoxelData array to file, dispatching by extension.

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
        VoxelData array to save.
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
        # ensure_voxeldata always returns a voxel-to-world-indexed DataArray (validate_voxeldata
        # unconditionally requires the index), so this always stores geometry as
        # attrs["voxel_to_world"] rather than dense z/y/x coordinate arrays.
        data_array = ensure_voxeldata(data_array)
        data_array = data_array.copy(deep=False)
        voxel_to_world = get_voxel_to_world_affine(data_array)
        units = get_voxel_to_world_units(data_array)
        # `pose` is its own plain, independently indexed coordinate (not owned by the
        # VoxelToWorldIndex -- see its docstring), so dropping the world coordinates
        # here leaves it untouched.
        data_array = data_array.drop_vars(WORLD_DIMS)
        data_array.attrs = {
            **data_array.attrs,
            "voxel_to_world": voxel_to_world,
            "voxel_to_world_units": units,
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

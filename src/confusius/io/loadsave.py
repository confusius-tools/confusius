"""Generic file loading and saving dispatcher."""

from pathlib import Path
from typing import Any

import voxeldata as vd
import xarray as xr

import confusius.io.echoframe as _echoframe
import confusius.io.iconeus_raw as _iconeus_raw
import confusius.io.nifti as _nifti
import confusius.io.scan as _scan
from confusius._utils.atlas import restore_atlas_cmap_and_norm
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
    - **Iconeus RAW** (`.raw`): loaded via
      [`load_iconeus_raw`][confusius.io.load_iconeus_raw]. If no metadata path is
      provided, [`load_iconeus_raw`][confusius.io.load_iconeus_raw] looks for the
      matching `.hraw` sidecar next to the RAW file.
    - **Zarr** (`.zarr`): opened via VoxelData's `load_array`, which restores spatial
      geometry and named frames. Must be a store previously written by
      [`save`][confusius.io.save] or VoxelData (identified by
      `attrs["voxel_to_world"]`); for an
      arbitrary/foreign Zarr store, use [`xarray.open_zarr`][xarray.open_zarr] directly
      and build a VoxelData array yourself (e.g. via
      [`create_voxeldata`][confusius.xarray.create_voxeldata]).

    If `attrs["rgb_lookup"]` is present but `attrs["cmap"]`/`attrs["norm"]` are missing
    (as happens after a save/load round-trip, since matplotlib colormap/norm objects are
    not JSON-serializable and are dropped on save), the colormap and normalization
    are rebuilt from the RGB lookup table so atlas-derived masks and annotations
    keep their canonical colors after reload.

    A Zarr-saved VoxelData array stores `attrs["voxel_to_world"]`
    instead of dense `z`/`y`/`x` coordinate arrays (see
    [`save`][confusius.io.save]); this rebuilds the world coordinates and
    `VoxelToWorldIndex` from that affine.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the file to load. For NIfTI files, a `{chunk}` placeholder in the
        filename loads matching integer chunks in numeric order along `pose`;
        see [load_nifti][confusius.io.load_nifti].
    variable : str, optional
        Zarr only. Name of the variable to extract as a DataArray. Required for
        datasets with multiple data variables; a single variable is selected
        automatically.
    **kwargs
        Additional keyword arguments forwarded to the underlying loader. Zarr
        defaults to `chunks="auto"` for Dask-backed loading; pass `chunks=None`
        for backend lazy indexing without Dask arrays.

    Returns
    -------
    xarray.DataArray
        Loaded VoxelData array.

    Raises
    ------
    ValueError
        If the file extension is not supported, the Zarr store lacks
        `attrs["voxel_to_world"]`, or multiple variables require an explicit choice.

    Notes
    -----
    Gzip-compressed NIfTI files (`.nii.gz`) do not support true random access. Repeated
    lazy partial reads, such as per-frame loops, may decompress the file again for each
    independent Dask computation. If the file fits in memory, use `.compute()` for eager
    processing, or `.persist()` before many downstream Dask operations. For larger
    repeated random-access workloads, prefer Zarr or uncompressed `.nii`.
    """
    path = check_path(path)
    name = path.name

    if name.endswith((".nii", ".nii.gz")):
        data_array = _nifti.load_nifti(path, **kwargs)
    elif name.endswith(".scan"):
        data_array = _scan.load_scan(path, **kwargs)
    elif name.endswith(".dat"):
        data_array = _echoframe.load_echoframe_dat(path, **kwargs)
    elif name.endswith(".raw"):
        data_array = _iconeus_raw.load_iconeus_raw(path, **kwargs)
    elif name.endswith(".zarr"):
        kwargs.setdefault("chunks", "auto")
        data_array = vd.load_array(path, variable=variable, **kwargs)
    else:
        raise ValueError(
            f"Unsupported file extension in {name!r}. Supported"
            " extensions are: .nii, .nii.gz, .scan, .dat, .raw, .zarr."
        )

    restore_atlas_cmap_and_norm(data_array)
    return data_array


def save(data_array: xr.DataArray, path: str | Path, **kwargs: Any) -> None:
    """Save a VoxelData array to file, dispatching by extension.

    Supported formats:

    - **NIfTI** (`.nii`, `.nii.gz`): saved via
      [`save_nifti`][confusius.io.save_nifti].
    - **Zarr** (`.zarr`): saved via VoxelData's `save`. Voxel-to-world geometry is
      stored as `attrs["voxel_to_world"]` rather than dense `z`/`y`/`x` coordinate
      arrays, since those are cheaply derived from the affine on load and, for
      oblique geometry, would otherwise duplicate a full dense array per axis.

    Parameters
    ----------
    data_array : xarray.DataArray
        VoxelData array to save.
    path : str or pathlib.Path
        Output path. The extension determines the format. For NIfTI, a `{chunk}`
        placeholder in the filename saves one file and JSON sidecar per pose,
        using non-negative integer `pose` labels as chunk indices.
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
        vd.save(ensure_voxeldata(data_array), path, **kwargs)
        return

    raise ValueError(
        f"Unsupported file extension in {name!r}. Supported"
        " extensions are: .nii, .nii.gz, .zarr."
    )

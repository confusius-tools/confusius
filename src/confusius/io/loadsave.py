"""Generic file loading and saving dispatcher."""

from pathlib import Path
from typing import Any

import xarray as xr

import confusius.io.nifti as _nifti
import confusius.io.scan as _scan
from confusius.io.utils import check_path


def load(path: str | Path, variable: str | None = None, **kwargs: Any) -> xr.DataArray:
    """Load a fUSI DataArray from file, dispatching by extension.

    Supported formats:

    - **NIfTI** (`.nii`, `.nii.gz`): loaded via [`load_nifti`][confusius.io.load_nifti].
    - **SCAN** (`.scan`): loaded via [`load_scan`][confusius.io.load_scan].
    - **Zarr** (`.zarr`): opened via [`xarray.open_zarr`][xarray.open_zarr] and a single
      variable is extracted. For loading the full dataset, use
      [`xarray.open_zarr`][xarray.open_zarr] directly.

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

    if name.endswith(".nii") or name.endswith(".nii.gz"):
        return _nifti.load_nifti(path, **kwargs)
    if name.endswith(".scan"):
        return _scan.load_scan(path, **kwargs)
    if name.endswith(".zarr"):
        ds = xr.open_zarr(path, **kwargs)
        if variable is not None:
            return ds[variable]
        first_var = next(iter(ds.data_vars))
        return ds[first_var]

    raise ValueError(
        f"Unsupported file extension in {name!r}. Supported"
        " extensions are: .nii, .nii.gz, .scan, .zarr."
    )


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

    if name.endswith(".nii") or name.endswith(".nii.gz"):
        _nifti.save_nifti(data_array, path, **kwargs)
        return
    if name.endswith(".zarr"):
        data_array.to_zarr(path, **kwargs)
        return

    raise ValueError(
        f"Unsupported file extension in {name!r}. Supported"
        " extensions are: .nii, .nii.gz, .zarr."
    )

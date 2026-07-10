"""Generic file loading and saving dispatcher."""

from pathlib import Path
from typing import Any

import xarray as xr

import confusius.io.nifti as _nifti
import confusius.io.scan as _scan
from confusius._utils.atlas import build_atlas_cmap_and_norm
from confusius._utils.io import restore_affines, zarr_safe_attrs
from confusius.io.utils import check_path


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

    if name.endswith(".nii") or name.endswith(".nii.gz"):
        data_array = _nifti.load_nifti(path, **kwargs)
    elif name.endswith(".scan"):
        data_array = _scan.load_scan(path, **kwargs)
    elif name.endswith(".zarr"):
        ds = xr.open_zarr(path, **kwargs)
        data_array = (
            ds[variable] if variable is not None else ds[next(iter(ds.data_vars))]
        )
    else:
        raise ValueError(
            f"Unsupported file extension in {name!r}. Supported"
            " extensions are: .nii, .nii.gz, .scan, .zarr."
        )

    _restore_atlas_cmap_and_norm(data_array)
    restore_affines(data_array.attrs)
    return data_array


def _restore_atlas_cmap_and_norm(data_array: xr.DataArray) -> None:
    """Rebuild `cmap`/`norm` attrs from `rgb_lookup` in place, when missing.

    Parameters
    ----------
    data_array : xarray.DataArray
        DataArray to update in place.

    Returns
    -------
    None
        This function mutates `data_array.attrs` and returns nothing.
    """
    if "rgb_lookup" not in data_array.attrs:
        return
    if "cmap" in data_array.attrs and "norm" in data_array.attrs:
        return
    cmap, norm = build_atlas_cmap_and_norm(data_array.attrs["rgb_lookup"])
    data_array.attrs["cmap"] = cmap
    data_array.attrs["norm"] = norm


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
        data_array = data_array.copy(deep=False)
        data_array.attrs = zarr_safe_attrs(data_array.attrs)
        data_array.to_zarr(path, **kwargs)
        return

    raise ValueError(
        f"Unsupported file extension in {name!r}. Supported"
        " extensions are: .nii, .nii.gz, .zarr."
    )

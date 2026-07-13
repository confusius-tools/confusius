"""Generic file loading and saving dispatcher."""

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

import confusius.io.nifti as _nifti
import confusius.io.scan as _scan
from confusius._utils.atlas import build_atlas_cmap_and_norm
from confusius._utils.stack import find_stack_level
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
    _restore_affines(data_array)
    return data_array


def _restore_affines(data_array: xr.DataArray) -> None:
    """Restore `attrs["affines"]` dict values to numpy arrays in place.

    Zarr stores the affines as nested lists (see
    [`save`][confusius.io.save]); this converts them back to numpy arrays so a
    Zarr round-trip matches the NIfTI and SCAN loaders. A no-op when `affines` is
    absent or its values are already arrays.

    Parameters
    ----------
    data_array : xarray.DataArray
        DataArray to update in place.

    Returns
    -------
    None
        This function mutates `data_array.attrs` and returns nothing.
    """
    affines = data_array.attrs.get("affines")
    if not isinstance(affines, dict):
        return
    data_array.attrs["affines"] = {
        key: np.asarray(value) for key, value in affines.items()
    }


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
        data_array.attrs = _zarr_safe_attrs(data_array.attrs)
        data_array.to_zarr(path, **kwargs)
        return

    raise ValueError(
        f"Unsupported file extension in {name!r}. Supported"
        " extensions are: .nii, .nii.gz, .zarr."
    )


def _to_json_serializable(value: Any) -> Any:
    """Recursively convert numpy containers and scalars to native Python objects.

    Parameters
    ----------
    value : Any
        Attribute value, possibly a numpy array or scalar or a `dict`/`list`/`tuple`
        nesting them.

    Returns
    -------
    Any
        Equivalent value with every `numpy.ndarray` replaced by a nested list and every
        numpy scalar replaced by its Python counterpart. Non-numpy values are returned
        unchanged.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _to_json_serializable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_to_json_serializable(item) for item in value]
    return value


def _zarr_safe_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    """Make attributes safe to store as Zarr attributes.

    Zarr stores attributes as JSON. Numpy arrays and scalars, including those nested
    inside dicts or lists such as `attrs["affines"]`, are converted to native Python
    objects. Any remaining value that cannot be JSON-encoded (e.g. the matplotlib
    colormap and normalization objects on atlas-derived data) is dropped with a warning;
    such `cmap`/`norm` attrs are rebuilt from `rgb_lookup` on load.

    Parameters
    ----------
    attrs : dict[str, Any]
        Attributes to sanitize.

    Returns
    -------
    dict[str, Any]
        Copy of `attrs` with numpy values converted to native Python and
        non-JSON-serializable entries removed.
    """
    safe: dict[str, Any] = {}
    dropped: list[str] = []
    for key, value in attrs.items():
        converted = _to_json_serializable(value)
        try:
            json.dumps(converted)
        except TypeError:
            dropped.append(key)
        else:
            safe[key] = converted

    if dropped:
        warnings.warn(
            f"Dropping non-JSON-serializable attrs from Zarr store: {dropped}.",
            stacklevel=find_stack_level(),
        )

    return safe

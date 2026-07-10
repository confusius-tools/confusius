"""File readers for ConfUSIus data formats (npe2).

These are called by napari when files are opened via File → Open, drag-and-drop,
or the CLI.

ConfUSIus loaders use voxel-affine coordinates: SCAN and NIfTI files are loaded into
DataArrays, typically with native voxel dimensions such as `k/j/i` plus linked
physical `z/y/x` coordinates. Napari display uses physical `z/y/x` axes:
axis-aligned data is promoted to a plain physical grid without interpolation, while
oblique or sheared data is resampled to an axis-aligned physical display grid that
napari can represent.

Each public function is a `get_reader` command: it receives the path, does a
lightweight validity check, and either returns `None` (cannot read) or a
`ReaderFunction` that does the actual loading.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from napari.layers.utils.layer_utils import calc_data_range
from napari.utils.colormaps import ensure_colormap
from napari.utils.notifications import show_warning

from confusius._utils.coordinates import get_coordinate_spacings_best_effort
from confusius._utils.napari import (
    build_direct_label_colormap,
    build_roi_labels_features,
    infer_layer_type,
)
from confusius.io import load
from confusius.plotting._utils import (
    convert_axis_aligned_voxel_affine_to_physical_grid,
    resample_voxel_affine_to_physical_grid,
)

if TYPE_CHECKING:
    import xarray as xr
    from napari.types import FullLayerData, PathOrPaths


def _get_napari_scale_translate_units(
    data: xr.DataArray,
) -> tuple[list[float], list[float], list[str | None], list[str], dict[str, float]]:
    """Return napari layer geometry metadata for `data`."""
    all_dims = list(data.dims)

    spacing, non_uniform = get_coordinate_spacings_best_effort(data)
    origin = data.fusi.origin
    scale = [spacing[str(d)] for d in all_dims]
    translate = [
        origin[d]
        if d in origin
        else (
            float(np.asarray(data.coords[d].values, dtype=float)[0])
            if d in data.coords
            else 0.0
        )
        for d in all_dims
    ]
    units = [
        data.coords[d].attrs.get("units") if d in data.coords else None
        for d in all_dims
    ]
    return scale, translate, units, non_uniform, spacing


def _convert_dataarray_to_layer_data(da: xr.DataArray, name: str) -> FullLayerData:
    """Convert a ConfUSIus DataArray to a napari FullLayerData tuple.

    Mirrors the logic of [`plot_napari`][confusius.plotting.plot_napari]

    * Napari readers use physical display space by default, so layers are shown on
      physical `z/y/x` axes.
    * Axis-aligned data is promoted to plain physical `z/y/x` dimensions without
      interpolation.
    * Oblique or sheared data is resampled to an axis-aligned physical `z/y/x`
      display grid before layer creation.
    * Uses
      [`get_coordinate_spacings_best_effort`][confusius._utils.coordinates.get_coordinate_spacings_best_effort]
      for the `scale`: uniform coordinates use their exact spacing; non-uniform
      coordinates fall back to the median diff and a napari warning is shown.
    * Includes [`origin`][confusius.xarray.FUSIAccessor.origin] as `translate`
      so the layer is positioned correctly in physical space.
    * Passes `axis_labels` and `units` from the DataArray dimensions and coordinate
      attributes. These are stored on the layer but napari does not yet propagate them
      to `viewer.dims.axis_labels` when loading via a reader plugin.
    * Integer-dtype arrays (e.g. atlas annotations, ROI masks) are returned as a
      `"labels"` layer, with a per-label `DirectLabelColormap` built from `da.attrs`
      when available. All other dtypes are returned as an `"image"` layer, reading
      `"cmap"` from `da.attrs` for the colormap when present, falling back to `"gray"`
      (with a napari warning) when the stored value is not a valid napari colormap.
    """

    source_da = da
    da = convert_axis_aligned_voxel_affine_to_physical_grid(da)
    da = resample_voxel_affine_to_physical_grid(da)

    all_dims = list(da.dims)

    scale, translate, all_units, non_uniform, spacing = (
        _get_napari_scale_translate_units(da)
    )
    for dim in non_uniform:
        show_warning(
            f"'{dim}' has non-uniform spacing; using median {spacing[dim]:.4g} "
            "(positions along this axis may be approximate)."
        )

    kwargs: dict[str, Any] = {
        "name": name,
        "scale": scale,
        "translate": translate,
        "axis_labels": all_dims,
        "metadata": {"xarray": da, "source_xarray": source_da},
    }
    if any(u is not None for u in all_units):
        kwargs["units"] = all_units

    layer_type = infer_layer_type(da.dtype)
    if layer_type == "labels":
        colormap = build_direct_label_colormap(da)
        if colormap is not None:
            kwargs["colormap"] = colormap
        if (roi_labels := da.attrs.get("roi_labels")) is not None:
            kwargs["features"] = build_roi_labels_features(roi_labels)
    else:
        # Pre-compute contrast limits so napari displays the image correctly on load.
        # In napari 0.6.6+ the deferred _should_calc_clims mechanism does not fire
        # reliably for non-numpy data during the insertion event. calc_data_range
        # samples a few planes, so it is fast even for large arrays.
        colormap = da.attrs.get("cmap", "gray")
        try:
            ensure_colormap(colormap)
        except (KeyError, TypeError):
            show_warning(
                f"{colormap!r} is not a valid napari colormap; falling back to 'gray'."
            )
            colormap = "gray"
        kwargs["colormap"] = colormap
        kwargs["blending"] = "additive"
        kwargs["contrast_limits"] = calc_data_range(da.data)

    return da.data, kwargs, layer_type


def _make_reader(path: str | Path) -> Callable[[PathOrPaths], list[FullLayerData]]:
    """Return a `ReaderFunction` for `path`.

    The returned function loads the file via [`confusius.load`][confusius.load] (which
    dispatches on extension) and converts the result to a `FullLayerData` tuple using
    the same voxel-affine-aware display rules as
    [`plot_napari`][confusius.plotting.plot_napari]. This function may raise; napari
    will surface any exception to the user.
    """

    def _read(_path: PathOrPaths) -> list[FullLayerData]:
        # Use the pre-validated `path` captured from the outer scope rather than
        # `_path`, which may be a list when napari replays the reader.
        da = load(path)
        name = Path(path).name
        return [_convert_dataarray_to_layer_data(da, name)]

    return _read


def read_nifti(
    path: PathOrPaths,
) -> Callable[[PathOrPaths], list[FullLayerData]] | None:
    """Get reader for NIfTI files (`.nii` / `.nii.gz`).

    Parameters
    ----------
    path : PathOrPaths
        Path passed by napari.

    Returns
    -------
    Callable or None
        Reader function, or `None` if the path is a list or the file does not exist
        (napari will fall back to other plugins).
    """
    if isinstance(path, list) or not isinstance(path, (str, Path)):
        return None
    if not Path(path).is_file():
        return None
    return _make_reader(path)


def read_scan(path: PathOrPaths) -> Callable[[PathOrPaths], list[FullLayerData]] | None:
    """Get reader for Iconeus SCAN files (`.scan`).

    Parameters
    ----------
    path : PathOrPaths
        Path passed by napari.

    Returns
    -------
    Callable or None
        Reader function, or `None` if the path is a list or the file does not exist
        (napari will fall back to other plugins).
    """
    if isinstance(path, list) or not isinstance(path, (str, Path)):
        return None
    if not Path(path).is_file():
        return None
    return _make_reader(path)


def read_zarr(path: PathOrPaths) -> Callable[[PathOrPaths], list[FullLayerData]] | None:
    """Get reader for Zarr stores (`.zarr`).

    Validates that the path is a directory containing at least one of the standard Zarr
    metadata files (`.zgroup`, `.zattrs`, `zarr.json`).

    Parameters
    ----------
    path : PathOrPaths
        Path passed by napari.

    Returns
    -------
    Callable or None
        Reader function, or `None` if the path is a list, not a directory, or contains
        no Zarr metadata files (napari will fall back to other plugins).
    """
    if isinstance(path, list) or not isinstance(path, (str, Path)):
        return None
    p = Path(path)
    if not p.is_dir():
        return None
    zarr_indicators = (".zgroup", ".zattrs", "zarr.json", ".zarray")
    if not any((p / indicator).exists() for indicator in zarr_indicators):
        return None
    return _make_reader(path)

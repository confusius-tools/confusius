"""File readers for ConfUSIus data formats (npe2).

These are called by napari when files are opened via File → Open, drag-and-drop,
or the CLI.

Each public function is a `get_reader` command: it receives the path, does a
lightweight validity check, and either returns `None` (cannot read) or a
`ReaderFunction` that does the actual loading.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from napari.layers.utils.layer_utils import calc_data_range
from napari.utils.notifications import show_warning

from confusius._utils import _compute_spacing_best_effort
from confusius.io import load

if TYPE_CHECKING:
    import xarray as xr
    from napari.types import FullLayerData, PathOrPaths


def _da_to_layer_data(da: xr.DataArray, name: str) -> FullLayerData:
    """Convert a ConfUSIus DataArray to a napari FullLayerData tuple.

    Mirrors the logic of [`plot_napari`][confusius.plotting.plot_napari]

    * Uses
      [`_compute_spacing_best_effort`][confusius._utils._compute_spacing_best_effort]
      for the `scale`: uniform coordinates use their exact spacing; non-uniform
      coordinates fall back to the median diff and a napari warning is shown.
    * Includes [`origin`][confusius.xarray.FUSIAccessor.origin] as `translate`
      so the layer is positioned correctly in physical space.
    * Reads `"cmap"` from `da.attrs` for the colormap when present.
    * Passes `axis_labels` and `units` from the DataArray dimensions and coordinate
      attributes. These are stored on the layer but napari does not yet propagate them
      to `viewer.dims.axis_labels` when loading via a reader plugin.
    """

    all_dims = list(da.dims)

    spacing, non_uniform = _compute_spacing_best_effort(da)
    for dim in non_uniform:
        show_warning(
            f"'{dim}' has non-uniform spacing; using median {spacing[dim]:.4g} "
            "(positions along this axis may be approximate)."
        )
    origin = da.fusi.origin

    scale: list[float] = [spacing[str(d)] for d in all_dims]
    translate: list[float] = [origin[d] for d in all_dims]
    all_units: list[str | None] = [
        da.coords[d].attrs.get("units") if d in da.coords else None for d in all_dims
    ]

    # Pre-compute contrast limits so napari displays the image correctly on load. In
    # napari 0.6.6+ the deferred _should_calc_clims mechanism does not fire reliably for
    # non-numpy data during the insertion event. calc_data_range samples a few planes,
    # so it is fast even for large arrays.
    contrast_limits = calc_data_range(da.data)

    kwargs: dict[str, Any] = {
        "name": name,
        "scale": scale,
        "translate": translate,
        "axis_labels": all_dims,
        "colormap": da.attrs.get("cmap", "gray"),
        "blending": "additive",
        "metadata": {"xarray": da},
        "contrast_limits": contrast_limits,
    }
    if any(u is not None for u in all_units):
        kwargs["units"] = all_units

    return da.data, kwargs, "image"


def _make_reader(path: str | Path) -> Callable[[PathOrPaths], list[FullLayerData]]:
    """Return a `ReaderFunction` for `path`.

    The returned function loads the file via [`confusius.load`][confusius.load] (which
    dispatches on extension) and converts the result to a `FullLayerData` tuple. This
    function may raise; napari will surface any exception to the user.
    """

    def _read(_path: PathOrPaths) -> list[FullLayerData]:
        # Use the pre-validated `path` captured from the outer scope rather than
        # `_path`, which may be a list when napari replays the reader.
        da = load(path)
        name = Path(path).name
        return [_da_to_layer_data(da, name)]

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

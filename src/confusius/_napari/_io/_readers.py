"""File readers for ConfUSIus data formats (npe2).

These are called by napari when files are opened via File → Open, drag-and-drop,
or the CLI.

Each public function is a `get_reader` command: it receives the path, does a
lightweight validity check, and either returns `None` (cannot read) or a
`ReaderFunction` that does the actual loading.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from confusius._utils.napari import convert_dataarray_to_layer_data
from confusius.io import load

if TYPE_CHECKING:
    from napari.types import FullLayerData, PathOrPaths


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
        return [convert_dataarray_to_layer_data(da, name)]

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

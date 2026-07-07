"""Zarr save/load for atlas Datasets."""

from pathlib import Path
from typing import Any

import xarray as xr

from confusius._utils.atlas import build_atlas_cmap_and_norm

_NONSERIALIZABLE_ANNOTATION_ATTRS = ("cmap", "norm")
"""`annotation` attrs that are matplotlib objects and cannot be written to Zarr.

They are dropped on save and rebuilt from `rgb_lookup` on load, so an atlas keeps its
canonical region colors across a round-trip without persisting non-serializable objects.
"""


def atlas_to_zarr(ds: xr.Dataset, path: str | Path, **kwargs: Any) -> None:
    """Save an atlas Dataset to a Zarr store.

    The non-serializable `cmap`/`norm` matplotlib objects in `annotation.attrs` are
    stripped before writing (the caller's in-memory Dataset is left untouched);
    [`atlas_from_zarr`][confusius.atlas.atlas_from_zarr] rebuilds them from `rgb_lookup`
    on load.

    Parameters
    ----------
    ds : xarray.Dataset
        Atlas Dataset to save.
    path : str or pathlib.Path
        Output Zarr store path.
    **kwargs
        Additional keyword arguments forwarded to
        [`xarray.Dataset.to_zarr`][xarray.Dataset.to_zarr].

    Returns
    -------
    None
        The Dataset is written to `path`.
    """
    annotation = ds["annotation"].copy()
    annotation.attrs = {
        k: v
        for k, v in annotation.attrs.items()
        if k not in _NONSERIALIZABLE_ANNOTATION_ATTRS
    }
    to_save = ds.copy()
    to_save["annotation"] = annotation
    to_save.to_zarr(path, **kwargs)


def atlas_from_zarr(path: str | Path, **kwargs: Any) -> xr.Dataset:
    """Load an atlas Dataset from a Zarr store.

    The `cmap`/`norm` colormap objects dropped on save are rebuilt into
    `annotation.attrs` from `rgb_lookup`. Meshes are resolved lazily by the `.atlas`
    accessor from `attrs["atlas_name"]`, so `get_mesh` on a loaded atlas requires the
    named atlas to be present in the local BrainGlobe cache (it is re-fetched by name if
    missing).

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the Zarr store.
    **kwargs
        Additional keyword arguments forwarded to
        [`xarray.open_zarr`][xarray.open_zarr].

    Returns
    -------
    xarray.Dataset
        The loaded atlas Dataset, with `cmap`/`norm` restored on `annotation.attrs`.
    """
    ds = xr.open_zarr(path, **kwargs)
    annotation = ds["annotation"]
    if "rgb_lookup" in annotation.attrs and not all(
        k in annotation.attrs for k in _NONSERIALIZABLE_ANNOTATION_ATTRS
    ):
        cmap, norm = build_atlas_cmap_and_norm(annotation.attrs["rgb_lookup"])
        annotation.attrs["cmap"] = cmap
        annotation.attrs["norm"] = norm
    return ds

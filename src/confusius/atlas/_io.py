"""Zarr save/load for atlas Datasets."""

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from confusius._utils.atlas import build_atlas_cmap_and_norm

_NONSERIALIZABLE_ANNOTATION_ATTRS = ("cmap", "norm")
"""`annotation` attrs that are matplotlib objects and cannot be written to Zarr.

They are dropped on save and rebuilt from `rgb_lookup` on load, so an atlas keeps its
canonical region colors across a round-trip without persisting non-serializable objects.
"""

_MESHES_SUBDIR = "meshes"
"""Subdirectory of the Zarr store that holds the bundled region OBJ meshes.

The `.obj` files are written as plain sibling files inside the store directory so they
travel with it when the store is copied or zipped. `atlas_from_zarr` re-points each ROI's
`mesh_filename` here, so `get_mesh` works on a loaded atlas without the BrainGlobe cache.
"""


def _encode_affines_attr(attrs: dict[str, Any]) -> dict[str, Any]:
    """Copy `attrs` with any `affines` matrices turned into JSON-native nested lists.

    A resampled atlas inherits an `affines` attr (a dict of `(4, 4)` numpy arrays mapping
    the grid to world space) from the fUSI volume it was warped onto. Zarr cannot serialize
    numpy arrays nested inside an attr, so each matrix is converted to a nested list;
    [`atlas_from_zarr`][confusius.atlas.atlas_from_zarr] restores them to arrays on load.

    Parameters
    ----------
    attrs : dict[str, typing.Any]
        Variable or Dataset attributes to encode.

    Returns
    -------
    dict[str, typing.Any]
        A shallow copy of `attrs` whose `affines` matrices (if any) are nested lists.
    """
    if "affines" not in attrs:
        return dict(attrs)
    encoded = dict(attrs)
    encoded["affines"] = {
        key: np.asarray(matrix).tolist() for key, matrix in attrs["affines"].items()
    }
    return encoded


def _decode_affines_attr(attrs: dict[str, Any]) -> None:
    """Restore `attrs["affines"]` list matrices to numpy arrays, in place.

    Parameters
    ----------
    attrs : dict[str, typing.Any]
        Variable or Dataset attributes read back from a Zarr store.

    Returns
    -------
    None
        `attrs` is mutated in place.
    """
    if "affines" in attrs:
        attrs["affines"] = {
            key: np.asarray(matrix, dtype=np.float64)
            for key, matrix in attrs["affines"].items()
        }


def _plan_mesh_bundle(structures_blob: str) -> tuple[str, dict[str, Path]]:
    """Basename each `mesh_filename` and list the OBJ files that must be copied.

    Pure planning step: does not touch the filesystem, so the caller can write the Zarr
    store first (the store directory must not already exist) and copy the meshes into it
    afterwards.

    Parameters
    ----------
    structures_blob : str
        Serialized structures list (`attrs["structures"]`), with complete `mesh_filename`
        paths.

    Returns
    -------
    blob : str
        A new serialized structures list whose non-null `mesh_filename` entries are bare
        basenames.
    to_copy : dict[str, pathlib.Path]
        Mapping from basename to source path for every referenced OBJ file that exists on
        disk (missing sources are skipped).
    """
    structures = json.loads(structures_blob)
    to_copy: dict[str, Path] = {}
    for record in structures:
        mesh_filename = record.get("mesh_filename")
        if mesh_filename is None:
            continue
        source = Path(mesh_filename)
        record["mesh_filename"] = source.name
        if source.is_file():
            to_copy[source.name] = source
    return json.dumps(structures), to_copy


def _rebase_meshes(structures_blob: str, meshes_dir: Path) -> str:
    """Re-point each non-null `mesh_filename` basename into `meshes_dir`.

    Parameters
    ----------
    structures_blob : str
        Serialized structures list read back from a store, with basename `mesh_filename`
        entries.
    meshes_dir : pathlib.Path
        Directory holding the bundled OBJ files.

    Returns
    -------
    str
        A new serialized structures list whose non-null `mesh_filename` entries are
        complete paths under `meshes_dir`.
    """
    structures = json.loads(structures_blob)
    for record in structures:
        if record.get("mesh_filename") is not None:
            record["mesh_filename"] = str(
                meshes_dir / Path(record["mesh_filename"]).name
            )
    return json.dumps(structures)


def atlas_to_zarr(ds: xr.Dataset, path: str | Path, **kwargs: Any) -> None:
    """Save an atlas Dataset to a Zarr store, bundling its region meshes.

    The region OBJ meshes are copied into a `meshes/` subdirectory of the store (as plain
    sibling files) and each ROI's `mesh_filename` is stored as a basename, so a loaded
    atlas can render meshes without the BrainGlobe cache. The non-serializable `cmap`/`norm`
    matplotlib objects in `annotation.attrs` are stripped before writing (the caller's
    in-memory Dataset is left untouched); [`atlas_from_zarr`][confusius.atlas.atlas_from_zarr]
    rebuilds them from `rgb_lookup` and re-points the mesh paths on load.

    Parameters
    ----------
    ds : xarray.Dataset
        Atlas Dataset to save.
    path : str or pathlib.Path
        Output Zarr store path. Must be a filesystem path (the meshes are written as
        sibling files); zarr store objects are not supported.
    **kwargs
        Additional keyword arguments forwarded to
        [`xarray.Dataset.to_zarr`][xarray.Dataset.to_zarr].

    Returns
    -------
    None
        The Dataset is written to `path`.
    """
    path = Path(path)

    annotation = ds["annotation"].copy()
    annotation.attrs = {
        k: v
        for k, v in annotation.attrs.items()
        if k not in _NONSERIALIZABLE_ANNOTATION_ATTRS
    }
    to_save = ds.copy()
    to_save["annotation"] = annotation

    # JSON-encode any `affines` matrices (a resampled atlas inherits them, as dicts of
    # numpy arrays, from the fUSI grid it was warped onto). zarr cannot serialize numpy
    # arrays nested inside an attr; atlas_from_zarr restores them to arrays on load.
    for name in list(to_save.data_vars):
        if "affines" in to_save[name].attrs:
            var = to_save[name].copy()
            var.attrs = _encode_affines_attr(var.attrs)
            to_save[name] = var
    to_save.attrs = _encode_affines_attr(to_save.attrs)

    # A nonlinear (displacement-field) mesh transform carries a decorative string
    # `component` coordinate that zarr v3 cannot serialize stably. Replace it with integer
    # indices, which serialize cleanly and keep `.fusi.spacing` defined on load; the
    # transform math uses the dimension order, not the component labels.
    if "mesh_vertex_transform" in to_save.data_vars and "component" in to_save.coords:
        to_save = to_save.assign_coords(component=np.arange(to_save.sizes["component"]))

    to_copy: dict[str, Path] = {}
    if "structures" in to_save.attrs:
        blob, to_copy = _plan_mesh_bundle(to_save.attrs["structures"])
        to_save.attrs = {**to_save.attrs, "structures": blob}

    # Write the store first (to_zarr requires the path not to exist yet), then copy the
    # OBJ files into its meshes/ subdirectory.
    to_save.to_zarr(path, **kwargs)
    if to_copy:
        meshes_dir = path / _MESHES_SUBDIR
        meshes_dir.mkdir(parents=True, exist_ok=True)
        for basename, source in to_copy.items():
            shutil.copyfile(source, meshes_dir / basename)


def atlas_from_zarr(path: str | Path, **kwargs: Any) -> xr.Dataset:
    """Load an atlas Dataset from a Zarr store.

    The `cmap`/`norm` colormap objects dropped on save are rebuilt into `annotation.attrs`
    from `rgb_lookup`, and each ROI's `mesh_filename` is re-pointed at the meshes bundled
    under the store's `meshes/` subdirectory, so `get_mesh` works on the loaded atlas
    without the BrainGlobe cache.

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
        The loaded atlas Dataset, with `cmap`/`norm` restored on `annotation.attrs` and
        `mesh_filename` paths pointing at the bundled meshes.
    """
    path = Path(path)
    ds = xr.open_zarr(path, **kwargs)

    # Restore any `affines` matrices JSON-encoded on save back to numpy arrays, so they
    # match the arrays a NIfTI-loaded volume carries.
    for name in ds.data_vars:
        _decode_affines_attr(ds[name].attrs)
    _decode_affines_attr(ds.attrs)

    annotation = ds["annotation"]
    if "rgb_lookup" in annotation.attrs and not all(
        k in annotation.attrs for k in _NONSERIALIZABLE_ANNOTATION_ATTRS
    ):
        cmap, norm = build_atlas_cmap_and_norm(annotation.attrs["rgb_lookup"])
        annotation.attrs["cmap"] = cmap
        annotation.attrs["norm"] = norm

    if "structures" in ds.attrs:
        ds.attrs["structures"] = _rebase_meshes(
            ds.attrs["structures"], path / _MESHES_SUBDIR
        )
    return ds

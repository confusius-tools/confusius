"""Zarr save/load for atlas Datasets."""

import json
import shutil
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr

from confusius._utils.atlas import restore_atlas_cmap_and_norm
from confusius._utils.io import make_attrs_zarr_safe, restore_affines_in_attrs
from confusius.io.utils import check_path

if TYPE_CHECKING:
    from brainglobe_atlasapi.structure_class import StructuresDict

_SERIALIZED_STRUCTURE_FIELDS = (
    "id",
    "acronym",
    "name",
    "rgb_triplet",
    "structure_id_path",
    "mesh_filename",
)
"""Fields kept when serializing a structure to JSON.

These are exactly the fields BrainGlobe's own `structures.json` carries and the minimum
[`brainglobe_atlasapi.structure_class.StructuresDict`][brainglobe_atlasapi.structure_class.StructuresDict]
needs to rebuild its hierarchy tree (`id`, `acronym`, `structure_id_path`) plus the
metadata the atlas surfaces (`name`, `rgb_triplet`, `mesh_filename`).
"""

_NONSERIALIZABLE_ANNOTATION_ATTRS = ("cmap", "norm")
"""`annotation` attrs that are matplotlib objects and cannot be written to Zarr.

They are dropped on save and rebuilt from `rgb_lookup` on load, so an atlas keeps its
canonical region colors across a round-trip without persisting non-serializable objects.
"""

_ZARR_V3_CONSOLIDATED_METADATA_WARNING = (
    "Consolidated metadata is currently not part in the Zarr format 3 specification."
)
"""Zarr v3 warning text emitted when consolidated metadata is written."""

_MESHES_SUBDIR = "meshes"
"""Subdirectory of the Zarr store that holds the bundled region OBJ meshes.

The `.obj` files are written as plain sibling files inside the store directory so they
travel with it when the store is copied or zipped. `load_atlas` re-points each ROI's
`mesh_filename` here, so `get_mesh` works on a loaded atlas without the BrainGlobe cache.
"""


def structures_to_json(structures: "StructuresDict") -> str:
    """Serialize a StructuresDict to a flat BrainGlobe `structures.json` string.

    The `treelib` hierarchy is never serialized directly; it is a derived index that
    [`structures_from_json`][confusius.io.atlas.structures_from_json] rebuilds from the flat
    list. `mesh_filename` is stored verbatim (the complete path), so a freshly fetched atlas
    points straight at the OBJ files in the BrainGlobe cache. When the atlas is saved with
    [`save_atlas`][confusius.io.save_atlas], the meshes are bundled into the store and the
    paths are re-pointed there on load.

    Parameters
    ----------
    structures : brainglobe_atlasapi.structure_class.StructuresDict
        BrainGlobe structure dictionary.

    Returns
    -------
    str
        JSON-encoded list of structure dictionaries, each holding
        `id`, `acronym`, `name`, `rgb_triplet`, `structure_id_path`, and
        `mesh_filename` (complete path or `None`).
    """
    structures_list = []
    for _, info in structures.items():
        record = {field: info[field] for field in _SERIALIZED_STRUCTURE_FIELDS[:-1]}
        mesh_filename = info.get("mesh_filename")
        record["mesh_filename"] = (
            str(mesh_filename) if mesh_filename is not None else None
        )
        structures_list.append(record)
    return json.dumps(structures_list)


def structures_from_json(blob: str) -> "StructuresDict":
    """Rebuild a StructuresDict (and its tree) from a serialized structures list.

    Parameters
    ----------
    blob : str
        JSON string produced by
        [`structures_to_json`][confusius.io.atlas.structures_to_json].

    Returns
    -------
    brainglobe_atlasapi.structure_class.StructuresDict
        Reconstructed structure dictionary with a freshly built hierarchy tree.
    """
    from brainglobe_atlasapi.structure_class import StructuresDict

    return StructuresDict(json.loads(blob))


def _check_zarr_suffix(path: Path) -> None:
    """Raise ValueError unless `path` carries the `.zarr` suffix.

    Parameters
    ----------
    path : pathlib.Path
        Candidate atlas store path.

    Raises
    ------
    ValueError
        If `path` does not end with the `.zarr` suffix.
    """
    if path.suffix != ".zarr":
        raise ValueError(
            f"Atlas store path must end with the '.zarr' suffix, got '{path.name}'."
        )


def _plan_mesh_bundle(structures_blob: str) -> tuple[str, dict[str, Path]]:
    """Basename each `mesh_filename` and list the OBJ files that must be copied.

    Pure planning step: does not touch the filesystem, so the caller can write the Zarr
    store first (the store directory must not already exist) and copy the meshes into it
    afterwards.

    Parameters
    ----------
    structures_blob : str
        Serialized structures list (from `structures_to_json`), with complete
        `mesh_filename` paths.

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


def save_atlas(ds: xr.Dataset, path: str | Path, **kwargs: Any) -> None:
    """Save an atlas Dataset to a Zarr store, bundling its region meshes.

    The in-memory `attrs["structures"]`
    [`StructuresDict`][brainglobe_atlasapi.structure_class.StructuresDict] is serialized to
    a flat JSON list for storage. The region OBJ meshes are copied into a `meshes/`
    subdirectory of the store (as plain sibling files) and each ROI's `mesh_filename` is
    stored as a basename, so a loaded atlas can render meshes without the BrainGlobe cache.
    The non-serializable `cmap`/`norm` matplotlib objects in `annotation.attrs` are stripped
    before writing (the caller's in-memory Dataset is left untouched);
    [`load_atlas`][confusius.io.load_atlas] rebuilds the `StructuresDict`, the `cmap`/`norm`
    from `rgb_lookup`, and re-points the mesh paths on load.

    Parameters
    ----------
    ds : xarray.Dataset
        Atlas Dataset to save.
    path : str or pathlib.Path
        Output Zarr store path; must end with the `.zarr` suffix. Must be a filesystem path
        (the meshes are written as sibling files); zarr store objects are not supported.
    **kwargs
        Additional keyword arguments forwarded to
        [`xarray.Dataset.to_zarr`][xarray.Dataset.to_zarr].

    Returns
    -------
    None
        The Dataset is written to `path`.

    Raises
    ------
    ValueError
        If `path` does not end with the `.zarr` suffix.
    """
    path = check_path(path)
    _check_zarr_suffix(path)

    annotation = ds["annotation"].copy()
    annotation.attrs = {
        k: v
        for k, v in annotation.attrs.items()
        if k not in _NONSERIALIZABLE_ANNOTATION_ATTRS
    }
    to_save = ds.copy()
    to_save["annotation"] = annotation

    # Serialize the in-memory StructuresDict to a flat JSON list and plan the mesh bundle.
    # Done before attr sanitization below, which would otherwise drop the (non-JSON)
    # StructuresDict along with the other non-serializable attrs.
    to_copy: dict[str, Path] = {}
    if "structures" in to_save.attrs:
        blob, to_copy = _plan_mesh_bundle(
            structures_to_json(to_save.attrs["structures"])
        )
        to_save.attrs = {**to_save.attrs, "structures": blob}

    # physical_to_base rides in a single attr as either a numpy affine or a displacement
    # field. A DataArray cannot live in JSON attrs, so a displacement field is moved into a
    # data variable of the same name for storage (load_atlas lifts it back into attrs); a
    # numpy affine is left in attrs and JSON-sanitized like any other array below.
    physical_to_base = to_save.attrs.get("physical_to_base")
    if isinstance(physical_to_base, xr.DataArray):
        to_save = to_save.assign(physical_to_base=physical_to_base)
        to_save.attrs = {
            k: v for k, v in to_save.attrs.items() if k != "physical_to_base"
        }

    # Sanitize numpy-valued attrs that zarr cannot serialize: the top-level numpy
    # `physical_to_base` affine, and any per-variable `affines` dicts (e.g. physical_to_sform
    # inherited from the fUSI grid a resampled atlas was warped onto). load_atlas restores
    # them to arrays on load. cmap/norm are stripped above, so nothing is dropped here.
    for name in list(to_save.data_vars):
        var = to_save[name].copy()
        var.attrs = make_attrs_zarr_safe(var.attrs)
        to_save[name] = var
    to_save.attrs = make_attrs_zarr_safe(to_save.attrs)

    # A nonlinear (displacement-field) physical_to_base transform carries a decorative
    # string `component` coordinate that zarr v3 cannot serialize stably. Replace it with
    # integer indices, which serialize cleanly and keep `.fusi.spacing` defined on load;
    # the transform math uses the dimension order, not the component labels.
    if "physical_to_base" in to_save.data_vars and "component" in to_save.coords:
        to_save = to_save.assign_coords(component=np.arange(to_save.sizes["component"]))

    # Write the store first (to_zarr requires the path not to exist yet), then copy the
    # OBJ files into its meshes/ subdirectory.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=_ZARR_V3_CONSOLIDATED_METADATA_WARNING,
        )
        to_save.to_zarr(path, **kwargs)
    if to_copy:
        meshes_dir = path / _MESHES_SUBDIR
        meshes_dir.mkdir(parents=True, exist_ok=True)
        for basename, source in to_copy.items():
            shutil.copyfile(source, meshes_dir / basename)


def load_atlas(path: str | Path, **kwargs: Any) -> xr.Dataset:
    """Load an atlas Dataset from a Zarr store.

    The serialized structures are rebuilt into a
    [`StructuresDict`][brainglobe_atlasapi.structure_class.StructuresDict] in
    `attrs["structures"]`. The `cmap`/`norm` colormap objects dropped on save are rebuilt
    into `annotation.attrs` from `rgb_lookup`, and each ROI's `mesh_filename` is re-pointed
    at the meshes bundled under the store's `meshes/` subdirectory, so `get_mesh` works on
    the loaded atlas without the BrainGlobe cache.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the Zarr store; must end with the `.zarr` suffix.
    **kwargs
        Additional keyword arguments forwarded to
        [`xarray.open_zarr`][xarray.open_zarr].

    Returns
    -------
    xarray.Dataset
        The loaded atlas Dataset, with `attrs["structures"]` rebuilt into a
        `StructuresDict`, `attrs["physical_to_base"]` restored to a numpy affine or
        displacement-field DataArray, `cmap`/`norm` restored on `annotation.attrs`, and
        `mesh_filename` paths pointing at the bundled meshes.

    Raises
    ------
    ValueError
        If `path` does not end with the `.zarr` suffix.
    """
    path = check_path(path)
    _check_zarr_suffix(path)
    ds = xr.open_zarr(path, **kwargs)

    # Restore any per-variable `affines` matrices (e.g. physical_to_sform) JSON-encoded on
    # save back to numpy arrays, so they match the arrays a NIfTI-loaded volume carries.
    for name in ds.data_vars:
        restore_affines_in_attrs(ds[name].attrs)
    restore_affines_in_attrs(ds.attrs)

    # Restore the physical_to_base transform to its single attr: a displacement field was
    # stored as a data variable (lift it back into attrs); a numpy affine was JSON-encoded
    # as a nested list (convert it back to an array).
    if "physical_to_base" in ds.data_vars:
        field = ds["physical_to_base"]
        ds = ds.drop_vars("physical_to_base")
        # `component` was that field's dimension coordinate; drop it once it is orphaned so
        # the loaded atlas keeps only its spatial dims.
        if "component" in ds.coords:
            ds = ds.drop_vars("component")
        ds.attrs["physical_to_base"] = field
    elif "physical_to_base" in ds.attrs:
        ds.attrs["physical_to_base"] = np.asarray(
            ds.attrs["physical_to_base"], dtype=np.float64
        )

    restore_atlas_cmap_and_norm(ds["annotation"])

    if "structures" in ds.attrs:
        blob = _rebase_meshes(ds.attrs["structures"], path / _MESHES_SUBDIR)
        ds.attrs["structures"] = structures_from_json(blob)
    return ds

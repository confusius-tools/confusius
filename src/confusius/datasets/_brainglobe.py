"""Fetcher for BrainGlobe brain atlases."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import dask.array as da
import numpy as np
import xarray as xr

from confusius._dims import VOXEL_DIMS
from confusius._utils.atlas import build_atlas_cmap_and_norm
from confusius.atlas._structures import _build_rgb_lookup
from confusius.xarray import create_voxeldata

if TYPE_CHECKING:
    from brainglobe_atlasapi import BrainGlobeAtlas
    from brainglobe_atlasapi.atlas_name import AtlasName


def _load_lazy_ngff_array(
    atlas: BrainGlobeAtlas, location: str, name: str, pyramid_level: int
) -> da.Array:
    """Lazily load one BrainGlobe v3 zarr array as a Dask array.

    ``brainglobe_atlasapi.core.Atlas.template``/``.annotation``/``.hemispheres``
    resolve the on-disk zarr path, download it if missing, then call
    `.compute()` before returning — there is no public lazy-loading entry
    point (see [brainglobe/brainglobe-atlasapi#882](https://github.com/brainglobe/brainglobe-atlasapi/issues/882)).
    This reimplements that resolve/download logic but stops short of
    `.compute()`, so it depends on `BrainGlobeAtlas` v3 internals
    (`root_dir`, `metadata`, `fs`, the `_template_pyramid_level`/
    `_annotation_pyramid_level` attrs) that aren't part of BrainGlobe's public
    API and may change without notice.

    Parameters
    ----------
    atlas : brainglobe_atlasapi.bg_atlas.BrainGlobeAtlas
        An already-loaded
        [`BrainGlobeAtlas`][brainglobe_atlasapi.bg_atlas.BrainGlobeAtlas] instance.
    location : str
        Atlas-relative directory containing the zarr store, e.g.
        `atlas.metadata["annotation_set"]["location"][1:]`.
    name : str
        Zarr store directory name, e.g. `brainglobe_atlasapi.descriptors.V3_ANNOTATION_NAME`.
    pyramid_level : int
        Resolution pyramid level to load, e.g. `atlas._annotation_pyramid_level`.

    Returns
    -------
    dask.array.Array
        Lazy array for the requested resolution level, not yet computed.
    """
    import ngff_zarr as nz
    from brainglobe_atlasapi.descriptors import remote_url_s3
    from fsspec.callbacks import TqdmCallback

    path = atlas.root_dir / location / name
    multiscale = nz.from_ngff_zarr(path)
    dataset_path = multiscale.metadata.datasets[pyramid_level].path
    resolution_path = path / dataset_path

    if not (resolution_path / "c").exists():
        remote_path = remote_url_s3.format(f"{location}/{name}/{dataset_path}/")
        atlas.fs.get(
            remote_path, resolution_path, recursive=True, callback=TqdmCallback()
        )

    return multiscale.images[pyramid_level].data


def _build_dataset_from_brainglobe(atlas: BrainGlobeAtlas) -> xr.Dataset:
    """Build an atlas Dataset from a loaded BrainGlobe atlas.

    Parameters
    ----------
    atlas : brainglobe_atlasapi.bg_atlas.BrainGlobeAtlas
        An already-loaded
        [`BrainGlobeAtlas`][brainglobe_atlasapi.bg_atlas.BrainGlobeAtlas] instance.

    Returns
    -------
    xarray.Dataset
        Atlas Dataset with data variables `reference`, `annotation`, and `hemispheres`
        on a common voxel-to-world `(k, j, i)` grid, with world `z`/`y`/`x` coordinates
        in millimetres.
    """
    metadata = atlas.metadata
    resolution_mm = [r * 1e-3 for r in metadata["resolution"]]

    voxel_to_world = np.eye(4, dtype=np.float64)
    voxel_to_world[:-1, :-1] = np.diag(resolution_mm)

    rgb_lookup = _build_rgb_lookup(atlas.structures)
    cmap, norm = build_atlas_cmap_and_norm(rgb_lookup)
    roi_labels = {
        int(sid): str(info["name"] + f" ({info['acronym']})")
        for sid, info in atlas.structures.items()
    }

    def _build(data: np.ndarray | da.Array, attrs: dict[str, object]) -> xr.DataArray:
        return create_voxeldata(
            data,
            dims=VOXEL_DIMS,
            voxel_to_world=voxel_to_world,
            attrs=attrs,
        )

    from brainglobe_atlasapi.descriptors import (
        V3_ANNOTATION_NAME,
        V3_HEMISPHERES_NAME,
        V3_TEMPLATE_NAME,
    )

    template_location = metadata["annotation_set"]["template"]["location"][1:]
    template = _load_lazy_ngff_array(
        atlas, template_location, V3_TEMPLATE_NAME, atlas._template_pyramid_level
    )
    reference = _build(template.astype(np.float32), {"cmap": "gray"})

    annotation_location = metadata["annotation_set"]["location"][1:]
    annotation_data = _load_lazy_ngff_array(
        atlas,
        annotation_location,
        V3_ANNOTATION_NAME,
        atlas._annotation_pyramid_level,
    )
    annotation = _build(
        annotation_data.view(np.int32),  # type: ignore
        {
            "rgb_lookup": rgb_lookup,
            "roi_labels": roi_labels,
            "cmap": cmap,
            "norm": norm,
        },
    )

    world_to_base = np.eye(4)

    if metadata["symmetric"]:
        # Synthesized in-memory from `shape` by BrainGlobe itself, not read from disk —
        # no laziness to preserve here.
        hemispheres_data = atlas.hemispheres
    else:
        hemispheres_data = _load_lazy_ngff_array(
            atlas,
            annotation_location,
            V3_HEMISPHERES_NAME,
            atlas._annotation_pyramid_level,
        )
    hemispheres = _build(
        hemispheres_data.view(np.int8),  # type: ignore
        {
            "left": int(getattr(atlas, "left_hemisphere_value", 1)),
            "right": int(getattr(atlas, "right_hemisphere_value", 2)),
        },
    )

    return xr.Dataset(
        {
            "reference": reference,
            "annotation": annotation,
            "hemispheres": hemispheres,
        },
        attrs={
            "name": metadata["name"],
            "citation": metadata["citation"],
            "species": metadata["species"],
            "orientation": metadata["orientation"],
            "structures": atlas.structures,
            "world_to_base": world_to_base,
        },
    )


def fetch_brainglobe_atlas(
    atlas_name: AtlasName,
    *,
    data_dir: str | Path | None = None,
    check_latest: bool = False,
) -> xr.Dataset:
    """Fetch a BrainGlobe brain atlas by name and return it as an atlas Dataset.

    Downloads the named atlas via the
    [BrainGlobe Atlas API](https://brainglobe.info/documentation/brainglobe-atlasapi/index.html)
    on first call, caching it in BrainGlobe's own atlas cache (shared with other
    BrainGlobe tools), then builds a self-describing atlas
    [`xarray.Dataset`][xarray.Dataset].

    Parameters
    ----------
    atlas_name : brainglobe_atlasapi.atlas_name.AtlasName
        BrainGlobe atlas name, e.g. `"allen_mouse_25um"`. See the
        [BrainGlobe atlas list](https://brainglobe.info/documentation/brainglobe-atlasapi/usage/atlas-details.html).
    data_dir : str or pathlib.Path, optional
        Directory in which BrainGlobe caches the atlas. If not provided, BrainGlobe uses
        its own default cache (`~/.brainglobe`), shared with other BrainGlobe tools.
    check_latest : bool, default: False
        Whether to check online for a newer atlas version. Left off by default so cached
        atlases load without a network round-trip.

    Returns
    -------
    xarray.Dataset
        Atlas Dataset with data variables `reference`, `annotation`, and `hemispheres`
        on a common voxel-to-world `(k, j, i)` grid with world `z`/`y`/`x` coordinates
        in millimetres, and the `.atlas` accessor for structure queries, masks, and
        meshes.

    Examples
    --------
    >>> atlas = fetch_brainglobe_atlas("allen_mouse_100um")
    >>> masks = atlas.atlas.get_masks("VISp")
    """
    from brainglobe_atlasapi import BrainGlobeAtlas

    bg_atlas = BrainGlobeAtlas(
        # BrainGlobeAtlas types atlas_name as a Literal of every known atlas name; we
        # accept any str so new atlases work without a stub bump.
        atlas_name,
        brainglobe_dir=data_dir,
        check_latest=check_latest,
    )
    return _build_dataset_from_brainglobe(bg_atlas)

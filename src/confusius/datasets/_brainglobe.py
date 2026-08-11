"""Fetcher for BrainGlobe brain atlases."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from confusius._dims import SPATIAL_DIMS, VOXEL_DIMS
from confusius._utils.atlas import build_atlas_cmap_and_norm
from confusius._utils.geometry import add_physical_coords_from_voxel_affine
from confusius.atlas._structures import _build_rgb_lookup

if TYPE_CHECKING:
    from brainglobe_atlasapi import BrainGlobeAtlas
    from brainglobe_atlasapi.atlas_name import AtlasName


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
        on a common voxel-affine `(k, j, i)` grid, with physical `z`/`y`/`x` coordinates
        in millimetres.
    """
    metadata = atlas.metadata
    resolution_mm = [r * 1e-3 for r in metadata["resolution"]]
    shape = metadata["shape"]

    voxel_coords = {
        dim: xr.Variable(dim, np.arange(shape[i], dtype=np.float64))
        for i, dim in enumerate(VOXEL_DIMS)
    }
    voxel_to_physical = np.eye(4, dtype=np.float64)
    voxel_to_physical[:-1, :-1] = np.diag(resolution_mm)
    physical_coord_attrs = {name: {"units": "mm"} for name in SPATIAL_DIMS}

    rgb_lookup = _build_rgb_lookup(atlas.structures)
    cmap, norm = build_atlas_cmap_and_norm(rgb_lookup)
    roi_labels = {
        int(sid): str(info["name"] + f" ({info['acronym']})")
        for sid, info in atlas.structures.items()
    }

    def _with_physical_coords(data: xr.DataArray) -> xr.DataArray:
        return add_physical_coords_from_voxel_affine(
            data,
            voxel_to_physical,
            voxel_dims=VOXEL_DIMS,
            physical_coord_names=SPATIAL_DIMS,
            physical_coord_attrs=physical_coord_attrs,
        )

    reference = _with_physical_coords(
        xr.DataArray(
            atlas.reference.astype(np.float32),
            dims=VOXEL_DIMS,
            coords=voxel_coords,
            attrs={"cmap": "gray"},
        )
    )

    annotation = _with_physical_coords(
        xr.DataArray(
            atlas.annotation.astype(np.int32),
            dims=VOXEL_DIMS,
            coords=voxel_coords,
            attrs={
                "rgb_lookup": rgb_lookup,
                "roi_labels": roi_labels,
                "cmap": cmap,
                "norm": norm,
            },
        )
    )

    physical_to_base = np.eye(4)

    hemispheres = _with_physical_coords(
        xr.DataArray(
            atlas.hemispheres.astype(np.int8),
            dims=VOXEL_DIMS,
            coords=voxel_coords,
            attrs={
                "left": int(getattr(atlas, "left_hemisphere_value", 1)),
                "right": int(getattr(atlas, "right_hemisphere_value", 2)),
            },
        )
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
            "physical_to_base": physical_to_base,
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
        on a common voxel-affine `(k, j, i)` grid with physical `z`/`y`/`x` coordinates
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

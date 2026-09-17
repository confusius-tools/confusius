"""Fetcher for BrainGlobe brain atlases."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from confusius._dims import VOXEL_DIMS
from confusius._utils.atlas import build_atlas_cmap_and_norm
from confusius.atlas._structures import _build_rgb_lookup
from confusius.xarray import create_voxeldata

if TYPE_CHECKING:
    from brainglobe_atlasapi import BrainGlobeAtlas
    from brainglobe_atlasapi.atlas_name import AtlasName


def _fetch_all_meshes(atlas: BrainGlobeAtlas) -> None:
    """Download every region mesh missing from the BrainGlobe cache in one batched call.

    BrainGlobe fetches meshes lazily, one S3 round trip per region on first access.
    Fetching them all up front in a single concurrent `s3fs` call is an order of
    magnitude faster and leaves the meshes available offline.

    Parameters
    ----------
    atlas : brainglobe_atlasapi.bg_atlas.BrainGlobeAtlas
        Loaded BrainGlobe atlas whose structures carry their `mesh_filename`.

    Warns
    -----
    UserWarning
        If the BrainGlobe S3 bucket is unreachable, or if some regions have no mesh on
        the remote store. Either way the affected meshes are left to BrainGlobe's lazy
        per-region download on first use.
    """
    missing = {
        str(structure["id"]): Path(structure["mesh_filename"])
        for structure in atlas.structures.values()
        if structure.get("mesh_filename") is not None
    }
    missing = {rid: path for rid, path in missing.items() if not path.exists()}
    if not missing:
        return

    import s3fs
    from brainglobe_atlasapi.descriptors import V3_MESHES_DIRECTORY, remote_url_s3
    from brainglobe_atlasapi.utils import check_s3_status
    from fsspec.callbacks import TqdmCallback

    if not check_s3_status(raise_error=False):
        warnings.warn(
            "BrainGlobe's S3 bucket is unreachable; region meshes will be downloaded "
            "lazily on first use instead.",
            stacklevel=2,
        )
        return

    location = atlas.metadata["annotation_set"]["location"].strip("/")
    remote_dir = remote_url_s3.format(f"{location}/{V3_MESHES_DIRECTORY}")
    fs = s3fs.S3FileSystem(anon=True)
    # One LIST call replaces BrainGlobe's per-mesh `fs.exists` and names the exact keys
    # absent remotely: a batched `fs.get` only raises a bare "The specified key does
    # not exist." with no path in it.
    remote_names = {Path(key).name for key in fs.ls(remote_dir)}
    absent = sorted(missing.keys() - remote_names, key=int)
    if absent:
        warnings.warn(
            f"BrainGlobe provides no mesh for region id(s) {absent} of atlas "
            f"'{atlas.atlas_name}'; requesting their mesh will fail.",
            stacklevel=2,
        )
        missing = {rid: path for rid, path in missing.items() if rid in remote_names}
        if not missing:
            return

    print(f"Downloading {atlas.atlas_name} atlas meshes:")
    try:
        fs.get(
            [f"{remote_dir}/{rid}" for rid in missing],
            [str(path) for path in missing.values()],
            callback=TqdmCallback(),
        )
    except BaseException:
        # Mirror BrainGlobe's `Structure._download_mesh`: never leave a partial mesh in
        # the cache, since BrainGlobe treats any existing file as a valid mesh.
        for path in missing.values():
            path.unlink(missing_ok=True)
        raise


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

    def _build(data: np.ndarray, attrs: dict[str, object]) -> xr.DataArray:
        return create_voxeldata(
            data,
            dims=VOXEL_DIMS,
            voxel_to_world=voxel_to_world,
            attrs=attrs,
        )

    reference = _build(atlas.template.astype(np.float32), {"cmap": "gray"})

    annotation = _build(
        atlas.annotation.view(np.int32),
        {
            "rgb_lookup": rgb_lookup,
            "roi_labels": roi_labels,
            "cmap": cmap,
            "norm": norm,
        },
    )

    world_to_base = np.eye(4)

    hemispheres = _build(
        atlas.hemispheres.view(np.int8),
        {
            "left": int(getattr(atlas, "left_hemisphere_value", 1)),
            "right": int(getattr(atlas, "right_hemisphere_value", 2)),
        },
    )

    _fetch_all_meshes(atlas)

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

    The first fetch of an atlas also downloads every region surface mesh, in one batched
    call rather than BrainGlobe's one-at-a-time lazy download, so
    [`get_meshes`][confusius.atlas.AtlasAccessor.get_meshes] works offline afterwards.
    If the BrainGlobe bucket is unreachable, meshes fall back to that lazy download on
    first use.

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

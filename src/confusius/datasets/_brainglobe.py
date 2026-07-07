"""Fetcher for BrainGlobe brain atlases."""

from __future__ import annotations

from pathlib import Path

import xarray as xr

from confusius.atlas import atlas_from_brainglobe


def fetch_brainglobe_atlas(
    atlas_name: str,
    *,
    data_dir: str | Path | None = None,
    check_latest: bool = False,
) -> xr.Dataset:
    """Fetch a BrainGlobe brain atlas by name and return it as an atlas Dataset.

    Downloads the named atlas via the
    [BrainGlobe Atlas API](https://brainglobe.info/documentation/brainglobe-atlasapi/index.html)
    on first call, caching it in BrainGlobe's own atlas cache (shared with other
    BrainGlobe tools), then builds a self-describing atlas
    [`xarray.Dataset`][xarray.Dataset] via
    [`atlas_from_brainglobe`][confusius.atlas.atlas_from_brainglobe].

    Parameters
    ----------
    atlas_name : str
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
        on a common `(z, y, x)` grid with physical coordinates in millimetres, and the
        `.atlas` accessor for structure queries, masks, and meshes.

    Examples
    --------
    >>> atlas = fetch_brainglobe_atlas("allen_mouse_100um")
    >>> masks = atlas.atlas.get_masks("VISp")
    """
    from brainglobe_atlasapi import BrainGlobeAtlas

    bg_atlas = BrainGlobeAtlas(
        atlas_name,
        brainglobe_dir=data_dir,
        check_latest=check_latest,
    )
    return atlas_from_brainglobe(bg_atlas)

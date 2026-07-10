"""Brain atlas integration via BrainGlobe.

An atlas is an [`xarray.Dataset`][xarray.Dataset] with a registered `.atlas` accessor.
Build one from BrainGlobe with
[`atlas_from_brainglobe`][confusius.atlas.atlas_from_brainglobe], save/load it with
[`atlas_to_zarr`][confusius.atlas.atlas_to_zarr] /
[`atlas_from_zarr`][confusius.atlas.atlas_from_zarr], and operate on it through
`ds.atlas.*` (see [`AtlasAccessor`][confusius.atlas.AtlasAccessor]). The core operations
are also exposed as standalone functions that take the Dataset as their first argument —
[`get_mesh`][confusius.atlas.get_mesh], [`get_masks`][confusius.atlas.get_masks], and
[`search`][confusius.atlas.search] — each of which validates its input as an atlas first.
"""

# Importing the accessor module registers the `.atlas` namespace as a side effect.
from confusius.atlas._accessor import AtlasAccessor, get_masks, get_mesh, search
from confusius.atlas._build import atlas_from_brainglobe
from confusius.atlas._io import atlas_from_zarr, atlas_to_zarr

__all__ = [
    "AtlasAccessor",
    "atlas_from_brainglobe",
    "atlas_from_zarr",
    "atlas_to_zarr",
    "get_masks",
    "get_mesh",
    "search",
]

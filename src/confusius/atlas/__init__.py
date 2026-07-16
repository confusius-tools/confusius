"""Brain atlases and region-aware utilities.

An atlas is an [`xarray.Dataset`][xarray.Dataset] with a registered `.atlas` accessor.
Fetch one from BrainGlobe with
[`fetch_brainglobe_atlas`][confusius.datasets.fetch_brainglobe_atlas], save/load it
with [`save_atlas`][confusius.io.save_atlas] /
[`load_atlas`][confusius.io.load_atlas], and operate on it through `ds.atlas.*` (see
[`AtlasAccessor`][confusius.atlas.AtlasAccessor]). The core operations are also exposed
as standalone functions that take the Dataset as their first argument —
[`get_atlas_mesh`][confusius.atlas.get_atlas_mesh],
[`get_atlas_masks`][confusius.atlas.get_atlas_masks], and
[`search_atlas`][confusius.atlas.search_atlas] — each of which validates its input as
an atlas first.
"""

# Importing the accessor module registers the `.atlas` namespace as a side effect.
from confusius.atlas._accessor import (
    AtlasAccessor,
    get_atlas_masks,
    get_atlas_mesh,
    search_atlas,
)

__all__ = [
    "AtlasAccessor",
    "get_atlas_masks",
    "get_atlas_mesh",
    "search_atlas",
]

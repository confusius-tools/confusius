"""Atlas-rendering helpers shared by atlas building and plotting code."""

import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap


def build_atlas_cmap_and_norm(
    rgb_lookup: dict[int, list[int]],
) -> tuple[ListedColormap, BoundaryNorm]:
    """Build a `ListedColormap` / `BoundaryNorm` pair from an RGB lookup.

    Using this colormap and norm, each region is always rendered with the same color
    regardless of which subset of IDs is currently displayed.

    Parameters
    ----------
    rgb_lookup : dict[int, list[int]]
        Mapping from structure id to `[r, g, b]` (0–255).

    Returns
    -------
    cmap : matplotlib.colors.ListedColormap
        The colormap to use for atlas rendering.
    norm : matplotlib.colors.BoundaryNorm
        The norm to use for atlas rendering.
    """
    # JSON (zarr) round-trips convert int keys to strings; normalize here.
    rgb_lookup = {int(k): v for k, v in rgb_lookup.items()}
    ordered_ids = sorted(rgb_lookup.keys())
    rgba = [list(np.array(rgb_lookup[sid]) / 255) + [1.0] for sid in ordered_ids]

    # Add a sentinel boundary one past the last ID so that the last structure falls in a
    # proper half-open bin [last_id, last_id+1) rather than being clipped. BoundaryNorm
    # maps values below the first boundary to the "under" color, so label 0 is rendered
    # transparently instead of stealing the first structure's color.
    boundaries = ordered_ids + [ordered_ids[-1] + 1]

    cmap = ListedColormap(rgba, N=len(ordered_ids))
    cmap.set_under((0.0, 0.0, 0.0, 0.0))
    norm = BoundaryNorm(boundaries, ncolors=len(ordered_ids), clip=False)
    return cmap, norm

"""Helpers for BrainGlobe structure trees and colormap construction."""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap

if TYPE_CHECKING:
    from brainglobe_atlasapi.structure_class import StructuresDict


def _build_lookup_df(structures: "StructuresDict") -> pd.DataFrame:
    """Build a lookup DataFrame from a BrainGlobe StructuresDict.

    Parameters
    ----------
    structures : brainglobe_atlasapi.structure_class.StructuresDict
        BrainGlobe structure dictionary.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns `id`, `acronym`, `name`, `rgb_triplet`.
    """
    rows = [
        {
            "id": sid,
            "acronym": info["acronym"],
            "name": info["name"],
            "rgb_triplet": info["rgb_triplet"],
        }
        for sid, info in structures.items()
    ]
    return pd.DataFrame(rows).set_index("id")


def _build_rgb_lookup(structures: "StructuresDict") -> dict[int, list[int]]:
    """Build an `{id: [r, g, b]}` dict from a BrainGlobe StructuresDict.

    Parameters
    ----------
    structures : brainglobe_atlasapi.structure_class.StructuresDict
        BrainGlobe structure dictionary.

    Returns
    -------
    dict[int, list[int]]
        Mapping from structure id to `[r, g, b]` in the 0–255 range.
    """
    return {sid: info["rgb_triplet"] for sid, info in structures.items()}


def _build_atlas_cmap_and_norm(
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


def _resolve_region_id(structures: "StructuresDict", region: int | str) -> int:
    """Return the integer id for a region given as an id or acronym.

    Parameters
    ----------
    structures : brainglobe_atlasapi.structure_class.StructuresDict
        BrainGlobe structure dictionary.
    region : int or str
        Integer structure id or acronym string.

    Returns
    -------
    int
        The resolved integer structure id.

    Raises
    ------
    KeyError
        If `region` is a string or integer not found in the structures dictionary.
    """
    if isinstance(region, str):
        acronym_map: dict[str, int] = {
            info["acronym"]: sid for sid, info in structures.items()
        }
        if region not in acronym_map:
            raise KeyError(
                f"Acronym '{region}' not found in atlas. "
                f"Use atlas.search() to find the correct acronym."
            )
        return acronym_map[region]

    if region not in structures:
        raise KeyError(
            f"Structure id {region} not found in atlas. "
            f"Use atlas.search() to browse available structures."
        )
    return int(region)


def _get_descendant_ids(structures: "StructuresDict", region_id: int) -> list[int]:
    """Return all descendant ids (including `region_id` itself).

    Parameters
    ----------
    structures : brainglobe_atlasapi.structure_class.StructuresDict
        BrainGlobe structure dictionary.
    region_id : int
        Root of the subtree.

    Returns
    -------
    list[int]
        All ids in the subtree rooted at `region_id`, inclusive.
    """
    subtree = structures.tree.subtree(region_id)
    return list(subtree.nodes.keys())


def _load_obj(
    path: Path,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int32]]:
    """Parse an OBJ file and return vertices and triangular faces.

    Only triangular faces are supported; polygons with more than three vertices are
    skipped silently. Vertex/texture/normal notation (`v/t/n`) is handled by taking only
    the vertex index.

    Parameters
    ----------
    path : pathlib.Path
        Path to the `.obj` file.

    Returns
    -------
    vertices : numpy.ndarray, shape (N, 3)
        Vertex coordinates as float64.
    faces : numpy.ndarray, shape (M, 3)
        Zero-indexed triangle face indices as int32.
    """
    vertices = []
    faces = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                parts = line.split()
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("f "):
                parts = line.split()[1:]
                if len(parts) != 3:
                    # Skip non-triangular faces.
                    continue
                # OBJ uses 1-indexed vertices; split on "/" to handle v/t/n notation.
                idx = [int(p.split("/")[0]) - 1 for p in parts]
                faces.append(idx)
    return np.array(vertices, dtype=np.float64), np.array(faces, dtype=np.int32)

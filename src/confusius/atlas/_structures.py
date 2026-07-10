"""Helpers for BrainGlobe structure trees and colormap construction."""

from typing import TYPE_CHECKING

import pandas as pd

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
        if region not in structures.acronym_to_id_map:
            raise KeyError(
                f"Acronym '{region}' not found in atlas. "
                f"Use atlas.search('{region}') to find the correct acronym."
            )
        return int(structures.acronym_to_id_map[region])

    if region not in structures:
        raise KeyError(
            f"Structure id {region} not found in atlas. "
            f"Use atlas.search({region}) to browse available structures."
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
    subtree = structures.tree.subtree(region_id)  # type: ignore
    return list(subtree.nodes.keys())

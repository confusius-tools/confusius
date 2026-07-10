"""Helpers for BrainGlobe structure trees and colormap construction."""

import json
from typing import TYPE_CHECKING

import pandas as pd

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


def structures_to_json(structures: "StructuresDict") -> str:
    """Serialize a StructuresDict to a flat BrainGlobe `structures.json` string.

    The `treelib` hierarchy is never serialized directly; it is a derived index that
    [`structures_from_json`][confusius.atlas._structures.structures_from_json] rebuilds
    from the flat list. `mesh_filename` is stored verbatim (the complete path), so a
    freshly fetched atlas points straight at the OBJ files in the BrainGlobe cache. When
    the atlas is saved with [`atlas_to_zarr`][confusius.atlas.atlas_to_zarr], the meshes
    are bundled into the store and the paths are re-pointed there on load.

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
        [`structures_to_json`][confusius.atlas._structures.structures_to_json].

    Returns
    -------
    brainglobe_atlasapi.structure_class.StructuresDict
        Reconstructed structure dictionary with a freshly built hierarchy tree.
    """
    from brainglobe_atlasapi.structure_class import StructuresDict

    return StructuresDict(json.loads(blob))


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

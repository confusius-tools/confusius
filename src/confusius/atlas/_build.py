"""Build a self-describing atlas `xarray.Dataset` from a BrainGlobe atlas."""

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from confusius._utils.atlas import build_atlas_cmap_and_norm
from confusius.atlas._structures import _build_rgb_lookup, structures_to_json

if TYPE_CHECKING:
    from brainglobe_atlasapi import BrainGlobeAtlas


def atlas_from_brainglobe(atlas: "BrainGlobeAtlas") -> xr.Dataset:
    """Build an atlas Dataset from a loaded BrainGlobe atlas.

    The returned Dataset is self-describing and serializable: the structure hierarchy
    rides in `attrs["structures"]` as a flat JSON list (rebuilt into a tree on demand by
    the `.atlas` accessor), so the whole atlas round-trips through
    [`atlas_from_zarr`][confusius.atlas.atlas_from_zarr].

    To fetch and build an atlas by name in one step, use
    [`fetch_brainglobe_atlas`][confusius.datasets.fetch_brainglobe_atlas].

    Parameters
    ----------
    atlas : brainglobe_atlasapi.bg_atlas.BrainGlobeAtlas
        An already-loaded
        [`BrainGlobeAtlas`][brainglobe_atlasapi.bg_atlas.BrainGlobeAtlas] instance.

    Returns
    -------
    xarray.Dataset
        Atlas Dataset with data variables `reference`, `annotation`, and `hemispheres`
        on a common `(z, y, x)` grid with physical coordinates in millimetres.

    Examples
    --------
    >>> from brainglobe_atlasapi import BrainGlobeAtlas
    >>> atlas = atlas_from_brainglobe(BrainGlobeAtlas("allen_mouse_25um"))
    """
    metadata = atlas.metadata
    resolution_mm = [r * 1e-3 for r in metadata["resolution"]]
    shape = metadata["shape"]

    coords = {
        dim: (
            np.arange(shape[i]) * resolution_mm[i],
            {"voxdim": resolution_mm[i], "units": "mm"},
        )
        for i, dim in enumerate(["z", "y", "x"])
    }

    rgb_lookup = _build_rgb_lookup(atlas.structures)
    cmap, norm = build_atlas_cmap_and_norm(rgb_lookup)
    roi_labels = {
        int(sid): str(info["name"] + f" ({info['acronym']})")
        for sid, info in atlas.structures.items()
    }

    reference = xr.DataArray(
        atlas.reference.astype(np.float32),
        dims=["z", "y", "x"],
        coords={d: xr.Variable(d, v, attrs=a) for d, (v, a) in coords.items()},
        attrs={"cmap": "gray"},
    )

    annotation = xr.DataArray(
        atlas.annotation.astype(np.int32),
        dims=["z", "y", "x"],
        coords={d: xr.Variable(d, v, attrs=a) for d, (v, a) in coords.items()},
        # cmap and norm are non-serializable, held in-memory only; rgb_lookup is the
        # serializable source of truth they are rebuilt from on load.
        attrs={
            "rgb_lookup": rgb_lookup,
            "roi_labels": roi_labels,
            "cmap": cmap,
            "norm": norm,
        },
    )

    # The mesh vertex transform is a pure base→current physical (mm) pull transform,
    # identity for a fresh atlas; OBJ vertices are converted microns→millimetres inside
    # get_mesh, not folded into this transform, so it composes uniformly with the
    # nonlinear (displacement-field) transforms that resample_like can produce.
    mesh_vertex_transform = np.eye(4)
    # RL midline in millimetres (base atlas space). For asr orientation: shape[2] is the
    # RL axis length (voxels), resolution[2] the voxel size in microns; the midline sits
    # at the centre of the volume.
    rl_midline = metadata["shape"][2] / 2 * metadata["resolution"][2] * 1e-3

    # hemispheres is a per-voxel left/right partition (1 = left, 2 = right). It is a data
    # variable, not a coordinate: as a coordinate it would ride along on `reference` and
    # `annotation` and be silently linear-interpolated (to fractional, meaningless labels)
    # by any regridding op — e.g. resampling into a non-orthogonal frame. As a data
    # variable it is on equal footing with `annotation` and only changes when explicitly
    # resampled with nearest-neighbour, which preserves the integer labels.
    hemispheres = xr.DataArray(
        atlas.hemispheres.astype(np.int8),
        dims=["z", "y", "x"],
        coords={d: xr.Variable(d, v, attrs=a) for d, (v, a) in coords.items()},
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
            "structures": structures_to_json(atlas.structures),
            "mesh_vertex_transform": mesh_vertex_transform.tolist(),
            "rl_midline": float(rl_midline),
        },
    )

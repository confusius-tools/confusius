---
icon: lucide/brain
---

# Atlases

!!! info "Coming soon"
    This page is currently under construction. The `atlas` module provides tools for
    loading and working with standard brain atlases via the
    [BrainGlobe Atlas API](https://brainglobe.info/documentation/brainglobe-atlasapi/index.html).

    An atlas is an [`xarray.Dataset`][xarray.Dataset] with `reference`, `annotation`, and
    per-voxel `hemispheres` data variables, and a registered `.atlas` accessor (see
    [`AtlasAccessor`][confusius.atlas.AtlasAccessor]) for all operations.

    **Loading and saving:**

    - [`atlas_from_brainglobe`][confusius.atlas.atlas_from_brainglobe]: Load any BrainGlobe
      atlas by name or from an existing instance, as a self-describing Dataset with
      physical coordinates in millimetres.
    - [`save_atlas`][confusius.io.save_atlas] /
      [`load_atlas`][confusius.io.load_atlas]: Save and reload the whole
      atlas (including its structure hierarchy) to and from a Zarr store.

    **Structure lookup:**

    - [`ds.atlas.lookup`][confusius.atlas.AtlasAccessor.lookup]: DataFrame of all
      structures with acronym, name, and RGB colour.
    - [`ds.atlas.search`][confusius.atlas.AtlasAccessor.search]: Search structures by
      substring or regex across acronym and name fields.
    - [`ds.atlas.ancestors`][confusius.atlas.AtlasAccessor.ancestors]: Return the ancestor
      nodes of a region, ordered from root down.

    **Masks and meshes:**

    - [`ds.atlas.get_masks`][confusius.atlas.AtlasAccessor.get_masks]: Build integer region
      masks stacked along a `mask` dimension, with optional per-region hemisphere filtering
      (`"left"`, `"right"`, or `"both"`). Descendant regions are included automatically.
    - [`ds.atlas.get_mesh`][confusius.atlas.AtlasAccessor.get_mesh]: Load the OBJ surface
      mesh for a region, clipped to a hemisphere if requested, in the atlas physical space
      (mm).

    **Registration:**

    - [`ds.atlas.resample_like`][confusius.atlas.AtlasAccessor.resample_like]: Resample the
      atlas onto the grid of a fUSI volume using a transform returned by
      [`register_volume`][confusius.registration.register_volume] — an affine, a B-spline,
      or a displacement field. Region meshes are warped through the same transform.
    - [`ds.atlas.resample`][confusius.atlas.AtlasAccessor.resample]: Resample onto an
      explicit output grid (shape, spacing, origin, dims).

    Please refer to the [API Reference](../api/atlas.md) for
    more information.

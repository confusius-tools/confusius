---
icon: lucide/brain
---

# Atlases

!!! info "Coming soon"
    This page is currently under construction. The `atlas` module provides tools for
    loading and working with standard brain atlases via the
    [BrainGlobe Atlas API](https://brainglobe.info/documentation/brainglobe-atlasapi/index.html):

    **Loading:**

    - [`Atlas.from_brainglobe`][confusius.atlas.Atlas.from_brainglobe]: Load any BrainGlobe
      atlas by name or from an existing instance. Exposes `reference`, `annotation`, and
      `hemispheres` as Xarray DataArrays with physical coordinates in millimetres.

    **Structure lookup:**

    - [`Atlas.lookup`][confusius.atlas.Atlas.lookup]: DataFrame of all structures with
      acronym, name, and RGB colour.
    - [`Atlas.search`][confusius.atlas.Atlas.search]: Search structures by substring or
      regex across acronym and name fields.
    - [`Atlas.ancestors`][confusius.atlas.Atlas.ancestors]: Return the ancestor nodes of a
      region, ordered from root down.

    **Masks and meshes:**

    - [`Atlas.get_masks`][confusius.atlas.Atlas.get_masks]: Build integer region masks
      stacked along a `masks` dimension, with optional per-region hemisphere filtering
      (`"left"`, `"right"`, or `"both"`). Descendant regions are included automatically.
    - [`Atlas.get_mesh`][confusius.atlas.Atlas.get_mesh]: Load the OBJ surface mesh for a
      region, clipped to a hemisphere if requested, in the atlas physical space (mm).

    **Registration:**

    - [`Atlas.resample_like`][confusius.atlas.Atlas.resample_like]: Resample the atlas onto
      the grid of a fUSI volume using a pull affine returned by
      [`register_volume`][confusius.registration.register_volume].

    Please refer to the [API Reference](../api/atlas.md) and [Roadmap](../roadmap.md) for
    more information.

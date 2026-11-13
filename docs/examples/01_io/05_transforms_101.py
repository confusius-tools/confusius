# %% [markdown]
# # Transforms 101
#
# A VoxelData array stores values on a voxel grid positioned in world space by a
# [voxel-to-world affine](../../../user-guide/voxeldata.md#world-space). The
# [`.fusi`][confusius.xarray.FUSIAccessor] accessor lets you translate, rotate, zoom,
# and flip that grid by updating its affine. **The data values are not resampled**:
# resampling occurs when plotting (e.g.,
# [`plot_volume`][confusius.plotting.plot_volume]) or when requested explicitly
# (e.g., [`resample_like`][confusius.registration.resample_like]).
#
# In this example, we'll transform one `z` slice of the mouse fUSI template from Pepe,
# Mariani et al. (2026)[^collection], keeping the full template as a fixed background.
#
# [^collection]: See [The ConfUSIus Dataset Collection](dataset_collection.md)
#     for available studies, formats, licenses, and reproducible subset downloads.

# %%
from collections.abc import Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import confusius as cf

# Adapt background color to the current Matplotlib style.
bg_color = mpl.colors.to_hex(mpl.rcParams["figure.facecolor"])

# Keep notebook output compact for large DataArray displays.
xr.set_options(display_expand_data=False, display_expand_attrs=False)

template = cf.datasets.fetch_template_pepe_mariani_2026().fusi.scale.power()

# World coordinates are lazily derived from the voxel-to-world affine. Read the z value
# of the first voxel of each plane in a 0.6 mm band.
slice_coords = template.sel(z=slice(5.8, 6.4)).z.values[:, 0, 0]

# Pass a list so the plane keeps its k dimension and stays a VoxelData array.
plane = template.sel(z=[slice_coords[3]], method="nearest")
plane

# %% [markdown]
# Before any transform, the selected plane aligns with the fourth `z` panel. In the
# composite plots, the template appears red, while the plane appears cyan. Where they
# overlap, the color is white.

# %%
# Options shared by the six-panel composites. The figure is wide, so use a large font.
# Normalize each panel's slices independently so the untouched plane maps to the same
# intensities as the template slice and their overlap is white.
panel_kwargs = {
    "slice_coords": slice_coords,
    "bg_color": bg_color,
    "fontsize": 20,
    "normalize_strategy": "per_slice",
}

_ = template.fusi.plot.composite(plane, **panel_kwargs)

# %% [markdown]
# ## Rotate
#
# [`.fusi.rotate()`][confusius.xarray.FUSIAccessor.rotate] takes an angle in degrees
# and a world-axis name. By default, it rotates about the **geometry center**: the
# midpoint of the array's bounding box, available as
# [`.fusi.affine.geometry_center`][confusius.xarray.FUSIAffineAccessor.geometry_center].
# Here, a 30° rotation about `z` turns the plane in place. The yellow dot marks the
# fixed point.

# %% tags=["thumbnail"]
rotated_about_center = plane.fusi.rotate(30, "z")

center = plane.fusi.affine.geometry_center
plotter = template.fusi.plot.composite(rotated_about_center, **panel_kwargs)
_ = plotter.axes.ravel()[3].plot(
    center["x"],
    center["y"],
    "o",
    color="yellow",
    markersize=15,
)

# %% [markdown]
# The composite plots only show the transformed plane sliced along the red template's
# `z` axis. When transforms change the plane's `z` position, each template panel will
# only show narrow intersections of the transformed plane. To help visualize the full 3D
# geometry, we can plot the original and transformed planes in 3D. The 3D views will
# show the original place in red, the transformed plane in cyan, and the transformation
# center in yellow. Rotation axes are also drawn as yellow arrows.

# As in the composite plots, `+x` points right, `+y` down, and `+z` toward the viewer.
# Positive rotations follow the right-hand rule in `(z, y, x)` world coordinates: with
# your right thumb along `+z`, your fingers curl from `+y` toward `+x`.


# %% tags=["collapse: Code for `plot_plane_in_3d` helper"]
def plot_plane_in_3d(before, after, center=None, axis=None, thickness=5):
    """Show a plane before (red) and after (cyan) a transform."""
    box = template.fusi.affine.bounding_box
    template_min = np.array([box[dim][0] for dim in "zyx"])
    template_max = np.array([box[dim][1] for dim in "zyx"])

    fig = plt.figure(figsize=(7, 4), layout="constrained", facecolor=bg_color)
    ax = fig.add_subplot(projection="3d", facecolor=bg_color)
    ax.computed_zorder = False  # Keep the pivot and arrows in front of the slabs.

    # Draw voxel-edge corners, including on cropped arrays. Thicken thin planes
    # for visibility unless a smaller thickness is requested.
    k = before.k.item()
    j_min, j_max = before.j.values[[0, -1]]
    i_min, i_max = before.i.values[[0, -1]]
    corners = np.array(
        [
            [kk, jj, ii, 1]
            for kk in (k - thickness / 2, k + thickness / 2)
            for jj in (j_min - 0.5, j_max + 0.5)
            for ii in (i_min - 0.5, i_max + 0.5)
        ]
    )
    faces = [
        [0, 1, 3, 2],
        [4, 5, 7, 6],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [0, 2, 6, 4],
        [1, 3, 7, 5],
    ]

    world_corners = []
    for array, color in ((before, "red"), (after, "cyan")):
        # Apply the affine, then reorder world (z, y, x) for display as (z, x, y).
        world = (array.fusi.affine.voxel_to_world @ corners.T)[:3].T[:, [0, 2, 1]]
        world_corners.append(world)
        ax.add_collection3d(
            Poly3DCollection(
                [world[face] for face in faces],
                facecolor=color,
                edgecolor=color,
                alpha=0.35,
            )
        )

    # Expand the template bounds to contain both slabs, including enlarged zooms.
    all_corners = np.concatenate(world_corners)
    limits_min = np.minimum(template_min[[0, 2, 1]], all_corners.min(axis=0))
    limits_max = np.maximum(template_max[[0, 2, 1]], all_corners.max(axis=0))
    extent = limits_max - limits_min
    padding = 0.02 * extent
    limits_min -= padding
    limits_max += padding

    # Mark the fixed point and positive world axes.
    if center is not None:
        if isinstance(center, Mapping):
            center = [center[name] for name in "zyx"]
        center = np.asarray(center, dtype=float)[[0, 2, 1]]

        arrow_length = 0.15 * np.linalg.norm(limits_max - limits_min)

        for name in axis or "":
            direction = np.zeros(3)
            direction["zxy".index(name)] = 1

            ax.quiver(
                *center,
                *direction,
                length=arrow_length,
                normalize=True,
                color="yellow",
                linewidth=2.5,
                arrow_length_ratio=0.25,
            )
            ax.text(
                *(center + 1.1 * arrow_length * direction),
                f"+{name}",
                color="yellow",
            )

        ax.plot(*center, "o", color="yellow", markersize=8)

    # Match the composite panels: x right, y down, +z toward the viewer.
    for setter, low, high in zip(
        (ax.set_xlim, ax.set_ylim, ax.set_zlim), limits_min, limits_max
    ):
        setter(low, high)
    ax.invert_zaxis()
    aspect = limits_max - limits_min
    aspect[2] *= 0.8  # Compress the displayed y dimension.
    ax.set_box_aspect(limits_max - limits_min, zoom=0.85)
    ax.set_proj_type("ortho")
    ax.view_init(elev=25, azim=20)
    ax.set_zlabel("y (mm)")

    # Put the world x and z labels on the front edges, away from the slabs.
    ax.xaxis.set_tick_params(labelcolor="none")
    ax.yaxis.set_tick_params(labelcolor="none")
    z_min, x_min, _ = limits_min
    z_max, x_max, y_max = limits_max
    gap = 0.02 * max(limits_max - limits_min)

    for x in ax.get_yticks():
        if x_min <= x <= x_max:
            ax.text(z_max, x, y_max + gap, f"{x:g}", ha="center", va="top")
    for z in ax.get_xticks()[::2]:
        if z_min <= z <= z_max:
            ax.text(z, x_max + gap, y_max, f"{z:g}", ha="left", va="center")
    ax.text(
        z_max, (x_min + x_max) / 2, y_max + 4 * gap, "x (mm)", ha="center", va="top"
    )
    ax.text(
        (z_min + z_max) / 2, x_max + 5 * gap, y_max, "z (mm)", ha="left", va="center"
    )
    return ax


# %%
_ = plot_plane_in_3d(plane, rotated_about_center, center=center, axis="z")

# %% [markdown]
# ### Rotate about a given point
#
# To rotate about a given point rather than the geometry center, pass its world position
# as `center`. Here, we select a point slightly offset from the top-left corner (yellow
# dot), so the plane swings around that point instead of turning in place.
#
# The first sampled voxel's world position is available as
# [`.fusi.origin`][confusius.xarray.FUSIAccessor.origin]; passing `center="origin"` uses
# it directly.

# %%
first_voxel = plane.fusi.origin
pivot = {dim: first_voxel[dim] + offset for dim, offset in zip("zyx", (0.0, 2.5, 2.5))}
rotated_about_pivot = plane.fusi.rotate(30, "z", center=pivot)

plotter = template.fusi.plot.composite(rotated_about_pivot, **panel_kwargs)
_ = plotter.axes.ravel()[3].plot(
    pivot["x"],
    pivot["y"],
    "o",
    color="yellow",
    markersize=15,
    clip_on=False,
)

# %%
_ = plot_plane_in_3d(plane, rotated_about_pivot, center=pivot, axis="z")

# %% [markdown]
# ### Rotate out of the plane
#
# Rotating about `y` tilts the plane away from its original `z` position. It now
# intersects several `z` panels, each showing a narrow band that shifts along `x`.

# %%
tilted = plane.fusi.rotate(15, "y")

_ = template.fusi.plot.composite(tilted, **panel_kwargs)

# %%
_ = plot_plane_in_3d(plane, tilted, center=center, axis="y")

# %% [markdown]
# A view sliced along `y` shows the full tilt. The yellow rectangle marks the
# [`.fusi.affine.bounding_box`][confusius.xarray.FUSIAffineAccessor.bounding_box] of the
# six `z` panels above: they cover only a small part of the tilted plane.

# %%
plotter = template.fusi.plot.composite(
    tilted, slice_mode="y", slice_coords=[3], bg_color=bg_color
)

bounding_box = template.sel(z=slice_coords).fusi.affine.bounding_box
z_min, z_max = bounding_box["z"]
x_min, x_max = bounding_box["x"]
_ = plotter.axes.ravel()[0].add_patch(
    Rectangle(
        (x_min, z_min), x_max - x_min, z_max - z_min, fill=False, edgecolor="yellow"
    )
)

# %% [markdown]
# ### Rotate about multiple axes
#
# `rotate` also accepts multiple angles and axes, applied in order about the fixed
# world axes. Here we rotate -10° about `z`, then 20° about `y`, then 30° about
# `x`. Changing the order changes the result, as rotations are not generally
# commutative. As before, the `z` panels show only narrow intersections of the
# transformed plane.

# %%
rotated_about_point = plane.fusi.rotate((-10, 20, 30), "zyx", center=pivot)

plotter = template.fusi.plot.composite(rotated_about_point, **panel_kwargs)
_ = plotter.axes.ravel()[3].plot(
    pivot["x"],
    pivot["y"],
    "o",
    color="yellow",
    markersize=15,
)

# %% [markdown]
# The 3D view shows the shared pivot and all three fixed world axes. The axes
# indicate the successive rotations, not the path taken by the plane.

# %%
_ = plot_plane_in_3d(plane, rotated_about_point, center=pivot, axis="zyx")

# %% [markdown]
# ## Translate
#
# [`translate`][confusius.xarray.FUSIAccessor.translate] shifts the grid by a
# `(z, y, x)` vector in world units (millimetres here). The vector `(-0.2, 1, 3)` moves
# the plane -0.2 mm along `z` (two panels), 1 mm along `y`, and 3 mm along `x`.

# %%
translated = plane.fusi.translate((-0.2, 1, 3))

_ = template.fusi.plot.composite(translated, **panel_kwargs)

# %% [markdown]
# In 3D, the displacement along all three axes is easier to see. A translation has
# no fixed point.

# %%
_ = plot_plane_in_3d(plane, translated)

# %% [markdown]
# ## Zoom
#
# [`zoom`][confusius.xarray.FUSIAccessor.zoom] scales the grid about its geometry
# center by default. A factor of 2 doubles its size along every axis, **including
# its thickness along `z`**, so the plane appears in several `z` panels. Its center
# stays fixed, while the brain appears twice as large.

# %%
zoomed_about_center = plane.fusi.zoom(2)

plotter = template.fusi.plot.composite(zoomed_about_center, **panel_kwargs)
_ = plotter.axes.ravel()[3].plot(
    center["x"],
    center["y"],
    "o",
    color="yellow",
    markersize=15,
)

# %% [markdown]
# The 3D view shows the enlarged plane and its doubled thickness.

# %%
_ = plot_plane_in_3d(plane, zoomed_about_center, center=center)

# %% [markdown]
# ### Zoom about a given point
#
# Zooming about a given point rather than the geometry center is similar to
# rotating about a given point. The zoom factor is applied to the distance of each
# voxel from the fixed point. Here, we zoom about the same pivot as the rotation above.
# The plane expands away from it, leaving only the pivot fixed.

# %%
zoomed_about_pivot = plane.fusi.zoom(2, center=pivot)

plotter = template.fusi.plot.composite(zoomed_about_pivot, **panel_kwargs)
_ = plotter.axes.ravel()[3].plot(
    pivot["x"],
    pivot["y"],
    "o",
    color="yellow",
    markersize=15,
    clip_on=False,
)

# %%
_ = plot_plane_in_3d(plane, zoomed_about_pivot, center=pivot)

# %% [markdown]
# ## Flip
#
# [`flip`][confusius.xarray.FUSIAccessor.flip] reflects the array across a plane
# perpendicular to a world axis. Like rotation and zoom, it uses the geometry
# center by default. To make the reflection clearer, we'll keep only one half
# of the brain and reflect it along world `x` about the **original plane's** center.
# This moves it to the opposite hemisphere without changing its data values.
#
# !!! warning "Anatomical orientation"
#     Flipping an axis can reverse anatomical left and right, depending on the
#     coordinate convention. A single-axis flip also reverses the handedness of
#     the coordinate system. Always verify anatomical orientation when flipping
#     data, especially before atlas registration or comparing datasets.
#
# We crop along `i` here because it follows the left-right direction in this template.

# %%
hemisphere = plane.isel(i=slice(0, plane.sizes["i"] // 2))
_ = template.fusi.plot.composite(hemisphere, **panel_kwargs)

# %%
flipped = hemisphere.fusi.flip("x", center=center)
_ = template.fusi.plot.composite(flipped, **panel_kwargs)

# %% [markdown]
# The 3D view shows the original and reflected halves on opposite sides of the
# fixed point (yellow). Unlike reversing the array with `isel`, `flip` changes
# only its voxel-to-world affine; the voxel labels and values stay unchanged.
# The full plane's `center` is passed explicitly because the cropped hemisphere
# has a different geometry center.

# %%
_ = plot_plane_in_3d(hemisphere, flipped, center=center)

# %% [markdown]
# ## Summary
#
# Every transform returns a new DataArray with unchanged data values and voxel
# labels (`k`, `j`, `i`). Only the voxel-to-world affine changes, along with the
# world coordinates derived from it. The cells below confirm this and compare
# the affines before and after zooming.
#
# !!! warning "Transform order matters!"
#     Transforms are not always commutative:
#
#     - **Translations** commute: their displacements add up.
#     - **Rotations** about the same axis and pivot commute, but rotations about
#       different axes generally do not.
#     - **Zooms and flips** about the same pivot commute with each other when
#       they scale or reflect along the fixed world axes.
#     - **Mixed transforms** generally do not commute. However, a uniform zoom
#       and rotation about the same fixed point do commute.
#
# To place a transformed array on another voxel grid, use
# [`resample_like`][confusius.registration.resample_like]. This is what
# [`plot_composite`][confusius.plotting.plot_composite] does internally to display
# the transformed plane on the template grid.

# %%
np.array_equal(zoomed_about_center.values, plane.values)

# %%
plane.fusi.affine.voxel_to_world

# %%
zoomed_about_center.fusi.affine.voxel_to_world

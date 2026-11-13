# %% [markdown]
# # Transforms 101
#
# A VoxelData array stores values on a voxel grid whose position in world space is
# defined by an affine transform (the voxel-to-world affine, see
# [World Space](../../../user-guide/voxeldata.md#world-space)). Thus, moving a VoxelData
# array in world space is as simple as applying a transformation to its affine. The
# [`.fusi`][confusius.xarray.FUSIAccessor] accessor provides methods to apply
# translations, rotations, and zooms. Note that these methods do not resample the data:
# they only update its position in world space. Resampling would happen during plotting
# (e.g., [`plot_volume`][confusius.plotting.plot_volume]) or when explicitly requested
# (e.g., [`resample_like`][confusius.registration.resample_like]).
#
# In this example, we'll use the mouse fUSI template from Pepe, Mariani et al.
# (2026)[^collection] to demonstrate the transforms. We'll use one `z`-slice of the
# template as our moving array and keep the full template as a fixed background.
#
# [^collection]: See [The ConfUSIus Dataset Collection](dataset_collection.md)
#     for available studies, formats, licenses, and reproducible subset downloads.

# %%
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
# Before any transform, the selected plane aligns with the fourth `z` panel. It appears
# cyan, or white where it overlaps the bright template. The other panels show only the
# template.

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
# ## Rotate about the grid origin
#
# [`.fusi.rotate()`][confusius.xarray.FUSIAccessor.rotate] takes an angle in
# degrees and a world-axis name. By default, it rotates about the grid origin: the world
# position of the array's first voxel. Here, that is the plane's top-left corner, marked
# by a yellow dot.

# %%
rotated_about_origin = plane.fusi.rotate(30, "z")

origin = plane.fusi.origin
plotter = template.fusi.plot.composite(rotated_about_origin, **panel_kwargs)
_ = plotter.axes.ravel()[3].plot(
    origin["x"], origin["y"], "o", color="yellow", markersize=15, clip_on=False
)

# %% [markdown]
# The composite plots show where the plane intersects each `z` slice. To see its
# orientation, we'll also use a 3D view. The collapsed helper below draws the original
# plane in red, the transformed plane in cyan, and any fixed point in yellow. Yellow
# arrows mark the positive world axes for rotations, with labels near their tips. The
# slabs are thickened for visibility except in the zoom comparisons.
#
# The 3D view matches the panels: `x` points right, `y` down, and `+z` toward the
# viewer. The `(z, y, x)` coordinates follow the right-hand rule: with your right thumb
# pointing along `+z`, your fingers curl from `+y` toward `+x`, the direction of this
# positive rotation.

# %% tags=["collapse: Code for `plot_plane_in_3d` helper"]
def plot_plane_in_3d(before, after, center=None, axis=None, thickness=5):
    """Show a plane before (red) and after (cyan) a transform."""
    box = template.fusi.affine.bounding_box
    template_min = box.sel(bound="min").values
    template_max = box.sel(bound="max").values

    fig = plt.figure(figsize=(7, 4.5), facecolor=bg_color)
    ax = fig.add_subplot(projection="3d", facecolor=bg_color)
    ax.set_position([0, 0, 1, 1])
    ax.computed_zorder = False  # Keep the pivot and arrows in front of the slabs.

    # Exaggerate thin planes for visibility unless a smaller thickness is requested.
    k, nj, ni = before.k.item(), before.sizes["j"], before.sizes["i"]
    corners = np.array(
        [
            [kk, jj, ii, 1]
            for kk in (k - thickness / 2, k + thickness / 2)
            for jj in (-0.5, nj - 0.5)
            for ii in (-0.5, ni - 0.5)
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

    # Show the fixed point and positive world axes for rotations or zooms.
    if center is not None:
        center = np.asarray(center, dtype=float)[[0, 2, 1]]
        for name in axis or "":
            end = center.copy()
            end["zxy".index(name)] = template_max["zyx".index(name)]
            direction = end - center
            ax.quiver(
                *center,
                *direction,
                color="yellow",
                linewidth=2.5,
                arrow_length_ratio=0.25,
            )
            ax.text(*(center + 0.85 * direction), f"+{name}", color="yellow")
        ax.plot(*center, "o", color="yellow", markersize=8)

    # Expand the template bounds to contain both slabs, including enlarged zooms.
    all_corners = np.concatenate(world_corners)
    limits_min = np.minimum(template_min[[0, 2, 1]], all_corners.min(axis=0))
    limits_max = np.maximum(template_max[[0, 2, 1]], all_corners.max(axis=0))
    extent = limits_max - limits_min
    padding = 0.02 * extent
    limits_min -= padding
    limits_max += padding

    # Match the composite panels: x right, y down, +z toward the viewer.
    for setter, low, high in zip(
        (ax.set_xlim, ax.set_ylim, ax.set_zlim), limits_min, limits_max
    ):
        setter(low, high)
    ax.invert_zaxis()
    ax.set_box_aspect(limits_max - limits_min, zoom=0.85)
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
_ = plot_plane_in_3d(
    plane, rotated_about_origin, center=list(origin.values()), axis="z"
)

# %% [markdown]
# ## Rotate about the geometry center
#
# With `center="geometry"`, the plane rotates around the center of its bounding box,
# available with
# [`.fusi.affine.geometry_center`][confusius.xarray.FUSIAffineAccessor.geometry_center].
# It now turns in place instead of around its corner.

# %% tags=["thumbnail"]
rotated_about_center = plane.fusi.rotate(30, "z", center="geometry")

center = plane.fusi.affine.geometry_center
plotter = template.fusi.plot.composite(rotated_about_center, **panel_kwargs)
_ = plotter.axes.ravel()[3].plot(
    center.sel(component="x"),
    center.sel(component="y"),
    "o",
    color="yellow",
    markersize=15,
)

# %%
_ = plot_plane_in_3d(plane, rotated_about_center, center=center, axis="z")

# %% [markdown]
# ## Rotate out of the plane
#
# Rotating about `y` tilts the plane away from its original `z` position. It now
# intersects several `z` panels, each showing a narrow band that shifts along `x`.

# %%
tilted = plane.fusi.rotate(15, "y", center="geometry")

_ = template.fusi.plot.composite(tilted, **panel_kwargs)

# %%
_ = plot_plane_in_3d(plane, tilted, center=center, axis="y")

# %% [markdown]
# A view sliced along `y` shows the full tilt. The yellow rectangle marks the
# [`.fusi.affine.bounding_box`][confusius.xarray.FUSIAffineAccessor.bounding_box] of the
# six `z` panels above: they cover only a small part of the tilted plane.

# %%
band_box = template.sel(z=slice_coords).fusi.affine.bounding_box
band_box

# %%
plotter = template.fusi.plot.composite(
    tilted, slice_mode="y", slice_coords=[3], bg_color=bg_color
)

z_min, z_max = band_box.sel(component="z").values
x_min, x_max = band_box.sel(component="x").values
_ = plotter.axes.ravel()[0].add_patch(
    Rectangle(
        (x_min, z_min), x_max - x_min, z_max - z_min, fill=False, edgecolor="yellow"
    )
)

# %% [markdown]
# ## Rotate about multiple axes and a chosen point
#
# [`.fusi.rotate`][confusius.xarray.FUSIAccessor.rotate] also accepts multiple angles
# and axes, applied in order about the fixed world axes. Here we rotate -10° about `z`,
# then 20° about `y`, then 30° about `x`. Changing the order changes the result.
#
# `center` can be any `(z, y, x)` world point. Here we choose a point on the plane, 2.5
# mm above and 4.5 mm to the right of its geometry center (yellow dot). Because the six
# panels span only 0.6 mm along `z`, they show just narrow sections of the rotated
# plane.

# %%
pivot = plane.fusi.affine.geometry_center + np.array((0, -2.5, 4.5))

rotated_about_point = plane.fusi.rotate((-10, 20, 30), "zyx", center=pivot)

plotter = template.fusi.plot.composite(rotated_about_point, **panel_kwargs)
_ = plotter.axes.ravel()[3].plot(
    pivot.sel(component="x"),
    pivot.sel(component="y"),
    "o",
    color="yellow",
    markersize=15,
)

# %% [markdown]
# The 3D view makes this easier to follow: the center of rotation stays fixed, and the
# plane rotates about it about the three axes.

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
# The 3D view shows the displacement along all three axes. Unlike rotation and zoom,
# translation has no fixed point.

# %%
_ = plot_plane_in_3d(plane, translated)

# %% [markdown]
# ## Zoom
#
# [`zoom`][confusius.xarray.FUSIAccessor.zoom] scales the grid around its origin by
# default. A factor of 2 doubles its size along every axis, including its thickness
# along `z`, so the plane appears in three panels. Because the origin stays fixed, the
# plane expands away from its top-left corner. Only the region near that corner remains
# inside the template.

# %%
zoomed_about_origin = plane.fusi.zoom(2)

plotter = template.fusi.plot.composite(zoomed_about_origin, **panel_kwargs)
_ = plotter.axes.ravel()[3].plot(
    origin["x"], origin["y"], "o", color="yellow", markersize=15, clip_on=False
)

# %% [markdown]
# The 3D view shows the scaling along every axis, including the thickness along `z`. Its
# limits expand to show the whole plane rather than cropping it to the template.

# %%
_ = plot_plane_in_3d(
    plane, zoomed_about_origin, center=list(origin.values())
)

# %% [markdown]
# With `center="geometry"`, the center of the plane's bounding box stays fixed. The
# brain remains centered and appears two times larger.

# %%
zoomed_about_center = plane.fusi.zoom(2, center="geometry")

plotter = template.fusi.plot.composite(zoomed_about_center, **panel_kwargs)
_ = plotter.axes.ravel()[3].plot(
    center.sel(component="x"),
    center.sel(component="y"),
    "o",
    color="yellow",
    markersize=15,
)

# %% [markdown]
# Compare this with the previous 3D view: the plane expands around its center rather
# than its corner, while its thickness also triples.

# %%
_ = plot_plane_in_3d(plane, zoomed_about_center, center=center)

# %% [markdown]
# ## Summary
#
# Each transform returns a new DataArray with the same values and the same
# (`k`, `j`, `i`) grid. Only the voxel-to-world affine changes, together with the world
# coordinates derived from it. The cells below confirm the unchanged values and compare
# the affines before and after zooming.
#
# !!! warning "Transform order matters!"
#     Transforms are generally not commutative:
#
#     - **Translations** commute: their displacements simply add up.
#     - **Rotations** commute about the same axis and fixed point, but generally not
#       about different axes.
#     - **Zooms** commute when applied about the same fixed point, even with different
#       scaling factors along each axis.
#     - **Mixed transforms** generally do not commute. Translation changes the result of
#       a subsequent rotation or zoom (and vice versa). However, a uniform zoom and a
#       rotation about the same fixed point do commute.
#
# To place a transformed array on another voxel grid, use
# [`resample_like`][confusius.registration.resample_like]. This is what
# [`plot_composite`][confusius.plotting.plot_composite] does internally to display the
# transformed plane on the template grid.

# %%
np.array_equal(zoomed_about_center.values, plane.values)

# %%
plane.fusi.affine.voxel_to_world

# %%
zoomed_about_center.fusi.affine.voxel_to_world

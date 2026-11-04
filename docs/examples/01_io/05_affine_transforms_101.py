# %% [markdown]
# # Affine transforms 101
#
# A VoxelData array keeps its values on a voxel grid. A voxel-to-world affine places
# that grid in world space (see [World Space](../../../user-guide/voxeldata.md#world-space)
# in The VoxelData Model). The [`.fusi`][confusius.xarray.FUSIAccessor] accessor moves
# the grid in world space. It changes only the affine: the data values and the voxel
# grid do not change. This example shows the three basic moves:
# [`translate`][confusius.xarray.FUSIAccessor.translate],
# [`rotate`][confusius.xarray.FUSIAccessor.rotate] and
# [`zoom`][confusius.xarray.FUSIAccessor.zoom]. It also shows how the `center`
# argument selects the point that does not move during a rotation or a zoom.

# %% [markdown]
# ## Fetch the template and select one plane
#
# This example uses the mouse fUSI template from Pepe, Mariani et al.
# (2026)[^collection]. We move one `z` plane of the template and keep the full template
# as a fixed background. Each composite figure shows the template in red and the moved
# plane in cyan, on six `z` planes around the plane that we move.
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

template = cf.datasets.fetch_template_pepe_mariani_2026()

# ConfUSIus stores world coordinates per voxel. Read the z value of the first voxel of
# each plane in a 0.6 mm band.
slice_coords = template.sel(z=slice(5.8, 6.4)).z.values[:, 0, 0]

# Pass a list so the plane keeps its k dimension and stays a VoxelData array.
plane = template.sel(z=[slice_coords[3]], method="nearest")
plane

# %% [markdown]
# Before a move, the plane sits exactly on the fourth `z` panel. The composite shows it
# in cyan, and in white where the template is also bright. The other panels show only
# the template.

# %%
# Options shared by the six-panel composites. The figure is wide, so use a large font.
panel_kwargs = dict(slice_coords=slice_coords, bg_color=bg_color, fontsize=20)

_ = template.fusi.plot.composite(plane, **panel_kwargs)

# %% [markdown]
# ## Rotate about the grid origin
#
# `rotate` takes an angle in degrees and the name of the world axis to turn about. By
# default, the fixed point of the rotation is the grid origin. The grid origin is the
# world position of the first voxel of the array. In these figures, it is the top-left
# corner of the plane. The yellow dot marks it.

# %%
rotated_about_origin = plane.fusi.rotate(30, "z")

origin = plane.fusi.origin
plotter = template.fusi.plot.composite(rotated_about_origin, **panel_kwargs)
_ = plotter.axes.ravel()[3].plot(
    origin["x"], origin["y"], "yo", markersize=15, clip_on=False
)

# %% [markdown]
# A 3D view shows the rotation axis. The collapsed `plot_plane_in_3d` helper draws the
# plane before (red) and after (cyan) a move, inside the bounding box of the template.
# It also draws the fixed point and an arrow from that point along the rotation axis.
# The arrow points in the positive direction of that axis. The plane is one voxel thick.
# The helper draws it thicker so that the slab is visible.
#
# The view has the same orientation as the composite panels: `x` to the right, `y`
# downward, and `z` as the depth axis, with `+z` toward you. This orientation keeps the
# right-hand rule of the `(z, y, x)` world coordinates. With the thumb of your right
# hand along `+z`, the fingers curl from `+y` toward `+x`. That is the direction in
# which the plane turned.


# %% tags=["collapse: Code for `plot_plane_in_3d` helper"]
def plot_plane_in_3d(before, after, center, axis):
    """Draw a plane before (red) and after (cyan) a move, inside the template box."""
    box = template.fusi.affine.bounding_box
    fig = plt.figure(figsize=(7, 4.5), facecolor=bg_color)
    ax = fig.add_subplot(projection="3d", facecolor=bg_color)
    # Fill the figure with the axes, below the legend. Together with the box zoom
    # further down, this keeps labels and slabs inside the figure.
    ax.set_position([0, 0, 1, 0.92])
    # Draw in call order, so the fixed point and the axis stay on top of the slabs.
    ax.computed_zorder = False

    # The voxel corners of the plane are the same before and after the move. Only the
    # voxel-to-world affine differs. The plane is one voxel thick along k. Draw it five
    # voxels thick so that the slab is visible.
    k, nj, ni = before.k.item(), before.sizes["j"], before.sizes["i"]
    corners = np.array(
        [
            [kk, jj, ii, 1]
            for kk in (k - 2.5, k + 2.5)
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
    for array, color in ((before, "red"), (after, "cyan")):
        # World corners in (z, y, x), reordered to the (z, x, y) display axes.
        world = (array.fusi.affine.voxel_to_world @ corners.T)[:3].T[:, [0, 2, 1]]
        ax.add_collection3d(
            Poly3DCollection(
                [world[face] for face in faces],
                facecolor=color,
                edgecolor=color,
                alpha=0.35,
            )
        )

    # Arrow from the fixed point along the rotation axis, in its positive direction, up
    # to the edge of the template box.
    center = np.asarray(center, dtype=float)[[0, 2, 1]]
    end = center.copy()
    end["zxy".index(axis)] = box.sel(component=axis, bound="max").item()
    ax.quiver(
        *center,
        *(end - center),
        color="yellow",
        linewidth=2.5,
        arrow_length_ratio=0.25,
        label=f"rotation axis (+{axis})",
    )
    ax.plot(*center, "yo", markersize=8, label="fixed point")

    # Same orientation as the composite panels: x to the right, y downward.
    for name, setter in zip("zxy", (ax.set_xlim, ax.set_ylim, ax.set_zlim)):
        setter(*box.sel(component=name).values)
    ax.invert_zaxis()
    extent = (box.sel(bound="max") - box.sel(bound="min")).values
    ax.set_box_aspect(extent[[0, 2, 1]], zoom=0.85)
    ax.view_init(elev=25, azim=20)
    ax.set_zlabel("y (mm)")
    # Matplotlib writes 3D tick labels next to the back panes, where the slabs hide
    # them. Hide the x and z tick labels and write them along the front edges instead.
    # Matplotlib's x axis holds world z, and its y axis holds world x.
    ax.xaxis.set_tick_params(labelcolor="none")
    ax.yaxis.set_tick_params(labelcolor="none")
    z_min, _, x_min = box.sel(bound="min").values
    z_max, y_max, x_max = box.sel(bound="max").values
    # Space between the box edge and the labels, in mm.
    gap = 0.4
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
    fig.legend(loc="upper center", ncol=2)
    return ax


# %%
_ = plot_plane_in_3d(
    plane, rotated_about_origin, center=list(origin.values()), axis="z"
)

# %% [markdown]
# ## Rotate about the geometry center
#
# Pass `center="geometry"` to turn the plane about the center of its bounding box. This
# point is available as
# [`geometry_center`][confusius.xarray.FUSIAffineAccessor.geometry_center]. The plane
# now turns in place. You can also pass a `(z, y, x)` world point as `center`.

# %% tags=["thumbnail"]
rotated_about_center = plane.fusi.rotate(30, "z", center="geometry")

center = plane.fusi.affine.geometry_center
plotter = template.fusi.plot.composite(rotated_about_center, **panel_kwargs)
_ = plotter.axes.ravel()[3].plot(
    center.sel(component="x"), center.sel(component="y"), "yo", markersize=15
)

# %%
_ = plot_plane_in_3d(plane, rotated_about_center, center=center, axis="z")

# %% [markdown]
# ## Rotate out of the plane
#
# A rotation about `y` tilts the plane out of its initial `z` position. The tilted
# plane now crosses several `z` panels. Each panel shows only the thin band where the
# tilted plane cuts through it. This band moves along `x` from one panel to the next.

# %%
tilted = plane.fusi.rotate(15, "y", center="geometry")

_ = template.fusi.plot.composite(tilted, **panel_kwargs)

# %%
_ = plot_plane_in_3d(plane, tilted, center=center, axis="y")

# %% [markdown]
# To see the full tilted plane, slice the composite along `y`. The yellow box is the
# [`bounding_box`][confusius.xarray.FUSIAffineAccessor.bounding_box] of the six `z`
# planes from the previous figures. Those panels showed only the part of the tilted
# plane inside this box.

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
# ## Translate
#
# `translate` moves the array by a `(z, y, x)` vector in world units. Here the plane
# moves 0.2 mm along `z`, 1 mm along `y` and 3 mm along `x`. The move along `z` puts the
# plane two panels later. The moves along `y` and `x` shift it inside that panel.

# %%
translated = plane.fusi.translate((0.2, 1, 3))

_ = template.fusi.plot.composite(translated, **panel_kwargs)

# %% [markdown]
# ## Zoom
#
# `zoom` scales the array by a factor, about the grid origin by default. A factor of 3
# makes the plane three times larger in each direction, which includes its thickness
# along `z`. As a result, the plane now shows in three panels. The origin does not move,
# so the plane grows away from its top-left corner. Only the part near that corner stays
# inside the template.

# %%
zoomed_about_origin = plane.fusi.zoom(3)

plotter = template.fusi.plot.composite(zoomed_about_origin, **panel_kwargs)
_ = plotter.axes.ravel()[3].plot(
    origin["x"], origin["y"], "yo", markersize=15, clip_on=False
)

# %% [markdown]
# With `center="geometry"`, the plane grows about its own center. The brain stays in the
# center of the panel and looks three times larger.

# %%
zoomed_about_center = plane.fusi.zoom(3, center="geometry")

plotter = template.fusi.plot.composite(zoomed_about_center, **panel_kwargs)
_ = plotter.axes.ravel()[3].plot(
    center.sel(component="x"), center.sel(component="y"), "yo", markersize=15
)

# %% [markdown]
# ## What changed, and what did not
#
# Each move returned a new DataArray with the same values and the same `k`, `j`, `i`
# grid as `plane`. Only the voxel-to-world affine changed, and with it the world
# coordinates derived from that affine. Moves apply in the order that you write them.
# Thus a rotation followed by a zoom differs from a zoom followed by a rotation. To
# resample a moved array onto another grid, use
# [`resample_like`][confusius.registration.resample_like]. This is what
# [`plot_composite`][confusius.plotting.plot_composite] did internally for each figure
# in this example.

# %%
np.array_equal(zoomed_about_center.values, plane.values)

# %%
zoomed_about_center.fusi.affine.voxel_to_world

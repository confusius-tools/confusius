# %% [markdown]
# # Load multi-pose recordings
#
# This example shows how to assemble a multi-pose fUSI recording that was not acquired
# through an Iconeus SCAN file, by loading each pose as a separate NIfTI file and
# stacking them into a single VoxelData array with pose-dependent voxel-to-world
# geometry.
#
# The [Pepe, Mariani et al. (2026)
# dataset][confusius.datasets.fetch_pepe_mariani_2026] contains transcranial mouse
# resting-state recordings acquired with a linear probe stepped across several
# positions. In the fUSI-BIDS export, each probe position is one file, distinguished by
# the `chunk-` entity (`chunk-0`, `chunk-1`, ...); this is exactly the "Other Systems"
# case described in the [Multi-Pose Imaging guide](../../../user-guide/multipose.md#other-systems)
# — data must be assembled manually rather than loaded as one file, unlike Iconeus
# SCAN's `3Dscan`/`4Dscan` modes.

# %% [markdown]
# ## Fetch one recording's chunks
#
# We select one subject, session, and acquisition so only the four chunks belonging to
# a single recording are downloaded, rather than the full multi-gigabyte dataset.

# %%
import re
from pathlib import Path

import matplotlib as mpl
import xarray as xr

import confusius as cf

# Adapt background color to the current Matplotlib style.
bg_color = mpl.colors.to_hex(mpl.rcParams["figure.facecolor"])

xr.set_options(display_expand_data=False)

subject = "cp230420a"
session = "1MEDISOses5"
acq = "3dfusi"

bids_root = cf.datasets.fetch_pepe_mariani_2026(
    datasets="rawdata",
    subjects=subject,
    sessions=session,
    acqs=acq,
    datatypes="fusi",
)

# %% [markdown]
# ## Locate and load every chunk
#
# Each chunk is an ordinary single-pose fUSI recording — `(time, k, j, i)` with a
# singleton `k`, since a linear probe images one elevation slice per pose — with its own
# `voxel_to_world` affine derived from that file's own NIfTI header. We sort by the
# `chunk-` index so poses end up in acquisition order, then load and compute each one
# (they're small enough to fit comfortably in memory).

# %%
fusi_dir = Path(bids_root) / f"sub-{subject}" / f"ses-{session}" / "fusi"
chunk_glob = (
    f"sub-{subject}_ses-{session}_task-rest_acq-{acq}_probe-linear_chunk-*_pwd.nii.gz"
)
chunk_paths = sorted(
    fusi_dir.glob(chunk_glob),
    key=lambda p: int(re.search(r"chunk-(\d+)", p.name).group(1)),
)
chunks = [cf.load(p).compute() for p in chunk_paths]

npose = len(chunks)
print(f"{npose} poses, each shaped {chunks[0].dims} = {chunks[0].shape}")

# %% [markdown]
# Each chunk's own origin confirms the poses sit at different physical positions along
# elevation (`z`), stepped by about 1 mm, and were acquired with slightly offset start
# times — this is the per-pose geometry we want to preserve, not collapse into one
# shared affine. `y`/`x` origins don't vary across chunks, so we only show `z`/`time`.

# %%
origins = [chunk.fusi.origin for chunk in chunks]
print("z origins:", tuple(origin["z"] for origin in origins))
print("time origins:", tuple(origin["time"] for origin in origins))

# %% [markdown]
# ## Stack the poses into one pose-dependent DataArray
#
# [`stack_poses`][confusius.multipose.stack_poses] concatenates independently loaded
# single-pose grids into one DataArray with a `pose` dimension, one voxel-to-world
# affine per pose. Poses were also acquired sequentially rather than simultaneously, so
# each chunk's `time` coordinate is offset from the others by a fraction of the
# repetition time; when per-pose timestamps differ, `stack_poses` keeps them as a
# pose-dependent `(time, pose)`-shaped `time` coordinate rather than collapsing them
# into one shared time axis.

# %%
multipose = cf.multipose.stack_poses(chunks)

# %% [markdown]
# ## Pose-transparent vs. pose-specific geometry
#
# Some geometry queries are well-defined without picking a pose: voxel spacing must be
# identical across poses (a stacked affine with differing scale is rejected at
# construction), so [`.fusi.spacing`][confusius.xarray.FUSIAccessor.spacing] works
# directly on the multi-pose array.
#
# Origin and direction, however, are inherently single-grid concepts—there is no one
# answer for "the origin" of a stack of differently-positioned grids—so they require
# selecting a scalar pose first.
#
# !!! warning "World-coordinate `.sel()` requires a scalar pose"
#     `.sel(z=..., y=..., x=...)` raises `ValueError` on pose-dependent data unless
#     `pose` was already reduced to a scalar, e.g. `.isel(pose=0).sel(z=..., y=...,
#     x=...)`. Resolving a world coordinate to a voxel means picking one pose's
#     affine first; ConfUSIus never resamples poses onto a shared grid implicitly.

# %%
pose0_origin = multipose.isel(pose=0).fusi.origin
pose3_origin = multipose.isel(pose=3).fusi.origin
print("spacing (pose-transparent):", multipose.fusi.spacing)
print("z origins (pose 0, 3):", (pose0_origin["z"], pose3_origin["z"]))
print("time origins (pose 0, 3):", (pose0_origin["time"], pose3_origin["time"]))

# %% [markdown]
# ## Visualize each pose
#
# We plot the temporal mean of each pose side by side with
# [`plot_volume`][confusius.plotting.plot_volume]'s `slice_mode="pose"`, which labels
# each panel by its `pose` value.
#
# !!! warning "World-coordinate slice modes require a scalar pose"
#     For the same reason as `.sel`, `plot_volume(..., slice_mode="z")` (or `"y"`/`"x"`)
#     also raises `ValueError` on pose-dependent data. Slicing along a native voxel dim
#     (`slice_mode="k"`/`"j"`/`"i"`, as below) or along `pose` itself works fine, since
#     neither requires picking one affine.

# %% tags=["thumbnail"]
mean_db = multipose.mean("time").fusi.scale.db().isel(k=0)

plotter = cf.plotting.plot_volume(
    mean_db,
    slice_mode="pose",
    cbar_label="Power Doppler (dB)",
    bg_color=bg_color,
)

# %% [markdown]
# ## Consolidate into a single volume
#
# Since every pose here shares the same rotation and is offset by a pure translation
# along elevation, [`consolidate_poses`][confusius.multipose.consolidate_poses] can
# merge `pose` and the swept voxel dimension (`k`, the default) into one physically
# ordered axis — the same operation `load_scan`-produced `3Dscan`/`4Dscan` data goes
# through, now working directly off the DataArray's own per-pose geometry rather than a
# separately stored affine.

# %%
consolidated = cf.multipose.consolidate_poses(multipose)
consolidated

# %% [markdown]
# The consolidated volume has a genuine `k` extent of `npose` elevation slices (each
# pose here contributes exactly one, since the probe is linear) and no more `pose`
# dimension, positioned by real physical location rather than acquisition order.

# %%
plotter = cf.plotting.plot_volume(
    consolidated.mean("time").fusi.scale.db(),
    slice_mode="k",
    cmap="gray",
    cbar_label="Power Doppler (dB)",
    bg_color=bg_color,
)

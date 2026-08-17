# %% [markdown]
# # Load a multi-pose recording from the Pepe, Mariani 2026 dataset
#
# This example shows how to assemble a multi-pose fUSI recording that was not acquired
# through an Iconeus SCAN file, by loading each pose as a separate NIfTI file and
# stacking them into a single [VoxelData][confusius._utils.geometry.VoxelToWorldIndex]
# DataArray with pose-dependent voxel-to-world geometry.
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
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import confusius as cf
from confusius._utils.geometry import get_voxel_to_world_affine

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
# elevation (`z`), stepped by about 1 mm — this is the per-pose geometry we want to
# preserve, not collapse into one shared affine.

# %%
for path, chunk in zip(chunk_paths, chunks, strict=True):
    print(path.stem, "origin:", chunk.fusi.origin)

# %% [markdown]
# ## Stack the poses into one pose-dependent DataArray
#
# [`create_fusi_dataarray`][confusius.xarray.create_fusi_dataarray] accepts an
# `(npose, 4, 4)` `voxel_to_world` stack — one affine per pose — alongside a `pose`
# dimension in `dims`. We stack each chunk's raw array along a new `pose` axis and each
# chunk's own affine into that stack; the resulting DataArray carries one physically
# meaningful voxel-to-world affine per pose, rather than approximating every pose with a
# single shared grid.
#
# Poses were also acquired sequentially rather than simultaneously, so each chunk's
# `time` coordinate is offset from the others by a fraction of the repetition time. We
# keep the first chunk's `time` as the array's main (regularly sampled) time axis and
# attach the real per-pose timestamps as a `pose_time` coordinate, mirroring how
# [`load_scan`][confusius.io.load_scan] handles `4Dscan` files.

# %%
voxel_to_world = np.stack([get_voxel_to_world_affine(chunk) for chunk in chunks])
data = np.stack([chunk.values for chunk in chunks], axis=1)  # (time, pose, k, j, i)
pose_time = np.stack([chunk.coords["time"].values for chunk in chunks], axis=1)

multipose = cf.xarray.create_fusi_dataarray(
    data,
    dims=("time", "pose", "k", "j", "i"),
    time=chunks[0].coords["time"],
    pose=np.arange(npose),
    voxel_to_world=voxel_to_world,
    name="pwd",
).assign_coords(pose_time=(("time", "pose"), pose_time))

multipose

# %% [markdown]
# ## Pose-transparent vs. pose-specific geometry
#
# Some geometry queries are well-defined without picking a pose: voxel spacing must be
# identical across poses (a ConfUSIus invariant, since a stacked affine with differing
# scale is rejected at construction), so [`.fusi.spacing`][confusius.xarray.FUSIAccessor.spacing]
# works directly on the multi-pose array.
#
# Origin and direction, however, are inherently single-grid concepts — there is no one
# answer for "the origin" of a stack of differently-positioned grids — so they require
# selecting a scalar pose first.

# %%
print("spacing (pose-transparent):", multipose.fusi.spacing)
print("pose 0 origin:", multipose.isel(pose=0).fusi.origin)
print("pose 3 origin:", multipose.isel(pose=3).fusi.origin)

# %% [markdown]
# ## Visualize each pose
#
# We plot the temporal mean of each pose side by side. Because each pose has its own
# affine, the `z` (elevation) position shown in each panel's title is genuinely that
# pose's physical location, not an arbitrary index.

# %% tags=["thumbnail"]
mean_db = multipose.mean("time").fusi.scale.db()

fig, axes = plt.subplots(1, npose, figsize=(4 * npose, 4), facecolor=bg_color)
for pose, ax in zip(range(npose), axes, strict=True):
    pose_slice = mean_db.isel(pose=pose, k=0)
    z = mean_db.isel(pose=pose).fusi.origin["z"]
    ax.imshow(
        pose_slice.values,
        cmap="gray",
        origin="lower",
        extent=[
            float(pose_slice.coords["x"].min()),
            float(pose_slice.coords["x"].max()),
            float(pose_slice.coords["y"].min()),
            float(pose_slice.coords["y"].max()),
        ],
    )
    ax.set_title(f"Pose {pose} (z = {z:.2f} mm)")
    ax.set_xlabel("x (mm)")
ax = axes[0]
_ = ax.set_ylabel("y (mm)")

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

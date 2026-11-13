# %% [markdown]
# # Load and save multi-pose recordings
#
# This example shows how to load and save multi-pose fUSI recordings split into several
# NIfTI files.
#
# The
# [Pepe, Mariani et al. (2026) dataset][confusius.datasets.fetch_pepe_mariani_2026][^collection]
# contains transcranial resting-state recordings acquired in mice. The authors used a
# linear probe stepped across several positions. Following the fUSI-BIDS specification,
# recordings are split into several NIfTI files, one per probe position, distinguished
# by the `chunk-` entity (`chunk-0`, `chunk-1`, ...).
#
# [^collection]: See [The ConfUSIus Dataset Collection](../io/dataset_collection.md)
#     for available studies, formats, licenses, and reproducible subset downloads.

# %% [markdown]
# ## Fetch one recording
#
# For this example, we select a single recording from the dataset.

# %%
from pathlib import Path
from tempfile import TemporaryDirectory

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
# ## Locate and load every pose
#
# The recording we fetched is split into four NIfTI files, one per pose. Each file
# contains a 2D+t fUSI recording of a single elevation slice, with its own spatial
# position and temporal offset. [`confusius.load`][confusius.load] can load all of them
# at once using a `{chunk}` placeholder in the filename pattern. The resulting
# [`DataArray`][xarray.DataArray] contains all poses stacked along a new `pose`
# dimension, with one voxel-to-world affine per pose and pose-dependent `time`
# coordinates.

# %%
fusi_dir = Path(bids_root) / f"sub-{subject}" / f"ses-{session}" / "fusi"
for path in fusi_dir.glob("*.nii.gz"):
    print(path.name)

chunk_pattern = f"sub-{subject}_ses-{session}_task-rest_acq-{acq}_probe-linear_chunk-{{chunk}}_pwd.nii.gz"
multipose = cf.load(fusi_dir / chunk_pattern)
multipose

# %% [markdown]
# !!! question "What if my NIfTI files don't have a pose index in their filename?"
#     Load the files individually and combine them with
#     [`stack_poses`][confusius.multipose.stack_poses], listing them in the desired
#     pose order. No particular filename convention is required:
#
#     ```python
#     paths = ["left_pwd.nii.gz", "center_pwd.nii.gz", "right_pwd.nii.gz"]
#     poses = [cf.load(path) for path in paths]
#     multipose = cf.multipose.stack_poses(poses)
#     ```
#
#     Each input must be a compatible single-pose VoxelData array with the same
#     time length and voxel grid. [`stack_poses`][confusius.multipose.stack_poses]
#     preserves each pose's geometry and, when timestamps differ, its `(time, pose)`
#     time coordinate. By default, pose labels are `0, 1, ...`; pass `pose=[...]` to
#     provide your own labels.

# %% [markdown]
# Looking at each pose's origin confirms they sit at different physical positions along
# the elevation axis (`z`), stepped by about 1 mm. Their time origins gives us their
# offset start times.

# %%
origins = [multipose.isel(pose=p).fusi.origin for p in range(multipose.sizes["pose"])]
for dim in ("z", "y", "x", "time"):
    print(f"{dim} origins:", tuple(round(origin[dim], 2) for origin in origins))

# %% [markdown]
# ## Pose-transparent vs. pose-specific geometry
#
# Some geometry queries are well-defined without picking a pose. For example, voxel
# spacing must be identical across poses so
# [`.fusi.spacing`][confusius.xarray.FUSIAccessor.spacing] works directly on the
# multi-pose array.
#
# Origin and direction, however, are inherently single-grid concepts, so they require
# selecting a scalar pose first.
#
# !!! warning "World-coordinate `.sel()` requires a scalar pose"
#     `.sel(z=..., y=..., x=...)` raises `ValueError` on pose-dependent data unless
#     `pose` was already reduced to a scalar, e.g. `.isel(pose=0).sel(z=..., y=...,
#     x=...)`. Resolving a world coordinate to a voxel means picking one pose first;
#     ConfUSIus never resamples poses onto a shared grid implicitly.

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
#     also raises `ValueError` on pose-dependent data, since display is always in
#     world space. Slicing along `pose` itself works fine, since it doesn't require
#     picking one affine.

# %% tags=["thumbnail"]
mean_db = multipose.mean("time").fusi.scale.db().isel(k=0)

plotter = cf.plotting.plot_volume(
    mean_db,
    slice_mode="pose",
    cbar_label="Power Doppler (dB)",
    bg_color=bg_color,
    yincrease=True,
)

# %% [markdown]
# ## Save back to per-pose NIfTI files
#
# Use the same `{chunk}` placeholder with
# [`confusius.save`][confusius.save] to write one NIfTI file and
# JSON sidecar per pose, preserving each pose's geometry and timing. Chunk indices come
# from the `pose` labels, which must be unique non-negative integers.
#
# This demonstration saves only the first ten frames to a temporary directory to keep
# the example quick and avoid leaving files behind.

# %%
with TemporaryDirectory() as output_dir:
    output_pattern = Path(output_dir) / chunk_pattern
    multipose.isel(time=slice(0, 10)).fusi.save(output_pattern)

# %% [markdown]
# ## Consolidate poses into a single volume
#
# Since every pose here shares the same rotation and is offset by a pure translation
# along elevation, [`consolidate_poses`][confusius.multipose.consolidate_poses] can
# merge `pose` and the swept voxel dimension (`k`, the default) into one physically
# ordered axis. This results in a single 3D+t recording with `npose` elevation slices,
# and a single voxel-to-world affine describing the entire volume.

# %%
consolidated = cf.multipose.consolidate_poses(multipose)
consolidated

# %% [markdown]
# The consolidated volume can now be plotted with `slice_mode="z"` to show the elevation
# slices in their physical order.

# %%
plotter = cf.plotting.plot_volume(
    consolidated.mean("time").fusi.scale.db(),
    slice_mode="z",
    cmap="gray",
    cbar_label="Power Doppler (dB)",
    bg_color=bg_color,
    yincrease=True,
)

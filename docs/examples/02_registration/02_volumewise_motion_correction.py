# %% [markdown]
# # Motion correction of a single recording
#
# This example shows how to correct frame-to-frame brain motion in one fUSI recording
# with [`register_volumewise`][confusius.registration.register_volumewise]. We reuse
# the exact acquisition and rigid-registration settings from the
# volumewise-registration GIF in the GUI guide: a short open-field excerpt from the
# [Cybis Pereira 2026 dataset](https://doi.org/10.1016/j.celrep.2025.116791). We then
# inspect three things that are useful in practice:
#
# - the motion diagnostics returned by
#   [`create_motion_dataframe`][confusius.registration.create_motion_dataframe];
# - a representative voxel trace before and after registration;
# - a compact raster view showing how one image column stabilizes over time.
#
# As in the GUI GIF, we register a 120-frame excerpt rather than the full recording.

# %% [markdown]
# ## Fetch and load a short motion-corrupted window
#
# This is the same open-field recording used by the GUI registration demo. The selected
# acquisition is a single 2D slice, so the data shape is `(time, z=1, y, x)`.


# %%
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import confusius as cf

bg_color = mpl.colors.to_hex(mpl.rcParams["figure.facecolor"])
xr.set_options(display_expand_data=False)

subject = "rat75"
session = "20220523"
acq = "slice32"
start_frame = 220
n_frames = 120

bids_root = cf.datasets.fetch_cybis_pereira_2026(
    datasets="rawdata",
    subjects=subject,
    sessions=session,
    acqs=acq,
)

# %%
pwd_path = (
    Path(bids_root)
    / f"sub-{subject}"
    / f"ses-{session}"
    / "fusi"
    / f"sub-{subject}_ses-{session}_task-openfield_acq-{acq}_pwd.nii.gz"
)

data = cf.load(pwd_path).isel(time=slice(start_frame, start_frame + n_frames)).compute()
data

# %% [markdown]
# ## Register every frame to the middle frame
#
# Registering to a central reference frame is a simple way to avoid anchoring the whole
# excerpt to one edge of the motion trajectory. Here we match the GUI demo settings:
# rigid transform, correlation metric, and a fixed `learning_rate=1.0`.

# %%
registered = cf.registration.register_volumewise(
    data,
    transform="rigid",
    metric="correlation",
    learning_rate=1.0,
    resample_interpolation="bspline",
    show_progress=False,
)

motion_df = registered.attrs["motion_params"]
motion_df.head()

# %% [markdown]
# `motion_df` is the output of
# [`create_motion_dataframe`][confusius.registration.create_motion_dataframe]. Because
# this recording has a singleton `z` axis, it is summarized as effective 2D motion:
# one in-plane `rotation`, in-plane translations (`trans_x`, `trans_y`), the
# framewise displacement summaries (`mean_fd`, `max_fd`, `rms_fd`), and the optimizer
# summaries (`final_metric_value`, `n_iterations`) added by
# [`register_volumewise`][confusius.registration.register_volumewise].

# %% [markdown]
# ## Plot the motion diagnostics
#
# The framewise displacement peak marks where the excerpt moves most strongly. The last
# panel is a useful sanity check: frames that systematically hit the maximum iteration
# count or converge to a much worse similarity metric deserve a closer look.

# %% tags=["thumbnail"]
fig, axes = cf.plotting.plot_motion_diagnostics(motion_df)
fig.patch.set_facecolor(bg_color)

# %% [markdown]
# ## Compare a representative voxel before and after registration
#
# To make the effect easy to see, we pick the voxel with the largest temporal standard
# deviation in the unregistered excerpt. In practice that usually lands on a large
# vessel, where motion-induced intensity changes are most visible.

# %%
std_map = data.squeeze("z", drop=True).std("time")
voxel_y, voxel_x = np.unravel_index(np.nanargmax(std_map.values), std_map.shape)

voxel_before = data.isel(z=0, y=voxel_y, x=voxel_x)
voxel_after = registered.isel(z=0, y=voxel_y, x=voxel_x)

fig, ax = plt.subplots(figsize=(9, 3.5), constrained_layout=True)
fig.patch.set_facecolor(bg_color)
ax.plot(voxel_before["time"], voxel_before, label="Before", lw=1.6, alpha=0.8)
ax.plot(voxel_after["time"], voxel_after, label="After", lw=1.6)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Power Doppler intensity")
ax.set_title(f"Voxel at y={voxel_y}, x={voxel_x}")
_ = ax.legend(frameon=False)

# %% [markdown]
# ## Check the alignment over time
#
# A compact way to inspect the correction inside the notebook is to follow one image
# column across time. Motion appears as slanted or wobbling vessel traces in this
# `y × time` raster, while a good correction makes those traces more horizontal and
# stable.

# %%
raster_before = data.fusi.scale.db().isel(z=0, x=voxel_x)
raster_after = registered.fusi.scale.db().isel(z=0, x=voxel_x)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True, sharey=True)
fig.patch.set_facecolor(bg_color)

vmin = float(np.nanpercentile(raster_before, 2))
vmax = float(np.nanpercentile(raster_before, 99.8))

for ax, raster, title in [
    (axes[0], raster_before, "Before motion correction"),
    (axes[1], raster_after, "After motion correction"),
]:
    ax.imshow(
        raster.T,
        aspect="auto",
        origin="upper",
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
        extent=[
            float(raster["time"].values[0]),
            float(raster["time"].values[-1]),
            float(raster["y"].values[0]),
            float(raster["y"].values[-1]),
        ],
    )
    ax.set_title(title)
    ax.set_xlabel("Time (s)")

_ = axes[0].set_ylabel("y (mm)")

# %% [markdown]
# For a full visual check of the corrected movie, open the result in napari with
# [`plot_napari`][confusius.plotting.plot_napari] or inspect it in the ConfUSIus
# plugin. We intentionally do not embed a full before/after GIF here because it would
# make the rendered notebook unnecessarily heavy for the docs site. For a full
# preprocessing workflow, you would usually run the same correction on the complete
# recording before downstream QC, decomposition, or connectivity analysis.

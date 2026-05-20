# %% [markdown]
# # ConfUSIus and Xarray 101
#
# This example demonstrates how to use ConfUSIus to load and handle fUSI data as
# [DataArray][xarray.DataArray] instances. We will use a small subset of the
# Nunez-Elizalde 2022 dataset and use a few basic Xarray operations to inspect, subset,
# and summarize the data.

# %% [markdown]
# ## Fetch one recording from the Nunez-Elizalde 2022 dataset
#
# ConfUSIus provides convenient functions to download public datasets. The data is
# cached after the first download, so subsequent runs will be faster. The
# [`fetch_nunez_elizalde_2022`][confusius.datasets.fetch_nunez_elizalde_2022] function
# allows you to specify which subjects, sessions, tasks, and acquisitions to download.
# Here we select one recording from subject `CR022`, session `20201011`, task
# `spontaneous`, and acquisition `slice03`.

# %%
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import xarray as xr

import confusius as cf
from confusius.datasets import fetch_nunez_elizalde_2022

# Adapt background color to the current Matplotlib style.
bg_color = mpl.colors.to_hex(mpl.rcParams["figure.facecolor"])

# Keep notebook output compact for large DataArray displays.
xr.set_options(display_expand_data=False)

bids_root = fetch_nunez_elizalde_2022(
    subjects="CR022",
    sessions="20201011",
    tasks="spontaneous",
    acqs="slice03",
)

# %% [markdown]
# ## Load the recording as a DataArray
#
# The downloaded dataset is stored following the [Brain Imaging Data Structure
# (BIDS)](https://bids.neuroimaging.io/), so we can use the `bids_root` path to
# construct the path to the desired recording. The [`confusius.load`][confusius.load]
# function reads any compatible fUSI file (NIfTI, Iconeus SCAN, Zarr, etc.) and returns
# a [DataArray][xarray.DataArray] with the data values, coordinates, and metadata all in
# one instance. In the recording representation below, note the named dimensions,
# physical coordinates, and representative acquisition attributes attached to the array.

# %%
pwd_path = (
    Path(bids_root)
    / "sub-CR022"
    / "ses-20201011"
    / "fusi"
    / "sub-CR022_ses-20201011_task-spontaneous_acq-slice03_pwd.nii.gz"
)
data = cf.load(pwd_path)
data

# %% [markdown]
# ConfUSIus also registers a [`.fusi`][confusius.xarray.FUSIAccessor] accessor on every
# DataArray, which exposes fUSI-specific helpers and methods. For example, we can
# inspect the sampling step along each dimension. Note that for this recording, the
# temporal step isn't defined because the time coordinate isn't regularly sampled. See
# the [Working with Xarray](../../../user-guide/xarray.md) guide for a broader overview
# of the accessor.

# %%
data.fusi.spacing

# %% [markdown]
# ## Lazy loading and when to compute
#
# ConfUSIus loads recordings lazily, so the underlying array starts out as a Dask array.
# This is convenient for working with large datasets that don't fit in memory, such as
# large beamformed IQ data. However, lazily loaded arrays may be chunked in ways that
# are not ideal for all operations. For example, chunking across the time dimension is
# ideal for processing volume-by-volume (such as when visualizing a recording in
# napari), but it will make temporal operations (e.g., averaging over time, frequency
# filtering, etc.) less efficient. Thus, when working with smaller recordings that do
# fit in memory, it is often simpler to call [`.compute`][xarray.DataArray.compute]
# right after loading to get a regular in-memory array. This is not required, but it is
# recommended to make subsequent operations more straightforward.

# %%
data = data.compute()

# %% [markdown]
# ## Positional and coordinate-based indexing
#
# Once the recording is in memory, we can keep using normal Xarray operations. A useful
# distinction is between [`.isel`][xarray.DataArray.isel], which indexes by integer
# position, and [`.sel`][xarray.DataArray.sel], which indexes by coordinate values.

# %%
first_50_volumes = data.isel(time=slice(0, 50))
first_50_volumes

# %% [markdown]
# Here we define a region of interest (ROI) in physical coordinates. This is often more
# meaningful than index-based slicing because the bounds are expressed directly in
# physical units (e.g., millimeters) rather than in terms of array indices, which may
# not be as intuitive to interpret.

# %%
roi = data.sel(y=slice(3.5, 7.5), x=slice(-2.0, 2.0))
roi

# %% [markdown]
# ## Plot a regional time course
#
# To summarize the ROI over time, we average over the spatial dimensions. With Xarray,
# reductions such as [`.mean`][xarray.DataArray.mean] take dimension names like
# `("z", "y", "x")` rather than integer axis indices, which makes the intent much
# clearer than the NumPy-style `axis=(...)` equivalent.

# %%
roi_trace = roi.mean(("z", "y", "x"))
roi_trace

# %% [markdown]
# The resulting 1D DataArray keeps the time coordinate from the original recording, so
# plotting it with Xarray's built-in plotting function automatically uses the time
# values on the *x*-axis. As one slightly more advanced example, we can also compute a
# rolling mean with Xarray to smooth short-term fluctuations, while still keeping the
# same labeled time axis.

# %%
roi_trace_smooth = roi_trace.rolling(time=15, center=True).mean()

fig, ax = plt.subplots(figsize=(7, 3), facecolor=bg_color)

roi_trace.plot(ax=ax, color="#d93a54", label="ROI mean")
roi_trace_smooth.plot(ax=ax, color="#3ad9a4", label="Rolling mean (15 samples)")

ax.set_title("Mean power Doppler intensity in a central ROI")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Power Doppler intensity (a.u.)")
_ = ax.legend(loc="upper right")

# %% [markdown]
# ## Compute and plot the mean power Doppler volume
#
# This recording has a single brain slice, so averaging over time leaves a 2D image. We
# convert it to decibel-scale and display it using ConfUSIus's
# [`plot_volume`][confusius.plotting.plot_volume] and
# [`.fusi.scale`][confusius.xarray.FUSIScaleAccessor] accessors.

# %% tags=["thumbnail"]
mean_db = data.mean("time").fusi.scale.db()

plotter = cf.plotting.plot_volume(
    mean_db, cmap="gray", cbar_label="Power Doppler (dB)", bg_color=bg_color
)

# %% [markdown]
# ## Mask low-intensity pixels with `where`
#
# Xarray's [`.where`][xarray.DataArray.where] keeps the same coordinates and dimensions
# while masking values that do not satisfy a condition. Here we suppress low-intensity
# pixels in the mean image before plotting it again.

# %%
mean_db_masked = mean_db.where(mean_db > -20)

plotter = mean_db_masked.fusi.plot.volume(
    cmap="gray", cbar_label="Power Doppler (dB)", bg_color=bg_color
)

# %% [markdown]
# ## Save a processed result
#
# ConfUSIus can also save DataArrays back to disk through the
# [`.fusi.save`][confusius.xarray.FUSIAccessor.save] method. For example, we can save
# the masked mean image as a NIfTI file.

# %%
mean_db_masked.fusi.save(
    "sub-CR022_ses-20201011_task-spontaneous_acq-slice03_desc-maskedmean_pwd.nii.gz"
)

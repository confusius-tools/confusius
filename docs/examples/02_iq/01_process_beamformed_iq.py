# %% [markdown]
# # Processing beamformed IQ recordings
#
# This example shows how to compute B-mode, power Doppler, and axial velocity from
# beamformed in-phase and quadrature (IQ) data. We use a short segment of an awake mouse
# recording from [Landemard et al. (2026)](https://doi.org/10.1038/s41586-026-10350-9)
# and compare the temporal-average maps, as in the ConfUSIus paper's IQ panel.

# %%
import dask
import matplotlib as mpl
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits

import confusius as cf

bg_color = mpl.colors.to_hex(mpl.rcParams["figure.facecolor"])

# %% [markdown]
# ## Load an IQ recording lazily
#
# [`fetch_sample_beamformed_iq`][confusius.datasets.fetch_sample_beamformed_iq] opens
# the recording lazily from the
# [ConfUSIus Dataset Collection](../../../user-guide/datasets.md). The IQ array is
# backed by Dask: opening it reads metadata and coordinates, but not the IQ samples.

# %%
iq = cf.datasets.fetch_sample_beamformed_iq()
iq

# %% [markdown] 
# ## Understanding chunks
#
# Zarr stores data in independently readable chunks. This allows for efficient access to
# subsets of the data. This particular IQ recording is stored in chunks of 300 frames
# along the time dimension. The chunking is visible in the VoxelData array
# representation above: each chunk has shape `(300, 1, 125, 80)` and size ~23 MiB.
#
# Dask breaks operations on the array into tasks that operate on individual chunks.
# These tasks are executed when [`.compute()`][xarray.DataArray.compute] is called.
# Here, our example beamformed IQ recording was chunked into blocks of 300 frames
# because that was the original ensemble size used during acquisition.
#
# !!! tip "Choosing chunk sizes"
#     If the chunk size doesn't match your processing workflow—say you want to process
#     blocks of 600 frames instead of 300—you can [re-chunk][xarray.DataArray.chunk] the
#     array. However, be aware that re-chunking an array does not change the actual
#     chunking on disk. Thus, if you re-chunk an IQ recording to use smaller chunks,
#     each new chunk will require reading the original chunk from disk. If you're
#     responsible for generating the IQ data, choose a chunk size that is appropriate
#     for your processing workflow!
#
# Here, we select the first 900 frames (3 chunks) to demonstrate the processing workflow
# on a small segment of the recording. We download the data to memory with
# [`.compute()`][xarray.DataArray.compute] to accelerate the. In practice, you'll often
# want to process entire IQ recordings lazily without downloading the entire dataset to
# memory.

# %%
iq = iq.isel(time=slice(0, 900)).compute()

# %% [markdown]
# ## Compute B-mode, power Doppler, and axial velocity
#
# Several measurements can be derived from beamformed IQ data. ConfUSIus provides
# convenient functions for computing:
#
# - B-mode: the amplitude of the IQ signal, which is a measure of tissue echogenicity.
# - Power Doppler: the power of the IQ signal after clutter filtering, which is a
#   measure of blood volume.
# - Axial velocity: the velocity of tissue or blood along the ultrasound beam, which is
#   a measure of motion.
#
# For blood measurements, these functions apply an SVD clutter filter to remove tissue
# contributions from the IQ signals. The filtered signal is used to compute power
# Doppler and axial blood velocity. For tissue measurements, we instead retain only the
# tissue signal, which is used to compute axial tissue velocity.

# %%
bmode = cf.iq.process_iq_to_bmode(iq, bmode_window_width=300)
# The low_cutoff/high_cutoff parameters define the cutoff index for the SVD clutter
# filter. This cutoff was chosen manually and may not be optimal for all recordings.
pwd = cf.iq.process_iq_to_power_doppler(iq, clutter_window_width=300, low_cutoff=40)
blood_vel = cf.iq.process_iq_to_axial_velocity(
    iq, clutter_window_width=300, low_cutoff=40
)
tissue_vel = cf.iq.process_iq_to_axial_velocity(
    iq, clutter_window_width=300, high_cutoff=40
)

# Since all four processing run on the same IQ data, we use dask.compute to compute all
# four outputs in parallel. If we had called .compute() on each output separately, Dask
# would have executed four separate computations and needed to download the IQ data four
# times.
with threadpool_limits(limits=1, user_api="blas"):
    bmode, pwd, blood_vel, tissue_vel = dask.compute(
        bmode, pwd, blood_vel, tissue_vel
    )

# %% [markdown]
# !!! tip "Dask, threading, and BLAS"
#     Since all four processing run on the same IQ data, we used
#     [`dask.compute`][dask.compute] to compute all four outputs in parallel. If we had
#     called [`.compute()`][xarray.DataArray.compute] on each output separately, Dask
#     would have executed four separate computations and needed to download the IQ data
#     four times.
#
#     Moreover, Dask tasks and ConfUSIus computations can each use multiple threads.
#     Therefore, it's recommended to limit the number of threads used by BLAS (the
#     underlying linear algebra library used by ConfUSIus) to avoid over-subscription. 
#     See the [IQ processing guide](../../../user-guide/beamformed-iq.md#working-with-large-recordings)
#     for more details.

# %% [markdown]
# ## Plot average maps
#
# For visualization, we average each output over time. For tissue velocity, we take the
# absolute value before averaging to avoid cancellation between opposite directions of
# motion. Blood axial velocity remains signed.
#
# We display the mean B-mode amplitude and power Doppler in relative decibels, and
# convert velocities from m/s to µm/s (tissue) or mm/s (blood).

# %% tags=["thumbnail"]
fig, axes = plt.subplots(2, 2, figsize=(8, 6), constrained_layout=True)
panels = (
    ("B-mode", bmode.mean("time").fusi.scale.db(factor=20), "gray", -60, 0, "dB"),
    ("Tissue velocity", abs(tissue_vel).mean("time") * 1e6, "inferno", 0, 5, "µm/s"),
    ("Power Doppler", pwd.mean("time").fusi.scale.db(), "gray", -20, 0, "dB"),
    ("Blood velocity", blood_vel.mean("time") * 1e3, "coolwarm", -10, 10, "mm/s"),
)
for ax, (title, image, cmap, vmin, vmax, unit) in zip(axes.flat, panels, strict=True):
    cf.plotting.plot_volume(
        image,
        slice_mode="z",
        axes=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        show_colorbar=False,
        show_titles=False,
        bg_color=bg_color,
    )
    ax.set_title(title)
    fig.colorbar(ax.collections[-1], ax=ax, label=unit, shrink=0.65)
plt.show()

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
# Zarr stores data in independently readable chunks, allowing subsets to be loaded
# without downloading the whole recording. Here, each chunk has shape
# `(300, 1, 125, 80)` and size ~23 MiB, as shown above. The 300-frame time chunks
# match the original ensemble size used during acquisition.
#
# Dask breaks processing into tasks that operate on these chunks. The tasks stay
# lazy: IQ samples are downloaded and processed when we call
# [`dask.compute`][dask.compute], not when we select data or set up processing.
#
# !!! tip "Choosing chunk sizes"
#     You can [rechunk][xarray.DataArray.chunk] the array to work with, for example,
#     600-frame blocks instead of 300. This changes Dask's layout, not the stored
#     chunks: smaller Dask chunks may still require reading a full storage chunk.
#     When writing IQ data, choose storage chunks that suit your workflow. Processing
#     window widths can also be set explicitly, independently of the chunking.
#
# We select the first 900 frames (three chunks) for this short demonstration. The
# slice stays lazy and preserves the 300-frame chunks without loading IQ samples.

# %%
iq = iq.isel(time=slice(0, 900))

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
# For blood measurements, an SVD clutter filter removes tissue contributions before
# computing power Doppler and axial blood velocity. For tissue velocity, we instead
# retain the tissue components.
#
# By default, each 300-frame time chunk is one processing window. The windows do
# not overlap, and each call below produces one output volume per window.

# %%
bmode = cf.iq.process_iq_to_bmode(iq)
# The low_cutoff/high_cutoff parameters define the cutoff index for the SVD clutter
# filter. This cutoff was chosen manually and may not be optimal for all recordings.
pwd = cf.iq.process_iq_to_power_doppler(iq, low_cutoff=40)
blood_vel = cf.iq.process_iq_to_axial_velocity(iq, low_cutoff=40)
tissue_vel = cf.iq.process_iq_to_axial_velocity(iq, high_cutoff=40)

with threadpool_limits(limits=1, user_api="blas"):
    bmode, pwd, blood_vel, tissue_vel = dask.compute(bmode, pwd, blood_vel, tissue_vel)

# %% [markdown]
# !!! tip "Dask, threading, and BLAS"
#     The processing calls above only build lazy computations. A single
#     [`dask.compute`][dask.compute] executes all four together, letting them share
#     IQ reads. Computing each output separately may download the same data again.
#
#     BLAS, the linear algebra library used by SVD, can use multiple threads. Dask
#     also runs tasks in parallel, so we limit BLAS to one thread to avoid too many
#     competing threads slowing processing down. See the
#     [IQ processing guide](../../../user-guide/beamformed-iq.md#working-with-large-recordings)
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

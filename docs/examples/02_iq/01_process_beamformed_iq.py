# %% [markdown]
# # Processing beamformed IQ recordings
#
# This example shows how to compute B-mode, power Doppler, and axial velocity from
# beamformed in-phase and quadrature (IQ) data. We use a short segment of an awake
# mouse recording from [Landemard et al. (2026)](https://doi.org/10.1038/s41586-026-10350-9)
# and compare the temporal-average maps, as in the ConfUSIus paper's IQ panel.

# %%
import dask
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from threadpoolctl import threadpool_limits

import confusius as cf

bg_color = mpl.colors.to_hex(mpl.rcParams["figure.facecolor"])

# %% [markdown]
# ## Load an IQ recording
#
# [`fetch_sample_beamformed_iq`][confusius.datasets.fetch_sample_beamformed_iq] opens
# the recording lazily from the public S3 collection. The IQ array is backed by
# Dask: opening it reads metadata and coordinates, but not the IQ samples.

# %%
iq = cf.datasets.fetch_sample_beamformed_iq()
print(iq)

# %% [markdown]
# ## Work with chunks
#
# Zarr stores the IQ samples in independently readable chunks of 300 frames. Dask
# uses chunks to build a lazy task graph: selecting a segment does not read its IQ
# data until we compute it. We load three chunks (about 72 MB uncompressed) into
# memory so all four calculations reuse the data without downloading it again.
#
# Storage chunks and processing windows are distinct. Below, we explicitly use
# 300-frame windows for B-mode and clutter filtering; loading into memory with
# `compute()` does not preserve Dask chunking.

# %%
iq = iq.isel(time=slice(0, 900)).compute()

# %% [markdown]
# Dask tasks and BLAS linear algebra can each use multiple threads. For this small
# example, we limit BLAS to one thread during computation to avoid competing thread
# pools. For larger workloads, a `dask.distributed.Client` offers worker management
# and a dashboard; see the [IQ processing guide](../../../user-guide/beamformed-iq.md#memory-and-performance-optimization).
# Performance depends on the workload and worker configuration.

# %% [markdown]
# ## Compute B-mode, power Doppler, and axial velocity
#
# B-mode represents the IQ amplitude, while power Doppler measures signal power
# after clutter filtering. With the default SVD-index filter, `low_cutoff=40` removes
# the first 40 modes to suppress tissue clutter; `high_cutoff=40` retains these modes
# to estimate tissue motion. These cutoffs are specific to this recording.
#
# We average each output over time. For tissue velocity, we take the absolute value
# before averaging to avoid cancellation between opposite directions of motion.
# Blood axial velocity remains signed.

# %%
bmode = cf.iq.process_iq_to_bmode(iq, bmode_window_width=300).mean("time")
pwd = cf.iq.process_iq_to_power_doppler(
    iq, clutter_window_width=300, low_cutoff=40
).mean("time")
blood_velocity = cf.iq.process_iq_to_axial_velocity(
    iq, clutter_window_width=300, low_cutoff=40, spatial_kernel=1
).mean("time")
tissue_velocity = abs(
    cf.iq.process_iq_to_axial_velocity(iq, clutter_window_width=300, high_cutoff=40)
).mean("time")
with threadpool_limits(limits=1, user_api="blas"):
    bmode, pwd, blood_velocity, tissue_velocity = dask.compute(
        bmode, pwd, blood_velocity, tissue_velocity, scheduler="threads"
    )
assert all(
    np.isfinite(image.values).all()
    for image in (bmode, pwd, blood_velocity, tissue_velocity)
)

# %% [markdown]
# ## Plot average maps
#
# We display the mean B-mode amplitude and power Doppler in relative decibels, and
# convert velocities from m/s to µm/s (tissue) or mm/s (blood). The signed blood
# velocity map shows motion along the ultrasound beam, not the total flow speed.

# %% tags=["thumbnail"]
fig, axes = plt.subplots(2, 2, figsize=(8, 6), constrained_layout=True)
panels = (
    ("B-mode", bmode.fusi.scale.db(factor=20), "gray", -60, 0, "dB"),
    ("Tissue velocity", tissue_velocity * 1e6, "inferno", 0, 5, "µm/s"),
    ("Power Doppler", pwd.fusi.scale.db(), "gray", -20, 0, "dB"),
    ("Blood axial velocity", blood_velocity * 1e3, "coolwarm", -10, 10, "mm/s"),
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

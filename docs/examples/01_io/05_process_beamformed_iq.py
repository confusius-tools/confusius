# %% [markdown]
# # From beamformed IQ to ultrasound images
#
# Compute B-mode, power Doppler, and axial velocity from a few blocks of the
# [Landemard et al. (2026)](https://doi.org/10.1038/s41586-026-10350-9) sample.
# The four temporal-average maps follow the ConfUSIus paper's IQ panel, using a
# much shorter segment. The recording is licensed under CC BY-NC 4.0.

# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import confusius as cf

bg_color = mpl.colors.to_hex(mpl.rcParams["figure.facecolor"])

# %% [markdown]
# ## Read three blocks
#
# The fetcher opens the remote Zarr store lazily. Select three 300-frame chunks
# before computing, so only this segment (about 72 MB uncompressed) enters memory.
# Keep its chunking for the processing windows and reuse the in-memory IQ data
# across all four calculations rather than downloading it repeatedly.

# %%
iq = cf.datasets.fetch_sample_beamformed_iq().isel(time=slice(0, 900))
iq = iq.compute().chunk({"time": 300})

# %% [markdown]
# ## Compute temporal averages
#
# With the default SVD-index filter, `low_cutoff=40` removes the first 40 modes
# to emphasize blood signals. Conversely, `high_cutoff=40` retains the tissue
# modes. As in the paper, average the absolute tissue velocity to avoid cancellation;
# blood axial velocity remains signed. These cutoffs are recording-specific.

# %%
bmode = cf.iq.process_iq_to_bmode(iq).mean("time").compute()
power = cf.iq.process_iq_to_power_doppler(iq, low_cutoff=40).mean("time").compute()
blood_velocity = (
    cf.iq.process_iq_to_axial_velocity(iq, low_cutoff=40, spatial_kernel=1)
    .mean("time")
    .compute()
)
tissue_velocity = (
    abs(cf.iq.process_iq_to_axial_velocity(iq, high_cutoff=40)).mean("time").compute()
)
assert all(
    np.isfinite(image.values).all()
    for image in (bmode, power, blood_velocity, tissue_velocity)
)

# %% [markdown]
# ## Compare the maps
#
# Convert the mean B-mode amplitude and power Doppler to relative decibels, and
# velocities from m/s to µm/s (tissue) or mm/s (blood). The signed blood map shows
# the direction of motion along the ultrasound beam, not total flow speed.

# %% tags=["thumbnail"]
fig, axes = plt.subplots(2, 2, figsize=(8, 6), constrained_layout=True)
panels = (
    ("B-mode", bmode.fusi.scale.db(factor=20), "gray", -60, 0, "dB"),
    ("Tissue velocity", tissue_velocity * 1e6, "inferno", 0, 5, "µm/s"),
    ("Power Doppler", power.fusi.scale.db(), "gray", -20, 0, "dB"),
    ("Blood axial velocity", blood_velocity * 1e3, "berlin", -10, 10, "mm/s"),
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

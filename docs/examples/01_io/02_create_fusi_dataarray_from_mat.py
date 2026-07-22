# # Create a fUSI DataArray from a MAT file
#
# This example downloads a power Doppler MAT file from the public dataset accompanying
# [Rabut *et al.* (2024)](https://doi.org/10.1126/scitranslmed.adj3143)[^1] and wraps it
# in a ConfUSIus [DataArray][xarray.DataArray] with
# [`create_fusi_dataarray`][confusius.xarray.create_fusi_dataarray].
#
# Since MAT files can contain any custom data, ConfUSIus cannot read them directly. The
# point of this example is to show the shortest path from a lab-specific MAT file array
# plus metadata to the standard ConfUSIus DataArray representation. This example then
# reproduces figure 4D from Rabut *et al.* (2024) using a simple general linear model.

# %%
from functools import partial
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pooch
import xarray as xr
from remotezip import RemoteZip

import confusius as cf
from confusius.glm import FirstLevelModel, gamma_hrf

# Adapt colors to the current Matplotlib style.
bg_color = mpl.colors.to_hex(mpl.rcParams["figure.facecolor"])
is_dark_theme = sum(mpl.colors.to_rgb(bg_color)) / 3 < 0.5
stat_cmap = "berlin" if is_dark_theme else None

# Keep notebook output compact for large DataArray displays.
_ = xr.set_options(display_expand_attrs=False, display_expand_data=False)

# %% [markdown]
# ## Download one recording from the Rabut *et al.* (2024) dataset
#
# The Caltech record stores all data in a single ZIP archive. We extract a single MAT file
# from the archive to avoid downloading the full dataset. The MAT file contains a power
# Doppler recording of a human subject performing a simple task, with accompanying
# timestamps, task labels, and ultrasound metadata.

# %%
RECORD_URL = "https://data.caltech.edu/api/records/f3y3k-em558/files/data.zip/content"
RECORD_DOI = "https://doi.org/10.22002/f3y3k-em558"
MEMBER = "data/human/S2R1.mat"

cache_dir = Path(pooch.os_cache("confusius")) / "rabut_2024_human_glm"
cache_dir.mkdir(parents=True, exist_ok=True)
mat_path = cache_dir / Path(MEMBER).name

if not mat_path.exists():
    with RemoteZip(RECORD_URL) as archive:
        with archive.open(MEMBER) as source, mat_path.open("wb") as target:
            target.write(source.read())

# %% [markdown]
# ## Load the power Doppler array and metadata
#
# This MAT file v7.3 is HDF5-backed, so we use [`h5py.File`][h5py.File] to open it.

# %%
with h5py.File(mat_path, "r") as mat:
    doppler = mat["dop"][()].astype("float32").transpose(0, 2, 1)
    task = mat["task"][:, 0].astype("float32")
    timestamps = mat["timestamps"][0]
    wavelength = float(mat["UF/Lambda"][0, 0])
    run_label = "".join(map(chr, mat["run_label"][()].ravel().astype(int))).strip()

print(run_label)
print(f"Data shape: {doppler.shape}")

# %% [markdown]
# ## Wrap the raw array with [`create_fusi_dataarray`][cf.create_fusi_dataarray]
#
# After transposing the MATLAB array, the Doppler movie is `(time, y, x)`. ConfUSIus
# adds the missing singleton `z` axis and returns the canonical `(time, z, y, x)`
# layout. The timestamps have small acquisition jitter, so we pass them as an exact
# coordinate. Following the authors' analysis code, we use the acoustic wavelength from
# `UF.Lambda` for axial spacing and the 0.3 mm probe pitch for lateral spacing. We keep
# depths between 10 and 30 mm to focus on the part of the image with good SNR.

# %%
dy = wavelength
dx = 0.3
y = dy / 2 + np.arange(doppler.shape[1]) * dy
x = dx / 2 + np.arange(doppler.shape[2]) * dx

power_doppler = cf.create_fusi_dataarray(
    doppler,
    dims=("time", "y", "x"),
    coords={"time": timestamps, "y": y, "x": x},
    dz=1.0,
    name="power_doppler",
    attrs={"source": RECORD_DOI, "source_member": MEMBER},
).sel(y=slice(10, 30))

time_step = float(np.median(np.diff(timestamps)))
power_doppler = cf.timing.resample_to_uniform_time(power_doppler, step=time_step)
task_da = xr.DataArray(task, dims="time", coords={"time": timestamps})
task_da = cf.timing.resample_to_uniform_time(
    task_da,
    start=float(power_doppler.time[0]),
    stop=float(power_doppler.time[-1]),
    step=time_step,
    method="nearest",
)
power_doppler

# %% [markdown]
# ## Plot the mean power Doppler image
#
# Once the custom file is represented as a DataArray, ConfUSIus plotting helpers work
# the same way as they do for built-in loaders.

# %% tags=["thumbnail"]
mean_doppler = power_doppler.mean("time").fusi.scale.db()
plotter = mean_doppler.fusi.plot.volume(
    cbar_label="Power Doppler (dB)",
    bg_color=bg_color,
)

# %% [markdown]
# ## Motion-correct, smooth, and fit a simple task GLM
#
# The paper reports rigid-body motion correction, 2D Gaussian spatial smoothing,
# temporal smoothing, detrending, and baseline scaling before the GLM. We reproduce all
# of these steps with ConfUSIus below.

# %%
# The default learning rate is conservative; for this recording, 1.0 recovers the motion
# better.
registered = power_doppler.fusi.register.volumewise(learning_rate=1.0)

# The paper reports a 2D Gaussian smoothing kernel with FWHM = 0.471 mm in both
# spatial dimensions.
smoothed = cf.spatial.smooth_volume(registered, fwhm=0.471)

# The paper reports a simple moving average for temporal smoothing.
filtered = smoothed.rolling(time=6, min_periods=1).mean()

# The paper reports baseline scaling to percent signal change relative to the mean of
# the rest blocks.
baseline = filtered.where(task_da == 0).mean("time")
scaled = 100 * filtered / baseline

time_values = power_doppler.time.values
edges = np.diff(np.r_[0.0, task_da.values, 0.0])
starts = np.flatnonzero(edges == 1.0)
stops = np.flatnonzero(edges == -1.0)
stop_times = np.r_[time_values, time_values[-1] + time_step][stops]
events = pd.DataFrame(
    {
        "onset": time_values[starts],
        "duration": stop_times - time_values[starts],
        "trial_type": "task",
    }
)


# We handle slow drift in FirstLevelModel with a cosine drift model. The task regressor
# is convolved with the single-gamma human fUSI HRF reported in the paper (`τ = 0.7`, `δ
# = 3 s`, `n = 3`).
tau = 0.7
n = 3
human_fusi_hrf = partial(
    gamma_hrf,
    time_length=16.0,
    peak_delay=(n - 1) * tau,
    dispersion=tau,
    onset=3.0,
)

model = FirstLevelModel(
    hrf_model=human_fusi_hrf, noise_model="ols", drift_model="cosine", low_cutoff=0.002
)
model.fit(scaled, events=events)

z_map = model.compute_contrast("task")
z_map

# %% [markdown]
# ## Plot the GLM map
#
# The z-map can be overlaid on the mean Doppler image with ConfUSIus's statistical map
# plotting helper. This is a lightweight reproduction of the Fig. 4D GLM map; here we
# threshold at the top 5% of positive task z-scores.

# %%
active_threshold = float(z_map.quantile(0.95))
active_voxels = z_map > active_threshold

plotter = z_map.fusi.plot.stat_map(
    bg_volume=mean_doppler,
    threshold=active_threshold,
    cmap=stat_cmap,
    cbar_label="task z-score",
    bg_color=bg_color,
)

# %% [markdown]
# ## Plot the strongest task-positive voxels
#
# We average the Doppler traces from those task-positive voxels. This gives a quick
# sanity check that the voxels highlighted by the GLM follow the task blocks more
# clearly than the whole-plane average.

# %%
active_signal = scaled.where(active_voxels).mean(("z", "y", "x"))
mean_signal = scaled.mean(("z", "y", "x"))
fig, ax = plt.subplots(figsize=(9, 4), facecolor=bg_color)
mean_signal.plot(ax=ax, color="#808080", label="Whole-plane mean")
active_signal.plot(ax=ax, color="#d93a54", label="Top task-positive voxels")
for idx, event in enumerate(events.itertuples()):
    label = "Task on" if idx == 0 else None
    ax.axvspan(
        event.onset,
        event.onset + event.duration,
        color="#3ad9a4",
        alpha=0.12,
        label=label,
    )
ax.set_title("Task-positive voxel time course")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Signal (% rest baseline)")

_ = ax.legend(loc="upper left")

# %% [markdown]
# [^1]: Rabut, Claire, et al. “Functional Ultrasound Imaging of Human Brain Activity
#       through an Acoustically Transparent Cranial Window.” *Science Translational
#       Medicine*, vol. 16, no. 749, May 2024, p. eadj3143. DOI.org (Crossref),
#       <https://doi.org/10.1126/scitranslmed.adj3143>.

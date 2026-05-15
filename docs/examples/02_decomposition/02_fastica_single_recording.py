# %% [markdown]
# # FastICA on a single fUSI recording
#
# This example shows how to use [FastICA][confusius.decomposition.FastICA] to decompose a
# fUSI recording into spatially independent components.
#
# [PCA](pca_single_recording.md) decomposes the data into orthogonal axes of maximum
# variance: the spatial maps are orthogonal and the resulting time courses are
# uncorrelated. FastICA goes further and searches for components that are statistically
# *independent*, a strictly stronger condition than mere decorrelation. In practice, each
# independent component tends to concentrate its spatial weight into a more localised,
# non-Gaussian pattern, which can separate sources that PCA would blend into a single
# axis of shared variance.
#
# We use the same spontaneous activity recording from the
# [Nunez-Elizalde 2022 dataset](https://doi.org/10.1016/j.neuron.2022.02.012) as in the
# [PCA example](pca_single_recording.md).

# %% [markdown]
# ## Load a fUSI recording
#
# We use a spontaneous activity recording from the
# [Nunez-Elizalde 2022 dataset](https://doi.org/10.1016/j.neuron.2022.02.012).
# See [`fetch_nunez_elizalde_2022`][confusius.datasets.fetch_nunez_elizalde_2022] for
# more details on how to download this dataset using ConfUSIus.

# %%
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import confusius as cf
from confusius.datasets import fetch_nunez_elizalde_2022
from confusius.decomposition import FastICA
from confusius.signal import standardize

# Keep figure backgrounds transparent in docs and standalone notebooks.
bg_color = "none"
# Match text and axes styling to the active Matplotlib theme.
fg_color = mpl.colors.to_hex(plt.rcParams["text.color"])
# Keep notebook output compact for large DataArray displays.
xr.set_options(display_expand_data=False)

bids_root = fetch_nunez_elizalde_2022(
    subjects="CR022",
    sessions="20201011",
    tasks="spontaneous",
    acqs="slice03",
)

pwd_path = (
    Path(bids_root)
    / "sub-CR022"
    / "ses-20201011"
    / "fusi"
    / "sub-CR022_ses-20201011_task-spontaneous_acq-slice03_pwd.nii.gz"
)
data = cf.load(pwd_path).compute()
data

# %% [markdown]
# ## Correct for brain motion
#
# This recording contains some brain motion, which we can mitigate by performing a rigid
# translation correction with
# [`register_volumewise`][confusius.registration.register_volumewise]. This is the same
# preprocessing step used in the [PCA example](pca_single_recording.md), and it helps
# avoid components dominated by motion artefacts.

# %%
data = cf.registration.register_volumewise(
    data, learning_rate=1e-2, show_progress=False
)

# %% [markdown]
# ## Fit a FastICA model
#
# Before fitting FastICA, we standardize the recording by centering and scaling each
# voxel's time series to zero mean and unit variance with
# [`standardize`][confusius.signal.standardize], for the same reasons as in the
# [PCA example](pca_single_recording.md).

# %%
data_std = standardize(data)

# %% [markdown]
# [FastICA][confusius.decomposition.FastICA] shares the same interface as
# [PCA][confusius.decomposition.PCA]: it expects a `(time, ...)` DataArray and exposes
# `fit`, `transform`, and `fit_transform`. With `mode="spatial"` (the default), the data
# is transposed to `(voxels, time)` internally so that ICA maximises independence across
# the *spatial* dimension. The resulting `maps_` are the independent spatial patterns;
# the associated time courses are recovered by projecting the standardised recording onto
# those maps.
#
# Unlike PCA, FastICA does not order its components by explained variance, so the
# component indices carry no intrinsic meaning. We fix `n_components=12` to limit the
# decomposition to a manageable number of patterns, and `random_state=0` for
# reproducibility since the ICA optimisation is sensitive to initialisation.

# %%
ica = FastICA(n_components=12, random_state=0, mode="spatial")
signals = ica.fit_transform(data_std)
signals

# %% [markdown]
# ## Independent spatial maps
#
# [`maps_`][confusius.decomposition.FastICA] is a `(component, y, x)` DataArray.
# Unlike [PCA maps](pca_single_recording.md), these spatial patterns are not constrained
# to be orthogonal — they are instead optimised to be as statistically independent as
# possible. In practice, this tends to produce maps with more spatially focused,
# non-Gaussian structure, often corresponding to individual vascular or functional
# territories.

# %% tags=["thumbnail"]
maps_12 = ica.maps_.isel(component=slice(0, 12))
vmax = float(np.abs(maps_12).max())
plotter = cf.plotting.plot_volume(
    maps_12,
    slice_mode="component",
    cmap="coolwarm",
    vmin=-vmax,
    vmax=vmax,
    ncols=4,
    show_axes=False,
    fontsize=24,
    bg_color=bg_color,
    fg_color=fg_color,
    cbar_label="Component weight",
)
_ = plotter.figure.suptitle("FastICA spatial maps (first 12 components)", fontsize=21)

# %% [markdown]
# ## Independent component time courses
#
# [`fit_transform`][confusius.decomposition.FastICA.fit_transform] returns the projection
# of the recording onto each independent spatial map: a `(time, component)` DataArray.
# Comparing the maps and time courses side by side helps to judge whether each component
# reflects a plausible functional or vascular source, or an artefact such as motion or
# physiological noise.

# %%
n_show = 6
fig = plt.figure(figsize=(10.5, 7.5), constrained_layout=True)
fig.patch.set_facecolor(bg_color)
gs = fig.add_gridspec(n_show, 2, width_ratios=[1, 3])

axes_tc = [fig.add_subplot(gs[i, 1]) for i in range(n_show)]
for ax in axes_tc[1:]:
    ax.sharex(axes_tc[0])

for i, comp in enumerate(range(n_show)):
    component_map = ica.maps_.isel(component=[comp])
    vmax = float(np.abs(component_map).max())
    cf.plotting.plot_volume(
        component_map,
        axes=fig.add_subplot(gs[i, 0]),
        slice_mode="component",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        show_axes=False,
        show_colorbar=False,
        show_titles=False,
        bg_color=bg_color,
        fg_color=fg_color,
    )

    signals.sel(component=comp).plot(ax=axes_tc[i], lw=1.1)
    axes_tc[i].set_title(f"IC {comp + 1}")
    axes_tc[i].set_ylabel("Signal")
    axes_tc[i].set_xlabel("")

for ax in axes_tc[:-1]:
    ax.tick_params(labelbottom=False)
axes_tc[-1].set_xlabel("Time (s)")
_ = fig.suptitle(
    "FastICA spatial maps and time courses (first 6 components)", fontsize=21
)

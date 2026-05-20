# %% [markdown]
# # FastICA on a single fUSI recording
#
# This example shows how to use [FastICA][confusius.decomposition.FastICA] to decompose
# a fUSI recording into independent components.
#
# [PCA](pca_single_recording.md) decomposes fUSI data into spatial maps and time courses
# by finding orthogonal axes that capture dominant covariance/correlation structure.
# The resulting components are uncorrelated but not necessarily independent: they may
# still share higher-order statistical structure.
#
# ICA uses a stronger statistical objective. Instead of diagonalising covariance, it
# searches for components that are as statistically independent as possible, using
# higher-order structure beyond variance and correlation[^1]. The interpretation of an
# ICA component depends on the orientation of the data:
#
# - **Temporal ICA** (`mode="temporal"`): the independent components are time courses
#   and the spatial maps are their voxel-wise mixing weights.
# - **Spatial ICA** (`mode="spatial"`): the independent components are spatial maps and
#   the corresponding time courses are their time-wise mixing weights.
#
# We start with temporal ICA because its orientation is directly comparable to the usual
# PCA decomposition, then contrast it with spatial ICA.
#
# Historically, temporal ICA was less used in resting-state fMRI because acquisitions had
# relatively few time points compared with the number of voxels. Spatial ICA was then
# better conditioned and became the conventional choice. With fUSI (and accelerated fMRI),
# temporal sampling is richer, so both temporal and spatial ICA are practical and provide
# complementary information.

# %% [markdown]
# ## Load a fUSI recording
#
# To demonstrate the use of ICA on fUSI data, we use the same spontaneous activity
# recording from the [Nunez-Elizalde 2022
# dataset](https://doi.org/10.1016/j.neuron.2022.02.012) as in the [PCA
# example](pca_single_recording.md). See the [Datasets](../../../user-guide/datasets.md)
# user guide for more details on how to download this dataset using ConfUSIus.

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
# ## Temporal ICA
#
# Temporal ICA treats time courses as signals and voxels as instances. It is useful for
# separating components that are temporally independent but may overlap in space, and can
# help separate low-frequency physiological fluctuations.
#
# Before fitting, we standardize the recording by centering and scaling each voxel's
# time series to zero mean and unit variance with
# [`standardize`][confusius.signal.standardize], for the same reasons as in the
# [PCA example](pca_single_recording.md).

# %%
data_std = standardize(data)

# %% [markdown]
# With `mode="temporal"`, [FastICA][confusius.decomposition.FastICA] operates on the
# same `(time, voxels)` orientation as [PCA][confusius.decomposition.PCA]. The
# algorithm finds `n_components` time courses that are as statistically independent and
# non-Gaussian as possible. The [`maps_`][confusius.decomposition.FastICA] attribute
# stores the corresponding spatial mixing weights — the voxel-space directions along
# which each independent time course has its strongest influence.

# %%
ica_t = FastICA(
    n_components=10,
    mode="temporal",
    random_state=42,
    fun="cube",
    max_iter=500,
)
signals_t = ica_t.fit_transform(data_std)
signals_t

# %% [markdown]
# Plotting the spatial mixing weights alongside the independent time courses gives a
# first sense of what each component captures.

# %%
n_show = 10
fig = plt.figure(figsize=(14, 20), constrained_layout=True)
fig.patch.set_facecolor(bg_color)
gs = fig.add_gridspec(n_show, 2, width_ratios=[1, 3])

axes_tc = [fig.add_subplot(gs[i, 1]) for i in range(n_show)]
for ax in axes_tc[1:]:
    ax.sharex(axes_tc[0])

for i, comp in enumerate(range(n_show)):
    component_map = ica_t.maps_.isel(component=[comp])
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
    )

    signals_t.sel(component=comp).plot(ax=axes_tc[i], lw=1.1)
    axes_tc[i].set_title(f"IC {comp + 1}")
    axes_tc[i].set_ylabel("Signal")
    axes_tc[i].set_xlabel("")

for ax in axes_tc[:-1]:
    ax.tick_params(labelbottom=False)
axes_tc[-1].set_xlabel("Time (s)")
_ = fig.suptitle(
    "Temporal ICA: mixing weights and time courses (first 10 components)", fontsize=21
)

# %% [markdown]
# ## Spatial ICA
#
# Spatial ICA (`mode="spatial"`, the default) treats spatial voxels as signals and time
# points as instances by transposing data to `(voxels, time)` before fitting. It is the
# conventional resting-state choice because the spatial dimension is usually much larger
# than the temporal one, making decomposition better conditioned. In practice, spatial ICA
# is effective at identifying spatially localized fluctuations and often better at
# reducing whole-brain motion-related structure.

# %%
ica_s = FastICA(
    n_components=10, mode="spatial", random_state=42, fun="cube", max_iter=500
)
signals_s = ica_s.fit_transform(data_std)
signals_s

# %% [markdown]
# ### Spatial maps and time courses
#
# [`maps_`][confusius.decomposition.FastICA] is a `(component, y, x)` DataArray whose
# rows are the independent spatial patterns themselves, in contrast to temporal ICA where
# `maps_` stores mixing weights. Comparing each map with its time course helps judge
# whether a component reflects a plausible functional or vascular source, or an artefact
# such as motion or physiological noise.

# %% tags=["thumbnail"]
n_show = 10
fig = plt.figure(figsize=(14, 20), constrained_layout=True)
fig.patch.set_facecolor(bg_color)
gs = fig.add_gridspec(n_show, 2, width_ratios=[1, 3])

axes_tc = [fig.add_subplot(gs[i, 1]) for i in range(n_show)]
for ax in axes_tc[1:]:
    ax.sharex(axes_tc[0])

for i, comp in enumerate(range(n_show)):
    component_map = ica_s.maps_.isel(component=[comp])
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
    )

    signals_s.sel(component=comp).plot(ax=axes_tc[i], lw=1.1)
    axes_tc[i].set_title(f"IC {comp + 1}")
    axes_tc[i].set_ylabel("Signal")
    axes_tc[i].set_xlabel("")

for ax in axes_tc[:-1]:
    ax.tick_params(labelbottom=False)
axes_tc[-1].set_xlabel("Time (s)")
_ = fig.suptitle("Spatial ICA: maps and time courses (first 10 components)", fontsize=21)

# %% [markdown]
# [^1]: Hyvärinen, A., and Oja, E. (2000). "Independent component analysis:
# algorithms and applications". *Neural Networks*, 13(4-5), 411-430.

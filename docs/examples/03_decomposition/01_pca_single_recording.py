# %% [markdown]
# # PCA on a single fUSI recording
#
# This example shows how to use principal component analysis (PCA) to decompose a fUSI
# recording into principal axes of variance.
#
# ConfUSIus supports two PCA orientations:
#
# - `mode="temporal"` (default): center across time for each voxel, then find orthogonal
#   voxel-space directions that maximize variance of projected time samples.
# - `mode="spatial"`: center across voxels for each time point (after transposition),
#   then find orthogonal spatial maps that maximize variance across voxels.
#
# Both return time courses with shape `(time, component)` via
# [`transform`][confusius.decomposition.PCA.transform], but the statistical objective is
# applied along different dimensions.
#
# PCA finds an ordered orthogonal set of axes in feature space[^1] such that projection
# onto the first *k* axes captures the maximum possible variance among all
# *k*-dimensional linear projections. Equivalently, these axes are the dominant
# eigenvectors of the covariance matrix, or of the correlation matrix if variables are
# properly standardized. If you are interested in linear covariance structure in your
# fUSI data, PCA is a useful place to start.

# %% [markdown]
# ## Load a fUSI recording
#
# To demonstrate the use of PCA on fUSI data, we use a spontaneous activity recording
# from the [Nunez-Elizalde 2022 dataset](https://doi.org/10.1016/j.neuron.2022.02.012).
# See the [Datasets](../../../user-guide/datasets.md) user guide for more details on how
# to download this dataset using ConfUSIus.

# %%
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import confusius as cf

# Adapt background color to the current Matplotlib style.
bg_color = mpl.colors.to_hex(mpl.rcParams["figure.facecolor"])

# Keep notebook output compact for large DataArray displays.
xr.set_options(display_expand_data=False)

bids_root = cf.datasets.fetch_nunez_elizalde_2022(
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
# transform correction with
# [`register_volumewise`][confusius.registration.register_volumewise]. This is a common
# preprocessing step for fUSI data, and it can help avoid spurious components driven by
# brain motion.


# %%
# The `learning_rate` controls the step size of the optimization. A value of 1e-2 is
# a common default that balances convergence speed and stability for typical fUSI data.
data = cf.registration.register_volumewise(data, learning_rate=1e-2)

# %% [markdown]
# ## Temporal PCA (`mode="temporal"`)
#
# Before performing PCA, we standardize the recording by centering and scaling each
# voxel's time series to zero mean and unit variance. The
# [`standardize`][confusius.signal.standardize] function can be used for this purpose.
# This ensures that PCA captures patterns of correlation rather than patterns of
# covariance; otherwise, PCA may be dominated by high-variance voxels, such as those
# near large blood vessels, which may not be of primary interest.

# %%
data_std = cf.signal.standardize(data)

# %% [markdown]
# In ConfUSIus, the [`PCA`][confusius.decomposition.PCA] model wraps the familiar
# scikit-learn [`PCA`][sklearn.decomposition.PCA] model while preserving the
# fUSI DataArray metadata and coordinates. [`PCA`][confusius.decomposition.PCA] expects
# the same arguments as the scikit-learn model, such as
# [`n_components`][confusius.decomposition.PCA] for the number of
# principal components to compute, and
# [`random_state`][confusius.decomposition.PCA] for reproducibility
# (see the API documentation for more details).
#
# By default, PCA uses `mode="temporal"` (fit on `(time, voxels)`). A
# `mode="spatial"` option is also available, analogous to spatial ICA.
#
# Here, we fit a PCA model with all available components.

# %%
pca_t = cf.decomposition.PCA(random_state=0, mode="temporal")
signals_t = pca_t.fit_transform(data_std)
signals_t

# %% [markdown]
# ## Explained variance
#
# [`explained_variance_ratio_`][confusius.decomposition.PCA]
# gives the fraction of total variance captured along each selected principal axis. For
# centered data with shape `(time, space)`, the rank is at most `min(space, time - 1)`,
# so a final near-zero entry appears in the spectrum; we omit it here for clarity. The
# scree plot on the left highlights the rapid decay of successive components, while the
# cumulative curve on the right shows how quickly the total explained variance saturates.
# These plots can help guide the choice of how many components to retain for further
# analysis, for example by selecting the "elbow" of the screen plot or a threshold for
# cumulative variance.

# %%
variance_ratio = pca_t.explained_variance_ratio_.isel(component=slice(None, -1))
component_ids = variance_ratio.component.values + 1
cumulative_variance = np.cumsum(variance_ratio.values) * 100

fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), constrained_layout=True)

axes[0].plot(
    component_ids, variance_ratio.values * 100, marker="o", ms=4, color="tab:blue"
)
axes[0].set_yscale("log")
axes[0].set_xlabel("Principal component")
axes[0].set_ylabel("Explained variance (%)")
axes[0].set_title("Scree plot")

axes[1].plot(component_ids, cumulative_variance, marker="o", ms=4, color="tab:blue")
axes[1].set_xlabel("Principal component")
axes[1].set_ylabel("Cumulative explained variance (%)")
_ = axes[1].set_title("Cumulative variance")

# %% [markdown]
# ## Temporal PCA maps and corresponding time courses
#
# [`maps_`][confusius.decomposition.PCA] stores principal axes in voxel space as a
# `(component, z, y, x)` DataArray. [`transform`][confusius.decomposition.PCA.transform]
# returned the associated temporal scores in `signals_t`, a `(time, component)`
# DataArray.
#
# Looking at both together is more informative than either alone. The maps show the
# dominant covariance structure in space, while the scores show how strongly each pattern
# is expressed over time.

# %% tags=["thumbnail"]
n_show = 6
fig = plt.figure(figsize=(10.5, 8.5), constrained_layout=True)
fig.patch.set_facecolor(bg_color)
gs = fig.add_gridspec(n_show, 2, width_ratios=[1, 3])

axes_tc = [fig.add_subplot(gs[i, 1]) for i in range(n_show)]
for ax in axes_tc[1:]:
    ax.sharex(axes_tc[0])

for i, comp in enumerate(range(n_show)):
    var = float(pca_t.explained_variance_ratio_.sel(component=comp)) * 100

    component_map = pca_t.maps_.isel(component=[comp])
    cf.plotting.plot_stat_map(
        component_map,
        axes=fig.add_subplot(gs[i, 0]),
        slice_mode="component",
        show_axes=False,
        show_colorbar=False,
        show_titles=False,
        bg_color=bg_color,
    )

    signals_t.sel(component=comp).plot(ax=axes_tc[i], lw=1.1)
    axes_tc[i].set_title(f"PC {comp + 1} ({var:.1f} %)")
    axes_tc[i].set_xlabel("")
    axes_tc[i].set_ylabel("")

for ax in axes_tc[:-1]:
    ax.tick_params(labelbottom=False)
axes_tc[-1].set_xlabel("Time (s)")
_ = fig.suptitle("Temporal PCA maps and time courses (first 6 components)", fontsize=21)

# %% [markdown]
# ## Spatial PCA (`mode="spatial"`)
#
# Spatial PCA transposes the matrix to `(voxels, time)` before decomposition. In this
# orientation, variance maximization is performed across voxels, yielding orthogonal
# spatial maps as principal components.

# %%
pca_s = cf.decomposition.PCA(random_state=0, mode="spatial")
signals_s = pca_s.fit_transform(data_std)
signals_s

# %% [markdown]
# As in temporal mode, we inspect maps and corresponding time courses jointly.

# %%
n_show = 6
fig = plt.figure(figsize=(10.5, 8.5), constrained_layout=True)
fig.patch.set_facecolor(bg_color)
gs = fig.add_gridspec(n_show, 2, width_ratios=[1, 3])

axes_tc = [fig.add_subplot(gs[i, 1]) for i in range(n_show)]
for ax in axes_tc[1:]:
    ax.sharex(axes_tc[0])

for i, comp in enumerate(range(n_show)):
    var = float(pca_s.explained_variance_ratio_.sel(component=comp)) * 100

    component_map = pca_s.maps_.isel(component=[comp])
    vmax = float(np.abs(component_map).max())
    cf.plotting.plot_stat_map(
        component_map,
        axes=fig.add_subplot(gs[i, 0]),
        slice_mode="component",
        cmap="coolwarm",
        vmax=vmax,
        show_axes=False,
        show_colorbar=False,
        show_titles=False,
        bg_color=bg_color,
    )

    signals_s.sel(component=comp).plot(ax=axes_tc[i], lw=1.1)
    axes_tc[i].set_title(f"PC {comp + 1} ({var:.1f} %)")
    axes_tc[i].set_xlabel("")
    axes_tc[i].set_ylabel("")

for ax in axes_tc[:-1]:
    ax.tick_params(labelbottom=False)
axes_tc[-1].set_xlabel("Time (s)")
_ = fig.suptitle("Spatial PCA maps and time courses (first 6 components)", fontsize=21)

# %% [markdown]
# [^1]: We usually consider voxels as features and time points as samples, but PCA can
# be applied in either orientation.

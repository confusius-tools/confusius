# %% [markdown]
# # PCA on a single fUSI recording
#
# This example shows how to use principal component analysis (PCA) to decompose a fUSI
# recording into principal axes of variance.
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
from confusius.decomposition import PCA
from confusius.signal import standardize

# Keep figure backgrounds transparent in docs and standalone notebooks.
bg_color = "none"
# Match text and axes styling to the active Matplotlib theme.
fg_color = mpl.colors.to_hex(plt.rcParams["text.color"])
# # Increase text size for dense decomposition figures.
# plt.rcParams.update(
#     {
#         "font.size": 15,
#         "axes.titlesize": 17,
#         "axes.labelsize": 15,
#         "xtick.labelsize": 13,
#         "ytick.labelsize": 13,
#         "legend.fontsize": 13,
#     }
# )
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
# [`register_volumewise`][confusius.registration.register_volumewise]. This is a common
# preprocessing step for fUSI data, and it can help avoid spurious components driven by
# brain motion.
#
# You may set `show_progress=True` to display a progress bar during registration, which
# can be helpful for long recordings.

# %%
# The `learning_rate` controls the step size of the optimization. A value of 1e-2 is
# a common default that balances convergence speed and stability for typical fUSI data.
data = cf.registration.register_volumewise(
    data, learning_rate=1e-2, show_progress=False
)

# %% [markdown]
# ## Fit a PCA model
#
# Before performing PCA, we standardize the recording by centering and scaling each
# voxel's time series to zero mean and unit variance. The
# [`standardize`][confusius.signal.standardize] function can be used for this purpose.
# This ensures that PCA captures patterns of correlation rather than patterns of
# covariance; otherwise, PCA may be dominated by high-variance voxels, such as those
# near large blood vessels, which may not be of primary interest.

# %%
data_std = standardize(data)

# %% [markdown]
# In ConfUSIus, the [`PCA`][confusius.decomposition.PCA] model wraps the familiar
# scikit-learn [`PCA`][sklearn.decomposition.PCA] model while preserving the
# fUSI DataArray metadata and coordinates. [`PCA`][confusius.decomposition.PCA] expects
# the same arguments as the scikit-learn model, such as
# [`n_components`][confusius.decomposition.PCA] for the number of
# principal components to compute, and
# [`random_state`][confusius.decomposition.PCA] for reproducibility
# (see the API documentation for more details). After fitting the model, the principal
# axes—also known as spatial components—are stored in the
# [`.maps_`][confusius.decomposition.PCA] attribute, and the corresponding time
# series—also known as temporal components or scores—can be obtained by transforming the
# data with the [`.transform`][confusius.decomposition.PCA.transform] method.
#
# Here, we fit a PCA model with all available components.

# %%
pca = PCA(random_state=0)
signals = pca.fit_transform(data_std)
signals

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
variance_ratio = pca.explained_variance_ratio_.isel(component=slice(None, -1))
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
# ## Visualize the principal axes of variance
#
# [`maps_`][confusius.decomposition.PCA] is a `(component, z, y, x)` DataArray.
# Each map is a principal axis in voxel space and indicates how strongly each voxel
# contributes to the corresponding component. Voxels with large positive weights tend to
# increase together along that axis, whereas voxels with large negative weights tend to
# vary in the opposite direction.
#
# These maps should not be interpreted as localized activation maps. Instead, they
# describe distributed patterns of covariation across the recording. In practice, the
# leading components often reflect broad anatomical or vascular structure, global signal
# fluctuations, artifacts, and other dominant sources of shared variance.

# %% tags=["thumbnail"]
maps_12 = pca.maps_.isel(component=slice(0, 12))
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
_ = plotter.figure.suptitle("PCA spatial maps (first 12 components)", fontsize=21)

# %% [markdown]
# ## Visualize the temporal components
#
# [`.transform`][confusius.decomposition.PCA.transform] returned the principal
# components, that is, the projection of the data onto each principal axis: a `(time,
# component)` DataArray. These time courses show how strongly each spatial pattern is
# expressed at each time point.
#
# Looking at the temporal scores together with the spatial maps is often more informative
# than inspecting either alone. A component may appear anatomically structured in space,
# while its score can reveal whether it reflects a slow drift, a transient fluctuation,
# or a more rhythmic pattern. Because PCA orders components by explained variance, the
# first components correspond to the most dominant temporal structure in the recording.

# %%
n_show = 6
fig = plt.figure(figsize=(10.5, 7.5), constrained_layout=True)
fig.patch.set_facecolor(bg_color)
gs = fig.add_gridspec(n_show, 2, width_ratios=[1, 3])

axes_tc = [fig.add_subplot(gs[i, 1]) for i in range(n_show)]
for ax in axes_tc[1:]:
    ax.sharex(axes_tc[0])

for i, comp in enumerate(range(n_show)):
    var = float(pca.explained_variance_ratio_.sel(component=comp)) * 100

    component_map = pca.maps_.isel(component=[comp])
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
    axes_tc[i].set_title(f"PC {comp + 1} ({var:.1f} %)")
    axes_tc[i].set_xlabel("")
    axes_tc[i].set_ylabel("")

for ax in axes_tc[:-1]:
    ax.tick_params(labelbottom=False)
axes_tc[-1].set_xlabel("Time (s)")
_ = fig.suptitle("PCA spatial maps and time courses (first 6 components)", fontsize=21)

# %% [markdown]
# [^1]: We usually consider voxels as features and time points as samples, but PCA can
# be applied in either orientation.

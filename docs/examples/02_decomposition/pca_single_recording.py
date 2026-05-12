# %% [markdown]
# # PCA on a single fUSI recording
#
# This example shows how to use [PCA][confusius.decomposition.PCA] to decompose a fUSI
# recording into principal components and reconstruct a denoised version of the data by
# retaining only the components that explain most of the variance.
#
# PCA finds an orthogonal basis that maximises explained variance. In fUSI, the leading
# components typically capture structured haemodynamic activity, while later components
# tend to represent noise and high-frequency fluctuations. This makes PCA a useful first
# step for data exploration and denoising.
#
# We use a spontaneous activity recording from the
# [Nunez-Elizalde 2022 dataset](https://doi.org/10.1016/j.neuron.2022.02.012).

# %% [markdown]
# ## Load and standardize the recording

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
# Before fitting PCA, we standardize the recording to z-scores with
# [`standardize`][confusius.signal.standardize]. This removes the DC offset and
# normalizes the scale across voxels so the decomposition is driven by temporal dynamics
# rather than differences in mean intensity.

# %%
data_std = standardize(data)
data_std

# %% [markdown]
# ## Fit PCA
#
# [PCA][confusius.decomposition.PCA] expects a `(time, ...)` DataArray. Here we retain
# 30 components, which is enough to capture most of the structured variance in a typical
# 2D fUSI recording while keeping the decomposition compact. The `random_state` argument
# fixes the SVD solver initialisation for reproducibility.

# %%
pca = PCA(n_components=30, random_state=0)
signals = pca.fit_transform(data_std)
signals

# %% [markdown]
# ## Scree plot
#
# The `explained_variance_ratio_` attribute tells us what fraction of total variance each
# component accounts for. Plotting this as a cumulative curve (scree plot) helps choose
# how many components to keep.

# %%
cumvar = np.cumsum(pca.explained_variance_ratio_.values)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

axes[0].bar(
    pca.explained_variance_ratio_.component.values,
    pca.explained_variance_ratio_.values * 100,
)
axes[0].set_xlabel("Component")
axes[0].set_ylabel("Explained variance (%)")
axes[0].set_title("Per-component variance")

axes[1].plot(pca.explained_variance_ratio_.component.values, cumvar * 100, marker="o", ms=4)
axes[1].axhline(90, color="tab:red", lw=1, ls="--", label="90 %")
axes[1].set_xlabel("Number of components")
axes[1].set_ylabel("Cumulative explained variance (%)")
axes[1].set_title("Cumulative variance")
axes[1].legend()

# %% [markdown]
# ## Component maps
#
# `components_` is a `(component, y, x)` DataArray. Each map shows the spatial
# distribution of one principal component — the regions that vary most together along
# that direction in data space.

# %% tags=["thumbnail"]
plotter = cf.plotting.plot_volume(
    pca.components_.isel(component=slice(0, 12)),
    slice_mode="component",
    cmap="coolwarm",
    ncols=4,
    show_axes=False,
    bg_color=bg_color,
    fg_color=fg_color,
    cbar_label="Component weight",
)
plotter.figure.suptitle("Principal component maps (first 12)", fontsize=11)
# %% [markdown]
# ## Component time courses
#
# `fit_transform` returns the projection of the data onto each component: a
# `(time, component)` DataArray. The time courses reveal the temporal dynamics
# associated with each spatial pattern.

# %%
fig, axes = plt.subplots(6, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
for ax, comp in zip(axes, range(6)):
    signals.sel(component=comp).plot(ax=ax, lw=0.8)
    var = float(pca.explained_variance_ratio_.sel(component=comp)) * 100
    ax.set_title(f"PC {comp}  ({var:.1f} %)", fontsize=8)
    ax.set_xlabel("")
axes[-1].set_xlabel("Time (s)")
plt.suptitle("PCA time courses (first 6 components)", fontsize=11)

# %% [markdown]
# ## Denoised reconstruction
#
# `inverse_transform` maps component-space signals back to the original spatial geometry.
# By passing only the leading $k$ components and zeroing the rest, we project the data
# onto a lower-dimensional subspace — effectively suppressing variance that the first $k$
# components do not explain.
#
# Here we keep the components that together account for at least 90 % of the total
# variance and compare the result to the original data on a single frame.

# %%
k = min(int(np.searchsorted(cumvar, 0.90)) + 1, pca.n_components_)

denoised_signals = signals.copy()
denoised_signals.loc[dict(component=slice(k, None))] = 0.0

denoised = pca.inverse_transform(denoised_signals)

# %%
frame_idx = 100

fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True, facecolor="none")

for ax, (title, vol, cmap) in zip(
    axes,
    [
        ("Original", data_std, "viridis"),
        (f"Denoised\n({k} components, {cumvar[k - 1] * 100:.1f} % variance)", denoised, "viridis"),
        ("Residual", data_std - denoised, "coolwarm"),
    ],
):
    cf.plotting.plot_volume(
        vol.isel(time=[frame_idx]),
        slice_mode="time",
        cmap=cmap,
        bg_color=bg_color,
        fg_color=fg_color,
        show_titles=False,
        show_colorbar=False,
        axes=ax,
    )
    ax.set_title(title, fontsize=9)

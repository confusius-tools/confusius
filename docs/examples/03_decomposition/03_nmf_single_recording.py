# %% [markdown] # NMF on a single fUSI recording
#
# This example shows how to use non-negative matrix factorization (NMF) to decompose a
# fUSI recording into non-negative spatial maps and their associated non-negative time
# courses. It complements the [PCA](pca_single_recording.md) and
# [FastICA](fastica_single_recording.md) examples in the same gallery.
#
# NMF is unique among the decomposers in ConfUSIus because it requires strictly
# non-negative inputs[^1]. Power Doppler fUSI signals are non-negative by construction,
# so they pass the constraint directly. However, raw power is dominated by each voxel's
# baseline intensity, which can make bright vessels dominate the factorization.
#
# A practical workaround is to center and scale each voxel across time, then split the
# standardized signal into separate positive and negative channels, preserving NMF's
# non-negativity constraint. NMF can then discover additive components in above-baseline
# and below-baseline fluctuations separately.

# %% [markdown]
# ## Load a fUSI recording
#
# We use the same spontaneous activity recording from the [Nunez-Elizalde 2022
# dataset](https://doi.org/10.1016/j.neuron.2022.02.012) as in the [PCA](pca_single_recording.md)
# and [FastICA](fastica_single_recording.md) examples. See the
# [Datasets](../../../user-guide/datasets.md) user guide for more details on how to
# download this dataset using ConfUSIus.

# %%
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
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
## Correct for brain motion
#
# As in the [PCA](pca_single_recording.md#correct-for-brain-motion) and
# [FastICA](fastica_single_recording.md#correct-for-brain-motion) examples, we first
# perform a rigid transformation correction with
# [`register_volumewise`][confusius.registration.register_volumewise] to mitigate brain
# motion.

# %%
data = cf.registration.register_volumewise(data, learning_rate=1e-2)

# %% [markdown]
# ## Standardize for NMF
#
# NMF requires non-negative inputs, but raw Power Doppler is dominated by each voxel's
# baseline intensity rather than the temporal fluctuations we want to group. We
# therefore:
#
# 1. Z-score each voxel's time series with [`standardize`][confusius.signal.standardize]
#    to remove its mean and put voxels on a comparable scale.
# 2. Split the standardized signal into separate positive and negative parts.
#
# This keeps the sign information— above-baseline versus below-baseline
# fluctuations—while still presenting a non-negative matrix to NMF.

# %%
z = cf.signal.standardize(data)
data_nmf = xr.concat(
    [z.clip(min=0), (-z).clip(min=0)],
    dim=xr.IndexVariable("sign", ["pos", "neg"]),
)

# %% [markdown]
# ## Fit temporal NMF
#
# [`NMF`][confusius.decomposition.NMF] wraps the familiar scikit-learn
# [`NMF`][sklearn.decomposition.NMF] estimator while preserving fUSI DataArray metadata
# and coordinates. With `mode="temporal"` (the default), it fits on `(time, voxels)`
# and returns:
#
# - [`maps_`][confusius.decomposition.NMF]: non-negative spatial maps. Because we split
#   the input into positive and negative channels, the maps here have shape
#   `(component, sign, z, y, x)`.
# - [`fit_transform`][confusius.decomposition.NMF.fit_transform]: non-negative time
#   courses of shape `(time, component)`.

# %%
nmf_t = cf.decomposition.NMF(n_components=10, random_state=0, max_iter=500)
signals = nmf_t.fit_transform(data_nmf)
signals

# %% [markdown]
# ## Reconstruction error
#
# [`reconstruction_err_`][confusius.decomposition.NMF] is the Frobenius norm of
# `X - WH`, where `W` are the spatial maps and `H` the time courses. It gives a sense
# of how well the chosen number of components explains the standardized data. A
# quantitative model-selection procedure is out of scope here, but the trace is useful
# when sweeping `n_components` and looking for diminishing returns.

# %%
print(f"reconstruction_err_: {nmf_t.reconstruction_err_:.3f}")
print(f"n_iter_: {nmf_t.n_iter_}")

# %% [markdown]
# ## Spatial maps and time courses
#
# Looking at the spatial maps and the associated time courses side by side is a useful
# first sanity check. Here each component has two map panels: one for above-baseline
# fluctuations (`pos`) and one for below-baseline fluctuations (`neg`). Localized,
# anatomically plausible structure paired with a clear transient in the time course
# tends to reflect a coherent spatiotemporal pattern, while diffuse maps paired with
# noisy or drift-like fluctuations often indicate residual motion or physiological
# artefacts.

# %% tags=["thumbnail"]
n_show = 10
fig = plt.figure(figsize=(11.5, 12.0), constrained_layout=True)
fig.patch.set_facecolor(bg_color)
gs = fig.add_gridspec(n_show, 2, width_ratios=[1.4, 3])

axes_tc = [fig.add_subplot(gs[i, 1]) for i in range(n_show)]
for ax in axes_tc[1:]:
    ax.sharex(axes_tc[0])

for i, comp in enumerate(range(n_show)):
    component_map = nmf_t.maps_.isel(component=comp)
    vmax = float(component_map.max())
    map_gs = gs[i, 0].subgridspec(1, 2, wspace=0.02)

    for j, sign in enumerate(["pos", "neg"]):
        cf.plotting.plot_stat_map(
            component_map.sel(sign=sign),
            axes=fig.add_subplot(map_gs[0, j]),
            slice_mode="z",
            vmax=vmax,
            show_axes=False,
            show_colorbar=False,
            show_titles=False,
            bg_color=bg_color,
        )

    signals.sel(component=comp).plot(ax=axes_tc[i], lw=1.1)
    axes_tc[i].set_title(f"Component {comp + 1}")
    axes_tc[i].set_ylabel("Signal")
    axes_tc[i].set_xlabel("")

for ax in axes_tc[:-1]:
    ax.tick_params(labelbottom=False)
axes_tc[-1].set_xlabel("Time (s)")
_ = fig.suptitle(
    "Temporal NMF: positive/negative maps and time courses (first 10 components)",
    fontsize=21,
)

# %% [markdown]
# ## Spatial NMF
#
# [`NMF`][confusius.decomposition.NMF] also accepts `mode="spatial"`, which transposes
# the data to `(voxels, time)` before fitting. The output convention is identical to
# temporal mode — [`maps_`][confusius.decomposition.NMF] still holds the non-negative
# spatial maps (here with `pos`/`neg` channels) and
# [`fit_transform`][confusius.decomposition.NMF.fit_transform] returns their
# non-negative time courses — so the choice between the two modes mirrors the
# temporal/spatial choice offered by [PCA](pca_single_recording.md) and
# [FastICA](fastica_single_recording.md).

# %%
nmf_s = cf.decomposition.NMF(
    n_components=10, mode="spatial", random_state=0, max_iter=500
)
signals_s = nmf_s.fit_transform(data_nmf)
signals_s

# %% [markdown]
# ### Spatial maps and time courses
#
# As in temporal mode, we inspect the positive and negative spatial maps and their
# corresponding time courses side by side.

# %%
n_show = 10
fig = plt.figure(figsize=(11.5, 12.0), constrained_layout=True)
fig.patch.set_facecolor(bg_color)
gs = fig.add_gridspec(n_show, 2, width_ratios=[1.4, 3])

axes_tc = [fig.add_subplot(gs[i, 1]) for i in range(n_show)]
for ax in axes_tc[1:]:
    ax.sharex(axes_tc[0])

for i, comp in enumerate(range(n_show)):
    component_map = nmf_s.maps_.isel(component=comp)
    vmax = float(component_map.max())
    map_gs = gs[i, 0].subgridspec(1, 2, wspace=0.02)

    for j, sign in enumerate(["pos", "neg"]):
        cf.plotting.plot_stat_map(
            component_map.sel(sign=sign),
            axes=fig.add_subplot(map_gs[0, j]),
            slice_mode="z",
            vmax=vmax,
            show_axes=False,
            show_colorbar=False,
            show_titles=False,
            bg_color=bg_color,
        )

    signals_s.sel(component=comp).plot(ax=axes_tc[i], lw=1.1)
    axes_tc[i].set_title(f"Component {comp + 1}")
    axes_tc[i].set_ylabel("Signal")
    axes_tc[i].set_xlabel("")

for ax in axes_tc[:-1]:
    ax.tick_params(labelbottom=False)
axes_tc[-1].set_xlabel("Time (s)")
_ = fig.suptitle(
    "Spatial NMF: positive/negative maps and time courses (first 10 components)",
    fontsize=21,
)

# %% [markdown]
# [^1]: Lee, D. D., and Seung, H. S. (1999). "Learning the parts of objects by
# non-negative matrix factorization". *Nature*, 401(6755), 788-791.

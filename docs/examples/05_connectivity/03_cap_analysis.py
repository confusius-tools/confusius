# %% [markdown]
# # Co-activation pattern analysis of a single recording
#
# Co-activation pattern (CAP) analysis clusters individual fUSI volumes into a small
# set of recurring whole-brain activity patterns. This example uses
# [`CAP`][confusius.connectivity.CAP] on the same spontaneous Nunez-Elizalde recording
# used by the connectivity examples: load the recording, motion-correct and clean it,
# cluster volumes, then inspect both the spatial CAP maps and their temporal dynamics.

# %% [markdown]
# ## Load and preprocess the recording
#
# We use the same `CR022 / 20201007 / slice02` recording as the atlas-based
# connectivity examples. CAPs are sensitive to global drift and motion, so we first resample to a
# uniform time grid, apply rigid volume-wise motion correction, then high-pass filter and
# z-score each voxel's time series.

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
    sessions="20201007",
    tasks="spontaneous",
    acqs="slice02",
)

# %%
pwd_path = (
    Path(bids_root)
    / "sub-CR022"
    / "ses-20201007"
    / "fusi"
    / "sub-CR022_ses-20201007_task-spontaneous_acq-slice02_pwd.nii.gz"
)
data = cf.timing.resample_to_uniform_time(cf.load(pwd_path)).compute()
data

# %%
data = cf.registration.register_volumewise(data, learning_rate=1e-1)

cleaned = cf.signal.clean(
    data, low_cutoff=0.01, filter_method="cosine", standardize_method="zscore"
).fillna(0)
cleaned

# %% [markdown]
# ## Fit CAPs
#
# [`CAP`][confusius.connectivity.CAP] flattens each volume into a feature vector and
# clusters the volumes with k-means. With the default `metric="correlation"`, each
# volume is spatially centered and L2-normalized before clustering, so CAP assignment is
# based on spatial pattern similarity rather than absolute signal scale.

# %%
n_caps = 6
caps = cf.connectivity.CAP(n_clusters=n_caps, metric="correlation", random_state=0)
caps.fit(cleaned)
caps.caps_

# %% [markdown]
# ## Plot spatial CAP maps
#
# `caps.caps_` contains one spatial map per cluster. The maps live in the preprocessed
# space used for clustering; positive and negative values show voxels that co-activate
# or anti-activate within each recurring pattern.

# %% tags=["thumbnail"]
# coolwarm's white midpoint reads as a washed-out hole on a dark background, so switch
# to berlin (Crameri's perceptually uniform diverging colormap, black midpoint) when the
# current Matplotlib style is dark.
is_dark_theme = sum(mpl.colors.to_rgb(bg_color)) / 3 < 0.5
cmap = "berlin" if is_dark_theme else "coolwarm"

anatomy = data.mean(dim="time").fusi.scale.db().expand_dims(cap=caps.caps_.cap)

fig, axes = plt.subplots(2, 3, figsize=(9, 5.5), constrained_layout=True)
fig.patch.set_facecolor(bg_color)

cf.plotting.plot_stat_map(
    caps.caps_,
    bg_volume=anatomy,
    slice_mode="cap",
    cmap=cmap,
    vmax=float(np.abs(caps.caps_).max()),
    show_axes=False,
    figure=fig,
    axes=axes,
    bg_color=bg_color,
)
_ = fig.suptitle("Co-activation pattern maps", fontsize=16)

# %% [markdown]
# ## Inspect CAP expression over time
#
# `labels_` stores the CAP assigned to each volume, and `scores_` stores the assignment
# quality. For the correlation metric, scores are cosine similarities to the assigned CAP
# center: larger values mean a volume better matches its CAP.

# %%
labels = caps.labels_[0]
scores = caps.scores_[0]

fig, axes = plt.subplots(2, 1, figsize=(10, 4.5), sharex=True, constrained_layout=True)
fig.patch.set_facecolor(bg_color)

(labels + 1).plot.line(ax=axes[0], drawstyle="steps-post", lw=1)
axes[0].set_ylabel("CAP")
axes[0].set_yticks(np.arange(1, n_caps + 1))
axes[0].set_title("CAP sequence")

scores.plot.line(ax=axes[1], lw=1, color="tab:orange")
axes[1].set_ylabel("Similarity")
axes[1].set_title("Assignment score")
_ = axes[1].set_xlabel("Time (s)")

# %% [markdown]
# ## Summarize temporal dynamics
#
# [`compute_temporal_metrics`][confusius.connectivity.CAP.compute_temporal_metrics]
# summarizes how often each CAP appears, how persistent its episodes are, and how often
# the recording switches from one CAP to another. Squaring the one-step transition
# matrix gives the two-step transition probabilities, i.e. where the sequence tends to
# land after two consecutive transitions.

# %%
metrics = caps.compute_temporal_metrics()
metrics

# %%
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
fig.patch.set_facecolor(bg_color)

fraction = metrics.temporal_fraction.sel(recording=0)
axes[0].bar(fraction.cap.values + 1, fraction.values * 100)
axes[0].set_xlabel("CAP")
axes[0].set_ylabel("Time assigned (%)")
axes[0].set_title("Temporal fraction")

transition_matrix = metrics.transition_matrix.sel(recording=0).assign_coords(
    cap_from=np.arange(1, n_caps + 1),
    cap_to=np.arange(1, n_caps + 1),
)
cf.plotting.plot_matrix(
    transition_matrix,
    ax=axes[1],
    cmap="magma",
    vmin=0,
    vmax=1,
    auto_range=False,
    cbar_label="P(next CAP)",
    title="One-step transitions",
    bg_color=bg_color,
)

transition_matrix_2 = xr.DataArray(
    np.linalg.matrix_power(transition_matrix.values, 2),
    dims=transition_matrix.dims,
    coords=transition_matrix.coords,
)
_ = cf.plotting.plot_matrix(
    transition_matrix_2,
    ax=axes[2],
    cmap="magma",
    vmin=0,
    vmax=1,
    auto_range=False,
    cbar_label="P(CAP after 2 steps)",
    title="Two-step transitions",
    bg_color=bg_color,
)

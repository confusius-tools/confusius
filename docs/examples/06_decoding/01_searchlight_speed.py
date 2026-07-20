# %% [markdown]
# # Searchlight decoding of locomotion speed
#
# This example maps which parts of a fUSI plane carry information about how fast the
# animal is moving, using a searchlight: a small cross-validated model run over the
# local neighborhood of every voxel.
#
# A searchlight complements the general linear model. The GLM asks, voxel by voxel,
# "does this voxel's signal track the regressor?". The searchlight asks "can the local
# pattern around this voxel predict the regressor?", which picks up information carried
# jointly by neighboring voxels rather than by any one of them alone.
#
# We follow the experimental setting and dataset of [Cybis Pereira et al.
# 2026](https://doi.org/10.1016/j.celrep.2025.116791), decoding locomotion speed from a
# single coronal plane, and compare the searchlight map against a GLM fit on the same
# data. Both analyses receive the same smoothing, the same nuisance removal and the same
# hemodynamically convolved speed regressor, so the only thing that differs between them
# is univariate versus multivariate, which is the comparison we actually want to make.

# %% [markdown]
# ## Load the recording and the tracking data

# %%
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.linear_model import RidgeCV

import confusius as cf

subject = "rat75"
session = "20220524"
acq = "slice32"

xr.set_options(display_expand_data=False)

bids_root = cf.datasets.fetch_cybis_pereira_2026(
    datasets="rawdata",
    subjects=subject,
    sessions=session,
    acqs=acq,
)

session_dir = Path(bids_root) / f"sub-{subject}" / f"ses-{session}"
stem = f"sub-{subject}_ses-{session}_task-openfield"

pwd_path = session_dir / "fusi" / f"{stem}_acq-{acq}_pwd.nii.gz"
motion_path = session_dir / "motion" / f"{stem}_tracksys-DLC_acq-{acq}_motion.tsv"

data = cf.load(pwd_path).compute()
data = cf.registration.register_volumewise(data, learning_rate=1)
data

# %% [markdown]
# ## Build the speed regressor
#
# The animal is tracked with DeepLabCut at 50 frames per second. We take the
# frame-to-frame displacement of the body marker, smooth it with a one second centered
# rolling mean, and resample it onto the fUSI volume acquisition times.

# %%
fps = 50
motion_df = pd.read_csv(motion_path, sep="\t")
squared_diff = motion_df[["body_x", "body_y"]].diff() ** 2
speed_df = fps * (squared_diff["body_x"] + squared_diff["body_y"]) ** 0.5
speed_df[0] = 0
speed = (
    xr.DataArray(
        speed_df,
        dims=["time"],
        coords={"time": 1 / fps * np.arange(len(speed_df))},
        name="speed",
    )
    .rolling(time=fps, min_periods=1, center=True)
    .mean()
)
speed = speed.interp(time=data.time, method="linear").ffill("time")

# %% [markdown]
# ## What the searchlight should actually predict
#
# The speed signal above is an instantaneous behavioral variable, but the power Doppler
# signal is hemodynamic: it lags the behavior and smooths it over several seconds.
# Asking a decoder to predict the instantaneous trace from that signal sets it an
# impossible target. We therefore decode the *hemodynamically convolved* speed
# regressor, which is the same quantity a GLM tests against.
#
# We build it with the modified Claron 2021 HRF, a rodent fUSI response function, and
# read it straight off a first-level design matrix. Building the design matrix here
# serves double duty: its `speed` column is the searchlight target, and the whole matrix
# is what we hand the GLM later, so the two analyses are guaranteed to see the same
# regressor.

# %%
modified_claron2021 = partial(cf.glm.claron2021_hrf, beta=6.7)

events = pd.DataFrame(
    {
        "onset": data.time.values,
        "duration": data.time.volume_acquisition_duration,
        "modulation": speed.values,
        "trial_type": "speed",
    }
)
confounds = cf.signal.compute_compcor_confounds(
    data,
    variance_threshold=0.05,
    n_components=3,
)


design_matrix = cf.glm.make_first_level_design_matrix(
    data.time.values,
    events=events,
    hrf_model=modified_claron2021,
    drift_model="cosine",
    low_cutoff=0.01,
    confounds=confounds,
)

# %% [markdown]
# The `speed` column of the design matrix is the raw speed trace convolved with the HRF.
# We read it out as the searchlight target.

# %%
speed_regressor = design_matrix["speed"].to_numpy()

# %% [markdown]
# ## Preprocess the data exactly as the GLM does
#
# The point of this example is to compare a univariate analysis with a multivariate one,
# which only works if everything *else* is held equal. So the decoder gets the same
# preprocessing the GLM applies.
#
# **Spatial smoothing.** `FirstLevelModel(smoothing_fwhm=0.3)` smooths each run with
# [`smooth_volume`][confusius.spatial.smooth_volume] before fitting. We apply the same
# call with the same value, rather than leaving the searchlight to work on unsmoothed
# data while the GLM enjoys the noise reduction.
#
# **Drift and confound removal.** Power Doppler carries slow drift and nuisance signal
# that have nothing to do with locomotion. The GLM handles them with cosine drift and
# [CompCor][confusius.signal.compute_compcor_confounds] regressors in its design; a
# decoder has no design matrix, so we remove the same regressors from the data up front
# with [`clean`][confusius.signal.clean], reusing the design matrix's own nuisance
# columns.
#
# The target is cleaned with the same regressors as the data, so both sides have the same
# nuisance structure removed. This mirrors what the GLM does implicitly when it fits the
# speed regressor and the nuisance regressors jointly.
#
# Every one of these operations is a fixed linear filter applied identically to every
# timepoint. None uses the relationship between the data and the target, so none can leak
# information across the cross-validation folds.

# %%
smoothing_fwhm = 0.3

confounds_np = design_matrix.drop(columns="speed").to_numpy()
confounds = xr.DataArray(
    confounds_np, dims=["time", "confound"], coords={"time": data.time}
)


cleaned = cf.signal.clean(
    cf.spatial.smooth_volume(data, smoothing_fwhm),
    standardize_method="zscore",
    confounds=confounds,
)
target = cf.signal.clean(
    xr.DataArray(
        speed_regressor, dims=["time"], coords={"time": data.time}, name="speed"
    ),
    standardize_method="zscore",
    confounds=confounds,
)

# %% [markdown]
# ## Run the searchlight
#
# Z-scoring during cleaning sets any zero-variance voxel, such as the corners outside the
# imaged plane, to NaN, and a neighborhood that includes one cannot be fit. We therefore
# pass a `mask` selecting the voxels that stayed finite through preprocessing; without it
# `SearchLight` would try to use every voxel and refuse to run. A real region-of-interest
# mask, such as an intensity-thresholded brain mask, would go here just the same.
#
# Two details matter for fUSI data:
#
# - `radius` is in the units of the data's spatial coordinates, not in voxel indices.
#   fUSI voxels are usually anisotropic, so an index-based radius would silently give
#   anisotropic neighborhoods. Each neighborhood is the set of voxels within `radius`
#   millimeters of the center.
# - Consecutive fUSI volumes are strongly autocorrelated, and the HRF convolution makes
#   the target smoother still. Cross-validating with shuffled folds would put
#   near-duplicate volumes in both the training and test sets and inflate the scores.
#   `SearchLight` therefore builds contiguous temporal folds by default. Each fold also
#   needs to be long enough to contain both quiet and active periods, since the animal
#   moves in bursts, which is why we keep the fold count low.
#
# The estimator is a `RidgeCV`: ridge regression that selects its own penalty from a
# grid. Neighboring fUSI voxels are highly correlated, and the right amount of
# regularization varies across the plane, so fixing a single penalty by hand would favor
# some regions arbitrarily. By default `RidgeCV` picks the penalty by leave-one-out
# generalized cross-validation, which does put temporally adjacent volumes in its train
# and test sets, unlike the contiguous folds we use for the outer searchlight
# cross-validation. That cannot leak: the penalty search runs entirely inside each outer
# training fold and never sees the outer test fold, so it only affects which penalty is
# chosen.

# %%
estimator = RidgeCV(alphas=np.logspace(0, 4, 9))
feature_mask = cleaned.notnull().all("time")

searchlight = cf.decoding.SearchLight(
    estimator=estimator, mask=feature_mask, radius=0.6, cv=2, n_jobs=-1
)
searchlight.fit(cleaned, target.values)
searchlight.scores_

# %% [markdown]
# ## Compare against a GLM
#
# We now fit a GLM with the design matrix built earlier, so it tests the same
# HRF-convolved speed regressor against the same drift model, on data smoothed with the
# same kernel.

# %%
glm = cf.glm.FirstLevelModel(smoothing_fwhm=smoothing_fwhm)
glm.fit(data, design_matrices=design_matrix)
z_scores = glm.compute_contrast("speed")

# %% [markdown]
# ## Compare the two maps
#
# The searchlight reports a cross-validated coefficient of determination, so values at or
# below zero mean the local neighborhood predicts the speed regressor no better than the
# fold mean; we clip the color scale at zero. The GLM reports a z-score for the speed
# contrast. Both maps cover the whole plane.

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 3), constrained_layout=True)

searchlight.scores_.plot(
    ax=axes[0], cmap="inferno", vmin=0, cbar_kwargs={"label": "Cross-validated $R^2$"}
)
axes[0].set_title("Searchlight decoding of speed")

z_scores.plot(ax=axes[1], cmap="coolwarm", center=0, cbar_kwargs={"label": "z-score"})
axes[1].set_title("GLM, same regressor")

for ax in axes:
    ax.set_aspect("equal")
    ax.invert_yaxis()

# %% [markdown]
# We quantify the spatial agreement between the two maps by the Dice overlap of their top
# 5 percent of voxels.

# %%
top_scores = searchlight.scores_.squeeze(drop=True)
top_z = z_scores.squeeze(drop=True)

selected_scores = top_scores >= top_scores.quantile(0.95)
selected_z = top_z >= top_z.quantile(0.95)
dice = float(
    2
    * (selected_scores & selected_z).sum()
    / (selected_scores.sum() + selected_z.sum())
)
print(f"Top-5% overlap, Dice = {dice:.3f}")

# %% [markdown]
# The overlap is partial: the two maps answer related but different questions. The GLM is univariate and asks, at
# each voxel, whether that voxel's signal tracks the speed regressor. The searchlight is
# multivariate and cross-validated, and asks whether the local pattern around each voxel
# predicts the regressor in held-out blocks of time. Matching the preprocessing and the
# regressor removes the reasons the two maps could differ artifactually, leaving the
# genuine difference between a univariate and a multivariate question.

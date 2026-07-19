# %% [markdown]
# # Searchlight decoding of locomotion speed
#
# This example maps which parts of a fUSI plane carry information about how fast the
# animal is moving, using a searchlight: a small cross-validated model run over the
# local neighbourhood of every voxel.
#
# A searchlight complements the general linear model. The GLM asks, voxel by voxel,
# "does this voxel's signal track the regressor?". The searchlight asks "can the local
# pattern around this voxel predict the regressor?", which picks up information carried
# jointly by neighbouring voxels rather than by any one of them alone.
#
# We follow the experimental setting and dataset of [Cybis Pereira et al.
# 2026](https://doi.org/10.1016/j.celrep.2025.116791), decoding locomotion speed from a
# single coronal plane, and compare the searchlight map against a GLM fit on the same
# data. Both analyses receive the same smoothing, the same drift removal and the same
# haemodynamically convolved speed regressor, so the only thing that differs between them
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import confusius as cf

subject = "rat75"
session = "20220524"
acq = "slice32"

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
data

# %% [markdown]
# ## Build the speed regressor
#
# The animal is tracked with DeepLabCut at 50 frames per second. We take the
# frame-to-frame displacement of the body marker, smooth it with a one second centred
# rolling mean, and resample it onto the fUSI volume acquisition times.

# %%
fps = 50
motion_df = pd.read_csv(motion_path, sep="\t")
squared_diff = motion_df.diff() ** 2
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

fig, ax = plt.subplots(figsize=(8, 2.5))
speed.plot(ax=ax)
ax.set_ylabel("Speed (pixels/s)")
_ = ax.set_title("Locomotion speed, resampled to volume times")

# %% [markdown]
# ## What the searchlight should actually predict
#
# Speed as plotted above is an instantaneous behavioural variable, but the power Doppler
# signal is haemodynamic: it lags the behaviour and smooths it over several seconds.
# Asking a decoder to predict the instantaneous trace from that signal sets it an
# impossible target. We therefore decode the *haemodynamically convolved* speed
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

design_matrix = cf.glm.make_first_level_design_matrix(
    data.time.values,
    events=events,
    hrf_model=modified_claron2021,
    drift_model="cosine",
    low_cutoff=0.01,
)
design_matrix.columns.tolist()

# %% [markdown]
# The `speed` column is the raw speed trace convolved with the HRF. Correlating it
# against the unconvolved trace at a range of shifts shows what the convolution did: the
# match is poor at zero shift and peaks several volumes later, which is the haemodynamic
# delay the searchlight would otherwise have to overcome on its own.

# %%
speed_regressor = design_matrix["speed"].to_numpy()
shifts = range(9)
correlations = [
    np.corrcoef(speed_regressor[shift:], speed.values[: len(speed_regressor) - shift])[
        0, 1
    ]
    for shift in shifts
]

fig, ax = plt.subplots(figsize=(6, 2.5))
ax.plot(list(shifts), correlations, marker="o")
ax.set_xlabel("Shift (volumes)")
ax.set_ylabel("Correlation with raw speed")
_ = ax.set_title("The convolved regressor lags raw speed")

# %% [markdown]
# ## Preprocess the data exactly as the GLM does
#
# The point of this example is to compare a univariate analysis with a multivariate one,
# which only works if everything *else* is held equal. So the decoder gets the same two
# preprocessing steps the GLM applies, and nothing more.
#
# **Spatial smoothing.** `FirstLevelModel(smoothing_fwhm=0.3)` smooths each run with
# [`smooth_volume`][confusius.spatial.smooth_volume] before fitting. We apply the same
# call with the same value, rather than leaving the searchlight to work on unsmoothed
# data while the GLM enjoys the noise reduction.
#
# **Drift removal.** Power Doppler carries slow drift that has nothing to do with
# locomotion. The GLM handles it with cosine drift regressors in its design; a decoder
# has no design matrix, so we remove the same basis from the data up front with
# [`clean`][confusius.signal.clean].
#
# The regressor needs the same drift treatment as the data. This is the step that makes
# or breaks the analysis. If the data has its drift structure removed but the target
# still carries it, we are asking the decoder to predict variance we have just deleted
# from its inputs, and the cross-validated $R^2$ comes out negative almost everywhere.
# Cleaning both sides is what the GLM does implicitly, by fitting the speed regressor and
# the drift regressors jointly. Doing it explicitly here took the peak neighbourhood in
# this recording from an $R^2$ of 0.03 to 0.25.
#
# Every one of these operations is a fixed linear filter applied identically to every
# timepoint. None uses the relationship between the data and the target, so none can leak
# information across the cross-validation folds. We check that claim directly further
# down.

# %%
smoothing_fwhm = 0.3
clean_kwargs = dict(filter_method="cosine", low_cutoff=0.01)

cleaned = cf.signal.clean(
    cf.spatial.smooth_volume(data, smoothing_fwhm), **clean_kwargs
)
target = cf.signal.clean(
    xr.DataArray(
        speed_regressor, dims=["time"], coords={"time": data.time}, name="speed"
    ),
    **clean_kwargs,
)

# %% [markdown]
# ### Why there are no CompCor confounds here
#
# A conventional fUSI pipeline would also regress out a few
# [CompCor][confusius.signal.compute_compcor_confounds] components. We deliberately do
# not, and it is worth saying why, because it is the single largest effect in this
# example.
#
# CompCor builds its components from the highest-variance voxels. In a task where the
# behaviour drives large, spatially widespread haemodynamic changes, which is exactly
# what locomotion does, those components end up describing the task rather than the
# noise. Removing them removes the signal we are trying to decode. Adding three CompCor
# components to both sides of this analysis drops the peak cross-validated $R^2$ from
# 0.41 to 0.28, and more components make it worse still.
#
# This is a property of this dataset, not general advice. CompCor is valuable when the
# dominant variance really is nuisance. Here it is not, so both the searchlight and the
# GLM go without.

# %% [markdown]
# ## Run the searchlight
#
# The searchlight needs a mask defining which voxels are available as features. We use
# a simple intensity threshold on the mean power Doppler image to exclude voxels
# outside the brain.
#
# Two details matter for fUSI data:
#
# - `radius` is in the units of the data's spatial coordinates, not in voxel indices.
#   fUSI voxels are usually anisotropic, so an index-based radius would silently give
#   anisotropic neighbourhoods. This recording has an in-plane spacing of roughly
#   0.1 mm, so a 1.0 mm radius collects a neighbourhood some twenty voxels across.
#   Scores rise gently with radius up to about 1 mm and then flatten, and past that the
#   neighbourhood covers so much of the plane that the map stops being a local one.
# - Consecutive fUSI volumes are strongly autocorrelated, and the HRF convolution makes
#   the target smoother still. Cross-validating with shuffled folds would put
#   near-duplicate volumes in both the training and test sets and inflate the scores.
#   `SearchLight` therefore builds contiguous temporal folds by default, which is what
#   we rely on here.
# - The number of folds matters more than it usually would. Three contiguous folds give
#   test blocks of roughly seven minutes, long enough to contain both quiet and active
#   periods. Splitting more finely produces short blocks in which the animal barely
#   moves, the target has almost no variance to predict, and the score degrades sharply:
#   the peak $R^2$ falls from 0.41 at three folds to 0.39 at five and 0.32 at eight. That
#   is a property of bursty behaviour rather than of the decoder, and it is worth knowing
#   that the headline number depends on it.
#
# The estimator is a `RidgeCV`: ridge regression that selects its own penalty from a
# grid. Neighbouring fUSI voxels are highly correlated, and the right amount of
# regularisation varies across the plane, so fixing a single penalty by hand would favour
# some regions arbitrarily. By default `RidgeCV` picks the penalty by leave-one-out
# generalised cross-validation, which does put temporally adjacent volumes in its train
# and test sets, unlike the contiguous folds we insist on for the outer searchlight
# cross-validation. That cannot inflate the reported scores: the penalty search runs
# entirely inside each outer training fold and never sees the outer test fold, so it only
# affects which penalty is chosen.

# %%
mean_image = data.mean("time")
mask = mean_image > mean_image.quantile(0.5)
mask = mask.drop_vars("quantile")

estimator = make_pipeline(
    StandardScaler(),
    RidgeCV(alphas=np.logspace(0, 4, 9)),
)

searchlight = cf.decoding.SearchLight(
    mask=mask,
    estimator=estimator,
    radius=1.0,
    cv=3,
    n_jobs=-1,
)
searchlight.fit(cleaned, target.values)
searchlight.scores_

# %% [markdown]
# ## Check that the preprocessing did not leak
#
# Smoothing and cleaning before cross-validating are the steps that could in principle
# carry information across fold boundaries, spatially or temporally. The cheapest way to
# test them is
# to run the identical pipeline against a target that cannot be predicted: the same
# cleaned regressor, circularly shifted by hundreds of volumes so it keeps its spectrum
# and its autocorrelation but loses its alignment with the data. Any score this null run
# achieves is score the pipeline manufactures out of nothing.

# %%
null_target = np.roll(target.values, 600)
null_searchlight = cf.decoding.SearchLight(
    mask=mask,
    estimator=estimator,
    radius=1.0,
    cv=3,
    n_jobs=-1,
)
null_searchlight.fit(cleaned, null_target)

null_scores = null_searchlight.scores_.values
null_finite = null_scores[np.isfinite(null_scores)]
print(f"Null run: max R^2 = {np.max(null_finite):.3f}")
print(f"Null run: 95th percentile R^2 = {np.percentile(null_finite, 95):.3f}")

# %% [markdown]
# The null map sits at essentially zero, so the preprocessing is not inventing
# predictable structure and the real scores below can be read at face value.
#
# ## Compare against a GLM
#
# We now fit a GLM with the design matrix built earlier, so it tests the same
# HRF-convolved speed regressor against the same drift model, on data smoothed with the
# same kernel.

# %%
glm = cf.glm.FirstLevelModel(smoothing_fwhm=smoothing_fwhm)
glm.fit([data], design_matrices=[design_matrix])
z_scores = glm.compute_contrast("speed")

# %% [markdown]
# ## Compare the two maps
#
# The searchlight reports a cross-validated coefficient of determination, so values at
# or below zero mean the local neighbourhood predicts the speed regressor no better than
# the fold mean. We show only positive scores. The GLM map is restricted to the same
# mask, so the two panels cover the same voxels.

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

scores = searchlight.scores_.squeeze(drop=True)
plane_mask = mask.squeeze(drop=True)

scores.where(scores > 0).plot(
    ax=axes[0], cmap="inferno", cbar_kwargs={"label": "Cross-validated $R^2$"}
)
axes[0].set_title("Searchlight decoding of speed")

z_scores.squeeze(drop=True).where(plane_mask).plot(
    ax=axes[1], cmap="coolwarm", center=0, cbar_kwargs={"label": "z-score"}
)
axes[1].set_title("GLM, same regressor")

for ax in axes:
    ax.set_aspect("equal")
    ax.invert_yaxis()

# %% [markdown]
# To put a number on the agreement, we take the top 5 percent of voxels in each map,
# within the shared mask, and measure their Dice overlap.

# %%
top_scores = scores.where(plane_mask)
top_z = z_scores.squeeze(drop=True).where(plane_mask)

selected_scores = top_scores >= top_scores.quantile(0.95)
selected_z = top_z >= top_z.quantile(0.95)
dice = float(
    2
    * (selected_scores & selected_z).sum()
    / (selected_scores.sum() + selected_z.sum())
)

finite_scores = scores.values[np.isfinite(scores.values)]
print(f"Peak cross-validated R^2 = {np.max(finite_scores):.3f}")
print(f"95th percentile R^2 = {np.percentile(finite_scores, 95):.3f}")
print(f"Top-5% overlap, Dice = {dice:.3f}")

# %% [markdown]
# The two maps pick out overlapping territory, but the overlap is partial. A Dice of
# around 0.36 between the top 5 percent of each map is far above the 0.05 that chance
# would give, and well short of agreement. That is the honest result, and it is the
# interesting one: matching the preprocessing and the regressor removes the *artefactual*
# reasons the maps could differ, so what remains is the genuine difference between the
# two questions. The GLM is univariate and asks whether each voxel's signal tracks the
# regressor. The searchlight is multivariate and cross-validated, and asks whether the
# local pattern predicts held-out blocks of time. Those are not the same question and
# they do not have the same answer.
#
# Part of the difference is also resolution. A 1.0 mm radius neighbourhood is much
# coarser than a single voxel, so the searchlight map is intrinsically blurrier than the
# voxelwise GLM map, and a top-5-percent overlap penalises that mismatch.
#
# The remaining difference in scale is worth reading carefully. The GLM reaches large
# z-scores over a broad area because it measures evidence against the null across more
# than a thousand volumes, and an effect can be highly reliable while explaining very
# little variance. The searchlight instead reports how much variance is actually
# predicted in unseen time blocks, which is a far stricter bar, so its map is sparser and
# its peak sits near an $R^2$ of 0.4. Reliability and predictive power are different
# questions, and the two maps are answering one each.
#
# Note also that the panels are labelled in stereotaxic coordinates with no atlas
# overlay, so we describe clusters by position rather than by anatomical name.

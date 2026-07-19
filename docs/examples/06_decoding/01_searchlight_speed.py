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
# We reproduce the setting of [Cybis Pereira et al.
# 2026](https://doi.org/10.1016/j.celrep.2025.116791), decoding locomotion speed from a
# single coronal plane, and compare the searchlight map against a lagged GLM fit on the
# same data.

# %% [markdown]
# ## Load the recording and the tracking data

# %%
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
# The animal moves in bursts: it is stationary most of the time and occasionally runs
# fast, so the speed distribution is strongly right-skewed. That matters for decoding.
# The searchlight scores each neighbourhood with a cross-validated coefficient of
# determination, which is a squared-error measure, so a raw speed target lets the few
# fastest moments dominate the score. Because the folds are contiguous blocks of time
# and the animal does not run equally in every block, a model trained on the quieter
# blocks is then judged mostly on bursts it never saw.
#
# Compressing the target with `log1p` puts the quiet and fast regimes on a more even
# footing. We decode this transformed speed, and use the same regressor for the GLM
# below so the two maps describe the same variable.

# %%
log_speed = np.log1p(speed).rename("log_speed")

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
#   0.1 mm, so a 0.3 mm radius collects a neighbourhood about three voxels wide.
# - Consecutive fUSI volumes are strongly autocorrelated, and consecutive speed values
#   are too. Cross-validating with shuffled folds would put near-duplicate volumes in
#   both the training and test sets and inflate the scores. `SearchLight` therefore
#   builds contiguous temporal folds by default, which is what we rely on here.
#
# The estimator is a ridge regression wrapped in a `RidgeCV`, so the penalty is chosen
# by an inner cross-validation within each neighbourhood's training folds. Neighbouring
# fUSI voxels are highly correlated, and the right amount of regularisation varies
# across the plane, so fixing a single penalty by hand would favour some regions
# arbitrarily.

# %%
mean_image = data.mean("time")
mask = mean_image > mean_image.quantile(0.5)
mask = mask.drop_vars("quantile")

estimator = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 4, 9)))

searchlight = cf.decoding.SearchLight(
    mask=mask,
    estimator=estimator,
    radius=0.3,
    cv=5,
    n_jobs=-1,
)
searchlight.fit(data, log_speed)
searchlight.scores_

# %% [markdown]
# ## Compare against a lagged GLM
#
# We fit the same regressor with a GLM, sweeping a few temporal lags and keeping the
# best z-score per voxel, so that the two maps are comparable.

# %%
confounds = cf.signal.compute_compcor_confounds(
    data,
    variance_threshold=0.05,
    n_components=3,
)

events = pd.DataFrame(
    {
        "onset": data.time.values,
        "duration": data.time.volume_acquisition_duration,
        "modulation": log_speed.values,
        "trial_type": "speed",
    }
)

glm = cf.glm.FirstLevelModel(smoothing_fwhm=0.3)
design_matrix = cf.glm.make_first_level_design_matrix(
    data.time.values,
    events=events,
    drift_model="cosine",
    low_cutoff=0.01,
    confounds=confounds.values,
)

lags = range(9)
max_lag = max(lags)
n_volumes = len(design_matrix)
data_window = data.isel(time=slice(max_lag, None))
fixed_design = design_matrix.iloc[max_lag:]

z_scores = []
for lag in lags:
    design = fixed_design.copy()
    design["speed"] = (
        design_matrix["speed"].iloc[max_lag - lag : n_volumes - lag].values
    )
    glm.fit([data_window], design_matrices=[design])
    z_scores.append(glm.compute_contrast("speed"))

best_z = xr.concat(z_scores, dim="lag").max("lag")

# %% [markdown]
# ## Compare the two maps
#
# The searchlight reports a cross-validated coefficient of determination, so values at
# or below zero mean the local neighbourhood predicts speed no better than the fold
# mean. We show only positive scores. The GLM map is restricted to the same mask, so
# the two panels cover the same voxels.

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

scores = searchlight.scores_.squeeze(drop=True)
plane_mask = mask.squeeze(drop=True)

scores.where(scores > 0).plot(
    ax=axes[0], cmap="inferno", cbar_kwargs={"label": "Cross-validated $R^2$"}
)
axes[0].set_title("Searchlight decoding of speed")

best_z.squeeze(drop=True).where(plane_mask).plot(
    ax=axes[1], cmap="coolwarm", center=0, cbar_kwargs={"label": "z-score"}
)
axes[1].set_title("Lagged GLM, best lag per voxel")

for ax in axes:
    ax.set_aspect("equal")
    ax.invert_yaxis()

# %% [markdown]
# The two maps highlight overlapping territory: the strongest searchlight cluster sits
# on the same dorsal structures the GLM flags most strongly. They are not identical,
# and should not be. The GLM is univariate and tests a linear relationship at a fixed
# lag, while the searchlight is multivariate, cross-validated, and rewards any locally
# distributed pattern that generalises to held-out time blocks.
#
# The difference in scale is worth noting. The GLM reaches large z-scores over a broad
# area because it measures evidence against the null across more than a thousand
# volumes, and an effect can be highly reliable while still explaining little variance.
# The searchlight instead reports how much variance is actually predicted in unseen
# time blocks, which is a much stricter bar, so its map is sparser and its peak sits
# near an $R^2$ of 0.12. Reliability and predictive power are different questions, and
# the two maps are answering one each.

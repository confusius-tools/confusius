# %% [markdown]
# # First-level GLM analysis of fUSI data
#
# The General Linear Model (GLM) is the workhorse of task-based neuroimaging analysis.
# It treats every voxel independently and asks a simple question: how much of this
# voxel's time course can be explained by the experimental paradigm, once we account for
# nuisance signals such as slow drifts and physiological noise? Fitting that model
# voxel-by-voxel turns a fUSI recording into a statistical map that highlights where the
# brain responded to the stimulus.
#
# In this example we run a complete first-level (single-subject) GLM on stimulus-evoked
# fUSI data from the [Khallaf et al. 2026
# dataset](https://doi.org/10.17617/3.7QCU1F)—functional ultrasound of a naked mole-rat
# exposed to repeated olfactory stimulation. The notebook will go through the following
# steps:
#
# 1. **Fetch and load** the five runs of the recording.
# 2. **Register** the recording to a reference template aligned with an anatomical atlas,
#    so we can define masks and draw region boundaries in a common space.
# 3. **Build a design matrix** per run from the stimulation events, a fUSI-specific HRF,
#    a drift model, and CompCor noise regressors.
# 4. **Fit** a [`FirstLevelModel`][confusius.glm.first_level.FirstLevelModel] across all
#    runs and **compute a contrast** for the stimulation condition.
# 5. **Threshold** the resulting map for statistical significance and visualize it.
#
#
# ## Fetch the olfactory-stimulation recordings
#
# As you may already know by now, ConfUSIus provides convenient functions to download
# public datasets. Using [`fetch_khallaf_2026`][confusius.datasets.fetch_khallaf_2026],
# we take subject `5622`, session `IPM`, and the `resampled` reconstruction (the runs
# already aligned to a common within-session space), which yields five task runs.

# %%
import warnings
from functools import partial
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import confusius as cf

# Adapt background color to the current Matplotlib style.
bg_color = mpl.colors.to_hex(mpl.rcParams["figure.facecolor"])

# Keep notebook output compact for large DataArray displays.
xr.set_options(display_expand_data=False)

bids_root = cf.datasets.fetch_khallaf_2026(
    datasets="rawdata",
    subjects="5622",
    sessions="IPM",
    reconstruction="resampled",
)

pwd_path_pattern = (
    Path(bids_root)
    / "sub-5622"
    / "ses-IPM"
    / "fusi"
    / "sub-5622_ses-IPM_task-olfactory_rec-resampled_run-*_space-5622run1_pwd.nii"
)

pwd_paths = sorted(Path(bids_root).rglob(str(pwd_path_pattern.relative_to(bids_root))))

# %% [markdown]
# The experimental paradigm is a **block design**: the stimulus (odour puff) is
# presented in five 30 s blocks (`trial_type` `"active"`, onsets at 30, 90, 150, 210 and
# 270 s) interleaved with 30 s of baseline. The single `events.tsv` table in the root of
# the dataset applies to every run, capturing the same `"active"` regressor that we are
# interested in.

# %%
events_path = Path(bids_root) / "events.tsv"
events = pd.read_csv(events_path, sep="\t")
events

# %% [markdown]
# ## Load the recordings and fix their spatial convention
#
# This dataset is stored with a spatial convention that differs from the [spatial
# convention that ConfUSIus assumes](../../../user-guide/spatial-conventions/) (the axes
# are labelled differently, the depth direction is flipped, and the stored
# physical-space affine is not metric). None of this is exactly a problem or specific to
# the GLM, but we need to force some changes to make the example run smoothly after
# that. The `_load_and_prepare_fusi` helper that we are using is described in the
# collapsed admonition. Feel free to go through it if you are interested (the docstring
# explains each step), otherwise jump to the next cell.The docstring documents each
# step in detail.


# %% [tags]
def _load_and_prepare_fusi(pwd_path: Path) -> xr.DataArray:
    """Load fUSI data and prepare geometry for analysis and visualization.

    The Khallaf et al. 2026 dataset is stored with a different spatial convention than
    the one ConfUSIus is comfortable with. The main differences are:

    - The elevation axis is the "y" axis in the dataset, not "z".
    - The axial/depth axis is the "z" axis in the dataset, not "y".
    - The axial/depth direction goes "towards" the transducer, not "away" from it,
      which is the default in ConfUSIus.
    - The `sform` of the files point to a physical space that is not metric and cannot
      be currently used (but the `sform_code` is valid, making the coordinates on
      loading to be at this space).

    In this function, we prepare the data as follows:

    1. Transpose "z" and "y".
    2. Match transpose "z" and "y" in the `physical_to_qform` affine.
    3. Apply the `physical_to_qform` to the coordinates to have metric coordinates.
    4. Shift the origin of the coordinates to the corner of the volume (nice to have).
    5. Convert the coordinates units to millimeter (nice to have).
    6. Flip the "y" axis to have a depth direction "away" from the tranducer.

    """
    # 1. Transpose "z" and "y" and rename the coordinates accordingly.
    with warnings.catch_warnings():
        # This dataset omits FrameAcquisitionDuration, which the fUSI-BIDS validator
        # flags on load; it is irrelevant to this analysis, so we silence it.
        warnings.filterwarnings(
            "ignore", message="fUSI-BIDS validation warning", category=UserWarning
        )
        da = cf.load(pwd_path)

    transpose_dims = ("y", "z", "x")
    if "time" in da.dims:
        transpose_dims = ("time",) + transpose_dims

    da = da.transpose(*transpose_dims).rename(y="z", z="y")

    # 2. Match transpose "z" and "y" in the `physical_to_qform` affine.
    transpose_z_y = np.eye(4)
    transpose_z_y[[0, 1]] = 0
    transpose_z_y[[0, 1], [1, 0]] = 1
    da.affines["physical_to_qform"] = (
        transpose_z_y.T @ da.affines["physical_to_qform"] @ transpose_z_y
    )

    # 3. Apply the `physical_to_qform` to the coordinates to have metric coordinates.
    da.fusi.affine.apply(da.affines["physical_to_qform"], inplace=True)

    # 4. Shift the origin of the coordinates to the corner of the volume (nice to have).
    shift_zyx = np.eye(4)
    shift_zyx[:3, 3] = -np.array([da.z.min(), da.y.min(), da.x.min()])
    da.fusi.affine.apply(shift_zyx, inplace=True)

    # 5. Convert the coordinates units to millimeter (nice to have).
    m_to_mm = np.eye(4)
    m_to_mm[:3, :3] *= 1e3
    da.fusi.affine.apply(m_to_mm, inplace=True)
    for dim in ("x", "y", "z"):
        da.coords[dim].attrs["voxdim"] = da.coords[dim].voxdim * 1e3
        da.coords[dim].attrs["units"] = "mm"

    # 6. Flip the "y" axis to have a depth direction "away" from the tranducer.
    flip_y = np.eye(4)
    flip_y[1, 1] = -1
    flip_y[1, 3] = da.y.max().item() + da.y.min().item()
    da.fusi.affine.apply(flip_y, inplace=True)
    da = da.isel(y=slice(None, None, -1))

    flip_z = np.eye(4)
    flip_z[0, 0] = -1
    flip_z[0, 3] = da.z.max().item() + da.z.min().item()
    da.fusi.affine.apply(flip_z, inplace=True)
    da = da.isel(z=slice(None, None, -1))

    return da


# %%
fusi_list = [_load_and_prepare_fusi(pwd_path) for pwd_path in pwd_paths]

fusi_list[0]

# %% [markdown]
# Averaging each run over time and then across runs gives a single, high-SNR
# power-Doppler volume. This averaged image carries no task information, but it is a
# clean anatomical reference that we use to drive registration in the next step.

# %%
average_fusi = xr.concat([fusi.mean("time") for fusi in fusi_list], dim="extra").mean(
    "extra"
)

# %% [markdown]
# ## Bring the data into a common anatomical space
#
# To interpret the activation map anatomically (and to define the masks the GLM needs)
# we align the recording to the
# [Pepe, Mariani et al. 2026 mouse fUSI template][confusius.datasets.fetch_template_pepe_mariani_2026],
# which is itself aligned to the Allen Mouse Brain atlas. This gives us a common space
# in which we can draw region boundaries and pull out anatomically defined masks.
#
# We register in two passes with
# [`register_volume`][confusius.registration.register_volume]: a first affine fit on the
# linear-scale mean image, then a refinement initialized from that result and run on the
# power-scaled image, which sharpens the alignment. Each call returns the (resampled)
# moving image, the estimated transform, and a diagnostics object; here we only keep the
# transform.
#
# !!! note
#     Registration results are sensitive to their arguments. See the
#     [registration examples](../../#registration)
#     for guidance on inspecting convergence and tuning the optimizer.

# %%
template_pepe_mariani = cf.datasets.fetch_template_pepe_mariani_2026()
atlas = cf.atlas.Atlas.from_brainglobe("allen_mouse_100um")

napari_transform = np.array(
    [
        [0.9964051642005587, 0.08471569367276423, 0.0, 1.017912047433129],
        [-0.08471569367276423, 0.9964051642005588, 0.0, -1.100018006023873],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

_, transform, _ = cf.registration.register_volume(
    average_fusi.fusi.scale.power(),
    template_pepe_mariani,
    transform_type="affine",
    learning_rate="auto",
    initialization=np.linalg.inv(napari_transform),
)

# %% [markdown]
# The refined transform lets us express the recording's physical-to-atlas mapping
# (`physical_to_sform`). With that affine in hand,
# [`resample_like`][confusius.registration.resample_like] reslices each volume onto the
# atlas grid. We resample the mean image (for display) and every individual run (to feed
# the GLM in atlas space).

# %%
average_fusi.affines["physical_to_sform"] = template_pepe_mariani.affines[
    "physical_to_sform"
] @ np.linalg.inv(transform)

resampled_average_in_atlas = cf.registration.resample_like(
    average_fusi,
    atlas.annotation,
    np.linalg.inv(average_fusi.affines["physical_to_sform"]),
)

resampled_fusi_list = []
for fusi in fusi_list:
    fusi.affines["physical_to_sform"] = average_fusi.affines["physical_to_sform"]
    resampled_fusi = cf.registration.resample_like(
        fusi,
        atlas.annotation,
        np.linalg.inv(fusi.affines["physical_to_sform"]),
    )
    resampled_fusi_list.append(resampled_fusi)

# %% [markdown]
# ## Choose a hemodynamic response function
#
# A stimulus does not produce an instantaneous change in the power-Doppler signal: the
# vascular response is delayed and smeared out in time. The GLM accounts for this by
# convolving the stimulation boxcar with a **hemodynamic response function (HRF)**. fUSI
# responses are faster and lack the post-stimulus undershoot of the BOLD signal, so we
# use [`claron2021_hrf`][confusius.glm.claron2021_hrf], an inverse-gamma HRF proposed for
# functional ultrasound, rather than a canonical BOLD HRF. We tune its `beta` scale
# parameter to `6.7` to better match this dataset's response; `partial` binds that value
# so the model can call the HRF with just the sampling interval.

# %%
modified_claron2021 = partial(cf.glm.claron2021_hrf, beta=6.7)

# Sample the kernel on a fine grid to visualize its shape.
hrf_kernel = modified_claron2021(dt=1.0)
hrf_time = np.linspace(0, 32, len(hrf_kernel))

fig, ax = plt.subplots(figsize=(7, 3), facecolor=bg_color)
ax.plot(hrf_time, hrf_kernel, color="#d93a54")
ax.set_xlabel("Time since stimulus onset (s)")
ax.set_ylabel("Response (a.u.)")
_ = ax.set_title("Claron et al. 2021 fUSI HRF (beta=6.7)")

# %% [markdown]
# ## Model physiological noise with CompCor
#
# Beyond the stimulus response, the signal contains structured nuisance fluctuations.
# [`compute_compcor_confounds`][confusius.signal.compute_compcor_confounds] extracts the
# leading principal components from a noise region — here the atlas `"fiber tracts"`,
# which carries little task signal — and we add them to the design as nuisance
# regressors (anatomical CompCor). We take three components per run.

# %%
mask = atlas.get_masks("fiber tracts").astype(bool)

# %% [markdown]
# ## Build the design matrix
#
# [`make_first_level_design_matrix`][confusius.glm.make_first_level_design_matrix]
# assembles, per run, the full set of regressors: the HRF-convolved `"active"`
# condition, the CompCor confounds, and a `"cosine"` **drift model** that high-pass
# filters slow scanner/physiological drifts below `0.01` Hz, plus a constant baseline
# column. Each run gets its own design matrix because its CompCor regressors are
# estimated from its own data.

# %%
design_matrices = []
for fusi in resampled_fusi_list:
    confounds = cf.signal.compute_compcor_confounds(
        fusi,
        mask.sel(mask="fiber tracts"),
        n_components=3,
    )

    design_matrices.append(
        cf.glm.make_first_level_design_matrix(
            fusi.time.values,
            events=events,
            hrf_model=modified_claron2021,
            drift_model="cosine",
            low_cutoff=0.01,
            confounds=confounds.to_numpy(),
            confound_names=confounds.component.values,
        )
    )

# %% [markdown]
# Visualizing the first run's design matrix makes the model concrete. Each column is a
# regressor and each row a volume (time runs top to bottom). The leftmost `active`
# column shows the HRF-convolved stimulation blocks; the CompCor and drift columns
# follow, and the constant column models the baseline.

# %%
design_matrix = design_matrices[0]

fig, ax = plt.subplots(figsize=(6, 5), facecolor=bg_color)
norm = mpl.colors.CenteredNorm()
ax.imshow(design_matrix.values, aspect="auto", norm=norm, interpolation="nearest")
ax.set_xticks(range(design_matrix.shape[1]))
ax.set_xticklabels([str(c) for c in design_matrix.columns], rotation=90, fontsize=8)
ax.set_ylabel("Volume")
_ = ax.set_title("Design matrix (run 1)")

# %% [markdown]
# ## Fit the first-level GLM
#
# [`FirstLevelModel`][confusius.glm.first_level.FirstLevelModel] fits the design to the
# data voxel by voxel. We apply a light Gaussian spatial smoothing (0.3 mm FWHM per axis)
# to boost SNR, and restrict the fit to the whole-brain `"root"` mask from the atlas.
# Passing the list of runs together combines them with a fixed-effects model; by default
# an AR(1) noise model accounts for temporal autocorrelation in the residuals.

# %%
glm = cf.glm.FirstLevelModel(
    smoothing_fwhm={"z": 0.3, "y": 0.3, "x": 0.3}, mask=atlas.get_masks("root")[0]
)
glm.fit(resampled_fusi_list, design_matrices=design_matrices)

# %% [markdown]
# ## Compute and display the activation map
#
# [`compute_contrast`][confusius.glm.first_level.FirstLevelModel.compute_contrast] turns
# the fitted model into a statistical map. Asking for the `"active"` contrast tests, at
# every voxel, whether the stimulation regressor has a non-zero effect, and returns a
# z-score map. We display it over a range of atlas slices with the region boundaries
# drawn on top for anatomical context.

# %%
z_score = glm.compute_contrast("active")

slice_coords = resampled_average_in_atlas.z[
    (resampled_average_in_atlas.z > 5) & (resampled_average_in_atlas.z < 9)
][::4]

vmax = 10
fig = cf.plotting.plot_stat_map(
    z_score,
    slice_coords=slice_coords,
    nrows=2,
    vmax=vmax,
    bg_color=bg_color,
)
fig.add_contours(
    atlas.annotation,
    linewidths=0.6,
    slice_coords=slice_coords,
    alpha=0.4,
)

# %% [markdown]
# ## Threshold for statistical significance
#
# The raw z-map shows an effect at every voxel; we need to keep only those that are
# statistically significant while controlling for the many thousands of simultaneous
# tests. [`apply_statistical_threshold`][confusius.glm.apply_statistical_threshold]
# applies a multiple-comparison correction — here Holm family-wise-error control at
# `alpha=0.01` — followed by a cluster-extent threshold that drops surviving clusters
# smaller than 50 voxels. We then set the zeroed-out voxels to `NaN` so they render
# transparently when overlaid.

# %%
thresholded_zscore, threshold = cf.stats.apply_statistical_threshold(
    z_score,
    mask=atlas.get_masks("root")[0],
    alpha=0.01,
    method="holm",
    cluster_threshold=50,
)

thresholded_zscore = thresholded_zscore.where(thresholded_zscore != 0, np.nan)

# %% [markdown]
# !!! tip "Explore the result interactively"
#     The thresholded map, the anatomical reference, and the atlas labels can be loaded
#     together into a [napari](https://napari.org/) viewer for 3D exploration:
#
#     ```python
#     viewer, _ = resampled_average_in_atlas.fusi.plot()
#     viewer, _ = thresholded_zscore.fusi.plot(
#         viewer=viewer, contrast_limits=(-vmax, vmax)
#     )
#     cf.plotting.plot_napari(atlas.annotation, viewer=viewer, layer_type="labels")
#     ```

# %% [markdown]
# Finally, we overlay the significant activation on the mean power-Doppler image (in
# decibels) with the atlas contours, giving a single figure that places the
# statistically significant odour response in its anatomical context.

# %% tags=["thumbnail"]
fig = cf.plotting.plot_volume(
    resampled_average_in_atlas.fusi.scale.db(),
    slice_coords=slice_coords,
    nrows=3,
    vmin=-20,
    vmax=0,
    bg_color=bg_color,
)
fig.add_volume(
    thresholded_zscore,
    slice_coords=slice_coords,
    alpha=None,
    vmin=-vmax,
    vmax=vmax,
)
fig.add_contours(
    atlas.annotation,
    linewidths=0.6,
    slice_coords=slice_coords,
    alpha=0.4,
)

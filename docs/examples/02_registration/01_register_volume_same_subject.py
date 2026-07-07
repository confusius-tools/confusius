# %% [markdown]
# # Registration of two sessions from the same subject
#
# This example shows how to align two power Doppler images acquired from the same
# subject in different sessions. We use
# [`register_volume`][confusius.registration.register_volume] with a rigid transform,
# which is appropriate when the imaged anatomy is the same but the probe placement
# differs slightly between the two recordings.
#
# We pick two `angio` acquisitions from the [Cybis Pereira 2026
# dataset](https://doi.org/10.1016/j.celrep.2025.116791) using
# [`fetch_cybis_pereira_2026`][confusius.datasets.fetch_cybis_pereira_2026]: subject
# `rat75`, slice `slice32`, recorded on consecutive days (sessions `20220523` and
# `20220524`).

# %% [markdown]
# ## Fetch and load both recordings
#
# Each recording is a single power Doppler image of one slice. We convert it to decibels
# for both display and registration, which is usually more stable on the log-compressed
# dynamic range. We also set the coordinate system to the NIfTI `qform`'s space, because
# the `sform` coordinate system is a weird non-metric space that is not consistent
# across session.

# %%
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import xarray as xr

import confusius as cf

# Adapt background color to the current Matplotlib style.
bg_color = mpl.colors.to_hex(mpl.rcParams["figure.facecolor"])

xr.set_options(display_expand_data=False)

sessions = ["20220523", "20220524"]
acq = "slice32"
bids_root = cf.datasets.fetch_cybis_pereira_2026(
    datasets="rawdata",
    subjects="rat75",
    sessions=sessions,
    datatypes="angio",
    acqs=acq,
)


def _load_angio_for_registration(session: str) -> xr.DataArray:
    """Load an angio acquisition and scale its intensity for registration."""
    path = (
        Path(bids_root)
        / "sub-rat75"
        / f"ses-{session}"
        / "angio"
        / f"sub-rat75_ses-{session}_acq-{acq}_rec-minframe2d_pwd.nii.gz"
    )
    angio = cf.load(path).fusi.scale.db().compute()
    return angio.fusi.affine.apply(angio.affines["physical_to_qform"])[0]


fixed = _load_angio_for_registration(sessions[0])

fixed
# %%
moving = _load_angio_for_registration(sessions[1])

moving

# %% [markdown]
# ## Inspect the misalignment before registration
#
# We visualise the alignment with
# [`plot_composite`][confusius.plotting.plot_composite], which resamples `moving` onto
# `fixed`'s grid and draws the two as a red/cyan composite: matched anatomy appear in
# white, while any residual red/cyan fringe reveals the displacement that
# [`register_volume`][confusius.registration.register_volume] will correct.

# %%
cf.plotting.plot_composite(fixed, moving, bg_color=bg_color)

# %% [markdown]
# ## Run the registration
#
# A rigid transform captures the rotation and translation difference between the two
# sessions. [`register_volume`][confusius.registration.register_volume] returns three
# values:
#
# 1. the moving image (only aligned to the fixed grid if `resample=True` is used);
# 2. the rigid transform matrix that maps fixed-physical coordinates to moving-physical
#    coordinates;
# 3. a [`RegistrationDiagnostics`][confusius.registration.RegistrationDiagnostics]
#    dataclass holding the per-iteration metric values and the optimizer stop
#    condition, which we use below to plot the convergence curve.

# !!! tip "Watch registration progress live"
#     Pass `show_progress=True` to
#     [`register_volume`][confusius.registration.register_volume] to follow the
#     optimization in real time. A live matplotlib window opens during the call and
#     updates at every iteration with both the similarity-metric curve and a
#     fixed/moving composite overlay. It is the fastest way to tell whether the
#     optimizer is making progress, stuck in a local minimum, or diverging—and to
#     decide which arguments to tweak from the warning above.
#
# !!! warning "Registration is sensitive to its arguments"
#     The result depends heavily on the choice of `transform_type`, `metric`,
#     `learning_rate`, `number_of_iterations`, `convergence_window_size`,
#     `initialization`, and the multi-resolution settings (`use_multi_resolution`,
#     `shrink_factors`, `smoothing_sigmas`). The default values are usually a good
#     starting point and work well in many cases, but you should definitely try
#     different arguments (start with the default!). If the result is not satisfactory,
#     inspect the
#     [`RegistrationDiagnostics`][confusius.registration.RegistrationDiagnostics]
#     convergence curve and the post-registration overlay, and sweep these arguments
#     until you get a stable, well-converged result.

# %%
registered, rigid_transform, diagnostics = cf.registration.register_volume(
    moving=moving,
    fixed=fixed,
    transform_type="rigid",
    learning_rate=1.0,
    show_progress=True,
)

print(f"Iterations: {diagnostics.n_iterations}")
print(f"Final metric: {diagnostics.final_metric_value:.4f}")
print(f"Stop condition: {diagnostics.stop_condition}")
rigid_transform

# %% [markdown]
# ## Check the alignment after registration
#
# Plotting the same fixed/moving overlay before and after registration makes the
# correction obvious: the red/cyan fringe in the first panel should be replaced
# by a more uniformly desaturated grey in the second.

# %% tags=["thumbnail"]
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.patch.set_facecolor(bg_color)

for ax, moving_view, title in [
    (axes[0], moving, "Before"),
    (axes[1], registered, "After"),
]:
    cf.plotting.plot_composite(fixed, moving_view, axes=ax, bg_color=bg_color)
    ax.set_title(title)

_ = fig.suptitle("Fixed (red) / moving (cyan)")

# %% [markdown]
# ## Inspect convergence with the registration diagnostics
#
# `diagnostics.metric_values` holds the optimizer's similarity-metric value at each
# iteration. With the default `metric="correlation"`, SimpleITK minimizes the negative
# normalized cross-correlation, so a lower (more negative) value means a better fit.
# The curve typically drops sharply at the start and then plateaus.
#

# %%
fig, ax = plt.subplots(figsize=(7, 3))
fig.patch.set_facecolor(bg_color)
ax.plot(diagnostics.metric_values, color="#d93a54")
ax.set_xlabel("Iteration")
ax.set_ylabel(f"Similarity metric ({diagnostics.metric})")
_ = ax.set_title(diagnostics.stop_condition)

# %% [markdown]
# The resulting rigid transform is encoded in physical units and can be reused, composed
# with other transforms, or applied to additional volumes from the same session with
# [`resample_volume`][confusius.registration.resample_volume].

# %% [markdown]
# ## Refine with a B-spline transform
#
# A rigid transform only models a single global rotation and translation. Any residual
# local mismatch—e.g. small elastic tissue deformation between sessions—needs a
# nonrigid model. [`register_volume`][confusius.registration.register_volume] with
# `transform_type="bspline"` fits a local displacement field on top of an initial
# transform, here the rigid transform found above.
#
# !!! warning "B-spline registration may need different parameters than rigid"
#     Successful non-linear registration is often very dependent on the choice of
#     parameters. If the B-spline fails to converge, or if defomations are unrealistic,
#     try adjusting the `learning_rate`, `mesh_size`, and multi-resolution settings. The
#     default values are usually a good starting point, but you may need to experiment
#     to find the right combination for your data. Here, we use a non-default mesh size
#     of `(6, 6, 6)` to allow less local deformations that the default `(10, 10, 10)`
#     mesh would allow.

# %%
registered_bspline, bspline_transform, diagnostics_bspline = (
    cf.registration.register_volume(
        moving=moving,
        fixed=fixed,
        transform_type="bspline",
        mesh_size=(6, 6, 6),
        learning_rate=1.0,
        initialization=rigid_transform,
        show_progress=True,
    )
)

print(f"Iterations: {diagnostics_bspline.n_iterations}")
print(f"Final metric: {diagnostics_bspline.final_metric_value:.4f}")
print(f"Stop condition: {diagnostics_bspline.stop_condition}")
bspline_transform

# %% [markdown]
# ## Check the alignment after B-spline refinement
#
# Comparing the rigid-only result against the B-spline-refined result shows the extra
# local correction the B-spline step adds on top of rigid, especially in deeper vessels
# where the brain might have contracted slightly between sessions.

# %%
print(f"Rigid final metric: {diagnostics.final_metric_value:.4f}")
print(f"Rigid + B-spline final metric: {diagnostics_bspline.final_metric_value:.4f}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.patch.set_facecolor(bg_color)

for ax, moving_view, title in [
    (axes[0], registered, "Rigid"),
    (axes[1], registered_bspline, "Rigid + B-spline"),
]:
    cf.plotting.plot_composite(fixed, moving_view, axes=ax, bg_color=bg_color)
    ax.set_title(title)

_ = fig.suptitle("Fixed (red) / moving (cyan)")

# %% [markdown]
# ## Inspect B-spline convergence

# %%
fig, ax = plt.subplots(figsize=(7, 3))
fig.patch.set_facecolor(bg_color)
ax.plot(diagnostics_bspline.metric_values, color="#d93a54")
ax.set_xlabel("Iteration")
ax.set_ylabel(f"Similarity metric ({diagnostics_bspline.metric})")
_ = ax.set_title(diagnostics_bspline.stop_condition)

# %% [markdown]
# Unlike the rigid transform, `bspline_transform` is a control-point `xarray.DataArray`
# rather than a homogeneous matrix—it has no closed-form inverse. To apply its
# approximate inverse as a warp on other data, sample it into a dense displacement field
# with
# [`bspline_to_displacement_field`][confusius.registration.bspline_to_displacement_field]
# and invert that field with
# [`invert_displacement_field`][confusius.registration.invert_displacement_field].

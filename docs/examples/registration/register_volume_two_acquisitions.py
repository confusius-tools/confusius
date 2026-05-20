# %% [markdown]
# # Registration of two acquisitions
#
# This example shows how to align two power Doppler images acquired from the same
# animal in different sessions. We use
# [`register_volume`][confusius.registration.register_volume] with an affine transform,
# which is appropriate when the imaged anatomy is the same but the probe placement
# differs slightly between the two recordings.
#
# We pick two `angio` acquisitions from the Cybis Pereira 2026 dataset using
# [`fetch_cybis_pereira_2026`][confusius.datasets.fetch_cybis_pereira_2026]: subject
# `rat75`, slice `slice32`, recorded on consecutive days (sessions `20220523` and
# `20220524`).

# %% [markdown]
# ## Fetch and load both recordings
#
# Each recording is a single power Doppler image of one slice. We convert it to decibels
# for both display and registration, which is usually more stable on the log-compressed
# dynamic range.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import xarray as xr

import confusius as cf
from confusius.datasets import fetch_cybis_pereira_2026
from confusius.registration import register_volume

# Keep figure backgrounds transparent in docs and standalone notebooks.
bg_color = "none"

xr.set_options(display_expand_data=False)

sessions = ["20220523", "20220524"]
acq = "slice32"
bids_root = fetch_cybis_pereira_2026(
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
    return cf.load(path).fusi.scale.db().compute()


fixed = _load_angio_for_registration(sessions[0])
moving = _load_angio_for_registration(sessions[1])

fixed
# %%
moving

# %% [markdown]
# ## Inspect the misalignment before registration
#
# The two images share anatomy but live on slightly different grids because the probe
# was re-positioned between sessions. We visualise the alignment with
# [`plot_composite`][confusius.plotting.plot_composite], which resamples `moving` onto
# `fixed`'s grid and draws the two as a red/cyan composite: matched anatomy appear in
# white, while any residual red/cyan fringe reveals the displacement that
# `register_volume` will correct.
#
# One subtlety: the two `angio` recordings sit at slightly different `z`
# coordinates in physical space, so resampling `moving` onto `fixed`'s grid would
# place every voxel outside the fixed slab and return an empty image. We force
# `moving.z` to match `fixed.z` before plotting so the overlay actually has
# something to show. The remaining in-plane misalignment is what we're after.

# %%
moving.coords["z"] = fixed.z
plotter = cf.plotting.plot_composite(fixed, moving, bg_color=bg_color)
plotter.show()

# %% [markdown]
# ## Run the registration
#
# An affine transform captures the rotation, translation, scaling and shear difference
# between the two sessions. [`register_volume`][confusius.registration.register_volume]
# returns three values:
#
# 1. the moving image (only aligned to the fixed grid if `resample=True` is used);
# 2. the affine matrix that maps fixed-physical coordinates to moving-physical coordinates;
# 3. a [`RegistrationDiagnostics`][confusius.registration.RegistrationDiagnostics]
#    dataclass holding the per-iteration metric values and the optimizer stop
#    condition, which we use below to plot the convergence curve.
#
# !!! warning "Registration is sensitive to its arguments"
#     The result depends heavily on the choice of `transform_type`, `metric`,
#     `learning_rate`, `number_of_iterations`, `convergence_window_size`,
#     `centering_initialization`, and the multi-resolution settings
#     (`use_multi_resolution`, `shrink_factors`, `smoothing_sigmas`). The values used in
#     this example were empirically found to work well in this case, but you should
#     definitely try different arguments (start with the default!) if the result is not
#     satisfactory—inspect the
#     [`RegistrationDiagnostics`][confusius.registration.RegistrationDiagnostics]
#     convergence curve and the post-registration overlay, and sweep these arguments
#     until you get a stable, well-converged result.

# %%
registered, affine, diagnostics = register_volume(
    moving=moving,
    fixed=fixed,
    show_progress=True,
    number_of_iterations=500,
    convergence_window_size=100,
    learning_rate=30,
)

print(f"Iterations: {diagnostics.n_iterations}")
print(f"Final metric: {diagnostics.final_metric_value:.4f}")
print(f"Stop condition: {diagnostics.stop_condition}")
affine

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
    plotter = cf.plotting.plot_composite(fixed, moving_view, axes=ax, bg_color=bg_color)
    ax.set_title(title)

fig.suptitle("Fixed (red) / moving (cyan)")
fig.show()

# %% [markdown]
# !!! tip "Watch registration progress live"
#     Pass `show_progress=True` to
#     [`register_volume`][confusius.registration.register_volume] to follow the
#     optimization in real time. A live matplotlib window opens during the call and
#     updates at every iteration with both the similarity-metric curve and a
#     fixed/moving composite overlay. It is the fastest way to tell whether the
#     optimizer is making progress, stuck in a local minimum, or diverging—and to
#     decide which arguments to tweak from the warning above.

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
ax.set_title(diagnostics.stop_condition)
fig.show()


# %% [markdown]
# The resulting affine encodes the registration transform in physical units and can be
# reused, composed with other transforms, or applied to additional volumes from the same
# session with [`resample_volume`][confusius.registration.resample_volume].

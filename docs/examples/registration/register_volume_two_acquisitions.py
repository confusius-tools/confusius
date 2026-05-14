# %% [markdown]
# # Registering two acquisitions
#
# This example shows how to align two power Doppler volumes acquired from the same
# animal in different sessions. We use
# [`register_volume`][confusius.registration.register_volume] with an affine transform,
# which is appropriate when the imaged anatomy is the same but the probe placement
# differs slightly between the two recordings.
#
# We pick two acquisitions from the Nunez-Elizalde 2022 dataset: subject `CR020`,
# slice `slice03`, task `spontaneous`, recorded two days apart (sessions `20191120`
# and `20191122`).

# %% [markdown]
# ## Fetch and load both recordings
#
# Each recording is a 2D+t power Doppler series of a single slice. We average over
# time to get one power Doppler image per session, then convert to decibels—both
# for display and for the registration itself, which is more stable on the
# log-compressed dynamic range.

# %%
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import confusius as cf
from confusius.datasets import fetch_nunez_elizalde_2022
from confusius.registration import register_volume, resample_like

# A transparent background looks better in the rendered notebook.
bg_color = mpl.colors.to_hex(mpl.rcParams["figure.facecolor"])
xr.set_options(display_expand_data=False)

bids_root = fetch_nunez_elizalde_2022(
    subjects="CR020",
    sessions=["20191120", "20191122"],
    tasks="spontaneous",
    acqs="slice03",
)


def load_mean_pwd_db(session: str) -> xr.DataArray:
    """Load and time-average the slice03 spontaneous recording, in decibels."""
    path = (
        Path(bids_root)
        / "sub-CR020"
        / f"ses-{session}"
        / "fusi"
        / f"sub-CR020_ses-{session}_task-spontaneous_acq-slice03_pwd.nii.gz"
    )
    return cf.load(path).compute().mean("time").fusi.scale.db()


fixed = load_mean_pwd_db("20191120")
moving = load_mean_pwd_db("20191122")

fixed

# %%
moving

# %% [markdown]
# ## Inspect the misalignment before registration
#
# The two volumes share anatomy but live on slightly different grids because the
# probe was re-positioned between sessions. We first put `moving` on the `fixed`
# grid with an identity transform via
# [`resample_like`][confusius.registration.resample_like]; any remaining mismatch is
# what `register_volume` will correct.
#
# To visualise the alignment we use confusius's
# [`plot_volume`][confusius.plotting.plot_volume] to draw the fixed image, then
# [`add_volume`][confusius.plotting.VolumePlotter.add_volume] to overlay the moving
# image with a translucent contrasting colormap. Overlapping anatomy renders as a
# blended purple; the red/blue fringe reveals the displacement we want to correct.

# %%
# Identity transform: no rotation/translation, just regrid moving onto fixed.
identity = np.eye(fixed.ndim + 1)
moving_on_fixed = resample_like(moving, fixed, identity)

plotter = cf.plotting.plot_volume(
    fixed,
    cmap="Reds",
    bg_color=bg_color,
    show_colorbar=False,
    show_titles=False,
)
plotter.add_volume(
    moving_on_fixed,
    cmap="Blues",
    alpha=0.5,
    match_coordinates=False,
    show_colorbar=False,
    show_titles=False,
)
plotter.figure.suptitle("Before registration — fixed (red) / moving (blue)")
plotter.figure.patch.set_alpha(0)

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
#     (`use_multi_resolution`, `shrink_factors`, `smoothing_sigmas`). The default
#     values used in this example were empirically found to work well in most
#     cases, but you should definitely try different arguments if the result is
#     not satisfactory—inspect the
#     [`RegistrationDiagnostics`][confusius.registration.RegistrationDiagnostics]
#     convergence curve and the post-registration overlay, and sweep these
#     arguments until you get a stable, well-converged result.

# %%
registered, affine, diagnostics = register_volume(
    moving=moving,
    fixed=fixed,
    convergence_window_size=50,
)

print(f"Iterations: {diagnostics.n_iterations}")
print(f"Final metric: {diagnostics.final_metric_value:.4f}")
print(f"Stop condition: {diagnostics.stop_condition}")
affine

# %% [markdown]
# ## Check the alignment after registration
#
# Plotting the same fixed/moving overlay before and after registration makes the
# correction obvious: the residual red/blue fringe in the first panel should be
# replaced by a more uniform blended purple in the second.

# %% tags=["thumbnail"]
fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="none")

for ax, moving_view, title in [
    (axes[0], moving_on_fixed, "Before"),
    (axes[1], registered, "After"),
]:
    panel = cf.plotting.plot_volume(
        fixed,
        axes=ax,
        cmap="Reds",
        bg_color=bg_color,
        show_colorbar=False,
        show_titles=False,
    )
    panel.add_volume(
        moving_view,
        cmap="Blues",
        alpha=0.5,
        match_coordinates=False,
        show_colorbar=False,
        show_titles=False,
    )
    ax.set_title(title)

fig.suptitle("Fixed (red) / moving (blue)")
fig.patch.set_alpha(0)

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
fig, ax = plt.subplots(figsize=(7, 3), facecolor="none")
ax.plot(diagnostics.metric_values, color="#d93a54")
ax.set_xlabel("Iteration")
ax.set_ylabel(f"Similarity metric ({diagnostics.metric})")
ax.set_title(diagnostics.stop_condition)
fig.patch.set_alpha(0)


# %% [markdown]
# The resulting affine encodes the registration transform in physical (millimeter)
# units and can be reused, composed with other transforms, or applied to additional
# volumes from the same session with
# [`resample_volume`][confusius.registration.resample_volume].

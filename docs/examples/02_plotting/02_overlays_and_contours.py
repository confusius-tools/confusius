# %% [markdown]
# # Overlays, ROI contours, and composites
#
# Building on [Plotting basics](plotting_basics.md), this example shows the three ways
# to combine more than one volume on the same axes: layering a statistical map over an
# anatomical background by chaining
# [`add_volume`][confusius.plotting.VolumePlotter.add_volume] onto the
# [`plot_volume`][confusius.plotting.plot_volume] call that created the figure,
# outlining atlas regions with
# [`add_contours`][confusius.plotting.VolumePlotter.add_contours], and rendering a
# red/cyan registration-QC composite with
# [`plot_composite`][confusius.plotting.plot_composite].
#
# We need an anatomical background, a statistical map, and an atlas registered into the
# recording's space to demonstrate all three. Getting there means registering a
# recording to the [Allen Mouse Brain Atlas][confusius.atlas] and computing a seed-based
# connectivity map — the same recipe as [Atlas-based seed connectivity
# maps](../connectivity/atlas_seed_map.md), collapsed below since it isn't the point of
# this example.

# %%
from pathlib import Path

import matplotlib as mpl
import numpy as np
import xarray as xr

import confusius as cf

# Adapt background color to the current Matplotlib style.
bg_color = mpl.colors.to_hex(mpl.rcParams["figure.facecolor"])

xr.set_options(display_expand_data=False)

# %% [markdown]
# ## Set up an anatomical background, a stat map, and an atlas
#
# This registers subject `CR022`'s session `20201007` recording to the [Pepe, Mariani
# 2026 fUSI template][confusius.datasets.fetch_template_pepe_mariani_2026], resamples the
# Allen atlas into the recording's native space, and computes one seed-based
# connectivity map (seeded on right retrosplenial cortex, `"RSP"`) with
# [`SeedBasedMaps`][confusius.connectivity.SeedBasedMaps]. `bg_volume` is the resampled
# Allen reference volume; `stat_map` is the RSP seed map; `atlas_native.annotation` is
# the resampled region label map.

# %% tags=["collapse: Registration and seed map setup"]
bids_root = cf.datasets.fetch_nunez_elizalde_2022(
    subjects="CR022", sessions="20201007", tasks="spontaneous", acqs="slice02"
)
data_path = (
    Path(bids_root)
    / "sub-CR022"
    / "ses-20201007"
    / "fusi"
    / "sub-CR022_ses-20201007_task-spontaneous_acq-slice02_pwd.nii.gz"
)
data = cf.timing.resample_to_uniform_time(cf.load(data_path))
moving = data.mean(dim="time").fusi.scale.db().compute()

# Copied and pasted transform after manual transformation in napari.
napari_affine = np.array(
    [
        [1.0, 0.0, 0.0, 5.594638656430411],
        [0.0, 1.0, 0.0, -2.50293925701927],
        [0.0, 0.0, 1.0, 5.6650243788545875],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
initialization = np.linalg.inv(napari_affine)

template = cf.datasets.fetch_template_pepe_mariani_2026().compute()
target_z = napari_affine[0, 3] + moving.fusi.origin["z"]
fixed = template.sel(z=slice(target_z - 1.0, target_z + 1.0))

registered, affine, _ = cf.registration.register_volume(
    moving=moving,
    fixed=fixed,
    transform_type="affine",
    metric="correlation",
    convergence_window_size=100,
    number_of_iterations=500,
    learning_rate=1,
    initialization=initialization,
    show_progress=False,
)

world_to_sform = template.attrs["affines"]["world_to_sform"]
subject_to_atlas = world_to_sform @ np.linalg.inv(affine)

atlas = cf.datasets.fetch_brainglobe_atlas("allen_mouse_100um", check_latest=False)
atlas_native = atlas.atlas.resample_like(moving, subject_to_atlas)
bg_volume = atlas_native.reference

seed_mask = atlas_native.atlas.get_masks("RSP", sides="right").isel(mask=0)
white_matter = atlas_native.atlas.get_masks("fiber tracts").isel(mask=0)
acompcor = cf.signal.compute_compcor_confounds(
    data, noise_mask=white_matter, n_components=1, variance_threshold=0.95
)

mapper = cf.connectivity.SeedBasedMaps(
    seed_masks=seed_mask.expand_dims(mask=["RSP_R"]),
    clean_kwargs={"low_cutoff": 0.01, "filter_method": "cosine", "confounds": acompcor},
)
mapper.fit(data)
stat_map = mapper.maps_  # A single seed region: `maps_`'s "region" dim is squeezed out.

# %% [markdown]
# ## Overlay a statistical map on an anatomical background
#
# `plot_volume` draws the background and returns the plotter managing the figure;
# chaining `add_volume` on it overlays the stat map onto the same axes, matched by
# `slice_mode` coordinate. The stat map's own colormap fully covers the background
# wherever it has a value; `threshold` masks out weak correlations so the anatomical
# background shows through there instead. `add_stat_map` does the same with the
# automatic colormap and range of `plot_stat_map`; `add_volume` is used here to pick
# an explicit colormap and range.

# %%
plotter = cf.plotting.plot_volume(bg_volume, cmap="gray", show_colorbar=False, bg_color=bg_color)
_ = plotter.add_volume(
    stat_map,
    cmap="berlin",
    vmin=-0.8,
    vmax=0.8,
    threshold=0.2,
    cbar_label="Pearson correlation",
)

# %% [markdown]
# `alpha` blends the overlay with the background instead of fully covering it — useful
# when you want the anatomy to stay visible everywhere, not just outside the
# thresholded stat map.

# %%
plotter = cf.plotting.plot_volume(bg_volume, cmap="gray", show_colorbar=False, bg_color=bg_color)
_ = plotter.add_volume(
    stat_map, cmap="berlin", vmin=-0.8, vmax=0.8, threshold=0.2, alpha=0.6
)

# %% [markdown]
# ## Outline ROIs with add_contours
#
# [`add_contours`][confusius.plotting.VolumePlotter.add_contours] draws contour lines
# for an integer label map on top of whatever is already on the axes — here, the
# resampled Allen annotation volume overlaid on the anatomical background. Leaving
# `colors` unset draws each region in its canonical Allen color, read from the label
# map's `attrs["cmap"]`/`attrs["norm"]` (the same convention `get_masks` uses).
#
# Passing `roi_labels` makes hovering the cursor over a voxel show the region name in
# the matplotlib status bar — useful when exploring a figure interactively, though it
# has no effect on a static, saved image. `atlas.lookup["acronym"]`, keyed by structure
# id, is exactly the `dict[int, str]` shape `roi_labels` expects.

# %% tags=["thumbnail"]
roi_labels = atlas_native.atlas.lookup["acronym"].to_dict()

plotter = cf.plotting.plot_volume(bg_volume, cmap="gray", show_colorbar=False, bg_color=bg_color)
_ = plotter.add_contours(atlas_native.annotation, roi_labels=roi_labels)

# %% [markdown]
# A single seed's mask — a flat, non-overlapping label map — draws the same way, with
# an explicit color instead of the atlas's per-region palette.

# %%
plotter = cf.plotting.plot_volume(bg_volume, cmap="gray", show_colorbar=False, bg_color=bg_color)
_ = plotter.add_contours(seed_mask, colors="yellow", linewidths=2.0)

# %% [markdown]
# ## Red/cyan composites for registration QC
#
# [`plot_composite`][confusius.plotting.plot_composite] renders two volumes as a single
# RGB image — the first in red, the second in cyan — so overlap desaturates to grey and
# misalignment shows up as separated color fringes. This is the same encoding
# [`register_volume`][confusius.registration.register_volume]'s own progress preview
# uses, and is the standard way to sanity-check a registration result: here, the atlas
# reference (red) against the recording it was resampled onto (cyan).

# %%
_ = cf.plotting.plot_composite(bg_volume, moving, resample=False, bg_color=bg_color)

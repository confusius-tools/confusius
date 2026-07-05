# %% [markdown]
# # Atlas-based seed connectivity maps
#
# This example computes voxel-wise seed-based functional connectivity maps: register a
# single-slice fUSI recording to an Allen-space template, bring an
# [Allen Mouse Brain Atlas][confusius.atlas.Atlas] into the recording's native space,
# pick four atlas regions of interest as seeds, and correlate each seed's signal
# against every voxel with [`SeedBasedMaps`][confusius.connectivity.SeedBasedMaps]. Each
# resulting map is displayed with
# [`plot_stat_map`][confusius.plotting.plot_stat_map], using the resampled Allen
# reference volume as background.
#
# We reuse the same recording, template, and registration workflow as the
# [Atlas-based region correlation matrix](atlas_correlation_matrix.md) example — see it
# for a detailed walkthrough of the registration steps condensed here.

# %% [markdown]
# ## Fetch the recording and register to the Allen atlas

# %%
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import confusius as cf

# Adapt background color to the current Matplotlib style.
bg_color = mpl.colors.to_hex(mpl.rcParams["figure.facecolor"])

xr.set_options(display_expand_data=False)

bids_root = cf.datasets.fetch_nunez_elizalde_2022(
    subjects="CR022",
    sessions="20201007",
    tasks="spontaneous",
    acqs="slice02",
)

data_path = (
    Path(bids_root)
    / "sub-CR022"
    / "ses-20201007"
    / "fusi"
    / "sub-CR022_ses-20201007_task-spontaneous_acq-slice02_pwd.nii.gz"
)
# The recording's timepoints are not perfectly uniformly spaced, so we resample to a
# uniform grid before any time-domain processing.
data = cf.timing.resample_to_uniform_time(cf.load(data_path))
moving = data.mean(dim="time").fusi.scale.db().compute()

template = cf.datasets.fetch_template_pepe_mariani_2026().compute()

# %% [markdown]
# As in the correlation-matrix example, we initialize registration with an affine
# measured once in Napari's registration widget, inverted so it maps template physical
# coordinates to recording physical coordinates, and crop the template to a thin band
# around the recording's expected location.

# %%
napari_affine = np.array(
    [
        [1.0, 0.0, 0.0, 5.594638656430411],
        [0.0, 1.0, 0.0, -2.50293925701927],
        [0.0, 0.0, 1.0, 5.6650243788545875],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
initialization = np.linalg.inv(napari_affine)

target_z = napari_affine[0, 3] + float(moving.z.values[0])
fixed = template.sel(z=slice(target_z - 1.0, target_z + 1.0))

registered, affine, diagnostics = cf.registration.register_volume(
    moving=moving,
    fixed=fixed,
    transform_type="affine",
    metric="correlation",
    convergence_window_size=100,
    number_of_iterations=500,
    learning_rate=1,
    initialization=initialization,
)

print(f"Initial metric: {diagnostics.metric_values[0]:.4f}")
print(f"Final metric: {diagnostics.final_metric_value:.4f}")

# %% [markdown]
# ## Resample the Allen atlas onto the recording's native grid
#
# Composing the template's `physical_to_sform` affine with the inverse of the estimated
# registration affine gives a single transform from the recording's native coordinates
# directly to Allen atlas coordinates.
# [`Atlas.resample_like`][confusius.atlas.Atlas.resample_like] resamples the atlas'
# reference volume, annotations, and hemisphere map onto that grid in one call, so
# `atlas_native.reference` can be used directly as the anatomical background for our
# stat maps below.

# %%
physical_to_sform = template.attrs["affines"]["physical_to_sform"]
subject_to_atlas = physical_to_sform @ np.linalg.inv(affine)

atlas = cf.atlas.Atlas.from_brainglobe("allen_mouse_100um")
atlas_native = atlas.resample_like(moving, subject_to_atlas)

# %% [markdown]
# ## Choose seed regions
#
# We pick one seed from each of four functional systems, to contrast their
# connectivity patterns: the primary somatosensory barrel field cortex (`"SSp-bfd"`), a
# classic resting-state seed with a strong, well-documented bilateral
# (interhemispheric) correlation signature; the retrosplenial cortex (`"RSP"`), a
# default-mode-like hub expected to correlate broadly across much of cortex; the
# hippocampus (`"HIP"`), which should instead correlate with a more localized,
# hippocampal-formation-restricted network; and the ventral posteromedial thalamic
# nucleus (`"VPM"`), a somatosensory relay nucleus expected to correlate with
# `"SSp-bfd"` through the thalamocortical pathway. All seeds are taken from the left
# hemisphere only, so that any right-hemisphere correlation in the resulting maps
# reflects genuine interhemispheric connectivity rather than the seed leaking into its
# own mask.
#
# [`Atlas.get_masks`][confusius.atlas.Atlas.get_masks] returns a stacked
# `(mask, z, y, x)` integer DataArray — one layer per requested region — which
# [`SeedBasedMaps`][confusius.connectivity.SeedBasedMaps] accepts directly as
# `seed_masks`.

# %%
seed_regions = ["SSp-bfd", "RSP", "HIP", "VPM"]
seed_masks = atlas_native.get_masks(seed_regions, sides="left")

# %% [markdown]
# ## Clean confounds
#
# As in the correlation-matrix example, we regress out an
# [aCompCor][confusius.signal.compute_compcor_confounds] component extracted from
# white-matter voxels (the Allen ontology's `"fiber tracts"` division) together with a
# `low_cutoff` high-pass filter for slow drift. Both are passed to `SeedBasedMaps` via
# `clean_kwargs`, which cleans the full voxel-wise recording *before* extracting the
# seed signals, so seeds and voxels are preprocessed consistently.

# %%
white_matter = atlas_native.get_masks("fiber tracts").isel(mask=0)
acompcor = cf.signal.compute_compcor_confounds(
    data, noise_mask=white_matter, n_components=1
)

# %% [markdown]
# ## Compute the seed-based correlation maps
#
# [`SeedBasedMaps`][confusius.connectivity.SeedBasedMaps] extracts each seed's average
# signal, correlates it against every voxel in the recording, and returns one Pearson r
# map per seed, stacked along a `region` dimension.

# %%
confounds = data.mean(("z", "y", "x"))
mapper = cf.connectivity.SeedBasedMaps(
    seed_masks=seed_masks,
    clean_kwargs={"low_cutoff": 0.01, "confounds": acompcor},
)
mapper.fit(data)
mapper.maps_

# %% [markdown]
# ## Plot the seed maps
#
# Each seed's map is plotted with [`plot_stat_map`][confusius.plotting.plot_stat_map]
# over the resampled Allen reference volume. Voxels with `|r| < 0.2` are masked out to
# suppress correlation noise, and `vmax=1.0` fixes the colormap to the full range of a
# Pearson correlation so the four seeds are directly comparable. We outline each seed's
# own ROI with
# [`VolumePlotter.add_contours`][confusius.plotting.VolumePlotter.add_contours],
# leaving `colors` unset so each region is drawn in its canonical Allen color (read
# from the atlas mask's `attrs["cmap"]`/`attrs["norm"]`, the same convention used by
# [`Atlas.get_masks`][confusius.atlas.Atlas.get_masks] elsewhere).

# %% tags=["thumbnail"]
fig, axes = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
fig.patch.set_facecolor(bg_color)

flat_axes = axes.ravel()
for i, (ax, region) in enumerate(zip(flat_axes, seed_regions)):
    seed_map = mapper.maps_.sel(region=region)
    plotter = cf.plotting.plot_stat_map(
        seed_map,
        bg_volume=atlas_native.reference,
        slice_mode="z",
        threshold=0.2,
        vmax=1.0,
        cbar_label="Pearson r",
        axes=ax,
        show_titles=False,
        show_axes=False,
        # A single shared colorbar for all four (identically-scaled) panels: adding
        # one per panel would stack up duplicates, since each call would otherwise
        # attach its own colorbar spanning every content axis in the figure.
        show_colorbar=(i == len(flat_axes) - 1),
        bg_color=bg_color,
    )
    plotter.add_contours(seed_masks.sel(mask=region), linewidths=1.5)
    ax.set_title(f"Seed: {region}")

_ = fig.suptitle("Seed-based connectivity maps", fontsize=16)

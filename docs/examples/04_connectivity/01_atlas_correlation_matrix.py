# %% [markdown]
# # Atlas-based region correlation matrix
#
# This example shows an end-to-end regional connectivity analysis: register a
# single-slice fUSI recording to an Allen-space template, bring an
# [Allen Mouse Brain Atlas][confusius.atlas.Atlas] into the recording's native space,
# extract region-averaged signals, and visualise their pairwise correlation with
# [`plot_matrix`][confusius.plotting.plot_matrix].
#
# We use the `fusi` acquisition from subject `CR022`, session `20201007`, in the
# [Nunez-Elizalde 2022 dataset](https://doi.org/10.1016/j.neuron.2022.02.012), and the
# [`Pepe, Mariani 2026`
# template][confusius.datasets.fetch_template_pepe_mariani_2026], which carries the
# affine transform required to bring it into Allen Common Coordinate Framework (CCF)
# space.

# %% [markdown]
# ## Fetch the recording and the template
#
# The recording is a single coronal slice with 755 timepoints. Registration works on a
# static anatomical image, so we use the temporal mean, converted to decibels for a more
# stable dynamic range.

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
# The recording's timepoints are not perfectly uniformly spaced (a common trait of raw
# scanner timestamps), so we resample to a uniform grid before any time-domain
# processing (filtering below requires it).
data = cf.timing.resample_to_uniform_time(cf.load(data_path))
moving = data.mean(dim="time").fusi.scale.db().compute()

template = cf.datasets.fetch_template_pepe_mariani_2026().compute()

data
# %%
template

# %% [markdown]
# ## Register the recording to the template
#
# The recording's `z` coordinate is arbitrary (it just indexes the single acquired
# slice), so it carries no information about where that slice actually sits along the
# template's anteroposterior axis. We use an affine measured once in Napari's
# registration widget as the initialization: it places the slice at its approximate
# location and already yields a good overlay, so registration only needs a small rigid
# refinement.
#
# `register_volume` expects a transform mapping `fixed` (template) physical coordinates
# to `moving` (recording) physical coordinates, so we invert the Napari affine—which
# instead describes how to place the recording *into* the template's coordinate
# system—before using it as `initialization`.

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

# Crop the template to a thin band around the recording's expected location: it
# stabilizes registration and keeps the resampled output close to the recording's
# thickness.
target_z = napari_affine[0, 3] + float(moving.z.values[0])
fixed = template.sel(z=slice(target_z - 1.0, target_z + 1.0))

initialized = cf.registration.resample_like(
    moving, fixed, initialization, default_value=float(moving.min())
)
cf.plotting.plot_composite(
    fixed,
    initialized,
    slice_coords=[target_z],
    normalize_strategy="per_slice",
    bg_color=bg_color,
)

# %% [markdown]
# We use an affine transform: on top of the rotation and translation a rigid
# transform would allow, it also captures small scale and shear differences between
# the recording and the template.

# %%
registered, affine, diagnostics = cf.registration.register_volume(
    moving=moving,
    fixed=fixed,
    transform_type="affine",
    metric="correlation",
    convergence_window_size=100,
    number_of_iterations=500,
    learning_rate=1,
    initialization=initialization,
    show_progress=True,
)

print(f"Initial metric: {diagnostics.metric_values[0]:.4f}")
print(f"Final metric: {diagnostics.final_metric_value:.4f}")
print(f"Iterations: {diagnostics.n_iterations}")
print(f"Stop condition: {diagnostics.stop_condition}")
affine

# %% [markdown]
# The initialization was already close, so the refinement is small. Comparing the
# overlay before and after registration, alignment is slightly better after the
# affine refinement, most noticeably around the anterior choroidal arteries in the
# bottom part of the field of view.

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.patch.set_facecolor(bg_color)
for ax, moving_view, title in [
    (axes[0], initialized, "Manual initialization"),
    (axes[1], registered, "Affine registration refinement"),
]:
    cf.plotting.plot_composite(
        fixed,
        moving_view,
        axes=ax,
        slice_coords=[target_z],
        normalize_strategy="per_slice",
        bg_color=bg_color,
    )
    ax.set_title(title)

_ = fig.suptitle("Template (red) / recording (cyan)")

# %% [markdown]
# ## Resample the Allen atlas onto the recording's native grid
#
# The template is not itself expressed in Allen space, but it carries the affine
# transform to get there in `template.attrs["affines"]["physical_to_sform"]`. Composing
# it with the inverse of the estimated registration affine gives a single transform
# from the recording's native coordinates directly to Allen atlas coordinates.

# %%
physical_to_sform = template.attrs["affines"]["physical_to_sform"]
subject_to_atlas = physical_to_sform @ np.linalg.inv(affine)

atlas = cf.atlas.Atlas.from_brainglobe("allen_mouse_100um")
atlas_native = atlas.resample_like(moving, subject_to_atlas)

plotter = cf.plotting.plot_volume(
    moving, slice_mode="z", cmap="gray", show_colorbar=False, bg_color=bg_color
)
plotter.add_contours(atlas_native.annotation)

# %% [markdown]
# ## Extract region signals and compute their correlation matrix
#
# [`Atlas.get_masks`][confusius.atlas.Atlas.get_masks] accepts parent acronyms from the
# Allen ontology (e.g. `"SSp-bfd"`) and automatically aggregates every descendant
# region, so we can request a handful of coarse regions per area of interest instead of
# individual cortical layers or thalamic nuclei. We pick three regions each from
# cortex, hippocampus, thalamus, and hypothalamus.
#
# We extract left and right hemispheres separately via `get_masks`'s `sides` argument:
# combining both sides into one mask would average left/right signals together and
# hide interhemispheric differences. Each region's `rid` is otherwise identical across
# hemispheres, so the two `extract_with_labels` calls must run separately (one per
# side)—a single stacked mask with both sides would have duplicate region ids across
# layers. Within each area, acronyms are listed medial-to-lateral; the left hemisphere
# is ordered lateral-to-medial and the right medial-to-lateral, so each area block
# reads as one continuous sweep across the slice, meeting at the midline in the middle.

# %%
groups = {
    "cortex": ["RSPv", "MOp", "SSp-bfd"],
    "hippocampus": ["DG", "CA3", "CA2"],
    "thalamus": ["PO", "VPM", "RT"],
    "hypothalamus": ["PH", "LHA", "ZI"],
}
region_acronyms = [acronym for acronyms in groups.values() for acronym in acronyms]

region_order = []
group_labels = []
for area, acronyms in groups.items():
    region_order += [f"{acronym}_L" for acronym in reversed(acronyms)]
    region_order += [f"{acronym}_R" for acronym in acronyms]
    group_labels += [area] * (2 * len(acronyms))

signals_by_side = []
for side, suffix in [("left", "L"), ("right", "R")]:
    side_masks = atlas_native.get_masks(region_acronyms, sides=side)
    side_signals = cf.extract.extract_with_labels(data, side_masks, reduction="mean")
    side_signals = side_signals.assign_coords(
        region=[f"{r}_{suffix}" for r in side_signals.coords["region"].values]
    )
    signals_by_side.append(side_signals)
# extract_with_labels does not guarantee any particular region order, so reindex
# explicitly into the left-right sweep computed above.
signals = xr.concat(signals_by_side, dim="region").sel(region=region_order)

signals

# %% [markdown]
# ## Clean the region signals
#
# Before correlating regions we remove nuisance variance that would otherwise inflate
# their apparent connectivity: a `low_cutoff` high-pass filter for slow drift, and
# [aCompCor][confusius.signal.compute_compcor_confounds] components regressed out.
# aCompCor components are extracted from white matter voxels—the Allen ontology's
# `"fiber tracts"` division, which aggregates every white-matter tract—so they must be
# computed from the voxelwise recording rather than the already-averaged regions.

# %%
white_matter = atlas_native.get_masks("fiber tracts").isel(mask=0)
acompcor = cf.signal.compute_compcor_confounds(
    data, noise_mask=white_matter, n_components=1
)
signals = cf.signal.clean(signals, low_cutoff=0.01, confounds=acompcor)

# %% [markdown]
# [`ConnectivityMatrix`][confusius.connectivity.ConnectivityMatrix] computes the Pearson
# correlation matrix between all region pairs from the `(time, region)` signals.

# %%
connectivity = cf.connectivity.ConnectivityMatrix(kind="correlation").fit_transform(
    [signals]
)[0]

# %% [markdown]
# ## Plot the correlation matrix
#
# [`plot_matrix`][confusius.plotting.plot_matrix]'s `groups` parameter annotates
# contiguous blocks of regions with colored strips—handy here to keep track of which
# brain area each region belongs to without cluttering the plot with per-region colors.

# %% tags=["thumbnail"]
fig, ax = cf.plotting.plot_matrix(
    connectivity,
    labels=region_order,
    groups=group_labels,
    vmax=0.8,
    cbar_label="correlation",
    title="Region correlation matrix",
    bg_color=bg_color,
)

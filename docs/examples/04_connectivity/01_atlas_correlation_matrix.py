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
from confusius.atlas import Atlas
from confusius.connectivity import ConnectivityMatrix
from confusius.datasets import (
    fetch_nunez_elizalde_2022,
    fetch_template_pepe_mariani_2026,
)
from confusius.extract import extract_with_labels
from confusius.plotting import plot_matrix
from confusius.registration import register_volume, resample_like

# Adapt background color to the current Matplotlib style.
bg_color = mpl.colors.to_hex(mpl.rcParams["figure.facecolor"])

xr.set_options(display_expand_data=False)

bids_root = fetch_nunez_elizalde_2022(
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
data = cf.load(data_path)
moving = data.mean(dim="time").fusi.scale.db().compute()

template = fetch_template_pepe_mariani_2026().compute()

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

initialized = resample_like(moving, fixed, initialization, default_value=float(moving.min()))
cf.plotting.plot_composite(
    fixed,
    initialized,
    slice_coords=[target_z],
    normalize_strategy="per_slice",
    bg_color=bg_color,
)

# %% [markdown]
# A rigid transform is appropriate here: the recording and the template share the same
# anatomy, so only a small residual translation/rotation should remain after
# initialization.

# %%
registered, affine, diagnostics = register_volume(
    moving=moving,
    fixed=fixed,
    transform_type="rigid",
    metric="correlation",
    initialization=initialization,
)

print(f"Initial metric: {diagnostics.metric_values[0]:.4f}")
print(f"Final metric: {diagnostics.final_metric_value:.4f}")
print(f"Iterations: {diagnostics.n_iterations}")
print(f"Stop condition: {diagnostics.stop_condition}")
affine

# %% [markdown]
# The initialization was already close, so the refinement is small—compare the
# overlay before and after registration.

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.patch.set_facecolor(bg_color)
for ax, moving_view, title in [
    (axes[0], initialized, "Before"),
    (axes[1], registered, "After"),
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

atlas = Atlas.from_brainglobe("allen_mouse_100um")
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
# cortex, hippocampus, thalamus, and hypothalamus, keeping the list ordered by area so
# that `groups` below can annotate contiguous blocks.

# %%
groups = {
    "cortex": ["MOp", "SSp-bfd", "RSPv"],
    "hippocampus": ["CA2", "CA3", "DG"],
    "thalamus": ["VPM", "PO", "RT"],
    "hypothalamus": ["ZI", "LHA", "PH"],
}
region_order = [acronym for acronyms in groups.values() for acronym in acronyms]
group_labels = [area for area, acronyms in groups.items() for _ in acronyms]

masks = atlas_native.get_masks(region_order)
signals = extract_with_labels(data, masks, reduction="mean")
# extract_with_labels does not guarantee any particular region order, so reindex
# explicitly to keep regions from the same area contiguous for the `groups` plot below.
signals = signals.sel(region=region_order)

signals

# %% [markdown]
# [`ConnectivityMatrix`][confusius.connectivity.ConnectivityMatrix] computes the Pearson
# correlation matrix between all region pairs from the `(time, region)` signals.

# %%
connectivity = ConnectivityMatrix(kind="correlation").fit_transform([signals])[0]
connectivity.shape

# %% [markdown]
# ## Plot the correlation matrix
#
# [`plot_matrix`][confusius.plotting.plot_matrix]'s `groups` parameter annotates
# contiguous blocks of regions with colored strips—handy here to keep track of which
# brain area each region belongs to without cluttering the plot with per-region colors.

# %% tags=["thumbnail"]
fig, ax = plot_matrix(
    connectivity,
    labels=region_order,
    groups=group_labels,
    tri="diag",
    grid="gray",
    cbar_label="correlation",
    title="Region correlation matrix",
    bg_color=bg_color,
)

# %% [markdown]
# # Save and reload a resampled atlas
#
# Aligning a brain atlas to a fUSI recording means registering the recording to a
# template and resampling the atlas—its reference volume, region annotations, and
# hemisphere map—onto the recording's native grid. That alignment is reused across many
# analyses of the same recording, and both the registration and the resampling can be
# costly, so it is worth computing it once and caching the result to disk.
#
# This example resamples the [Allen Mouse Brain Atlas][confusius.atlas] onto a
# recording's grid, saves it to Zarr format with
# [`save_atlas`][confusius.io.save_atlas], and reads it back with
# [`load_atlas`][confusius.io.load_atlas]. The Zarr dataset bundles the structure
# hierarchy and the region meshes alongside the arrays, so the reloaded atlas is
# immediately usable for masks, meshes, and plotting.

# %% [markdown]
# ## Register the recording to the template
#
# We reuse the recording, template, and affine registration from the [Atlas-based region
# correlation matrix](../connectivity/atlas_correlation_matrix.md) example—see it for a
# full walkthrough. Briefly: we take the temporal mean of a 2D+t recording from the
# [Nunez-Elizalde 2022 dataset][confusius.datasets.fetch_nunez_elizalde_2022] and
# register it to the [Pepe, Mariani 2026
# template][confusius.datasets.fetch_template_pepe_mariani_2026]. The latter carries an
# affine transform to the Allen Common Coordinate Framework (CCF) space. Code for data
# fetching and registration is available in the folded cell below.

# %% tags=["collapse: Data fetching and registration"]
import warnings
from pathlib import Path

import matplotlib as mpl
import numpy as np
import xarray as xr

import confusius as cf

# Adapt background color to the current Matplotlib style.
bg_color = mpl.colors.to_hex(mpl.rcParams["figure.facecolor"])

xr.set_options(display_expand_data=False)

# The ConfUSIus datasets directory is a convenient place to cache the resampled atlas.
CONFUSIUS_DATA_DIR = cf.datasets.get_datasets_dir()

bids_root = cf.datasets.fetch_nunez_elizalde_2022(
    subjects="CR022",
    sessions="20201007",
    tasks="spontaneous",
    acqs="slice02",
    print_citation=False,
)

data_path = (
    Path(bids_root)
    / "sub-CR022"
    / "ses-20201007"
    / "fusi"
    / "sub-CR022_ses-20201007_task-spontaneous_acq-slice02_pwd.nii.gz"
)
moving = cf.load(data_path).mean(dim="time").fusi.scale.db().compute()

template = cf.datasets.fetch_template_pepe_mariani_2026(print_citation=False).compute()

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

# %% [markdown]
# ## Resample the atlas and save it
#
# Composing the template's `physical_to_sform` affine with the inverse of the estimated
# registration affine gives a single transform from the recording's native coordinates
# to Allen atlas coordinates.
# [`resample_like`][confusius.atlas.AtlasAccessor.resample_like] then brings the atlas'
# reference, annotations, and hemisphere maps onto the recording's grid in one call.
# This is the expensive step we want to avoid repeating, so we write the result to a
# Zarr dataset right away and drop it from memory.

# %%
physical_to_sform = template.attrs["affines"]["physical_to_sform"]
subject_to_atlas = physical_to_sform @ np.linalg.inv(affine)

atlas = cf.datasets.fetch_brainglobe_atlas("allen_mouse_100um")

# `<dataset>.atlas` exposes the atlas accessor on the xarray.Dataset.
resampled_atlas = atlas.atlas.resample_like(moving, subject_to_atlas)

store_path = f"{CONFUSIUS_DATA_DIR}/resampled_allen_mouse_100um.zarr"
cf.io.save_atlas(resampled_atlas, store_path, mode="w")
del resampled_atlas

# %% [markdown]
# ## Reload the resampled atlas
#
# Later, in a new session, or a downstream analysis, the aligned atlas is one
# [`load_atlas`][confusius.io.load_atlas] call away, with no registration or resampling
# to redo. Overlaying its region annotations on the recording confirms the reloaded
# atlas is correctly aligned.

# %% tags=["thumbnail"]
resampled_atlas = cf.io.load_atlas(store_path)

plotter = cf.plotting.plot_volume(moving, bg_color=bg_color)
_ = plotter.add_contours(resampled_atlas.annotation)

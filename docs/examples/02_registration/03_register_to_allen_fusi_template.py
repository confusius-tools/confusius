# %% [markdown]
# # Register a recording to an Allen fUSI template
#
# This example shows the full workflow for aligning a single-slice fUSI recording to one
# of ConfUSIus's Allen-space fUSI templates: fetch the recording and the template,
# initialize and refine the registration, inspect diagnostic overlays, resample the
# [Allen Mouse Brain Atlas][confusius.atlas] onto the recording's native grid, and save
# the aligned atlas for reuse.
#
# We use an awake freely-running acquisition from subject `CR022`, session `20201007`,
# in the [Nunez-Elizalde 2022 dataset][confusius.datasets.fetch_nunez_elizalde_2022],
# and the [Pepe, Mariani 2026 fUSI template][confusius.datasets.fetch_template_pepe_mariani_2026],
# which carries the affine transform required to bring it into Allen Common Coordinate
# Framework (CCF) space.

# %% [markdown]
# ## Fetch the recording and the template
#
# The recording is a single coronal slice imaged for approximately 4 minutes at 3.33 Hz.
# Registration works on a static anatomical image, so we use the temporal mean,
# converted to decibels for a more stable dynamic range.

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

# The ConfUSIus datasets directory is a convenient place to cache the resampled atlas.
CONFUSIUS_DATA_DIR = cf.datasets.get_datasets_dir()

template = cf.datasets.fetch_template_pepe_mariani_2026().compute()

bids_root = cf.datasets.fetch_nunez_elizalde_2022(
    subjects="CR022", sessions="20201007", tasks="spontaneous", acqs="slice02"
)

# %%
data_path = (
    Path(bids_root)
    / "sub-CR022"
    / "ses-20201007"
    / "fusi"
    / "sub-CR022_ses-20201007_task-spontaneous_acq-slice02_pwd.nii.gz"
)
moving = cf.load(data_path).mean(dim="time").fusi.scale.db().compute()

moving

# %% [markdown]
# ## Register the recording to the template
#
# The template is a 3D volume but the recording is a single slice. We can still register
# the recording to the template, but we need to initialize the registration with a rough
# guess of where the recording sits in the template, otherwise the registration
# algorithm may not converge to the right slice. To initialize the registration, we use
# an affine transform obtained using [napari's manual transform
# tool](https://napari.org/stable/howtos/layers/image.html#buttons) by placing the
# recording at an approximate location on the template. The transform is not perfect
# (notice the slight misalignment toward the bottom of the field of view), but it is
# close enough to allow the registration algorithm to converge to a good solution.
#
# [`register_volume`][confusius.registration.register_volume] expects a transform
# mapping `fixed` (template) physical coordinates to `moving` (recording) physical
# coordinates, so we invert the napari affine—which instead describes how to place the
# recording *into* the template's coordinate system—before using it as `initialization`.

# %%
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

# Crop the template to a thin band around the recording's expected location to improve
# registration speed and visualization.
target_z = napari_affine[0, 3] + float(moving.z.values[0])
fixed = template.sel(z=slice(target_z - 1.0, target_z + 1.0)).fusi.scale.db()

initialized = cf.registration.resample_like(moving, fixed, initialization)
_ = cf.plotting.plot_composite(
    fixed,
    initialized,
    slice_coords=[target_z],
    normalize_strategy="per_slice",
    bg_color=bg_color,
)

# %% [markdown]
# We use an affine transform: on top of the rotation and translation a rigid transform
# would allow, it also captures small scale and shear differences between the recording
# and the template.

# %%
registered, affine, _ = cf.registration.register_volume(
    moving=moving,
    fixed=fixed,
    transform_type="affine",
    metric="correlation",
    convergence_window_size=50,
    number_of_iterations=500,
    learning_rate=1,
    initialization=initialization,
    show_progress=True,
)

# %% [markdown]
# The initialization was already close, so the refinement is small. Comparing the
# overlay before and after registration, alignment is slightly better after the affine
# refinement, most noticeably around the anterior choroidal arteries in the bottom part
# of the field of view.

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
# it with the inverse of the estimated registration affine gives a single transform from
# the recording's native coordinates directly to Allen atlas coordinates.
#
# [`resample_like`][confusius.atlas.AtlasAccessor.resample_like] then brings the atlas's
# reference volume, annotations, and hemisphere map onto the recording's grid in one
# call.

# %%
physical_to_sform = template.attrs["affines"]["physical_to_sform"]
subject_to_atlas = physical_to_sform @ np.linalg.inv(affine)

atlas = cf.datasets.fetch_brainglobe_atlas("allen_mouse_100um", check_latest=False)
resampled_atlas = atlas.atlas.resample_like(moving, subject_to_atlas)

# %% [markdown]
# ## Save and reload the resampled atlas
#
# Registration and atlas resampling are expensive, so for downstream analyses of the
# same recording it is worth caching the aligned atlas once and reloading it later.
# [`save_atlas`][confusius.io.save_atlas] writes the arrays together with the structure
# hierarchy and meshes, so the reloaded atlas is immediately usable for masks, meshes,
# and plotting.

# %% tags=["thumbnail"]
store_path = f"{CONFUSIUS_DATA_DIR}/resampled_allen_mouse_100um.zarr"
cf.io.save_atlas(resampled_atlas, store_path, mode="w")
del resampled_atlas

resampled_atlas = cf.io.load_atlas(store_path)

plotter = cf.plotting.plot_volume(
    moving, slice_mode="z", cmap="gray", show_colorbar=False, bg_color=bg_color
)
_ = plotter.add_contours(resampled_atlas.annotation)

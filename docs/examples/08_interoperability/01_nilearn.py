# %% [markdown]
# # Use nilearn on a VoxelData array
#
# This example shows how to pass data between ConfUSIus and
# [nilearn](https://nilearn.github.io). [`to_nifti`][confusius.io.to_nifti] converts a
# VoxelData array to an in-memory NIfTI image. [`from_nifti`][confusius.io.from_nifti]
# converts a NIfTI image back to a VoxelData array.
#
# We use nilearn to find the clusters of a statistical map from [Cybis Pereira *et al.*
# 2026](https://doi.org/10.1016/j.celrep.2025.116791)[^collection]. Then we plot these
# clusters with ConfUSIus.
#
# !!! warning "nilearn is not a dependency of ConfUSIus" ConfUSIus does not install
# nilearn. If nilearn is not in your environment, install it before you run this
# example:
#
#     ```bash pip install nilearn ```
#
# [^collection]: See [The ConfUSIus Dataset Collection](../io/dataset_collection.md) for
# available studies, formats, licenses, and reproducible subset downloads.
#
# ## Load a statistical map
#
# [`fetch_cybis_pereira_2026`][confusius.datasets.fetch_cybis_pereira_2026] downloads
# the `glm-speed` results of one rat. The file that we load contains t-statistic maps
# for the contrast between fUSI signal and animal speed. Its `time` dimension holds the
# lag between speed and fUSI. We keep the map at a lag of 2 s as we know this is a lag
# that shows a strong correlation.

# %%
from pathlib import Path

import matplotlib as mpl
import xarray as xr
from nilearn.reporting import get_clusters_table

import confusius as cf

# Adapt background color to the current Matplotlib style.
bg_color = mpl.colors.to_hex(mpl.rcParams["figure.facecolor"])

# Keep notebook output compact for large DataArray displays.
_ = xr.set_options(display_expand_attrs=False, display_expand_data=False)

subject = "rat75"
acq = "slice32"

bids_root = cf.datasets.fetch_cybis_pereira_2026(
    datasets="glm-speed",
    subjects=subject,
    acqs=acq,
)

stat_path = (
    Path(bids_root)
    / "derivatives/glm-speed"
    / f"sub-{subject}"
    / f"sub-{subject}_acq-{acq}_rec-staticsvd60clutterwindowwidth200dopplerwindowwidth40_contrast-speed_stat-t_control-bonferroni_alpha-001_statmap.nii"
)

stat = cf.load(stat_path).sel(time=2, method="nearest")
stat.attrs["long_name"] = "t statistic"

# %% [markdown]
# ## Convert the map to a NIfTI image
#
# nilearn reads the world coordinates of an image from `img.affine`. NiBabel [sets this
# affine](https://nipy.org/nibabel/nifti_images.html#choosing-the-image-affine) to the
# `sform` when `sform_code` is not 0. If `sform_code` is 0, NiBabel uses the `qform`.
#
# In this file, the `qform` points to the probe space. In that space, the imaging
# plane is aligned with the `x` and `y` axes, which makes the cluster coordinates easy
# to read. Thus we move the world coordinates to that space with
# [`affine.apply`][confusius.xarray.FUSIAffineAccessor.apply]. Then we set `sform_code`
# to 0, and [`to_nifti`][confusius.io.to_nifti] writes only the `qform`.

# %%
stat = stat.fusi.affine.apply("world_to_qform")
stat.attrs["sform_code"] = 0

img = cf.io.to_nifti(stat)

# %% [markdown]
# ## Find the clusters with nilearn
#
# [`get_clusters_table`][nilearn.reporting.get_clusters_table] finds the clusters of
# voxels with a value more than `stat_threshold`. It returns a table with the peak of
# each cluster, in the world coordinates of the image. With `return_label_maps=True`, it
# also returns the cluster labels as NIfTI images.
#
# nilearn gives the cluster size in full cubic millimeters. As a result, a cluster
# smaller than 1 mm³ shows a size of 0.

# %%
table, label_maps = get_clusters_table(
    img,
    stat_threshold=5,
    return_label_maps=True,
)

table

# %% [markdown]
# ## Convert the cluster labels to a VoxelData array
#
# [`from_nifti`][confusius.io.from_nifti] converts the label image to a VoxelData
# array. This array has the same world coordinates as `stat`.

# %%
labels = cf.io.from_nifti(label_maps[0])
labels

# %% [markdown]
# ## Plot the clusters
#
# We plot the contour of each cluster on the statistical map. Then we mark the peak of
# each cluster at its coordinates from the nilearn table.

# %% tags=["thumbnail"]
plotter = cf.plotting.plot_stat_map(stat, bg_color=bg_color)
plotter.add_contours(labels)
ax = plotter.axes[0][0]

colors = []
for line in ax.get_lines():
    color = line.get_color()
    if color not in colors:
        colors.append(color)

for (i, row), color in zip(table.iterrows(), colors):
    ax.plot(
        row["X"],
        row["Y"],
        marker="o",
        color=color,
        markersize=7,
        label=f"Cluster ID {i + 1}",
    )

_ = ax.legend()

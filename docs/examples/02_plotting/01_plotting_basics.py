# %% [markdown]
# # Plotting basics
#
# This example introduces ConfUSIus's volume plotting: a grid of matplotlib axes, one
# per slice coordinate, filled in by [`plot_volume`][confusius.plotting.plot_volume] and
# the [`VolumePlotter`][confusius.plotting.VolumePlotter] it returns. We build up a
# single-volume figure step by step — the slice grid, colormap and value range — before
# moving on to overlays and ROI contours in [Overlays, ROI contours, and
# composites](overlays_and_contours.md).
#
# We use an awake freely-running acquisition from subject `CR022`, session `20201007`,
# in the [Nunez-Elizalde 2022 dataset][confusius.datasets.fetch_nunez_elizalde_2022].

# %% [markdown]
# ## Load a volume
#
# See [ConfUSIus and Xarray 101](../io/confusius_xarray_101.md) for a walkthrough of
# `cf.load`. `VolumePlotter` plots one 2D slice per subplot, so we reduce the
# recording's `time` dimension to a single static volume with a temporal mean,
# converted to decibels for a more stable dynamic range.

# %%
from pathlib import Path

import matplotlib as mpl
import xarray as xr

import confusius as cf

# Adapt background color to the current Matplotlib style.
bg_color = mpl.colors.to_hex(mpl.rcParams["figure.facecolor"])

xr.set_options(display_expand_data=False)

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
volume = cf.load(data_path).mean(dim="time").fusi.scale.db().compute()

# %% [markdown]
# ## Plotting a volume
#
# [`plot_volume`][confusius.plotting.plot_volume] is the entry point for every figure:
# it creates the figure, lays out one axes per slice coordinate, and returns the
# [`VolumePlotter`][confusius.plotting.VolumePlotter] managing them. Our recording has a
# single `z` slice, so this creates one axes. Later calls chain
# [`add_volume`][confusius.plotting.VolumePlotter.add_volume]/[`add_contours`][confusius.plotting.VolumePlotter.add_contours]
# onto the same plotter to overlay more data — see [Overlays, ROI contours, and
# composites](overlays_and_contours.md). You should never need to construct
# `VolumePlotter` directly.
#
# If your panels ever come out rotated 90° from what you expect, pass
# `transpose=True` to swap each panel's row/column display axes — most volumes don't
# need it, but it depends on how the recording's native in-plane axes happen to map to
# rows/columns.

# %%
_ = cf.plotting.plot_volume(volume, bg_color=bg_color)

# %% [markdown]
# ## Multi-slice grids
#
# For a volume with several coordinates along `slice_mode`, `plot_volume` lays out one
# panel per coordinate automatically, computing `nrows`/`ncols` unless you provide them.
# We use a real multi-slice angiography acquisition from the same session — a static 3D
# power Doppler volume with 41 `z` slices spanning 4 mm — and `slice_coords` to pick out
# three representative slices instead of showing all 41.

# %% tags=["thumbnail"]
angio_path = (
    Path(bids_root)
    / "sub-CR022"
    / "ses-20201007"
    / "angio"
    / "sub-CR022_ses-20201007_pwd.nii.gz"
)
angio = cf.load(angio_path).fusi.scale.db().compute()

_ = cf.plotting.plot_volume(
    angio, slice_coords=[0.5, 2.0, 3.5], cmap="gray", show_colorbar=False, bg_color=bg_color
)

# %% [markdown]
# ## Colormap and value range
#
# Without an explicit `vmin`/`vmax`, `plot_volume` scales the colormap to the 2nd/98th
# percentile of the data — robust to a few extreme voxels, but adjustable by passing
# `vmin`/`vmax` directly.

# %%
_ = cf.plotting.plot_volume(volume, bg_color=bg_color)

# %%
_ = cf.plotting.plot_volume(volume, cmap="magma", vmin=-20, vmax=0, bg_color=bg_color)

# %% [markdown]
# `threshold`/`threshold_mode` mask out values below (or above) a cutoff — useful for
# statistical maps, so we leave it for [Overlays, ROI contours, and
# composites](overlays_and_contours.md) where we have one to show.
#
# ## Saving and displaying figures
#
# `VolumePlotter.savefig`/`show`/`close` forward to the underlying matplotlib figure —
# convenient so you don't need to keep a separate reference to `plotter.figure` around.

# %%
import tempfile

plotter = cf.plotting.plot_volume(volume, cmap="gray", bg_color=bg_color)
output_path = Path(tempfile.gettempdir()) / "plotting_basics.png"
plotter.savefig(output_path, dpi=150)
plotter.close()
print(f"Saved to {output_path}")

# %% [markdown]
# `plot_volume`'s siblings — [`plot_stat_map`][confusius.plotting.plot_stat_map],
# [`plot_contours`][confusius.plotting.plot_contours], and
# [`plot_composite`][confusius.plotting.plot_composite] — cover the other common
# one-shot patterns; we use them next in [Overlays, ROI contours, and
# composites](overlays_and_contours.md).

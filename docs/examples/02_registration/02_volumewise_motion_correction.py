# %% [markdown]
# # Volumewise motion correction
#
# This example shows how to correct frame-to-frame brain motion in one fUSI recording
# with [`register_volumewise`][confusius.registration.register_volumewise]. We reuse
# the exact acquisition and rigid-registration settings from the
# volumewise-registration GIF in the GUI guide: a short open-field excerpt from the
# [Cybis Pereira 2026 dataset](https://doi.org/10.1016/j.celrep.2025.116791). We then
# inspect three things that are useful in practice:
#
# - the motion diagnostics returned by
#   [`create_motion_dataframe`][confusius.registration.create_motion_dataframe];
# - a before/after GIF of the registered movie;
# - the time series of one representative voxel before and after registration.
#
# As in the GUI GIF, we register a 120-frame excerpt rather than the full recording.

# %% [markdown]
# ## Fetch and load a short motion-corrupted window
#
# This is the same open-field recording used by the GUI registration demo. The selected
# acquisition is a single 2D slice, so the data shape is `(time, z=1, y, x)`.


# %%
from base64 import b64encode
from io import BytesIO
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from IPython.display import HTML
from PIL import Image, ImageDraw

import confusius as cf

bg_color = mpl.colors.to_hex(mpl.rcParams["figure.facecolor"])
xr.set_options(display_expand_data=False)

subject = "rat75"
session = "20220523"
acq = "slice32"
start_frame = 220
n_frames = 120

bids_root = cf.datasets.fetch_cybis_pereira_2026(
    datasets="rawdata",
    subjects=subject,
    sessions=session,
    acqs=acq,
)

pwd_path = (
    Path(bids_root)
    / f"sub-{subject}"
    / f"ses-{session}"
    / "fusi"
    / f"sub-{subject}_ses-{session}_task-openfield_acq-{acq}_pwd.nii.gz"
)

data = cf.load(pwd_path).isel(time=slice(start_frame, start_frame + n_frames)).compute()
data

# %% [markdown]
# ## Register every frame to the middle frame
#
# Registering to a central reference frame is a simple way to avoid anchoring the whole
# excerpt to one edge of the motion trajectory. Here we match the GUI demo settings:
# rigid transform, correlation metric, and a fixed `learning_rate=1.0`.

# %%
reference_time = data.sizes["time"] // 2
registered = cf.registration.register_volumewise(
    data,
    reference_time=reference_time,
    transform="rigid",
    metric="correlation",
    learning_rate=1.0,
    n_jobs=-1,
    resample_interpolation="bspline",
    show_progress=False,
)

motion_df = registered.attrs["motion_params"]
motion_df.head()

# %% [markdown]
# `motion_df` is the output of
# [`create_motion_dataframe`][confusius.registration.create_motion_dataframe]. For this
# 2D+t example we focus on the in-plane rotation (`rot_z`), the in-plane translations,
# the framewise displacement summaries (`mean_fd`, `max_fd`, `rms_fd`), and the
# optimizer summaries (`final_metric_value`, `n_iterations`) added by
# [`register_volumewise`][confusius.registration.register_volumewise].

# %% [markdown]
# ## Plot the motion diagnostics
#
# The framewise displacement peak marks where the excerpt moves most strongly. The last
# panel is a useful sanity check: frames that systematically hit the maximum iteration
# count or converge to a much worse similarity metric deserve a closer look.

# %%
fig, axes = plt.subplots(4, 1, figsize=(9, 9), sharex=True, constrained_layout=True)
fig.patch.set_facecolor(bg_color)

time = motion_df.index.to_numpy(dtype=float)

axes[0].plot(time, np.rad2deg(motion_df["rot_z"]), color="#4c78a8", lw=1.6)
axes[0].set_ylabel("Rotation (deg)")
axes[0].set_title("In-plane motion estimates")

axes[1].plot(time, motion_df["trans_x"], label="x", lw=1.6)
axes[1].plot(time, motion_df["trans_y"], label="y", lw=1.6)
axes[1].set_ylabel("Translation (mm)")
axes[1].legend(frameon=False, ncol=2)

axes[2].plot(time, motion_df["mean_fd"], label="Mean FD", lw=1.8)
axes[2].plot(time, motion_df["max_fd"], label="Max FD", lw=1.2, alpha=0.8)
axes[2].set_ylabel("Displacement (mm)")
axes[2].legend(frameon=False, ncol=2)

metric_color = "#d93a54"
iteration_color = "#3ad9a4"
ax_metric = axes[3]
ax_metric.plot(time, motion_df["final_metric_value"], color=metric_color, lw=1.8)
ax_metric.set_ylabel("Final metric", color=metric_color)
ax_metric.tick_params(axis="y", colors=metric_color)
ax_metric.spines["left"].set_color(metric_color)
ax_metric.set_xlabel("Time (s)")
ax_metric.set_title("Optimizer summary")

ax_iterations = ax_metric.twinx()
ax_iterations.plot(
    time,
    motion_df["n_iterations"],
    color=iteration_color,
    lw=1.2,
    alpha=0.9,
)
ax_iterations.set_ylabel("Iterations", color=iteration_color)
ax_iterations.tick_params(axis="y", colors=iteration_color)
ax_iterations.spines["right"].set_color(iteration_color)

# %% [markdown]
# ## Compare a representative voxel before and after registration
#
# To make the effect easy to see, we pick the voxel with the largest temporal standard
# deviation in the unregistered excerpt. In practice that usually lands on a large
# vessel, where motion-induced intensity changes are most visible.

# %% tags=["thumbnail"]
std_map = data.squeeze("z", drop=True).std("time")
voxel_y, voxel_x = np.unravel_index(np.nanargmax(std_map.values), std_map.shape)

voxel_before = data.isel(z=0, y=voxel_y, x=voxel_x)
voxel_after = registered.isel(z=0, y=voxel_y, x=voxel_x)

fig, ax = plt.subplots(figsize=(9, 3.5), constrained_layout=True)
fig.patch.set_facecolor(bg_color)
ax.plot(voxel_before["time"], voxel_before, label="Before", lw=1.6, alpha=0.8)
ax.plot(voxel_after["time"], voxel_after, label="After", lw=1.6)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Power Doppler intensity")
ax.set_title(f"Voxel at y={voxel_y}, x={voxel_x}")
ax.legend(frameon=False)

# %%
data_db = data.fusi.scale.db()
registered_db = registered.fusi.scale.db()

# %% [markdown]
# ## Build a before/after GIF
#
# A side-by-side movie is often the fastest qualitative check. We render the raw and
# registered slices with a shared contrast scale so the residual jitter is easy to spot.


# %%
def _to_uint8_frame(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Convert one 2D image to a clipped 8-bit grayscale frame.

    Parameters
    ----------
    values : numpy.ndarray
        Two-dimensional image data.
    vmin : float
        Lower display bound.
    vmax : float
        Upper display bound.

    Returns
    -------
    numpy.ndarray
        Unsigned 8-bit image with shape `(y, x)`.
    """
    if vmax <= vmin:
        vmax = vmin + 1.0
    clipped = np.clip((values - vmin) / (vmax - vmin), 0, 1)
    return (255 * clipped).astype(np.uint8)


# %%
def _movie_html(before: xr.DataArray, after: xr.DataArray) -> HTML:
    """Return an inline HTML `<img>` tag for a side-by-side animated GIF.

    Parameters
    ----------
    before : xarray.DataArray
        Unregistered movie with dims `(time, y, x)`.
    after : xarray.DataArray
        Registered movie with the same dims and coordinates as `before`.

    Returns
    -------
    IPython.display.HTML
        HTML object embedding the GIF as a data URI.
    """
    font = ImageDraw.Draw(Image.new("RGB", (1, 1))).getfont()
    vmin = float(np.nanpercentile(before, 2))
    vmax = float(np.nanpercentile(before, 99.8))
    pad = 8
    frames: list[Image.Image] = []

    for i, t in enumerate(before["time"].values):
        left = Image.fromarray(_to_uint8_frame(before.isel(time=i).values, vmin, vmax))
        right = Image.fromarray(_to_uint8_frame(after.isel(time=i).values, vmin, vmax))
        left = left.convert("RGB").resize((320, 260))
        right = right.convert("RGB").resize((320, 260))

        canvas = Image.new(
            "RGB", (left.width + right.width + 3 * pad, left.height + 40), "black"
        )
        canvas.paste(left, (pad, 28))
        canvas.paste(right, (left.width + 2 * pad, 28))

        draw = ImageDraw.Draw(canvas)
        draw.text((pad, 6), "Before", fill="white", font=font)
        draw.text((left.width + 2 * pad, 6), "After", fill="white", font=font)
        draw.text((canvas.width - 70, 6), f"{float(t):.1f}s", fill="white", font=font)
        frames.append(canvas)

    buffer = BytesIO()
    frames[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
    )
    gif_base64 = b64encode(buffer.getvalue()).decode("ascii")
    return HTML(
        f'<img src="data:image/gif;base64,{gif_base64}" alt="Before/after volumewise registration GIF" />'
    )


movie_before = data_db.squeeze("z", drop=True)
movie_after = registered_db.squeeze("z", drop=True)
_movie_html(movie_before, movie_after)

# %% [markdown]
# Even on this short excerpt, the registered movie is visibly more stable and the mean
# image is slightly sharper. For a full preprocessing workflow, you would usually run
# the same correction on the complete recording before downstream QC, decomposition, or
# connectivity analysis.

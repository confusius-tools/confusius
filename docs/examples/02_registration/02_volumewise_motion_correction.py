# %% [markdown]
# # Motion correction of a single recording
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
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

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

# %%
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
registered = cf.registration.register_volumewise(
    data,
    transform="rigid",
    metric="correlation",
    learning_rate=1.0,
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

# %% tags=["thumbnail"]
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

# %%
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
_ = ax.legend(frameon=False)

# %% [markdown]
# ## Build a before/after GIF
#
# A side-by-side movie is often the fastest qualitative check. We render the raw and
# registered slices with a shared contrast scale so the residual jitter is easy to spot.


# %%
def _gif_html(before: xr.DataArray, after: xr.DataArray) -> HTML:
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
    vmin = float(np.nanpercentile(before, 2))
    vmax = float(np.nanpercentile(before, 99.8))
    if vmax <= vmin:
        vmax = vmin + 1.0

    pad = 12
    panel_size = (480, 390)
    header_height = 56
    try:
        font = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 24)
    except (OSError, ValueError):
        font = ImageFont.load_default()

    frames: list[Image.Image] = []

    for i, t in enumerate(before["time"].values):
        left = np.clip((before.isel(time=i).values - vmin) / (vmax - vmin), 0, 1)
        right = np.clip((after.isel(time=i).values - vmin) / (vmax - vmin), 0, 1)
        left = Image.fromarray((255 * left).astype(np.uint8)).convert("RGB")
        right = Image.fromarray((255 * right).astype(np.uint8)).convert("RGB")
        left = left.resize(panel_size, Image.Resampling.LANCZOS)
        right = right.resize(panel_size, Image.Resampling.LANCZOS)

        canvas = Image.new(
            "RGB",
            (left.width + right.width + 3 * pad, left.height + header_height),
            "black",
        )
        canvas.paste(left, (pad, header_height))
        canvas.paste(right, (left.width + 2 * pad, header_height))

        draw = ImageDraw.Draw(canvas)
        draw.text((pad, 14), "Before", fill="white", font=font)
        draw.text((left.width + 2 * pad, 14), "After", fill="white", font=font)
        timestamp = f"{float(t):.1f}s"
        timestamp_width = draw.textlength(timestamp, font=font)
        draw.text(
            (canvas.width - pad - timestamp_width, 14),
            timestamp,
            fill="white",
            font=font,
        )
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


movie_before = data.fusi.scale.db().squeeze("z", drop=True)
movie_after = registered.fusi.scale.db().squeeze("z", drop=True)
_gif_html(movie_before, movie_after)

# %% [markdown]
# Even on this short excerpt, the registered movie is visibly more stable. For a full
# preprocessing workflow, you would usually run the same correction on the complete
# recording before downstream QC, decomposition, or connectivity analysis.

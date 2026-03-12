"""Generate documentation images for the Visualization user guide.

Usage
-----
1. Fill in the constant variables below with paths to example datasets on your machine:

    - `ZARR_PATH`: 3D or 4D fUSI recording in Zarr format, with a variable containing
      the power Doppler data.
    - `ATLAS_ZARR_PATH` and `ATLAS_VARIABLE`: Zarr store and variable name for a labelled
      integer atlas mask aligned to the recording. Used for the napari Labels layer and
      volume-with-contours overlay.
    - `VOLUME_3D_PATH` and `VOLUME_3D_VARIABLE`: Zarr store and variable name for a true 3D
      recording (z > 1) to use in the multi-slice volume example.
    - `MASK_ZARR_PATH` and `MASK_VARIABLE`: Zarr store and variable name for a binary brain
      mask (True/1 inside brain, False/0 outside) to use for voxel selection in the
      carpet plot.

2. Run from the project root::

       uv run docs/images/visualization/generate_images.py

All images are saved to docs/images/visualization/.

Notes
-----
- Napari screenshots are taken programmatically. Review them after the script
  finishes and retake manually (File > Save Screenshot in Napari) if the canvas
  renders poorly (e.g., all-black).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import napari
import numpy as np
import xarray as xr
from brainglobe_atlasapi import BrainGlobeAtlas
from napari.qt import get_qapp
from napari.utils.colormaps import DirectLabelColormap
from qtpy.QtCore import Qt

import confusius as cf  # noqa: F401  # Register xarray accessors.

HERE = Path(__file__).parent

# == Fill in before running ===================================================

ZARR_PATH = "../../../data/sub-ALD030_ses-ChATChR2FibreMidThalamus_task-awake_acq-motor2dot2_proc-staticsvd50_pwd.zarr/"
VARIABLE = "power_doppler"  # Variable name inside the Zarr store.

# Labelled integer atlas mask (one positive integer per brain region).
# Used for: napari Labels layer and volume-with-contours overlay.
ATLAS_ZARR_PATH = "../../../data/regions.zarr"
ATLAS_VARIABLE = "allen"

# Path to a true 3D recording (z > 1) for the multi-slice volume example.
VOLUME_3D_PATH = "../../../data/angio.zarr"
VOLUME_3D_VARIABLE = "angio"

# Binary brain mask (True/1 inside brain, 0 outside).
# Used for: carpet plot voxel selection.
MASK_ZARR_PATH = "../../../data/brain_mask.zarr/"
MASK_VARIABLE = "brain_mask"

# =============================================================================


def _resolve(path: str) -> str:
    """Resolve a path relative to this script's directory if not absolute."""
    p = Path(path)
    return str((HERE / p).resolve() if not p.is_absolute() else p)


ZARR_PATH = _resolve(ZARR_PATH)
ATLAS_ZARR_PATH = _resolve(ATLAS_ZARR_PATH)
MASK_ZARR_PATH = _resolve(MASK_ZARR_PATH)
VOLUME_3D_PATH = _resolve(VOLUME_3D_PATH)

# --------------------------------------------------------------------------- #
# Load power Doppler data                                                       #
# --------------------------------------------------------------------------- #

print("Loading power Doppler data …")
pwd = xr.open_zarr(ZARR_PATH)[VARIABLE]
print(f"  {pwd.dims}, shape {dict(pwd.sizes)}")

print("Computing mean volume …")
mean_vol = pwd.mean("time").compute()

# --------------------------------------------------------------------------- #
# Load labelled atlas mask                                                      #
# --------------------------------------------------------------------------- #

print("Loading atlas mask …")
atlas_mask = xr.open_zarr(ATLAS_ZARR_PATH)[ATLAS_VARIABLE].compute()
n_atlas_regions = int(np.sum(np.unique(atlas_mask.values) != 0))
print(f"  {atlas_mask.dims}, {n_atlas_regions} region(s)")

# --------------------------------------------------------------------------- #
# Build Allen atlas colormap                                                    #
# --------------------------------------------------------------------------- #

# Maps Allen CCF integer IDs to normalized RGB tuples (0–1 range).
atlas = BrainGlobeAtlas("allen_mouse_25um")
id_to_rgb = {
    s["id"]: tuple(c / 255 for c in s["rgb_triplet"]) for s in atlas.structures_list
}
print(f"  Built Allen atlas colormap ({len(id_to_rgb)} entries).")

# --------------------------------------------------------------------------- #
# Load binary brain mask (for carpet plot)                                      #
# --------------------------------------------------------------------------- #

print("Loading binary brain mask …")
brain_mask = xr.open_zarr(MASK_ZARR_PATH)[MASK_VARIABLE].compute() > 0

# --------------------------------------------------------------------------- #
# 1 & 2. Napari images                                                          #
# --------------------------------------------------------------------------- #

print("\n── Napari images ─────────────────────────────────────────────────────")

try:
    import napari
    from napari.qt import get_qapp
    from qtpy.QtCore import Qt

    def _napari_screenshot(viewer: "napari.Viewer", path: str) -> None:
        """Take a full-window napari screenshot without displaying the window.

        QWidget.grab() (used by canvas_only=False) requires the widget to have
        been shown at least once for its layout to be initialised. Setting
        WA_DontShowOnScreen before show() runs the full Qt layout pipeline
        without actually mapping the window on screen, so the tiling WM never
        sees or resizes it.
        """
        win = viewer.window._qt_window
        win.setAttribute(Qt.WA_DontShowOnScreen)
        win.show()
        win.resize(1100, 750)
        get_qapp().processEvents()
        viewer.camera.zoom *= 1.2
        get_qapp().processEvents()
        viewer.screenshot(path=path, canvas_only=False)

    # ── Image 1: overview (mean volume) ──────────────────────────────────────
    print("Generating napari-overview.png …")

    viewer = napari.Viewer(show=False)
    mean_vol.fusi.scale.db().fusi.plot.napari(
        viewer=viewer,
        contrast_limits=(-15, 0),
        colormap="gray",
        name="Power Doppler (dB)",
    )
    _napari_screenshot(viewer, str(HERE / "napari-overview.png"))
    viewer.close()
    print("  Saved napari-overview.png")

    # ── Image 2: power Doppler + Allen atlas Labels layer ─────────────────────
    print("Generating napari-labels.png …")

    viewer2 = napari.Viewer(show=False)
    viewer2 = mean_vol.fusi.scale.db().fusi.plot.napari(
        viewer=viewer2,
        contrast_limits=(-15, 0),
        colormap="gray",
        name="Power Doppler (dB)",
    )

    # Build RGBA colormap: transparent background, Allen atlas colors at 70% opacity.
    label_colormap = DirectLabelColormap(
        color_dict={
            0: np.zeros(4),
            None: np.zeros(4),
            **{
                lbl: np.array(list(rgb) + [0.7])
                for lbl, rgb in id_to_rgb.items()
                if lbl > 0
            },
        }
    )

    # plot_napari with layer_type="labels" handles scale/translate from
    # the DataArray coordinates automatically.
    viewer2 = atlas_mask.fusi.plot.napari(
        viewer=viewer2,
        layer_type="labels",
        colormap=label_colormap,
        name="Allen atlas",
        opacity=0.6,
    )

    _napari_screenshot(viewer2, str(HERE / "napari-labels.png"))
    viewer2.close()
    print("  Saved napari-labels.png")

except Exception as exc:
    print(f"  Napari screenshot failed: {exc}")
    print(
        "  → Take Napari screenshots manually:\n"
        "      mean_vol.fusi.scale.db().fusi.plot.napari(\n"
        "          contrast_limits=(-15, 0), colormap='gray'\n"
        "      )\n"
        "  Then use File > Save Screenshot in the Napari menu."
    )

# --------------------------------------------------------------------------- #
# 3. Napari 3D orbit GIF                                                        #
# --------------------------------------------------------------------------- #

print("\n── Napari 3D orbit GIF ────────────────────────────────────────────────")

try:
    print("Generating napari-3d-orbit.gif …")

    vol_3d_orbit = xr.open_zarr(VOLUME_3D_PATH)[VOLUME_3D_VARIABLE]
    if "time" in vol_3d_orbit.dims:
        vol_3d_orbit = vol_3d_orbit.mean("time").compute()

    viewer_orbit = napari.Viewer(show=False)
    # dim_order=("z","x","y") puts depth (y) last so vispy's turntable "up"
    # axis aligns with y, making azimuth orbit around the depth axis.
    # roll=90 in camera angles corrects the displayed orientation so that
    # depth (y) appears vertical (brain surface at top) across all frames.
    viewer_orbit = vol_3d_orbit.fusi.plot.napari(
        viewer=viewer_orbit,
        colormap="gray",
        name="Angiography",
        dim_order=("z", "x", "y"),
    )
    viewer_orbit.dims.ndisplay = 3

    win = viewer_orbit.window._qt_window
    win.setAttribute(Qt.WA_DontShowOnScreen)
    win.show()
    win.resize(1100, 750)
    get_qapp().processEvents()

    # Start at azimuth=270 to show the standard x-y fUSI face first.
    # roll=-90 corrects vertical orientation so brain surface is at top.
    viewer_orbit.camera.angles = (-90.0, -15.0, 270.0)
    viewer_orbit.camera.zoom *= 0.8
    get_qapp().processEvents()

    N_FRAMES = 36  # One frame every 10°, full 360° orbit.
    GIF_WIDTH = 1100  # Match the logical 1100×750 window size of the static PNGs.
    from PIL import Image

    frames_pil = []
    for i in range(N_FRAMES):
        azimuth = 270.0 + i * (360.0 / N_FRAMES)
        viewer_orbit.camera.angles = (-90.0, -15.0, azimuth)
        get_qapp().processEvents()
        # Capture full window (menus included), drop alpha for GIF.
        raw = viewer_orbit.screenshot(canvas_only=False)[..., :3]
        h, w = raw.shape[:2]
        scale = GIF_WIDTH / w
        frames_pil.append(
            Image.fromarray(raw).resize((GIF_WIDTH, int(h * scale)), Image.LANCZOS)
        )

    viewer_orbit.close()

    # Quantize all frames against a shared palette derived from frame 0
    # (which contains the full scene: menus, canvas, dark background). Using a
    # single palette avoids per-frame colour shifts and flickering. dither=0
    # keeps edges clean — important for scientific greyscale content.
    palette_src = frames_pil[0].quantize(colors=256, dither=0)
    quantized = [frame.quantize(palette=palette_src, dither=0) for frame in frames_pil]

    out_path = str(HERE / "napari-3d-orbit.gif")
    quantized[0].save(
        out_path,
        save_all=True,
        append_images=quantized[1:],
        duration=1000 // 8,  # ~8 fps — slow enough to appreciate the 3D structure.
        loop=0,  # Loop forever.
    )
    print("  Saved napari-3d-orbit.gif")
except Exception as exc:
    print(f"  Napari 3D orbit GIF failed: {exc}")


# --------------------------------------------------------------------------- #
# 4. Volume grid — mean volume (matplotlib)                                     #
# --------------------------------------------------------------------------- #

_SAVEFIG_KWARGS = {"dpi": 150, "bbox_inches": "tight", "transparent": True}


def _save_light_dark(plotter, stem: str) -> None:
    """Save light and dark variants of a VolumePlotter figure."""
    plotter.savefig(str(HERE / f"{stem}-light.png"), **_SAVEFIG_KWARGS)
    plotter.close()


def _save_light_dark_carpet(fig_light, fig_dark, stem: str) -> None:
    """Save light and dark variants of a carpet plot figure."""
    fig_light.savefig(str(HERE / f"{stem}-light.png"), **_SAVEFIG_KWARGS)
    plt.close(fig_light)
    fig_dark.savefig(str(HERE / f"{stem}-dark.png"), **_SAVEFIG_KWARGS)
    plt.close(fig_dark)


print("\n── Matplotlib images ─────────────────────────────────────────────────")
print("Generating plot-volume-grid-light/dark.png …")

for black_bg, suffix in [(False, "light"), (True, "dark")]:
    plotter = mean_vol.fusi.scale.db().fusi.plot.volume(
        slice_mode="z",
        cmap="gray",
        vmin=-15,
        vmax=0,
        cbar_label="Power Doppler (dB)",
        black_bg=black_bg,
    )
    plotter.savefig(str(HERE / f"plot-volume-grid-{suffix}.png"), **_SAVEFIG_KWARGS)
    plotter.close()
print("  Saved plot-volume-grid-light.png and plot-volume-grid-dark.png")

# --------------------------------------------------------------------------- #
# 5. Volume grid — true 3D volume (matplotlib)                       #
# --------------------------------------------------------------------------- #

print("Generating plot-volume-3d-light/dark.png …")
vol_3d = xr.open_zarr(VOLUME_3D_PATH)[VOLUME_3D_VARIABLE]
if "time" in vol_3d.dims:
    vol_3d = vol_3d.mean("time").compute()
for black_bg, suffix in [(False, "light"), (True, "dark")]:
    plotter_3d = vol_3d.fusi.plot.volume(
        slice_mode="z",
        cmap="gray",
        show_colorbar=False,
        black_bg=black_bg,
    )
    plotter_3d.savefig(str(HERE / f"plot-volume-3d-{suffix}.png"), **_SAVEFIG_KWARGS)
    plotter_3d.close()
print("  Saved plot-volume-3d-light.png and plot-volume-3d-dark.png")

# --------------------------------------------------------------------------- #
# 6. Volume grid — sliced 3D volume (matplotlib)                       #
# --------------------------------------------------------------------------- #

print("Generating plot-sliced-volume-3d-light/dark.png …")
vol_3d = xr.open_zarr(VOLUME_3D_PATH)[VOLUME_3D_VARIABLE]
if "time" in vol_3d.dims:
    vol_3d = vol_3d.mean("time").compute()
for black_bg, suffix in [(False, "light"), (True, "dark")]:
    plotter_3d = vol_3d.fusi.plot.volume(
        nrows=1,
        slice_mode="z",
        slice_coords=(1.6, 2.4, 3.2),
        cmap="inferno",
        show_colorbar=False,
        black_bg=black_bg,
    )
    plotter_3d.savefig(
        str(HERE / f"plot-sliced-volume-3d-{suffix}.png"), **_SAVEFIG_KWARGS
    )
    plotter_3d.close()
print("  Saved plot-sliced-volume-3d-light.png and plot-sliced-volume-3d-dark.png")

# --------------------------------------------------------------------------- #
# 7. Volume with atlas contours overlaid (matplotlib)                           #
# --------------------------------------------------------------------------- #

print("Generating volume-with-contours-light/dark.png …")

overlay_colors = id_to_rgb if id_to_rgb else "white"

for black_bg, suffix in [(False, "light"), (True, "dark")]:
    plotter_overlay = mean_vol.fusi.scale.db().fusi.plot.volume(
        slice_mode="z",
        cmap="gray",
        vmin=-15,
        vmax=0,
        cbar_label="Power Doppler (dB)",
        black_bg=black_bg,
    )
    plotter_overlay.add_contours(atlas_mask, colors=overlay_colors)
    plotter_overlay.savefig(
        str(HERE / f"volume-with-contours-{suffix}.png"), **_SAVEFIG_KWARGS
    )
    plotter_overlay.close()
print("  Saved volume-with-contours-light.png and volume-with-contours-dark.png")

# --------------------------------------------------------------------------- #
# 8. draw_napari_labels — interactive ROI drawing                               #
# --------------------------------------------------------------------------- #

print("\n── draw_napari_labels ────────────────────────────────────────────────")
print("Generating napari-draw-labels.png …")

try:
    viewer_draw, labels_layer = cf.plotting.draw_napari_labels(
        mean_vol.fusi.scale.db(),
        contrast_limits=(-15, 0),
        colormap="gray",
    )

    # Simulate two painted ROIs by scattering overlapping discs around each
    # centre — this mimics what a napari brush stroke looks like (irregular
    # blob rather than a perfect rectangle).
    ny, nx = labels_layer.data.shape[-2], labels_layer.data.shape[-1]
    y_idx, x_idx = np.ogrid[:ny, :nx]

    def _paint_blob(
        data: np.ndarray,
        cy: int,
        cx: int,
        radius: int,
        label: int,
        rng: np.random.Generator,
    ) -> None:
        """Paint an irregular blob by scattering overlapping discs."""
        n_strokes = 18
        for _ in range(n_strokes):
            dy = int(rng.integers(-radius // 2, radius // 2 + 1))
            dx = int(rng.integers(-radius // 2, radius // 2 + 1))
            r = int(rng.integers(radius // 3, radius // 2 + 1))
            cy_ = int(np.clip(cy + dy, 0, data.shape[-2] - 1))
            cx_ = int(np.clip(cx + dx, 0, data.shape[-1] - 1))
            mask = (y_idx - cy_) ** 2 + (x_idx - cx_) ** 2 <= r**2
            data[..., mask] = label

    rng = np.random.default_rng(0)
    # ROI 1 (label=1): left hemisphere, mid-depth region.
    _paint_blob(labels_layer.data, ny // 3, nx // 4, ny // 16, 1, rng)
    # ROI 2 (label=2): right hemisphere, similar depth.
    _paint_blob(labels_layer.data, ny // 3, 3 * nx // 4, ny // 16, 2, rng)

    labels_layer.refresh()

    _napari_screenshot(viewer_draw, str(HERE / "napari-draw-labels.png"))
    viewer_draw.close()
    print("  Saved napari-draw-labels.png")
except Exception as exc:
    print(f"  draw_napari_labels screenshot failed: {exc}")

# --------------------------------------------------------------------------- #
# 9. Carpet plot (matplotlib)                                                   #
# --------------------------------------------------------------------------- #

print("Generating carpet-plot-light/dark.png …")

for black_bg, suffix in [(False, "light"), (True, "dark")]:
    fig, _ax = pwd.fusi.plot.carpet(
        mask=brain_mask,
        detrend_order=1,
        standardize=True,
        cmap="gray",
        figsize=(12, 4),
        black_bg=black_bg,
    )
    fig.savefig(str(HERE / f"carpet-plot-{suffix}.png"), **_SAVEFIG_KWARGS)
    plt.close(fig)
print("  Saved carpet-plot-light.png and carpet-plot-dark.png")

# --------------------------------------------------------------------------- #

print("\nDone! Rebuild the docs with `just docs` to see the images in place.")

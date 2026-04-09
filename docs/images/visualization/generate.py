"""Generate documentation images for the Visualization user guide.

Data is fetched automatically from the Nunez-Elizalde et al. (2022) fUSI-BIDS
dataset on OSF (https://osf.io/43skw/) via `confusius.datasets`. The first run
downloads ~30 MB; subsequent runs use the local cache.

Atlas overlays (`napari-labels.png` and `volume-with-contours-*.png`) are generated
from the Allen CCF labels in the dataset derivatives (`derivatives/allenccf_align`).
You can still override this by setting `ATLAS_MASK_PATH` to a local Zarr or NIfTI
integer-label mask.

Usage
-----
Run from the project root::

    uv run docs/images/visualization/generate.py

All images are saved to docs/images/visualization/.

Notes
-----
- napari screenshots are taken programmatically. Review them after the script
  finishes and retake manually (File > Save Screenshot in napari) if the canvas
  renders poorly (e.g., all-black).
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import napari
import numpy as np
from napari.qt import get_qapp
from qtpy.QtCore import Qt
from rich.console import Console

import confusius as cf  # noqa: F401  # Register xarray accessors.
from confusius.datasets import fetch_nunez_elizalde_2022

HERE = Path(__file__).parent

_SUBJECT = "CR022"
_SESSION = "20201011"
_TASK = "spontaneous"
_ACQ_SLICE = "slice03"

_DERIVATIVE_ATLAS_REL_PATH = (
    Path("derivatives")
    / "allenccf_align"
    / f"sub-{_SUBJECT}"
    / f"ses-{_SESSION}"
    / "fusi"
    / f"sub-{_SUBJECT}_ses-{_SESSION}_space-fusi_desc-allenccf_dseg.nii.gz"
)
_STRUCTURE_TREE_FILENAME = "structure_tree_safe_2017.csv"
_DERIVATIVE_STRUCTURE_TREE_REL_PATH = (
    Path("derivatives") / "allenccf_align" / _STRUCTURE_TREE_FILENAME
)

_MEAN_DB_LIMITS = (-20.0, 0.0)
_ANGIO_DB_LIMITS = (-20.0, 0.0)

_ORBIT_ELEVATION = 0.0
_ORBIT_START_AZIMUTH = 0.0
_ORBIT_ROLL = 15.0

# ---------------------------------------------------------------------------
# Optional override path to an integer-labelled atlas mask (Zarr or NIfTI).
# If left as None, this script tries to use the derivative Allen CCF atlas
# from the fetched dataset.
# ---------------------------------------------------------------------------

ATLAS_MASK_PATH: str | None = None

console = Console()


def _section(title: str) -> None:
    console.rule(f"[bold]{title}[/bold]")


def _ok(message: str) -> None:
    console.print(f"[green]✓[/green] {message}")


def _warn(message: str) -> None:
    console.print(f"[yellow]![/yellow] {message}")


def _try_int(raw_value: str | None) -> int | None:
    """Parse an integer-like CSV field; return None for missing/invalid values."""
    if raw_value is None:
        return None

    value = raw_value.strip()
    if not value:
        return None

    try:
        return int(float(value))
    except ValueError:
        return None


def _hex_triplet_to_rgb(raw_hex: str) -> tuple[float, float, float] | None:
    """Convert a `RRGGBB` color string into RGB floats in [0, 1]."""
    value = raw_hex.strip().lstrip("#")
    if len(value) != 6:
        return None

    try:
        return tuple(int(value[i : i + 2], 16) / 255 for i in (0, 2, 4))
    except ValueError:
        return None


def _require_structure_tree_csv(bids_root: Path) -> Path:
    """Return structure-tree CSV path from fetched dataset, or fail."""
    csv_path = bids_root / _DERIVATIVE_STRUCTURE_TREE_REL_PATH
    if csv_path.exists():
        return csv_path

    raise RuntimeError(
        "Missing required derivative structure tree CSV: "
        f"{_DERIVATIVE_STRUCTURE_TREE_REL_PATH}. "
        "Recreate and publish dataset_index.json with this file included."
    )


def _build_structure_tree_colormap(
    csv_path: Path,
    atlas_labels: np.ndarray,
) -> tuple[dict[int, tuple[float, float, float]], str, float]:
    """Build `label -> rgb` lookup from the structure-tree CSV.

    The Nunez derivative dseg can encode labels as `graph_order`/`sphinx_id`
    rather than Allen `id`. We auto-detect the best matching key column.
    """
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    label_set = {int(v) for v in atlas_labels.tolist() if int(v) > 0}
    key_candidates = ("graph_order", "sphinx_id", "id")

    best_key = key_candidates[0]
    best_hits = -1
    for key in key_candidates:
        values = {
            parsed
            for row in rows
            if (parsed := _try_int(row.get(key))) is not None and parsed > 0
        }
        hits = len(values & label_set)
        if hits > best_hits:
            best_key = key
            best_hits = hits

    id_to_rgb: dict[int, tuple[float, float, float]] = {}
    for row in rows:
        label = _try_int(row.get(best_key))
        if label is None or label <= 0 or label not in label_set:
            continue

        raw_hex = row.get("color_hex_triplet")
        if raw_hex is None:
            continue

        rgb = _hex_triplet_to_rgb(raw_hex)
        if rgb is not None:
            id_to_rgb[label] = rgb

    coverage = len(id_to_rgb) / len(label_set) if label_set else 0.0
    return id_to_rgb, best_key, coverage


def _collect_isocortex_labels(csv_path: Path, key_name: str) -> set[int]:
    """Return atlas labels belonging to Isocortex descendants.

    Uses Allen structure id 315 (Isocortex) in `structure_id_path`.
    """
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    cortex_labels: set[int] = set()
    for row in rows:
        label = _try_int(row.get(key_name))
        if label is None or label <= 0:
            continue

        structure_path = (row.get("structure_id_path") or "").strip()
        if "/315/" in structure_path:
            cortex_labels.add(label)

    return cortex_labels


def _build_symmetric_cortex_rois(
    atlas_mask_2d: np.ndarray,
    isocortex_labels: set[int],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Build bilateral ROI masks from a single cortical atlas label."""
    atlas_2d = np.asarray(atlas_mask_2d)
    if atlas_2d.ndim != 2:
        raise ValueError(f"Expected 2D atlas mask, got shape {atlas_2d.shape}.")

    ny, nx = atlas_2d.shape
    x_mid = nx // 2
    best_label = -1
    best_score = -1
    best_left = None
    best_right = None

    for raw_label in np.unique(atlas_2d):
        label = int(raw_label)
        if label <= 0 or label not in isocortex_labels:
            continue

        mask = atlas_2d == label
        left = mask.copy()
        left[:, x_mid:] = False
        right = mask.copy()
        right[:, :x_mid] = False

        left_count = int(np.count_nonzero(left))
        right_count = int(np.count_nonzero(right))
        score = min(left_count, right_count)
        if score > best_score:
            best_label = label
            best_score = score
            best_left = left
            best_right = right

    if best_left is None or best_right is None or best_score <= 0:
        raise RuntimeError("Could not find a bilateral isocortex region in atlas mask.")

    return best_left, best_right, best_label


def _normalized_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Return normalized correlation between two arrays."""
    a0 = a.astype(float) - float(np.mean(a))
    b0 = b.astype(float) - float(np.mean(b))
    denom = float(np.sqrt(np.sum(a0 * a0) * np.sum(b0 * b0)))
    if denom == 0.0:
        return 0.0
    return float(np.sum(a0 * b0) / denom)


def _best_matching_z_coordinate(reference_2d, volume_3d) -> float:
    """Find the z coordinate in `volume_3d` best matching `reference_2d`."""
    scores = [
        _normalized_correlation(
            np.asarray(reference_2d.values),
            np.asarray(volume_3d.isel(z=i).values),
        )
        for i in range(volume_3d.sizes["z"])
    ]
    best_idx = int(np.argmax(scores))
    return float(volume_3d["z"].values[best_idx])


# ---------------------------------------------------------------------------
# Fetch dataset and load data
# ---------------------------------------------------------------------------

_section("Load Data")
console.print("Fetching Nunez-Elizalde 2022 dataset")
bids_root = fetch_nunez_elizalde_2022(
    subjects=[_SUBJECT],
    sessions=[_SESSION],
    tasks=[_TASK],
)

_FUSI_PATH = (
    bids_root
    / f"sub-{_SUBJECT}/ses-{_SESSION}/fusi"
    / f"sub-{_SUBJECT}_ses-{_SESSION}_task-{_TASK}_acq-{_ACQ_SLICE}_pwd.nii.gz"
)
_ANGIO_PATH = (
    bids_root
    / f"sub-{_SUBJECT}/ses-{_SESSION}/angio"
    / f"sub-{_SUBJECT}_ses-{_SESSION}_pwd.nii.gz"
)

console.print("Loading power Doppler data")
pwd = cf.load(_FUSI_PATH)
console.print(f"  {pwd.dims}, shape {dict(pwd.sizes)}")

console.print("Computing mean volume")
mean_vol = pwd.mean("time").compute()

# Brain mask: voxels above the 40th percentile of the mean image.
# Used for the carpet plot voxel selection.
brain_mask = mean_vol > float(mean_vol.quantile(0.4))

console.print("Loading angiography (3D volume)")
vol_3d = cf.load(_ANGIO_PATH).compute()
vol_3d_name = "Angiography"

console.print(f"  {vol_3d.dims}, shape {dict(vol_3d.sizes)}")
_z_values = np.asarray(vol_3d["z"].values, dtype=float)
if _z_values.size >= 5:
    _margin_idx = max(1, int(round(0.12 * (_z_values.size - 1))))
    _z_start = float(_z_values[_margin_idx])
    _z_stop = float(_z_values[-_margin_idx - 1])
    if _z_stop <= _z_start:
        _z_start = float(_z_values[1])
        _z_stop = float(_z_values[-2])
else:
    _z_start = float(_z_values.min())
    _z_stop = float(_z_values.max())

SLICED_Z_COORDS = tuple(np.linspace(_z_start, _z_stop, 3).tolist())
console.print(
    "  Using interior linearly spaced z slices for static views: "
    f"{[round(v, 3) for v in SLICED_Z_COORDS]}"
)

# ---------------------------------------------------------------------------
# Optionally load atlas mask
# ---------------------------------------------------------------------------

atlas_mask = None
id_to_rgb: dict = {}
structure_tree_csv: Path | None = None
isocortex_labels: set[int] = set()

atlas_mask_path = (
    Path(ATLAS_MASK_PATH)
    if ATLAS_MASK_PATH is not None
    else bids_root / _DERIVATIVE_ATLAS_REL_PATH
)

if atlas_mask_path is not None and not atlas_mask_path.exists():
    raise RuntimeError(
        "Missing required derivative atlas file: "
        f"{_DERIVATIVE_ATLAS_REL_PATH}. "
        "Recreate and publish dataset_index.json with this file included."
    )

if atlas_mask_path is not None:
    console.print("Loading atlas mask")
    atlas_mask = cf.load(atlas_mask_path).compute()

    if "z" in atlas_mask.dims and "z" in mean_vol.dims:
        if atlas_mask.sizes["z"] != mean_vol.sizes["z"]:
            if mean_vol.sizes["z"] == 1 and vol_3d.sizes.get("z", 0) > 1:
                target_z = _best_matching_z_coordinate(mean_vol.isel(z=0), vol_3d)
                source_z = float(
                    atlas_mask["z"].sel(z=target_z, method="nearest").item()
                )
                atlas_mask = atlas_mask.sel(z=[target_z], method="nearest")
                atlas_mask = atlas_mask.assign_coords(z=mean_vol["z"])
                console.print(
                    "  Matched atlas slice by image similarity: "
                    f"run z={float(mean_vol['z'].item()):.3f} -> atlas z={source_z:.3f}."
                )
            else:
                source_z = np.asarray(
                    atlas_mask["z"].sel(z=mean_vol["z"], method="nearest").values
                )
                atlas_mask = atlas_mask.sel(z=mean_vol["z"], method="nearest")
                atlas_mask = atlas_mask.assign_coords(z=mean_vol["z"])
                console.print(
                    "  Matched atlas z slice(s) "
                    f"{source_z.tolist()} to fUSI z {np.asarray(mean_vol['z'].values).tolist()}."
                )

    atlas_mask = atlas_mask.round().astype(np.int32)
    atlas_labels = np.unique(atlas_mask.values)
    n_atlas_regions = int(np.sum(atlas_labels != 0))
    console.print(f"  {atlas_mask.dims}, {n_atlas_regions} region(s)")

    structure_tree_csv = _require_structure_tree_csv(bids_root)
    id_to_rgb, key_name, coverage = _build_structure_tree_colormap(
        structure_tree_csv,
        atlas_labels,
    )
    if coverage < 1.0:
        raise RuntimeError(
            "Incomplete structure-tree color coverage for atlas labels: "
            f"{coverage:.1%}."
        )
    isocortex_labels = _collect_isocortex_labels(structure_tree_csv, key_name)
    console.print(
        "  Built colormap from structure tree "
        f"({key_name}, {len(id_to_rgb)} labels, {coverage:.1%} coverage)."
    )


def _napari_screenshot(
    viewer: "napari.Viewer",
    path: str,
    *,
    zoom_factor: float = 1.2,
) -> None:
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
    viewer.camera.zoom *= zoom_factor
    get_qapp().processEvents()
    viewer.screenshot(path=path, canvas_only=False)


_SAVEFIG_KWARGS = {"dpi": 150, "bbox_inches": "tight", "transparent": True}


def _save_light_dark(plotter, stem: str) -> None:
    """Save light and dark variants of a VolumePlotter figure."""
    plotter.savefig(str(HERE / f"{stem}-light.png"), **_SAVEFIG_KWARGS)
    plotter.close()


# ---------------------------------------------------------------------------
# 1. napari — overview (mean volume)
# ---------------------------------------------------------------------------

_section("napari images")

try:
    viewer = napari.Viewer(show=False)
    mean_vol.fusi.scale.db().fusi.plot.napari(
        viewer=viewer,
        contrast_limits=_MEAN_DB_LIMITS,
        colormap="gray",
        name="Power Doppler (dB)",
    )
    _napari_screenshot(viewer, str(HERE / "napari-overview.png"))
    viewer.close()
    _ok("Saved napari-overview.png")
except Exception as exc:
    _warn(f"napari-overview.png failed: {exc}")

# ---------------------------------------------------------------------------
# 2. napari — power Doppler + Allen atlas Labels layer (atlas mask required)
# ---------------------------------------------------------------------------

if atlas_mask is not None:
    try:
        from napari.utils.colormaps import DirectLabelColormap

        viewer2 = napari.Viewer(show=False)
        viewer2, _layer2 = mean_vol.fusi.scale.db().fusi.plot.napari(
            viewer=viewer2,
            contrast_limits=_MEAN_DB_LIMITS,
            colormap="gray",
            name="Power Doppler (dB)",
        )
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
        viewer2, _labels_layer = atlas_mask.fusi.plot.napari(
            viewer=viewer2,
            layer_type="labels",
            colormap=label_colormap,
            name="Allen atlas",
            opacity=0.6,
        )
        _napari_screenshot(viewer2, str(HERE / "napari-labels.png"))
        viewer2.close()
        _ok("Saved napari-labels.png")
    except Exception as exc:
        _warn(f"napari-labels.png failed: {exc}")
else:
    _warn("Skipping napari-labels.png (atlas mask unavailable)")

# ---------------------------------------------------------------------------
# 3. napari 3D orbit GIF
# ---------------------------------------------------------------------------

_section("napari 3D orbit GIF")

try:
    viewer_orbit = napari.Viewer(show=True)
    viewer_orbit, _orbit_layer = vol_3d.fusi.scale.db().fusi.plot.napari(
        viewer=viewer_orbit,
        colormap="gray",
        contrast_limits=_ANGIO_DB_LIMITS,
        name=vol_3d_name,
        dim_order=("z", "y", "x"),
    )
    viewer_orbit.dims.ndisplay = 3

    win = viewer_orbit.window._qt_window
    win.setAttribute(Qt.WA_DontShowOnScreen)
    win.show()
    win.resize(1100, 750)
    get_qapp().processEvents()

    viewer_orbit.camera.angles = (_ORBIT_ELEVATION, _ORBIT_START_AZIMUTH, _ORBIT_ROLL)
    try:
        viewer_orbit.camera.up_direction = (0.0, -1.0, 0.0)
    except Exception:
        pass
    viewer_orbit.camera.zoom *= 0.9
    get_qapp().processEvents()

    N_FRAMES = 72  # One frame every 5°, full 360° orbit.
    GIF_WIDTH = 1100
    from PIL import Image

    frames_pil = []
    for i in range(N_FRAMES):
        azimuth = _ORBIT_START_AZIMUTH + i * (360.0 / N_FRAMES)
        viewer_orbit.camera.angles = (_ORBIT_ELEVATION, azimuth, _ORBIT_ROLL)
        get_qapp().processEvents()
        raw = viewer_orbit.screenshot(canvas_only=False)[..., :3]
        h, w = raw.shape[:2]
        scale = GIF_WIDTH / w
        frames_pil.append(
            Image.fromarray(raw).resize((GIF_WIDTH, int(h * scale)), Image.LANCZOS)
        )

    viewer_orbit.close()

    palette_src = frames_pil[0].quantize(colors=256, dither=0)
    quantized = [frame.quantize(palette=palette_src, dither=0) for frame in frames_pil]

    out_path = str(HERE / "napari-3d-orbit.gif")
    quantized[0].save(
        out_path,
        save_all=True,
        append_images=quantized[1:],
        duration=1000 // 8,
        loop=0,
    )
    _ok("Saved napari-3d-orbit.gif")
except Exception as exc:
    _warn(f"napari 3D orbit GIF failed: {exc}")

# ---------------------------------------------------------------------------
# 4. Volume grid — mean volume (matplotlib)
# ---------------------------------------------------------------------------

_section("Matplotlib images")

for black_bg, suffix in [(False, "light"), (True, "dark")]:
    plotter = mean_vol.fusi.scale.db().fusi.plot.volume(
        slice_mode="z",
        cmap="gray",
        vmin=_MEAN_DB_LIMITS[0],
        vmax=_MEAN_DB_LIMITS[1],
        cbar_label="Power Doppler (dB)",
        black_bg=black_bg,
    )
    plotter.savefig(str(HERE / f"plot-volume-grid-{suffix}.png"), **_SAVEFIG_KWARGS)
    plotter.close()
_ok("Saved plot-volume-grid-light.png and plot-volume-grid-dark.png")

# ---------------------------------------------------------------------------
# 5. Volume grid — true 3D volume (matplotlib)
# ---------------------------------------------------------------------------

for black_bg, suffix in [(False, "light"), (True, "dark")]:
    plotter_3d = vol_3d.fusi.scale.db().fusi.plot.volume(
        slice_mode="z",
        cmap="gray",
        vmin=_ANGIO_DB_LIMITS[0],
        vmax=_ANGIO_DB_LIMITS[1],
        show_colorbar=False,
        black_bg=black_bg,
    )
    plotter_3d.savefig(str(HERE / f"plot-volume-3d-{suffix}.png"), **_SAVEFIG_KWARGS)
    plotter_3d.close()
_ok("Saved plot-volume-3d-light.png and plot-volume-3d-dark.png")

# ---------------------------------------------------------------------------
# 6. Volume grid — sliced 3D volume (matplotlib)
# ---------------------------------------------------------------------------

for black_bg, suffix in [(False, "light"), (True, "dark")]:
    plotter_3d = vol_3d.fusi.scale.db().fusi.plot.volume(
        nrows=1,
        slice_mode="z",
        slice_coords=SLICED_Z_COORDS,
        cmap="inferno",
        vmin=_ANGIO_DB_LIMITS[0],
        vmax=_ANGIO_DB_LIMITS[1],
        show_colorbar=False,
        black_bg=black_bg,
    )
    plotter_3d.savefig(
        str(HERE / f"plot-sliced-volume-3d-{suffix}.png"), **_SAVEFIG_KWARGS
    )
    plotter_3d.close()
_ok("Saved plot-sliced-volume-3d-light.png and plot-sliced-volume-3d-dark.png")

# ---------------------------------------------------------------------------
# 7. Volume with atlas contours overlaid (atlas mask required)
# ---------------------------------------------------------------------------

if atlas_mask is not None:
    overlay_colors = id_to_rgb if id_to_rgb else "white"
    for black_bg, suffix in [(False, "light"), (True, "dark")]:
        plotter_overlay = mean_vol.fusi.scale.db().fusi.plot.volume(
            slice_mode="z",
            cmap="gray",
            vmin=_MEAN_DB_LIMITS[0],
            vmax=_MEAN_DB_LIMITS[1],
            cbar_label="Power Doppler (dB)",
            black_bg=black_bg,
        )
        plotter_overlay.add_contours(atlas_mask, colors=overlay_colors)
        plotter_overlay.savefig(
            str(HERE / f"volume-with-contours-{suffix}.png"), **_SAVEFIG_KWARGS
        )
        plotter_overlay.close()
    _ok("Saved volume-with-contours-light.png and volume-with-contours-dark.png")
else:
    _warn("Skipping volume-with-contours-*.png (atlas mask unavailable)")

# ---------------------------------------------------------------------------
# 8. draw_napari_labels — interactive ROI drawing
# ---------------------------------------------------------------------------

_section("draw_napari_labels")

try:
    viewer_draw, labels_layer = cf.plotting.draw_napari_labels(
        mean_vol.fusi.scale.db(),
        contrast_limits=_MEAN_DB_LIMITS,
        colormap="gray",
    )

    labels_layer.data[...] = 0

    if atlas_mask is None or not isocortex_labels:
        raise RuntimeError("Atlas-driven cortex ROIs are unavailable.")

    atlas_values = np.asarray(atlas_mask.values)
    atlas_2d = atlas_values[0] if atlas_values.ndim == 3 else atlas_values
    left_roi, right_roi, label_id = _build_symmetric_cortex_rois(
        atlas_2d,
        isocortex_labels,
    )

    labels_layer.data[..., left_roi] = 1
    labels_layer.data[..., right_roi] = 2
    console.print(
        f"  Painted symmetric cortex ROIs from atlas segmentation (label {label_id})"
    )

    labels_layer.refresh()

    _napari_screenshot(
        viewer_draw,
        str(HERE / "napari-draw-labels.png"),
        zoom_factor=1.0,
    )
    viewer_draw.close()
    _ok("Saved napari-draw-labels.png")
except Exception as exc:
    _warn(f"draw_napari_labels screenshot failed: {exc}")

# ---------------------------------------------------------------------------
# 9. Carpet plot (matplotlib)
# ---------------------------------------------------------------------------

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
_ok("Saved carpet-plot-light.png and carpet-plot-dark.png")

# ---------------------------------------------------------------------------

_ok("Done! Rebuild docs with `just docs` to preview changes")

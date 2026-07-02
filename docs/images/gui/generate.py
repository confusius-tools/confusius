"""Generate documentation images for the GUI guide.

Two datasets are fetched automatically via `confusius.datasets`:

- **Nunez-Elizalde et al. (2022)** fUSI-BIDS dataset on OSF
  (https://osf.io/43skw/) — used for the Data I/O, Signals, and QC
  screenshots. First run downloads ~30 MB.
- **Cybis Pereira et al. (2026)** fUSI-BIDS dataset on OSF
  (https://osf.io/2v6f7/) — used for the Video panel GIF. First run
  downloads ~200 MB (raw fUSI + DLC video).

Subsequent runs use the local cache.

Usage
-----
Run from the project root::

    uv run docs/images/gui/generate.py

Outputs (all saved to docs/images/gui/):

- `plugin-data-io.png` — Data I/O panel with a scan loaded.
- `plugin-signals.png` — Signals panel in hover mode.
- `plugin-signals-points.png` — Signals panel in points mode.
- `plugin-signals-labels.png` — Signals panel in labels mode.
- `plugin-events-create.gif` — GIF of creating events with the Start/End workflow.
- `plugin-qc.png` — QC panel with DVARS, carpet, and CV computed.
- `plugin-video.gif` — Video panel with video synced to the fUSI acquisition.

Notes
-----
- Screenshots are taken programmatically using `WA_DontShowOnScreen` so no window
  appears on screen. Review them after the script finishes and retake manually
  (File > Save Screenshot in napari) if the canvas renders poorly (e.g. all-black).
- The ConfUSIus widget is instantiated directly and docked on the right, exactly as
  napari would do when the user opens it via Plugins > ConfUSIus.
"""

import csv
from pathlib import Path

import napari
import numpy as np
from napari.layers import Image
from napari.qt import get_qapp
from qtpy.QtCore import QEventLoop, Qt, QTimer
from rich.console import Console

import confusius as cf  # noqa: F401  # Register xarray accessors.
from confusius.datasets import fetch_cybis_pereira_2026, fetch_nunez_elizalde_2022

HERE = Path(__file__).parent

_SUBJECT = "CR022"
_SESSION = "20201011"
_TASK = "spontaneous"
_ACQ_SLICE = "slice04"

_SLICE_INDEX = int(_ACQ_SLICE.replace("slice", ""))

_ROI_STRUCTURE_ID = 1089
_ROI_STRUCTURE_NAME = "HIP"

_DERIVATIVE_ATLAS_REL_PATH = (
    Path("derivatives")
    / "allenccf_align"
    / f"sub-{_SUBJECT}"
    / f"ses-{_SESSION}"
    / "fusi"
    / f"sub-{_SUBJECT}_ses-{_SESSION}_space-fusi_desc-allenccf_dseg.nii.gz"
)
_DERIVATIVE_STRUCTURE_TREE_REL_PATH = (
    Path("derivatives") / "allenccf_align" / "structure_tree_safe_2017.csv"
)
_ANGIO_REL_PATH = (
    Path(f"sub-{_SUBJECT}")
    / f"ses-{_SESSION}"
    / "angio"
    / f"sub-{_SUBJECT}_ses-{_SESSION}_pwd.nii.gz"
)

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


def _detect_structure_label_key(csv_path: Path, atlas_labels: np.ndarray) -> str:
    """Detect which CSV integer column encodes atlas labels."""
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    label_set = {int(v) for v in atlas_labels.tolist() if int(v) > 0}
    candidates = ("graph_order", "sphinx_id", "id")

    best_key = candidates[0]
    best_hits = -1
    for key in candidates:
        values = {
            parsed
            for row in rows
            if (parsed := _try_int(row.get(key))) is not None and parsed > 0
        }
        hits = len(values & label_set)
        if hits > best_hits:
            best_hits = hits
            best_key = key

    if best_hits < len(label_set):
        raise RuntimeError(
            f"Incomplete atlas label coverage with key '{best_key}': "
            f"{best_hits}/{len(label_set)}."
        )
    return best_key


def _collect_labels_for_structure(
    csv_path: Path,
    key_name: str,
    structure_id: int,
) -> set[int]:
    """Return atlas labels belonging to descendants of a structure id."""
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    structure_labels: set[int] = set()
    token = f"/{structure_id}/"
    for row in rows:
        label = _try_int(row.get(key_name))
        if label is None or label <= 0:
            continue
        structure_path = (row.get("structure_id_path") or "").strip()
        if token in structure_path:
            structure_labels.add(label)

    return structure_labels


def _build_bilateral_roi_masks(
    atlas_mask_2d: np.ndarray,
    roi_labels: set[int],
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Build bilateral ROI masks from all labels in a target structure."""
    atlas_2d = np.asarray(atlas_mask_2d)
    if atlas_2d.ndim != 2:
        raise ValueError(f"Expected 2D atlas mask, got shape {atlas_2d.shape}.")

    _, nx = atlas_2d.shape
    x_mid = nx // 2
    labels_in_slice = sorted(
        int(raw_label)
        for raw_label in np.unique(atlas_2d)
        if int(raw_label) > 0 and int(raw_label) in roi_labels
    )
    if not labels_in_slice:
        raise RuntimeError("Could not find target ROI labels in atlas slice.")

    roi_mask = np.isin(atlas_2d, labels_in_slice)
    left = roi_mask.copy()
    left[:, x_mid:] = False
    right = roi_mask.copy()
    right[:, :x_mid] = False
    if int(np.count_nonzero(left)) == 0 or int(np.count_nonzero(right)) == 0:
        raise RuntimeError(
            "Target ROI does not have bilateral coverage in atlas slice."
        )

    return left, right, labels_in_slice


def _point_inside_mask_near_centroid(mask_2d: np.ndarray) -> tuple[float, float]:
    """Return an in-mask point closest to the mask centroid."""
    points = np.argwhere(mask_2d)
    if points.size == 0:
        raise RuntimeError("Cannot compute point for an empty ROI mask.")

    centroid = np.mean(points, axis=0)
    distances2 = np.sum((points - centroid) ** 2, axis=1)
    best_idx = int(np.argmin(distances2))
    y, x = points[best_idx]
    return float(y), float(x)


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
    acqs=[_ACQ_SLICE],
)

_FUSI_PATH = (
    bids_root
    / f"sub-{_SUBJECT}/ses-{_SESSION}/fusi"
    / f"sub-{_SUBJECT}_ses-{_SESSION}_task-{_TASK}_acq-{_ACQ_SLICE}_pwd.nii.gz"
)

console.print("Loading data")
da = cf.load(_FUSI_PATH)
console.print(f"  {da.dims}, shape {dict(da.sizes)}")

console.print("Computing display contrast limits")
_mean_display = da.mean("time").compute()
DISPLAY_GAMMA = 0.4
DISPLAY_CONTRAST = (
    float(da.min()),
    float(da.quantile(0.9995)),
)
console.print(
    "  contrast_limits="
    f"({DISPLAY_CONTRAST[0]:.1f}, {DISPLAY_CONTRAST[1]:.1f}), gamma={DISPLAY_GAMMA}"
)

console.print("Preparing atlas-driven cortex ROIs for labels screenshot")
angio = cf.load(bids_root / _ANGIO_REL_PATH).compute()
atlas_path = bids_root / _DERIVATIVE_ATLAS_REL_PATH
if not atlas_path.exists():
    raise RuntimeError(
        "Missing required derivative atlas file: "
        f"{_DERIVATIVE_ATLAS_REL_PATH}. "
        "Recreate and publish dataset_index.json with this file included."
    )

structure_tree_csv = bids_root / _DERIVATIVE_STRUCTURE_TREE_REL_PATH
if not structure_tree_csv.exists():
    raise RuntimeError(
        "Missing required derivative structure tree CSV: "
        f"{_DERIVATIVE_STRUCTURE_TREE_REL_PATH}. "
        "Recreate and publish dataset_index.json with this file included."
    )

atlas_mask = cf.load(atlas_path).compute().round().astype(np.int32)
target_z = _best_matching_z_coordinate(_mean_display.isel(z=0), angio)
atlas_slice = atlas_mask.sel(z=[target_z], method="nearest")

atlas_labels = np.unique(np.asarray(atlas_mask.values))
label_key = _detect_structure_label_key(structure_tree_csv, atlas_labels)
roi_labels = _collect_labels_for_structure(
    structure_tree_csv,
    label_key,
    _ROI_STRUCTURE_ID,
)

roi_mask = np.isin(np.asarray(atlas_slice.values), list(roi_labels))
if not np.any(roi_mask):
    raise RuntimeError(
        f"No {_ROI_STRUCTURE_NAME} labels found in selected atlas slice."
    )

atlas_2d = np.asarray(atlas_slice.values)[0]
GUI_LEFT_ROI, GUI_RIGHT_ROI, GUI_LABEL_IDS = _build_bilateral_roi_masks(
    atlas_2d,
    roi_labels,
)
left_y, left_x = _point_inside_mask_near_centroid(GUI_LEFT_ROI)
right_y, right_x = _point_inside_mask_near_centroid(GUI_RIGHT_ROI)
GUI_POINT_LEFT = np.array([0.0, left_y, left_x])
GUI_POINT_RIGHT = np.array([0.0, right_y, right_x])
console.print(
    "  Using atlas z="
    f"{float(atlas_slice['z'].values[0]):.3f} "
    f"for acq-slice{_SLICE_INDEX:02d} {_ROI_STRUCTURE_NAME} ROIs "
    f"({len(GUI_LABEL_IDS)} labels)"
)

# ---------------------------------------------------------------------------
# Video dataset (Cybis-Pereira 2026) — used for the time-scrubbing plugin GIF.
# Scan is already in display orientation upstream; no preprocessing needed.
# ---------------------------------------------------------------------------

_VIDEO_DATASETS = ["rawdata", "dlc-videos"]
_VIDEO_SUBJECT = "rat75"
_VIDEO_SESSION = "20220525"
_VIDEO_ACQ_SLICE = "slice37"

console.print("Fetching Cybis-Pereira 2026 dataset (for video GIF)")
video_bids_root = fetch_cybis_pereira_2026(
    datasets=_VIDEO_DATASETS,
    subjects=[_VIDEO_SUBJECT],
    sessions=[_VIDEO_SESSION],
    acqs=[_VIDEO_ACQ_SLICE],
)

_VIDEO_FUSI_PATH = (
    video_bids_root
    / f"sub-{_VIDEO_SUBJECT}/ses-{_VIDEO_SESSION}/fusi"
    / f"sub-{_VIDEO_SUBJECT}_ses-{_VIDEO_SESSION}_task-openfield_acq-{_VIDEO_ACQ_SLICE}_pwd.nii.gz"
)
if not _VIDEO_FUSI_PATH.is_file():
    raise FileNotFoundError(f"Video-dataset fUSI not found at {_VIDEO_FUSI_PATH}")

_VIDEO_MP4_PATH = (
    video_bids_root
    / "derivatives/dlc-videos"
    / f"sub-{_VIDEO_SUBJECT}/ses-{_VIDEO_SESSION}/video"
    / f"sub-{_VIDEO_SUBJECT}_ses-{_VIDEO_SESSION}_task-openfield_tracksys-DLC_acq-{_VIDEO_ACQ_SLICE}_video.mp4"
)
if not _VIDEO_MP4_PATH.is_file():
    raise FileNotFoundError(f"Video not found at {_VIDEO_MP4_PATH}")

console.print("Computing video-dataset display contrast")
_video_pwd = cf.load(_VIDEO_FUSI_PATH)
VIDEO_TIME_AXIS = list(_video_pwd.dims).index("time")
VIDEO_DISPLAY_CONTRAST = (
    float(_video_pwd.min()),
    float(_video_pwd.quantile(0.9995)),
)
del _video_pwd


def _napari_screenshot(viewer: napari.Viewer, path: str) -> None:
    """Take a full-window napari screenshot without displaying the window.

    `QWidget.grab()` (used by `canvas_only=False`) requires the widget to have
    been shown at least once for its layout to be initialised. Setting
    `WA_DontShowOnScreen` before `show()` runs the full Qt layout pipeline
    without actually mapping the window on screen, so the tiling WM never
    sees or resizes it.
    """
    win = viewer.window._qt_window
    win.setAttribute(Qt.WA_DontShowOnScreen)
    win.show()
    win.resize(1400, 900)
    get_qapp().processEvents()
    get_qapp().processEvents()
    viewer.screenshot(path=path, canvas_only=False)


def _qt_sleep(ms: int) -> None:
    """Block for *ms* milliseconds while keeping the Qt event loop running.

    Unlike `time.sleep`, this allows QTimers and QPropertyAnimations to fire
    normally, which is required for accordion animations to complete.
    """
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def _open_accordion(widget, idx: int) -> None:
    """Show accordion panel *idx* and hide the others — no animation.

    Directly sets panel visibility rather than going through the animated click
    handler. Animations depend on the widget being already laid out (so that
    `p.height()` is non-zero), which is not guaranteed in the headless
    screenshot setup.
    """
    btns_and_icons = widget._accordion_btns
    container = btns_and_icons[0][0].parent()
    layout = container.layout()

    for i, (btn, _) in enumerate(btns_and_icons):
        active = i == idx
        btn.blockSignals(True)
        btn.setChecked(active)
        btn.blockSignals(False)
        # The container layout interleaves buttons and panels: btn0, panel0,
        # btn1, panel1, … so panel i is at layout index 2*i + 1.
        item = layout.itemAt(2 * i + 1)
        if item and item.widget():
            panel = item.widget()
            panel.setMaximumHeight(16777215)
            panel.setVisible(active)

    get_qapp().processEvents()


def _accordion_index(widget, title: str) -> int:
    """Return the accordion position of the section with the given title.

    Resolving the index from the section title keeps the screenshot code robust
    to accordion reordering or the insertion of new sections.
    """
    return list(widget._accordion_panels).index(title)


# ---------------------------------------------------------------------------
# 1. Data I/O panel — file loaded, save section visible
# ---------------------------------------------------------------------------

_section("Plugin Screenshots")

try:
    from confusius._napari._widget import ConfUSIusWidget
    from confusius.plotting.napari import plot_napari

    viewer = napari.Viewer(show=False)
    _viewer, _layer = plot_napari(
        da,
        viewer=viewer,
        gamma=DISPLAY_GAMMA,
        contrast_limits=DISPLAY_CONTRAST,
    )

    widget = ConfUSIusWidget(viewer)
    viewer.window.add_dock_widget(widget, name="ConfUSIus", area="right")
    _qt_sleep(200)  # Let the viewer and sliders settle.

    # Data I/O is accordion index 0 — already open by default.
    _napari_screenshot(viewer, str(HERE / "plugin-data-io.png"))
    viewer.close()
    _ok("Saved plugin-data-io.png")
except Exception as exc:
    _warn(f"plugin-data-io.png failed: {exc}")

# ---------------------------------------------------------------------------
# 2. Signals panel — hover mode, voxel time series at spatial centre
# ---------------------------------------------------------------------------

try:
    import numpy as np

    viewer2 = napari.Viewer(show=False)
    _viewer2, _layer2 = plot_napari(
        da,
        viewer=viewer2,
        gamma=DISPLAY_GAMMA,
        contrast_limits=DISPLAY_CONTRAST,
    )

    widget2 = ConfUSIusWidget(viewer2)
    viewer2.window.add_dock_widget(widget2, name="ConfUSIus", area="right")
    _qt_sleep(200)

    # Open Signals panel (index 2).
    _open_accordion(widget2, 2)

    # Retrieve the Signals panel from the accordion container layout.
    _container2 = widget2._accordion_btns[0][0].parent()
    ts_panel = _container2.layout().itemAt(2 * 2 + 1).widget()

    # Open the bottom dock with the signals plotter.
    plotter = ts_panel._ensure_plotter()
    _qt_sleep(350)  # Let the dock resize QTimer.singleShot(200, …) fire.

    # Inject a signal from the spatial centre of the volume directly,
    # bypassing the actual mouse event so no display is needed.
    layer2 = viewer2.layers[0]
    center_data = np.array([0] + [s // 2 for s in layer2.data.shape[1:]], dtype=float)
    plotter._current_layer = layer2
    plotter._cursor_pos = np.array(layer2.data_to_world(center_data))
    plotter._update_plot()
    get_qapp().processEvents()

    viewer2.window._qt_window.resize(1400, 1050)
    get_qapp().processEvents()
    _napari_screenshot(viewer2, str(HERE / "plugin-signals.png"))
    viewer2.close()
    _ok("Saved plugin-signals.png")
except Exception as exc:
    _warn(f"plugin-signals.png failed: {exc}")

# ---------------------------------------------------------------------------
# 3. QC panel — DVARS, Carpet plot, and CV computed and displayed
# ---------------------------------------------------------------------------

try:
    from confusius.plotting.image import _prepare_carpet_data
    from confusius.qc import compute_cv, compute_dvars

    viewer3 = napari.Viewer(show=False)
    _viewer3, _layer3 = plot_napari(
        da,
        viewer=viewer3,
        gamma=DISPLAY_GAMMA,
        contrast_limits=DISPLAY_CONTRAST,
    )
    layer_name = viewer3.layers[0].name

    widget3 = ConfUSIusWidget(viewer3)
    viewer3.window.add_dock_widget(widget3, name="ConfUSIus", area="right")
    _qt_sleep(200)

    # Open the Quality Control panel, resolving its position by name so the
    # screenshot does not break when sections are reordered or inserted.
    _open_accordion(widget3, _accordion_index(widget3, "Quality Control"))
    qc_panel = widget3._accordion_panels["Quality Control"]

    # Select the layer in the QC panel.
    idx = qc_panel._layer_combo.findText(layer_name)
    if idx >= 0:
        qc_panel._layer_combo.setCurrentIndex(idx)

    # Compute QC metrics synchronously (bypasses the background thread).
    console.print("  Computing DVARS")
    results = {"dvars": compute_dvars(da)}
    console.print("  Computing CV")
    results["cv"] = compute_cv(da)
    console.print("  Computing carpet plot")
    results["carpet"] = _prepare_carpet_data(da)

    # Inject results — this creates the bottom dock and draws the plots.
    qc_panel._on_compute_returned(results, da, layer_name)
    get_qapp().processEvents()

    # Wait for the dock resize QTimer.singleShot(200, …) to fire.
    _qt_sleep(350)

    # Taller window so the bottom dock with plots is clearly visible.
    viewer3.window._qt_window.resize(1400, 1050)
    get_qapp().processEvents()
    _napari_screenshot(viewer3, str(HERE / "plugin-qc.png"))
    viewer3.close()
    _ok("Saved plugin-qc.png")
except Exception as exc:
    _warn(f"plugin-qc.png failed: {exc}")

# ---------------------------------------------------------------------------
# 4. Signals panel — points mode, 2 points with distinct face colours
# ---------------------------------------------------------------------------

try:
    import numpy as np

    viewer4 = napari.Viewer(show=False)
    _viewer4, _layer4 = plot_napari(
        da,
        viewer=viewer4,
        gamma=DISPLAY_GAMMA,
        contrast_limits=DISPLAY_CONTRAST,
    )

    widget4 = ConfUSIusWidget(viewer4)
    viewer4.window.add_dock_widget(widget4, name="ConfUSIus", area="right")
    _qt_sleep(200)

    # Open Signals panel (index 2).
    _open_accordion(widget4, 2)
    _container4 = widget4._accordion_btns[0][0].parent()
    ts_panel4 = _container4.layout().itemAt(2 * 2 + 1).widget()

    layer4 = viewer4.layers[0]
    shape4 = layer4.data.shape[1:]  # (z, y, x)
    scale_3d4 = layer4.scale[1:]
    translate_3d4 = layer4.translate[1:]

    # Place points at the centroids of the two atlas-derived cortical ROIs.
    pt_red = GUI_POINT_LEFT
    pt_teal = GUI_POINT_RIGHT
    pts_layer4 = viewer4.add_points(
        np.array([pt_red, pt_teal]),
        name="ROI Points",
        scale=scale_3d4,
        translate=translate_3d4,
        face_color=["#e94b5f", "#3ad9a4"],
        size=2.0,
        out_of_slice_display=True,
    )

    # Open the bottom dock.
    plotter4 = ts_panel4._ensure_plotter()
    _qt_sleep(350)

    # Re-activate the image layer so the x-axis dropdown picks up its xarray dims
    # (the Points layer added above steals focus, making the combo empty).
    viewer4.layers.selection.active = layer4
    get_qapp().processEvents()

    # Select the Points radio button on the panel so the UI reflects the correct state
    # (radio checked, combo enabled and showing "ROI Points"). The radio toggle fires
    # _on_source_mode_changed → _sync_source_to_plotter, which sets the layer and mode
    # on the plotter automatically.
    ts_panel4._radio_points.setChecked(True)
    get_qapp().processEvents()

    viewer4.window._qt_window.resize(1400, 1050)
    get_qapp().processEvents()
    _napari_screenshot(viewer4, str(HERE / "plugin-signals-points.png"))
    viewer4.close()
    _ok("Saved plugin-signals-points.png")
except Exception as exc:
    _warn(f"plugin-signals-points.png failed: {exc}")

# ---------------------------------------------------------------------------
# 5. Signals panel — labels mode, 3 labelled regions (auto-generated)
# ---------------------------------------------------------------------------

try:
    import numpy as np

    viewer5 = napari.Viewer(show=False)
    _viewer5, _layer5 = plot_napari(
        da,
        viewer=viewer5,
        gamma=DISPLAY_GAMMA,
        contrast_limits=DISPLAY_CONTRAST,
    )

    widget5 = ConfUSIusWidget(viewer5)
    viewer5.window.add_dock_widget(widget5, name="ConfUSIus", area="right")
    _qt_sleep(200)

    # Open Signals panel (index 2).
    _open_accordion(widget5, 2)
    _container5 = widget5._accordion_btns[0][0].parent()
    ts_panel5 = _container5.layout().itemAt(2 * 2 + 1).widget()

    layer5 = viewer5.layers[0]
    shape5 = layer5.data.shape[1:]  # (z, y, x)
    scale_3d5 = layer5.scale[1:]
    translate_3d5 = layer5.translate[1:]

    # Two symmetric cortex ROIs derived from Allen atlas segmentation.
    labels_data = np.zeros(shape5, dtype=np.int32)
    labels_data[0, GUI_LEFT_ROI] = 1
    labels_data[0, GUI_RIGHT_ROI] = 2

    labels_layer5 = viewer5.add_labels(
        labels_data,
        name="Brain Regions",
        scale=scale_3d5,
        translate=translate_3d5,
    )

    # Open the bottom dock.
    plotter5 = ts_panel5._ensure_plotter()
    _qt_sleep(350)

    # Re-activate the image layer so the x-axis dropdown picks up its xarray dims
    # (the Labels layer added above steals focus, making the combo empty).
    viewer5.layers.selection.active = layer5
    get_qapp().processEvents()

    # Select the Labels radio button on the panel so the UI reflects the correct state
    # (radio checked, combo enabled and showing "Brain Regions"). The radio toggle fires
    # _on_source_mode_changed → _sync_source_to_plotter automatically.
    ts_panel5._radio_labels.setChecked(True)
    get_qapp().processEvents()

    viewer5.window._qt_window.resize(1400, 1050)
    get_qapp().processEvents()
    _napari_screenshot(viewer5, str(HERE / "plugin-signals-labels.png"))
    viewer5.close()
    _ok("Saved plugin-signals-labels.png")
except Exception as exc:
    _warn(f"plugin-signals-labels.png failed: {exc}")

# ---------------------------------------------------------------------------
# 6. Video panel — time-scrubbing GIF via the plugin
# ---------------------------------------------------------------------------

try:
    from PIL import Image as _PILImage

    from confusius._napari._data._load_panel import DataPanel
    from confusius._napari._video._video_panel import VideoPanel

    viewer6 = napari.Viewer(show=False)
    widget6 = ConfUSIusWidget(viewer6)
    viewer6.window.add_dock_widget(widget6, name="ConfUSIus", area="right")
    _qt_sleep(200)

    # Load fUSI via the Data I/O panel (scan already in display orientation).
    data_panel = widget6.findChild(DataPanel)
    if data_panel is None:
        raise RuntimeError("DataPanel not found in ConfUSIusWidget")
    data_panel._path_edit.setText(str(_VIDEO_FUSI_PATH))
    data_panel._compute_check.setChecked(False)  # Load eagerly.

    _load_loop = QEventLoop()
    viewer6.layers.events.inserted.connect(lambda _e: _load_loop.quit())
    data_panel._load()
    _load_loop.exec()

    fusi_layer = viewer6.layers[-1]
    assert isinstance(fusi_layer, Image)
    fusi_layer.gamma = DISPLAY_GAMMA
    fusi_layer.contrast_limits = VIDEO_DISPLAY_CONTRAST
    _qt_sleep(100)

    # Attach video via the Video panel, using the fUSI layer as reference.
    video_panel = widget6.findChild(VideoPanel)
    if video_panel is None:
        raise RuntimeError("VideoPanel not found in ConfUSIusWidget")
    ref_idx = video_panel._ref_combo.findText(fusi_layer.name)
    if ref_idx >= 0:
        video_panel._ref_combo.setCurrentIndex(ref_idx)
    video_panel._path_edit.setText(str(_VIDEO_MP4_PATH))
    video_panel._load_from_path()
    _qt_sleep(200)

    # Open the Video accordion section (index 1).
    _open_accordion(widget6, 1)

    # Size the window, then refit camera to layers (napari "home" button).
    win6 = viewer6.window._qt_window
    win6.setAttribute(Qt.WA_DontShowOnScreen)
    win6.show()
    win6.resize(1400, 900)
    get_qapp().processEvents()
    viewer6.reset_view()
    _qt_sleep(100)

    # --- GIF: scrub the time slider so the grid viewers animate. ---
    # Follows the 3D-orbit GIF pattern in docs/images/visualization/generate.py:
    # capture full-window frames, shared-palette quantize, save as animated GIF.
    N_GIF_FRAMES = 60
    GIF_FPS = 12
    GIF_WIDTH = 1100
    # Scrub from 2 s to 17 s of scan world time. Use `set_point` (world
    # coordinate) instead of `set_current_step` (index), because fUSI and
    # video layers have different time scales in the shared grid.
    GIF_T_START_S, GIF_T_STOP_S = 2.0, 17.0
    step_times = np.linspace(GIF_T_START_S, GIF_T_STOP_S, N_GIF_FRAMES)

    frames_pil: list = []
    for t in step_times:
        viewer6.dims.set_point(VIDEO_TIME_AXIS, float(t))
        get_qapp().processEvents()
        get_qapp().processEvents()
        raw = viewer6.screenshot(canvas_only=False)[..., :3]
        h, w = raw.shape[:2]
        scale = GIF_WIDTH / w
        frames_pil.append(
            _PILImage.fromarray(raw).resize(
                (GIF_WIDTH, int(h * scale)), _PILImage.Resampling.LANCZOS
            )
        )

    palette_src = frames_pil[0].quantize(colors=256, dither=0)
    quantized = [frame.quantize(palette=palette_src, dither=0) for frame in frames_pil]

    gif_path = str(HERE / "plugin-video.gif")
    quantized[0].save(
        gif_path,
        save_all=True,
        append_images=quantized[1:],
        duration=1000 // GIF_FPS,
        loop=0,
    )
    viewer6.close()
    _ok("Saved plugin-video.gif")
except Exception as exc:
    _warn(f"plugin-video.gif failed: {exc}")

# ---------------------------------------------------------------------------
# 7. Events panel — GIF of creating events with the Start/End workflow
# ---------------------------------------------------------------------------

try:
    from matplotlib import font_manager
    from PIL import Image as _PILImage
    from PIL import ImageDraw, ImageFont
    from qtpy.QtCore import QPoint
    from qtpy.QtWidgets import QScrollArea

    from confusius._napari._video._video_panel import VideoPanel

    da_gif = cf.load(_VIDEO_FUSI_PATH)

    viewer8 = napari.Viewer(show=False)
    _viewer8, fusi8 = plot_napari(
        da_gif,
        viewer=viewer8,
        gamma=DISPLAY_GAMMA,
        contrast_limits=VIDEO_DISPLAY_CONTRAST,
    )
    widget8 = ConfUSIusWidget(viewer8)
    viewer8.window.add_dock_widget(widget8, name="ConfUSIus", area="right")
    _qt_sleep(200)

    # Two behavioural events to annotate, with absolute world-time onsets and
    # durations (seconds): a "rearing" event, then a "grooming" event.
    gif_time = np.asarray(da_gif.coords["time"].values, dtype=float)
    gif_t0 = float(gif_time[0])
    rearing_onset, rearing_duration = gif_t0 + 300.0, 20.0
    grooming_onset, grooming_duration = gif_t0 + 355.0, 40.0
    settle_t = grooming_onset + grooming_duration

    # --- Labels layer aligned to the fUSI spatial axes, two painted regions. ---
    da_meta8 = fusi8.metadata["xarray"]
    spatial8 = [i for i, d in enumerate(da_meta8.dims) if d in ("pose", "z", "y", "x")]
    spatial_shape8 = tuple(da_meta8.shape[i] for i in spatial8)
    spatial_scale8 = tuple(float(fusi8.scale[i]) for i in spatial8)
    spatial_translate8 = tuple(float(fusi8.translate[i]) for i in spatial8)
    _ny8, _nx8 = spatial_shape8[-2], spatial_shape8[-1]
    _yy8, _xx8 = np.ogrid[:_ny8, :_nx8]
    _r8 = 0.05 * min(_ny8, _nx8)
    _blob1 = ((_yy8 - 0.10 * _ny8) ** 2 + (_xx8 - 0.63 * _nx8) ** 2) < _r8**2
    _blob2 = ((_yy8 - 0.35 * _ny8) ** 2 + (_xx8 - 0.73 * _nx8) ** 2) < _r8**2
    _label_data8 = np.zeros(spatial_shape8, dtype=np.int32)
    _label_data8[0][_blob1] = 1
    _label_data8[0][_blob2] = 2
    labels8 = viewer8.add_labels(
        _label_data8,
        name="Labels (3D)",
        scale=spatial_scale8,
        translate=spatial_translate8,
        opacity=0.7,
    )

    # --- Behavioural video via the Video panel; group fUSI + labels in one cell. ---
    video_panel8 = widget8.findChild(VideoPanel)
    _ref_idx8 = video_panel8._ref_combo.findText(fusi8.name)
    if _ref_idx8 >= 0:
        video_panel8._ref_combo.setCurrentIndex(_ref_idx8)
    video_panel8._path_edit.setText(str(_VIDEO_MP4_PATH))
    video_panel8._load_from_path()
    _qt_sleep(300)
    # stride=2 keeps [fUSI, Labels] overlaid in one grid cell; the video gets its own.
    viewer8.grid.stride = 2
    get_qapp().processEvents()

    events_panel8 = widget8._accordion_panels["Events"]

    # --- Signals plotter in Labels mode (mean signal per region) with the cursor. ---
    signals_panel8 = widget8._accordion_panels["Signals"]
    plotter8 = signals_panel8._ensure_plotter()
    _qt_sleep(350)
    plotter8.set_source_mode("labels")
    plotter8.set_labels_layer(labels8)
    plotter8.set_ref_layers([fusi8])
    plotter8._cursor_world = rearing_onset
    plotter8.set_show_cursor(True)
    plotter8._update_plot_from_labels()

    # Select the fUSI so events and the overlay read its true time coordinate, then
    # put the slider at the first onset and activate the overlay.
    viewer8.layers.selection = {fusi8}
    viewer8.dims.set_point(VIDEO_TIME_AXIS, rearing_onset)
    widget8._time_overlay.check()

    # Open the Events accordion and show the window so the geometry is final.
    _open_accordion(widget8, _accordion_index(widget8, "Events"))
    win8 = viewer8.window._qt_window
    win8.setAttribute(Qt.WA_DontShowOnScreen)
    win8.show()
    win8.resize(1400, 1050)
    get_qapp().processEvents()
    viewer8.reset_view()
    get_qapp().processEvents()

    # Scroll the sidebar down so the Events panel is fully visible.
    scroll8 = widget8.findChild(QScrollArea)
    if scroll8 is not None and scroll8.widget() is not None:
        first_btn8 = widget8._accordion_btns[0][0]
        scroll8.verticalScrollBar().setValue(
            first_btn8.mapTo(scroll8.widget(), QPoint(0, 0)).y()
        )
    get_qapp().processEvents()

    # --- GIF frame capture --------------------------------------------------
    GIF_WIDTH8 = 1100
    GIF_FPS8 = 12
    frames8: list = []

    try:
        badge_font = ImageFont.truetype(font_manager.findfont("DejaVu Sans:bold"), 30)
    except (OSError, ValueError):
        badge_font = ImageFont.load_default()

    def _grab8(badge_text: str | None = None, repeat: int = 1) -> None:
        raw = viewer8.screenshot(canvas_only=False)[..., :3]
        h, w = raw.shape[:2]
        scale = GIF_WIDTH8 / w
        frame = _PILImage.fromarray(raw).resize(
            (GIF_WIDTH8, int(h * scale)), _PILImage.Resampling.LANCZOS
        )
        if badge_text is not None:
            draw = ImageDraw.Draw(frame, "RGBA")
            text_w = draw.textlength(badge_text, font=badge_font)
            pad = 16
            box_w, box_h = text_w + 2 * pad, 52
            box_x = (frame.width - box_w) / 2
            box_y = frame.height - box_h - 28
            draw.rounded_rectangle(
                (box_x, box_y, box_x + box_w, box_y + box_h),
                radius=12,
                fill=(20, 20, 28, 225),
            )
            draw.text(
                (box_x + pad, box_y + 10),
                badge_text,
                font=badge_font,
                fill=(233, 75, 95, 255),
            )
        for _ in range(repeat):
            frames8.append(frame)

    def _set_cursor8(t: float) -> None:
        viewer8.dims.set_point(VIDEO_TIME_AXIS, t)
        plotter8._cursor_world = t
        # Re-render the labels-mode plot so the cursor line moves to the new time
        # (the video frame in the grid also updates to this time step).
        plotter8.set_show_cursor(True)
        plotter8._update_plot_from_labels()
        widget8._time_overlay.update()
        get_qapp().processEvents()
        get_qapp().processEvents()

    def _type_name8(name: str) -> None:
        """Type *name* into the event-name field one character at a time."""
        events_panel8._name_edit.setText("")
        get_qapp().processEvents()
        _grab8(repeat=2)
        for i in range(1, len(name) + 1):
            events_panel8._name_edit.setText(name[:i])
            get_qapp().processEvents()
            _grab8()
        _grab8(repeat=2)

    def _annotate8(name: str, onset: float, duration: float) -> None:
        """Drive the full type → Start → scrub → End workflow for one event."""
        _set_cursor8(onset)
        _type_name8(name)
        # Start (S) marks the onset at the current time.
        events_panel8._on_start()
        get_qapp().processEvents()
        _grab8(badge_text="S  ·  Start", repeat=7)
        # Scrub the time slider forward to the offset.
        for t in np.linspace(onset, onset + duration, 12):
            _set_cursor8(float(t))
            _grab8()
        # End (E) creates the event, shading the plot and filling the table.
        events_panel8._on_end()
        get_qapp().processEvents()
        get_qapp().processEvents()
        _grab8(badge_text="E  ·  End", repeat=7)

    # 1. Annotate the "rearing" event.
    _annotate8("rearing", rearing_onset, rearing_duration)

    # 2. Travel forward (no recording) to the "grooming" onset.
    for t in np.linspace(rearing_onset + rearing_duration, grooming_onset, 14):
        _set_cursor8(float(t))
        _grab8()

    # 3. Annotate the "grooming" event.
    _annotate8("grooming", grooming_onset, grooming_duration)

    # 4. Settle inside the rearing event so the overlay names the active event.
    _set_cursor8(settle_t)
    _grab8(repeat=12)

    # --- Assemble the GIF (shared-palette quantize, like the video GIF). ---
    palette_src8 = frames8[0].quantize(colors=256, dither=0)
    quantized8 = [frame.quantize(palette=palette_src8, dither=0) for frame in frames8]
    gif_path8 = str(HERE / "plugin-events-create.gif")
    quantized8[0].save(
        gif_path8,
        save_all=True,
        append_images=quantized8[1:],
        duration=1000 // GIF_FPS8,
        loop=0,
    )
    viewer8.close()
    _ok("Saved plugin-events-create.gif")
except Exception as exc:
    _warn(f"plugin-events-create.gif failed: {exc}")

# ---------------------------------------------------------------------------

_ok("Done! Rebuild docs with `just docs` to preview changes")

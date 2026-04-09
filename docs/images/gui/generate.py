"""Generate documentation images for the GUI guide.

Data is fetched automatically from the Nunez-Elizalde et al. (2022) fUSI-BIDS
dataset on OSF (https://osf.io/43skw/) via `confusius.datasets`.  The first run
downloads ~30 MB; subsequent runs use the local cache.

Usage
-----
Run from the project root::

    uv run docs/images/gui/generate.py

All images are saved to docs/images/gui/.

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
from napari.qt import get_qapp
from qtpy.QtCore import QEventLoop, Qt, QTimer
from rich.console import Console

import confusius as cf  # noqa: F401  # Register xarray accessors.
from confusius.datasets import fetch_nunez_elizalde_2022

HERE = Path(__file__).parent

_SUBJECT = "CR022"
_SESSION = "20201011"
_TASK = "spontaneous"
_ACQ_SLICE = "slice03"
_SLICE_INDEX = int(_ACQ_SLICE.replace("slice", ""))

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


def _collect_isocortex_labels(csv_path: Path, key_name: str) -> set[int]:
    """Return atlas labels belonging to Isocortex descendants."""
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

    _, nx = atlas_2d.shape
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
cortex_labels = _collect_isocortex_labels(structure_tree_csv, label_key)

cortex_mask = np.isin(np.asarray(atlas_slice.values), list(cortex_labels))
if not np.any(cortex_mask):
    raise RuntimeError("No isocortex labels found in selected atlas slice.")

atlas_2d = np.asarray(atlas_slice.values)[0]
GUI_LEFT_ROI, GUI_RIGHT_ROI, GUI_LABEL_ID = _build_symmetric_cortex_rois(
    atlas_2d,
    cortex_labels,
)
_left_points = np.argwhere(GUI_LEFT_ROI)
_right_points = np.argwhere(GUI_RIGHT_ROI)
GUI_POINT_LEFT = np.array(
    [0.0, np.mean(_left_points[:, 0]), np.mean(_left_points[:, 1])]
)
GUI_POINT_RIGHT = np.array(
    [0.0, np.mean(_right_points[:, 0]), np.mean(_right_points[:, 1])]
)
console.print(
    "  Using atlas z="
    f"{float(atlas_slice['z'].values[0]):.3f} "
    f"for acq-slice{_SLICE_INDEX:02d} cortex ROIs (label {GUI_LABEL_ID})"
)


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


# ---------------------------------------------------------------------------
# 1. Data I/O panel — file loaded, save section visible
# ---------------------------------------------------------------------------

_section("Plugin Screenshots")

try:
    from confusius._napari._widget import ConfUSIusWidget
    from confusius.plotting.image import plot_napari

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

    # Open Signals panel (index 1).
    _open_accordion(widget2, 1)

    # Retrieve the Signals panel from the accordion container layout.
    _container2 = widget2._accordion_btns[0][0].parent()
    ts_panel = _container2.layout().itemAt(2 * 1 + 1).widget()

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

    # Open QC panel (index 2).
    _open_accordion(widget3, 2)

    # Retrieve the QCPanel widget from the accordion container layout.
    # Layout interleaves buttons and panels: btn0, panel0, btn1, panel1, …
    _container3 = widget3._accordion_btns[0][0].parent()
    qc_panel = _container3.layout().itemAt(2 * 2 + 1).widget()

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

    # Open Signals panel (index 1).
    _open_accordion(widget4, 1)
    _container4 = widget4._accordion_btns[0][0].parent()
    ts_panel4 = _container4.layout().itemAt(2 * 1 + 1).widget()

    layer4 = viewer4.layers[0]
    shape4 = layer4.data.shape[1:]  # (z, y, x)
    scale_3d4 = layer4.scale[1:]
    translate_3d4 = layer4.translate[1:]

    # Place points at the centroids of the two atlas-derived cortical ROIs.
    pt_yellow = GUI_POINT_LEFT
    pt_cyan = GUI_POINT_RIGHT
    pts_layer4 = viewer4.add_points(
        np.array([pt_yellow, pt_cyan]),
        name="ROI Points",
        scale=scale_3d4,
        translate=translate_3d4,
        face_color=["#ffd33d", "#00b4d8"],
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

    # Open Signals panel (index 1).
    _open_accordion(widget5, 1)
    _container5 = widget5._accordion_btns[0][0].parent()
    ts_panel5 = _container5.layout().itemAt(2 * 1 + 1).widget()

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

_ok("Done! Rebuild docs with `just docs` to preview changes")

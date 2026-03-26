"""Generate documentation images for the GUI guide.

Usage
-----
1. Fill in the constant below with the path to an example dataset on your machine:

    - `ZARR_PATH`: 4D fUSI recording in Zarr format (or any format supported by
      `confusius.load`). Used to populate the viewer before taking screenshots.

2. Run from the project root::

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

from pathlib import Path

import napari
from napari.qt import get_qapp
from qtpy.QtCore import QEventLoop, Qt, QTimer

import confusius as cf  # noqa: F401  # Register xarray accessors.

HERE = Path(__file__).parent

# == Fill in before running ===================================================

ZARR_PATH = "../../../data/sub-ALD030_ses-ChATChR2FibreMidThalamus_task-awake_acq-motor2dot2_proc-staticsvd50_pwd.zarr/"

# Path to a hand-drawn labels layer (NIfTI or Zarr) for the signal labels
# screenshot. Set to None to fall back to the auto-generated rectangular ROIs.
LABELS_PATH = "../../../data/sub-ALD030_ses-ChATChR2FibreMidThalamus_task-awake_acq-motor2dot2_proc-staticsvd50_labels.zarr/"

# =============================================================================


def _resolve(path: str) -> str:
    """Resolve a path relative to this script's directory if not absolute."""
    p = Path(path)
    return str((HERE / p).resolve() if not p.is_absolute() else p)


ZARR_PATH = _resolve(ZARR_PATH)


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
# Load data
# ---------------------------------------------------------------------------

print("Loading data …")
da = cf.load(ZARR_PATH)
print(f"  {da.dims}, shape {dict(da.sizes)}")

# ---------------------------------------------------------------------------
# 1. Data I/O panel — file loaded, save section visible
# ---------------------------------------------------------------------------

print("\n── Plugin screenshots ────────────────────────────────────────────────")
print("Generating plugin-data-io.png …")

try:
    from confusius._napari._widget import ConfUSIusWidget
    from confusius.plotting.image import plot_napari

    viewer = napari.Viewer(show=False)
    _viewer, _layer = plot_napari(da, viewer=viewer, gamma=0.5)

    widget = ConfUSIusWidget(viewer)
    viewer.window.add_dock_widget(widget, name="ConfUSIus", area="right")
    _qt_sleep(200)  # Let the viewer and sliders settle.

    # Data I/O is accordion index 0 — already open by default.
    _napari_screenshot(viewer, str(HERE / "plugin-data-io.png"))
    viewer.close()
    print("  Saved plugin-data-io.png")
except Exception as exc:
    print(f"  plugin-data-io.png failed: {exc}")

# ---------------------------------------------------------------------------
# 2. Signals panel — hover mode, voxel time series at spatial centre
# ---------------------------------------------------------------------------

print("Generating plugin-signals.png …")

try:
    import numpy as np

    viewer2 = napari.Viewer(show=False)
    _viewer2, _layer2 = plot_napari(da, viewer=viewer2, gamma=0.5)

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
    print("  Saved plugin-signals.png")
except Exception as exc:
    print(f"  plugin-signals.png failed: {exc}")

# ---------------------------------------------------------------------------
# 3. QC panel — DVARS, Carpet plot, and CV computed and displayed
# ---------------------------------------------------------------------------

print("Generating plugin-qc.png …")

try:
    from confusius.plotting.image import _prepare_carpet_data
    from confusius.qc import compute_cv, compute_dvars

    viewer3 = napari.Viewer(show=False)
    _viewer3, _layer3 = plot_napari(da, viewer=viewer3, gamma=0.5)
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
    print("  Computing DVARS …")
    results = {"dvars": compute_dvars(da)}
    print("  Computing CV …")
    results["cv"] = compute_cv(da)
    print("  Computing carpet plot …")
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
    print("  Saved plugin-qc.png")
except Exception as exc:
    print(f"  plugin-qc.png failed: {exc}")

# ---------------------------------------------------------------------------
# 4. Signals panel — points mode, 2 points with distinct face colours
# ---------------------------------------------------------------------------

print("Generating plugin-signals-points.png …")

try:
    import numpy as np

    viewer4 = napari.Viewer(show=False)
    _viewer4, _layer4 = plot_napari(da, viewer=viewer4, gamma=0.5)

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

    # Two points placed symmetrically about the volume centre: 20 voxels up (-y) and
    # ±15 voxels in x (30 voxels apart). Colours match the ConfUSIus brand palette:
    # yellow (#ffd33d) and a complementary cyan (#00b4d8).
    center4 = np.array([shape4[0] // 2, shape4[1] // 2, shape4[2] // 2], dtype=float)
    pt_yellow = center4 + np.array([0, -25, -15])
    pt_cyan = center4 + np.array([0, -25, +15])
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
    print("  Saved plugin-signals-points.png")
except Exception as exc:
    print(f"  plugin-signals-points.png failed: {exc}")

# ---------------------------------------------------------------------------
# 5. Signals panel — labels mode, 3 labelled regions
# ---------------------------------------------------------------------------

print("Generating plugin-signals-labels.png …")

try:
    import numpy as np

    viewer5 = napari.Viewer(show=False)
    _viewer5, _layer5 = plot_napari(da, viewer=viewer5, gamma=0.5)

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

    if LABELS_PATH is not None:
        # Load the hand-drawn labels file provided by the user.
        labels_da = cf.load(_resolve(LABELS_PATH))
        labels_data = labels_da.values.astype(np.int32)
    else:
        # Fallback: three small, localised ROI blobs at different positions.
        labels_data = np.zeros(shape5, dtype=np.int32)
        r = max(2, min(shape5) // 8)  # Half-width of each cubic ROI (mm-agnostic).
        centers = [
            (shape5[0] // 4, shape5[1] // 3, shape5[2] // 2),
            (shape5[0] // 2, shape5[1] // 2, shape5[2] // 3),
            (shape5[0] * 3 // 4, shape5[1] * 2 // 3, shape5[2] * 2 // 3),
        ]
        for label_id, (zc, yc, xc) in enumerate(centers, start=1):
            labels_data[
                max(0, zc - r) : min(shape5[0], zc + r),
                max(0, yc - r) : min(shape5[1], yc + r),
                max(0, xc - r) : min(shape5[2], xc + r),
            ] = label_id

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
    print("  Saved plugin-signals-labels.png")
except Exception as exc:
    print(f"  plugin-signals-labels.png failed: {exc}")

# ---------------------------------------------------------------------------

print("\nDone! Rebuild the docs with `just docs` to see the images in place.")

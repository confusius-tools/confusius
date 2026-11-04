"""Live ROI confounds retain preprocessing-input semantics and stable selection."""

import numpy as np
from napari.layers import Image
from numpy.testing import assert_allclose
from qtpy.QtCore import Qt

from confusius._napari._preprocessing._panel import PreprocessingPanel
from confusius._napari._signals._store import SignalStore
from confusius.plotting.napari import plot_napari
from confusius.signal import clean


def test_roi_confound_uses_input_and_survives_renames(
    make_napari_viewer, sample_voxeldata_3dt, qtbot
):
    """A shared ROI is selected once and re-extracted from preprocessing input."""
    viewer = make_napari_viewer()
    reference_data = sample_voxeldata_3dt.copy(
        data=np.roll(sample_voxeldata_3dt.values, 2, axis=0)
    )
    _, reference = plot_napari(
        reference_data,
        viewer=viewer,
        name="reference",
        show_colorbar=False,
        show_scale_bar=False,
    )
    _, target = plot_napari(
        sample_voxeldata_3dt,
        viewer=viewer,
        name="input",
        show_colorbar=False,
        show_scale_bar=False,
    )
    points = viewer.add_points(
        [[1, 1, 1]],
        name="ROI",
        scale=target.scale[-3:],
        translate=target.translate[-3:],
    )
    store = SignalStore()
    controller = store.live_sources(viewer)
    controller.add_source(points)
    assert isinstance(reference, Image)
    controller.set_references([reference])
    for signal in store.live_signals():
        if signal.source_type == "point":
            store.set_live_signal_visible(signal.id, False)
    panel = PreprocessingPanel(viewer, store)
    qtbot.addWidget(panel)
    panel._source_combo.setCurrentIndex(panel._source_combo.findData(target))
    assert panel._confounds_list.count() == 1
    item = panel._confounds_list.item(0)
    assert item is not None
    signal_id = item.data(Qt.ItemDataRole.UserRole)
    item.setSelected(True)
    reference.name = "reference renamed"
    target.name = "input renamed"
    points.name = "ROI renamed"
    assert panel._source_combo.currentData() is target
    renamed_item = panel._confounds_list.item(0)
    assert renamed_item is not None
    assert renamed_item.data(Qt.ItemDataRole.UserRole) == signal_id
    assert renamed_item.isSelected()
    panel._apply_btn.click()
    qtbot.waitUntil(lambda: "input renamed — cleaned" in viewer.layers, timeout=5000)
    confounds = (
        sample_voxeldata_3dt.isel(k=1, j=1, i=1)
        .drop_vars(["z", "y", "x"])
        .expand_dims("confound")
    )
    expected = clean(
        sample_voxeldata_3dt,
        confounds=confounds,
        detrend_order=None,
        standardize_method=None,
        standardize_confounds=True,
        ensure_finite=False,
    )
    assert_allclose(
        viewer.layers["input renamed — cleaned"].data,
        expected.values,
        rtol=1e-5,
        atol=1e-8,
    )
    controller.close()

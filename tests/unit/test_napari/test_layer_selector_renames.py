"""Regression tests for layer identity across selector refreshes."""

from __future__ import annotations

import numpy as np
import pytest
from qtpy.QtTest import QSignalSpy

from confusius._napari._data._save_panel import SavePanel
from confusius._napari._qc._panel import QCPanel
from confusius._napari._registration._panel import RegistrationPanel
from confusius._napari._registration._panel_transforms import (
    get_selected_initial_transform_payload,
    get_selected_manual_initialization_layer,
)
from confusius._napari._registration._transform_payloads import (
    make_affine_transform_payload,
)
from confusius._napari._signals._panel import SignalPanel
from confusius._napari._video._video_panel import VideoPanel
from confusius.registration import RegistrationDiagnostics


@pytest.mark.parametrize(
    ("panel_type", "combo_name", "layer_type"),
    [
        (RegistrationPanel, "_moving_combo", "image"),
        (RegistrationPanel, "_fixed_combo", "image"),
        (RegistrationPanel, "_moving_mask_combo", "labels"),
        (RegistrationPanel, "_fixed_mask_combo", "labels"),
        (RegistrationPanel, "_transform_target_combo", "image"),
        (SignalPanel, "_points_combo", "points"),
        (SignalPanel, "_labels_combo", "labels"),
        (SignalPanel, "_ref_combo", "image"),
        (QCPanel, "_layer_combo", "image"),
        (VideoPanel, "_ref_combo", "image"),
        (SavePanel, "_layer_combo", "image"),
        (SavePanel, "_template_combo", "image"),
    ],
)
def test_rename_preserves_selected_layer(
    make_napari_viewer, sample_voxeldata_3dt, panel_type, combo_name, layer_type
):
    """Selected and unselected renames update text without switching layers."""
    viewer = make_napari_viewer()
    panel = panel_type(viewer)
    if layer_type == "image":
        data = sample_voxeldata_3dt.values
    elif layer_type == "labels":
        data = np.zeros((2, 3, 4), dtype=np.uint8)
    else:
        data = np.array([[0, 0, 0]])
    add_layer = getattr(viewer, f"add_{layer_type}")
    first, selected = (
        add_layer(
            data,
            name=name,
            metadata={"xarray": sample_voxeldata_3dt} if layer_type == "image" else {},
        )
        for name in ("first", "selected")
    )
    combo = getattr(panel, combo_name)
    combo.setCurrentText(selected.name)
    spy = QSignalSpy(combo.currentIndexChanged)

    selected.name = "renamed selected"
    assert combo.currentText() == selected.name
    assert combo.currentData() is selected
    assert len(spy) == 0

    first.name = "renamed first"
    assert combo.findText(first.name) >= 0
    assert combo.currentData() is selected
    assert len(spy) == 0

    # Napari coerces duplicate names; selectors must show the final unique name.
    selected.name = first.name
    assert selected.name != first.name
    assert combo.currentText() == selected.name
    assert combo.currentData() is selected

    viewer.layers.remove(selected)
    assert combo.currentData() is not selected
    assert combo.findText(selected.name) == -1


@pytest.mark.parametrize(
    ("panel_type", "combo_name", "sentinel"),
    [
        (RegistrationPanel, "_moving_mask_combo", ""),
        (RegistrationPanel, "_fixed_mask_combo", ""),
        (RegistrationPanel, "_initialization_combo", "none"),
        (SignalPanel, "_ref_combo", "All image layers"),
        (SavePanel, "_template_combo", None),
    ],
)
def test_rename_preserves_optional_selection(
    make_napari_viewer, sample_voxeldata_3d, panel_type, combo_name, sentinel
):
    """Renaming does not replace an optional choice with a layer."""
    viewer = make_napari_viewer()
    panel = panel_type(viewer)
    layer = viewer.add_labels(np.zeros((2, 3, 4), dtype=np.uint8), name="mask")
    image = viewer.add_image(
        sample_voxeldata_3d.values,
        name="image",
        metadata={"xarray": sample_voxeldata_3d},
    )
    combo = getattr(panel, combo_name)
    if sentinel is None:
        combo.setCurrentIndex(-1)
    else:
        combo.setCurrentText(sentinel)
    index = combo.currentIndex()
    layer.name = "renamed mask"
    image.name = "renamed image"
    assert combo.currentIndex() == index
    assert combo.currentData() is None


@pytest.mark.parametrize("kind", ["layer", "manual"])
def test_rename_preserves_registration_transform(
    make_napari_viewer, sample_voxeldata_3d, kind
):
    """Decorated transform entries keep their payload and layer identity."""
    viewer = make_napari_viewer()
    panel = RegistrationPanel(viewer)
    for name in ("first", "selected"):
        layer = viewer.add_image(
            sample_voxeldata_3d.values,
            name=name,
            metadata={"xarray": sample_voxeldata_3d},
        )
        affine = np.eye(4)
        affine[0, 3] = 1
        if kind == "manual":
            layer.affine = affine
        else:
            layer.metadata["confusius_transform"] = make_affine_transform_payload(
                affine,
                reference=sample_voxeldata_3d,
                source_layer_name=name,
                target_layer_name="target",
                operation="register_volume",
                transform_model="affine",
                metric="correlation",
                diagnostics=RegistrationDiagnostics(
                    metric="correlation",
                    metric_values=np.array([-1.0]),
                    final_metric_value=-1.0,
                    n_iterations=1,
                    stop_condition="done",
                    status="completed",
                ),
            )
    # Metadata changes are not part of layer-list refresh events.
    panel._refresh_layers()
    combos = (panel._transform_source_combo, panel._initialization_combo)
    for combo in combos:
        index = next(
            i for i in range(combo.count()) if combo.itemData(i) == (kind, layer)
        )
        combo.setCurrentIndex(index)
    layer.name = "renamed selected"
    for combo in combos:
        assert combo.currentData() == (kind, layer)
        if kind == "manual":
            assert combo.currentText() == f"{layer.name} (manual)"
        else:
            # Stored transform names describe their provenance, not the layer name.
            assert combo.currentText() == layer.metadata["confusius_transform"]["name"]
    if kind == "layer":
        assert (
            get_selected_initial_transform_payload(panel)
            is layer.metadata["confusius_transform"]
        )
    else:
        assert get_selected_manual_initialization_layer(panel) is layer

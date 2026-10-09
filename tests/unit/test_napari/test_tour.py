"""Unit tests for the guided tour's Atlas steps.

The tour is built against a real plugin widget but never started: the steps'
`pre_action` callbacks are called directly, since those are what reveal UI that only
exists after user actions, and `on_close` must put that UI back.
"""

from __future__ import annotations

import pytest
from qtpy.QtWidgets import QApplication

from confusius._napari._atlas._panel import AtlasPanel
from confusius._napari._tour import build_default_tour
from confusius._napari._widget import ConfUSIusWidget


@pytest.fixture
def widget(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = ConfUSIusWidget(viewer)
    viewer.window.add_dock_widget(widget, area="right")
    # Keep the Atlas panel off the network when the tour expands it.
    atlas_panel = widget._accordion_panels["Atlas"]
    assert isinstance(atlas_panel, AtlasPanel)
    atlas_panel._atlases_listed = True
    return widget


def _steps_by_title(tour) -> dict:
    return {step.title: step for step in tour._steps}


def test_atlas_steps_reveal_hidden_ui_and_restore_it(widget) -> None:
    atlas_panel = widget._accordion_panels["Atlas"]
    assert isinstance(atlas_panel, AtlasPanel)
    tour = build_default_tour(widget, is_dark=True)
    steps = _steps_by_title(tour)

    steps["Region Masks"].pre_action()
    QApplication.processEvents()
    assert atlas_panel._masks_group.isVisibleTo(atlas_panel)
    assert steps["Region Masks"].target() is atlas_panel._masks_group

    steps["fUSI Templates"].pre_action()
    QApplication.processEvents()
    assert atlas_panel._templates_page_btn.isChecked()
    assert steps["fUSI Templates"].target() is atlas_panel._template_group

    assert tour._on_close is not None
    tour._on_close()
    QApplication.processEvents()
    assert not atlas_panel._masks_group.isVisibleTo(atlas_panel)
    assert atlas_panel._atlases_page_btn.isChecked()


def test_atlas_steps_follow_data_io(widget) -> None:
    titles = [step.title for step in build_default_tour(widget, is_dark=True)._steps]
    atlas_index = titles.index("Atlas")
    assert titles[atlas_index - 1] == "Layer Saving"
    assert titles[atlas_index : atlas_index + 4] == [
        "Atlas",
        "BrainGlobe Atlases",
        "Region Masks",
        "fUSI Templates",
    ]
    assert titles[atlas_index + 4] == "Video"

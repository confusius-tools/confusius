"""Unit tests for the DataPanel widget.

`_on_load_returned` is exercised directly with a pre-built DataArray so tests don't
depend on the background `thread_worker` machinery used by the real load path.
"""

from __future__ import annotations

import pytest

from confusius._napari._data._load_panel import DataPanel


@pytest.fixture
def viewer(make_napari_viewer):
    return make_napari_viewer()


@pytest.fixture
def data_panel(viewer):
    return DataPanel(viewer)


class TestOnLoadReturned:
    """Loaded DataArrays are added to the viewer as the expected layer type."""

    def test_integer_dtype_adds_labels_layer(
        self, data_panel, viewer, sample_roi_labels
    ) -> None:
        from napari.layers import Labels

        data_panel._on_load_returned(sample_roi_labels)

        assert len(viewer.layers) == 1
        assert isinstance(viewer.layers[0], Labels)

    def test_float_dtype_adds_image_layer(
        self, data_panel, viewer, sample_3d_volume
    ) -> None:
        from napari.layers import Image

        data_panel._on_load_returned(sample_3d_volume)

        assert len(viewer.layers) == 1
        assert isinstance(viewer.layers[0], Image)

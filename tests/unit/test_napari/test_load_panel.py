"""Unit tests for the DataPanel widget.

`_on_load_returned` is exercised directly with a pre-built DataArray so tests don't
depend on the background `thread_worker` machinery used by the real load path.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import confusius._napari._data._load_panel as load_module
from confusius._napari._data._load_panel import DataPanel


@pytest.fixture
def viewer(make_napari_viewer):
    return make_napari_viewer()


@pytest.fixture
def data_panel(viewer):
    return DataPanel(viewer)


def test_load_button_requires_nonempty_path(data_panel):
    assert not data_panel._load_btn.isEnabled()
    data_panel._path_edit.setText("scan.nii.gz")
    assert data_panel._load_btn.isEnabled()
    data_panel._path_edit.setText("   ")
    assert not data_panel._load_btn.isEnabled()
    assert data_panel._file_group.title() == "Load Data"


@pytest.mark.parametrize("suffix", [".mp4", ".mov", ".avi", ".MP4"])
def test_video_paths_use_video_loader(data_panel, tmp_path, monkeypatch, suffix):
    path = tmp_path / f"clip{suffix}"
    path.touch()
    video_loader = MagicMock()
    fusi_loader = MagicMock()
    monkeypatch.setattr(data_panel.video_panel, "add_video", video_loader)
    monkeypatch.setattr(load_module, "_load_file", fusi_loader)
    data_panel._compute_check.setChecked(True)
    data_panel._path_edit.setText(str(path))
    assert not data_panel._compute_check.isEnabled()
    data_panel._load_btn.click()
    video_loader.assert_called_once_with(path)
    fusi_loader.assert_not_called()
    data_panel._path_edit.setText("scan.nii.gz")
    assert data_panel._compute_check.isEnabled()
    assert data_panel._compute_check.isChecked()


@pytest.mark.parametrize("suffix", [".nii.gz", ".scan", ".zarr"])
def test_data_paths_use_fusi_loader(data_panel, tmp_path, monkeypatch, suffix):
    path = tmp_path / f"scan{suffix}"
    if suffix == ".zarr":
        path.mkdir()
    else:
        path.touch()
    worker = MagicMock()
    fusi_loader = MagicMock(return_value=worker)
    video_loader = MagicMock()
    monkeypatch.setattr(load_module, "_load_file", fusi_loader)
    monkeypatch.setattr(data_panel.video_panel, "add_video", video_loader)
    monkeypatch.setattr(load_module, "show_error", MagicMock())
    data_panel._path_edit.setText(str(path))
    data_panel._compute_check.setChecked(True)
    data_panel._load_btn.click()
    fusi_loader.assert_called_once_with(path, lazy=True)
    worker.start.assert_called_once()
    video_loader.assert_not_called()
    assert not data_panel._load_btn.isEnabled()
    assert not data_panel._compute_check.isEnabled()
    worker.errored.connect.call_args.args[0](RuntimeError("failed"))
    assert data_panel._load_btn.isEnabled()
    assert data_panel._compute_check.isEnabled()


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
        self, data_panel, viewer, sample_voxeldata_3d
    ) -> None:
        from napari.layers import Image

        data_panel._on_load_returned(sample_voxeldata_3d)

        assert len(viewer.layers) == 1
        assert isinstance(viewer.layers[0], Image)

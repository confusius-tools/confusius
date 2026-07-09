"""Unit tests for napari sample helpers."""

from __future__ import annotations

from npe2 import PluginManifest

from unittest.mock import Mock

import numpy as np
import pytest
import xarray as xr

from confusius._napari._sample import (
    _SAMPLE_ACQ,
    _SAMPLE_RELATIVE_PATH,
    _SAMPLE_SESSION,
    _SAMPLE_SUBJECT,
    _SAMPLE_TASK,
    SampleDownloadCancelledError,
    _resolve_nunez_elizalde_2022_sample_path,
    _update_progress_dialog,
    open_nunez_elizalde_2022_sample,
)


def test_manifest_registers_nunez_elizalde_2022_sample():
    """The napari manifest exposes the sample via `sample_data`."""
    manifest = PluginManifest.from_file("src/confusius/_napari/napari.yaml")

    sample = manifest.contributions.sample_data[0]
    assert sample.key == "nunez-elizalde-2022"
    assert sample.display_name == "Nunez-Elizalde 2022"
    assert sample.command == "confusius.open_sample_nunez_elizalde_2022"


def test_update_progress_dialog_raises_when_cancelled():
    """Cancelling the popup aborts the sample download."""
    dialog = Mock()
    dialog.wasCanceled.return_value = True

    with pytest.raises(SampleDownloadCancelledError, match="cancelled"):
        _update_progress_dialog(dialog, 1, 10, "Downloading")


def test_open_sample_sets_default_gamma_and_shows_scale_bar(monkeypatch, tmp_path):
    """The sample loader uses a softer default gamma and enables the scale bar."""

    class _Dialog:
        def __init__(self, *args, **kwargs):
            pass

        def setWindowTitle(self, *args, **kwargs):
            pass

        def setLabelText(self, *args, **kwargs):
            pass

        def setCancelButtonText(self, *args, **kwargs):
            pass

        def setMinimumDuration(self, *args, **kwargs):
            pass

        def setAutoClose(self, *args, **kwargs):
            pass

        def setAutoReset(self, *args, **kwargs):
            pass

        def show(self):
            pass

        def close(self):
            pass

        def setRange(self, *args, **kwargs):
            pass

        def setValue(self, *args, **kwargs):
            pass

        def wasCanceled(self):
            return False

    class _ScaleBar:
        def __init__(self):
            self.visible = False

    class _Viewer:
        def __init__(self):
            self.scale_bar = _ScaleBar()
            self.window = Mock()
            self.window._qt_window = object()

    viewer = _Viewer()

    monkeypatch.setattr("confusius._napari._sample.QProgressDialog", _Dialog)
    monkeypatch.setattr(
        "confusius._napari._sample._resolve_nunez_elizalde_2022_sample_path",
        lambda progress_callback=None: tmp_path / "sample.nii.gz",
    )
    monkeypatch.setattr(
        "confusius._napari._sample.napari.current_viewer", lambda: viewer
    )

    da = xr.DataArray(
        np.zeros((2, 3), dtype=np.float32),
        dims=["z", "x"],
        coords={"z": np.arange(2), "x": np.arange(3)},
    )

    class _Loaded:
        def compute(self):
            return da

    monkeypatch.setattr("confusius._napari._sample.load", lambda path: _Loaded())

    [(data, kwargs, layer_type)] = open_nunez_elizalde_2022_sample()
    assert isinstance(data, np.ndarray)
    assert layer_type == "image"
    assert kwargs["gamma"] == 0.4
    assert viewer.scale_bar.visible is True


def test_resolve_nunez_elizalde_2022_sample_path(monkeypatch, tmp_path):
    """The napari sample helper fetches the expected recording subset."""
    captured: dict[str, object] = {}

    def fake_fetch_nunez_elizalde_2022(**kwargs):
        captured.update(kwargs)
        return tmp_path

    monkeypatch.setattr(
        "confusius._napari._sample.fetch_nunez_elizalde_2022",
        fake_fetch_nunez_elizalde_2022,
    )

    result = _resolve_nunez_elizalde_2022_sample_path()

    assert result == tmp_path / _SAMPLE_RELATIVE_PATH
    assert captured == {
        "datasets": "rawdata",
        "subjects": _SAMPLE_SUBJECT,
        "sessions": _SAMPLE_SESSION,
        "tasks": _SAMPLE_TASK,
        "acqs": _SAMPLE_ACQ,
        "datatypes": "fusi",
        "progress_callback": None,
    }

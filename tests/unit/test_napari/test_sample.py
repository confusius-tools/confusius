"""Unit tests for napari sample helpers."""

from __future__ import annotations

from npe2 import PluginManifest

from unittest.mock import Mock

import pytest

from confusius._napari._sample import (
    _SAMPLE_ACQ,
    _SAMPLE_RELATIVE_PATH,
    _SAMPLE_SESSION,
    _SAMPLE_SUBJECT,
    _SAMPLE_TASK,
    SampleDownloadCancelledError,
    _resolve_nunez_elizalde_2022_sample_path,
    _update_progress_dialog,
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

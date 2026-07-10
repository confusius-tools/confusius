"""Unit tests for napari sample helpers."""

from __future__ import annotations

from unittest.mock import Mock

from npe2 import PluginManifest

import numpy as np
import pytest
import xarray as xr

from confusius._napari._sample import (
    _AWAKE_MOUSE_ACQ,
    _AWAKE_MOUSE_SESSION,
    _AWAKE_MOUSE_SUBJECT,
    _AWAKE_MOUSE_TASK,
    _RAT_REGISTRATION_ACQ,
    _RAT_REGISTRATION_SESSIONS,
    _RAT_REGISTRATION_SUBJECT,
    SampleDownloadCancelledError,
    SampleFileSpec,
    SampleSpec,
    _load_sample_dataarray,
    _resolve_awake_mouse_recording,
    _resolve_rat_registration_pair,
    _update_progress_dialog,
    open_awake_mouse_recording_sample,
    open_rat_registration_pair_sample,
)


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


class _Dims:
    def __init__(self):
        self.axis_labels = ("0", "1")


class _Viewer:
    def __init__(self):
        self.scale_bar = _ScaleBar()
        self.dims = _Dims()
        self.window = Mock()
        self.window._qt_window = object()


class _ImmediateTimer:
    """Stand-in for `QTimer` that runs the callback synchronously.

    `_open_sample` defers the viewer axis-label update via `QTimer.singleShot`.
    Running it inline keeps the update inside the test instead of leaking a live
    timer that fires (and errors) during a later test's event loop.
    """

    @staticmethod
    def singleShot(msec, callback):
        callback()


def test_manifest_registers_samples():
    """The napari manifest exposes the built-in samples via `sample_data`."""
    manifest = PluginManifest.from_file("src/confusius/_napari/napari.yaml")

    samples = {sample.key: sample for sample in manifest.contributions.sample_data}
    assert set(samples) == {"awake-mouse-recording", "rat-registration-pair"}
    assert (
        samples["awake-mouse-recording"].display_name == "Awake mouse recording (2D+t)"
    )
    assert (
        samples["awake-mouse-recording"].command
        == "confusius.open_sample_awake_mouse_recording"
    )
    assert (
        samples["rat-registration-pair"].display_name
        == "Rat registration pair (2D slices)"
    )
    assert (
        samples["rat-registration-pair"].command
        == "confusius.open_sample_rat_registration_pair"
    )


def test_update_progress_dialog_raises_when_cancelled():
    """Cancelling the popup aborts the sample download."""
    dialog = Mock()
    dialog.wasCanceled.return_value = True

    with pytest.raises(SampleDownloadCancelledError, match="cancelled"):
        _update_progress_dialog(dialog, 1, 10, "Downloading")


def test_open_awake_mouse_sample_sets_default_gamma_and_shows_scale_bar(
    monkeypatch, tmp_path
):
    """The awake mouse sample uses a softer gamma and enables the scale bar."""
    viewer = _Viewer()

    monkeypatch.setattr("confusius._napari._sample.QProgressDialog", _Dialog)
    monkeypatch.setattr("confusius._napari._sample.QTimer", _ImmediateTimer)
    monkeypatch.setattr(
        "confusius._napari._sample._SAMPLE_SPECS",
        {
            "awake-mouse-recording": SampleSpec(
                title="ConfUSIus sample",
                initial_status="Checking sample cache...",
                files_resolver=lambda progress_callback=None: [
                    SampleFileSpec(path=tmp_path / "sample.nii.gz", name="awake-mouse")
                ],
                gamma=0.4,
            )
        },
    )
    monkeypatch.setattr(
        "confusius._napari._sample.napari.current_viewer", lambda: viewer
    )

    da = xr.DataArray(
        np.zeros((2, 3), dtype=np.float32),
        dims=["z", "x"],
        coords={"z": np.arange(2), "x": np.arange(3)},
    )
    monkeypatch.setattr(
        "confusius._napari._sample._load_sample_dataarray",
        lambda path, affine_key: da,
    )

    [(data, kwargs, layer_type)] = open_awake_mouse_recording_sample()
    assert isinstance(data, np.ndarray)
    assert layer_type == "image"
    assert kwargs["gamma"] == 0.4
    assert viewer.scale_bar.visible is True
    # The sample's dims are pushed onto the viewer sliders (napari does not do
    # this for the sample path on its own).
    assert viewer.dims.axis_labels == ("z", "x")


def test_open_rat_registration_pair_loads_two_layers_with_qform(monkeypatch, tmp_path):
    """The rat registration sample loads both layers in qform space."""
    monkeypatch.setattr("confusius._napari._sample.QProgressDialog", _Dialog)
    monkeypatch.setattr("confusius._napari._sample.napari.current_viewer", lambda: None)

    rat_layers = [
        SampleFileSpec(
            path=tmp_path / "fixed.nii.gz",
            name="fixed",
            layer_kwargs={"colormap": "red", "blending": "additive"},
        ),
        SampleFileSpec(
            path=tmp_path / "moving.nii.gz",
            name="moving",
            layer_kwargs={"colormap": "cyan", "blending": "additive"},
        ),
    ]
    monkeypatch.setattr(
        "confusius._napari._sample._SAMPLE_SPECS",
        {
            "rat-registration-pair": SampleSpec(
                title="ConfUSIus sample",
                initial_status="Checking sample cache...",
                files_resolver=lambda progress_callback=None: rat_layers,
                gamma=0.4,
                affine_key="physical_to_qform",
            )
        },
    )

    da = xr.DataArray(
        np.zeros((2, 3), dtype=np.float32),
        dims=["z", "x"],
        coords={"z": np.arange(2), "x": np.arange(3)},
    )
    seen_affine_keys: list[str | None] = []

    def fake_load_sample_dataarray(path, affine_key):
        seen_affine_keys.append(affine_key)
        return da

    monkeypatch.setattr(
        "confusius._napari._sample._load_sample_dataarray",
        fake_load_sample_dataarray,
    )

    layers = open_rat_registration_pair_sample()
    assert len(layers) == 2
    assert seen_affine_keys == ["physical_to_qform", "physical_to_qform"]
    assert layers[0][1]["colormap"] == "red"
    assert layers[0][1]["blending"] == "additive"
    assert layers[1][1]["colormap"] == "cyan"
    assert layers[1][1]["blending"] == "additive"


def test_load_sample_dataarray_applies_affine_when_requested(monkeypatch, tmp_path):
    """Sample loading can switch arrays into an alternate affine space."""
    transformed = xr.DataArray(
        np.ones((2, 3), dtype=np.float32),
        dims=["z", "x"],
        coords={"z": np.arange(2), "x": np.arange(3)},
    )
    applied: list[str] = []

    class _AffineAccessor:
        def apply(self, key: str):
            applied.append(key)
            return transformed, np.eye(4)

    sample = Mock()
    sample.fusi.affine = _AffineAccessor()

    class _Loaded:
        def compute(self):
            return sample

    monkeypatch.setattr("confusius._napari._sample.load", lambda path: _Loaded())

    result = _load_sample_dataarray(tmp_path / "sample.nii.gz", "physical_to_qform")
    assert result is transformed
    assert applied == ["physical_to_qform"]


def test_resolve_awake_mouse_recording(monkeypatch, tmp_path):
    """The awake mouse sample fetches the expected recording subset."""
    captured: dict[str, object] = {}

    def fake_fetch_nunez_elizalde_2022(**kwargs):
        captured.update(kwargs)
        return tmp_path

    monkeypatch.setattr(
        "confusius._napari._sample.fetch_nunez_elizalde_2022",
        fake_fetch_nunez_elizalde_2022,
    )

    [result] = _resolve_awake_mouse_recording()

    assert result.path == (
        tmp_path
        / f"sub-{_AWAKE_MOUSE_SUBJECT}"
        / f"ses-{_AWAKE_MOUSE_SESSION}"
        / "fusi"
        / (
            f"sub-{_AWAKE_MOUSE_SUBJECT}_ses-{_AWAKE_MOUSE_SESSION}"
            f"_task-{_AWAKE_MOUSE_TASK}_acq-{_AWAKE_MOUSE_ACQ}_pwd.nii.gz"
        )
    )
    assert captured == {
        "datasets": "rawdata",
        "subjects": _AWAKE_MOUSE_SUBJECT,
        "sessions": _AWAKE_MOUSE_SESSION,
        "tasks": _AWAKE_MOUSE_TASK,
        "acqs": _AWAKE_MOUSE_ACQ,
        "datatypes": "fusi",
        "progress_callback": None,
    }


def test_resolve_rat_registration_pair(monkeypatch, tmp_path):
    """The rat registration sample fetches the expected angiography subset."""
    captured: dict[str, object] = {}

    def fake_fetch_cybis_pereira_2026(**kwargs):
        captured.update(kwargs)
        return tmp_path

    monkeypatch.setattr(
        "confusius._napari._sample.fetch_cybis_pereira_2026",
        fake_fetch_cybis_pereira_2026,
    )

    result = _resolve_rat_registration_pair()

    assert [spec.path for spec in result] == [
        (
            tmp_path
            / f"sub-{_RAT_REGISTRATION_SUBJECT}"
            / f"ses-{session}"
            / "angio"
            / (
                f"sub-{_RAT_REGISTRATION_SUBJECT}_ses-{session}"
                f"_acq-{_RAT_REGISTRATION_ACQ}_rec-minframe2d_pwd.nii.gz"
            )
        )
        for session in _RAT_REGISTRATION_SESSIONS
    ]
    assert captured == {
        "datasets": "rawdata",
        "subjects": _RAT_REGISTRATION_SUBJECT,
        "sessions": list(_RAT_REGISTRATION_SESSIONS),
        "datatypes": "angio",
        "acqs": _RAT_REGISTRATION_ACQ,
    }

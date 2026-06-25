"""Unit tests for the napari-backed registration progress reporter."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import SimpleITK as sitk

from confusius._napari._registration._progress import (
    NapariProgressBridge,
    NapariVolumeProgress,
    make_napari_progress_factory,
)


@pytest.fixture
def fixed_img_2d():
    """Small 2D SimpleITK image with a bright square."""
    arr = np.zeros((16, 16), dtype=np.float32)
    arr[6:10, 6:10] = 1.0
    img = sitk.GetImageFromArray(arr.T)
    img.SetSpacing((1.0, 1.0))
    return img


@pytest.fixture
def moving_img_2d(fixed_img_2d):
    """Same image shifted by one pixel."""
    arr = sitk.GetArrayFromImage(fixed_img_2d).T
    shifted = np.roll(arr, 1, axis=0).astype(np.float32)
    img = sitk.GetImageFromArray(shifted.T)
    img.SetSpacing(fixed_img_2d.GetSpacing())
    return img


def _make_registration_method(ndim: int = 2) -> sitk.ImageRegistrationMethod:
    """Return a minimally configured ImageRegistrationMethod."""
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsCorrelation()
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsGradientDescent(
        learningRate=0.5,
        numberOfIterations=5,
        convergenceMinimumValue=1e-7,
        convergenceWindowSize=3,
    )
    reg.SetShrinkFactorsPerLevel(shrinkFactors=[1])
    reg.SetSmoothingSigmasPerLevel(smoothingSigmas=[0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()
    reg.SetInitialTransform(sitk.TranslationTransform(ndim), inPlace=True)
    return reg


class _SignalSpy:
    """Collect emitted payloads from a Qt signal."""

    def __init__(self) -> None:
        self.payloads: list[Any] = []

    def __call__(self, payload: Any) -> None:
        self.payloads.append(payload)


class TestNapariProgressBridge:
    """Signal bridge behaviour."""

    def test_iterated_signal_is_emitted(self, qtbot):
        bridge = NapariProgressBridge()
        spy = _SignalSpy()
        bridge.iterated.connect(spy)
        with qtbot.waitSignal(bridge.iterated, timeout=1000):
            bridge.iterated.emit(np.zeros((2, 2), dtype=np.float32))
        assert len(spy.payloads) == 1
        np.testing.assert_array_equal(spy.payloads[0], np.zeros((2, 2)))

    def test_finished_signal_is_emitted(self, qtbot):
        bridge = NapariProgressBridge()
        with qtbot.waitSignal(bridge.finished, timeout=1000):
            bridge.finished.emit()


class TestNapariVolumeProgress:
    """Per-iteration reporter behaviour."""

    def test_update_resamples_and_emits_array(self, qtbot, fixed_img_2d, moving_img_2d):
        reg = _make_registration_method(ndim=2)
        bridge = NapariProgressBridge()
        spy = _SignalSpy()
        bridge.iterated.connect(spy)

        reporter = NapariVolumeProgress(
            bridge,
            reg,
            fixed_img_2d,
            moving_img_2d,
            # default_value is required by `_resample_intermediate`.
            resample_kwargs={"interpolation": "linear", "default_value": 0.0},
        )

        with qtbot.waitSignal(bridge.iterated, timeout=2000):
            reporter.update()

        assert len(spy.payloads) == 1
        arr = spy.payloads[0]
        # `.T` restores numpy axis order, matching `register_volume`.
        assert arr.shape == (16, 16)
        assert arr.dtype == np.float32

    def test_close_emits_finished_signal(self, qtbot, fixed_img_2d, moving_img_2d):
        reg = _make_registration_method(ndim=2)
        bridge = NapariProgressBridge()
        reporter = NapariVolumeProgress(
            bridge,
            reg,
            fixed_img_2d,
            moving_img_2d,
            resample_kwargs={"default_value": 0.0},
        )
        with qtbot.waitSignal(bridge.finished, timeout=1000):
            reporter.close()


class TestMakeNapariProgressFactory:
    """Factory closure behaviour."""

    def test_factory_returns_napari_volume_progress(
        self, qtbot, fixed_img_2d, moving_img_2d
    ):
        bridge = NapariProgressBridge()
        factory = make_napari_progress_factory(bridge)
        reg = _make_registration_method(ndim=2)

        plotter = factory(
            reg,
            fixed_img_2d,
            moving_img_2d,
            plot_metric=True,
            plot_composite=True,
            resample_kwargs={"default_value": 0.0},
        )

        assert isinstance(plotter, NapariVolumeProgress)
        assert plotter._bridge is bridge
        assert plotter._method is reg
        assert plotter._fixed_img is fixed_img_2d
        assert plotter._moving_img is moving_img_2d


class TestRegisterVolumeWithNapariFactory:
    """End-to-end: register_volume calls the injected napari factory."""

    def test_factory_is_invoked_and_iterated_signal_fires(self, qtbot):
        import xarray as xr

        from confusius.registration.volume import register_volume

        arr = np.zeros((16, 16), dtype=np.float32)
        arr[6:10, 6:10] = 1.0
        da = xr.DataArray(
            arr,
            dims=("y", "x"),
            coords={
                "y": np.arange(16) * 0.1,
                "x": np.arange(16) * 0.1,
            },
        )

        bridge = NapariProgressBridge()
        spy = _SignalSpy()
        bridge.iterated.connect(spy)
        factory = make_napari_progress_factory(bridge)

        with qtbot.waitSignal(bridge.finished, timeout=5000):
            result, _transform, _diagnostics = register_volume(
                da,
                da,
                transform_type="translation",
                show_progress=True,
                progress_plotter=factory,
                plot_metric=True,
                plot_composite=False,
            )

        # The translator iterates at least once, so we should have received
        # at least one intermediate resampled array.
        assert len(spy.payloads) >= 1
        for payload in spy.payloads:
            assert payload.shape == da.shape
        assert result.shape == da.shape

"""Unit tests for the napari-backed registration progress reporter."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import SimpleITK as sitk

from confusius._napari._registration._progress import (
    NapariProgressBridge,
    NapariVolumeProgress,
    NapariVolumewiseProgress,
    NapariVolumewiseProgressBridge,
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

    def test_metric_updated_signal_is_emitted(self, qtbot):
        bridge = NapariProgressBridge()
        with qtbot.waitSignal(bridge.metric_updated, timeout=1000):
            bridge.metric_updated.emit(0.42)


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

    def test_update_emits_metric_value(self, qtbot, fixed_img_2d, moving_img_2d):
        """`update()` also forwards the current optimizer metric value."""
        reg = _make_registration_method(ndim=2)
        bridge = NapariProgressBridge()
        metric_spy = _SignalSpy()
        bridge.metric_updated.connect(metric_spy)

        reporter = NapariVolumeProgress(
            bridge,
            reg,
            fixed_img_2d,
            moving_img_2d,
            resample_kwargs={"default_value": 0.0},
        )

        with qtbot.waitSignal(bridge.metric_updated, timeout=2000):
            reporter.update()

        assert len(metric_spy.payloads) == 1
        assert isinstance(metric_spy.payloads[0], float)
        assert np.isfinite(metric_spy.payloads[0])

    def test_update_skips_metric_when_plot_metric_false(
        self, qtbot, fixed_img_2d, moving_img_2d
    ):
        """`plot_metric=False` suppresses the metric_updated emission."""
        reg = _make_registration_method(ndim=2)
        bridge = NapariProgressBridge()
        metric_spy = _SignalSpy()
        bridge.metric_updated.connect(metric_spy)

        reporter = NapariVolumeProgress(
            bridge,
            reg,
            fixed_img_2d,
            moving_img_2d,
            plot_metric=False,
            resample_kwargs={"default_value": 0.0},
        )
        # Iterate and confirm the metric signal never fires. We trigger the
        # iterated signal first to give the metric a chance to emit, then
        # check the spy.
        with qtbot.waitSignal(bridge.iterated, timeout=2000):
            reporter.update()
        assert metric_spy.payloads == []

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


class TestNapariVolumewiseProgressBridge:
    """Signal bridge behaviour for volumewise registration."""

    def test_frame_progress_signal_is_emitted(self, qtbot):
        bridge = NapariVolumewiseProgressBridge()
        payloads: list[tuple[int, int]] = []
        bridge.frame_progress.connect(lambda completed, total: payloads.append((completed, total)))

        with qtbot.waitSignal(bridge.frame_progress, timeout=1000):
            bridge.frame_progress.emit(1, 3)

        assert payloads == [(1, 3)]

    def test_frame_completed_signal_is_emitted(self, qtbot):
        bridge = NapariVolumewiseProgressBridge()
        payloads: list[tuple[int, np.ndarray]] = []
        bridge.frame_completed.connect(
            lambda index, array: payloads.append((index, array))
        )
        expected = np.ones((2, 2), dtype=np.float32)

        with qtbot.waitSignal(bridge.frame_completed, timeout=1000):
            bridge.frame_completed.emit(2, expected)

        assert len(payloads) == 1
        assert payloads[0][0] == 2
        np.testing.assert_array_equal(payloads[0][1], expected)

    def test_finished_signal_is_emitted(self, qtbot):
        bridge = NapariVolumewiseProgressBridge()
        with qtbot.waitSignal(bridge.finished, timeout=1000):
            bridge.finished.emit()


class TestNapariVolumewiseProgress:
    """Aggregate per-frame progress for volumewise registration."""

    def test_frame_completed_emits_progress_and_array(self, qtbot):
        import xarray as xr

        bridge = NapariVolumewiseProgressBridge()
        reporter = NapariVolumewiseProgress(bridge, n_frames=3)
        progress_payloads: list[tuple[int, int]] = []
        frame_payloads: list[tuple[int, np.ndarray]] = []
        bridge.frame_progress.connect(
            lambda completed, total: progress_payloads.append((completed, total))
        )
        bridge.frame_completed.connect(
            lambda index, array: frame_payloads.append((index, array))
        )
        frame = xr.DataArray(np.ones((2, 2), dtype=np.float32), dims=("y", "x"))
        diagnostics = object()

        with qtbot.waitSignals(
            [bridge.frame_progress, bridge.frame_completed], timeout=1000
        ):
            reporter.frame_completed(1, frame, diagnostics)  # type: ignore[arg-type]

        assert progress_payloads == [(1, 3)]
        assert len(frame_payloads) == 1
        assert frame_payloads[0][0] == 1
        np.testing.assert_array_equal(frame_payloads[0][1], frame.values)

    def test_close_emits_finished_signal(self, qtbot):
        bridge = NapariVolumewiseProgressBridge()
        reporter = NapariVolumewiseProgress(bridge, n_frames=3)

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

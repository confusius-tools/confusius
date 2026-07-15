"""Unit tests for the napari-backed registration progress reporter."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import SimpleITK as sitk

from confusius._napari._registration._progress import (
    NapariRegistrationProgressPlotter,
    NapariRegistrationProgressPlotterBridge,
    NapariRegistrationProgressReporter,
    NapariRegistrationProgressReporterBridge,
    make_napari_progress_factory,
)
from confusius.registration import RegistrationDiagnostics


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


class TestNapariRegistrationProgressPlotter:
    """Per-iteration reporter behaviour."""

    def test_update_resamples_and_emits_array(self, qtbot, fixed_img_2d, moving_img_2d):
        reg = _make_registration_method(ndim=2)
        bridge = NapariRegistrationProgressPlotterBridge()
        spy = _SignalSpy()
        bridge.iterated.connect(spy)

        reporter = NapariRegistrationProgressPlotter(
            bridge,
            reg,
            fixed_img_2d,
            moving_img_2d,
            resample_kwargs={"interpolation": "linear", "fill_value": 0.0},
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
        bridge = NapariRegistrationProgressPlotterBridge()
        metric_spy = _SignalSpy()
        bridge.metric_updated.connect(metric_spy)

        reporter = NapariRegistrationProgressPlotter(
            bridge,
            reg,
            fixed_img_2d,
            moving_img_2d,
            resample_kwargs={"fill_value": 0.0},
        )

        expected_metric = float(reg.GetMetricValue())

        with qtbot.waitSignal(bridge.metric_updated, timeout=2000):
            reporter.update()

        assert len(metric_spy.payloads) == 1
        emitted_metric = metric_spy.payloads[0]
        if np.isnan(expected_metric):
            assert np.isnan(emitted_metric)
        else:
            assert emitted_metric == pytest.approx(expected_metric)

    def test_update_skips_metric_when_plot_metric_false(
        self, qtbot, fixed_img_2d, moving_img_2d
    ):
        """`plot_metric=False` suppresses the metric_updated emission."""
        reg = _make_registration_method(ndim=2)
        bridge = NapariRegistrationProgressPlotterBridge()
        metric_spy = _SignalSpy()
        bridge.metric_updated.connect(metric_spy)

        reporter = NapariRegistrationProgressPlotter(
            bridge,
            reg,
            fixed_img_2d,
            moving_img_2d,
            plot_metric=False,
            resample_kwargs={"fill_value": 0.0},
        )
        # Iterate and confirm the metric signal never fires. We trigger the
        # iterated signal first to give the metric a chance to emit, then
        # check the spy.
        with qtbot.waitSignal(bridge.iterated, timeout=2000):
            reporter.update()
        assert metric_spy.payloads == []

    def test_close_emits_finished_signal(self, qtbot, fixed_img_2d, moving_img_2d):
        reg = _make_registration_method(ndim=2)
        bridge = NapariRegistrationProgressPlotterBridge()
        reporter = NapariRegistrationProgressPlotter(
            bridge,
            reg,
            fixed_img_2d,
            moving_img_2d,
            resample_kwargs={"fill_value": 0.0},
        )
        with qtbot.waitSignal(bridge.finished, timeout=1000):
            reporter.close()


class TestNapariRegistrationProgressReporter:
    """Aggregate per-frame progress for volumewise registration."""

    def test_frame_completed_emits_progress_and_array(self, qtbot):
        import xarray as xr

        bridge = NapariRegistrationProgressReporterBridge()
        reporter = NapariRegistrationProgressReporter(bridge, n_frames=3)
        progress_payloads: list[tuple[int, int]] = []
        frame_payloads: list[tuple[int, np.ndarray]] = []
        bridge.frame_progress.connect(
            lambda completed, total: progress_payloads.append((completed, total))
        )
        bridge.frame_completed.connect(
            lambda index, array: frame_payloads.append((index, array))
        )
        frame = xr.DataArray(np.ones((2, 2), dtype=np.float32), dims=("y", "x"))
        diagnostics = RegistrationDiagnostics(
            metric="correlation",
            metric_values=np.array([-1.0]),
            final_metric_value=-1.0,
            n_iterations=1,
            stop_condition="done",
            status="completed",
        )

        with qtbot.waitSignals(
            [bridge.frame_progress, bridge.frame_completed], timeout=1000
        ):
            reporter.frame_completed(1, frame, diagnostics)

        assert progress_payloads == [(1, 3)]
        assert len(frame_payloads) == 1
        assert frame_payloads[0][0] == 1
        np.testing.assert_array_equal(frame_payloads[0][1], frame.values)

    def test_frame_completed_accumulates_unique_progress(self, qtbot):
        import xarray as xr

        bridge = NapariRegistrationProgressReporterBridge()
        reporter = NapariRegistrationProgressReporter(bridge, n_frames=3)
        progress_payloads: list[tuple[int, int]] = []
        bridge.frame_progress.connect(
            lambda completed, total: progress_payloads.append((completed, total))
        )
        frame = xr.DataArray(np.ones((2, 2), dtype=np.float32), dims=("y", "x"))
        diagnostics = RegistrationDiagnostics(
            metric="correlation",
            metric_values=np.array([-1.0]),
            final_metric_value=-1.0,
            n_iterations=1,
            stop_condition="done",
            status="completed",
        )

        reporter.frame_completed(1, frame, diagnostics)
        reporter.frame_completed(2, frame, diagnostics)

        qtbot.waitUntil(lambda: len(progress_payloads) == 2, timeout=1000)
        assert progress_payloads == [(1, 3), (2, 3)]

    def test_close_emits_finished_signal(self, qtbot):
        bridge = NapariRegistrationProgressReporterBridge()
        reporter = NapariRegistrationProgressReporter(bridge, n_frames=3)

        with qtbot.waitSignal(bridge.finished, timeout=1000):
            reporter.close()


class TestMakeNapariProgressFactory:
    """Factory closure behaviour."""

    def test_factory_returns_napari_volume_progress(
        self, qtbot, fixed_img_2d, moving_img_2d
    ):
        bridge = NapariRegistrationProgressPlotterBridge()
        factory = make_napari_progress_factory(bridge)
        reg = _make_registration_method(ndim=2)

        plotter = factory(
            reg,
            fixed_img_2d,
            moving_img_2d,
            plot_metric=True,
            plot_composite=True,
            resample_kwargs={"fill_value": 0.0},
        )

        assert isinstance(plotter, NapariRegistrationProgressPlotter)

        with qtbot.waitSignals(
            [bridge.metric_updated, bridge.iterated, bridge.finished], timeout=2000
        ):
            plotter.update()
            plotter.close()


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

        bridge = NapariRegistrationProgressPlotterBridge()
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

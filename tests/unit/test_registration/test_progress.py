"""Unit tests for MatplotlibVolumeRegistrationProgressPlotter."""

import builtins
import sys
import types
from typing import Any, cast

import matplotlib
import numpy as np
import pytest
import SimpleITK as sitk

matplotlib.use("Agg")

from confusius.registration.diagnostics import RegistrationDiagnostics
from confusius.registration.progress import (
    MatplotlibVolumeRegistrationProgressPlotter,
)
from confusius.registration.volumewise_progress import (
    MatplotlibVolumewiseRegistrationProgressPlotter,
)


@pytest.fixture(autouse=True)
def close_matplotlib_figures():
    """Close figures opened by progress-plotter tests."""
    import matplotlib.pyplot as plt

    close = plt.close
    yield
    close("all")


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------


def _make_diagnostics(
    final_metric_value: float, n_iterations: int
) -> RegistrationDiagnostics:
    """Return minimal registration diagnostics for progress-plotter tests."""
    return RegistrationDiagnostics(
        metric="correlation",
        metric_values=np.asarray([final_metric_value]),
        final_metric_value=final_metric_value,
        n_iterations=n_iterations,
        stop_condition="done",
        status="completed",
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


@pytest.fixture
def fixed_img_3d():
    """Small 3D SimpleITK image with a bright cube."""
    arr = np.zeros((8, 8, 8), dtype=np.float32)
    arr[3:5, 3:5, 3:5] = 1.0
    img = sitk.GetImageFromArray(arr.T)
    img.SetSpacing((1.0, 1.0, 1.0))
    return img


@pytest.fixture
def moving_img_3d(fixed_img_3d):
    """Same 3D image shifted by one voxel."""
    arr = sitk.GetArrayFromImage(fixed_img_3d).T
    shifted = np.roll(arr, 1, axis=0).astype(np.float32)
    img = sitk.GetImageFromArray(shifted.T)
    img.SetSpacing(fixed_img_3d.GetSpacing())
    return img


def _make_registration_method():
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
    reg.SetInitialTransform(sitk.TranslationTransform(2), inPlace=True)
    return reg


# ---------------------------------------------------------------------------
# MatplotlibVolumeRegistrationProgressPlotter
# ---------------------------------------------------------------------------


class TestMatplotlibVolumeRegistrationProgressPlotterInstantiation:
    """Smoke tests for plotter construction."""

    def test_importerror_from_ipython_detection_falls_back_to_script_mode(
        self, fixed_img_2d, moving_img_2d, monkeypatch
    ):
        """Missing IPython support falls back cleanly to non-notebook mode."""
        reg = _make_registration_method()
        original_import = builtins.__import__

        def _guarded_import(name, *args, **kwargs):
            if name == "IPython.core.getipython":
                raise ImportError("no ipython")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _guarded_import)

        plotter = MatplotlibVolumeRegistrationProgressPlotter(
            reg,
            fixed_img_2d,
            moving_img_2d,
            plot_metric=True,
            plot_composite=False,
        )

        assert plotter._notebook is False
        plotter.figure.clf()

    def test_metric_only(self, fixed_img_2d, moving_img_2d):
        """Plotter with only metric panel is created without error."""
        reg = _make_registration_method()
        plotter = MatplotlibVolumeRegistrationProgressPlotter(
            reg,
            fixed_img_2d,
            moving_img_2d,
            plot_metric=True,
            plot_composite=False,
        )
        plotter.figure.clf()

    def test_composite_only(self, fixed_img_2d, moving_img_2d):
        """Plotter with only composite panel is created without error."""
        reg = _make_registration_method()
        plotter = MatplotlibVolumeRegistrationProgressPlotter(
            reg,
            fixed_img_2d,
            moving_img_2d,
            plot_metric=False,
            plot_composite=True,
        )
        plotter.figure.clf()

    def test_both_panels(self, fixed_img_2d, moving_img_2d):
        """Plotter with both panels is created without error."""
        reg = _make_registration_method()
        plotter = MatplotlibVolumeRegistrationProgressPlotter(
            reg,
            fixed_img_2d,
            moving_img_2d,
            plot_metric=True,
            plot_composite=True,
        )
        plotter.figure.clf()


class TestMatplotlibVolumeRegistrationProgressPlotterUpdate:
    """Tests for metric_values population and composite rendering."""

    def test_notebook_mode_uses_display_and_closes_figure(
        self, fixed_img_2d, moving_img_2d, monkeypatch
    ):
        """Notebook mode renders via IPython display and closes on finish."""
        import matplotlib.pyplot as plt

        reg = _make_registration_method()
        display_calls: list[tuple[object, bool]] = []
        closed_figures: list[object] = []

        fake_getipython = cast(Any, types.ModuleType("IPython.core.getipython"))

        class ZMQInteractiveShell:
            pass

        fake_getipython.get_ipython = lambda: ZMQInteractiveShell()
        fake_display = cast(Any, types.ModuleType("IPython.display"))
        fake_display.display = lambda fig, clear=False: display_calls.append(
            (fig, clear)
        )
        monkeypatch.setitem(sys.modules, "IPython.core.getipython", fake_getipython)
        monkeypatch.setitem(sys.modules, "IPython.display", fake_display)
        monkeypatch.setattr(plt, "close", lambda fig: closed_figures.append(fig))

        plotter = MatplotlibVolumeRegistrationProgressPlotter(
            reg,
            fixed_img_2d,
            moving_img_2d,
            plot_metric=False,
            plot_composite=True,
        )

        plotter.update()
        plotter.close()

        assert display_calls
        assert display_calls[-1][0] is plotter.figure
        assert display_calls[-1][1] is True
        assert closed_figures == [plotter.figure]

    def test_metric_values_populated_after_registration(
        self, fixed_img_2d, moving_img_2d
    ):
        """metric_values contains one entry per iteration after registration."""
        reg = _make_registration_method()
        plotter = MatplotlibVolumeRegistrationProgressPlotter(
            reg,
            fixed_img_2d,
            moving_img_2d,
            plot_metric=True,
            plot_composite=False,
        )
        reg.AddCommand(sitk.sitkIterationEvent, plotter.update)
        reg.AddCommand(sitk.sitkEndEvent, plotter.close)
        reg.Execute(
            sitk.Cast(fixed_img_2d, sitk.sitkFloat32),
            sitk.Cast(moving_img_2d, sitk.sitkFloat32),
        )
        assert len(plotter.metric_values) > 0
        plotter.figure.clf()

    def test_metric_values_are_floats(self, fixed_img_2d, moving_img_2d):
        """All recorded metric values are finite floats."""
        reg = _make_registration_method()
        plotter = MatplotlibVolumeRegistrationProgressPlotter(
            reg,
            fixed_img_2d,
            moving_img_2d,
            plot_metric=True,
            plot_composite=False,
        )
        reg.AddCommand(sitk.sitkIterationEvent, plotter.update)
        reg.AddCommand(sitk.sitkEndEvent, plotter.close)
        reg.Execute(
            sitk.Cast(fixed_img_2d, sitk.sitkFloat32),
            sitk.Cast(moving_img_2d, sitk.sitkFloat32),
        )
        assert all(np.isfinite(v) for v in plotter.metric_values)
        plotter.figure.clf()

    def test_composite_panel_rendered_after_registration(
        self, fixed_img_2d, moving_img_2d
    ):
        """Composite panel renders without error after at least one iteration."""
        reg = _make_registration_method()
        plotter = MatplotlibVolumeRegistrationProgressPlotter(
            reg,
            fixed_img_2d,
            moving_img_2d,
            plot_metric=False,
            plot_composite=True,
        )
        reg.AddCommand(sitk.sitkIterationEvent, plotter.update)
        reg.AddCommand(sitk.sitkEndEvent, plotter.close)
        reg.Execute(
            sitk.Cast(fixed_img_2d, sitk.sitkFloat32),
            sitk.Cast(moving_img_2d, sitk.sitkFloat32),
        )
        plotter.figure.clf()

    def test_3d_composite_panel_rendered(self, fixed_img_3d, moving_img_3d):
        """Composite mosaic renders for 3D images without error."""
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
        reg.SetInitialTransform(sitk.TranslationTransform(3), inPlace=True)

        plotter = MatplotlibVolumeRegistrationProgressPlotter(
            reg,
            fixed_img_3d,
            moving_img_3d,
            plot_metric=False,
            plot_composite=True,
        )
        reg.AddCommand(sitk.sitkIterationEvent, plotter.update)
        reg.AddCommand(sitk.sitkEndEvent, plotter.close)
        reg.Execute(
            sitk.Cast(fixed_img_3d, sitk.sitkFloat32),
            sitk.Cast(moving_img_3d, sitk.sitkFloat32),
        )
        plotter.figure.clf()


class TestMatplotlibVolumeRegistrationProgressPlotterResampleKwargs:
    """Tests for intermediate-resample settings."""

    def test_none_interpolation_falls_back_to_linear(self, fixed_img_2d, moving_img_2d):
        """A `None` interpolation override falls back to linear at render time."""
        reg = _make_registration_method()
        plotter = MatplotlibVolumeRegistrationProgressPlotter(
            reg,
            fixed_img_2d,
            moving_img_2d,
            plot_metric=False,
            plot_composite=True,
            resample_kwargs={"interpolation": None},
        )

        plotter.update()

        assert plotter._composite_im is not None
        plotter.figure.clf()

    def test_invalid_interpolation_raises_on_update(self, fixed_img_2d, moving_img_2d):
        """Unknown interpolation names raise a clear ValueError during rendering."""
        reg = _make_registration_method()
        plotter = MatplotlibVolumeRegistrationProgressPlotter(
            reg,
            fixed_img_2d,
            moving_img_2d,
            plot_metric=False,
            plot_composite=True,
            resample_kwargs={"interpolation": "bogus"},
        )

        with pytest.raises(ValueError, match="Invalid `interpolation`"):
            plotter.update()

        plotter.figure.clf()

    def test_default_fill_value_is_moving_min(self, fixed_img_2d, moving_img_2d):
        """When resample_kwargs omits fill_value, it defaults to moving_img.min()."""
        reg = _make_registration_method()
        plotter = MatplotlibVolumeRegistrationProgressPlotter(
            reg, fixed_img_2d, moving_img_2d, plot_metric=False, plot_composite=True
        )
        expected = float(sitk.GetArrayFromImage(moving_img_2d).min())
        assert plotter._fill_value == pytest.approx(expected)
        plotter.figure.clf()

    def test_explicit_fill_value_is_respected(self, fixed_img_2d, moving_img_2d):
        """Explicit fill_value in resample_kwargs overrides the auto-default."""
        reg = _make_registration_method()
        plotter = MatplotlibVolumeRegistrationProgressPlotter(
            reg,
            fixed_img_2d,
            moving_img_2d,
            plot_metric=False,
            plot_composite=True,
            resample_kwargs={"fill_value": -60.0},
        )
        assert plotter._fill_value == pytest.approx(-60.0)
        plotter.figure.clf()

    def test_explicit_interpolation_is_stored(self, fixed_img_2d, moving_img_2d):
        """interpolation key in resample_kwargs is stored and later used."""
        reg = _make_registration_method()
        plotter = MatplotlibVolumeRegistrationProgressPlotter(
            reg,
            fixed_img_2d,
            moving_img_2d,
            plot_metric=False,
            plot_composite=True,
            resample_kwargs={"interpolation": "nearest"},
        )
        assert plotter._interpolation == "nearest"
        plotter.figure.clf()


class TestRegisterVolumeShowProgress:
    """Integration: show_progress=True wires correctly through register_volume."""

    def test_show_progress_true_does_not_raise(self):
        """register_volume with show_progress=True completes without error."""
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
        result, _, _ = register_volume(
            da,
            da,
            transform_type="translation",
            show_progress=True,
            plot_metric=True,
            plot_composite=False,
        )
        assert result.shape == da.shape


class TestMatplotlibVolumewiseRegistrationProgressPlotter:
    """Unit tests for MatplotlibVolumewiseRegistrationProgressPlotter."""

    def test_updates_completed_frames_by_index(self):
        """Out-of-order frame completion fills the matching slots."""
        import xarray as xr

        frame = xr.DataArray(np.zeros((2, 2)), dims=("y", "x"))
        with pytest.warns(UserWarning, match="non-interactive"):
            plotter = MatplotlibVolumewiseRegistrationProgressPlotter(
                3,
                reference=frame,
                time_coords=np.asarray([0.0, 0.5, 1.0]),
                time_units="s",
                redraw_every=1,
            )

        plotter.frame_completed(2, frame, np.eye(3), _make_diagnostics(-0.2, 4))
        plotter.frame_completed(0, frame, np.eye(3), _make_diagnostics(-1.0, 2))

        np.testing.assert_allclose(plotter.metric_values, [-1.0, np.nan, -0.2])
        np.testing.assert_allclose(plotter.n_iterations, [2, np.nan, 4])
        assert plotter._optimizer_ax.get_xlabel() == "Time (s)"
        plotter.close()
        plotter.figure.clf()

    def test_default_redraw_every_skips_first_render(self):
        """Default redraw cadence buffers early frames without drawing."""
        import xarray as xr

        frame = xr.DataArray(np.zeros((2, 2)), dims=("y", "x"))
        with pytest.warns(UserWarning, match="non-interactive"):
            plotter = MatplotlibVolumewiseRegistrationProgressPlotter(
                3, reference=frame
            )

        plotter.frame_completed(0, frame, np.eye(3), _make_diagnostics(-1.0, 2))

        np.testing.assert_allclose(plotter.metric_values, [-1.0, np.nan, np.nan])
        assert len(plotter._metric_line.get_xdata()) == 0
        plotter.figure.clf()

    def test_updates_3d_motion_and_fd(self):
        """3D affines populate rotation, translation, and FD lines."""
        import xarray as xr

        reference = xr.DataArray(
            np.zeros((2, 2, 2)),
            dims=("z", "y", "x"),
            coords={
                "z": np.arange(2),
                "y": np.arange(2),
                "x": np.arange(2),
            },
        )
        with pytest.warns(UserWarning, match="non-interactive"):
            plotter = MatplotlibVolumewiseRegistrationProgressPlotter(
                2, reference=reference, redraw_every=1
            )

        affine0 = np.eye(4)
        affine1 = np.eye(4)
        affine1[2, 3] = 1.0
        plotter.frame_completed(0, reference, affine0, _make_diagnostics(-1.0, 2))
        plotter.frame_completed(1, reference, affine1, _make_diagnostics(-0.5, 3))

        assert "rot_x" in plotter._motion_values
        assert "trans_x" in plotter._motion_values
        np.testing.assert_allclose(plotter._fd_values["mean_fd"][0], 1.0)
        plotter.figure.clf()

    def test_none_affine_update_is_ignored(self):
        """Missing affine slots are ignored by the motion-value updater."""
        import xarray as xr

        frame = xr.DataArray(np.zeros((2, 2)), dims=("y", "x"))
        with pytest.warns(UserWarning, match="non-interactive"):
            plotter = MatplotlibVolumewiseRegistrationProgressPlotter(
                1, reference=frame
            )

        plotter._update_motion_values(0)

        assert plotter._motion_values == {}
        plotter.figure.clf()

    def test_missing_ipython_uses_script_mode(self, monkeypatch):
        """ImportError while detecting IPython falls back to script rendering."""
        import xarray as xr

        original_import = builtins.__import__

        def _guarded_import(name, *args, **kwargs):
            if name == "IPython.core.getipython":
                raise ImportError("no IPython")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _guarded_import)
        frame = xr.DataArray(np.zeros((2, 2)), dims=("y", "x"))
        with pytest.warns(UserWarning, match="non-interactive"):
            plotter = MatplotlibVolumewiseRegistrationProgressPlotter(
                1, reference=frame
            )

        assert not plotter._notebook
        plotter.figure.clf()

    def test_notebook_mode_displays_and_closes(self, monkeypatch):
        """Notebook mode renders through IPython display and closes on finish."""
        import matplotlib.pyplot as plt
        import xarray as xr

        display_calls: list[tuple[object, bool]] = []
        closed_figures: list[object] = []
        fake_getipython = cast(Any, types.ModuleType("IPython.core.getipython"))

        class ZMQInteractiveShell:
            pass

        fake_getipython.get_ipython = lambda: ZMQInteractiveShell()
        fake_display = cast(Any, types.ModuleType("IPython.display"))
        fake_display.display = lambda fig, clear=False: display_calls.append(
            (fig, clear)
        )
        monkeypatch.setitem(sys.modules, "IPython.core.getipython", fake_getipython)
        monkeypatch.setitem(sys.modules, "IPython.display", fake_display)
        monkeypatch.setattr(plt, "close", lambda fig: closed_figures.append(fig))

        frame = xr.DataArray(np.zeros((2, 2)), dims=("y", "x"))
        plotter = MatplotlibVolumewiseRegistrationProgressPlotter(
            1, reference=frame, redraw_every=1
        )
        plotter.frame_completed(0, frame, np.eye(3), _make_diagnostics(-1.0, 2))
        plotter.close()

        assert display_calls
        assert closed_figures == [plotter.figure]

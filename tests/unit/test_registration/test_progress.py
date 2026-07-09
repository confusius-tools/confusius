"""Unit tests for MatplotlibRegistrationProgressPlotter."""

import builtins
import sys
import types

import matplotlib
import numpy as np
import pytest
import SimpleITK as sitk

matplotlib.use("Agg")

from confusius.registration.progress import (  # noqa: E402
    MatplotlibRegistrationProgressPlotter,
)

# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------


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
# MatplotlibRegistrationProgressPlotter
# ---------------------------------------------------------------------------


class TestMatplotlibRegistrationProgressPlotterInstantiation:
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

        plotter = MatplotlibRegistrationProgressPlotter(
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
        plotter = MatplotlibRegistrationProgressPlotter(
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
        plotter = MatplotlibRegistrationProgressPlotter(
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
        plotter = MatplotlibRegistrationProgressPlotter(
            reg,
            fixed_img_2d,
            moving_img_2d,
            plot_metric=True,
            plot_composite=True,
        )
        plotter.figure.clf()


class TestMatplotlibRegistrationProgressPlotterUpdate:
    """Tests for metric_values population and composite rendering."""

    def test_notebook_mode_uses_display_and_closes_figure(
        self, fixed_img_2d, moving_img_2d, monkeypatch
    ):
        """Notebook mode renders via IPython display and closes on finish."""
        import matplotlib.pyplot as plt

        reg = _make_registration_method()
        display_calls: list[tuple[object, bool]] = []
        closed_figures: list[object] = []

        fake_getipython = types.ModuleType("IPython.core.getipython")

        class ZMQInteractiveShell:
            pass

        setattr(fake_getipython, "get_ipython", lambda: ZMQInteractiveShell())
        fake_display = types.ModuleType("IPython.display")
        setattr(
            fake_display,
            "display",
            lambda fig, clear=False: display_calls.append((fig, clear)),
        )
        monkeypatch.setitem(sys.modules, "IPython.core.getipython", fake_getipython)
        monkeypatch.setitem(sys.modules, "IPython.display", fake_display)
        monkeypatch.setattr(plt, "close", lambda fig: closed_figures.append(fig))

        plotter = MatplotlibRegistrationProgressPlotter(
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
        plotter = MatplotlibRegistrationProgressPlotter(
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
        plotter = MatplotlibRegistrationProgressPlotter(
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
        plotter = MatplotlibRegistrationProgressPlotter(
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

        plotter = MatplotlibRegistrationProgressPlotter(
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


class TestMatplotlibRegistrationProgressPlotterResampleKwargs:
    """Tests for intermediate-resample settings."""

    def test_none_interpolation_falls_back_to_linear(
        self, fixed_img_2d, moving_img_2d
    ):
        """A `None` interpolation override falls back to linear at render time."""
        reg = _make_registration_method()
        plotter = MatplotlibRegistrationProgressPlotter(
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

    def test_invalid_interpolation_raises_on_update(
        self, fixed_img_2d, moving_img_2d
    ):
        """Unknown interpolation names raise a clear ValueError during rendering."""
        reg = _make_registration_method()
        plotter = MatplotlibRegistrationProgressPlotter(
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
        plotter = MatplotlibRegistrationProgressPlotter(
            reg, fixed_img_2d, moving_img_2d, plot_metric=False, plot_composite=True
        )
        expected = float(sitk.GetArrayFromImage(moving_img_2d).min())
        assert plotter._fill_value == pytest.approx(expected)
        plotter.figure.clf()

    def test_explicit_fill_value_is_respected(self, fixed_img_2d, moving_img_2d):
        """Explicit fill_value in resample_kwargs overrides the auto-default."""
        reg = _make_registration_method()
        plotter = MatplotlibRegistrationProgressPlotter(
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
        plotter = MatplotlibRegistrationProgressPlotter(
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

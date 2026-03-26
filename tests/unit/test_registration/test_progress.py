"""Unit tests for RegistrationProgressPlotter."""

import matplotlib
import numpy as np
import pytest
import SimpleITK as sitk

matplotlib.use("Agg")

from confusius.registration._progress import (  # noqa: E402
    RegistrationProgressPlotter,
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
# RegistrationProgressPlotter
# ---------------------------------------------------------------------------


class TestRegistrationProgressPlotterInstantiation:
    """Smoke tests for plotter construction."""

    def test_metric_only(self, fixed_img_2d, moving_img_2d):
        """Plotter with only metric panel is created without error."""
        reg = _make_registration_method()
        plotter = RegistrationProgressPlotter(
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
        plotter = RegistrationProgressPlotter(
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
        plotter = RegistrationProgressPlotter(
            reg,
            fixed_img_2d,
            moving_img_2d,
            plot_metric=True,
            plot_composite=True,
        )
        plotter.figure.clf()


class TestRegistrationProgressPlotterUpdate:
    """Tests for metric_values population and composite rendering."""

    def test_metric_values_populated_after_registration(
        self, fixed_img_2d, moving_img_2d
    ):
        """metric_values contains one entry per iteration after registration."""
        reg = _make_registration_method()
        plotter = RegistrationProgressPlotter(
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
        plotter = RegistrationProgressPlotter(
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
        plotter = RegistrationProgressPlotter(
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

        plotter = RegistrationProgressPlotter(
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
        result, _ = register_volume(
            da,
            da,
            transform_type="translation",
            show_progress=True,
            plot_metric=True,
            plot_composite=False,
        )
        assert result.shape == da.shape

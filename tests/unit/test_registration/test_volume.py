"""Unit tests for single-volume registration."""

import signal
from collections.abc import Callable
from threading import Event
from types import FrameType
from typing import TypeGuard

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal

from confusius._utils.coordinates import get_grid_kwargs_from_dataarray
from confusius.registration.bspline import (
    invert_displacement_field,
    sample_displacement_field,
    sample_displacement_field_like,
    sitk_bspline_to_dataarray,
)
from confusius.registration.diagnostics import RegistrationDiagnostics
from confusius.registration.resampling import resample_like, resample_volume
from confusius.registration.volume import register_volume


def _is_signal_handler(
    handler: object,
) -> TypeGuard[Callable[[int, FrameType | None], object]]:
    """Return whether `handler` is a callable Python SIGINT handler."""
    return callable(handler)


class TestRegisterVolumeSigint:
    """Ctrl+C handling exposed through the public `register_volume` API."""

    def test_first_ctrl_c_returns_aborted_result_and_restores_handler(
        self, sample_singleton_z_dataarray_spatial, monkeypatch
    ):
        """First Ctrl+C sets the cooperative abort event and restores SIGINT afterwards."""
        import SimpleITK as sitk

        previous_handler = signal.getsignal(signal.SIGINT)

        def fake_execute(self, fixed, moving):
            del self, fixed, moving
            handler = signal.getsignal(signal.SIGINT)
            assert _is_signal_handler(handler)
            handler(signal.SIGINT, None)
            return sitk.TranslationTransform(3)

        monkeypatch.setattr(sitk.ImageRegistrationMethod, "Execute", fake_execute)

        _result, _transform, diagnostics = register_volume(
            sample_singleton_z_dataarray_spatial,
            sample_singleton_z_dataarray_spatial,
            transform_type="translation",
        )

        assert diagnostics.status == "aborted"
        assert signal.getsignal(signal.SIGINT) is previous_handler

    def test_second_ctrl_c_raises_keyboardinterrupt(
        self, sample_singleton_z_dataarray_spatial, monkeypatch
    ):
        """Second Ctrl+C falls back to the previous default SIGINT handler."""
        import SimpleITK as sitk

        previous_handler = signal.getsignal(signal.SIGINT)

        def fake_execute(self, fixed, moving):
            del self, fixed, moving
            handler = signal.getsignal(signal.SIGINT)
            assert _is_signal_handler(handler)
            handler(signal.SIGINT, None)
            handler(signal.SIGINT, None)
            return sitk.TranslationTransform(3)

        monkeypatch.setattr(sitk.ImageRegistrationMethod, "Execute", fake_execute)

        with pytest.raises(KeyboardInterrupt):
            register_volume(
                sample_singleton_z_dataarray_spatial,
                sample_singleton_z_dataarray_spatial,
                transform_type="translation",
            )

        assert signal.getsignal(signal.SIGINT) is previous_handler

    def test_second_ctrl_c_ignores_when_previous_handler_ignores(
        self, sample_singleton_z_dataarray_spatial, monkeypatch
    ):
        """Second Ctrl+C is ignored when the previous SIGINT handler ignored it."""
        import SimpleITK as sitk

        previous_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        def fake_execute(self, fixed, moving):
            del self, fixed, moving
            handler = signal.getsignal(signal.SIGINT)
            assert _is_signal_handler(handler)
            handler(signal.SIGINT, None)
            handler(signal.SIGINT, None)
            return sitk.TranslationTransform(3)

        monkeypatch.setattr(sitk.ImageRegistrationMethod, "Execute", fake_execute)

        try:
            _result, _transform, diagnostics = register_volume(
                sample_singleton_z_dataarray_spatial,
                sample_singleton_z_dataarray_spatial,
                transform_type="translation",
            )
        finally:
            signal.signal(signal.SIGINT, previous_handler)

        assert diagnostics.status == "aborted"

    def test_second_ctrl_c_calls_previous_custom_handler(
        self, sample_singleton_z_dataarray_spatial, monkeypatch
    ):
        """Second Ctrl+C delegates to a previous custom handler when one is installed."""
        import SimpleITK as sitk

        previous_handler = signal.getsignal(signal.SIGINT)
        calls: list[tuple[int, object]] = []

        def custom_handler(signum: int, frame: object) -> None:
            calls.append((signum, frame))

        signal.signal(signal.SIGINT, custom_handler)

        def fake_execute(self, fixed, moving):
            del self, fixed, moving
            handler = signal.getsignal(signal.SIGINT)
            assert _is_signal_handler(handler)
            handler(signal.SIGINT, None)
            handler(signal.SIGINT, None)
            return sitk.TranslationTransform(3)

        monkeypatch.setattr(sitk.ImageRegistrationMethod, "Execute", fake_execute)

        try:
            _result, _transform, diagnostics = register_volume(
                sample_singleton_z_dataarray_spatial,
                sample_singleton_z_dataarray_spatial,
                transform_type="translation",
            )
        finally:
            signal.signal(signal.SIGINT, previous_handler)

        assert diagnostics.status == "aborted"
        assert len(calls) == 1
        assert calls[0][0] == signal.SIGINT


class TestRegisterVolumeValidation:
    """Input validation for register_volume."""

    def test_time_dimension_raises(self, sample_singleton_z_dataarray):
        """DataArray with a time dimension raises ValueError."""
        with pytest.raises(ValueError, match="spatial-only"):
            register_volume(sample_singleton_z_dataarray, sample_singleton_z_dataarray)

    def test_nan_in_moving_raises(self, sample_singleton_z_dataarray_spatial):
        """moving with NaN values raises ValueError."""
        moving = sample_singleton_z_dataarray_spatial.copy()
        moving.values[0, 0] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            register_volume(
                moving,
                sample_singleton_z_dataarray_spatial,
                transform_type="translation",
            )

    def test_nan_in_fixed_raises(self, sample_singleton_z_dataarray_spatial):
        """fixed with NaN values raises ValueError."""
        fixed = sample_singleton_z_dataarray_spatial.copy()
        fixed.values[0, 0] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            register_volume(
                sample_singleton_z_dataarray_spatial,
                fixed,
                transform_type="translation",
            )

    def test_wrong_ndim_1d_raises(self):
        """1D input raises ValueError."""
        da = xr.DataArray(np.zeros(10), dims=("x",))
        with pytest.raises(ValueError, match="must contain all spatial dimensions"):
            register_volume(da, da)

    def test_wrong_ndim_4d_raises(self):
        """4D input raises ValueError."""
        da = xr.DataArray(np.zeros((4, 4, 4, 4)), dims=("a", "b", "c", "d"))
        with pytest.raises(ValueError, match="Unexpected dimensions"):
            register_volume(da, da)

    def test_invalid_initialization_raises(self, sample_singleton_z_dataarray_spatial):
        """Unknown initialization mode raises ValueError."""
        with pytest.raises(ValueError, match="Invalid initialization"):
            register_volume(
                sample_singleton_z_dataarray_spatial,
                sample_singleton_z_dataarray_spatial,
                initialization="moments",  # ty: ignore[invalid-argument-type]
            )

    def test_non_array_initialization_raises_value_error(
        self, sample_singleton_z_dataarray_spatial
    ):
        """A non-ndarray sequence raises ValueError, not an unhashable TypeError."""
        with pytest.raises(ValueError, match="Invalid initialization"):
            register_volume(
                sample_singleton_z_dataarray_spatial,
                sample_singleton_z_dataarray_spatial,
                initialization=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # ty: ignore[invalid-argument-type]
            )

    def test_shape_mismatch_no_error(
        self, sample_2d_image, sample_singleton_z_dataarray_spatial
    ):
        """Different shapes do not raise an error."""
        moving = sample_singleton_z_dataarray_spatial.isel(y=slice(16), x=slice(16))
        result, _, _ = register_volume(
            moving,
            sample_singleton_z_dataarray_spatial,
            transform_type="translation",
            resample=False,
        )
        assert result.shape == moving.shape

    def test_abort_event_returns_partial_result(
        self, sample_singleton_z_dataarray_spatial
    ):
        """A pre-set abort event returns an aborted diagnostics record."""
        abort_event = Event()
        abort_event.set()

        result, _transform, diagnostics = register_volume(
            sample_singleton_z_dataarray_spatial,
            sample_singleton_z_dataarray_spatial,
            transform_type="translation",
            abort_event=abort_event,
        )

        assert result.shape == sample_singleton_z_dataarray_spatial.shape
        assert diagnostics.status == "aborted"
        assert diagnostics.n_iterations == 0

    def test_unknown_runtime_error_is_passed_through(
        self, sample_singleton_z_dataarray_spatial, monkeypatch
    ):
        """Unknown SimpleITK runtime errors are re-raised unchanged."""
        import SimpleITK as sitk

        error = RuntimeError("boom")

        def fake_execute(self, fixed, moving):
            del self, fixed, moving
            raise error

        monkeypatch.setattr(sitk.ImageRegistrationMethod, "Execute", fake_execute)

        with pytest.raises(RuntimeError) as excinfo:
            register_volume(
                sample_singleton_z_dataarray_spatial,
                sample_singleton_z_dataarray_spatial,
                transform_type="translation",
            )

        assert excinfo.value is error

    def test_bspline_scale_error_raises_clearer_message(
        self, sample_singleton_z_dataarray_spatial, monkeypatch
    ):
        """Known SimpleITK scale failures are rewritten to actionable errors."""
        import SimpleITK as sitk

        def fake_execute(self, fixed, moving):
            del self, fixed, moving
            raise RuntimeError(
                "Exception thrown in SimpleITK ImageRegistrationMethod_Execute: "
                "ITK ERROR: GradientDescentOptimizerv4Template: "
                "m_Scales values must be > epsilon.[1e-20, 1e-12]"
            )

        monkeypatch.setattr(sitk.ImageRegistrationMethod, "Execute", fake_execute)

        with pytest.raises(
            RuntimeError, match="could not compute valid optimizer scales"
        ):
            register_volume(
                sample_singleton_z_dataarray_spatial,
                sample_singleton_z_dataarray_spatial,
                transform_type="bspline",
                learning_rate=1.0,
            )

    def test_bspline_scale_error_with_auto_learning_rate_suggests_fixed_rate(
        self, sample_singleton_z_dataarray_spatial, monkeypatch
    ):
        """Auto-learning-rate scale failures suggest retrying with a fixed rate."""
        import SimpleITK as sitk

        def fake_execute(self, fixed, moving):
            del self, fixed, moving
            raise RuntimeError(
                "Exception thrown in SimpleITK ImageRegistrationMethod_Execute: "
                "ITK ERROR: GradientDescentOptimizerv4Template: "
                "m_Scales values must be > epsilon.[1e-20, 1e-12]"
            )

        monkeypatch.setattr(sitk.ImageRegistrationMethod, "Execute", fake_execute)

        with pytest.raises(
            RuntimeError,
            match="Retry with a fixed `learning_rate` such as `0.1` or `0.01`",
        ):
            register_volume(
                sample_singleton_z_dataarray_spatial,
                sample_singleton_z_dataarray_spatial,
                transform_type="bspline",
                learning_rate="auto",
            )

    def test_mismatched_spatial_units_raise(self, sample_singleton_z_dataarray_spatial):
        """moving and fixed must agree on spatial coordinate units when declared."""
        moving = sample_singleton_z_dataarray_spatial.copy()
        fixed = sample_singleton_z_dataarray_spatial.copy()
        moving.coords["y"].attrs["units"] = "mm"
        moving.coords["x"].attrs["units"] = "mm"
        fixed.coords["y"].attrs["units"] = "um"
        fixed.coords["x"].attrs["units"] = "um"

        with pytest.raises(ValueError, match="units"):
            register_volume(moving, fixed, transform_type="translation")


class TestRegisterVolumeOutput:
    """Output properties for register_volume."""

    def test_without_coords_raises(self, sample_2d_image):
        """DataArray without coordinates is rejected."""
        da = xr.DataArray(sample_2d_image[np.newaxis], dims=("z", "y", "x"))
        with pytest.raises(ValueError, match="Missing required coordinate"):
            register_volume(da, da, transform_type="translation")

    def test_singleton_z_without_voxdim_raises(self, sample_2d_image):
        """Singleton spatial dimensions need spacing metadata for registration."""
        da = xr.DataArray(
            sample_2d_image[np.newaxis],
            dims=("z", "y", "x"),
            coords={"z": [0.0], "y": np.arange(32.0), "x": np.arange(32.0)},
        )

        with pytest.warns(UserWarning, match="spacing is undefined"):
            with pytest.raises(ValueError, match="SimpleITK image.*z"):
                register_volume(da, da, transform_type="translation")

    def test_returns_affine_matrix(self, sample_singleton_z_dataarray_spatial):
        """register_volume returns a (4, 4) numpy affine matrix for single-slice input."""
        _, affine, _ = register_volume(
            sample_singleton_z_dataarray_spatial,
            sample_singleton_z_dataarray_spatial,
            transform_type="translation",
        )
        assert isinstance(affine, np.ndarray)
        assert affine.shape == (4, 4)

    def test_bspline_returns_dataarray_transform(
        self, sample_singleton_z_dataarray_spatial
    ):
        """register_volume with bspline returns a DataArray for the transform."""
        _, bspline_tx, _ = register_volume(
            sample_singleton_z_dataarray_spatial,
            sample_singleton_z_dataarray_spatial,
            transform_type="bspline",
        )
        assert isinstance(bspline_tx, xr.DataArray)
        assert bspline_tx.attrs.get("transform_type") == "bspline_transform"
        assert bspline_tx.dims[0] == "component"
        np.testing.assert_array_equal(
            bspline_tx.coords["component"].values, ["z", "y", "x"]
        )

    def test_bspline_control_point_domain_matches_each_axis_extent(self):
        """Each axis's control-point domain scales with its own physical extent.

        Regression test for a bug where `sitk_bspline_to_dataarray` assumed
        SimpleITK reverses axis order relative to the DataArray (`(x, y, z)` vs.
        `(z, y, x)`), when in fact this codebase's convention (see
        `dataarray_to_sitk_image`) never reverses axes: sitk axis `i` maps directly
        to DataArray dim `i`. On an anisotropic image, the erroneous reversal
        swapped the y/x control-point grids: `y`'s spacing was computed from `x`'s
        physical domain and vice versa. Isotropic test fixtures never exposed this
        because swapping equal-sized, equal-spacing axes is a no-op.
        """
        img = np.zeros((20, 40), dtype=np.float32)
        img[6:14, 10:30] = 100.0
        # Single-slice recording: singleton `z` axis with anisotropic in-plane spacing.
        da = xr.DataArray(
            img[np.newaxis],
            dims=("z", "y", "x"),
            coords={
                "z": xr.DataArray([0.0], dims=("z",), attrs={"voxdim": 0.5}),
                "y": np.arange(20) * 0.5,
                "x": np.arange(40) * 0.1,
            },
        )
        _, bspline_tx, _ = register_volume(
            da,
            da,
            transform_type="bspline",
            mesh_size=(1, 4, 4),
        )

        y_span = float(
            bspline_tx.coords["y"].values[-1] - bspline_tx.coords["y"].values[0]
        )
        x_span = float(
            bspline_tx.coords["x"].values[-1] - bspline_tx.coords["x"].values[0]
        )
        # The control-point domain is padded beyond the image FOV for boundary
        # support, so spans are somewhat larger than the raw physical extent (9.5 mm
        # for y, 3.9 mm for x). Padding scales with each axis's own extent (same mesh
        # size, so padding is proportional to domain size), so the span ratio should
        # track the physical extent ratio (9.5 / 3.9 ~= 2.44) rather than being
        # swapped with the other axis's.
        assert y_span / x_span == pytest.approx(9.5 / 3.9, rel=0.3)

    def test_resample_true_coords_match_fixed(
        self, sample_2d_image, sample_singleton_z_dataarray_spatial
    ):
        """resample=True output coordinates match the fixed volume, not moving."""
        moving = sample_singleton_z_dataarray_spatial.isel(y=slice(16), x=slice(16))
        result, _, _ = register_volume(
            moving,
            sample_singleton_z_dataarray_spatial,
            transform_type="translation",
            resample=True,
        )
        assert_allclose(
            result.coords["y"].values,
            sample_singleton_z_dataarray_spatial.coords["y"].values,
        )
        assert_allclose(
            result.coords["x"].values,
            sample_singleton_z_dataarray_spatial.coords["x"].values,
        )

    def test_resample_true_inherits_fixed_affines(
        self, sample_singleton_z_dataarray_spatial
    ):
        """resample=True output inherits physical-space affines from `fixed`."""
        moving = sample_singleton_z_dataarray_spatial.isel(
            y=slice(16), x=slice(16)
        ).copy()
        fixed = sample_singleton_z_dataarray_spatial.copy()
        moving.attrs["affines"] = {"physical_to_lab": np.diag([2.0, 2.0, 1.0])}
        fixed.attrs["affines"] = {"physical_to_lab": np.diag([3.0, 3.0, 1.0])}

        result, _, _ = register_volume(
            moving,
            fixed,
            transform_type="translation",
            resample=True,
        )

        assert "registration" not in result.attrs
        assert_allclose(
            result.attrs["affines"]["physical_to_lab"],
            fixed.attrs["affines"]["physical_to_lab"],
        )


class TestRegisterVolumeMask:
    """Metric masks for register_volume."""

    def test_integer_label_mask_matches_boolean_mask(
        self, sample_2d_image, sample_singleton_z_dataarray_spatial
    ):
        """A single-label integer mask registers identically to its boolean form.

        Guards against single-label integer masks (e.g. ``{0, 512}`` from
        `Atlas.get_masks`) reaching SimpleITK's metric mask uncoerced: 512 wraps to 0
        under the `numpy.uint8` cast, which silently empties the mask and turns
        registration into a no-op.
        """
        shift = 2
        shifted = np.roll(np.roll(sample_2d_image, shift, axis=0), shift, axis=1)
        fixed = sample_singleton_z_dataarray_spatial
        moving = xr.DataArray(shifted[np.newaxis], dims=fixed.dims, coords=fixed.coords)

        region = np.zeros(fixed.shape, dtype=bool)
        region[:, 4:28, 4:28] = True  # covers the bright square in both volumes
        bool_mask = xr.DataArray(region, dims=fixed.dims, coords=fixed.coords)
        # 512 is a multiple of 256: a uint8 cast of the raw integer mask wraps it to 0.
        int_mask = xr.DataArray(
            region.astype(np.int32) * 512, dims=fixed.dims, coords=fixed.coords
        )

        _, affine_bool, _ = register_volume(
            moving,
            fixed,
            fixed_mask=bool_mask,
            transform_type="translation",
            resample=False,
        )
        _, affine_int, _ = register_volume(
            moving,
            fixed,
            fixed_mask=int_mask,
            transform_type="translation",
            resample=False,
        )

        # The masked registration must actually recover the planted shift; otherwise the
        # equality check would also pass for a silently-emptied (no-op) mask.
        assert not np.allclose(affine_bool, np.eye(4), atol=1e-2)
        assert_allclose(affine_int, affine_bool)

    def test_both_masks_coerced_to_bool(
        self, sample_2d_image, sample_singleton_z_dataarray_spatial
    ):
        """Both fixed_mask and moving_mask are coerced to boolean."""
        shift = 2
        shifted = np.roll(np.roll(sample_2d_image, shift, axis=0), shift, axis=1)
        fixed = sample_singleton_z_dataarray_spatial
        moving = xr.DataArray(shifted[np.newaxis], dims=fixed.dims, coords=fixed.coords)

        region = np.zeros(fixed.shape, dtype=bool)
        region[:, 4:28, 4:28] = True
        fixed_mask = xr.DataArray(region, dims=fixed.dims, coords=fixed.coords)
        moving_mask = xr.DataArray(region, dims=fixed.dims, coords=fixed.coords)

        _, affine, _ = register_volume(
            moving,
            fixed,
            fixed_mask=fixed_mask,
            moving_mask=moving_mask,
            transform_type="translation",
            resample=False,
        )
        assert not np.allclose(affine, np.eye(4), atol=1e-2)


class TestRegisterVolumeResample:
    """Behaviour of the resample parameter."""

    def test_no_resample_returns_moving_values_unchanged(
        self, sample_2d_image, sample_singleton_z_dataarray_spatial
    ):
        """resample=False returns moving values without modification."""
        rng = np.random.default_rng(0)
        shift = rng.integers(1, 4, size=2)
        shifted = np.roll(
            np.roll(sample_2d_image, int(shift[0]), axis=0), int(shift[1]), axis=1
        )
        moving = xr.DataArray(
            shifted[np.newaxis],
            dims=sample_singleton_z_dataarray_spatial.dims,
            coords=sample_singleton_z_dataarray_spatial.coords,
        )
        result, _, _ = register_volume(
            moving,
            sample_singleton_z_dataarray_spatial,
            transform_type="translation",
            resample=False,
        )
        assert_array_equal(result.values, moving.values)

    def test_resample_true_aligns_to_fixed(
        self, sample_2d_image, sample_singleton_z_dataarray_spatial
    ):
        """resample=True produces output close to fixed (the registration target)."""
        # Use a fixed shift of 2 pixels to avoid wrap-around contamination from np.roll.
        shift = 2
        shifted = np.roll(np.roll(sample_2d_image, shift, axis=0), shift, axis=1)
        moving = xr.DataArray(
            shifted[np.newaxis],
            dims=sample_singleton_z_dataarray_spatial.dims,
            coords=sample_singleton_z_dataarray_spatial.coords,
        )
        result, _, _ = register_volume(
            moving,
            sample_singleton_z_dataarray_spatial,
            transform_type="translation",
            learning_rate=1.0,
            number_of_iterations=200,
            resample=True,
        )
        # Compare only the interior to avoid boundary wrap-around artifacts.
        margin = shift + 1
        assert_allclose(
            result.values[margin:-margin, margin:-margin],
            sample_singleton_z_dataarray_spatial.values[margin:-margin, margin:-margin],
            atol=10.0,
        )


class TestRegisterVolumeAccuracy:
    """Registration accuracy for register_volume."""

    def test_identical_volumes_unchanged_2d(self, sample_singleton_z_dataarray_spatial):
        """Registering identical 2D volumes produces nearly identical output."""
        result, _, _ = register_volume(
            sample_singleton_z_dataarray_spatial,
            sample_singleton_z_dataarray_spatial,
            transform_type="translation",
            resample=True,
        )
        assert_allclose(
            result.values, sample_singleton_z_dataarray_spatial.values, atol=1e-3
        )

    def test_identical_volumes_unchanged_3d(self, sample_3d_dataarray_spatial):
        """Registering identical 3D volumes produces nearly identical output."""
        result, _, _ = register_volume(
            sample_3d_dataarray_spatial,
            sample_3d_dataarray_spatial,
            transform_type="translation",
            resample=True,
        )
        assert_allclose(result.values, sample_3d_dataarray_spatial.values, atol=1e-3)

    def test_3d_recovers_known_shift(self, sample_3d_array):
        """Registration recovers a known 3D translation."""
        shifted = np.roll(sample_3d_array, 2, axis=0)
        spacing = (1.0, 1.0, 1.0)
        fixed = xr.DataArray(
            sample_3d_array,
            dims=("z", "y", "x"),
            coords={
                d: np.arange(sample_3d_array.shape[i]) * spacing[i]
                for i, d in enumerate(("z", "y", "x"))
            },
        )
        moving = xr.DataArray(shifted, dims=fixed.dims, coords=fixed.coords)
        result, _, _ = register_volume(
            moving,
            fixed,
            transform_type="translation",
            learning_rate=1.0,
            number_of_iterations=200,
            resample=True,
        )
        # Compare only the interior to avoid boundary wrap-around artifacts.
        margin = 3
        assert_allclose(
            result.values[margin:-margin, margin:-margin, margin:-margin],
            fixed.values[margin:-margin, margin:-margin, margin:-margin],
            atol=10.0,
        )

    def test_optimizer_weights_freezes_rotation(
        self, sample_singleton_z_dataarray_spatial
    ):
        """Setting rotation weight to 0 produces the same result as translation-only."""
        da = sample_singleton_z_dataarray_spatial
        _, affine_translation, _ = register_volume(da, da, transform_type="translation")
        # 3D rigid with rotations frozen: [rot_x, rot_y, rot_z, tx, ty, tz] with
        # weight [0, 0, 0, 1, 1, 1].
        _, affine_frozen, _ = register_volume(
            da,
            da,
            transform_type="rigid",
            optimizer_weights=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        )
        # The rotation sub-matrix should be identity (no rotation applied).
        assert_allclose(affine_frozen[:3, :3], np.eye(3), atol=1e-4)


class TestRegisterVolumeThinDims:
    """register_volume with volumes that have a unitary or thin dimension."""

    def test_3d_volume_with_depth_1_does_not_crash(self):
        """3D volume with depth=1 (coronal fUSI scan) registers without error."""
        arr = np.zeros((1, 32, 32), dtype=np.float32)
        arr[0, 12:20, 12:20] = 1.0
        da = xr.DataArray(
            arr,
            dims=("z", "y", "x"),
            coords={
                "z": xr.DataArray([0.0], dims=("z",), attrs={"voxdim": 0.2}),
                "y": np.arange(32) * 0.1,
                "x": np.arange(32) * 0.1,
            },
        )
        result, _, _ = register_volume(da, da, transform_type="translation")
        assert result.shape == da.shape

    def test_3d_volume_with_depth_1_preserves_output_shape_on_resample(self):
        """resample=True preserves the original shape for a depth-1 volume."""
        arr = np.zeros((1, 32, 32), dtype=np.float32)
        arr[0, 12:20, 12:20] = 1.0
        da = xr.DataArray(
            arr,
            dims=("z", "y", "x"),
            coords={
                "z": xr.DataArray([0.0], dims=("z",), attrs={"voxdim": 0.2}),
                "y": np.arange(32) * 0.1,
                "x": np.arange(32) * 0.1,
            },
        )
        result, _, _ = register_volume(
            da, da, transform_type="translation", resample=True
        )
        assert result.shape == da.shape

    def test_float32_moving_float64_fixed_does_not_crash(
        self, sample_singleton_z_dataarray_spatial
    ):
        """float32 moving and float64 fixed register without a dtype mismatch error.

        Regression test: CenteredTransformInitializer requires both images to share the
        same pixel type. Mixed dtypes (e.g. float32 template vs. float64 mean of NIfTI
        data) previously raised a RuntimeError.
        """
        moving = sample_singleton_z_dataarray_spatial  # float32
        fixed = sample_singleton_z_dataarray_spatial.astype(np.float64)
        result, _, _ = register_volume(moving, fixed, transform_type="translation")
        assert result.shape == fixed.shape

    def test_3d_volume_with_depth_2_does_not_crash(self):
        """3D volume with depth=2 (below the 4-voxel threshold) registers without error."""
        arr = np.zeros((2, 16, 16), dtype=np.float32)
        arr[:, 6:10, 6:10] = 1.0
        da = xr.DataArray(
            arr,
            dims=("z", "y", "x"),
            coords={
                "z": np.arange(2) * 0.5,
                "y": np.arange(16) * 0.1,
                "x": np.arange(16) * 0.1,
            },
        )
        result, _, _ = register_volume(da, da, transform_type="translation")
        assert result.shape == da.shape


class TestResampleVolume:
    """Unit tests for the low-level resample_volume."""

    def test_time_dimension_moving_works(
        self,
        sample_2d_image,
        sample_singleton_z_dataarray,
        sample_singleton_z_dataarray_spatial,
    ):
        """moving with a time dimension resamples each frame with the same transform."""
        result = resample_volume(
            sample_singleton_z_dataarray,
            np.eye(4),
            **get_grid_kwargs_from_dataarray(sample_singleton_z_dataarray_spatial),
        )
        assert "time" in result.dims
        assert result.shape == sample_singleton_z_dataarray.shape
        assert_allclose(
            result.coords["time"].values,
            sample_singleton_z_dataarray.coords["time"].values,
        )

    def test_3d_time_dimension_moving_works(
        self, sample_3d_dataarray, sample_3d_dataarray_spatial
    ):
        """3D moving with time dimension resamples each frame with the same transform."""
        result = resample_volume(
            sample_3d_dataarray,
            np.eye(4),
            **get_grid_kwargs_from_dataarray(sample_3d_dataarray_spatial),
        )
        assert "time" in result.dims
        assert result.shape == sample_3d_dataarray.shape
        assert_allclose(
            result.coords["time"].values, sample_3d_dataarray.coords["time"].values
        )

    def test_wrong_ndim_raises(self):
        """1D input raises ValueError."""
        da = xr.DataArray(np.zeros(10), dims=("x",))
        with pytest.raises(ValueError, match="must contain all spatial dimensions"):
            resample_volume(
                da, np.eye(2), shape=[10], spacing=[1.0], origin=[0.0], dims=["x"]
            )

    def test_affine_shape_mismatch_raises(self, sample_singleton_z_dataarray_spatial):
        """Affine with wrong shape raises ValueError."""
        with pytest.raises(ValueError, match="affine shape"):
            resample_volume(
                sample_singleton_z_dataarray_spatial,
                np.eye(3),  # wrong: 2D affine for a 3D single-slice volume
                **get_grid_kwargs_from_dataarray(sample_singleton_z_dataarray_spatial),
            )

    def test_output_shape_matches_requested_shape(
        self, sample_2d_image, sample_singleton_z_dataarray_spatial
    ):
        """Output shape matches the requested shape, not the moving shape."""
        moving = sample_singleton_z_dataarray_spatial.isel(y=slice(16), x=slice(16))
        result = resample_volume(
            moving,
            np.eye(4),
            **get_grid_kwargs_from_dataarray(sample_singleton_z_dataarray_spatial),
        )
        assert result.shape == sample_singleton_z_dataarray_spatial.shape

    def test_coords_reconstructed_from_origin_and_spacing(
        self, sample_singleton_z_dataarray_spatial
    ):
        """Output coordinates are reconstructed from origin and spacing, not copied."""
        grid = get_grid_kwargs_from_dataarray(sample_singleton_z_dataarray_spatial)
        result = resample_volume(
            sample_singleton_z_dataarray_spatial, np.eye(4), **grid
        )
        for i, d in enumerate(sample_singleton_z_dataarray_spatial.dims):
            expected = (
                grid["origin"][i] + np.arange(grid["shape"][i]) * grid["spacing"][i]
            )
            assert_allclose(result.coords[d].values, expected)

    def test_matches_register_volume_resample(
        self, sample_2d_image, sample_singleton_z_dataarray_spatial
    ):
        """resample_volume matches register_volume(resample=True) on a shifted image."""
        rng = np.random.default_rng(42)
        shift = rng.integers(3, 6, size=2)
        shifted = np.roll(
            np.roll(sample_2d_image, int(shift[0]), axis=0), int(shift[1]), axis=1
        )
        moving = xr.DataArray(
            shifted[np.newaxis],
            dims=sample_singleton_z_dataarray_spatial.dims,
            coords=sample_singleton_z_dataarray_spatial.coords,
        )
        resampled_direct, affine, _ = register_volume(
            moving,
            sample_singleton_z_dataarray_spatial,
            transform_type="translation",
            resample=True,
        )
        result = resample_volume(
            moving,
            affine,
            **get_grid_kwargs_from_dataarray(sample_singleton_z_dataarray_spatial),
        )
        assert_allclose(result.values, resampled_direct.values, atol=1e-5)


class TestInitialization:
    """Tests for the initialization parameter of register_volume."""

    def test_wrong_shape_raises(self, sample_singleton_z_dataarray_spatial):
        """Affine initialization with wrong shape raises ValueError."""
        with pytest.raises(ValueError, match="initialization shape"):
            register_volume(
                sample_singleton_z_dataarray_spatial,
                sample_singleton_z_dataarray_spatial,
                transform_type="bspline",
                initialization=np.eye(3),  # wrong: 2D affine for 3D single-slice images
            )

    def test_bspline_with_affine_initialization_stores_pre_affine(
        self, sample_singleton_z_dataarray_spatial
    ):
        """B-spline result stores the pre-affine when affine initialization is given."""
        pre_affine = np.eye(4)
        _, bspline_tx, diagnostics = register_volume(
            sample_singleton_z_dataarray_spatial,
            sample_singleton_z_dataarray_spatial,
            transform_type="bspline",
            initialization=pre_affine,
            mesh_size=(1, 1, 1),
            number_of_iterations=1,
            learning_rate=0.1,
        )
        assert diagnostics.status == "completed"
        assert isinstance(bspline_tx, xr.DataArray)
        assert "affines" in bspline_tx.attrs
        assert "bspline_initialization" in bspline_tx.attrs["affines"]

    def test_bspline_without_affine_initialization_has_no_pre_affine(
        self, sample_singleton_z_dataarray_spatial
    ):
        """B-spline result without affine initialization has no bspline_initialization key."""
        _, bspline_tx, _ = register_volume(
            sample_singleton_z_dataarray_spatial,
            sample_singleton_z_dataarray_spatial,
            transform_type="bspline",
        )
        assert isinstance(bspline_tx, xr.DataArray)
        affines = bspline_tx.attrs.get("affines", {})
        assert "bspline_initialization" not in affines

    def test_center_moments_uses_moments_initializer(
        self, sample_singleton_z_dataarray_spatial, monkeypatch
    ):
        """center_moments uses SimpleITK's moments-based centering initializer."""
        import SimpleITK as sitk

        original_initializer = sitk.CenteredTransformInitializer
        calls = []

        def wrapped_initializer(fixed, moving, transform, operation_mode):
            calls.append(operation_mode)
            return original_initializer(fixed, moving, transform, operation_mode)

        monkeypatch.setattr(sitk, "CenteredTransformInitializer", wrapped_initializer)

        register_volume(
            sample_singleton_z_dataarray_spatial,
            sample_singleton_z_dataarray_spatial,
            transform_type="affine",
            initialization="center_moments",
        )

        assert calls == [sitk.CenteredTransformInitializerFilter.MOMENTS]


class TestResampleVolumeWithBspline:
    """Tests for resample_volume and resample_like with a B-spline DataArray transform."""

    def test_resample_like_with_bspline_matches_direct_resample(
        self, sample_2d_image, sample_singleton_z_dataarray_spatial
    ):
        """resample_like with a B-spline DataArray matches register_volume(resample=True)."""
        rng = np.random.default_rng(0)
        shift = rng.integers(3, 6, size=2)
        shifted = np.roll(
            np.roll(sample_2d_image, int(shift[0]), axis=0), int(shift[1]), axis=1
        )
        moving = xr.DataArray(
            shifted[np.newaxis],
            dims=sample_singleton_z_dataarray_spatial.dims,
            coords=sample_singleton_z_dataarray_spatial.coords,
        )
        resampled_direct, bspline_tx, _ = register_volume(
            moving,
            sample_singleton_z_dataarray_spatial,
            transform_type="bspline",
            resample=True,
            mesh_size=(1, 1, 1),
            learning_rate=0.1,
        )
        assert isinstance(bspline_tx, xr.DataArray)
        result = resample_like(moving, sample_singleton_z_dataarray_spatial, bspline_tx)
        np.testing.assert_allclose(result.values, resampled_direct.values, atol=1e-5)

    def test_resample_like_with_composite_bspline_matches_direct_resample(
        self, sample_2d_image, sample_singleton_z_dataarray_spatial
    ):
        """resample_like with composite B-spline matches register_volume(resample=True)."""
        rng = np.random.default_rng(1)
        shift = rng.integers(2, 4, size=2)
        shifted = np.roll(
            np.roll(sample_2d_image, int(shift[0]), axis=0), int(shift[1]), axis=1
        )
        moving = xr.DataArray(
            shifted[np.newaxis],
            dims=sample_singleton_z_dataarray_spatial.dims,
            coords=sample_singleton_z_dataarray_spatial.coords,
        )
        # First pass: affine registration.
        _, affine_tx, _ = register_volume(
            moving,
            sample_singleton_z_dataarray_spatial,
            transform_type="affine",
        )
        # Second pass: B-spline refinement on top of the affine.
        resampled_direct, bspline_tx, _ = register_volume(
            moving,
            sample_singleton_z_dataarray_spatial,
            transform_type="bspline",
            initialization=affine_tx,
            resample=True,
            mesh_size=(1, 1, 1),
            learning_rate=0.1,
        )
        assert isinstance(bspline_tx, xr.DataArray)
        result = resample_like(moving, sample_singleton_z_dataarray_spatial, bspline_tx)
        np.testing.assert_allclose(result.values, resampled_direct.values, atol=1e-5)


class TestDisplacementField:
    """Tests for displacement-field sampling, inversion, and resampling."""

    def test_sitk_bspline_to_dataarray_rejects_non_bspline_transform(self):
        """Non-B-spline SimpleITK transforms are rejected."""
        import SimpleITK as sitk

        with pytest.raises(TypeError, match="BSplineTransform"):
            sitk_bspline_to_dataarray(sitk.AffineTransform(2))

    def test_sample_displacement_field_wrong_type_attr_raises(self):
        """A DataArray with the wrong `type` attr is rejected."""
        transform = xr.DataArray(
            np.zeros((2, 4, 4)),
            dims=["component", "y", "x"],
            coords={"component": ["y", "x"], "y": np.arange(4.0), "x": np.arange(4.0)},
            attrs={
                "type": "displacement_field_transform",
                "order": 3,
                "direction": np.eye(2).tolist(),
            },
        )
        with pytest.raises(ValueError, match="bspline_transform"):
            sample_displacement_field(
                transform,
                shape=[4, 4],
                spacing=[1.0, 1.0],
                origin=[0.0, 0.0],
                dims=["y", "x"],
            )

    def test_sample_displacement_field_missing_required_attr_raises(self):
        """Missing B-spline metadata is rejected."""
        transform = xr.DataArray(
            np.zeros((2, 4, 4)),
            dims=["component", "y", "x"],
            coords={"component": ["y", "x"], "y": np.arange(4.0), "x": np.arange(4.0)},
            attrs={"transform_type": "bspline_transform", "order": 3},
        )
        with pytest.raises(ValueError, match="direction"):
            sample_displacement_field(
                transform,
                shape=[4, 4],
                spacing=[1.0, 1.0],
                origin=[0.0, 0.0],
                dims=["y", "x"],
            )

    def test_sample_displacement_field_wrong_first_dim_raises(self):
        """A B-spline DataArray without leading 'component' is rejected."""
        transform = xr.DataArray(
            np.zeros((4, 4, 2)),
            dims=["y", "x", "component"],
            coords={"y": np.arange(4.0), "x": np.arange(4.0), "component": ["y", "x"]},
            attrs={
                "transform_type": "bspline_transform",
                "order": 3,
                "direction": np.eye(2).tolist(),
            },
        )
        with pytest.raises(ValueError, match="'component' as its first dimension"):
            sample_displacement_field(
                transform,
                shape=[4, 4],
                spacing=[1.0, 1.0],
                origin=[0.0, 0.0],
                dims=["y", "x"],
            )

    def test_sample_displacement_field_like_time_reference_raises(
        self, sample_singleton_z_dataarray
    ):
        """The `_like` wrapper rejects references with a time dimension."""
        transform = xr.DataArray(
            np.zeros((2, 4, 4)),
            dims=["component", "y", "x"],
            coords={"component": ["y", "x"], "y": np.arange(4.0), "x": np.arange(4.0)},
            attrs={
                "transform_type": "bspline_transform",
                "order": 3,
                "direction": np.eye(2).tolist(),
            },
        )
        with pytest.raises(ValueError, match="time dimension"):
            sample_displacement_field_like(transform, sample_singleton_z_dataarray)

    def test_invert_displacement_field_wrong_type_attr_raises(self):
        """A DataArray with the wrong `type` attr is rejected."""
        field = xr.DataArray(
            np.zeros((2, 4, 4)),
            dims=["component", "y", "x"],
            coords={"component": [0, 1], "y": np.arange(4.0), "x": np.arange(4.0)},
            attrs={"transform_type": "bspline_transform"},
        )
        with pytest.raises(ValueError, match="displacement_field_transform"):
            invert_displacement_field(field)

    def test_invert_displacement_field_wrong_first_dim_raises(self):
        """A DataArray without 'component' as its first dimension is rejected."""
        field = xr.DataArray(
            np.zeros((4, 4, 2)),
            dims=["y", "x", "component"],
            coords={"y": np.arange(4.0), "x": np.arange(4.0), "component": [0, 1]},
            attrs={"type": "displacement_field_transform"},
        )
        with pytest.raises(ValueError, match="'component' as its first dimension"):
            invert_displacement_field(field)

    def test_invert_displacement_field_singleton_z_without_voxdim_raises(self):
        """Displacement-field inversion validates field spacing before inversion."""
        field = xr.DataArray(
            np.zeros((3, 1, 4, 4)),
            dims=["component", "z", "y", "x"],
            coords={
                "component": [0, 1, 2],
                "z": [0.0],
                "y": np.arange(4.0),
                "x": np.arange(4.0),
            },
            attrs={"type": "displacement_field_transform"},
        )

        with pytest.warns(UserWarning, match="spacing is undefined"):
            with pytest.raises(ValueError, match="convert displacement field.*z"):
                invert_displacement_field(field)

    def test_resample_displacement_field_singleton_z_without_voxdim_raises(self):
        """Displacement-field resampling requires defined singleton spacing."""
        moving = xr.DataArray(
            np.zeros((1, 4, 4)),
            dims=("z", "y", "x"),
            coords={
                "z": xr.DataArray([0.0], dims=("z",), attrs={"voxdim": 0.2}),
                "y": np.arange(4.0),
                "x": np.arange(4.0),
            },
        )
        field = xr.DataArray(
            np.zeros((3, 1, 4, 4)),
            dims=["component", "z", "y", "x"],
            coords={
                "component": [0, 1, 2],
                "z": [0.0],
                "y": np.arange(4.0),
                "x": np.arange(4.0),
            },
            attrs={"type": "displacement_field_transform"},
        )

        with pytest.warns(UserWarning, match="spacing is undefined"):
            with pytest.raises(ValueError, match="convert displacement field.*z"):
                resample_volume(
                    moving,
                    field,
                    shape=[1, 4, 4],
                    spacing=[0.2, 1.0, 1.0],
                    origin=[0.0, 0.0, 0.0],
                    dims=["z", "y", "x"],
                )

    def test_sample_displacement_field_returns_valid_dataarray(
        self, sample_singleton_z_dataarray_spatial
    ):
        """Sampling an identity B-spline transform yields a near-zero dense field."""
        _, bspline_tx, _ = register_volume(
            sample_singleton_z_dataarray_spatial,
            sample_singleton_z_dataarray_spatial,
            transform_type="bspline",
        )
        grid = get_grid_kwargs_from_dataarray(sample_singleton_z_dataarray_spatial)
        field = sample_displacement_field(bspline_tx, **grid)

        assert field.attrs["type"] == "displacement_field_transform"
        assert field.dims[0] == "component"
        np.testing.assert_array_equal(field.coords["component"].values, ["z", "y", "x"])
        assert field.shape == (3, *sample_singleton_z_dataarray_spatial.shape)
        assert_allclose(field.values, 0.0, atol=1e-6)

    def test_sample_displacement_field_like_matches_explicit_grid(
        self, sample_singleton_z_dataarray_spatial
    ):
        """The `_like` wrapper matches explicit-grid sampling."""
        _, bspline_tx, _ = register_volume(
            sample_singleton_z_dataarray_spatial,
            sample_singleton_z_dataarray_spatial,
            transform_type="bspline",
        )

        by_grid = sample_displacement_field(
            bspline_tx,
            **get_grid_kwargs_from_dataarray(sample_singleton_z_dataarray_spatial),
        )
        by_reference = sample_displacement_field_like(
            bspline_tx, sample_singleton_z_dataarray_spatial
        )

        assert_array_equal(by_reference.coords["component"].values, ["z", "y", "x"])
        assert_allclose(by_reference.values, by_grid.values, atol=1e-6)
        assert_allclose(
            by_reference.coords["y"].values,
            sample_singleton_z_dataarray_spatial.coords["y"].values,
        )
        assert_allclose(
            by_reference.coords["x"].values,
            sample_singleton_z_dataarray_spatial.coords["x"].values,
        )

    def test_invert_displacement_field_undoes_translation(self):
        """Inverting a constant translation field approximately negates it.

        Only the interior of the grid is checked: pixels near the boundary map
        outside the field's domain under the translation, which the inversion
        cannot resolve (there is nothing to invert against there).
        """
        shape = (12, 12)
        dims = ["y", "x"]
        translation = np.array([2.0, -1.5])
        array = np.broadcast_to(translation[:, None, None], (2, *shape)).astype(
            np.float64
        )
        field = xr.DataArray(
            array.copy(),
            dims=["component", *dims],
            coords={
                "component": [0, 1],
                "y": np.arange(shape[0], dtype=np.float64),
                "x": np.arange(shape[1], dtype=np.float64),
            },
            attrs={"type": "displacement_field_transform"},
        )

        inverted = invert_displacement_field(field)

        assert inverted.attrs["type"] == "displacement_field_transform"
        interior = np.s_[:, 4:8, 4:8]
        assert_allclose(inverted.values[interior], -array[interior], atol=1e-2)

    def test_resample_volume_with_displacement_field_matches_bspline(
        self, sample_2d_image, sample_singleton_z_dataarray_spatial
    ):
        """resample_volume with a displacement field matches the equivalent B-spline resample."""
        rng = np.random.default_rng(2)
        shift = rng.integers(3, 6, size=2)
        shifted = np.roll(
            np.roll(sample_2d_image, int(shift[0]), axis=0), int(shift[1]), axis=1
        )
        moving = xr.DataArray(
            shifted[np.newaxis],
            dims=sample_singleton_z_dataarray_spatial.dims,
            coords=sample_singleton_z_dataarray_spatial.coords,
        )
        _, bspline_tx, _ = register_volume(
            moving,
            sample_singleton_z_dataarray_spatial,
            transform_type="bspline",
        )
        grid = get_grid_kwargs_from_dataarray(sample_singleton_z_dataarray_spatial)
        field = sample_displacement_field(bspline_tx, **grid)

        result_bspline = resample_volume(moving, bspline_tx, **grid)
        result_field = resample_volume(moving, field, **grid)

        assert_allclose(result_field.values, result_bspline.values, atol=1e-4)

    def test_matches_bspline_with_singleton_spatial_dim(
        self, sample_2d_image, sample_singleton_z_dataarray_spatial
    ):
        """A singleton spatial dim (a single 2D slice stored as (1, y, x)) must not
        produce NaN spacing anywhere in the field round trip.

        Regression test: `coords[dim].diff(dim)` is empty for a length-1 axis, so
        `.mean()` silently returns NaN. Field construction/consumption must fall back
        to the `voxdim` coordinate attribute instead (via the `fusi` accessor), as
        `resample_volume`'s own grid handling already does.
        """
        # The fixture is already a single-slice (1, y, x) volume with a singleton z.
        fixed = sample_singleton_z_dataarray_spatial.copy()
        fixed.coords["z"].attrs["voxdim"] = 0.5

        rng = np.random.default_rng(3)
        shift = rng.integers(3, 6, size=2)
        shifted = np.roll(
            np.roll(sample_2d_image, int(shift[0]), axis=0), int(shift[1]), axis=1
        )
        moving = xr.DataArray(shifted[np.newaxis], dims=fixed.dims, coords=fixed.coords)
        _, bspline_tx, _ = register_volume(moving, fixed, transform_type="bspline")

        grid = get_grid_kwargs_from_dataarray(fixed)
        field = sample_displacement_field(bspline_tx, **grid)
        assert not np.isnan(field.values).any()

        result_bspline = resample_volume(moving, bspline_tx, **grid)
        result_field = resample_volume(moving, field, **grid)
        assert_allclose(result_field.values, result_bspline.values, atol=1e-4)

        inverse_field = invert_displacement_field(field)
        assert not np.isnan(inverse_field.values).any()

    def test_invert_displacement_field_with_singleton_spatial_dim_is_nonzero(
        self, sample_2d_image, sample_singleton_z_dataarray_spatial
    ):
        """Inverting a field with a singleton spatial axis must not silently no-op.

        Regression test: `InvertDisplacementFieldImageFilter` requires an N-D image
        with N-component vectors and silently returns an all-zero field when any
        spatial axis has size 1, since it has no local neighborhood to compute a
        fixed-point update from along that axis. `invert_displacement_field` must
        expand the degenerate axis before inverting and crop it back down afterward,
        rather than passing the degenerate field straight to the filter.
        """
        # The fixture is already a single-slice (1, y, x) volume with a singleton z.
        fixed = sample_singleton_z_dataarray_spatial.copy()
        fixed.coords["z"].attrs["voxdim"] = 0.5

        rng = np.random.default_rng(4)
        shift = rng.integers(3, 6, size=2)
        shifted = np.roll(
            np.roll(sample_2d_image, int(shift[0]), axis=0), int(shift[1]), axis=1
        )
        moving = xr.DataArray(shifted[np.newaxis], dims=fixed.dims, coords=fixed.coords)
        _, bspline_tx, _ = register_volume(moving, fixed, transform_type="bspline")

        grid = get_grid_kwargs_from_dataarray(fixed)
        field = sample_displacement_field(bspline_tx, **grid)
        inverse_field = invert_displacement_field(field)

        # The degenerate z axis has no spatial variation to invert against, so its
        # displacement component is ~zero (platform-dependent floating-point noise,
        # not exactly 0.0), but y/x must be genuinely inverted, not silently zeroed
        # out along with it.
        assert_allclose(inverse_field.values[0], 0.0, atol=1e-9)
        assert np.abs(inverse_field.values[1:]).max() > 0.1


class TestResampleLike:
    """Unit tests for resample_like."""

    def test_time_dimension_moving_works(
        self, sample_singleton_z_dataarray, sample_singleton_z_dataarray_spatial
    ):
        """moving with a time dimension resamples each frame with the same transform."""
        result = resample_like(
            sample_singleton_z_dataarray,
            sample_singleton_z_dataarray_spatial,
            np.eye(4),
        )
        assert "time" in result.dims
        assert result.shape == sample_singleton_z_dataarray.shape
        assert_allclose(
            result.coords["time"].values,
            sample_singleton_z_dataarray.coords["time"].values,
        )

    def test_time_dimension_reference_raises(
        self, sample_singleton_z_dataarray, sample_singleton_z_dataarray_spatial
    ):
        """reference with a time dimension raises ValueError."""
        with pytest.raises(ValueError, match="time"):
            resample_like(
                sample_singleton_z_dataarray_spatial,
                sample_singleton_z_dataarray,
                np.eye(4),
            )

    def test_mismatched_units_between_moving_and_reference_raise(
        self, sample_singleton_z_dataarray_spatial
    ):
        """moving and reference must agree on spatial coordinate units when declared."""
        moving = sample_singleton_z_dataarray_spatial.copy()
        reference = sample_singleton_z_dataarray_spatial.copy()
        moving.coords["y"].attrs["units"] = "mm"
        moving.coords["x"].attrs["units"] = "mm"
        reference.coords["y"].attrs["units"] = "um"
        reference.coords["x"].attrs["units"] = "um"

        with pytest.raises(ValueError, match="units"):
            resample_like(moving, reference, np.eye(4))

    def test_mismatched_units_between_transform_and_reference_raise(
        self, sample_singleton_z_dataarray_spatial
    ):
        """DataArray transforms must agree with the reference units when declared."""
        reference = sample_singleton_z_dataarray_spatial.copy()
        reference.coords["y"].attrs["units"] = "mm"
        reference.coords["x"].attrs["units"] = "mm"
        transform = xr.DataArray(
            np.zeros((2, 2, 2), dtype=np.float64),
            dims=["component", "y", "x"],
            coords={
                "component": np.arange(2),
                "y": xr.Variable("y", [0.0, 1.0], attrs={"units": "um"}),
                "x": xr.Variable("x", [0.0, 1.0], attrs={"units": "um"}),
            },
            attrs={"type": "displacement_field_transform"},
        )

        with pytest.raises(ValueError, match="units"):
            resample_like(reference, reference, transform)

    def test_wrong_ndim_reference_raises(self):
        """1D reference raises ValueError."""
        da = xr.DataArray(np.zeros(10), dims=("x",))
        with pytest.raises(ValueError, match="must contain all spatial dimensions"):
            resample_like(da, da, np.eye(2))

    def test_default_fill_is_moving_min(self, sample_singleton_z_dataarray_spatial):
        """Out-of-FOV voxels default to moving.min(), not 0.0."""
        moving = xr.DataArray(
            np.ones((1, 8, 8), dtype=np.float32) * 5.0,
            dims=("z", "y", "x"),
            coords={
                "z": sample_singleton_z_dataarray_spatial.coords["z"],
                "y": sample_singleton_z_dataarray_spatial.coords["y"].values[:8],
                "x": sample_singleton_z_dataarray_spatial.coords["x"].values[:8],
            },
        )
        result = resample_like(moving, sample_singleton_z_dataarray_spatial, np.eye(4))
        assert float(result.values[0, -1, -1]) == pytest.approx(5.0, abs=1e-5)

    def test_explicit_fill_value_overrides(self, sample_singleton_z_dataarray_spatial):
        """Explicit fill_value overrides the auto-default."""
        moving = xr.DataArray(
            np.ones((1, 8, 8), dtype=np.float32) * 5.0,
            dims=("z", "y", "x"),
            coords={
                "z": sample_singleton_z_dataarray_spatial.coords["z"],
                "y": sample_singleton_z_dataarray_spatial.coords["y"].values[:8],
                "x": sample_singleton_z_dataarray_spatial.coords["x"].values[:8],
            },
        )
        result = resample_like(
            moving, sample_singleton_z_dataarray_spatial, np.eye(4), fill_value=0.0
        )
        assert float(result.values[0, -1, -1]) == pytest.approx(0.0, abs=1e-5)

    def test_output_coords_match_reference(
        self, sample_2d_image, sample_singleton_z_dataarray_spatial
    ):
        """Output coordinates match reference, not moving."""
        moving = sample_singleton_z_dataarray_spatial.isel(y=slice(16), x=slice(16))
        result = resample_like(moving, sample_singleton_z_dataarray_spatial, np.eye(4))
        assert_allclose(
            result.coords["y"].values,
            sample_singleton_z_dataarray_spatial.coords["y"].values,
        )
        assert_allclose(
            result.coords["x"].values,
            sample_singleton_z_dataarray_spatial.coords["x"].values,
        )

    def test_inherits_reference_affines(self, sample_singleton_z_dataarray_spatial):
        """resample_like output inherits physical-space affines from `reference`."""
        moving = sample_singleton_z_dataarray_spatial.isel(
            y=slice(16), x=slice(16)
        ).copy()
        reference = sample_singleton_z_dataarray_spatial.copy()
        moving.attrs["affines"] = {"physical_to_lab": np.diag([2.0, 2.0, 1.0])}
        reference.attrs["affines"] = {"physical_to_lab": np.diag([3.0, 3.0, 1.0])}

        result = resample_like(moving, reference, np.eye(4))

        assert "registration" not in result.attrs
        assert_allclose(
            result.attrs["affines"]["physical_to_lab"],
            reference.attrs["affines"]["physical_to_lab"],
        )

    def test_matches_register_volume_resample_2d(
        self, sample_2d_image, sample_singleton_z_dataarray_spatial
    ):
        """resample_like matches register_volume(resample=True) on a shifted 2D image."""
        rng = np.random.default_rng(42)
        shift = rng.integers(3, 6, size=2)
        shifted = np.roll(
            np.roll(sample_2d_image, int(shift[0]), axis=0), int(shift[1]), axis=1
        )
        moving = xr.DataArray(
            shifted[np.newaxis],
            dims=sample_singleton_z_dataarray_spatial.dims,
            coords=sample_singleton_z_dataarray_spatial.coords,
        )
        resampled_direct, affine, _ = register_volume(
            moving,
            sample_singleton_z_dataarray_spatial,
            transform_type="translation",
            resample=True,
        )
        result = resample_like(moving, sample_singleton_z_dataarray_spatial, affine)
        assert_allclose(result.values, resampled_direct.values, atol=1e-5)

    def test_matches_register_volume_resample_3d(
        self, sample_3d_array, sample_3d_dataarray_spatial
    ):
        """resample_like matches register_volume(resample=True) in 3D."""
        shifted = np.roll(sample_3d_array, 2, axis=0)
        moving = xr.DataArray(
            shifted,
            dims=sample_3d_dataarray_spatial.dims,
            coords=sample_3d_dataarray_spatial.coords,
        )
        resampled_direct, affine, _ = register_volume(
            moving,
            sample_3d_dataarray_spatial,
            transform_type="translation",
            learning_rate=1.0,
            number_of_iterations=200,
            resample=True,
        )
        result = resample_like(moving, sample_3d_dataarray_spatial, affine)
        assert_allclose(result.values, resampled_direct.values, atol=1e-5)

    def test_matches_register_volume_with_affine_initialization(
        self, sample_2d_image, sample_singleton_z_dataarray_spatial
    ):
        """resample_like matches register_volume(resample=True) when affine initialization is used.

        Regression test for a bug where CompositeTransform sub-transforms were
        composed in the wrong order in _sitk_linear_transform_to_affine, causing
        the returned affine matrix to differ from the transform actually applied
        during resampling.
        """
        rng = np.random.default_rng(42)
        shift = rng.integers(2, 4, size=2)
        shifted = np.roll(
            np.roll(sample_2d_image, int(shift[0]), axis=0), int(shift[1]), axis=1
        )
        moving = xr.DataArray(
            shifted[np.newaxis],
            dims=sample_singleton_z_dataarray_spatial.dims,
            coords=sample_singleton_z_dataarray_spatial.coords,
        )
        _, affine_init, _ = register_volume(
            moving, sample_singleton_z_dataarray_spatial, transform_type="translation"
        )
        resampled_direct, affine, _ = register_volume(
            moving,
            sample_singleton_z_dataarray_spatial,
            transform_type="affine",
            initialization=affine_init,
            resample=True,
        )
        result = resample_like(moving, sample_singleton_z_dataarray_spatial, affine)
        assert_allclose(result.values, resampled_direct.values, atol=1e-5)

    def test_matches_resample_volume(
        self, sample_2d_image, sample_singleton_z_dataarray_spatial
    ):
        """resample_like and resample_volume produce identical results."""
        moving = sample_singleton_z_dataarray_spatial.isel(y=slice(16), x=slice(16))
        affine = np.eye(4)
        result_like = resample_like(
            moving, sample_singleton_z_dataarray_spatial, affine
        )
        result_vol = resample_volume(
            moving,
            affine,
            **get_grid_kwargs_from_dataarray(sample_singleton_z_dataarray_spatial),
        )
        assert_allclose(result_like.values, result_vol.values, atol=1e-10)


class TestRegisterVolumeDiagnostics:
    """Diagnostics object returned by register_volume."""

    def test_returns_diagnostics_with_consistent_fields(
        self, sample_2d_image, sample_singleton_z_dataarray_spatial
    ):
        """register_volume returns a fully populated RegistrationDiagnostics."""
        shifted = np.roll(np.roll(sample_2d_image, 2, axis=0), 2, axis=1)
        moving = xr.DataArray(
            shifted[np.newaxis],
            dims=sample_singleton_z_dataarray_spatial.dims,
            coords=sample_singleton_z_dataarray_spatial.coords,
        )

        max_iters = 50
        _, _, diagnostics = register_volume(
            moving,
            sample_singleton_z_dataarray_spatial,
            transform_type="translation",
            number_of_iterations=max_iters,
        )

        assert isinstance(diagnostics, RegistrationDiagnostics)
        assert diagnostics.metric == "correlation"
        # metric_values is a 1D numpy array, one entry per iteration.
        assert isinstance(diagnostics.metric_values, np.ndarray)
        assert diagnostics.metric_values.ndim == 1
        assert diagnostics.metric_values.shape == (diagnostics.n_iterations,)
        # final_metric_value mirrors metric_values[-1] when at least one
        # iteration ran.
        assert diagnostics.n_iterations >= 1
        assert diagnostics.n_iterations <= max_iters
        assert diagnostics.final_metric_value == pytest.approx(
            float(diagnostics.metric_values[-1])
        )
        # SimpleITK populates a non-empty stop condition string at the end.
        assert isinstance(diagnostics.stop_condition, str)
        assert diagnostics.stop_condition != ""

    def test_metric_field_echoes_argument(self, sample_singleton_z_dataarray_spatial):
        """The `metric` field on diagnostics matches the metric argument."""
        _, _, diagnostics = register_volume(
            sample_singleton_z_dataarray_spatial,
            sample_singleton_z_dataarray_spatial,
            transform_type="translation",
            metric="mattes_mi",
        )
        assert diagnostics.metric == "mattes_mi"


class TestRegisterVolumeFillValue:
    """fill_value applies to both the final resample output and the progress plotter."""

    def test_explicit_fill_value_appears_in_out_of_fov_voxels(self):
        """Out-of-FOV voxels in the registered output are filled with fill_value."""
        # moving is a small sub-region of fixed; after registration the output grid
        # is fixed-sized, so voxels outside moving's FOV must be filled.
        fixed = xr.DataArray(
            np.ones((1, 16, 16), dtype=np.float32),
            dims=("z", "y", "x"),
            coords={
                "z": xr.DataArray([0.0], dims=("z",), attrs={"voxdim": 0.1}),
                "y": np.arange(16) * 0.1,
                "x": np.arange(16) * 0.1,
            },
        )
        # moving covers only the central 8x8 region.
        moving = xr.DataArray(
            np.ones((1, 8, 8), dtype=np.float32) * 2.0,
            dims=("z", "y", "x"),
            coords={
                "z": xr.DataArray([0.0], dims=("z",), attrs={"voxdim": 0.1}),
                "y": np.arange(4, 12) * 0.1,
                "x": np.arange(4, 12) * 0.1,
            },
        )
        sentinel = -99.0
        result, _, _ = register_volume(
            moving,
            fixed,
            transform_type="translation",
            fill_value=sentinel,
        )
        # Out-of-FOV voxels (corners) should be exactly fill_value.
        assert float(result.values[0, 0, 0]) == pytest.approx(sentinel, abs=1e-5)

    def test_default_fill_value_is_moving_min(self):
        """When fill_value is None, out-of-FOV voxels are filled with moving.min()."""
        fixed = xr.DataArray(
            np.ones((1, 16, 16), dtype=np.float32),
            dims=("z", "y", "x"),
            coords={
                "z": xr.DataArray([0.0], dims=("z",), attrs={"voxdim": 0.1}),
                "y": np.arange(16) * 0.1,
                "x": np.arange(16) * 0.1,
            },
        )
        moving = xr.DataArray(
            np.ones((1, 8, 8), dtype=np.float32) * 2.0,
            dims=("z", "y", "x"),
            coords={
                "z": xr.DataArray([0.0], dims=("z",), attrs={"voxdim": 0.1}),
                "y": np.arange(4, 12) * 0.1,
                "x": np.arange(4, 12) * 0.1,
            },
        )
        result, _, _ = register_volume(
            moving,
            fixed,
            transform_type="translation",
        )
        # Default fill should be moving.min() == 2.0, not 0.0.
        assert float(result.values[0, 0, 0]) == pytest.approx(
            float(moving.min()), abs=1e-5
        )


class TestRegisterVolumePreSetAbort:
    """Pre-set abort_event short-circuits before SimpleITK Execute is called."""

    def test_bspline_abort_returns_initial_bspline_transform(
        self, sample_singleton_z_dataarray_spatial
    ):
        """Pre-aborted bspline returns a DataArray without forcing a bspline fit."""
        abort_event = Event()
        abort_event.set()

        _, transform, diagnostics = register_volume(
            sample_singleton_z_dataarray_spatial,
            sample_singleton_z_dataarray_spatial,
            transform_type="bspline",
            abort_event=abort_event,
        )

        assert diagnostics.status == "aborted"
        assert diagnostics.n_iterations == 0
        assert (
            diagnostics.stop_condition
            == "Registration aborted before optimisation started."
        )
        # The returned DataArray wraps the initial (unoptimised) bspline — its
        # coefficients differ from a real registration only in that no iterations ran.
        assert isinstance(transform, xr.DataArray)
        assert transform.attrs.get("transform_type") == "bspline_transform"

    def test_affine_initialization_abort_returns_initialization_affine(
        self, sample_singleton_z_dataarray_spatial
    ):
        """Pre-aborted linear registration returns the provided affine initialization.

        The transform must match the initialization matrix — not the default
        identity/TranslationTransform fallback used when no initialization is set —
        so downstream consumers can rely on a coherent aborted transform.
        """
        pre_affine = np.array(
            [
                [1.0, 0.0, 0.0, 0.5],
                [0.0, 1.0, 0.0, -0.25],
                [0.0, 0.0, 1.0, 0.1],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        abort_event = Event()
        abort_event.set()

        _, transform, diagnostics = register_volume(
            sample_singleton_z_dataarray_spatial,
            sample_singleton_z_dataarray_spatial,
            transform_type="rigid",
            initialization=pre_affine,
            abort_event=abort_event,
        )

        assert diagnostics.status == "aborted"
        assert diagnostics.n_iterations == 0
        assert_allclose(transform, pre_affine)


class TestRegisterVolumeConvergesBeforeFirstIteration:
    """`final_metric_value` falls back to the optimizer's metric when no iteration event fires."""

    def test_final_metric_value_pulled_from_optimizer_when_no_iterations(
        self, sample_singleton_z_dataarray_spatial
    ):
        """When SimpleITK converges before any iteration event, final_metric_value is
        the optimizer's current metric, not NaN.

        Achieved by raising `convergence_minimum_value` above the metric for identical
        images and shrinking the window to 1, so the convergence checker passes at
        iteration 0 before any iteration event fires.
        """
        _, _, diagnostics = register_volume(
            sample_singleton_z_dataarray_spatial,
            sample_singleton_z_dataarray_spatial,
            transform_type="translation",
            number_of_iterations=100,
            convergence_minimum_value=1.0,
            convergence_window_size=1,
        )

        assert diagnostics.n_iterations == 0
        assert diagnostics.status == "completed"
        assert np.isfinite(diagnostics.final_metric_value)
        assert "Convergence checker passed at iteration 0" in diagnostics.stop_condition


class TestRegisterVolumeFromWorkerThread:
    """`register_volume` works when called from a non-main thread."""

    def test_register_volume_runs_in_non_main_thread(
        self, sample_singleton_z_dataarray_spatial
    ):
        """Calling `register_volume` from a worker thread bypasses SIGINT wiring.

        The non-main-thread branch of `abort_on_sigint` skips installing a SIGINT
        handler and simply yields the abort event, so registration runs to
        completion without trying to mutate the main thread's signal handlers.
        """
        import threading

        from confusius.registration.diagnostics import RegistrationDiagnostics

        result_holder: dict[str, object] = {}

        def worker() -> None:
            assert threading.current_thread() is not threading.main_thread()
            result, transform, diagnostics = register_volume(
                sample_singleton_z_dataarray_spatial,
                sample_singleton_z_dataarray_spatial,
                transform_type="translation",
                number_of_iterations=2,
            )
            result_holder["result"] = result
            result_holder["transform"] = transform
            result_holder["diagnostics"] = diagnostics

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()

        result = result_holder["result"]
        diagnostics = result_holder["diagnostics"]
        assert isinstance(result, xr.DataArray)
        assert isinstance(diagnostics, RegistrationDiagnostics)
        assert result.shape == sample_singleton_z_dataarray_spatial.shape
        assert diagnostics.status == "completed"

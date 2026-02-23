"""Unit tests for single-volume registration."""

import numpy as np
import pytest
import SimpleITK as sitk
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal

from confusius.registration.resampling import resample_like, resample_volume
from confusius.registration.volume import (
    _expand_thin_dims,
    _validate_register_volume_inputs,
    register_volume,
)


def _base_validation_kwargs(da: xr.DataArray) -> dict:
    """Return a minimal valid kwargs dict for _validate_register_volume_inputs."""
    return dict(
        moving=da,
        fixed=da,
        transform="rigid",
        metric="correlation",
        number_of_histogram_bins=50,
        learning_rate="auto",
        number_of_iterations=100,
        convergence_window_size=10,
        initialization="geometry",
        shrink_factors=(6, 2, 1),
        smoothing_sigmas=(6, 2, 1),
        resample_interpolation="linear",
    )


class TestRegisterVolumeValidation:
    """Input validation for register_volume."""

    def test_time_dimension_raises(self, sample_2d_dataarray):
        """DataArray with a time dimension raises ValueError."""
        with pytest.raises(ValueError, match="spatial-only"):
            register_volume(sample_2d_dataarray, sample_2d_dataarray)

    def test_wrong_ndim_1d_raises(self):
        """1D input raises ValueError."""
        da = xr.DataArray(np.zeros(10), dims=("x",))
        with pytest.raises(ValueError, match="2D or 3D"):
            register_volume(da, da)

    def test_wrong_ndim_4d_raises(self):
        """4D input raises ValueError."""
        da = xr.DataArray(np.zeros((4, 4, 4, 4)), dims=("a", "b", "c", "d"))
        with pytest.raises(ValueError, match="2D or 3D"):
            register_volume(da, da)

    def test_shape_mismatch_no_error(
        self, sample_2d_image, sample_2d_dataarray_spatial
    ):
        """Different shapes do not raise an error."""
        moving = sample_2d_dataarray_spatial.isel(y=slice(16), x=slice(16))
        result, _ = register_volume(
            moving, sample_2d_dataarray_spatial, transform="translation"
        )
        assert result.shape == moving.shape


class TestValidateRegisterVolumeInputs:
    """Unit tests for the _validate_register_volume_inputs helper."""

    def test_valid_inputs_do_not_raise(self, sample_2d_dataarray_spatial):
        """A fully valid set of inputs raises no exception."""
        _validate_register_volume_inputs(
            **_base_validation_kwargs(sample_2d_dataarray_spatial)
        )

    def test_invalid_transform_raises(self, sample_2d_dataarray_spatial):
        """Unrecognised transform value raises ValueError."""
        kwargs = _base_validation_kwargs(sample_2d_dataarray_spatial)
        kwargs["transform"] = "foobar"
        with pytest.raises(ValueError, match="transform"):
            _validate_register_volume_inputs(**kwargs)

    def test_invalid_metric_raises(self, sample_2d_dataarray_spatial):
        """Unrecognised metric value raises ValueError."""
        kwargs = _base_validation_kwargs(sample_2d_dataarray_spatial)
        kwargs["metric"] = "ncc"
        with pytest.raises(ValueError, match="metric"):
            _validate_register_volume_inputs(**kwargs)

    def test_invalid_initialization_raises(self, sample_2d_dataarray_spatial):
        """Unrecognised initialization value raises ValueError."""
        kwargs = _base_validation_kwargs(sample_2d_dataarray_spatial)
        kwargs["initialization"] = "random"
        with pytest.raises(ValueError, match="initialization"):
            _validate_register_volume_inputs(**kwargs)

    def test_invalid_resample_interpolation_raises(self, sample_2d_dataarray_spatial):
        """Unrecognised resample_interpolation value raises ValueError."""
        kwargs = _base_validation_kwargs(sample_2d_dataarray_spatial)
        kwargs["resample_interpolation"] = "nearest"
        with pytest.raises(ValueError, match="resample_interpolation"):
            _validate_register_volume_inputs(**kwargs)

    def test_negative_learning_rate_raises(self, sample_2d_dataarray_spatial):
        """Negative learning rate raises ValueError."""
        kwargs = _base_validation_kwargs(sample_2d_dataarray_spatial)
        kwargs["learning_rate"] = -0.5
        with pytest.raises(ValueError, match="learning_rate"):
            _validate_register_volume_inputs(**kwargs)

    def test_zero_learning_rate_raises(self, sample_2d_dataarray_spatial):
        """Zero learning rate raises ValueError."""
        kwargs = _base_validation_kwargs(sample_2d_dataarray_spatial)
        kwargs["learning_rate"] = 0.0
        with pytest.raises(ValueError, match="learning_rate"):
            _validate_register_volume_inputs(**kwargs)

    def test_nan_learning_rate_raises(self, sample_2d_dataarray_spatial):
        """NaN learning rate raises ValueError."""
        kwargs = _base_validation_kwargs(sample_2d_dataarray_spatial)
        kwargs["learning_rate"] = float("nan")
        with pytest.raises(ValueError, match="learning_rate"):
            _validate_register_volume_inputs(**kwargs)

    def test_zero_number_of_iterations_raises(self, sample_2d_dataarray_spatial):
        """Zero number_of_iterations raises ValueError."""
        kwargs = _base_validation_kwargs(sample_2d_dataarray_spatial)
        kwargs["number_of_iterations"] = 0
        with pytest.raises(ValueError, match="number_of_iterations"):
            _validate_register_volume_inputs(**kwargs)

    def test_float_number_of_iterations_raises(self, sample_2d_dataarray_spatial):
        """Float number_of_iterations raises ValueError."""
        kwargs = _base_validation_kwargs(sample_2d_dataarray_spatial)
        kwargs["number_of_iterations"] = 10.5  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="number_of_iterations"):
            _validate_register_volume_inputs(**kwargs)

    def test_zero_convergence_window_size_raises(self, sample_2d_dataarray_spatial):
        """Zero convergence_window_size raises ValueError."""
        kwargs = _base_validation_kwargs(sample_2d_dataarray_spatial)
        kwargs["convergence_window_size"] = 0
        with pytest.raises(ValueError, match="convergence_window_size"):
            _validate_register_volume_inputs(**kwargs)

    def test_zero_number_of_histogram_bins_raises(self, sample_2d_dataarray_spatial):
        """Zero number_of_histogram_bins raises ValueError."""
        kwargs = _base_validation_kwargs(sample_2d_dataarray_spatial)
        kwargs["number_of_histogram_bins"] = 0
        with pytest.raises(ValueError, match="number_of_histogram_bins"):
            _validate_register_volume_inputs(**kwargs)

    def test_mismatched_shrink_smoothing_raises(self, sample_2d_dataarray_spatial):
        """shrink_factors and smoothing_sigmas with different lengths raise ValueError."""
        kwargs = _base_validation_kwargs(sample_2d_dataarray_spatial)
        kwargs["shrink_factors"] = (4, 2, 1)
        kwargs["smoothing_sigmas"] = (2, 1)
        with pytest.raises(ValueError, match="shrink_factors"):
            _validate_register_volume_inputs(**kwargs)


class TestRegisterVolumeOutput:
    """Output properties for register_volume."""

    def test_without_coords_uses_unit_spacing(self, sample_2d_image):
        """DataArray without coordinates warns for both spacing and origin."""
        da = xr.DataArray(sample_2d_image, dims=("y", "x"))
        with pytest.warns(UserWarning):
            register_volume(da, da, transform="translation")

    def test_returns_affine_matrix(self, sample_2d_dataarray_spatial):
        """register_volume returns a (3, 3) numpy affine matrix for 2D input."""
        _, affine = register_volume(
            sample_2d_dataarray_spatial,
            sample_2d_dataarray_spatial,
            transform="translation",
        )
        assert isinstance(affine, np.ndarray)
        assert affine.shape == (3, 3)

    def test_bspline_returns_none_affine(self, sample_2d_dataarray_spatial):
        """register_volume with bspline returns None for the affine."""
        _, affine = register_volume(
            sample_2d_dataarray_spatial,
            sample_2d_dataarray_spatial,
            transform="bspline",
        )
        assert affine is None

    def test_resample_true_coords_match_fixed(
        self, sample_2d_image, sample_2d_dataarray_spatial
    ):
        """resample=True output coordinates match the fixed volume, not moving."""
        moving = sample_2d_dataarray_spatial.isel(y=slice(16), x=slice(16))
        result, _ = register_volume(
            moving, sample_2d_dataarray_spatial, transform="translation", resample=True
        )
        assert_allclose(
            result.coords["y"].values, sample_2d_dataarray_spatial.coords["y"].values
        )
        assert_allclose(
            result.coords["x"].values, sample_2d_dataarray_spatial.coords["x"].values
        )


class TestRegisterVolumeResample:
    """Behaviour of the resample parameter."""

    def test_no_resample_returns_moving_values_unchanged(
        self, sample_2d_image, sample_2d_dataarray_spatial
    ):
        """resample=False returns moving values without modification."""
        rng = np.random.default_rng(0)
        shift = rng.integers(1, 4, size=2)
        shifted = np.roll(
            np.roll(sample_2d_image, int(shift[0]), axis=0), int(shift[1]), axis=1
        )
        moving = xr.DataArray(
            shifted,
            dims=sample_2d_dataarray_spatial.dims,
            coords=sample_2d_dataarray_spatial.coords,
        )
        result, _ = register_volume(
            moving, sample_2d_dataarray_spatial, transform="translation", resample=False
        )
        assert_array_equal(result.values, moving.values)

    def test_resample_true_modifies_values(
        self, sample_2d_image, sample_2d_dataarray_spatial
    ):
        """resample=True produces output values different from the unregistered moving image."""
        rng = np.random.default_rng(0)
        shift = rng.integers(3, 6, size=2)
        shifted = np.roll(
            np.roll(sample_2d_image, int(shift[0]), axis=0), int(shift[1]), axis=1
        )
        moving = xr.DataArray(
            shifted,
            dims=sample_2d_dataarray_spatial.dims,
            coords=sample_2d_dataarray_spatial.coords,
        )
        result, _ = register_volume(
            moving, sample_2d_dataarray_spatial, transform="translation", resample=True
        )
        assert not np.allclose(result.values, moving.values, atol=1e-3)


class TestRegisterVolumeAccuracy:
    """Registration accuracy for register_volume."""

    def test_identical_volumes_unchanged_2d(self, sample_2d_dataarray_spatial):
        """Registering identical 2D volumes produces nearly identical output."""
        result, _ = register_volume(
            sample_2d_dataarray_spatial,
            sample_2d_dataarray_spatial,
            transform="translation",
            resample=True,
        )
        assert_allclose(result.values, sample_2d_dataarray_spatial.values, atol=1e-3)

    def test_identical_volumes_unchanged_3d(self, sample_3d_dataarray_spatial):
        """Registering identical 3D volumes produces nearly identical output."""
        result, _ = register_volume(
            sample_3d_dataarray_spatial,
            sample_3d_dataarray_spatial,
            transform="translation",
            resample=True,
        )
        assert_allclose(result.values, sample_3d_dataarray_spatial.values, atol=1e-3)

    def test_3d_recovers_known_shift(self, sample_3d_volume):
        """Registration recovers a known 3D translation."""
        shifted = np.roll(sample_3d_volume, 2, axis=0)
        spacing = (1.0, 1.0, 1.0)
        fixed = xr.DataArray(
            sample_3d_volume,
            dims=("z", "y", "x"),
            coords={
                d: np.arange(sample_3d_volume.shape[i]) * spacing[i]
                for i, d in enumerate(("z", "y", "x"))
            },
        )
        moving = xr.DataArray(shifted, dims=fixed.dims, coords=fixed.coords)
        result, _ = register_volume(
            moving,
            fixed,
            transform="translation",
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

    def test_optimizer_weights_freezes_rotation(self, sample_2d_dataarray_spatial):
        """Setting rotation weight to 0 produces the same result as translation-only."""
        da = sample_2d_dataarray_spatial
        _, affine_translation = register_volume(da, da, transform="translation")
        # 2D rigid with rotation frozen: [rotation, tx, ty] with weight [0, 1, 1].
        _, affine_frozen = register_volume(
            da, da, transform="rigid", optimizer_weights=[0.0, 1.0, 1.0]
        )
        assert affine_translation is not None
        assert affine_frozen is not None
        # The rotation sub-matrix should be identity (no rotation applied).
        assert_allclose(affine_frozen[:2, :2], np.eye(2), atol=1e-4)


class TestExpandThinDims:
    """Unit tests for _expand_thin_dims."""

    def test_no_expansion_when_dims_large_enough(self):
        """Image with all dims >= 4 is returned unchanged (same object)."""
        img = sitk.Image(8, 8, sitk.sitkFloat32)
        assert _expand_thin_dims(img) is img

    def test_thin_dim_expanded(self):
        """A dimension of size 1 is expanded to at least 4."""
        img = sitk.Image(1, 8, sitk.sitkFloat32)
        expanded = _expand_thin_dims(img)
        assert expanded.GetSize()[0] >= 4

    def test_physical_extent_preserved(self):
        """Expanding a thin dim keeps the physical size (size * spacing) constant."""
        img = sitk.Image(1, 8, sitk.sitkFloat32)
        img.SetSpacing((2.0, 1.0))
        expanded = _expand_thin_dims(img)
        original_extent = img.GetSize()[0] * img.GetSpacing()[0]
        expanded_extent = expanded.GetSize()[0] * expanded.GetSpacing()[0]
        assert abs(expanded_extent - original_extent) < 1e-6

    def test_3d_thin_dim_expanded(self):
        """A 3D image with one thin dimension is correctly expanded."""
        img = sitk.Image(1, 16, 16, sitk.sitkFloat32)
        expanded = _expand_thin_dims(img)
        assert expanded.GetSize()[0] >= 4
        assert expanded.GetSize()[1] == 16
        assert expanded.GetSize()[2] == 16


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
                "z": np.array([0.0]),
                "y": np.arange(32) * 0.1,
                "x": np.arange(32) * 0.1,
            },
        )
        with pytest.warns(UserWarning, match="spacing is undefined"):
            result, _ = register_volume(da, da, transform="translation")
        assert result.shape == da.shape

    def test_3d_volume_with_depth_1_preserves_output_shape_on_resample(self):
        """resample=True preserves the original shape for a depth-1 volume."""
        arr = np.zeros((1, 32, 32), dtype=np.float32)
        arr[0, 12:20, 12:20] = 1.0
        da = xr.DataArray(
            arr,
            dims=("z", "y", "x"),
            coords={
                "z": np.array([0.0]),
                "y": np.arange(32) * 0.1,
                "x": np.arange(32) * 0.1,
            },
        )
        with pytest.warns(UserWarning, match="spacing is undefined"):
            result, _ = register_volume(da, da, transform="translation", resample=True)
        assert result.shape == da.shape

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
        result, _ = register_volume(da, da, transform="translation")
        assert result.shape == da.shape


class TestResampleVolume:
    """Unit tests for the low-level resample_volume."""

    def _grid_from_da(self, da: xr.DataArray) -> dict:
        """Extract explicit grid kwargs from a DataArray."""
        return dict(
            shape=[da.sizes[d] for d in da.dims],
            spacing=[float(da.coords[d].diff(d).mean()) for d in da.dims],
            origin=[float(da.coords[d][0]) for d in da.dims],
            dims=list(da.dims),
        )

    def test_time_dimension_moving_raises(
        self, sample_2d_image, sample_2d_dataarray, sample_2d_dataarray_spatial
    ):
        """moving with a time dimension raises ValueError."""
        with pytest.raises(ValueError, match="time"):
            resample_volume(
                sample_2d_dataarray,
                np.eye(3),
                **self._grid_from_da(sample_2d_dataarray_spatial),
            )

    def test_wrong_ndim_raises(self):
        """1D input raises ValueError."""
        da = xr.DataArray(np.zeros(10), dims=("x",))
        with pytest.raises(ValueError, match="2D or 3D"):
            resample_volume(
                da, np.eye(2), shape=[10], spacing=[1.0], origin=[0.0], dims=["x"]
            )

    def test_affine_shape_mismatch_raises(self, sample_2d_dataarray_spatial):
        """Affine with wrong shape raises ValueError."""
        with pytest.raises(ValueError, match="affine shape"):
            resample_volume(
                sample_2d_dataarray_spatial,
                np.eye(4),
                **self._grid_from_da(sample_2d_dataarray_spatial),
            )

    def test_output_shape_matches_requested_shape(
        self, sample_2d_image, sample_2d_dataarray_spatial
    ):
        """Output shape matches the requested shape, not the moving shape."""
        moving = sample_2d_dataarray_spatial.isel(y=slice(16), x=slice(16))
        result = resample_volume(
            moving, np.eye(3), **self._grid_from_da(sample_2d_dataarray_spatial)
        )
        assert result.shape == sample_2d_dataarray_spatial.shape

    def test_coords_reconstructed_from_origin_and_spacing(
        self, sample_2d_dataarray_spatial
    ):
        """Output coordinates are reconstructed from origin and spacing, not copied."""
        grid = self._grid_from_da(sample_2d_dataarray_spatial)
        result = resample_volume(sample_2d_dataarray_spatial, np.eye(3), **grid)
        for i, d in enumerate(sample_2d_dataarray_spatial.dims):
            expected = (
                grid["origin"][i] + np.arange(grid["shape"][i]) * grid["spacing"][i]
            )
            assert_allclose(result.coords[d].values, expected)

    def test_matches_register_volume_resample(
        self, sample_2d_image, sample_2d_dataarray_spatial
    ):
        """resample_volume matches register_volume(resample=True) on a shifted image."""
        rng = np.random.default_rng(42)
        shift = rng.integers(3, 6, size=2)
        shifted = np.roll(
            np.roll(sample_2d_image, int(shift[0]), axis=0), int(shift[1]), axis=1
        )
        moving = xr.DataArray(
            shifted,
            dims=sample_2d_dataarray_spatial.dims,
            coords=sample_2d_dataarray_spatial.coords,
        )
        resampled_direct, affine = register_volume(
            moving, sample_2d_dataarray_spatial, transform="translation", resample=True
        )
        assert affine is not None
        result = resample_volume(
            moving, affine, **self._grid_from_da(sample_2d_dataarray_spatial)
        )
        assert_allclose(result.values, resampled_direct.values, atol=1e-5)


class TestResampleLike:
    """Unit tests for resample_like."""

    def test_time_dimension_moving_raises(
        self, sample_2d_dataarray, sample_2d_dataarray_spatial
    ):
        """moving with a time dimension raises ValueError."""
        with pytest.raises(ValueError, match="time"):
            resample_like(sample_2d_dataarray, sample_2d_dataarray_spatial, np.eye(3))

    def test_time_dimension_reference_raises(
        self, sample_2d_dataarray, sample_2d_dataarray_spatial
    ):
        """reference with a time dimension raises ValueError."""
        with pytest.raises(ValueError, match="time"):
            resample_like(sample_2d_dataarray_spatial, sample_2d_dataarray, np.eye(3))

    def test_wrong_ndim_reference_raises(self):
        """1D reference raises ValueError."""
        da = xr.DataArray(np.zeros(10), dims=("x",))
        with pytest.raises(ValueError, match="2D or 3D"):
            resample_like(da, da, np.eye(2))

    def test_output_coords_match_reference(
        self, sample_2d_image, sample_2d_dataarray_spatial
    ):
        """Output coordinates match reference, not moving."""
        moving = sample_2d_dataarray_spatial.isel(y=slice(16), x=slice(16))
        result = resample_like(moving, sample_2d_dataarray_spatial, np.eye(3))
        assert_allclose(
            result.coords["y"].values, sample_2d_dataarray_spatial.coords["y"].values
        )
        assert_allclose(
            result.coords["x"].values, sample_2d_dataarray_spatial.coords["x"].values
        )

    def test_matches_register_volume_resample_2d(
        self, sample_2d_image, sample_2d_dataarray_spatial
    ):
        """resample_like matches register_volume(resample=True) on a shifted 2D image."""
        rng = np.random.default_rng(42)
        shift = rng.integers(3, 6, size=2)
        shifted = np.roll(
            np.roll(sample_2d_image, int(shift[0]), axis=0), int(shift[1]), axis=1
        )
        moving = xr.DataArray(
            shifted,
            dims=sample_2d_dataarray_spatial.dims,
            coords=sample_2d_dataarray_spatial.coords,
        )
        resampled_direct, affine = register_volume(
            moving, sample_2d_dataarray_spatial, transform="translation", resample=True
        )
        assert affine is not None
        result = resample_like(moving, sample_2d_dataarray_spatial, affine)
        assert_allclose(result.values, resampled_direct.values, atol=1e-5)

    def test_matches_register_volume_resample_3d(
        self, sample_3d_volume, sample_3d_dataarray_spatial
    ):
        """resample_like matches register_volume(resample=True) in 3D."""
        shifted = np.roll(sample_3d_volume, 2, axis=0)
        moving = xr.DataArray(
            shifted,
            dims=sample_3d_dataarray_spatial.dims,
            coords=sample_3d_dataarray_spatial.coords,
        )
        resampled_direct, affine = register_volume(
            moving,
            sample_3d_dataarray_spatial,
            transform="translation",
            learning_rate=1.0,
            number_of_iterations=200,
            resample=True,
        )
        assert affine is not None
        result = resample_like(moving, sample_3d_dataarray_spatial, affine)
        assert_allclose(result.values, resampled_direct.values, atol=1e-5)

    def test_matches_resample_volume(
        self, sample_2d_image, sample_2d_dataarray_spatial
    ):
        """resample_like and resample_volume produce identical results."""
        moving = sample_2d_dataarray_spatial.isel(y=slice(16), x=slice(16))
        affine = np.eye(3)
        result_like = resample_like(moving, sample_2d_dataarray_spatial, affine)
        result_vol = resample_volume(
            moving,
            affine,
            shape=[
                sample_2d_dataarray_spatial.sizes[d]
                for d in sample_2d_dataarray_spatial.dims
            ],
            spacing=[
                float(sample_2d_dataarray_spatial.coords[d].diff(d).mean())
                for d in sample_2d_dataarray_spatial.dims
            ],
            origin=[
                float(sample_2d_dataarray_spatial.coords[d][0])
                for d in sample_2d_dataarray_spatial.dims
            ],
            dims=[str(d) for d in sample_2d_dataarray_spatial.dims],
        )
        assert_allclose(result_like.values, result_vol.values, atol=1e-10)

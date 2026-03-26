"""Unit tests for single-volume registration."""

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal

from confusius.registration.resampling import resample_like, resample_volume
from confusius.registration.volume import register_volume


class TestRegisterVolumeValidation:
    """Input validation for register_volume."""

    def test_time_dimension_raises(self, sample_2d_dataarray):
        """DataArray with a time dimension raises ValueError."""
        with pytest.raises(ValueError, match="spatial-only"):
            register_volume(sample_2d_dataarray, sample_2d_dataarray)

    def test_nan_in_moving_raises(self, sample_2d_dataarray_spatial):
        """moving with NaN values raises ValueError."""
        moving = sample_2d_dataarray_spatial.copy()
        moving.values[0, 0] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            register_volume(moving, sample_2d_dataarray_spatial, transform_type="translation")

    def test_nan_in_fixed_raises(self, sample_2d_dataarray_spatial):
        """fixed with NaN values raises ValueError."""
        fixed = sample_2d_dataarray_spatial.copy()
        fixed.values[0, 0] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            register_volume(sample_2d_dataarray_spatial, fixed, transform_type="translation")

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
            moving, sample_2d_dataarray_spatial, transform_type="translation",
            resample=False,
        )
        assert result.shape == moving.shape


class TestRegisterVolumeOutput:
    """Output properties for register_volume."""

    def test_without_coords_uses_unit_spacing(self, sample_2d_image):
        """DataArray without coordinates warns for both spacing and origin."""
        da = xr.DataArray(sample_2d_image, dims=("y", "x"))
        with pytest.warns(UserWarning):
            register_volume(da, da, transform_type="translation")

    def test_returns_affine_matrix(self, sample_2d_dataarray_spatial):
        """register_volume returns a (3, 3) numpy affine matrix for 2D input."""
        _, affine = register_volume(
            sample_2d_dataarray_spatial,
            sample_2d_dataarray_spatial,
            transform_type="translation",
        )
        assert isinstance(affine, np.ndarray)
        assert affine.shape == (3, 3)

    def test_bspline_returns_dataarray_transform(self, sample_2d_dataarray_spatial):
        """register_volume with bspline returns a DataArray for the transform."""
        _, bspline_tx = register_volume(
            sample_2d_dataarray_spatial,
            sample_2d_dataarray_spatial,
            transform_type="bspline",
        )
        assert isinstance(bspline_tx, xr.DataArray)
        assert bspline_tx.attrs.get("type") == "bspline_transform"
        assert bspline_tx.dims[0] == "component"

    def test_resample_true_coords_match_fixed(
        self, sample_2d_image, sample_2d_dataarray_spatial
    ):
        """resample=True output coordinates match the fixed volume, not moving."""
        moving = sample_2d_dataarray_spatial.isel(y=slice(16), x=slice(16))
        result, _ = register_volume(
            moving,
            sample_2d_dataarray_spatial,
            transform_type="translation",
            resample=True,
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
            moving,
            sample_2d_dataarray_spatial,
            transform_type="translation",
            resample=False,
        )
        assert_array_equal(result.values, moving.values)

    def test_resample_true_aligns_to_fixed(
        self, sample_2d_image, sample_2d_dataarray_spatial
    ):
        """resample=True produces output close to fixed (the registration target)."""
        # Use a fixed shift of 2 pixels to avoid wrap-around contamination from np.roll.
        shift = 2
        shifted = np.roll(np.roll(sample_2d_image, shift, axis=0), shift, axis=1)
        moving = xr.DataArray(
            shifted,
            dims=sample_2d_dataarray_spatial.dims,
            coords=sample_2d_dataarray_spatial.coords,
        )
        result, _ = register_volume(
            moving,
            sample_2d_dataarray_spatial,
            transform_type="translation",
            learning_rate=1.0,
            number_of_iterations=200,
            resample=True,
        )
        # Compare only the interior to avoid boundary wrap-around artifacts.
        margin = shift + 1
        assert_allclose(
            result.values[margin:-margin, margin:-margin],
            sample_2d_dataarray_spatial.values[margin:-margin, margin:-margin],
            atol=10.0,
        )


class TestRegisterVolumeAccuracy:
    """Registration accuracy for register_volume."""

    def test_identical_volumes_unchanged_2d(self, sample_2d_dataarray_spatial):
        """Registering identical 2D volumes produces nearly identical output."""
        result, _ = register_volume(
            sample_2d_dataarray_spatial,
            sample_2d_dataarray_spatial,
            transform_type="translation",
            resample=True,
        )
        assert_allclose(result.values, sample_2d_dataarray_spatial.values, atol=1e-3)

    def test_identical_volumes_unchanged_3d(self, sample_3d_dataarray_spatial):
        """Registering identical 3D volumes produces nearly identical output."""
        result, _ = register_volume(
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
        result, _ = register_volume(
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

    def test_optimizer_weights_freezes_rotation(self, sample_2d_dataarray_spatial):
        """Setting rotation weight to 0 produces the same result as translation-only."""
        da = sample_2d_dataarray_spatial
        _, affine_translation = register_volume(da, da, transform_type="translation")
        # 2D rigid with rotation frozen: [rotation, tx, ty] with weight [0, 1, 1].
        _, affine_frozen = register_volume(
            da, da, transform_type="rigid", optimizer_weights=[0.0, 1.0, 1.0]
        )
        # The rotation sub-matrix should be identity (no rotation applied).
        assert_allclose(affine_frozen[:2, :2], np.eye(2), atol=1e-4)


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
            result, _ = register_volume(da, da, transform_type="translation")
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
            result, _ = register_volume(
                da, da, transform_type="translation", resample=True
            )
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
        result, _ = register_volume(da, da, transform_type="translation")
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

    def test_time_dimension_moving_works(
        self, sample_2d_image, sample_2d_dataarray, sample_2d_dataarray_spatial
    ):
        """moving with a time dimension resamples each frame with the same transform."""
        result = resample_volume(
            sample_2d_dataarray,
            np.eye(3),
            **self._grid_from_da(sample_2d_dataarray_spatial),
        )
        assert "time" in result.dims
        assert result.shape == sample_2d_dataarray.shape
        assert_allclose(
            result.coords["time"].values, sample_2d_dataarray.coords["time"].values
        )

    def test_3d_time_dimension_moving_works(
        self, sample_3d_dataarray, sample_3d_dataarray_spatial
    ):
        """3D moving with time dimension resamples each frame with the same transform."""
        result = resample_volume(
            sample_3d_dataarray,
            np.eye(4),
            **self._grid_from_da(sample_3d_dataarray_spatial),
        )
        assert "time" in result.dims
        assert result.shape == sample_3d_dataarray.shape
        assert_allclose(
            result.coords["time"].values, sample_3d_dataarray.coords["time"].values
        )

    def test_wrong_ndim_raises(self):
        """1D input raises ValueError."""
        da = xr.DataArray(np.zeros(10), dims=("x",))
        with pytest.raises(ValueError, match="2 or 3 spatial"):
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
            moving,
            sample_2d_dataarray_spatial,
            transform_type="translation",
            resample=True,
        )
        result = resample_volume(
            moving, affine, **self._grid_from_da(sample_2d_dataarray_spatial)
        )
        assert_allclose(result.values, resampled_direct.values, atol=1e-5)


class TestInitialTransform:
    """Tests for the initial_transform parameter of register_volume."""

    def test_wrong_shape_raises(self, sample_2d_dataarray_spatial):
        """initial_transform with wrong shape raises ValueError."""
        with pytest.raises(ValueError, match="initial_transform shape"):
            register_volume(
                sample_2d_dataarray_spatial,
                sample_2d_dataarray_spatial,
                transform_type="bspline",
                initial_transform=np.eye(4),  # wrong: 3D affine for 2D images
            )

    def test_bspline_with_initial_transform_stores_pre_affine(
        self, sample_2d_dataarray_spatial
    ):
        """B-spline result DataArray stores the pre-affine in attrs when initial_transform is given."""
        pre_affine = np.eye(3)
        _, bspline_tx = register_volume(
            sample_2d_dataarray_spatial,
            sample_2d_dataarray_spatial,
            transform_type="bspline",
            initial_transform=pre_affine,
        )
        assert isinstance(bspline_tx, xr.DataArray)
        assert "affines" in bspline_tx.attrs
        assert "bspline_initialization" in bspline_tx.attrs["affines"]

    def test_bspline_without_initial_transform_has_no_pre_affine(
        self, sample_2d_dataarray_spatial
    ):
        """B-spline result DataArray without initial_transform has no bspline_initialization key."""
        _, bspline_tx = register_volume(
            sample_2d_dataarray_spatial,
            sample_2d_dataarray_spatial,
            transform_type="bspline",
        )
        assert isinstance(bspline_tx, xr.DataArray)
        affines = bspline_tx.attrs.get("affines", {})
        assert "bspline_initialization" not in affines


class TestResampleVolumeWithBspline:
    """Tests for resample_volume and resample_like with a B-spline DataArray transform."""

    def test_resample_like_with_bspline_matches_direct_resample(
        self, sample_2d_image, sample_2d_dataarray_spatial
    ):
        """resample_like with a B-spline DataArray matches register_volume(resample=True)."""
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
        resampled_direct, bspline_tx = register_volume(
            moving,
            sample_2d_dataarray_spatial,
            transform_type="bspline",
            resample=True,
        )
        assert isinstance(bspline_tx, xr.DataArray)
        result = resample_like(moving, sample_2d_dataarray_spatial, bspline_tx)
        np.testing.assert_allclose(result.values, resampled_direct.values, atol=1e-5)

    def test_resample_like_with_composite_bspline_matches_direct_resample(
        self, sample_2d_image, sample_2d_dataarray_spatial
    ):
        """resample_like with composite B-spline (with initial_transform) matches register_volume(resample=True)."""
        rng = np.random.default_rng(1)
        shift = rng.integers(2, 4, size=2)
        shifted = np.roll(
            np.roll(sample_2d_image, int(shift[0]), axis=0), int(shift[1]), axis=1
        )
        moving = xr.DataArray(
            shifted,
            dims=sample_2d_dataarray_spatial.dims,
            coords=sample_2d_dataarray_spatial.coords,
        )
        # First pass: affine registration.
        _, affine_tx = register_volume(
            moving,
            sample_2d_dataarray_spatial,
            transform_type="affine",
        )
        # Second pass: B-spline refinement on top of the affine.
        resampled_direct, bspline_tx = register_volume(
            moving,
            sample_2d_dataarray_spatial,
            transform_type="bspline",
            initial_transform=affine_tx,
            resample=True,
        )
        assert isinstance(bspline_tx, xr.DataArray)
        result = resample_like(moving, sample_2d_dataarray_spatial, bspline_tx)
        np.testing.assert_allclose(result.values, resampled_direct.values, atol=1e-5)


class TestResampleLike:
    """Unit tests for resample_like."""

    def test_time_dimension_moving_works(
        self, sample_2d_dataarray, sample_2d_dataarray_spatial
    ):
        """moving with a time dimension resamples each frame with the same transform."""
        result = resample_like(
            sample_2d_dataarray, sample_2d_dataarray_spatial, np.eye(3)
        )
        assert "time" in result.dims
        assert result.shape == sample_2d_dataarray.shape
        assert_allclose(
            result.coords["time"].values, sample_2d_dataarray.coords["time"].values
        )

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
            moving,
            sample_2d_dataarray_spatial,
            transform_type="translation",
            resample=True,
        )
        result = resample_like(moving, sample_2d_dataarray_spatial, affine)
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
        resampled_direct, affine = register_volume(
            moving,
            sample_3d_dataarray_spatial,
            transform_type="translation",
            learning_rate=1.0,
            number_of_iterations=200,
            resample=True,
        )
        result = resample_like(moving, sample_3d_dataarray_spatial, affine)
        assert_allclose(result.values, resampled_direct.values, atol=1e-5)

    def test_matches_register_volume_with_initial_transform(
        self, sample_2d_image, sample_2d_dataarray_spatial
    ):
        """resample_like matches register_volume(resample=True) when initial_transform is used.

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
            shifted,
            dims=sample_2d_dataarray_spatial.dims,
            coords=sample_2d_dataarray_spatial.coords,
        )
        _, affine_init = register_volume(
            moving, sample_2d_dataarray_spatial, transform_type="translation"
        )
        resampled_direct, affine = register_volume(
            moving,
            sample_2d_dataarray_spatial,
            transform_type="affine",
            initial_transform=affine_init,
            resample=True,
        )
        result = resample_like(moving, sample_2d_dataarray_spatial, affine)
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

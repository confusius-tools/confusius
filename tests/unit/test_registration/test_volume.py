"""Unit tests for single-volume registration."""

from threading import Event

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal

from confusius._dims import SPATIAL_DIMS, VOXEL_DIMS
from confusius._utils.coordinates import get_grid_info_from_dataarray
from confusius._utils.geometry import (
    get_affine_orientation_matrix,
    get_voxel_to_world_affine,
)
from confusius.registration._utils import (
    build_voxel_to_world_plane_initial_transform,
    dataarray_to_sitk_image,
    get_defined_spatial_spacing,
)
from confusius.registration.bspline import (
    invert_displacement_field,
    sample_displacement_field,
    sample_displacement_field_like,
    sitk_bspline_to_dataarray,
)
from confusius.registration.diagnostics import RegistrationDiagnostics
from confusius.registration.resampling import resample_like, resample_volume
from confusius.registration.volume import register_volume
from confusius.xarray import create_fusi_dataarray


def _make_voxel_to_world_2d() -> xr.DataArray:
    """Create a small singleton-k voxel-to-world test image."""
    yy, xx = np.mgrid[-1.0:1.0:32j, -1.0:1.0:40j]
    values = np.exp(-((xx - 0.2) ** 2 + (yy + 0.1) ** 2) / 0.15).astype(np.float32)
    return create_fusi_dataarray(
        values[np.newaxis],
        dims=("k", "j", "i"),
        voxel_to_world=np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.2, 0.05, 10.0],
                [0.0, 0.08, 0.18, 20.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )


def _make_voxel_to_world_3d() -> xr.DataArray:
    """Create a small singleton-k voxel-to-world test image with an oblique j/i plane."""
    return _make_voxel_to_world_2d()


def _resample_volume_grid_kwargs(data: xr.DataArray) -> dict:
    """Build `resample_volume`'s position-anchored output-grid kwargs."""
    _, spacing = get_defined_spatial_spacing(data)
    origin = data.fusi.origin
    return {
        "output_shape": [int(data.sizes[dim]) for dim in VOXEL_DIMS],
        "output_spacing": spacing,
        "output_origin": [origin[name] for name in SPATIAL_DIMS],
        "output_direction": get_affine_orientation_matrix(
            get_voxel_to_world_affine(data)
        ),
    }


def _add_identity_voxel_to_world(data: xr.DataArray) -> xr.DataArray:
    """Attach axis-aligned voxel-to-world geometry to a test array."""
    extra_coords = {
        str(dim): data.coords[dim]
        for dim in data.dims
        if dim not in {*VOXEL_DIMS, "time", "pose"} and dim in data.coords
    }
    spacing = []
    origin = []
    for dim in VOXEL_DIMS:
        if dim not in data.coords:
            spacing.append(1.0)
            origin.append(0.0)
            continue
        values = np.asarray(data.coords[dim].values, dtype=np.float64)
        origin.append(float(values[0]))
        spacing.append(float(values[1] - values[0]) if values.size > 1 else 1.0)
    return create_fusi_dataarray(
        data.values,
        dims=tuple(str(dim) for dim in data.dims),
        time=data.coords.get("time"),
        pose=data.coords.get("pose"),
        extra_coords=extra_coords,
        spacing=spacing,
        origin=origin,
        attrs=data.attrs.copy(),
        name=str(data.name) if data.name is not None else None,
    )


def _make_voxel_to_world_3d_slab() -> xr.DataArray:
    """Create a small 3D voxel-to-world slab with a singleton slice dimension."""
    return create_fusi_dataarray(
        np.zeros((1, 5, 6), dtype=np.float32),
        dims=("k", "j", "i"),
        voxel_to_world=np.array(
            [
                [0.4, 0.0, 0.0, 10.0],
                [0.0, 2.0, 0.0, 20.0],
                [0.0, 0.0, 3.0, 30.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
    )


def _make_voxel_to_world_3d_slab_flipped_normal() -> xr.DataArray:
    """Voxel-to-world slab like `_make_voxel_to_world_3d_slab` with the k-axis normal flipped."""
    return create_fusi_dataarray(
        np.zeros((1, 5, 6), dtype=np.float32),
        dims=("k", "j", "i"),
        voxel_to_world=np.array(
            [
                [-0.4, 0.0, 0.0, 10.0],
                [0.0, 2.0, 0.0, 20.0],
                [0.0, 0.0, 3.0, 30.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
    )


class TestRegisterVolumeValidation:
    """Input validation for register_volume."""

    def test_time_dimension_raises(self, sample_fusi_2dt_registration):
        """DataArray with a time dimension raises ValueError."""
        with pytest.raises(ValueError, match="spatial-only"):
            register_volume(sample_fusi_2dt_registration, sample_fusi_2dt_registration)

    def test_nan_in_moving_raises(self, sample_fusi_2d_registration):
        """moving with NaN values raises ValueError."""
        moving = sample_fusi_2d_registration.copy()
        moving.values[0, 0] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            register_volume(
                moving,
                sample_fusi_2d_registration,
                transform_type="translation",
            )

    def test_nan_in_fixed_raises(self, sample_fusi_2d_registration):
        """fixed with NaN values raises ValueError."""
        fixed = sample_fusi_2d_registration.copy()
        fixed.values[0, 0] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            register_volume(
                sample_fusi_2d_registration, fixed, transform_type="translation"
            )

    def test_wrong_ndim_1d_raises(self):
        """1D input raises ValueError."""
        da = xr.DataArray(np.zeros(10), dims=("i",), coords={"i": np.arange(10)})
        with pytest.raises(
            ValueError,
            match="native voxel dimensions|at least 2 spatial dimensions|defined spatial spacing",
        ):
            register_volume(da, da)

    def test_wrong_ndim_4d_raises(self):
        """4D input raises ValueError."""
        da = xr.DataArray(np.zeros((4, 4, 4, 4)), dims=("a", "b", "c", "d"))
        with pytest.raises(ValueError, match="at most 3 spatial dimensions"):
            register_volume(da, da)

    def test_invalid_initialization_raises(self, sample_fusi_2d_registration):
        """Unknown initialization mode raises ValueError."""
        with pytest.raises(ValueError, match="Invalid initialization"):
            register_volume(
                sample_fusi_2d_registration,
                sample_fusi_2d_registration,
                initialization="moments",  # ty: ignore[invalid-argument-type]
            )

    def test_non_array_initialization_raises_value_error(
        self, sample_fusi_2d_registration
    ):
        """A non-ndarray sequence raises ValueError, not an unhashable TypeError."""
        with pytest.raises(ValueError, match="Invalid initialization"):
            register_volume(
                sample_fusi_2d_registration,
                sample_fusi_2d_registration,
                initialization=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # ty: ignore[invalid-argument-type]
            )

    def test_invalid_learning_rate_raises(self, sample_fusi_2d_registration):
        """A non-positive learning_rate raises ValueError."""
        with pytest.raises(ValueError, match="learning_rate must be a positive"):
            register_volume(
                sample_fusi_2d_registration,
                sample_fusi_2d_registration,
                learning_rate=-1.0,
            )

    def test_shape_mismatch_no_error(self, sample_fusi_2d_registration):
        """Different shapes do not raise an error."""
        moving = sample_fusi_2d_registration.isel(j=slice(16), i=slice(16))
        result, _, _ = register_volume(
            moving,
            sample_fusi_2d_registration,
            transform_type="translation",
            resample=False,
        )
        assert result.shape == moving.shape

    def test_abort_event_returns_partial_result(self, sample_fusi_2d_registration):
        """A pre-set abort event returns an aborted diagnostics record."""
        abort_event = Event()
        abort_event.set()

        result, _transform, diagnostics = register_volume(
            sample_fusi_2d_registration,
            sample_fusi_2d_registration,
            transform_type="translation",
            abort_event=abort_event,
        )

        assert result.shape == sample_fusi_2d_registration.shape
        assert diagnostics.status == "aborted"
        assert diagnostics.n_iterations == 0

    def test_unknown_runtime_error_is_passed_through(
        self, sample_fusi_2d_registration, monkeypatch
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
                sample_fusi_2d_registration,
                sample_fusi_2d_registration,
                transform_type="translation",
            )

        assert excinfo.value is error

    def test_bspline_scale_error_raises_clearer_message(
        self, sample_fusi_2d_registration, monkeypatch
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
                sample_fusi_2d_registration,
                sample_fusi_2d_registration,
                transform_type="bspline",
                learning_rate=1.0,
            )

    def test_bspline_scale_error_with_auto_learning_rate_suggests_fixed_rate(
        self, sample_fusi_2d_registration, monkeypatch
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
                sample_fusi_2d_registration,
                sample_fusi_2d_registration,
                transform_type="bspline",
                learning_rate="auto",
            )

    def test_mismatched_spatial_units_raise(self, sample_fusi_2d_registration):
        """moving and fixed must agree on spatial coordinate units when declared."""
        moving = sample_fusi_2d_registration.copy()
        fixed = sample_fusi_2d_registration.copy()
        moving.coords["y"].attrs["units"] = "mm"
        moving.coords["x"].attrs["units"] = "mm"
        fixed.coords["y"].attrs["units"] = "um"
        fixed.coords["x"].attrs["units"] = "um"

        with pytest.raises(ValueError, match="units"):
            register_volume(moving, fixed, transform_type="translation")


class TestSimpleITKGeometry:
    """SimpleITK conversion preserves ConfUSIus spatial geometry."""

    def test_dataarray_to_sitk_image_sets_voxel_to_world_origin_spacing_direction(self):
        """Voxel-to-world DataArrays map to SimpleITK origin/spacing/direction."""
        data = _make_voxel_to_world_2d()

        image = dataarray_to_sitk_image(data)

        assert_allclose(image.GetOrigin(), (0.0, 10.0, 20.0))
        assert_allclose(
            image.GetSpacing(),
            (1.0, np.hypot(0.2, 0.08), np.hypot(0.05, 0.18)),
        )
        assert_allclose(
            np.array(image.GetDirection()).reshape(3, 3),
            data.fusi.direction,
        )

    def test_dataarray_to_sitk_image_without_voxel_to_world_index_raises(self):
        """A DataArray without a `VoxelToWorldIndex` is not a supported input.

        ConfUSIus has a single canonical spatial data shape: `k`/`j`/`i` voxel dims
        backed by a `VoxelToWorldIndex`. Plain `z`/`y`/`x` coordinates with no index
        attached are not a supported fallback shape, so `dataarray_to_sitk_image`
        must raise instead of silently deriving a grid from the coordinates.
        """
        da = xr.DataArray(
            np.zeros((3, 4, 5), dtype=np.float32),
            dims=("z", "y", "x"),
            coords={
                "z": np.arange(3, dtype=np.float64) * 1.0,
                "y": np.arange(4, dtype=np.float64) * 0.2,
                "x": np.arange(5, dtype=np.float64) * 0.3,
            },
        )

        with pytest.raises(ValueError, match=r"fusi\.affine\.set_voxel_to_world"):
            dataarray_to_sitk_image(da)

    def test_undefined_voxel_to_world_spacing_raises_repair_hint(self):
        """An irregular voxel-to-world coordinate raises a `voxdim`-repair error."""
        da = create_fusi_dataarray(
            np.zeros((1, 4, 5)),
            dims=("k", "j", "i"),
            voxel_to_world=np.eye(4),
        ).assign_coords(j=np.array([0.0, 1.0, 3.0, 6.0]))
        with pytest.raises(ValueError, match="voxdim"):
            get_defined_spatial_spacing(da)


class TestRegisterVolumeOutput:
    """Output properties for register_volume."""

    def test_without_coords_raises(self, sample_fusi_2d_registration):
        """DataArray without coordinates is rejected."""
        da = xr.DataArray(sample_fusi_2d_registration.values, dims=("k", "j", "i"))
        with pytest.raises(
            ValueError,
            match="native voxel dimensions|at least 2 spatial dimensions|defined spatial spacing",
        ):
            register_volume(da, da, transform_type="translation")

    def test_returns_affine_matrix(self, sample_fusi_2d_registration):
        """register_volume returns a (4, 4) numpy affine matrix."""
        _, affine, _ = register_volume(
            sample_fusi_2d_registration,
            sample_fusi_2d_registration,
            transform_type="translation",
        )
        assert isinstance(affine, np.ndarray)
        assert affine.shape == (4, 4)

    def test_bspline_returns_dataarray_transform(self, sample_fusi_2d_registration):
        """register_volume with bspline returns a DataArray for the transform."""
        _, bspline_tx, _ = register_volume(
            sample_fusi_2d_registration,
            sample_fusi_2d_registration,
            transform_type="bspline",
        )
        assert isinstance(bspline_tx, xr.DataArray)
        assert bspline_tx.attrs.get("type") == "bspline_transform"
        assert bspline_tx.dims[0] == "component"
        np.testing.assert_array_equal(
            bspline_tx.coords["component"].values, ["k", "j", "i"]
        )

    def test_bspline_control_point_domain_matches_each_axis_extent(self):
        """Each axis's control-point domain scales with its own world extent.

        Regression test for a bug where `sitk_bspline_to_dataarray` assumed
        SimpleITK reverses axis order relative to the DataArray (`(x, y, z)` vs.
        `(z, y, x)`), when in fact this codebase's convention (see
        `dataarray_to_sitk_image`) never reverses axes: sitk axis `i` maps directly
        to DataArray dim `i`. On an anisotropic image, the erroneous reversal
        swapped the y/x control-point grids: `y`'s spacing was computed from `x`'s
        world domain and vice versa. Isotropic test fixtures never exposed this
        because swapping equal-sized, equal-spacing axes is a no-op.
        """
        img = np.zeros((1, 20, 40), dtype=np.float32)
        img[:, 6:14, 10:30] = 100.0
        da = create_fusi_dataarray(img, dims=("k", "j", "i"), spacing=(1.0, 0.5, 0.1))
        _, bspline_tx, _ = register_volume(
            da,
            da,
            transform_type="bspline",
            mesh_size=(1, 4, 4),
        )

        y_span = float(bspline_tx.fusi.spacing["j"] * (bspline_tx.sizes["j"] - 1))
        x_span = float(bspline_tx.fusi.spacing["i"] * (bspline_tx.sizes["i"] - 1))
        # The control-point domain is padded beyond the image FOV for boundary
        # support, so spans are somewhat larger than the raw world extent (9.5 mm
        # for y, 3.9 mm for x). Padding scales with each axis's own extent (same mesh
        # size, so padding is proportional to domain size), so the span ratio should
        # track the world extent ratio (9.5 / 3.9 ~= 2.44) rather than being
        # swapped with the other axis's.
        assert y_span / x_span == pytest.approx(9.5 / 3.9, rel=0.3)

    def test_resample_true_coords_match_fixed(self, sample_fusi_2d_registration):
        """resample=True output coordinates match the fixed volume, not moving."""
        moving = sample_fusi_2d_registration.isel(j=slice(16), i=slice(16))
        result, _, _ = register_volume(
            moving,
            sample_fusi_2d_registration,
            transform_type="translation",
            resample=True,
        )
        assert_allclose(
            result.coords["j"].values,
            sample_fusi_2d_registration.coords["j"].values,
        )
        assert_allclose(
            result.coords["i"].values,
            sample_fusi_2d_registration.coords["i"].values,
        )

    def test_resample_true_inherits_fixed_affines(self, sample_fusi_2d_registration):
        """resample=True output inherits world-space affines from `fixed`."""
        moving = sample_fusi_2d_registration.isel(j=slice(16), i=slice(16)).copy()
        fixed = sample_fusi_2d_registration.copy()
        moving.attrs["affines"] = {"world_to_lab": np.diag([2.0, 2.0, 1.0])}
        fixed.attrs["affines"] = {"world_to_lab": np.diag([3.0, 3.0, 1.0])}

        result, _, _ = register_volume(
            moving,
            fixed,
            transform_type="translation",
            resample=True,
        )

        assert "registration" not in result.attrs
        assert_allclose(
            result.attrs["affines"]["world_to_lab"],
            fixed.attrs["affines"]["world_to_lab"],
        )

    def test_resample_true_inherits_fixed_voxel_to_world_geometry(self):
        """resample=True output inherits voxel-to-world geometry from the fixed grid."""
        moving = _make_voxel_to_world_3d()
        fixed = _make_voxel_to_world_3d()

        result, _, _ = register_volume(
            moving,
            fixed,
            transform_type="translation",
            resample=True,
        )

        assert_allclose(
            get_voxel_to_world_affine(result), get_voxel_to_world_affine(fixed)
        )
        assert type(result.xindexes["x"]).__name__ == "VoxelToWorldIndex"
        assert result.coords["i"].dims == fixed.coords["i"].dims


class TestRegisterVolumeMask:
    """Metric masks for register_volume."""

    def test_integer_label_mask_matches_boolean_mask(self, sample_fusi_2d_registration):
        """A single-label integer mask registers identically to its boolean form.

        Guards against single-label integer masks (e.g. `{0, 512}` from
        `Atlas.get_masks`) reaching SimpleITK's metric mask uncoerced: 512 wraps to 0
        under the `numpy.uint8` cast, which silently empties the mask and turns
        registration into a no-op.
        """
        shift = 2
        shifted = np.roll(
            np.roll(sample_fusi_2d_registration.values, shift, axis=1),
            shift,
            axis=2,
        )
        fixed = sample_fusi_2d_registration
        moving = xr.DataArray(
            shifted,
            dims=fixed.dims,
            coords=fixed.coords,
            attrs=fixed.attrs,
        )

        region = np.zeros(fixed.shape, dtype=bool)
        region[:, 4:28, 4:28] = True  # covers the bright square in both volumes
        bool_mask = xr.DataArray(
            region, dims=fixed.dims, coords=fixed.coords, attrs=fixed.attrs
        )
        # 512 is a multiple of 256: a uint8 cast of the raw integer mask wraps it to 0.
        int_mask = xr.DataArray(
            region.astype(np.int32) * 512,
            dims=fixed.dims,
            coords=fixed.coords,
            attrs=fixed.attrs,
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

    def test_both_masks_coerced_to_bool(self, sample_fusi_2d_registration):
        """Both fixed_mask and moving_mask are coerced to boolean."""
        shift = 2
        shifted = np.roll(
            np.roll(sample_fusi_2d_registration.values, shift, axis=1),
            shift,
            axis=2,
        )
        fixed = sample_fusi_2d_registration
        moving = xr.DataArray(
            shifted,
            dims=fixed.dims,
            coords=fixed.coords,
            attrs=fixed.attrs,
        )

        region = np.zeros(fixed.shape, dtype=bool)
        region[:, 4:28, 4:28] = True
        fixed_mask = xr.DataArray(
            region, dims=fixed.dims, coords=fixed.coords, attrs=fixed.attrs
        )
        moving_mask = xr.DataArray(
            region, dims=fixed.dims, coords=fixed.coords, attrs=fixed.attrs
        )

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
        self, sample_fusi_2d_registration
    ):
        """resample=False returns moving values without modification."""
        rng = np.random.default_rng(0)
        shift = rng.integers(1, 4, size=2)
        shifted = np.roll(
            np.roll(sample_fusi_2d_registration.values, int(shift[0]), axis=1),
            int(shift[1]),
            axis=2,
        )
        moving = xr.DataArray(
            shifted,
            dims=sample_fusi_2d_registration.dims,
            coords=sample_fusi_2d_registration.coords,
            attrs=sample_fusi_2d_registration.attrs,
        )
        result, _, _ = register_volume(
            moving,
            sample_fusi_2d_registration,
            transform_type="translation",
            resample=False,
        )
        assert_array_equal(result.values, moving.values)

    def test_resample_true_aligns_to_fixed(self, sample_fusi_2d_registration):
        """resample=True produces output close to fixed (the registration target)."""
        # Use a fixed shift of 2 pixels to avoid wrap-around contamination from np.roll.
        shift = 2
        shifted = np.roll(
            np.roll(sample_fusi_2d_registration.values, shift, axis=1),
            shift,
            axis=2,
        )
        moving = xr.DataArray(
            shifted,
            dims=sample_fusi_2d_registration.dims,
            coords=sample_fusi_2d_registration.coords,
            attrs=sample_fusi_2d_registration.attrs,
        )
        result, _, _ = register_volume(
            moving,
            sample_fusi_2d_registration,
            transform_type="translation",
            learning_rate=1.0,
            number_of_iterations=200,
            resample=True,
        )
        # Compare only the interior to avoid boundary wrap-around artifacts.
        margin = shift + 1
        assert_allclose(
            result.values[:, margin:-margin, margin:-margin],
            sample_fusi_2d_registration.values[:, margin:-margin, margin:-margin],
            atol=10.0,
        )


class TestRegisterVolumeAccuracy:
    """Registration accuracy for register_volume."""

    def test_identical_volumes_unchanged_2d(self, sample_fusi_2d_registration):
        """Registering identical 2D volumes produces nearly identical output."""
        result, _, _ = register_volume(
            sample_fusi_2d_registration,
            sample_fusi_2d_registration,
            transform_type="translation",
            resample=True,
        )
        assert_allclose(result.values, sample_fusi_2d_registration.values, atol=1e-3)

    def test_identical_volumes_unchanged_3d(self, sample_fusi_3d_registration):
        """Registering identical 3D volumes produces nearly identical output."""
        result, _, _ = register_volume(
            sample_fusi_3d_registration,
            sample_fusi_3d_registration,
            transform_type="translation",
            resample=True,
        )
        assert_allclose(result.values, sample_fusi_3d_registration.values, atol=1e-3)

    def test_3d_recovers_known_shift(self, sample_fusi_3d_registration):
        """Registration recovers a known 3D translation."""
        shifted = np.roll(sample_fusi_3d_registration.values, 2, axis=0)
        spacing = (1.0, 1.0, 1.0)
        fixed = _add_identity_voxel_to_world(
            xr.DataArray(
                sample_fusi_3d_registration.values,
                dims=("k", "j", "i"),
                coords={
                    d: np.arange(sample_fusi_3d_registration.values.shape[i])
                    * spacing[i]
                    for i, d in enumerate(("k", "j", "i"))
                },
            )
        )
        moving = xr.DataArray(
            shifted, dims=fixed.dims, coords=fixed.coords, attrs=fixed.attrs
        )
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

    def test_optimizer_weights_freezes_rotation(self, sample_fusi_2d_registration):
        """Setting rotation weight to 0 produces the same result as translation-only."""
        da = sample_fusi_2d_registration
        _, _affine_translation, _ = register_volume(
            da, da, transform_type="translation"
        )
        # 3D rigid with rotation frozen: [rx, ry, rz, tx, ty, tz] with weight [0,0,0,1,1,1].
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
        da = _add_identity_voxel_to_world(
            xr.DataArray(
                arr,
                dims=("k", "j", "i"),
                coords={
                    "k": np.array([0.0]),
                    "j": np.arange(32) * 0.1,
                    "i": np.arange(32) * 0.1,
                },
            )
        )
        result, _, _ = register_volume(da, da, transform_type="translation")
        assert result.shape == da.shape

    def test_3d_volume_with_depth_1_preserves_output_shape_on_resample(self):
        """resample=True preserves the original shape for a depth-1 volume."""
        arr = np.zeros((1, 32, 32), dtype=np.float32)
        arr[0, 12:20, 12:20] = 1.0
        da = _add_identity_voxel_to_world(
            xr.DataArray(
                arr,
                dims=("k", "j", "i"),
                coords={
                    "k": np.array([0.0]),
                    "j": np.arange(32) * 0.1,
                    "i": np.arange(32) * 0.1,
                },
            )
        )
        result, _, _ = register_volume(
            da, da, transform_type="translation", resample=True
        )
        assert result.shape == da.shape

    def test_float32_moving_float64_fixed_does_not_crash(
        self, sample_fusi_2d_registration
    ):
        """float32 moving and float64 fixed register without a dtype mismatch error.

        Regression test: CenteredTransformInitializer requires both images to share the
        same pixel type. Mixed dtypes (e.g. float32 template vs. float64 mean of NIfTI
        data) previously raised a RuntimeError.
        """
        moving = sample_fusi_2d_registration  # float32
        fixed = sample_fusi_2d_registration.astype(np.float64)
        result, _, _ = register_volume(moving, fixed, transform_type="translation")
        assert result.shape == fixed.shape

    def test_3d_volume_with_depth_2_does_not_crash(self):
        """3D volume with depth=2 (below the 4-voxel threshold) registers without error."""
        arr = np.zeros((2, 16, 16), dtype=np.float32)
        arr[:, 6:10, 6:10] = 1.0
        da = _add_identity_voxel_to_world(
            xr.DataArray(
                arr,
                dims=("k", "j", "i"),
                coords={
                    "k": np.arange(2) * 0.5,
                    "j": np.arange(16) * 0.1,
                    "i": np.arange(16) * 0.1,
                },
            )
        )
        result, _, _ = register_volume(da, da, transform_type="translation")
        assert result.shape == da.shape


class TestResampleVolume:
    """Unit tests for the low-level resample_volume."""

    def test_time_dimension_moving_works(
        self, sample_fusi_2d_registration, sample_fusi_2dt_registration
    ):
        """moving with a time dimension resamples each frame with the same transform."""
        result = resample_volume(
            sample_fusi_2dt_registration,
            np.eye(4),
            **_resample_volume_grid_kwargs(sample_fusi_2d_registration),
        )
        assert "time" in result.dims
        assert result.shape == sample_fusi_2dt_registration.shape
        assert_allclose(
            result.coords["time"].values,
            sample_fusi_2dt_registration.coords["time"].values,
        )

    def test_3d_time_dimension_moving_works(
        self, sample_fusi_3dt_registration, sample_fusi_3d_registration
    ):
        """3D moving with time dimension resamples each frame with the same transform."""
        result = resample_volume(
            sample_fusi_3dt_registration,
            np.eye(4),
            **_resample_volume_grid_kwargs(sample_fusi_3d_registration),
        )
        assert "time" in result.dims
        assert result.shape == sample_fusi_3dt_registration.shape
        assert_allclose(
            result.coords["time"].values,
            sample_fusi_3dt_registration.coords["time"].values,
        )

    def test_wrong_ndim_raises(self):
        """1D input raises ValueError."""
        da = xr.DataArray(np.zeros(10), dims=("i",), coords={"i": np.arange(10)})
        with pytest.raises(
            ValueError,
            match="native voxel dimensions|at least 2 spatial dimensions|defined spatial spacing",
        ):
            resample_volume(
                da,
                np.eye(2),
                output_shape=[10, 10, 10],
                output_spacing=[1.0, 1.0, 1.0],
                output_origin=[0.0, 0.0, 0.0],
            )

    def test_affine_shape_mismatch_raises(self, sample_fusi_2d_registration):
        """Affine with wrong shape raises ValueError."""
        with pytest.raises(ValueError, match="affine shape"):
            resample_volume(
                sample_fusi_2d_registration,
                np.eye(3),
                **_resample_volume_grid_kwargs(sample_fusi_2d_registration),
            )

    def test_output_shape_wrong_length_raises(self, sample_fusi_2d_registration):
        """output_shape with the wrong number of entries raises ValueError."""
        with pytest.raises(ValueError, match="output_shape must have"):
            resample_volume(
                sample_fusi_2d_registration,
                np.eye(4),
                output_shape=[10, 10],
                output_spacing=[1.0, 1.0, 1.0],
                output_origin=[0.0, 0.0, 0.0],
            )

    def test_output_direction_wrong_shape_raises(self, sample_fusi_2d_registration):
        """output_direction with the wrong shape raises ValueError."""
        with pytest.raises(ValueError, match="output_direction must have shape"):
            resample_volume(
                sample_fusi_2d_registration,
                np.eye(4),
                output_shape=[10, 10, 10],
                output_spacing=[1.0, 1.0, 1.0],
                output_origin=[0.0, 0.0, 0.0],
                output_direction=np.eye(2),
            )

    def test_output_shape_matches_requested_shape(self, sample_fusi_2d_registration):
        """Output shape matches the requested shape, not the moving shape."""
        moving = sample_fusi_2d_registration.isel(j=slice(16), i=slice(16))
        result = resample_volume(
            moving,
            np.eye(4),
            **_resample_volume_grid_kwargs(sample_fusi_2d_registration),
        )
        assert result.shape == sample_fusi_2d_registration.shape

    def test_coords_reconstructed_from_origin_and_spacing(
        self, sample_fusi_2d_registration
    ):
        """Output geometry is reconstructed from the requested spacing/origin/direction."""
        grid = _resample_volume_grid_kwargs(sample_fusi_2d_registration)
        result = resample_volume(sample_fusi_2d_registration, np.eye(4), **grid)
        for dim, size in zip(VOXEL_DIMS, grid["output_shape"], strict=True):
            assert result.sizes[dim] == size
        assert_allclose(
            [result.fusi.spacing[dim] for dim in VOXEL_DIMS], grid["output_spacing"]
        )
        assert_allclose(
            [result.fusi.origin[name] for name in SPATIAL_DIMS], grid["output_origin"]
        )

    def test_matches_register_volume_resample(self, sample_fusi_2d_registration):
        """resample_volume matches register_volume(resample=True) on a shifted image."""
        rng = np.random.default_rng(42)
        shift = rng.integers(3, 6, size=2)
        shifted = np.roll(
            np.roll(sample_fusi_2d_registration.values, int(shift[0]), axis=1),
            int(shift[1]),
            axis=2,
        )
        moving = xr.DataArray(
            shifted,
            dims=sample_fusi_2d_registration.dims,
            coords=sample_fusi_2d_registration.coords,
            attrs=sample_fusi_2d_registration.attrs,
        )
        resampled_direct, affine, _ = register_volume(
            moving,
            sample_fusi_2d_registration,
            transform_type="translation",
            resample=True,
        )
        result = resample_volume(
            moving,
            affine,
            **_resample_volume_grid_kwargs(sample_fusi_2d_registration),
        )
        assert_allclose(result.values, resampled_direct.values, atol=1e-5)


class TestInitialization:
    """Tests for the initialization parameter of register_volume."""

    def test_wrong_shape_raises(self, sample_fusi_2d_registration):
        """Affine initialization with wrong shape raises ValueError."""
        with pytest.raises(ValueError, match="initialization shape"):
            register_volume(
                sample_fusi_2d_registration,
                sample_fusi_2d_registration,
                transform_type="bspline",
                initialization=np.eye(3),  # wrong: 2D affine for 3D images
            )

    def test_plane_initializer_aligns_voxel_to_world_slabs(self):
        """The voxel-to-world slab initializer rotates and translates planes into coincidence."""
        fixed = _make_voxel_to_world_3d_slab()
        rotation = np.array(
            [
                [np.cos(np.deg2rad(2.5)), -np.sin(np.deg2rad(2.5)), 0.0, 0.0],
                [np.sin(np.deg2rad(2.5)), np.cos(np.deg2rad(2.5)), 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        translation = np.eye(4, dtype=np.float64)
        translation[:3, 3] = [-25.0, 4.0, 7.5]
        expected_transform = translation @ rotation
        moving = fixed.fusi.affine.apply(expected_transform)

        initial_transform = build_voxel_to_world_plane_initial_transform(fixed, moving)
        seeded = moving.fusi.affine.apply(np.linalg.inv(initial_transform))

        assert_allclose(initial_transform, expected_transform, atol=1e-10)
        assert_allclose(seeded.fusi.direction, fixed.fusi.direction, atol=1e-10)
        assert_allclose(seeded.coords["k"].values, fixed.coords["k"].values, atol=1e-10)
        assert_allclose(seeded.coords["j"].values, fixed.coords["j"].values, atol=1e-10)
        assert_allclose(seeded.coords["i"].values, fixed.coords["i"].values, atol=1e-10)

    def test_plane_initializer_identity_rotation_for_parallel_normals(self):
        """Parallel slice normals use the identity-rotation shortcut."""
        fixed = _make_voxel_to_world_3d_slab()
        moving = _make_voxel_to_world_3d_slab()

        initial_transform = build_voxel_to_world_plane_initial_transform(fixed, moving)

        assert_allclose(initial_transform[:3, :3], np.eye(3), atol=1e-10)
        assert_allclose(initial_transform[:3, 3], 0.0, atol=1e-10)

    def test_plane_initializer_flips_antiparallel_normals(self):
        """Antiparallel slice normals fall back to the axis-flip rotation formula."""
        fixed = _make_voxel_to_world_3d_slab()
        moving = _make_voxel_to_world_3d_slab_flipped_normal()

        initial_transform = build_voxel_to_world_plane_initial_transform(fixed, moving)
        rotation = initial_transform[:3, :3]

        # The rotation must be a proper rotation (orthogonal, determinant 1) that maps
        # fixed's plane back onto moving's flipped plane.
        assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-10)
        assert_allclose(np.linalg.det(rotation), 1.0, atol=1e-10)
        seeded = moving.fusi.affine.apply(np.linalg.inv(initial_transform))
        assert_allclose(
            seeded.fusi.direction[:, 0], fixed.fusi.direction[:, 0], atol=1e-10
        )

    def test_plane_initializer_requires_single_singleton_dim(self):
        """A voxel-to-world slab without exactly one singleton spatial axis is rejected."""
        base = xr.DataArray(
            np.zeros((2, 5, 6), dtype=np.float32),
            dims=("k", "j", "i"),
            coords={
                "k": np.arange(2, dtype=np.float64),
                "j": np.arange(5, dtype=np.float64),
                "i": np.arange(6, dtype=np.float64),
            },
        )
        fixed = _add_identity_voxel_to_world(base)
        with pytest.raises(ValueError, match="exactly one singleton"):
            build_voxel_to_world_plane_initial_transform(fixed, fixed)

    def test_plane_initializer_requires_voxel_to_world_geometry(self):
        """Non-voxel-to-world inputs (plain z/y/x dims, no voxel_to_world index) are rejected."""
        da = xr.DataArray(
            np.zeros((1, 5, 6), dtype=np.float32),
            dims=("z", "y", "x"),
            coords={
                "z": np.array([0.0]),
                "y": np.arange(5, dtype=np.float64),
                "x": np.arange(6, dtype=np.float64),
            },
        )
        with pytest.raises(ValueError, match="Unexpected dimensions"):
            build_voxel_to_world_plane_initial_transform(da, da)

    def test_plane_initializer_rejects_extra_dims(self):
        """Plane initialization only accepts spatial-only canonical DataArrays."""
        da = create_fusi_dataarray(
            np.zeros((2, 1, 5, 6), dtype=np.float32),
            dims=("component", "k", "j", "i"),
            extra_coords={"component": ["a", "b"]},
            voxel_to_world=np.eye(4),
        )
        with pytest.raises(ValueError, match="Unexpected dimensions"):
            build_voxel_to_world_plane_initial_transform(da, da)

    def test_linear_initial_transform_is_not_shifted_by_geometry_centering(self):
        """A supplied initial affine is used directly, without extra centering shift."""
        fixed = _add_identity_voxel_to_world(
            xr.DataArray(
                np.arange(16, dtype=np.float32).reshape(1, 4, 4),
                dims=("k", "j", "i"),
                coords={
                    "k": np.arange(1, dtype=np.float64),
                    "j": np.arange(4, dtype=np.float64),
                    "i": np.arange(4, dtype=np.float64),
                },
            )
        )
        moving = _add_identity_voxel_to_world(
            xr.DataArray(
                fixed.values.copy(),
                dims=fixed.dims,
                coords={
                    "k": fixed.coords["k"].values,
                    "j": fixed.coords["j"].values + 10.0,
                    "i": fixed.coords["i"].values + 20.0,
                },
            )
        )
        initial_transform = np.eye(4, dtype=np.float64)
        initial_transform[1:3, 3] = [20.0, 10.0]

        _, transform, _ = register_volume(
            moving,
            fixed,
            transform_type="affine",
            initialization=initial_transform,
            optimizer_weights=[0.0] * 12,
            learning_rate=1.0,
            number_of_iterations=1,
            resample=False,
        )

        assert_allclose(transform, initial_transform)

    def test_bspline_with_affine_initialization_stores_pre_affine(
        self, sample_fusi_2d_registration
    ):
        """B-spline result stores the pre-affine when affine initialization is given."""
        pre_affine = np.eye(4)
        _, bspline_tx, diagnostics = register_volume(
            sample_fusi_2d_registration,
            sample_fusi_2d_registration,
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
        self, sample_fusi_2d_registration
    ):
        """B-spline result without affine initialization has no bspline_initialization key."""
        _, bspline_tx, _ = register_volume(
            sample_fusi_2d_registration,
            sample_fusi_2d_registration,
            transform_type="bspline",
            mesh_size=(1, 1, 1),
            number_of_iterations=1,
            learning_rate=0.1,
        )
        assert isinstance(bspline_tx, xr.DataArray)
        affines = bspline_tx.attrs.get("affines", {})
        assert "bspline_initialization" not in affines

    def test_center_moments_uses_moments_initializer(
        self, sample_fusi_2d_registration, monkeypatch
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
            sample_fusi_2d_registration,
            sample_fusi_2d_registration,
            transform_type="affine",
            initialization="center_moments",
        )

        assert calls == [sitk.CenteredTransformInitializerFilter.MOMENTS]


class TestResampleVolumeWithBspline:
    """Tests for resample_volume and resample_like with a B-spline DataArray transform."""

    def test_resample_like_with_bspline_matches_direct_resample(
        self, sample_fusi_3d_feature_registration
    ):
        """resample_like with a B-spline DataArray matches register_volume(resample=True).

        Uses `sample_fusi_3d_feature_registration.values` (non-singleton, several scattered features)
        rather than the singleton-k 2D registration fixture, so every axis actually
        supports a control-point grid and no control point is left without any
        image gradient to constrain it. `mesh_size`/`learning_rate` are kept coarse
        and fixed (not this project's `mesh_size=(10, 10, 10)`/`"auto"` defaults) so
        the test stays fast and deterministic -- it only checks that resampling via
        the returned transform matches `register_volume`'s own direct resample, not
        registration accuracy, so a coarse mesh is enough.
        """
        rng = np.random.default_rng(0)
        shift = rng.integers(3, 6, size=3)
        shifted = np.roll(
            sample_fusi_3d_feature_registration.values,
            tuple(int(s) for s in shift),
            axis=(0, 1, 2),
        )
        moving = xr.DataArray(
            shifted,
            dims=sample_fusi_3d_feature_registration.dims,
            coords=sample_fusi_3d_feature_registration.coords,
            attrs=sample_fusi_3d_feature_registration.attrs,
        )
        resampled_direct, bspline_tx, _ = register_volume(
            moving,
            sample_fusi_3d_feature_registration,
            transform_type="bspline",
            mesh_size=(2, 2, 2),
            learning_rate=0.5,
            number_of_iterations=5,
            resample=True,
        )
        assert isinstance(bspline_tx, xr.DataArray)
        result = resample_like(moving, sample_fusi_3d_feature_registration, bspline_tx)
        np.testing.assert_allclose(result.values, resampled_direct.values, atol=1e-5)

    def test_resample_like_with_composite_bspline_matches_direct_resample(
        self, sample_fusi_3d_feature_registration
    ):
        """resample_like with composite B-spline matches register_volume(resample=True).

        See `test_resample_like_with_bspline_matches_direct_resample` for why this
        uses the off-centre-feature `sample_fusi_3d_feature_registration.values` and a coarse, fixed
        mesh/learning rate instead of a singleton-z fixture and this project's
        `mesh_size=(10, 10, 10)`/`"auto"` defaults.
        """
        rng = np.random.default_rng(1)
        shift = rng.integers(2, 4, size=3)
        shifted = np.roll(
            sample_fusi_3d_feature_registration.values,
            tuple(int(s) for s in shift),
            axis=(0, 1, 2),
        )
        moving = xr.DataArray(
            shifted,
            dims=sample_fusi_3d_feature_registration.dims,
            coords=sample_fusi_3d_feature_registration.coords,
            attrs=sample_fusi_3d_feature_registration.attrs,
        )
        # First pass: affine registration.
        _, affine_tx, _ = register_volume(
            moving,
            sample_fusi_3d_feature_registration,
            transform_type="affine",
        )
        # Second pass: B-spline refinement on top of the affine.
        resampled_direct, bspline_tx, _ = register_volume(
            moving,
            sample_fusi_3d_feature_registration,
            transform_type="bspline",
            initialization=affine_tx,
            mesh_size=(2, 2, 2),
            learning_rate=0.5,
            number_of_iterations=5,
            resample=True,
        )
        assert isinstance(bspline_tx, xr.DataArray)
        result = resample_like(moving, sample_fusi_3d_feature_registration, bspline_tx)
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
            dims=["component", "j", "i"],
            coords={"component": ["j", "i"], "j": np.arange(4.0), "i": np.arange(4.0)},
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
                dims=["j", "i"],
            )

    def test_sample_displacement_field_missing_required_attr_raises(self):
        """Missing B-spline metadata is rejected."""
        transform = _add_identity_voxel_to_world(
            xr.DataArray(
                np.zeros((2, 4, 4)),
                dims=["component", "j", "i"],
                coords={
                    "component": ["j", "i"],
                    "j": np.arange(4.0),
                    "i": np.arange(4.0),
                },
                attrs={"type": "bspline_transform"},
            )
        )
        with pytest.raises(ValueError, match="order"):
            sample_displacement_field(
                transform,
                shape=[4, 4],
                spacing=[1.0, 1.0],
                origin=[0.0, 0.0],
                dims=["j", "i"],
            )

    def test_sample_displacement_field_wrong_first_dim_raises(self):
        """A B-spline DataArray without leading 'component' is rejected."""
        transform = xr.DataArray(
            np.zeros((4, 4, 2)),
            dims=["j", "i", "component"],
            coords={"j": np.arange(4.0), "i": np.arange(4.0), "component": ["j", "i"]},
            attrs={
                "type": "bspline_transform",
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
                dims=["j", "i"],
            )

    def test_sample_displacement_field_wrong_direction_shape_raises(self):
        """A direction matrix whose shape doesn't match `dims` is rejected."""
        transform = _add_identity_voxel_to_world(
            xr.DataArray(
                np.zeros((2, 4, 4)),
                dims=["component", "j", "i"],
                coords={
                    "component": ["j", "i"],
                    "j": np.arange(4.0),
                    "i": np.arange(4.0),
                },
                attrs={"type": "bspline_transform", "order": 3},
            )
        )
        with pytest.raises(ValueError, match="direction must have shape"):
            sample_displacement_field(
                transform,
                shape=[4, 4],
                spacing=[1.0, 1.0],
                origin=[0.0, 0.0],
                dims=["j", "i"],
                direction=np.eye(3),
            )

    def test_sample_displacement_field_like_time_reference_raises(
        self, sample_fusi_2dt_registration
    ):
        """The `_like` wrapper rejects references with a time dimension."""
        transform = xr.DataArray(
            np.zeros((2, 4, 4)),
            dims=["component", "j", "i"],
            coords={"component": ["j", "i"], "j": np.arange(4.0), "i": np.arange(4.0)},
            attrs={
                "type": "bspline_transform",
                "order": 3,
                "direction": np.eye(2).tolist(),
            },
        )
        with pytest.raises(ValueError, match="time dimension"):
            sample_displacement_field_like(transform, sample_fusi_2dt_registration)

    def test_sample_displacement_field_like_singleton_dim_without_voxdim_raises(self):
        """Thin references without `voxdim` are rejected during field sampling."""
        transform = xr.DataArray(
            np.zeros((3, 4, 4, 4)),
            dims=["component", "k", "j", "i"],
            coords={
                "component": ["k", "j", "i"],
                "k": np.arange(4.0),
                "j": np.arange(4.0),
                "i": np.arange(4.0),
            },
            attrs={
                "type": "bspline_transform",
                "order": 3,
                "direction": np.eye(3).tolist(),
            },
        )
        reference = xr.DataArray(
            np.zeros((1, 8, 8), dtype=np.float32),
            dims=("k", "j", "i"),
            coords={
                "k": np.array([0.0]),
                "j": np.arange(8, dtype=np.float64) * 0.1,
                "i": np.arange(8, dtype=np.float64) * 0.1,
            },
        )

        with pytest.raises(
            ValueError,
            match="native voxel dimensions|at least 2 spatial dimensions|defined spatial spacing",
        ):
            sample_displacement_field_like(transform, reference)

    def test_invert_displacement_field_wrong_type_attr_raises(self):
        """A DataArray with the wrong `type` attr is rejected."""
        field = xr.DataArray(
            np.zeros((2, 4, 4)),
            dims=["component", "j", "i"],
            coords={"component": ["j", "i"], "j": np.arange(4.0), "i": np.arange(4.0)},
            attrs={"type": "bspline_transform"},
        )
        with pytest.raises(ValueError, match="displacement_field_transform"):
            invert_displacement_field(field)

    def test_invert_displacement_field_wrong_first_dim_raises(self):
        """A DataArray without 'component' as its first dimension is rejected."""
        field = xr.DataArray(
            np.zeros((4, 4, 2)),
            dims=["j", "i", "component"],
            coords={"j": np.arange(4.0), "i": np.arange(4.0), "component": ["j", "i"]},
            attrs={"type": "displacement_field_transform"},
        )
        with pytest.raises(ValueError, match="'component' as its first dimension"):
            invert_displacement_field(field)

    def test_sample_displacement_field_returns_valid_dataarray(
        self, sample_fusi_2d_registration
    ):
        """Sampling an identity B-spline transform yields a near-zero dense field."""
        _, bspline_tx, _ = register_volume(
            sample_fusi_2d_registration,
            sample_fusi_2d_registration,
            transform_type="bspline",
        )
        grid = get_grid_info_from_dataarray(sample_fusi_2d_registration)
        field = sample_displacement_field(bspline_tx, **grid)

        assert field.attrs["type"] == "displacement_field_transform"
        assert field.dims[0] == "component"
        np.testing.assert_array_equal(field.coords["component"].values, ["k", "j", "i"])
        assert_allclose(field.fusi.direction, np.eye(3))
        assert field.shape == (3, *sample_fusi_2d_registration.shape)
        assert_allclose(field.values, 0.0, atol=1e-6)

    def test_sample_displacement_field_like_matches_explicit_grid(
        self, sample_fusi_2d_registration
    ):
        """The `_like` wrapper matches explicit-grid sampling."""
        _, bspline_tx, _ = register_volume(
            sample_fusi_2d_registration,
            sample_fusi_2d_registration,
            transform_type="bspline",
        )

        by_grid = sample_displacement_field(
            bspline_tx,
            **get_grid_info_from_dataarray(sample_fusi_2d_registration),
        )
        by_reference = sample_displacement_field_like(
            bspline_tx, sample_fusi_2d_registration
        )

        assert_array_equal(by_reference.coords["component"].values, ["k", "j", "i"])
        assert_allclose(by_reference.values, by_grid.values, atol=1e-6)
        assert_allclose(
            by_reference.fusi.direction,
            sample_fusi_2d_registration.fusi.direction,
        )
        assert_allclose(
            by_reference.coords["j"].values,
            sample_fusi_2d_registration.coords["j"].values,
        )
        assert_allclose(
            by_reference.coords["i"].values,
            sample_fusi_2d_registration.coords["i"].values,
        )

    def test_sample_and_invert_displacement_field_preserve_direction(self):
        """Direction survives field sampling and inversion on oblique grids."""
        rotation = np.array(
            [
                [np.cos(np.deg2rad(10.0)), -np.sin(np.deg2rad(10.0)), 0.0, 0.0],
                [np.sin(np.deg2rad(10.0)), np.cos(np.deg2rad(10.0)), 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        fixed = _make_voxel_to_world_3d_slab().fusi.affine.apply(rotation)
        _, bspline_tx, _ = register_volume(fixed, fixed, transform_type="bspline")

        field = sample_displacement_field_like(bspline_tx, fixed)
        inverted = invert_displacement_field(field)

        assert_allclose(field.fusi.direction, fixed.fusi.direction)
        assert_allclose(inverted.fusi.direction, fixed.fusi.direction)
        assert_allclose(field.values, 0.0, atol=1e-6)
        assert_allclose(inverted.values, 0.0, atol=1e-6)

    def test_invert_displacement_field_without_voxel_to_world_index_raises(self):
        """A displacement field without a voxel-to-world index is rejected."""
        field = xr.DataArray(
            np.zeros((3, 1, 8, 8), dtype=np.float64),
            dims=["component", "k", "j", "i"],
            coords={
                "component": ["k", "j", "i"],
                "k": np.array([0.0]),
                "j": np.arange(8, dtype=np.float64) * 0.1,
                "i": np.arange(8, dtype=np.float64) * 0.1,
            },
            attrs={"type": "displacement_field_transform"},
        )

        with pytest.raises(ValueError, match="native voxel dimensions"):
            invert_displacement_field(field)

    def test_invert_displacement_field_undoes_translation(self):
        """Inverting a constant translation field approximately negates it.

        Only the interior of the grid is checked: pixels near the boundary map
        outside the field's domain under the translation, which the inversion
        cannot resolve (there is nothing to invert against there).
        """
        shape = (12, 12, 12)
        dims = ["k", "j", "i"]
        translation = np.array([1.0, 2.0, -1.5])
        array = np.broadcast_to(translation[:, None, None, None], (3, *shape)).astype(
            np.float64
        )
        field = _add_identity_voxel_to_world(
            xr.DataArray(
                array.copy(),
                dims=["component", *dims],
                coords={
                    "component": dims,
                    "k": np.arange(shape[0], dtype=np.float64),
                    "j": np.arange(shape[1], dtype=np.float64),
                    "i": np.arange(shape[2], dtype=np.float64),
                },
                attrs={"type": "displacement_field_transform"},
            )
        )

        inverted = invert_displacement_field(field)

        assert inverted.attrs["type"] == "displacement_field_transform"
        interior = np.s_[:, 4:8, 4:8, 4:8]
        assert_allclose(inverted.values[interior], -array[interior], atol=1e-2)

    def test_resample_volume_with_displacement_field_matches_bspline(
        self, sample_fusi_2d_registration
    ):
        """resample_volume with a displacement field matches the equivalent B-spline resample."""
        rng = np.random.default_rng(2)
        shift = rng.integers(3, 6, size=2)
        shifted = np.roll(
            np.roll(sample_fusi_2d_registration.values, int(shift[0]), axis=1),
            int(shift[1]),
            axis=2,
        )
        moving = xr.DataArray(
            shifted,
            dims=sample_fusi_2d_registration.dims,
            coords=sample_fusi_2d_registration.coords,
            attrs=sample_fusi_2d_registration.attrs,
        )
        _, bspline_tx, _ = register_volume(
            moving,
            sample_fusi_2d_registration,
            transform_type="bspline",
        )
        grid = get_grid_info_from_dataarray(sample_fusi_2d_registration)
        field = sample_displacement_field(bspline_tx, **grid)

        resample_grid = _resample_volume_grid_kwargs(sample_fusi_2d_registration)
        result_bspline = resample_volume(moving, bspline_tx, **resample_grid)
        result_field = resample_volume(moving, field, **resample_grid)

        assert_allclose(result_field.values, result_bspline.values, atol=1e-4)

    def test_matches_bspline_with_singleton_spatial_dim(
        self, sample_fusi_2d_registration
    ):
        """A singleton spatial dim (a single 2D slice stored as (1, y, x)) must not
        produce NaN spacing anywhere in the field round trip.

        Regression test: `coords[dim].diff(dim)` is empty for a length-1 axis, so
        `.mean()` silently returns NaN. Field construction/consumption must fall back
        to the `voxdim` coordinate attribute instead (via the `fusi` accessor), as
        `resample_volume`'s own grid handling already does.
        """
        fixed = sample_fusi_2d_registration

        rng = np.random.default_rng(3)
        shift = rng.integers(3, 6, size=2)
        shifted = np.roll(
            np.roll(sample_fusi_2d_registration.values, int(shift[0]), axis=1),
            int(shift[1]),
            axis=2,
        )
        moving = xr.DataArray(
            shifted, dims=fixed.dims, coords=fixed.coords, attrs=fixed.attrs
        )
        _, bspline_tx, _ = register_volume(moving, fixed, transform_type="bspline")

        grid = get_grid_info_from_dataarray(fixed)
        field = sample_displacement_field(bspline_tx, **grid)
        assert not np.isnan(field.values).any()

        resample_grid = _resample_volume_grid_kwargs(fixed)
        result_bspline = resample_volume(moving, bspline_tx, **resample_grid)
        result_field = resample_volume(moving, field, **resample_grid)
        assert_allclose(result_field.values, result_bspline.values, atol=1e-4)

        inverse_field = invert_displacement_field(field)
        assert not np.isnan(inverse_field.values).any()

    def test_invert_displacement_field_with_singleton_spatial_dim_is_nonzero(
        self, sample_fusi_2d_registration
    ):
        """Inverting a field with a singleton spatial axis must not silently no-op.

        Regression test: `InvertDisplacementFieldImageFilter` requires an N-D image
        with N-component vectors and silently returns an all-zero field when any
        spatial axis has size 1, since it has no local neighborhood to compute a
        fixed-point update from along that axis. `invert_displacement_field` must
        expand the degenerate axis before inverting and crop it back down afterward,
        rather than passing the degenerate field straight to the filter.
        """
        fixed = sample_fusi_2d_registration

        rng = np.random.default_rng(4)
        shift = rng.integers(3, 6, size=2)
        shifted = np.roll(
            np.roll(sample_fusi_2d_registration.values, int(shift[0]), axis=1),
            int(shift[1]),
            axis=2,
        )
        moving = xr.DataArray(
            shifted, dims=fixed.dims, coords=fixed.coords, attrs=fixed.attrs
        )
        _, bspline_tx, _ = register_volume(moving, fixed, transform_type="bspline")

        grid = get_grid_info_from_dataarray(fixed)
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
        self, sample_fusi_2dt_registration, sample_fusi_2d_registration
    ):
        """moving with a time dimension resamples each frame with the same transform."""
        result = resample_like(
            sample_fusi_2dt_registration, sample_fusi_2d_registration, np.eye(4)
        )
        assert "time" in result.dims
        assert result.shape == sample_fusi_2dt_registration.shape
        assert_allclose(
            result.coords["time"].values,
            sample_fusi_2dt_registration.coords["time"].values,
        )

    def test_time_dimension_reference_raises(
        self, sample_fusi_2dt_registration, sample_fusi_2d_registration
    ):
        """reference with a time dimension raises ValueError."""
        with pytest.raises(ValueError, match="time"):
            resample_like(
                sample_fusi_2d_registration,
                sample_fusi_2dt_registration,
                np.eye(3),
            )

    def test_singleton_reference_dim_without_voxdim_raises_helpful_error(self):
        """Thin references without `voxdim` are rejected with a repair hint."""
        reference = xr.DataArray(
            np.zeros((1, 8, 8), dtype=np.float32),
            dims=("k", "j", "i"),
            coords={
                "k": np.array([0.0]),
                "j": np.arange(8, dtype=np.float64) * 0.1,
                "i": np.arange(8, dtype=np.float64) * 0.1,
            },
        )
        moving = reference.copy()

        with pytest.raises(
            ValueError,
            match="native voxel dimensions|at least 2 spatial dimensions|defined spatial spacing",
        ):
            resample_like(moving, reference, np.eye(4))

    def test_mismatched_units_between_moving_and_reference_raise(
        self, sample_fusi_2d_registration
    ):
        """moving and reference must agree on spatial coordinate units when declared."""
        moving = sample_fusi_2d_registration.copy()
        reference = sample_fusi_2d_registration.copy()
        moving.coords["y"].attrs["units"] = "mm"
        moving.coords["x"].attrs["units"] = "mm"
        reference.coords["y"].attrs["units"] = "um"
        reference.coords["x"].attrs["units"] = "um"

        with pytest.raises(ValueError, match="units"):
            resample_like(moving, reference, np.eye(3))

    def test_mismatched_units_between_transform_and_reference_raise(
        self, sample_fusi_2d_registration
    ):
        """DataArray transforms must agree with the reference units when declared."""
        reference = sample_fusi_2d_registration.copy()
        reference.coords["y"].attrs["units"] = "mm"
        reference.coords["x"].attrs["units"] = "mm"
        transform = xr.DataArray(
            np.zeros((2, 2, 2), dtype=np.float64),
            dims=["component", "j", "i"],
            coords={
                "component": np.arange(2),
                "j": [0.0, 1.0],
                "i": [0.0, 1.0],
                "y": xr.Variable("j", [0.0, 1.0], attrs={"units": "um"}),
                "x": xr.Variable("i", [0.0, 1.0], attrs={"units": "um"}),
            },
            attrs={"type": "displacement_field_transform"},
        )

        with pytest.raises(ValueError, match="units"):
            resample_like(reference, reference, transform)

    def test_wrong_ndim_reference_raises(self):
        """1D reference raises ValueError."""
        da = xr.DataArray(np.zeros(10), dims=("i",), coords={"i": np.arange(10)})
        with pytest.raises(
            ValueError,
            match="native voxel dimensions|at least 2 spatial dimensions|defined spatial spacing",
        ):
            resample_like(da, da, np.eye(2))

    def test_default_fill_is_moving_min(self, sample_fusi_2d_registration):
        """Out-of-FOV voxels default to moving.min(), not 0.0."""
        moving = sample_fusi_2d_registration.isel(j=slice(8), i=slice(8)).copy()
        moving.values[:] = 5.0
        result = resample_like(moving, sample_fusi_2d_registration, np.eye(4))
        assert float(result.values[-1, -1, -1]) == pytest.approx(5.0, abs=1e-5)

    def test_explicit_default_value_overrides(self, sample_fusi_2d_registration):
        """Explicit default_value overrides the auto-default."""
        moving = sample_fusi_2d_registration.isel(j=slice(8), i=slice(8)).copy()
        moving.values[:] = 5.0
        result = resample_like(
            moving, sample_fusi_2d_registration, np.eye(4), default_value=0.0
        )
        assert float(result.values[-1, -1, -1]) == pytest.approx(0.0, abs=1e-5)

    def test_output_coords_match_reference(self, sample_fusi_2d_registration):
        """Output coordinates match reference, not moving."""
        moving = sample_fusi_2d_registration.isel(j=slice(16), i=slice(16))
        result = resample_like(moving, sample_fusi_2d_registration, np.eye(4))
        assert_allclose(
            result.coords["j"].values,
            sample_fusi_2d_registration.coords["j"].values,
        )
        assert_allclose(
            result.coords["i"].values,
            sample_fusi_2d_registration.coords["i"].values,
        )

    def test_inherits_reference_affines(self, sample_fusi_2d_registration):
        """resample_like output inherits world-space affines from `reference`."""
        moving = sample_fusi_2d_registration.isel(j=slice(16), i=slice(16)).copy()
        reference = sample_fusi_2d_registration.copy()
        moving.attrs["affines"] = {"world_to_lab": np.diag([2.0, 2.0, 1.0])}
        reference.attrs["affines"] = {"world_to_lab": np.diag([3.0, 3.0, 1.0])}

        result = resample_like(moving, reference, np.eye(4))

        assert "registration" not in result.attrs
        assert_allclose(
            result.attrs["affines"]["world_to_lab"],
            reference.attrs["affines"]["world_to_lab"],
        )

    def test_inherits_reference_voxel_to_world_geometry(self):
        """resample_like output inherits reference's grid, not moving's."""
        moving = _make_voxel_to_world_3d_slab()
        reference = moving.fusi.affine.apply(
            np.array(
                [
                    [1.0, 0.0, 0.0, 100.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        )

        result = resample_like(moving, reference, np.eye(4))

        assert_allclose(
            get_voxel_to_world_affine(result), get_voxel_to_world_affine(reference)
        )
        assert not np.allclose(
            get_voxel_to_world_affine(result), get_voxel_to_world_affine(moving)
        )
        assert type(result.xindexes["x"]).__name__ == "VoxelToWorldIndex"
        assert result.coords["i"].dims == reference.coords["i"].dims

    def test_uses_actual_grid_of_cropped_and_strided_reference(self):
        """resample_like places output at reference's actual grid, not its stale affine.

        Regression test: `reference.voxel_to_world` describes voxel-space *values*
        (crop/stride-invariant by design), not the world location of `reference`'s
        array *positions*. Deriving the output grid straight from that affine silently
        misplaced the resampled content whenever `reference` had been cropped or
        strided from a larger array; the correct grid must come from
        `reference.fusi.origin`/`.fusi.spacing`, which account for the crop/stride.
        """
        base = _add_identity_voxel_to_world(
            xr.DataArray(
                np.arange(4 * 20 * 20, dtype=np.float32).reshape(4, 20, 20),
                dims=("k", "j", "i"),
                coords={
                    "k": np.arange(4.0),
                    "j": np.arange(20.0),
                    "i": np.arange(20.0),
                },
            )
        )
        reference = base.isel(k=slice(1, 3), j=slice(2, 10, 2), i=slice(1, 15, 3))

        result = resample_like(reference, reference, np.eye(4), interpolation="nearest")

        assert_allclose(
            [result.fusi.origin[name] for name in ("z", "y", "x")], [1.0, 2.0, 1.0]
        )
        assert_allclose(
            [result.fusi.spacing[dim] for dim in ("k", "j", "i")], [1.0, 2.0, 3.0]
        )
        assert_allclose(result.values, reference.values)

    def test_matches_register_volume_resample_2d(self, sample_fusi_2d_registration):
        """resample_like matches register_volume(resample=True) on a shifted 2D image."""
        rng = np.random.default_rng(42)
        shift = rng.integers(3, 6, size=2)
        shifted = np.roll(
            np.roll(sample_fusi_2d_registration.values, int(shift[0]), axis=1),
            int(shift[1]),
            axis=2,
        )
        moving = xr.DataArray(
            shifted,
            dims=sample_fusi_2d_registration.dims,
            coords=sample_fusi_2d_registration.coords,
            attrs=sample_fusi_2d_registration.attrs,
        )
        resampled_direct, affine, _ = register_volume(
            moving,
            sample_fusi_2d_registration,
            transform_type="translation",
            resample=True,
        )
        result = resample_like(moving, sample_fusi_2d_registration, affine)
        assert_allclose(result.values, resampled_direct.values, atol=1e-5)

    def test_matches_register_volume_resample_3d(self, sample_fusi_3d_registration):
        """resample_like matches register_volume(resample=True) in 3D."""
        shifted = np.roll(sample_fusi_3d_registration.values, 2, axis=0)
        moving = xr.DataArray(
            shifted,
            dims=sample_fusi_3d_registration.dims,
            coords=sample_fusi_3d_registration.coords,
            attrs=sample_fusi_3d_registration.attrs,
        )
        resampled_direct, affine, _ = register_volume(
            moving,
            sample_fusi_3d_registration,
            transform_type="translation",
            learning_rate=1.0,
            number_of_iterations=200,
            resample=True,
        )
        result = resample_like(moving, sample_fusi_3d_registration, affine)
        assert_allclose(result.values, resampled_direct.values, atol=1e-5)

    def test_matches_register_volume_with_affine_initialization(
        self, sample_fusi_2d_registration
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
            np.roll(sample_fusi_2d_registration.values, int(shift[0]), axis=1),
            int(shift[1]),
            axis=2,
        )
        moving = xr.DataArray(
            shifted,
            dims=sample_fusi_2d_registration.dims,
            coords=sample_fusi_2d_registration.coords,
            attrs=sample_fusi_2d_registration.attrs,
        )
        _, affine_init, _ = register_volume(
            moving, sample_fusi_2d_registration, transform_type="translation"
        )
        resampled_direct, affine, _ = register_volume(
            moving,
            sample_fusi_2d_registration,
            transform_type="affine",
            initialization=affine_init,
            resample=True,
        )
        result = resample_like(moving, sample_fusi_2d_registration, affine)
        assert_allclose(result.values, resampled_direct.values, atol=1e-5)

    def test_matches_resample_volume(self, sample_fusi_2d_registration):
        """resample_like and resample_volume produce identical results."""
        moving = sample_fusi_2d_registration.isel(j=slice(16), i=slice(16))
        affine = np.eye(4)
        result_like = resample_like(moving, sample_fusi_2d_registration, affine)
        result_vol = resample_volume(
            moving,
            affine,
            **_resample_volume_grid_kwargs(sample_fusi_2d_registration),
        )
        assert_allclose(result_like.values, result_vol.values, atol=1e-10)


class TestRegisterVolumeDiagnostics:
    """Diagnostics object returned by register_volume."""

    def test_returns_diagnostics_with_consistent_fields(
        self, sample_fusi_2d_registration
    ):
        """register_volume returns a fully populated RegistrationDiagnostics."""
        shifted = np.roll(
            np.roll(sample_fusi_2d_registration.values, 2, axis=1), 2, axis=2
        )
        moving = xr.DataArray(
            shifted,
            dims=sample_fusi_2d_registration.dims,
            coords=sample_fusi_2d_registration.coords,
            attrs=sample_fusi_2d_registration.attrs,
        )

        max_iters = 50
        _, _, diagnostics = register_volume(
            moving,
            sample_fusi_2d_registration,
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

    def test_metric_field_echoes_argument(self, sample_fusi_2d_registration):
        """The `metric` field on diagnostics matches the metric argument."""
        _, _, diagnostics = register_volume(
            sample_fusi_2d_registration,
            sample_fusi_2d_registration,
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
        fixed = _add_identity_voxel_to_world(
            xr.DataArray(
                np.ones((1, 16, 16), dtype=np.float32),
                dims=("k", "j", "i"),
                coords={
                    "k": np.arange(1),
                    "j": np.arange(16) * 0.1,
                    "i": np.arange(16) * 0.1,
                },
            )
        )
        # moving covers only the central 8x8 region.
        moving = _add_identity_voxel_to_world(
            xr.DataArray(
                np.ones((1, 8, 8), dtype=np.float32) * 2.0,
                dims=("k", "j", "i"),
                coords={
                    "k": np.arange(1),
                    "j": np.arange(4, 12) * 0.1,
                    "i": np.arange(4, 12) * 0.1,
                },
            )
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
        fixed = _add_identity_voxel_to_world(
            xr.DataArray(
                np.ones((1, 16, 16), dtype=np.float32),
                dims=("k", "j", "i"),
                coords={
                    "k": np.arange(1),
                    "j": np.arange(16) * 0.1,
                    "i": np.arange(16) * 0.1,
                },
            )
        )
        moving = _add_identity_voxel_to_world(
            xr.DataArray(
                np.ones((1, 8, 8), dtype=np.float32) * 2.0,
                dims=("k", "j", "i"),
                coords={
                    "k": np.arange(1),
                    "j": np.arange(4, 12) * 0.1,
                    "i": np.arange(4, 12) * 0.1,
                },
            )
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

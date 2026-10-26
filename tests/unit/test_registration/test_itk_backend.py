"""Backend-independent geometry, precision, and inverse-transform regressions."""

import gc
import warnings
import weakref
from threading import Event
from types import SimpleNamespace

import itk
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation

from confusius._utils.geometry import get_voxel_to_world_affine
from confusius.registration import (
    invert_displacement_field,
    register_volume,
    resample_like,
    sample_displacement_field,
    sample_displacement_field_like,
)
from confusius.registration.affines import (
    affine_to_itk_linear_transform,
    itk_linear_transform_to_affine,
)
from confusius.xarray import create_voxeldata


@pytest.fixture
def bspline_coefficients():
    """Return a valid zero-displacement cubic coefficient lattice."""
    return create_voxeldata(
        np.zeros((3, 5, 5, 5)),
        dims=["component", "k", "j", "i"],
        extra_coords={"component": ["k", "j", "i"]},
        spacing=[1, 1, 1],
        attrs={"type": "bspline_transform", "order": 3},
    )


@pytest.mark.parametrize(
    "interpolation,order", [("nearest", 0), ("linear", 1), ("bspline", 3)]
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("reverse", [False, True])
def test_resampling_matches_scipy_on_cropped_oblique_grid(
    sample_voxeldata_3d_feature_registration,
    interpolation,
    order,
    dtype,
    reverse,
):
    """Pull geometry and interpolation match an independent world-space reference."""
    source = sample_voxeldata_3d_feature_registration.isel(
        k=slice(1, 9, 2),
        j=slice(3, 15, 2),
        i=slice(2, 18, 2),
    )
    if reverse:
        source = source.isel(i=slice(None, None, -1))
    rotation = np.eye(4)
    rotation[:3, :3] = Rotation.from_euler("xyz", [0.12, -0.2, 0.3]).as_matrix()
    source = source.fusi.affine.apply(rotation)
    rng = np.random.default_rng(48)
    source = source.copy(data=rng.normal(size=source.shape).astype(dtype))
    pull = np.eye(4)
    pull[:3, :3] = Rotation.from_euler("xyz", [0.01, 0.02, -0.03]).as_matrix()
    pull[:3, 3] = [0.17, -0.26, 0.31]
    actual = resample_like(
        source, source, pull, interpolation=interpolation, fill_value=-17
    )

    basis = source.fusi.direction @ np.diag(
        [source.fusi.spacing[d] for d in ("k", "j", "i")]
    )
    origin = np.array([source.fusi.origin[d] for d in ("z", "y", "x")])
    grid = np.indices(source.shape).reshape(3, -1)
    world = basis @ grid + origin[:, None]
    indices = np.linalg.solve(
        basis, pull[:3, :3] @ world + pull[:3, 3, None] - origin[:, None]
    )
    expected = map_coordinates(
        source.values.astype(np.float64),
        indices,
        order=order,
        mode="mirror" if order == 3 else "nearest",
    )
    outside = np.any(
        (indices < -0.5) | (indices >= np.array(source.shape)[:, None] - 0.5), axis=0
    )
    expected[outside] = -17
    assert_allclose(
        actual.values,
        expected.reshape(source.shape).astype(dtype),
        rtol=2e-6,
        atol=2e-6,
    )
    assert_array_equal(actual.coords["x"], source.coords["x"])
    assert actual.dtype == dtype


@pytest.mark.parametrize(
    "dtype,offset", [(np.uint64, 2**63 + 101), (np.int64, -(2**60) + 103)]
)
@pytest.mark.parametrize("extra", [False, True])
def test_nearest_preserves_large_integer_labels(dtype, offset, extra):
    """Nearest interpolation gathers labels without converting them to float64."""
    shape = (2, 3, 4)
    values = np.arange(np.prod(shape), dtype=dtype).reshape(shape) + dtype(offset)
    dims = ["k", "j", "i"]
    if extra:
        values = np.stack([values, values + dtype(17)])
        dims = ["region", *dims]
    source = create_voxeldata(values, dims=dims, spacing=[1, 1, 1])
    pull = np.eye(4)
    pull[2, 3] = 1
    reference = source.isel(region=0, drop=True) if extra else source
    actual = resample_like(
        source, reference, pull, interpolation="nearest", fill_value=0
    )
    expected = np.zeros_like(values)
    expected[..., :-1] = values[..., 1:]
    assert_array_equal(actual.values, expected)
    assert actual.dtype == dtype


def test_sampled_cubic_field_matches_itk_points_in_double_precision():
    """Sampling includes oblique control geometry and the pre-affine in pull order."""
    rng = np.random.default_rng(21)
    shape = (6, 7, 8)
    coefficients = rng.normal(0, 0.013, size=(3, *shape))
    direction = Rotation.from_euler("xyz", [0.2, -0.1, 0.3]).as_matrix()
    spacing = [1.2, 1.7, 2.3]
    origin = [-4.2, 1.1, 2.7]
    pre_affine = np.eye(4)
    pre_affine[:3, :3] = Rotation.from_euler("xyz", [0.01, 0.02, -0.03]).as_matrix()
    pre_affine[:3, 3] = [0.12, -0.2, 0.08]
    transform = create_voxeldata(
        coefficients,
        dims=["component", "k", "j", "i"],
        extra_coords={"component": ["k", "j", "i"]},
        spacing=spacing,
        origin=origin,
        direction=direction,
        attrs={
            "type": "bspline_transform",
            "order": 3,
            "affines": {"bspline_initialization": pre_affine.tolist()},
        },
    )
    output_origin = np.array(origin) + direction @ (np.array(spacing) * 2)
    actual = sample_displacement_field(
        transform,
        dims=["k", "j", "i"],
        shape=[3, 4, 5],
        spacing=[0.3, 0.4, 0.5],
        origin=output_origin,
        direction=direction,
    )
    reference = itk.BSplineTransform[itk.D, 3, 3].New()
    fixed = np.r_[shape, origin, spacing, direction.ravel()].astype(np.float64)
    reference.SetFixedParameters(itk.OptimizerParameters[itk.D](fixed))
    parameters = np.ascontiguousarray(coefficients.transpose(0, 3, 2, 1)).ravel()
    reference.SetParameters(itk.OptimizerParameters[itk.D](parameters))
    expected = np.empty(actual.shape)
    for index in np.ndindex(actual.shape[1:]):
        point = output_origin + direction @ (np.array([0.3, 0.4, 0.5]) * index)
        warped = np.asarray(reference.TransformPoint(tuple(point)))
        expected[(slice(None), *index)] = (
            pre_affine[:3, :3] @ warped + pre_affine[:3, 3] - point
        )
    assert_allclose(actual.values, expected, rtol=0, atol=1e-14)
    assert_allclose(actual.fusi.direction, direction, atol=1e-14)


@pytest.mark.parametrize("order", [0, 1, 2])
def test_lower_order_bspline_sampling_and_resampling(order):
    """Serialized lower-order splines remain usable despite wheel template limits."""
    coefficients = np.broadcast_to(
        np.array([0.12, -0.13, 0.17])[:, None, None, None], (3, 7, 7, 7)
    ).copy()
    transform = create_voxeldata(
        coefficients,
        dims=["component", "k", "j", "i"],
        extra_coords={"component": ["k", "j", "i"]},
        spacing=[1, 1, 1],
        origin=[-3, -3, -3],
        attrs={"type": "bspline_transform", "order": order},
    )
    source = create_voxeldata(
        np.arange(27, dtype=float).reshape(3, 3, 3),
        dims=["k", "j", "i"],
        spacing=[0.2, 0.2, 0.2],
        origin=[0, 0, 0],
    )
    field = sample_displacement_field_like(transform, source)
    shift = np.zeros(3) if order == 0 else coefficients[:, 0, 0, 0]
    expected = np.broadcast_to(shift[:, None, None, None], field.shape)
    assert_allclose(field.values, expected, atol=1e-14)
    affine = np.eye(4)
    affine[:3, 3] = shift
    assert_allclose(
        resample_like(source, source, transform).values,
        resample_like(source, source, affine).values,
        atol=1e-12,
    )


@pytest.mark.parametrize("shape", [(1, 2, 3), (9, 11, 13)])
def test_inverse_field_matches_analytic_affine_on_oblique_grid(shape):
    """Inverse vectors are computed in world space, including thin-axis behavior."""
    direction = Rotation.from_euler("xyz", [0.12, -0.23, 0.34]).as_matrix()
    spacing = np.array([1.2, 0.7, 1.5])
    world = np.einsum(
        "ij,j...->i...", direction @ np.diag(spacing), np.indices(shape, dtype=float)
    )
    matrix = np.diag([0.02, -0.03, 0.01]) if min(shape) >= 4 else np.zeros((3, 3))
    shift = np.array([0.1, -0.12, 0.17])[:, None, None, None]
    values = np.einsum("ij,j...->i...", matrix, world) + shift
    field = create_voxeldata(
        values,
        dims=["component", "k", "j", "i"],
        extra_coords={"component": ["k", "j", "i"]},
        spacing=spacing.tolist(),
        origin=[0, 0, 0],
        direction=direction,
        attrs={"type": "displacement_field_transform"},
    )
    actual = invert_displacement_field(
        field, max_iterations=100, max_error_tolerance=1e-12
    )
    expected = (
        np.einsum("ij,j...->i...", np.linalg.inv(np.eye(3) + matrix), world - shift)
        - world
    )
    selection = np.s_[:, 2:-2, 2:-2, 2:-2] if min(shape) >= 4 else np.s_[:]
    assert_allclose(actual.values[selection], expected[selection], atol=1e-11)
    assert_allclose(actual.fusi.direction, direction, atol=1e-14)


def test_large_affine_offset_does_not_erase_small_matrix_entries():
    """Affine conversion must not recover its matrix by subtracting large points."""
    affine = np.eye(4)
    affine[0, 1] = 1e-6
    affine[:3, 3] = [1e20, -1e20, 2e20]
    assert_array_equal(
        itk_linear_transform_to_affine(affine_to_itk_linear_transform(affine)), affine
    )


def test_db_optimizer_scaling_keeps_zero_power_finite(sample_voxeldata_2d_registration):
    """Zeros may enter decibel optimization without generating infinite gradients."""
    source = sample_voxeldata_2d_registration
    actual, transform, diagnostics = register_volume(
        source,
        source,
        transform_type="translation",
        fixed_intensity_scaling="db",
        moving_intensity_scaling=2.0,
        learning_rate=0.1,
        itk_threads=1,
    )
    assert_allclose(transform, np.eye(4), atol=1e-6)
    assert_allclose(actual.values, source.values, atol=1e-6)
    assert_allclose(diagnostics.metric_values, -1, atol=1e-6)


@pytest.mark.parametrize(
    "function,kwargs",
    [(register_volume, {}), (resample_like, {"transform": np.eye(4)})],
)
def test_complex_data_is_rejected_without_losing_phase(
    sample_voxeldata_2d_registration,
    function,
    kwargs,
):
    """An unsupported image dtype must not silently discard its imaginary part."""
    source = sample_voxeldata_2d_registration.astype(np.complex64) * (1 + 2j)
    with pytest.raises(TypeError, match="complex"):
        function(source, source, fill_value=0, **kwargs)


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"transform_type": "invalid"}, "Invalid transform"),
        ({"metric": "invalid"}, "Invalid metric"),
        ({"resample_interpolation": "nearest"}, "Invalid resample_interpolation"),
        (
            {"number_of_iterations": 0},
            "number_of_iterations must be a positive integer",
        ),
        (
            {"convergence_window_size": 0},
            "convergence_window_size must be a positive integer",
        ),
        (
            {"number_of_histogram_bins": 0},
            "number_of_histogram_bins must be a positive integer",
        ),
        (
            {"shrink_factors": [2, 1], "smoothing_sigmas": [0]},
            "must have the same length",
        ),
    ],
)
def test_registration_rejects_invalid_optimizer_settings(
    sample_voxeldata_2d_registration,
    kwargs,
    message,
):
    """Invalid modes and optimizer bounds must fail before backend execution."""
    source = sample_voxeldata_2d_registration
    with pytest.raises(ValueError, match=message):
        register_volume(source, source, **kwargs)


@pytest.mark.parametrize(
    "fixed_scaling,moving_scaling",
    [("db", "none"), ("none", "db")],
)
def test_db_optimizer_scaling_rejects_negative_power(
    sample_voxeldata_2d_registration,
    fixed_scaling,
    moving_scaling,
):
    """Negative power must be rejected before it produces NaN optimizer inputs."""
    source = sample_voxeldata_2d_registration - 1
    with pytest.raises(ValueError, match="requires nonnegative values"):
        register_volume(
            source,
            source,
            fixed_intensity_scaling=fixed_scaling,
            moving_intensity_scaling=moving_scaling,
        )


@pytest.mark.parametrize(
    "dims,shape",
    [(["z", "y", "x"], [3, 3, 3]), (["k", "j", "i"], [3, 3])],
)
def test_field_sampling_rejects_invalid_output_grid(bspline_coefficients, dims, shape):
    """An explicit output grid requires three native voxel axes and three sizes."""
    with pytest.raises(ValueError, match="native voxel dimensions"):
        sample_displacement_field(
            bspline_coefficients,
            dims=dims,
            shape=shape,
            spacing=[1, 1, 1],
            origin=[0, 0, 0],
        )


@pytest.mark.parametrize("order", [-1, 4])
def test_field_sampling_rejects_unsupported_spline_order(bspline_coefficients, order):
    """Only the supported polynomial orders can be sampled."""
    transform = bspline_coefficients.assign_attrs(order=order)
    with pytest.raises(ValueError, match="order must be between 0 and 3"):
        sample_displacement_field_like(
            transform, bspline_coefficients.isel(component=0, drop=True)
        )


@pytest.mark.parametrize(
    "name,value,message",
    [
        ("max_iterations", 0, "max_iterations must be positive"),
        ("max_error_tolerance", -0.1, "finite and nonnegative"),
        ("max_error_tolerance", float("nan"), "finite and nonnegative"),
        ("max_error_tolerance", float("inf"), "finite and nonnegative"),
    ],
)
def test_inverse_field_rejects_invalid_solver_settings(
    bspline_coefficients,
    name,
    value,
    message,
):
    """Invalid solver bounds must be rejected rather than return an unchecked field."""
    field = sample_displacement_field_like(
        bspline_coefficients,
        bspline_coefficients.isel(component=0, drop=True),
    )
    with pytest.raises(ValueError, match=message):
        invert_displacement_field(field, **{name: value})


@pytest.mark.parametrize(
    "dtype", [np.int16, np.uint32, np.int64, np.uint64, np.float32]
)
def test_cubic_overshoot_saturates_instead_of_wrapping(dtype):
    """Pixel-range saturation matches ITK, including exact 64-bit integer limits."""
    limits = np.finfo(dtype) if np.issubdtype(dtype, np.floating) else np.iinfo(dtype)
    values = np.full((4, 4, 8), limits.min, dtype=dtype)
    values[:, :, 2:6] = limits.max
    source = create_voxeldata(values, dims=["k", "j", "i"], spacing=[1, 1, 1])
    pull = np.eye(4)
    pull[2, 3] = 0.25
    actual = resample_like(source, source, pull, interpolation="bspline", fill_value=0)
    assert_array_equal(actual.values[:, :, 2], np.full((4, 4), limits.max, dtype=dtype))
    assert_array_equal(actual.values[:, :, 0], np.full((4, 4), limits.min, dtype=dtype))


def test_multiresolution_abort_stops_future_levels_and_releases_reporter(
    sample_voxeldata_3d_registration,
):
    """Cancellation stops the whole pyramid and detaches C++ callback ownership."""
    event = Event()
    levels = []
    reporter_refs = []
    closed = []

    class Reporter:
        def __init__(self, registration):
            self.registration = registration

        def update(self):
            """Request cancellation after the first optimizer update."""
            levels.append(self.registration.GetCurrentLevel())
            event.set()

        def close(self):
            """Record finalization without retaining the reporter."""
            closed.append(True)

    def factory(registration, *_images, **_kwargs):
        """Build a reporter that keeps a registration reference like GUI reporters."""
        reporter = Reporter(registration)
        reporter_refs.append(weakref.ref(reporter))
        return reporter

    source = sample_voxeldata_3d_registration
    _, _, diagnostics = register_volume(
        source,
        source,
        transform_type="translation",
        show_progress=True,
        progress_plotter=factory,
        abort_event=event,
        resample=False,
        use_multi_resolution=True,
        shrink_factors=[2, 1],
        smoothing_sigmas=[1, 0],
        itk_threads=1,
    )
    gc.collect()
    assert diagnostics.status == "aborted"
    assert diagnostics.n_iterations == 2
    assert levels == [0, 0]
    assert closed == [True]
    assert reporter_refs[0]() is None


@pytest.mark.parametrize("depths", [(1, 1), (2, 2), (3, 3), (4, 4), (1, 4), (4, 1)])
@pytest.mark.parametrize("thin_axis", [0, 1, 2])
@pytest.mark.parametrize("use_multi_resolution", [False, True])
def test_registration_preserves_native_image_and_mask_grids(
    sample_voxeldata_2d_registration,
    depths,
    thin_axis,
    use_multi_resolution,
):
    """Thin grids stay unchanged, with finite-difference gradients and no smoothing."""
    fixed_depth, moving_depth = depths
    values = np.repeat(sample_voxeldata_2d_registration.values, fixed_depth, axis=0)
    values += np.arange(fixed_depth, dtype=np.float32)[:, None, None] * 10
    values = np.moveaxis(values, 0, thin_axis)
    moving_values = np.moveaxis(
        np.repeat(sample_voxeldata_2d_registration.values, moving_depth, axis=0),
        0,
        thin_axis,
    )
    spacing = np.array([1.7, 0.2, 0.3])
    origin = np.array([2.5, -3.0, 4.0])
    direction = Rotation.from_euler("xyz", [0.12, -0.23, 0.34]).as_matrix()
    source = create_voxeldata(
        values,
        dims=["k", "j", "i"],
        spacing=spacing.tolist(),
        origin=origin.tolist(),
        direction=direction,
    )
    moving = create_voxeldata(
        moving_values,
        dims=("k", "j", "i"),
        spacing=spacing.tolist(),
        origin=origin.tolist(),
        direction=direction,
    )
    mask = source.copy(data=np.indices(source.shape).sum(axis=0) % 2 == 0)
    moving_mask = moving.copy(data=np.indices(moving.shape).sum(axis=0) % 2 == 0)
    use_gradient_filters = min(depths) >= 4
    observed = {}

    def factory(registration, *_images, **_kwargs):
        """Inspect the real optimizer inputs through the public progress hook."""
        image = registration.GetFixedImage()
        metric = itk.down_cast(registration.GetMetric())
        observed["values"] = itk.array_from_image(image).T
        observed["moving"] = itk.array_from_image(registration.GetMovingImage()).T
        observed["spacing"] = np.array(image.GetSpacing())
        observed["origin"] = np.array(image.GetOrigin())
        observed["direction"] = itk.array_from_matrix(image.GetDirection())
        observed["fixed_mask"] = itk.array_from_image(
            itk.down_cast(metric.GetFixedImageMask()).GetImage()
        ).T
        observed["moving_mask"] = itk.array_from_image(
            itk.down_cast(metric.GetMovingImageMask()).GetImage()
        ).T
        observed["gradient_filters"] = (
            metric.GetUseFixedImageGradientFilter(),
            metric.GetUseMovingImageGradientFilter(),
        )
        observed["sigmas"] = list(registration.GetSmoothingSigmasPerLevel())
        return SimpleNamespace(update=lambda: None, close=lambda: None)

    event = Event()
    event.set()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        register_volume(
            moving,
            source,
            fixed_mask=mask,
            moving_mask=moving_mask,
            transform_type="translation",
            use_multi_resolution=use_multi_resolution,
            shrink_factors=(2, 1),
            smoothing_sigmas=(1, 0),
            show_progress=True,
            progress_plotter=factory,
            plot_metric=False,
            plot_composite=False,
            abort_event=event,
            resample=False,
        )
    assert_array_equal(observed["values"], values)
    assert_array_equal(observed["moving"], moving_values)
    assert_array_equal(observed["fixed_mask"], mask.values)
    assert_array_equal(observed["moving_mask"], moving_mask.values)
    assert_allclose(observed["direction"], direction, atol=1e-14)
    assert_allclose(observed["spacing"], spacing, atol=1e-14)
    assert_allclose(observed["origin"], origin, atol=1e-14)
    assert observed["gradient_filters"] == (use_gradient_filters, use_gradient_filters)
    expected_sigmas = [0]
    if use_multi_resolution:
        expected_sigmas = [1, 0] if use_gradient_filters else [0, 0]
    assert observed["sigmas"] == expected_sigmas
    assert len(caught) == int(use_multi_resolution and not use_gradient_filters)
    if caught:
        assert "Gaussian pyramid smoothing" in str(caught[0].message)


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("oblique", [False, True])
@pytest.mark.parametrize("use_multi_resolution", [False, True])
def test_thin_registration_recovers_masked_in_plane_translation(
    sample_voxeldata_2d_registration, depth, oblique, use_multi_resolution
):
    """Masked thin grids recover observable in-plane motion, including oblique planes."""
    direction = (
        Rotation.from_euler("xyz", [0.12, -0.23, 0.34]).as_matrix()
        if oblique
        else np.eye(3)
    )
    values = np.repeat(sample_voxeldata_2d_registration.values, depth, axis=0)
    fixed = create_voxeldata(
        values,
        dims=("k", "j", "i"),
        spacing=(1.7, 0.1, 0.1),
        origin=(2.5, -3.0, 4.0),
        direction=direction,
    )
    moving = fixed.copy(data=np.roll(values, (2, 2), axis=(1, 2)))
    region = np.zeros(fixed.shape, dtype=bool)
    region[:, 4:28, 4:28] = True
    mask = fixed.copy(data=region)
    result, affine, diagnostics = register_volume(
        moving,
        fixed,
        fixed_mask=mask,
        moving_mask=mask,
        transform_type="translation",
        initialization=None,
        use_multi_resolution=use_multi_resolution,
        shrink_factors=(2, 1),
        smoothing_sigmas=(0, 0),
        learning_rate=0.01,
        number_of_iterations=150,
        itk_threads=1,
    )
    assert_allclose(affine[:3, :3], np.eye(3), atol=1e-12)
    assert_allclose((direction.T @ affine[:3, 3])[1:], [0.2, 0.2], atol=0.005)
    # World-coordinate differences can drift along an unobservable oblique normal.
    if not oblique:
        assert abs(affine[0, 3]) < 1e-6
    assert_allclose(
        get_voxel_to_world_affine(result), get_voxel_to_world_affine(fixed), atol=1e-12
    )
    assert np.mean((result.values - fixed.values) ** 2) < 0.02 * np.mean(
        (moving.values - fixed.values) ** 2
    )
    assert np.isfinite(diagnostics.metric_values).all()
    assert diagnostics.n_iterations > 0


def test_zero_worker_count_is_rejected(sample_voxeldata_2d_registration):
    """Zero is not a valid joblib-style worker count."""
    source = sample_voxeldata_2d_registration
    with pytest.raises(ValueError, match="must not be zero"):
        resample_like(source, source, np.eye(4), itk_threads=0)

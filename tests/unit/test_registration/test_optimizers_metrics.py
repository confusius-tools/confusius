"""Additional ITK solvers and metrics, exercised through public registration APIs."""

from inspect import signature
from threading import Event
from types import SimpleNamespace

import itk
import numpy as np
import pytest
from numpy.testing import assert_allclose

from confusius.registration import register_volume, register_volumewise
from confusius.xarray import create_voxeldata
from confusius.xarray.registration import FUSIRegistrationAccessor


@pytest.mark.parametrize(
    "left,right",
    [
        (register_volume, register_volumewise),
        (FUSIRegistrationAccessor.to_volume, FUSIRegistrationAccessor.volumewise),
        (register_volume, FUSIRegistrationAccessor.to_volume),
        (register_volumewise, FUSIRegistrationAccessor.volumewise),
    ],
)
def test_shared_argument_order(left, right):
    """Common arguments have the same relative order across the public APIs."""
    left_names = [
        name.replace("transform_type", "transform")
        for name in signature(left).parameters
    ]
    right_names = [
        name.replace("transform_type", "transform")
        for name in signature(right).parameters
    ]
    common = set(left_names) & set(right_names)
    assert [name for name in left_names if name in common] == [
        name for name in right_names if name in common
    ]
    assert signature(left).parameters["optimizer"].default == "auto"
    assert signature(right).parameters["optimizer"].default == "auto"


@pytest.mark.parametrize(
    "function", [register_volumewise, FUSIRegistrationAccessor.volumewise]
)
def test_volumewise_input_argument_group(function):
    """Masks immediately follow the fixed reference, before registration settings."""
    assert list(signature(function).parameters)[1:5] == [
        "fixed",
        "fixed_mask",
        "moving_mask",
        "reference_time",
    ]


@pytest.mark.parametrize("transform", ["translation", "rigid", "affine", "bspline"])
def test_auto_selects_parametric_gradient_descent(shifted_gaussian, transform):
    """Automatic selection preserves the explicit gradient-descent result."""
    moving, fixed = shifted_gaussian
    results = [
        register_volume(
            moving,
            fixed,
            transform_type=transform,
            optimizer=optimizer,
            metric="mean_squares",
            initialization=None,
            learning_rate=0.05,
            number_of_iterations=2,
            mesh_size=(1, 1, 1),
            itk_threads=1,
        )
        for optimizer in ("auto", "gradient_descent")
    ]
    assert_allclose(results[0][0], results[1][0], rtol=0, atol=0)
    assert_allclose(results[0][1], results[1][1], rtol=0, atol=0)
    assert_allclose(
        results[0][2].metric_values, results[1][2].metric_values, rtol=0, atol=0
    )


@pytest.fixture
def shifted_gaussian():
    """A volumetric Gaussian with an exactly known one-voxel pull translation."""
    k, j, i = np.mgrid[:16, :16, :16]
    values = np.exp(-((k - 7) ** 2 + (j - 7) ** 2 + (i - 7) ** 2) / 12)
    fixed = create_voxeldata(
        values.astype(np.float64),
        dims=["k", "j", "i"],
        spacing=[0.2, 0.2, 0.2],
    )
    moving = fixed.copy(data=np.roll(fixed.values, 1, axis=2))
    return moving, fixed


def test_mean_squares_matches_numpy(shifted_gaussian):
    """The metric equals a naive mean squared difference, including its sign."""
    moving, fixed = shifted_gaussian
    _, _, diagnostics = register_volume(
        moving,
        fixed,
        transform_type="translation",
        metric="mean_squares",
        initialization=None,
        number_of_iterations=1,
        learning_rate=1e-9,
        itk_threads=1,
        resample=False,
    )
    expected = np.mean((moving.values - fixed.values) ** 2)
    assert_allclose(diagnostics.metric_values, expected, rtol=1e-10)
    assert diagnostics.final_metric_value > 0


def test_joint_histogram_mi_matches_itk(shifted_gaussian):
    """Joint-histogram MI uses the requested histogram size and native objective."""
    moving, fixed = shifted_gaussian
    images = [itk.image_from_array(data.values.T.copy()) for data in (fixed, moving)]
    for image in images:
        image.SetSpacing([0.2, 0.2, 0.2])
    reference = itk.JointHistogramMutualInformationImageToImageMetricv4[
        type(images[0]), type(images[1])
    ].New(NumberOfHistogramBins=24)
    reference.SetFixedImage(images[0])
    reference.SetMovingImage(images[1])
    reference.SetFixedTransform(itk.TranslationTransform[itk.D, 3].New())
    reference.SetMovingTransform(itk.TranslationTransform[itk.D, 3].New())
    reference.Initialize()
    expected = reference.GetValue()
    _, _, diagnostics = register_volume(
        moving,
        fixed,
        transform_type="translation",
        metric="joint_mi",
        number_of_histogram_bins=24,
        initialization=None,
        number_of_iterations=1,
        learning_rate=1e-9,
        itk_threads=1,
        resample=False,
    )
    assert_allclose(diagnostics.metric_values, expected, rtol=1e-10)
    assert diagnostics.final_metric_value < 0


@pytest.mark.parametrize("optimizer", ["lbfgs", "one_plus_one"])
def test_solvers_recover_translation(shifted_gaussian, optimizer):
    """Quasi-Newton and stochastic search recover a known world-space translation."""
    moving, fixed = shifted_gaussian
    result, affine, diagnostics = register_volume(
        moving,
        fixed,
        transform_type="translation",
        metric="mean_squares",
        optimizer=optimizer,
        optimizer_seed=42,
        initialization=None,
        number_of_iterations=200,
        learning_rate=0.2,
        itk_threads=1,
    )
    assert_allclose(affine[:3, 3], [0, 0, 0.2], atol=0.035)
    interior = np.s_[3:-3, 3:-3, 3:-3]
    before = np.mean((moving.values[interior] - fixed.values[interior]) ** 2)
    after = np.mean((result.values[interior] - fixed.values[interior]) ** 2)
    assert after < before * 0.1
    assert 0 < diagnostics.n_iterations <= 200
    assert diagnostics.n_iterations == len(diagnostics.metric_values)
    assert np.isfinite(diagnostics.metric_values).all()
    assert diagnostics.final_metric_value == diagnostics.metric_values[-1]


def test_evolutionary_seed_and_native_cost_cache(shifted_gaussian):
    """Seeded runs are reproducible and report the accepted cost, not an unused cache."""
    moving, fixed = shifted_gaussian
    results = [
        register_volume(
            moving,
            fixed,
            transform_type="translation",
            metric="mean_squares",
            optimizer="one_plus_one",
            optimizer_seed=123,
            initialization=None,
            number_of_iterations=5,
            learning_rate=0.01,
            itk_threads=1,
        )
        for _ in range(2)
    ]
    assert_allclose(results[0][1], results[1][1], rtol=0, atol=0)
    assert_allclose(
        results[0][2].metric_values, results[1][2].metric_values, rtol=0, atol=0
    )
    assert (results[0][2].metric_values > 0).all()
    assert (np.diff(results[0][2].metric_values) <= 0).all()


@pytest.mark.parametrize("optimizer", ["lbfgs", "one_plus_one"])
def test_progress_and_multilevel_cancellation(shifted_gaussian, optimizer):
    """Native iteration events, not L-BFGS cost events, drive progress and cancellation."""
    moving, fixed = shifted_gaussian
    event = Event()
    seen = []
    levels = []
    closed = []

    def factory(method, *_images, **_kwargs):
        """Read solver-native values at the public progress hook."""

        def update():
            """Cancel after the second reported optimizer iteration."""
            levels.append(method.GetCurrentLevel())
            seen.append(method.GetOptimizer().GetValue())
            if len(seen) == 2:
                event.set()

        return SimpleNamespace(update=update, close=lambda: closed.append(True))

    _, _, diagnostics = register_volume(
        moving,
        fixed,
        transform_type="translation",
        metric="mean_squares",
        optimizer=optimizer,
        number_of_iterations=50,
        learning_rate=0.05,
        initialization=None,
        show_progress=True,
        progress_plotter=factory,
        abort_event=event,
        itk_threads=1,
        use_multi_resolution=True,
        shrink_factors=[2, 1],
        smoothing_sigmas=[1, 0],
    )
    assert diagnostics.status == "aborted"
    assert 2 <= diagnostics.n_iterations <= 3
    assert_allclose(diagnostics.metric_values, seen)
    assert set(levels) == {0}
    assert closed == [True]


@pytest.mark.parametrize("optimizer", ["lbfgs", "one_plus_one"])
def test_standard_progress_uses_native_cost(shifted_gaussian, optimizer, request):
    """The standard plot reports accepted costs, including the evolutionary cache."""
    import matplotlib.pyplot as plt

    from confusius.registration import MatplotlibRegistrationProgressPlotter

    moving, fixed = shifted_gaussian
    figures = []

    def factory(*args, **kwargs):
        """Keep a public figure reference to inspect the plotted curve."""
        plotter = MatplotlibRegistrationProgressPlotter(*args, **kwargs)
        figures.append(plt.gcf())
        request.addfinalizer(lambda: plt.close(figures[0]))
        return plotter

    _, _, diagnostics = register_volume(
        moving,
        fixed,
        transform_type="translation",
        metric="mean_squares",
        optimizer=optimizer,
        number_of_iterations=3,
        learning_rate=0.05,
        initialization=None,
        itk_threads=1,
        show_progress=True,
        progress_plotter=factory,
        plot_composite=False,
    )
    assert_allclose(figures[0].axes[0].lines[0].get_ydata(), diagnostics.metric_values)
    assert (diagnostics.metric_values > 0).all()


@pytest.mark.parametrize("metric", ["mean_squares", "joint_mi"])
def test_additional_metrics_with_syn(sample_voxeldata_2d_registration, metric):
    """The extra metrics also configure SyN without substituting its solver."""
    data = sample_voxeldata_2d_registration
    result, field, diagnostics = register_volume(
        data,
        data,
        transform_type="syn",
        metric=metric,
        number_of_iterations=1,
        learning_rate=1e-9,
        itk_threads=1,
    )
    assert_allclose(result.values, data.values, atol=1e-5)
    assert_allclose(field.values, 0, atol=1e-7)
    assert np.isfinite(diagnostics.final_metric_value)


def test_lbfgs_bspline_identity(sample_voxeldata_2d_registration):
    """L-BFGS configures a compact B-spline lattice and handles immediate convergence."""
    data = sample_voxeldata_2d_registration
    result, coefficients, diagnostics = register_volume(
        data,
        data,
        transform_type="bspline",
        metric="mean_squares",
        optimizer="lbfgs",
        mesh_size=(1, 3, 3),
        number_of_iterations=2,
        itk_threads=1,
    )
    assert_allclose(coefficients.values, 0, atol=1e-12)
    assert_allclose(result.values, data.values)
    assert diagnostics.final_metric_value == 0


@pytest.mark.parametrize("optimizer", ["lbfgs", "one_plus_one"])
def test_accessors_and_batch_forward_selection(
    sample_voxeldata_2dt_registration, optimizer
):
    """Both accessors and the batch engine use the selected native solver."""
    data = sample_voxeldata_2dt_registration.isel(time=slice(0, 2))
    result = data.fusi.register.volumewise(
        transform="translation",
        optimizer=optimizer,
        optimizer_seed=23,
        metric="mean_squares",
        number_of_iterations=2,
        n_jobs=1,
        keep_transforms=True,
        show_progress=False,
    )
    assert_allclose(result.values, data.values)
    assert_allclose(result.attrs["motion_params"].final_metric_value, 0)
    for matrix in result.attrs["registration_transforms"]:
        assert_allclose(matrix, np.eye(4))
    _, matrix, _ = data.isel(time=0).fusi.register.to_volume(
        data.isel(time=0),
        transform="translation",
        optimizer=optimizer,
        optimizer_seed=23,
        metric="mean_squares",
        number_of_iterations=2,
        itk_threads=1,
    )
    assert_allclose(matrix, np.eye(4))


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"optimizer": "invalid"}, "Invalid optimizer"),
        ({"optimizer_seed": -1}, "optimizer_seed"),
        ({"optimizer_seed": 2**31}, "optimizer_seed"),
        ({"optimizer_seed": 1.5}, "optimizer_seed"),
        ({"optimizer": "lbfgs", "optimizer_weights": [1, 1, 1]}, "optimizer_weights"),
        (
            {"optimizer": "one_plus_one", "optimizer_weights": [1, 1, 1]},
            "optimizer_weights",
        ),
        ({"optimizer": "lbfgs", "transform_type": "syn"}, "SyN uses its own solver"),
        (
            {"optimizer": "gradient_descent", "transform_type": "syn"},
            "SyN uses its own solver",
        ),
        (
            {"optimizer": "one_plus_one", "transform_type": "syn"},
            "SyN uses its own solver",
        ),
        (
            {"optimizer": "one_plus_one", "transform_type": "bspline"},
            "linear transforms",
        ),
    ],
)
def test_rejects_invalid_or_unsupported_selection(
    sample_voxeldata_2d_registration, kwargs, message
):
    """Never silently ignore unsupported solvers, seeds, or parameter weights."""
    with pytest.raises(ValueError, match=message):
        register_volume(
            sample_voxeldata_2d_registration, sample_voxeldata_2d_registration, **kwargs
        )

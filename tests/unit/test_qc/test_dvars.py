"""Formula-based tests for raw and relative DVARS."""

import warnings
from typing import Literal

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import iqr, norm

from confusius.qc import compute_dvars
from confusius.signal import censor_samples
from confusius.xarray import create_voxeldata


@pytest.fixture
def signals():
    """Return reproducible heterogeneous AR(1) signals with a positive baseline."""
    rng = np.random.default_rng(42)
    values = rng.standard_normal((200, 12))
    for t in range(1, len(values)):
        values[t] += 0.5 * values[t - 1]
    values = 100 + values * np.linspace(1, 10, 12)
    return xr.DataArray(
        values, dims=("time", "region"), coords={"time": np.arange(200) * 0.1}
    )


def reference_dvars(values, estimator):
    """Calculate a baseline independently using scalar autocovariances.

    Parameters
    ----------
    values : (time, feature) numpy.ndarray
        Nonconstant input signals.
    estimator : str
        Baseline estimator, or `raw` for no standardization.

    Returns
    -------
    numpy.ndarray
        Expected DVARS including its artificial first sample.
    """
    raw = np.sqrt(np.mean(np.diff(values, axis=0) ** 2, axis=1, dtype=np.float64))
    if estimator == "median":
        raw /= np.sqrt(np.median(raw**2))
    elif estimator == "ar1":
        q25, q75 = np.quantile(values, [0.25, 0.75], axis=0, method="lower")
        sd = (q75 - q25) / (norm.ppf(0.75) - norm.ppf(0.25))
        rho = []
        for column in values.T:
            centered = column - column.mean()
            # The library uses the biased Yule-Walker autocovariance convention.
            rho.append(np.dot(centered[:-1], centered[1:]) / np.dot(centered, centered))
        diff_sd = sd * np.sqrt(2 * (1 - np.array(rho)))
        denominator = np.sqrt(np.mean(diff_sd**2))
        raw /= denominator
    return np.insert(raw, 0, np.nan)


@pytest.mark.parametrize("estimator", ["raw", "median", "ar1"])
@pytest.mark.parametrize("features", [1, 12])
def test_matches_independent_formula(signals, estimator, features):
    """Both variants and all baselines match independently calculated values."""
    signals = signals.isel(region=slice(features))
    kwargs = (
        {"variant": "raw"} if estimator == "raw" else {"baseline_estimator": estimator}
    )
    actual = compute_dvars(signals, normalization_factor=None, **kwargs)
    assert_allclose(actual, reference_dvars(signals.values, estimator), rtol=1e-10)
    assert np.isnan(actual.values[0])
    assert np.isfinite(actual.values[1:]).all()
    xr.testing.assert_identical(actual.time, signals.time)
    assert actual.attrs == {
        "long_name": "DVARS" if estimator == "raw" else "relative DVARS",
        "units": "a.u.",
    }


def test_raw_intensity_normalization(signals):
    """Raw normalization scales by the retained signals' median intensity."""
    actual = compute_dvars(signals, variant="raw")
    expected = reference_dvars(
        signals.values * (1000 / np.median(signals.values)), "raw"
    )
    assert_allclose(actual, expected)


def test_default_is_median(signals):
    """The new default uses the empirical baseline and does not warn."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = compute_dvars(signals)
    assert_allclose(actual, reference_dvars(signals.values, "median"))


def test_median_excludes_artificial_sample():
    """A deterministic asymmetric example catches estimating after padding."""
    signals = xr.DataArray([0.0, 1.0, 4.0, 14.0], dims="time")
    # Differences are 1, 3, 10; sqrt(median(square(differences))) is 3.
    assert_allclose(compute_dvars(signals), np.array([np.nan, 1, 3, 10]) / 3)


def test_median_baseline_resists_an_isolated_spike():
    """Two extreme differences do not change a majority unit-change baseline."""
    values = np.arange(101, dtype=float)
    values[50] += 1000
    expected = np.ones(101)
    expected[0] = np.nan
    expected[50:52] = [1001, 999]
    assert_allclose(compute_dvars(xr.DataArray(values, dims="time")), expected)


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("keep_first", [False, True])
def test_sample_mask_first_frame_policy(signals, lazy, keep_first):
    """An explicit first-frame policy gives an aligned boolean censoring mask."""
    data = signals.chunk({"time": 25, "region": 5}) if lazy else signals
    dvars = compute_dvars(data)
    mask = (dvars < 1.2).compute()
    assert not mask.values[0]
    mask[{"time": 0}] = keep_first
    expected = reference_dvars(signals.values, "median") < 1.2
    expected[0] = keep_first
    xr.testing.assert_identical(
        censor_samples(signals, mask), signals.isel(time=expected)
    )


def test_estimators_agree_for_homogeneous_stationary_noise():
    """Long independent Gaussian signals put all baselines near the same scale."""
    signals = xr.DataArray(
        np.random.default_rng(7).normal(size=(4000, 100)), dims=("time", "region")
    )
    expected = reference_dvars(signals.values, "median")
    for estimator in ("median", "ar1"):
        actual = compute_dvars(signals, baseline_estimator=estimator)
        assert_allclose(actual, expected, rtol=0.03)


@pytest.mark.parametrize("estimator", ["median", "ar1"])
@pytest.mark.parametrize("scale", [7.0, 0.1])
def test_relative_scale_invariance(signals, estimator, scale):
    """Positive global scaling leaves relative DVARS unchanged."""
    actual = compute_dvars(signals * scale, baseline_estimator=estimator)
    expected = compute_dvars(signals, baseline_estimator=estimator)
    assert_allclose(actual, expected, rtol=1e-12)


@pytest.mark.parametrize("estimator", ["median", "ar1"])
def test_zero_variance_filtering(signals, estimator):
    """Constant features are excluded before normalization and differencing."""
    padded = xr.concat([signals, xr.zeros_like(signals.isel(region=[0]))], dim="region")
    actual = compute_dvars(padded, baseline_estimator=estimator)
    assert_allclose(actual, reference_dvars(signals.values, estimator))


@pytest.mark.parametrize("lazy", [False, True])
def test_lower_iqr_excludes_short_sparse_transient(lazy):
    """Nipype-style lower quartiles exclude a transient retained by linear quartiles."""
    data = xr.DataArray(
        [[0.0, 1.0], [0.0, 2.0], [0.0, 3.0], [100.0, 4.0]], dims=("time", "feature")
    )
    if lazy:
        data = data.chunk({"time": 2, "feature": 1})
    assert_allclose(compute_dvars(data).compute(), [np.nan, 1, 1, 1])
    assert_allclose(
        compute_dvars(data, variant="raw", normalization_factor=None).compute(),
        [np.nan, 1, 1, 1],
    )
    assert_allclose(
        compute_dvars(data, remove_zero_variance=False).compute(),
        [np.nan, 1, 1, np.sqrt(10001)],
    )


@pytest.mark.parametrize("lazy", [False, True])
def test_zero_iqr_filter_can_exclude_sparse_transients(lazy):
    """Disabling robust-variance filtering retains isolated changes in longer signals."""
    values = np.column_stack([np.zeros(8), np.arange(1.0, 9.0)])
    values[-1, 0] = 100
    data = xr.DataArray(values, dims=("time", "feature"))
    if lazy:
        data = data.chunk({"time": 2, "feature": 1})
    assert_allclose(compute_dvars(data).compute(), [np.nan] + [1.0] * 7)
    expected = np.array([np.nan] + [1.0] * 6 + [np.sqrt(10001)])
    assert_allclose(compute_dvars(data, remove_zero_variance=False).compute(), expected)


def test_variance_tolerance(signals):
    """Near-constant features are filtered at the requested threshold."""
    values = signals.values.copy()
    values[:, 0] = 1e-6 * np.arange(len(values))
    actual = compute_dvars(signals.copy(data=values), variance_tolerance=0.01)
    assert_allclose(actual, reference_dvars(values[:, 1:], "median"))


@pytest.mark.parametrize("estimator", ["median", "ar1"])
def test_dimension_order_and_dask(signals, estimator):
    """Flattening supports arbitrary feature names, axis order, and Dask chunks."""
    spatial = create_voxeldata(
        signals.values.reshape(200, 2, 2, 3),
        dims=("time", "k", "j", "i"),
        time=signals.time.values,
        spacing=(1, 1, 1),
    )
    inputs = [
        signals.transpose("region", "time"),
        signals.transpose("region", "time").chunk({"time": 13, "region": 5}),
        spatial,
        spatial.chunk({"time": 20}),
        spatial.chunk({"time": 13, "k": 1, "j": 1, "i": 2}),
    ]
    for data in inputs:
        result = compute_dvars(data, baseline_estimator=estimator)
        assert_allclose(result, reference_dvars(signals.values, estimator))
        if data.chunks is not None:
            assert result.chunks is not None
        else:
            assert isinstance(result.data, np.ndarray)


def test_single_time_series(signals):
    """A time-only input is treated as one feature."""
    actual = compute_dvars(signals.isel(region=0))
    assert_allclose(actual, reference_dvars(signals.values[:, :1], "median"))


def test_no_filter_keeps_constant_features(signals):
    """Disabling filtering retains constant features in raw RMS averaging."""
    values = np.column_stack([signals.values, np.zeros(len(signals))])
    actual = compute_dvars(
        xr.DataArray(values, dims=("time", "feature")),
        variant="raw",
        normalization_factor=None,
        remove_zero_variance=False,
    )
    assert_allclose(actual, reference_dvars(values, "raw"))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("lazy", [False, True])
def test_floating_precision_and_spatial_accumulation(signals, dtype, lazy, monkeypatch):
    """Keep input precision for differences while accumulating spatially in float64."""
    data = signals.astype(dtype)
    expected = reference_dvars(data.values, "median")
    difference_dtypes = []
    original_diff = np.diff

    def track_diff(values, *args, **kwargs):
        """Record the precision used by temporal differencing."""
        difference_dtypes.append(values.dtype)
        return original_diff(values, *args, **kwargs)

    monkeypatch.setattr("confusius.qc.dvars.np.diff", track_diff)
    if lazy:
        data = data.chunk({"time": 25, "region": 5})
    actual = compute_dvars(data).compute(scheduler="synchronous")
    assert difference_dtypes
    assert all(observed == np.dtype(dtype) for observed in difference_dtypes)
    assert actual.dtype == np.float64
    assert_allclose(actual, expected, rtol=1e-12)


@pytest.mark.parametrize("lazy", [False, True])
def test_integer_differences_do_not_overflow(lazy):
    """Conversion to float occurs before differencing fixed-width integers."""
    data = xr.DataArray(np.array([0, 200, 0, 200], dtype=np.uint8), dims="time")
    if lazy:
        data = data.chunk({"time": 2})
        assert_array_equal(compute_dvars(data).compute(), [np.nan, 1, 1, 1])
    else:
        assert_array_equal(
            compute_dvars(data, variant="raw", normalization_factor=None),
            [np.nan, 200, 200, 200],
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"variant": "raw", "normalization_factor": None},
        {"baseline_estimator": "median"},
    ],
)
def test_no_unneeded_estimators(signals, monkeypatch, kwargs):
    """Unfiltered raw and median paths require neither IQR nor AR(1)."""

    def unexpected(*args, **kwargs):
        """Fail if an unused estimator is called."""
        pytest.fail("Unexpected IQR or AR(1) estimation")

    monkeypatch.setattr("confusius.qc.dvars.estimate_ar_coeffs", unexpected)
    monkeypatch.setattr("confusius.qc.dvars.sp_stats.iqr", unexpected)
    actual = compute_dvars(signals, remove_zero_variance=False, **kwargs)
    assert_allclose(
        actual,
        reference_dvars(
            signals.values, "raw" if kwargs.get("variant") == "raw" else "median"
        ),
    )


def test_ar1_does_not_warn(signals):
    """The supported model-based estimator does not issue a deprecation warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        compute_dvars(signals, baseline_estimator="ar1")


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"variant": "unknown"}, "variant"),
        ({"baseline_estimator": "unknown"}, "baseline_estimator"),
        ({"baseline_estimator": "legacy"}, "baseline_estimator"),
        ({"variance_tolerance": -1}, "variance_tolerance"),
        ({"variance_tolerance": np.nan}, "variance_tolerance"),
        ({"variant": "raw", "normalization_factor": 0}, "normalization_factor"),
        ({"variant": "raw", "normalization_factor": np.inf}, "normalization_factor"),
    ],
)
def test_invalid_parameters(signals, kwargs, message):
    """Invalid estimator, variant, and numerical controls raise early."""
    with pytest.raises(ValueError, match=message):
        compute_dvars(signals, **kwargs)


@pytest.mark.parametrize(
    "values, dims, message",
    [
        (np.ones((5, 2)), ("sample", "region"), "must have a 'time' dimension"),
        (np.ones((1, 2)), ("time", "region"), "more than 1 timepoint"),
        (np.ones((0, 2)), ("time", "region"), "more than 1 timepoint"),
        (np.ones((5, 0)), ("time", "region"), "nonempty"),
        (np.ones((5, 2)), ("time", "region"), "All signals have variance"),
        (np.array([[1, np.nan], [2, 3]]), ("time", "region"), "finite"),
        (np.array([[1, np.inf], [2, 3]]), ("time", "region"), "finite"),
    ],
)
def test_invalid_data(values, dims, message):
    """Malformed and nonfinite inputs are rejected with informative errors."""
    with pytest.raises(ValueError, match=message):
        compute_dvars(xr.DataArray(values, dims=dims))


@pytest.mark.parametrize("values", [["a", "b"], [1 + 2j, 3 + 4j]])
def test_non_real_data(values):
    """String and complex inputs cannot define real RMS intensity changes."""
    with pytest.raises(TypeError, match="real numeric"):
        compute_dvars(xr.DataArray(values, dims="time"))


def test_zero_empirical_baseline():
    """Sparse changes can have zero median DVARS despite positive temporal IQR."""
    with pytest.raises(ValueError, match="baseline must be finite and positive"):
        compute_dvars(xr.DataArray([0.0, 0, 0, 1, 1, 1], dims="time"))


def test_zero_intensity_median():
    """Raw intensity normalization cannot divide by a zero signal median."""
    with pytest.raises(ValueError, match="Median signal intensity"):
        compute_dvars(xr.DataArray([-2.0, -1, 0, 1, 2], dims="time"), variant="raw")


def test_zero_baseline_without_filtering():
    """Constant data without filtering still fails on its relative baseline."""
    with pytest.raises(ValueError, match="baseline must be finite and positive"):
        compute_dvars(xr.DataArray(np.ones(5), dims="time"), remove_zero_variance=False)


@pytest.mark.parametrize("case", ["median", "ar1", "raw", "raw_normalized"])
@pytest.mark.parametrize("remove_zero_variance", [True, False])
def test_preserves_laziness_and_reads_blocks_once(
    signals, case, remove_zero_variance, monkeypatch
):
    """Construction runs no tasks and computation reads each input block once."""
    import dask.array as da
    from dask import delayed
    from dask.callbacks import Callback

    reads = []
    iqr_calls = []
    original_iqr = iqr

    def load_block(index):
        """Track materialization of a source block."""
        reads.append(index)
        return signals.values[:, index : index + 3]

    def track_iqr(*args, **kwargs):
        """Track expensive temporal quantile estimation."""
        iqr_calls.append(1)
        return original_iqr(*args, **kwargs)

    values = da.concatenate(
        [
            da.from_delayed(delayed(load_block)(i), shape=(200, 3), dtype=float)
            for i in range(0, 12, 3)
        ],
        axis=1,
    ).rechunk({0: 25})
    lazy_signals = signals.copy(data=values)
    variant: Literal["raw", "relative"] = (
        "raw" if case.startswith("raw") else "relative"
    )
    estimator: Literal["median", "ar1"] = "ar1" if case == "ar1" else "median"
    normalization = 1000 if case == "raw_normalized" else None
    scaled = (
        signals.values * (1000 / np.median(signals.values))
        if normalization
        else signals.values
    )
    expected = reference_dvars(scaled, "raw" if variant == "raw" else estimator)
    monkeypatch.setattr("confusius.qc.dvars.sp_stats.iqr", track_iqr)
    tasks = []
    with Callback(pretask=lambda key, graph, state: tasks.append(key)):
        result = compute_dvars(
            lazy_signals,
            variant=variant,
            baseline_estimator=estimator,
            normalization_factor=normalization,
            remove_zero_variance=remove_zero_variance,
        )
    assert tasks == []
    assert reads == []
    assert iqr_calls == []
    assert result.chunks is not None
    actual = result.compute(scheduler="synchronous")
    assert_allclose(actual, expected, rtol=1e-12)
    assert actual.name is None
    xr.testing.assert_identical(actual.time, signals.time)
    assert sorted(reads) == [0, 3, 6, 9]
    expected_calls = (
        (1 if case == "raw_normalized" else 4)
        if remove_zero_variance or case == "ar1"
        else 0
    )
    assert len(iqr_calls) == expected_calls


@pytest.mark.parametrize(
    "values, remove, message",
    [
        ([1.0, np.nan, 2, 3], True, "finite"),
        ([1.0, 1, 1, 1], True, "All signals have variance"),
        ([1.0, 1, 1, 1], False, "baseline must be finite and positive"),
        ([0.0, 0, 0, 1, 1, 1], True, "baseline must be finite and positive"),
    ],
)
def test_lazy_validation_at_compute(values, remove, message):
    """Data-dependent validation is deferred but not omitted."""
    signals = xr.DataArray(values, dims="time").chunk({"time": 2})
    result = compute_dvars(signals, remove_zero_variance=remove)
    with pytest.raises(ValueError, match=message):
        result.compute()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("case", ["ar1", "raw", "raw_normalized"])
def test_lazy_matches_eager_across_precisions(signals, dtype, case):
    """Lazy model-based and raw paths match eager results for uneven spatial chunks."""
    data = signals.astype(dtype)
    variant: Literal["raw", "relative"] = "relative" if case == "ar1" else "raw"
    normalization = 1000 if case == "raw_normalized" else None
    expected = compute_dvars(
        data,
        variant=variant,
        baseline_estimator="ar1",
        normalization_factor=normalization,
    )
    lazy = compute_dvars(
        data.chunk({"time": 13, "region": 5}),
        variant=variant,
        baseline_estimator="ar1",
        normalization_factor=normalization,
    )
    assert lazy.chunks is not None
    assert_allclose(
        lazy.compute(), expected, rtol=1e-6 if dtype == np.float32 else 1e-12
    )


@pytest.mark.parametrize("normalization_factor", [None, 1000])
@pytest.mark.parametrize("remove_zero_variance", [False, True])
def test_lazy_raw_uses_global_retained_intensity_median(
    signals, normalization_factor, remove_zero_variance
):
    """Normalized raw DVARS uses the global median, never a mean of block medians."""
    values = signals.values.copy()
    values[:, :3] = 0
    values[:, 3:6] += 1000
    retained = values[:, 3:] if remove_zero_variance else values
    if normalization_factor is not None:
        retained = retained * (normalization_factor / np.median(retained))
    expected = reference_dvars(retained, "raw")
    lazy = signals.copy(data=values).chunk({"time": 25, "region": 5})
    actual = compute_dvars(
        lazy,
        variant="raw",
        normalization_factor=normalization_factor,
        remove_zero_variance=remove_zero_variance,
    )
    assert actual.chunks is not None
    assert_allclose(actual.compute(), expected)


@pytest.mark.parametrize("estimator", ["median", "ar1"])
def test_lazy_constant_spatial_block(signals, estimator):
    """An entirely filtered block contributes zero, not an error or biased mean."""
    values = signals.values.copy()
    values[:, :3] = 0
    lazy = signals.copy(data=values).chunk({"time": 25, "region": 3})
    assert_allclose(
        compute_dvars(lazy, baseline_estimator=estimator).compute(),
        reference_dvars(values[:, 3:], estimator),
    )


def test_standardize_removed(signals):
    """The removed argument is not silently accepted."""
    with pytest.raises(TypeError, match="standardize"):
        compute_dvars(signals, standardize=True)  # ty: ignore[unknown-argument]
    with pytest.raises(TypeError):
        compute_dvars(signals, False)  # ty: ignore[too-many-positional-arguments]

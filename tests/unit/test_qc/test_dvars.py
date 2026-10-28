"""Formula-based tests for raw and relative DVARS."""

import warnings
from contextlib import nullcontext

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import iqr

from confusius.qc import compute_dvars
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
    raw = np.sqrt(np.mean(np.diff(values, axis=0) ** 2, axis=1))
    if estimator == "median":
        raw /= np.sqrt(np.median(raw**2))
    elif estimator in ("ar1", "legacy"):
        sd = iqr(values, axis=0, scale="normal", interpolation="lower")
        rho = []
        for column in values.T:
            centered = column - column.mean()
            # The library uses the biased Yule-Walker autocovariance convention.
            rho.append(np.dot(centered[:-1], centered[1:]) / np.dot(centered, centered))
        diff_sd = sd * np.sqrt(2 * (1 - np.array(rho)))
        denominator = (
            np.mean(diff_sd) if estimator == "legacy" else np.sqrt(np.mean(diff_sd**2))
        )
        raw /= denominator
    return np.insert(raw, 0, raw.min())


@pytest.mark.parametrize("estimator", ["raw", "median", "ar1", "legacy"])
@pytest.mark.parametrize("features", [1, 12])
def test_matches_independent_formula(signals, estimator, features):
    """Both variants and all baselines match independently calculated values."""
    signals = signals.isel(region=slice(features))
    context = (
        pytest.warns(DeprecationWarning, match="legacy")
        if estimator == "legacy"
        else nullcontext()
    )
    kwargs = (
        {"variant": "raw"} if estimator == "raw" else {"baseline_estimator": estimator}
    )
    with context:
        actual = compute_dvars(signals, normalization_factor=None, **kwargs)
    assert_allclose(actual, reference_dvars(signals.values, estimator), rtol=1e-10)
    xr.testing.assert_identical(actual.time, signals.time)
    assert actual.attrs == {
        "long_name": "DVARS" if estimator == "raw" else "relative DVARS",
        "units": "a.u." if estimator == "raw" else "1",
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
    assert_allclose(compute_dvars(signals), np.array([1, 1, 3, 10]) / 3)


def test_ar1_and_legacy_differ_for_heterogeneous_variance(signals):
    """RMS aggregation gives a smaller trace than legacy mean-SD aggregation."""
    with pytest.warns(DeprecationWarning, match="legacy"):
        legacy = compute_dvars(signals, baseline_estimator="legacy")
    ar1 = compute_dvars(signals, baseline_estimator="ar1")
    assert_allclose(legacy, reference_dvars(signals.values, "legacy"))
    assert_allclose(ar1, reference_dvars(signals.values, "ar1"))
    assert np.all(legacy.values > ar1.values * 1.1)


def test_estimators_agree_for_homogeneous_stationary_noise():
    """Long independent Gaussian signals put all baselines near the same scale."""
    signals = xr.DataArray(
        np.random.default_rng(7).normal(size=(4000, 100)), dims=("time", "region")
    )
    expected = reference_dvars(signals.values, "median")
    for estimator in ("median", "ar1", "legacy"):
        context = (
            pytest.warns(DeprecationWarning) if estimator == "legacy" else nullcontext()
        )
        with context:
            actual = compute_dvars(signals, baseline_estimator=estimator)
        assert_allclose(actual, expected, rtol=0.03)


@pytest.mark.parametrize("estimator", ["median", "ar1", "legacy"])
@pytest.mark.parametrize("scale", [7.0, 0.1])
def test_relative_scale_invariance(signals, estimator, scale):
    """Positive global scaling leaves relative DVARS unchanged."""
    context = (
        pytest.warns(DeprecationWarning) if estimator == "legacy" else nullcontext()
    )
    with context:
        actual = compute_dvars(signals * scale, baseline_estimator=estimator)
        expected = compute_dvars(signals, baseline_estimator=estimator)
    assert_allclose(actual, expected, rtol=1e-12)


@pytest.mark.parametrize("estimator", ["median", "ar1", "legacy"])
def test_zero_variance_filtering(signals, estimator):
    """Constant features are excluded before normalization and differencing."""
    padded = xr.concat([signals, xr.zeros_like(signals.isel(region=[0]))], dim="region")
    context = (
        pytest.warns(DeprecationWarning) if estimator == "legacy" else nullcontext()
    )
    with context:
        actual = compute_dvars(padded, baseline_estimator=estimator)
    assert_allclose(actual, reference_dvars(signals.values, estimator))


def test_median_negative_scale_invariance(signals):
    """Empirical normalization is also invariant to a sign reversal."""
    assert_allclose(compute_dvars(-3 * signals), compute_dvars(signals), rtol=1e-12)


def test_variance_tolerance(signals):
    """Near-constant features are filtered at the requested threshold."""
    values = signals.values.copy()
    values[:, 0] = 1e-6 * np.arange(len(values))
    actual = compute_dvars(signals.copy(data=values), variance_tolerance=0.01)
    assert_allclose(actual, reference_dvars(values[:, 1:], "median"))


@pytest.mark.parametrize("estimator", ["median", "ar1", "legacy"])
def test_dimension_order_and_dask(signals, estimator):
    """Flattening supports arbitrary feature names, axis order, and Dask chunks."""
    spatial = create_voxeldata(
        signals.values.reshape(200, 2, 2, 3),
        dims=("time", "k", "j", "i"),
        time=signals.time.values,
        spacing=(1, 1, 1),
    )
    inputs = [signals.transpose("region", "time"), spatial, spatial.chunk({"time": 20})]
    context = (
        pytest.warns(DeprecationWarning) if estimator == "legacy" else nullcontext()
    )
    with context:
        for data in inputs:
            result = compute_dvars(data, baseline_estimator=estimator)
            assert_allclose(result, reference_dvars(signals.values, estimator))
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


def test_integer_differences_do_not_overflow():
    """Conversion to float occurs before differencing fixed-width integers."""
    data = xr.DataArray(np.array([0, 200, 0, 200], dtype=np.uint8), dims="time")
    assert_array_equal(
        compute_dvars(data, variant="raw", normalization_factor=None), [200] * 4
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


def test_legacy_warning_points_to_caller(signals):
    """Legacy emits exactly one actionable warning attributed to the caller."""
    with pytest.warns(
        DeprecationWarning, match="neither preserves legacy numerical values"
    ) as caught:
        compute_dvars(signals, baseline_estimator="legacy")
    assert len(caught) == 1
    assert caught[0].filename == __file__


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"variant": "unknown"}, "variant"),
        ({"baseline_estimator": "unknown"}, "baseline_estimator"),
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


def test_standardize_removed(signals):
    """The removed argument is not silently accepted."""
    with pytest.raises(TypeError, match="standardize"):
        compute_dvars(signals, standardize=True)  # ty: ignore[unknown-argument]
    with pytest.raises(TypeError):
        compute_dvars(signals, False)  # ty: ignore[too-many-positional-arguments]

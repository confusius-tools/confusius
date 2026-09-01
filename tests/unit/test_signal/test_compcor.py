"""Tests for CompCor functions."""

from typing import NotRequired, TypedDict

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from confusius.multipose.timing import build_consolidated_time_coordinate
from confusius.signal import compute_compcor_confounds


class _CompcorKwargs(TypedDict):
    n_components: int
    detrend: bool
    noise_mask: NotRequired[xr.DataArray]
    variance_threshold: NotRequired[float]


def _create_mask_like(data, mask_values):
    """Create a mask with coordinates matching the data."""
    return xr.DataArray(mask_values, dims=data.dims, coords=data.coords)


def _assert_components_match_reference_svd(
    signals: np.ndarray,
    selected_voxels: np.ndarray,
    components: xr.DataArray,
    n_components: int,
) -> None:
    """Compare CompCor components with sign-insensitive standardized SVD."""
    noise_signals = signals[:, selected_voxels]
    noise_signals = noise_signals - noise_signals.mean(axis=0)
    noise_signals = noise_signals / noise_signals.std(axis=0, ddof=1)
    reference, *_ = np.linalg.svd(noise_signals, full_matrices=False)

    for i in range(n_components):
        corr = np.abs(np.corrcoef(components.values[:, i], reference[:, i])[0, 1])
        assert corr > 0.9999


def test_compute_compcor_detrending(make_sample_timeseries):
    """Test that detrending changes the extracted components."""
    n_time = 100
    n_voxels = 50

    signals = make_sample_timeseries(n_time=n_time, n_voxels=n_voxels)

    # Add linear trend to first 20 voxels (noise region)
    trend = np.linspace(-1, 1, n_time)
    signals.values[:, :20] += trend[:, np.newaxis] * 2

    mask_values = np.zeros(n_voxels, dtype=bool)
    mask_values[:20] = True
    noise_mask = _create_mask_like(signals.isel(time=0), mask_values)

    # Extract without detrending
    components_no_detrend = compute_compcor_confounds(
        signals, noise_mask=noise_mask, n_components=5, detrend=False
    )

    # Extract with detrending
    components_detrend = compute_compcor_confounds(
        signals, noise_mask=noise_mask, n_components=5, detrend=True
    )

    # Results should differ (trend affects variance)
    # Use correlation to check they're not just sign-flipped versions
    max_corr = 0
    for i in range(5):
        corr = np.abs(
            np.corrcoef(
                components_no_detrend.values[:, i], components_detrend.values[:, i]
            )[0, 1]
        )
        max_corr = max(max_corr, corr)

    # At least one component should differ substantially
    assert max_corr < 0.99


def test_compute_compcor_4d_imaging_acompcor(sample_voxeldata_3dt):
    """Test aCompCor on 4D imaging data."""
    spatial_shape = sample_voxeldata_3dt.shape[1:]
    mask_values = np.zeros(spatial_shape, dtype=bool)
    mask_values[1:3, 2:4, 3:5] = True
    noise_mask = _create_mask_like(sample_voxeldata_3dt.isel(time=0), mask_values)

    components = compute_compcor_confounds(
        sample_voxeldata_3dt,
        noise_mask=noise_mask,
        n_components=3,
        detrend=False,
    )

    selected_voxels = mask_values.ravel()
    flattened = sample_voxeldata_3dt.values.reshape(
        sample_voxeldata_3dt.sizes["time"], -1
    )
    _assert_components_match_reference_svd(flattened, selected_voxels, components, 3)


def test_compute_compcor_4d_imaging_tcompcor(sample_voxeldata_3dt):
    """Test tCompCor on 4D imaging data."""
    components = compute_compcor_confounds(
        sample_voxeldata_3dt,
        variance_threshold=0.2,
        n_components=3,
        detrend=False,
    )

    flattened = sample_voxeldata_3dt.values.reshape(
        sample_voxeldata_3dt.sizes["time"], -1
    )
    variances = flattened.var(axis=0)
    selected_voxels = variances >= np.quantile(variances, 0.8)
    _assert_components_match_reference_svd(flattened, selected_voxels, components, 3)


@pytest.mark.parametrize("region_id", [1009, 1008])
def test_compute_compcor_integer_label_mask(make_sample_timeseries, region_id):
    """Integer single-label masks ({0, region_id}) must select only in-region voxels.

    `Atlas.get_masks` returns integer masks whose foreground voxels carry the region
    id rather than `True`. The selection must be identical to the equivalent boolean
    mask and must not fall back to selecting every voxel. Both odd and even region ids
    are checked because a bitwise (rather than logical) AND is parity-dependent.
    """
    signals = make_sample_timeseries(n_time=100, n_voxels=50)

    int_values = np.zeros(50, dtype=np.int32)
    int_values[10:20] = region_id
    int_mask = _create_mask_like(signals.isel(time=0), int_values)
    bool_mask = _create_mask_like(signals.isel(time=0), int_values.astype(bool))

    from_int = compute_compcor_confounds(signals, noise_mask=int_mask, n_components=3)
    from_bool = compute_compcor_confounds(signals, noise_mask=bool_mask, n_components=3)
    explicit = compute_compcor_confounds(
        signals.isel(space=slice(10, 20)),
        noise_mask=_create_mask_like(
            signals.isel(time=0, space=slice(10, 20)), np.ones(10, dtype=bool)
        ),
        n_components=3,
    )

    assert from_int.shape == (100, 3)
    # Absolute value guards against arbitrary SVD sign flips.
    assert_allclose(np.abs(from_int.values), np.abs(from_bool.values))
    assert_allclose(np.abs(from_int.values), np.abs(explicit.values))


def test_compute_compcor_requires_mode():
    """Test error when neither noise_mask nor variance_threshold specified."""
    signals = xr.DataArray(
        np.random.randn(100, 50),
        dims=["time", "space"],
        coords={"time": np.arange(100) * 0.1},
    )

    with pytest.raises(ValueError, match="Must specify at least one"):
        compute_compcor_confounds(signals, n_components=5)


def test_compute_compcor_invalid_variance_threshold(make_sample_timeseries):
    """Test error for invalid variance threshold values."""
    signals = make_sample_timeseries()

    # Test threshold <= 0
    with pytest.raises(ValueError, match="must be in range"):
        compute_compcor_confounds(signals, variance_threshold=0.0, n_components=5)

    # Test threshold >= 1
    with pytest.raises(ValueError, match="must be in range"):
        compute_compcor_confounds(signals, variance_threshold=1.0, n_components=5)

    # Test negative threshold
    with pytest.raises(ValueError, match="must be in range"):
        compute_compcor_confounds(signals, variance_threshold=-0.1, n_components=5)


def test_compute_compcor_invalid_n_components(make_sample_timeseries):
    """Test error for invalid n_components."""
    signals = make_sample_timeseries()
    noise_mask = np.ones(50, dtype=bool)

    with pytest.raises(ValueError, match="must be positive"):
        compute_compcor_confounds(signals, noise_mask=noise_mask, n_components=0)  # ty: ignore[invalid-argument-type]

    with pytest.raises(ValueError, match="must be positive"):
        compute_compcor_confounds(signals, noise_mask=noise_mask, n_components=-1)  # ty: ignore[invalid-argument-type]


def test_compute_compcor_no_time_dimension():
    """Test error when signals lack time dimension."""
    signals = xr.DataArray(
        np.random.randn(50, 10),
        dims=["space", "sample"],
    )
    noise_mask = np.ones(50, dtype=bool)

    with pytest.raises(ValueError, match="must have a 'time' dimension"):
        compute_compcor_confounds(signals, noise_mask=noise_mask, n_components=5)  # ty: ignore[invalid-argument-type]


def test_compute_compcor_mask_shape_mismatch(make_sample_timeseries):
    """Test error when mask shape doesn't match signals."""
    signals = make_sample_timeseries(n_voxels=50)

    noise_mask = xr.DataArray(
        np.ones(30, dtype=bool), dims=["space"], coords={"space": np.arange(30)}
    )

    with pytest.raises(ValueError, match="has size 30, expected 50"):
        compute_compcor_confounds(signals, noise_mask=noise_mask, n_components=5)


def test_compute_compcor_mask_rejects_subset_dims(sample_voxeldata_3dt):
    """A mask missing one of data's native voxel dims is rejected upfront."""
    noise_mask = xr.DataArray(
        np.ones(
            (sample_voxeldata_3dt.sizes["k"], sample_voxeldata_3dt.sizes["j"]),
            dtype=bool,
        ),
        dims=["k", "j"],
        coords={
            "k": sample_voxeldata_3dt.coords["k"],
            "j": sample_voxeldata_3dt.coords["j"],
        },
    )

    with pytest.raises(
        ValueError, match="native voxel dimensions|missing voxel dimension"
    ):
        compute_compcor_confounds(
            sample_voxeldata_3dt,
            noise_mask=noise_mask,
            n_components=3,
        )


def test_compute_compcor_empty_mask(make_sample_timeseries):
    """Test error when no voxels are selected."""
    signals = make_sample_timeseries()

    noise_mask = _create_mask_like(signals.isel(time=0), np.zeros(50, dtype=bool))

    with pytest.raises(ValueError, match="No voxels selected"):
        compute_compcor_confounds(signals, noise_mask=noise_mask, n_components=5)


def test_compute_compcor_too_few_voxels(make_sample_timeseries):
    """Test error when selected voxels < n_components."""
    signals = make_sample_timeseries()

    # Only 3 voxels selected
    mask_values = np.zeros(50, dtype=bool)
    mask_values[:3] = True
    noise_mask = _create_mask_like(signals.isel(time=0), mask_values)

    with pytest.raises(ValueError, match="less than n_components"):
        compute_compcor_confounds(signals, noise_mask=noise_mask, n_components=5)


def test_compute_compcor_scaling_invariance(make_sample_timeseries):
    """Test that global scaling doesn't change component directions."""
    signals = make_sample_timeseries(n_time=100, n_voxels=50)
    mask_values = np.zeros(50, dtype=bool)
    mask_values[:20] = True
    noise_mask = _create_mask_like(signals.isel(time=0), mask_values)

    # Original components
    comp1 = compute_compcor_confounds(
        signals, noise_mask=noise_mask, n_components=5, detrend=False
    )

    # Scaled signals
    signals_scaled = signals * 2.0
    comp2 = compute_compcor_confounds(
        signals_scaled, noise_mask=noise_mask, n_components=5, detrend=False
    )

    # After normalization, components should be very similar
    # (allowing for sign flips)
    for i in range(5):
        corr = np.abs(np.corrcoef(comp1.values[:, i], comp2.values[:, i])[0, 1])
        assert corr > 0.99


def test_compute_compcor_ignores_zero_variance_voxels(make_sample_timeseries):
    """Constant voxels are dropped before CompCor SVD."""
    signals = make_sample_timeseries(n_time=100, n_voxels=10)
    signals.values[:, :3] = 5.0

    all_voxels_mask = _create_mask_like(
        signals.isel(time=0), np.ones(signals.sizes["space"], dtype=bool)
    )
    varying_voxels = signals.isel(space=slice(3, None))
    varying_mask = _create_mask_like(
        varying_voxels.isel(time=0), np.ones(varying_voxels.sizes["space"], dtype=bool)
    )

    result = compute_compcor_confounds(
        signals,
        noise_mask=all_voxels_mask,
        n_components=3,
        detrend=False,
    )
    expected = compute_compcor_confounds(
        varying_voxels,
        noise_mask=varying_mask,
        n_components=3,
        detrend=False,
    )

    assert_allclose(np.abs(result.values), np.abs(expected.values))


def test_compute_compcor_all_zero_variance_selected_voxels_error(
    make_sample_timeseries,
):
    """CompCor fails when every selected voxel is constant."""
    signals = make_sample_timeseries(n_time=100, n_voxels=10)
    signals.values[:, :4] = 2.0

    mask_values = np.zeros(signals.sizes["space"], dtype=bool)
    mask_values[:4] = True
    noise_mask = _create_mask_like(signals.isel(time=0), mask_values)

    with pytest.raises(ValueError, match="All voxels have variance below tolerance"):
        compute_compcor_confounds(
            signals,
            noise_mask=noise_mask,
            n_components=2,
            detrend=False,
        )


def test_compute_compcor_tcompcor_selects_high_variance(make_sample_timeseries, rng):
    """Test that tCompCor selects high-variance voxels."""
    n_time = 100
    n_voxels = 50

    signals = make_sample_timeseries(n_time=n_time, n_voxels=n_voxels)

    high_var_voxels = [0, 1, 2, 3, 4]
    for i in high_var_voxels:
        signals.values[:, i] = rng.normal(0, 10, n_time)

    for i in range(5, n_voxels):
        signals.values[:, i] = rng.normal(0, 0.1, n_time)

    components = compute_compcor_confounds(
        signals, variance_threshold=0.1, n_components=3, detrend=False
    )

    assert components.shape == (n_time, 3)

    gram = components.values.T @ components.values
    assert_allclose(gram, np.eye(3), atol=1e-10)


def test_compute_compcor_reproducibility(make_sample_timeseries):
    """Test that repeated calls give identical results."""
    signals = make_sample_timeseries(n_time=100, n_voxels=50)
    mask_values = np.zeros(50, dtype=bool)
    mask_values[:20] = True
    noise_mask = _create_mask_like(signals.isel(time=0), mask_values)

    comp1 = compute_compcor_confounds(
        signals, noise_mask=noise_mask, n_components=5, detrend=True
    )
    comp2 = compute_compcor_confounds(
        signals, noise_mask=noise_mask, n_components=5, detrend=True
    )

    xr.testing.assert_allclose(comp1, comp2)


def test_compute_compcor_orthonormal_output(make_sample_timeseries):
    """Test that output components are orthonormal (U^T U = I)."""
    signals = make_sample_timeseries(n_time=100, n_voxels=50)
    mask_values = np.zeros(50, dtype=bool)
    mask_values[:20] = True
    noise_mask = _create_mask_like(signals.isel(time=0), mask_values)

    components = compute_compcor_confounds(
        signals, noise_mask=noise_mask, n_components=5, detrend=False
    )

    gram = components.values.T @ components.values
    assert_allclose(gram, np.eye(5), atol=1e-10)


@pytest.mark.parametrize(
    "use_noise_mask,use_variance_threshold",
    [
        (True, False),  # aCompCor: noise_mask only
        (False, True),  # tCompCor: variance_threshold only
        (True, True),  # Hybrid: both noise_mask and variance_threshold
    ],
    ids=["acompcor", "tcompcor", "hybrid"],
)
def test_compute_compcor_reference_svd(
    make_sample_timeseries, rng, use_noise_mask, use_variance_threshold
):
    """Compare against PCA (standardized SVD) implementation.

    Tests all three modes: aCompCor, tCompCor, and hybrid.
    """
    n_time = 100
    n_voxels = 50
    signals = make_sample_timeseries(n_time=n_time, n_voxels=n_voxels)

    # Add varying variance to voxels for tCompCor
    for i in range(n_voxels):
        signals.values[:, i] *= (i + 1) * 0.1

    # Build kwargs based on mode
    kwargs: _CompcorKwargs = {"n_components": 5, "detrend": False}

    if use_noise_mask:
        mask_values = np.zeros(n_voxels, dtype=bool)
        mask_values[:30] = True
        kwargs["noise_mask"] = _create_mask_like(signals.isel(time=0), mask_values)

    if use_variance_threshold:
        kwargs["variance_threshold"] = 0.5

    components = compute_compcor_confounds(signals, **kwargs)

    # Determine which voxels were selected for reference implementation
    selected_voxels = np.ones(n_voxels, dtype=bool)

    if use_noise_mask:
        selected_voxels = selected_voxels & kwargs["noise_mask"].values

    if use_variance_threshold:
        masked_signals = signals.values[:, selected_voxels]
        variances = masked_signals.var(axis=0)
        threshold_value = np.quantile(variances, 1 - kwargs["variance_threshold"])
        high_var_mask = np.zeros(n_voxels, dtype=bool)
        high_var_mask[selected_voxels] = variances >= threshold_value
        selected_voxels = high_var_mask

    _assert_components_match_reference_svd(
        signals.values, selected_voxels, components, 5
    )


def test_compute_compcor_single_timepoint():
    """Test error with single timepoint."""
    signals = xr.DataArray(
        np.random.randn(1, 50),
        dims=["time", "space"],
        coords={"time": [0.0], "space": np.arange(50)},
    )
    noise_mask = _create_mask_like(signals.isel(time=0), np.ones(50, dtype=bool))

    with pytest.raises(ValueError, match="more than 1 timepoint"):
        compute_compcor_confounds(signals, noise_mask=noise_mask, n_components=5)


def test_compute_compcor_time_chunked():
    """Test error when time dimension is chunked."""
    import dask.array as da

    signals_data = da.from_array(np.random.randn(100, 50), chunks=(50, 50))
    signals = xr.DataArray(
        signals_data,
        dims=["time", "space"],
        coords={"time": np.arange(100) * 0.1, "space": np.arange(50)},
    )

    noise_mask = _create_mask_like(signals.isel(time=0), np.ones(50, dtype=bool))

    with pytest.raises(ValueError, match="chunked along the 'time' dimension"):
        compute_compcor_confounds(signals, noise_mask=noise_mask, n_components=5)


def test_compute_compcor_dask_path(make_sample_timeseries):
    """Test CompCor uses the Dask SVD path when data are Dask-backed."""
    signals = make_sample_timeseries(n_time=100, n_voxels=50).chunk(
        {"time": -1, "space": 10}
    )
    noise_mask = _create_mask_like(signals.isel(time=0), np.ones(50, dtype=bool))

    components = compute_compcor_confounds(
        signals, noise_mask=noise_mask, n_components=5, detrend=False
    )

    assert isinstance(components.data, da.Array)
    assert components.shape == (100, 5)
    assert_allclose(
        components.compute().values.T @ components.compute().values,
        np.eye(5),
        atol=1e-10,
    )


def test_compute_compcor_dask_svd_compressed_path(rng):
    """Above the size/n_components gate, the Dask path must use svd_compressed too.

    Built from a known, well-separated ground-truth spectrum (plus a small noise
    floor), same reasoning as `test_top_left_singular_vectors_randomized_path_
    matches_exact_svd`: with no real spectral gap, individual singular vectors
    are numerically ill-defined, so a meaningful comparison needs one.
    """
    n_time, n_voxels, n_components = 300, 20_000, 5
    u_true, _ = np.linalg.qr(rng.standard_normal((n_time, n_components)))
    v_true, _ = np.linalg.qr(rng.standard_normal((n_voxels, n_components)))
    s_true = np.array([50.0, 40.0, 30.0, 20.0, 10.0])
    values = u_true @ np.diag(s_true) @ v_true.T
    values += 0.01 * rng.standard_normal((n_time, n_voxels))

    assert values.size > 500 * 500
    assert n_components < 0.8 * min(n_time, n_voxels)

    signals = xr.DataArray(
        values,
        dims=["time", "space"],
        coords={"time": np.arange(n_time) * 0.1, "space": np.arange(n_voxels)},
    ).chunk({"time": -1, "space": 2000})
    noise_mask = _create_mask_like(signals.isel(time=0), np.ones(n_voxels, dtype=bool))

    components = compute_compcor_confounds(
        signals, noise_mask=noise_mask, n_components=n_components, detrend=False
    )
    assert isinstance(components.data, da.Array)
    assert components.shape == (n_time, n_components)

    selected_voxels = np.ones(n_voxels, dtype=bool)
    _assert_components_match_reference_svd(
        values, selected_voxels, components.compute(), n_components
    )


def test_compute_compcor_explained_variance_ratio():
    """Test explained variance ratio against known ground truth.

    Creates synthetic data from orthogonal components with known singular values,
    then verifies that the computed variance ratios match the expected values.
    """
    n_time = 100
    n_voxels = 20
    n_components = 3

    rng = np.random.default_rng(42)

    # Create orthonormal temporal components (columns of U in SVD)
    U_basis = rng.normal(size=(n_time, n_components))
    U_basis, _ = np.linalg.qr(U_basis)

    # Create orthonormal voxel loadings (columns of V in SVD)
    V_basis = rng.normal(size=(n_voxels, n_components))
    V_basis, _ = np.linalg.qr(V_basis)

    # Set specific singular values to control variance contributions
    singular_values = np.array([10.0, 5.0, 2.0])

    # Construct signals: X = U @ diag(s) @ V^T
    signals_data = U_basis @ np.diag(singular_values) @ V_basis.T

    signals = xr.DataArray(
        signals_data,
        dims=["time", "space"],
        coords={"time": np.arange(n_time) * 0.1, "space": np.arange(n_voxels)},
    )

    # Our function standardizes (z-score) before PCA, so we need to compute
    # expected variance ratio after standardization
    signals_centered = signals - signals.mean(dim="time")
    signals_std = signals_centered / signals_centered.std(dim="time")

    # Compute expected variance ratio via SVD
    _, s, _ = np.linalg.svd(signals_std.values, full_matrices=False)
    expected_ratio = (s[:n_components] ** 2) / (s**2).sum()

    # Test with all voxels
    mask = xr.DataArray(
        np.ones(n_voxels, dtype=bool),
        dims=["space"],
        coords={"space": np.arange(n_voxels)},
    )

    components = compute_compcor_confounds(
        signals, noise_mask=mask, n_components=n_components, detrend=False
    )

    variance_ratio = components.coords["explained_variance_ratio"].values

    # Should match expected values exactly (within numerical precision)
    assert_allclose(variance_ratio, expected_ratio, rtol=1e-10)

    # Verify descending order
    assert np.all(np.diff(variance_ratio) <= 0)

    # Test extracting all components - should sum to 1.0
    components_all = compute_compcor_confounds(
        signals, noise_mask=mask, n_components=min(n_time, n_voxels), detrend=False
    )

    variance_ratio_all = components_all.coords["explained_variance_ratio"].values
    assert_allclose(variance_ratio_all.sum(), 1.0, rtol=1e-10)


def test_compute_compcor_confounds_multipose_pools_poses_and_time_coordinate(
    sample_voxeldata_3dt_pose,
):
    """Test CompCor pools voxels across poses and derives a consolidated `time`.

    `pose` is stacked into `space` like every other spatial dim, so noise/high-variance
    voxels are selected jointly across poses. The original per-pose `(time, pose)`
    `time` coordinate would otherwise get silently broadcast to `(time, space)` by
    that stacking (one timestamp per voxel instead of one per timepoint), so the
    result must instead carry a `time`-only coordinate rebuilt with the same
    reference/duration accounting `consolidate_poses` uses.
    """
    n_time = sample_voxeldata_3dt_pose.sizes["time"]
    n_components = 2

    result = compute_compcor_confounds(
        sample_voxeldata_3dt_pose, variance_threshold=0.5, n_components=n_components
    )

    assert result.dims == ("time", "component")
    assert result.coords["time"].dims == ("time",)

    flat_signals = sample_voxeldata_3dt_pose.values.reshape(n_time, -1)
    variances = flat_signals.var(axis=0)
    threshold = np.quantile(variances, 0.5)
    selected_voxels = variances >= threshold
    _assert_components_match_reference_svd(
        flat_signals, selected_voxels, result, n_components
    )

    time_coord = sample_voxeldata_3dt_pose.coords["time"]
    base_time_coord = xr.DataArray(
        time_coord.isel(pose=0).values, dims=["time"], attrs=dict(time_coord.attrs)
    )
    expected_time = build_consolidated_time_coordinate(
        base_time_coord, time_coord.values, dict(time_coord.attrs)
    )
    assert_allclose(result.coords["time"].values, expected_time.values)


# ---------------------------------------------------------------------------
# _left_singular_vectors_via_eigh
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n_time", "n_voxels"),
    [(20, 200), (200, 20), (50, 50)],
    ids=["wide (time < voxels)", "tall (time > voxels)", "square"],
)
def test_left_singular_vectors_via_eigh_matches_svd(rng, n_time, n_voxels):
    from confusius.signal.confounds import _left_singular_vectors_via_eigh

    values = rng.standard_normal((n_time, n_voxels))

    U, s = _left_singular_vectors_via_eigh(values)
    U_ref, s_ref, _ = np.linalg.svd(values, full_matrices=False)

    k = min(n_time, n_voxels)
    assert U.shape == (n_time, k)
    assert s.shape == (k,)
    assert_allclose(s, s_ref, rtol=1e-6, atol=1e-8)

    # Sign of each singular vector is arbitrary; compare columns up to sign.
    for i in range(k):
        assert_allclose(np.abs(U[:, i]), np.abs(U_ref[:, i]), rtol=1e-5, atol=1e-8)


# ---------------------------------------------------------------------------
# _top_left_singular_vectors
# ---------------------------------------------------------------------------


def test_top_left_singular_vectors_uses_exact_path_for_small_matrices(rng):
    """Below the size/n_components gate, the result must be exactly the eigh path."""
    from confusius.signal.confounds import (
        _left_singular_vectors_via_eigh,
        _top_left_singular_vectors,
    )

    values = rng.standard_normal((30, 40))
    U, s = _top_left_singular_vectors(values, n_components=5)
    U_ref, s_ref = _left_singular_vectors_via_eigh(values)

    assert_allclose(s, s_ref[:5])
    assert_allclose(U, U_ref[:, :5])


def test_top_left_singular_vectors_randomized_path_matches_exact_svd(rng):
    """Above the gate, randomized SVD must still closely match the true top components.

    Built from a known, well-separated ground-truth spectrum (plus a small noise
    floor) rather than pure noise: with no real spectral gap, individual singular
    vectors past the first are numerically ill-defined (any rotation within a
    near-degenerate subspace is an equally valid solution), so comparing them
    directly would not be a meaningful test.
    """
    from confusius.signal.confounds import _top_left_singular_vectors

    n_time, n_voxels, n_components = 300, 20_000, 5
    u_true, _ = np.linalg.qr(rng.standard_normal((n_time, n_components)))
    v_true, _ = np.linalg.qr(rng.standard_normal((n_voxels, n_components)))
    s_true = np.array([50.0, 40.0, 30.0, 20.0, 10.0])
    values = u_true @ np.diag(s_true) @ v_true.T
    # Noise floor's characteristic top singular value is roughly
    # noise_scale * (sqrt(n_time) + sqrt(n_voxels)) (Marchenko-Pastur edge);
    # keep it well below the smallest true singular value (10) so all 5 kept
    # components are cleanly separated from noise, not just the top few.
    values += 0.01 * rng.standard_normal((n_time, n_voxels))

    assert values.size > 500 * 500
    assert n_components < 0.8 * min(n_time, n_voxels)

    U, s = _top_left_singular_vectors(values, n_components)
    U_ref, s_ref, _ = np.linalg.svd(values, full_matrices=False)

    assert U.shape == (n_time, n_components)
    assert s.shape == (n_components,)
    assert_allclose(s, s_ref[:n_components], rtol=1e-3)
    for i in range(n_components):
        corr = np.abs(np.corrcoef(U[:, i], U_ref[:, i])[0, 1])
        assert corr > 0.999


def test_top_left_singular_vectors_reproducible(rng):
    """A fixed random_state must give identical results across calls."""
    from confusius.signal.confounds import _top_left_singular_vectors

    values = rng.standard_normal((300, 20_000))
    U1, s1 = _top_left_singular_vectors(values, n_components=5)
    U2, s2 = _top_left_singular_vectors(values, n_components=5)

    assert_allclose(U1, U2)
    assert_allclose(s1, s2)

"""Tests for CompCor functions."""

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from confusius.signal import compute_compcor_confounds


def _create_mask_like(data, mask_values):
    """Create a mask with coordinates matching the data."""
    return xr.DataArray(mask_values, dims=data.dims, coords=data.coords)


def test_compute_compcor_detrending(sample_timeseries):
    """Test that detrending changes the extracted components."""
    n_time = 100
    n_voxels = 50

    signals = sample_timeseries(n_time=n_time, n_voxels=n_voxels)

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


def test_compute_compcor_with_combined_mask(sample_timeseries):
    """Test combining multiple masks (e.g., brain mask AND WM mask)."""
    signals = sample_timeseries(n_time=100, n_voxels=50)

    brain_mask_values = np.zeros(50, dtype=bool)
    brain_mask_values[10:40] = True
    brain_mask = _create_mask_like(signals.isel(time=0), brain_mask_values)

    wm_mask_values = np.zeros(50, dtype=bool)
    wm_mask_values[15:25] = True
    wm_mask = _create_mask_like(signals.isel(time=0), wm_mask_values)

    combined_mask = brain_mask & wm_mask

    components = compute_compcor_confounds(
        signals,
        noise_mask=combined_mask,
        n_components=3,
        detrend=False,
    )

    assert components.shape == (100, 3)


def test_compute_compcor_4d_imaging_acompcor(sample_4d_volume):
    """Test aCompCor on 4D imaging data."""
    spatial_shape = sample_4d_volume.shape[1:]
    mask_values = np.zeros(spatial_shape, dtype=bool)
    mask_values[1:3, 2:4, 3:5] = True
    noise_mask = _create_mask_like(sample_4d_volume.isel(time=0), mask_values)

    components = compute_compcor_confounds(
        sample_4d_volume,
        noise_mask=noise_mask,
        n_components=3,
        detrend=False,
    )

    # Check shape: (time, n_components)
    assert components.shape == (sample_4d_volume.sizes["time"], 3)


def test_compute_compcor_4d_imaging_tcompcor(sample_4d_volume):
    """Test tCompCor on 4D imaging data."""
    components = compute_compcor_confounds(
        sample_4d_volume,
        variance_threshold=0.2,
        n_components=3,
        detrend=False,
    )

    # Check shape
    assert components.shape == (sample_4d_volume.sizes["time"], 3)


def test_compute_compcor_xarray_noise_mask(sample_timeseries):
    """Test with xarray DataArray noise mask."""
    signals = sample_timeseries(n_time=100, n_voxels=50)

    mask_values = np.zeros(50, dtype=bool)
    mask_values[:20] = True
    noise_mask = _create_mask_like(signals.isel(time=0), mask_values)

    components = compute_compcor_confounds(
        signals, noise_mask=noise_mask, n_components=5, detrend=False
    )

    assert components.shape == (100, 5)


def test_compute_compcor_requires_mode():
    """Test error when neither noise_mask nor variance_threshold specified."""
    signals = xr.DataArray(
        np.random.randn(100, 50),
        dims=["time", "space"],
        coords={"time": np.arange(100) * 0.1},
    )

    with pytest.raises(ValueError, match="Must specify at least one"):
        compute_compcor_confounds(signals, n_components=5)


def test_compute_compcor_hybrid_mode(sample_timeseries, rng):
    """Test hybrid mode combining noise_mask and variance_threshold."""
    n_time = 100
    n_voxels = 50
    signals = sample_timeseries(n_time=n_time, n_voxels=n_voxels)

    mask_values = np.zeros(n_voxels, dtype=bool)
    mask_values[:20] = True
    noise_mask = _create_mask_like(signals.isel(time=0), mask_values)

    for i in range(5):
        signals.values[:, i] = rng.normal(0, 10, n_time)

    for i in range(5, n_voxels):
        signals.values[:, i] = rng.normal(0, 0.1, n_time)

    components = compute_compcor_confounds(
        signals,
        noise_mask=noise_mask,
        variance_threshold=0.5,
        n_components=3,
        detrend=False,
    )

    assert components.shape == (n_time, 3)


def test_compute_compcor_invalid_variance_threshold(sample_timeseries):
    """Test error for invalid variance threshold values."""
    signals = sample_timeseries()

    # Test threshold <= 0
    with pytest.raises(ValueError, match="must be in range"):
        compute_compcor_confounds(signals, variance_threshold=0.0, n_components=5)

    # Test threshold >= 1
    with pytest.raises(ValueError, match="must be in range"):
        compute_compcor_confounds(signals, variance_threshold=1.0, n_components=5)

    # Test negative threshold
    with pytest.raises(ValueError, match="must be in range"):
        compute_compcor_confounds(signals, variance_threshold=-0.1, n_components=5)


def test_compute_compcor_invalid_n_components(sample_timeseries):
    """Test error for invalid n_components."""
    signals = sample_timeseries()
    noise_mask = np.ones(50, dtype=bool)

    with pytest.raises(ValueError, match="must be positive"):
        compute_compcor_confounds(signals, noise_mask=noise_mask, n_components=0)

    with pytest.raises(ValueError, match="must be positive"):
        compute_compcor_confounds(signals, noise_mask=noise_mask, n_components=-1)


def test_compute_compcor_no_time_dimension():
    """Test error when signals lack time dimension."""
    signals = xr.DataArray(
        np.random.randn(50, 10),
        dims=["space", "sample"],
    )
    noise_mask = np.ones(50, dtype=bool)

    with pytest.raises(ValueError, match="must have a 'time' dimension"):
        compute_compcor_confounds(signals, noise_mask=noise_mask, n_components=5)


def test_compute_compcor_mask_shape_mismatch(sample_timeseries):
    """Test error when mask shape doesn't match signals."""
    signals = sample_timeseries(n_voxels=50)

    noise_mask = xr.DataArray(
        np.ones(30, dtype=bool), dims=["space"], coords={"space": np.arange(30)}
    )

    with pytest.raises(ValueError, match="does not match"):
        compute_compcor_confounds(signals, noise_mask=noise_mask, n_components=5)


def test_compute_compcor_empty_mask(sample_timeseries):
    """Test error when no voxels are selected."""
    signals = sample_timeseries()

    noise_mask = _create_mask_like(signals.isel(time=0), np.zeros(50, dtype=bool))

    with pytest.raises(ValueError, match="No voxels selected"):
        compute_compcor_confounds(signals, noise_mask=noise_mask, n_components=5)


def test_compute_compcor_too_few_voxels(sample_timeseries):
    """Test error when selected voxels < n_components."""
    signals = sample_timeseries()

    # Only 3 voxels selected
    mask_values = np.zeros(50, dtype=bool)
    mask_values[:3] = True
    noise_mask = _create_mask_like(signals.isel(time=0), mask_values)

    with pytest.raises(ValueError, match="less than n_components"):
        compute_compcor_confounds(signals, noise_mask=noise_mask, n_components=5)


def test_compute_compcor_scaling_invariance(sample_timeseries):
    """Test that global scaling doesn't change component directions."""
    signals = sample_timeseries(n_time=100, n_voxels=50)
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


def test_compute_compcor_tcompcor_selects_high_variance(sample_timeseries, rng):
    """Test that tCompCor selects high-variance voxels."""
    n_time = 100
    n_voxels = 50

    signals = sample_timeseries(n_time=n_time, n_voxels=n_voxels)

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


def test_compute_compcor_reproducibility(sample_timeseries):
    """Test that repeated calls give identical results."""
    signals = sample_timeseries(n_time=100, n_voxels=50)
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


def test_compute_compcor_orthonormal_output(sample_timeseries):
    """Test that output components are orthonormal (U^T U = I)."""
    signals = sample_timeseries(n_time=100, n_voxels=50)
    mask_values = np.zeros(50, dtype=bool)
    mask_values[:20] = True
    noise_mask = _create_mask_like(signals.isel(time=0), mask_values)

    components = compute_compcor_confounds(
        signals, noise_mask=noise_mask, n_components=5, detrend=False
    )

    gram = components.values.T @ components.values
    assert_allclose(gram, np.eye(5), atol=1e-10)


def test_compute_compcor_orthogonality(sample_timeseries):
    """Test that components are orthonormal (PCA property)."""
    signals = sample_timeseries(n_time=200, n_voxels=50)
    mask_values = np.zeros(50, dtype=bool)
    mask_values[:30] = True
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
    sample_timeseries, rng, use_noise_mask, use_variance_threshold
):
    """Compare against PCA (standardized SVD) implementation.

    Tests all three modes: aCompCor, tCompCor, and hybrid.
    """
    n_time = 100
    n_voxels = 50
    signals = sample_timeseries(n_time=n_time, n_voxels=n_voxels)

    # Add varying variance to voxels for tCompCor
    for i in range(n_voxels):
        signals.values[:, i] *= (i + 1) * 0.1

    # Build kwargs based on mode
    kwargs = {"n_components": 5, "detrend": False}

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

    # Reference implementation: manual PCA
    noise_signals = signals.values[:, selected_voxels]
    noise_signals_centered = noise_signals - noise_signals.mean(axis=0)
    noise_signals_std = noise_signals_centered / noise_signals_centered.std(
        axis=0, ddof=1
    )

    U, s, Vt = np.linalg.svd(noise_signals_std, full_matrices=False)
    ref_components = U[:, :5]

    # Compare components (allowing for sign flips)
    for i in range(5):
        corr = np.abs(np.corrcoef(components.values[:, i], ref_components[:, i])[0, 1])
        assert corr > 0.9999


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

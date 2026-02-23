"""Tests for signal filtering functions."""

import dask.array as da
import numpy as np
import pytest
import scipy.signal
import xarray as xr

from confusius.signal import filter_butterworth


def create_signals_with_time(shape, sampling_rate=100, **kwargs):
    """Helper to create DataArray with time coordinates."""
    data = np.random.randn(*shape)
    n_time = shape[0]
    dims = kwargs.pop("dims", ["time"] + [f"dim_{i}" for i in range(1, len(shape))])
    coords = kwargs.pop("coords", {})
    coords["time"] = np.arange(n_time) / sampling_rate
    return xr.DataArray(data, dims=dims, coords=coords, **kwargs)


class TestFilterButterworth:
    """Tests for Butterworth filtering."""

    def test_lowpass_matches_scipy(self, sample_timeseries):
        """Low-pass filter should match scipy.signal.butter + sosfiltfilt."""
        n_timepoints = 500
        n_voxels = 10
        sampling_rate = 100
        signals = sample_timeseries(
            n_time=n_timepoints, n_voxels=n_voxels, sampling_rate=sampling_rate
        )
        data = signals.values

        # Apply our filter.
        low_pass = 0.1
        order = 5
        filtered = filter_butterworth(
            signals,
            high_cutoff=low_pass,
            order=order,
        )

        # Apply scipy filter directly.
        sos = scipy.signal.butter(
            order, low_pass, btype="lowpass", fs=100, output="sos"
        )
        expected = scipy.signal.sosfiltfilt(sos, data, axis=0)

        np.testing.assert_allclose(filtered.values, expected, rtol=1e-10)

    def test_highpass_matches_scipy(self, sample_timeseries):
        """High-pass filter should match scipy.signal.butter + sosfiltfilt."""
        n_timepoints = 500
        n_voxels = 10
        sampling_rate = 100
        signals = sample_timeseries(
            n_time=n_timepoints, n_voxels=n_voxels, sampling_rate=sampling_rate
        )
        data = signals.values

        # Apply our filter.
        high_pass = 0.01
        order = 5
        filtered = filter_butterworth(
            signals,
            low_cutoff=high_pass,
            order=order,
        )

        # Apply scipy filter directly.
        sos = scipy.signal.butter(
            order, high_pass, btype="highpass", fs=100, output="sos"
        )
        expected = scipy.signal.sosfiltfilt(sos, data, axis=0)

        np.testing.assert_allclose(filtered.values, expected, rtol=1e-10)

    def test_bandpass_matches_scipy(self, sample_timeseries):
        """Band-pass filter should match scipy.signal.butter + sosfiltfilt."""
        n_timepoints = 500
        n_voxels = 10
        sampling_rate = 100
        signals = sample_timeseries(
            n_time=n_timepoints, n_voxels=n_voxels, sampling_rate=sampling_rate
        )
        data = signals.values

        # Apply our filter (keep frequencies between 0.01 and 0.1 Hz).
        low_cutoff = 0.01
        high_cutoff = 0.1
        order = 5
        filtered = filter_butterworth(
            signals,
            low_cutoff=low_cutoff,
            high_cutoff=high_cutoff,
            order=order,
        )

        # Compute expected using the same sampling rate our code derives (median of
        # diffs), so that tiny float64 rounding differences don't break the comparison.
        computed_fs = 1.0 / float(np.median(np.diff(signals.coords["time"].values)))
        sos = scipy.signal.butter(
            order,
            [low_cutoff, high_cutoff],
            btype="bandpass",
            fs=computed_fs,
            output="sos",
        )
        expected = scipy.signal.sosfiltfilt(sos, data, axis=0)

        np.testing.assert_allclose(filtered.values, expected, rtol=1e-10)

    def test_lowpass_attenuates_high_frequencies(self):
        """Low-pass filter should attenuate high frequencies."""
        # Create signal with low and high frequency components.
        n_timepoints = 1000
        sampling_rate = 1000  # 1000 Hz.
        t = np.arange(n_timepoints) / sampling_rate

        # Low frequency: 5 Hz.
        low_freq_signal = np.sin(2 * np.pi * 5 * t)
        # High frequency: 100 Hz.
        high_freq_signal = 0.5 * np.sin(2 * np.pi * 100 * t)
        # Combined signal.
        combined = low_freq_signal + high_freq_signal

        signals = xr.DataArray(
            combined[:, np.newaxis], dims=["time", "voxels"], coords={"time": t}
        )

        # Low-pass filter at 10 Hz (should keep 5 Hz, remove 100 Hz).
        filtered = filter_butterworth(signals, high_cutoff=10, order=5)

        # Check middle portion to avoid edge effects from filtering.
        middle_slice = slice(100, 900)
        np.testing.assert_allclose(
            filtered.values[middle_slice, 0], low_freq_signal[middle_slice], atol=0.1
        )

    def test_highpass_attenuates_low_frequencies(self):
        """High-pass filter should attenuate low frequencies."""
        # Create signal with low and high frequency components.
        n_timepoints = 1000
        sampling_rate = 1000  # 1000 Hz.
        t = np.arange(n_timepoints) / sampling_rate

        # Low frequency: 5 Hz.
        low_freq_signal = np.sin(2 * np.pi * 5 * t)
        # High frequency: 100 Hz.
        high_freq_signal = 0.5 * np.sin(2 * np.pi * 100 * t)
        # Combined signal.
        combined = low_freq_signal + high_freq_signal

        signals = xr.DataArray(
            combined[:, np.newaxis], dims=["time", "voxels"], coords={"time": t}
        )

        # High-pass filter at 50 Hz (should remove 5 Hz, keep 100 Hz).
        filtered = filter_butterworth(signals, low_cutoff=50, order=5)

        # Check middle portion to avoid edge effects from filtering.
        middle_slice = slice(100, 900)
        np.testing.assert_allclose(
            filtered.values[middle_slice, 0], high_freq_signal[middle_slice], atol=0.1
        )

    def test_output_actually_filtered(self):
        """Low-pass filter attenuates high-frequency energy."""
        rng = np.random.default_rng(42)
        n = 500
        sampling_rate = 100
        t = np.arange(n) / sampling_rate
        # Signal: 1 Hz (pass) + 40 Hz (stop).
        signal = np.sin(2 * np.pi * 1 * t) + np.sin(2 * np.pi * 40 * t)
        signals = xr.DataArray(
            signal[:, np.newaxis], dims=["time", "voxels"], coords={"time": t}
        )
        filtered = filter_butterworth(signals, high_cutoff=5, order=5)

        # Middle portion only — avoid edge effects.
        mid = slice(100, 400)
        # The 1 Hz component should survive; 40 Hz should be heavily attenuated.
        low_freq_ref = np.sin(2 * np.pi * 1 * t[mid])
        np.testing.assert_allclose(filtered.values[mid, 0], low_freq_ref, atol=0.1)

    def test_single_vs_multiple_voxels_consistency(self, sample_timeseries):
        """Filtering single voxel should match first column of multi-voxel result."""
        n_timepoints = 500
        n_voxels = 50
        sampling_rate = 100
        multi_voxel = sample_timeseries(
            n_time=n_timepoints, n_voxels=n_voxels, sampling_rate=sampling_rate
        )
        data = multi_voxel.values

        # Filter single voxel.
        single_voxel = xr.DataArray(
            data[:, 0],
            dims=["time"],
            coords={"time": np.arange(n_timepoints) / sampling_rate},
        )
        filtered_single = filter_butterworth(single_voxel, high_cutoff=0.1, order=5)

        # Filter multiple voxels.
        filtered_multi = filter_butterworth(multi_voxel, high_cutoff=0.1, order=5)

        # First column of multi-voxel result should match single voxel result.
        np.testing.assert_allclose(
            filtered_multi.values[:, 0], filtered_single.values, rtol=1e-10
        )

    def test_preserves_shape_and_coords_multidimensional(self):
        """Filter should preserve shape and coordinates for multi-dimensional data."""
        signals = xr.DataArray(
            np.random.randn(100, 5, 10, 20, 30),
            dims=["time", "pose", "z", "y", "x"],
            coords={
                "time": np.arange(100) * 0.01,
                "pose": np.arange(5),
            },
        )
        filtered = filter_butterworth(signals, high_cutoff=0.1, order=5)

        assert filtered.shape == signals.shape
        assert filtered.dims == signals.dims
        np.testing.assert_array_equal(filtered.coords["time"], signals.coords["time"])
        np.testing.assert_array_equal(filtered.coords["pose"], signals.coords["pose"])

    def test_dask_array_support(self):
        """Filter should work with Dask-backed arrays."""
        np.random.seed(42)
        sampling_rate = 100
        data = da.from_array(
            np.random.randn(500, 50),
            chunks=(500, 25),  # Chunk spatial, not time.
        )
        signals = xr.DataArray(
            data,
            dims=["time", "voxels"],
            coords={"time": np.arange(500) / sampling_rate},
        )

        filtered = filter_butterworth(signals, high_cutoff=0.1, order=5)

        # Should still be Dask-backed.
        assert isinstance(filtered.data, da.Array)

        # Compute and check it matches eager version.
        signals_eager = signals.compute()
        filtered_eager = filter_butterworth(signals_eager, high_cutoff=0.1, order=5)

        np.testing.assert_allclose(
            filtered.compute().values, filtered_eager.values, rtol=1e-10
        )

    def test_raises_when_no_time_dimension(self):
        """Should raise ValueError if no time dimension."""
        signals = xr.DataArray(np.random.randn(50, 100), dims=["voxels", "samples"])
        with pytest.raises(ValueError, match="must have a 'time' dimension"):
            filter_butterworth(signals, high_cutoff=0.1)

    def test_raises_when_no_cutoffs(self):
        """Should raise ValueError if both low_cutoff and high_cutoff are None."""
        signals = create_signals_with_time(
            (100, 50), sampling_rate=100, dims=["time", "voxels"]
        )
        with pytest.raises(
            ValueError,
            match="At least one of 'low_cutoff' or 'high_cutoff' must be specified",
        ):
            filter_butterworth(signals)

    def test_raises_when_negative_order(self):
        """Should raise ValueError if order is not positive."""
        signals = create_signals_with_time(
            (100, 50), sampling_rate=100, dims=["time", "voxels"]
        )
        with pytest.raises(ValueError, match="'order' must be positive"):
            filter_butterworth(signals, high_cutoff=0.1, order=0)

    def test_raises_when_lowpass_negative(self):
        """Should raise ValueError if high_cutoff is negative."""
        signals = create_signals_with_time(
            (100, 50), sampling_rate=100, dims=["time", "voxels"]
        )
        with pytest.raises(ValueError, match="'high_cutoff' must be positive"):
            filter_butterworth(signals, high_cutoff=-0.1)

    def test_raises_when_highpass_negative(self):
        """Should raise ValueError if low_cutoff is negative."""
        signals = create_signals_with_time(
            (100, 50), sampling_rate=100, dims=["time", "voxels"]
        )
        with pytest.raises(ValueError, match="'low_cutoff' must be positive"):
            filter_butterworth(signals, low_cutoff=-0.1)

    def test_raises_when_lowpass_exceeds_nyquist(self):
        """Should raise ValueError if high_cutoff >= Nyquist frequency."""
        sampling_rate = 100
        signals = create_signals_with_time(
            (100, 50), sampling_rate=sampling_rate, dims=["time", "voxels"]
        )
        nyquist = sampling_rate / 2.0

        with pytest.raises(ValueError, match="must be less than Nyquist frequency"):
            filter_butterworth(signals, high_cutoff=nyquist + 1)

    def test_raises_when_highpass_exceeds_nyquist(self):
        """Should raise ValueError if low_cutoff >= Nyquist frequency."""
        sampling_rate = 100
        signals = create_signals_with_time(
            (100, 50), sampling_rate=sampling_rate, dims=["time", "voxels"]
        )
        nyquist = sampling_rate / 2.0

        with pytest.raises(ValueError, match="must be less than Nyquist frequency"):
            filter_butterworth(signals, low_cutoff=nyquist + 1)

    def test_raises_when_high_cutoff_not_greater_than_low_cutoff(self):
        """Should raise ValueError if high_cutoff <= low_cutoff for band-pass."""
        signals = create_signals_with_time(
            (100, 50), sampling_rate=100, dims=["time", "voxels"]
        )
        with pytest.raises(
            ValueError, match="'high_cutoff' .* must be greater than 'low_cutoff'"
        ):
            filter_butterworth(signals, low_cutoff=0.1, high_cutoff=0.01)

    def test_raises_with_too_few_timepoints(self):
        """Should raise ValueError if too few timepoints for the filter order."""
        # With order=5, n_sections=3, min_samples = 3*(2*3+1)+1 = 22.
        signals = create_signals_with_time(
            (20, 50), sampling_rate=100, dims=["time", "voxels"]
        )
        with pytest.raises(ValueError, match="requires at least .* timepoints"):
            filter_butterworth(signals, high_cutoff=0.1, order=5)

    def test_raises_when_nonuniform_time(self):
        """Should raise ValueError if time coordinates are not uniformly sampled."""
        # Introduce a gap at the midpoint (simulating a dropped volume).
        time = np.concatenate([np.arange(50), np.arange(51, 101)]) / 100.0
        signals = xr.DataArray(
            np.random.randn(100, 10),
            dims=["time", "voxels"],
            coords={"time": time},
        )
        with pytest.raises(ValueError, match="Non-uniform 'time' coordinates"):
            filter_butterworth(signals, high_cutoff=0.1)

    def test_raises_when_time_is_chunked(self):
        """Should raise ValueError if time dimension is chunked."""
        sampling_rate = 100
        data = da.from_array(
            np.random.randn(500, 50),
            chunks=(100, 50),  # Time is chunked!
        )
        signals = xr.DataArray(
            data,
            dims=["time", "voxels"],
            coords={"time": np.arange(500) / sampling_rate},
        )

        with pytest.raises(ValueError, match="chunked along the 'time' dimension"):
            filter_butterworth(signals, high_cutoff=0.1)

    def test_different_orders_produce_different_results(self):
        """Different filter orders should produce different filtered outputs."""
        np.random.seed(42)
        signals = create_signals_with_time(
            (500, 50), sampling_rate=100, dims=["time", "voxels"]
        )

        results = {}
        for order in [1, 2, 5, 8]:
            filtered = filter_butterworth(signals, high_cutoff=0.1, order=order)
            results[order] = filtered.values

        # Different orders should produce different results.
        assert not np.allclose(results[1], results[5])
        assert not np.allclose(results[2], results[8])

    def test_time_not_first_dimension(self):
        """Filter should work when time is not the first dimension."""
        np.random.seed(42)
        sampling_rate = 100
        signals = xr.DataArray(
            np.random.randn(50, 500),
            dims=["voxels", "time"],
            coords={"time": np.arange(500) / sampling_rate},
        )

        filtered = filter_butterworth(signals, high_cutoff=0.1, order=5)

        # Should preserve structure regardless of dim order.
        assert filtered.shape == signals.shape
        assert filtered.dims == signals.dims

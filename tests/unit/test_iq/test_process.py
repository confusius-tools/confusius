"""Unit tests for IQ processing functions."""

import dask.array as da
import numpy as np
import numpy.typing as npt
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from confusius.iq.process import (
    compute_axial_velocity_volume,
    compute_bmode_volume,
    compute_power_doppler_volume,
    compute_processed_volume_timings,
    process_iq_blocks,
    process_iq_to_axial_velocity,
    process_iq_to_bmode,
    process_iq_to_power_doppler,
)


class TestComputeProcessedVolumeTimes:
    """Tests for compute_processed_volume_timings function."""

    @staticmethod
    def _make_iq(
        time_values: npt.ArrayLike,
        *,
        volume_acquisition_duration: float,
        volume_acquisition_reference: str = "start",
    ) -> xr.DataArray:
        time_values = np.asarray(time_values, dtype=np.float64)
        return xr.DataArray(
            np.ones((time_values.size, 1, 1, 1), dtype=np.complex128),
            dims=("time", "z", "y", "x"),
            coords={
                "time": xr.DataArray(
                    time_values,
                    dims=("time",),
                    attrs={
                        "units": "s",
                        "volume_acquisition_duration": volume_acquisition_duration,
                        "volume_acquisition_reference": volume_acquisition_reference,
                    },
                ),
                "z": [0],
                "y": [0],
                "x": [0],
            },
        )

    @pytest.mark.parametrize(
        ("output_ref", "expected"),
        [
            ("start", [0.0]),  # onset of first frame
            ("center", [0.5]),  # 0.0 + 0.5 * 10 * 0.1 = 0.5
            ("end", [1.0]),  # 0.0 + 1.0 * 10 * 0.1 = 1.0
        ],
    )
    def test_single_window_output_reference(self, output_ref, expected):
        """Single window with different output timing references."""
        iq = self._make_iq(np.arange(10) * 0.1, volume_acquisition_duration=0.1)
        result, durations = compute_processed_volume_timings(
            iq,
            clutter_window_width=10,
            clutter_window_stride=10,
            inner_window_width=10,
            inner_window_stride=10,
            processed_time_reference=output_ref,
        )
        assert_allclose(result, expected)
        assert_allclose(durations, [1.0])

    def test_defaults_to_input_time_reference(self):
        """Processed timings reuse the input reference when none is provided."""
        iq = self._make_iq(
            np.arange(10) * 0.1,
            volume_acquisition_duration=0.1,
            volume_acquisition_reference="center",
        )

        result, durations = compute_processed_volume_timings(
            iq,
            clutter_window_width=10,
            clutter_window_stride=10,
            inner_window_width=10,
            inner_window_stride=10,
        )

        assert_allclose(result, [0.45])
        assert_allclose(durations, [1.0])

    def test_single_frame_center(self):
        """Single-frame window: center is onset + volume_duration / 2."""
        # Verifies the bin model: a single frame at t=0 with duration 2 ms has
        # its center at 1 ms, not 0 ms (which the old discrete midpoint gave).
        iq = self._make_iq([0.0], volume_acquisition_duration=0.002)
        result, durations = compute_processed_volume_timings(
            iq,
            clutter_window_width=1,
            clutter_window_stride=1,
            inner_window_width=1,
            inner_window_stride=1,
            processed_time_reference="center",
        )
        assert_allclose(result, [0.001])
        assert_allclose(durations, [0.002])

    def test_multiple_windows_values(self):
        """Multiple windows produce correct timestamps."""
        iq = self._make_iq(np.arange(100) * 0.1, volume_acquisition_duration=0.1)
        result, durations = compute_processed_volume_timings(
            iq,
            clutter_window_width=50,
            clutter_window_stride=50,
            inner_window_width=25,
            inner_window_stride=25,
            processed_time_reference="center",
        )
        # 2 outer windows (start at 0, 50), 2 inner windows each (offset 0, 25).
        # Centers: 0.0 + 0.5*25*0.1=1.25, 2.5+1.25=3.75, 5.0+1.25=6.25, 7.5+1.25=8.75.
        assert_allclose(result, [1.25, 3.75, 6.25, 8.75])
        assert_allclose(durations, [2.5, 2.5, 2.5, 2.5])

    @pytest.mark.parametrize(
        ("volume_ref", "output_ref", "expected"),
        [
            # Input timestamps are centers; recover onset: 0.0 - 0.5*0.1 = -0.05.
            ("center", "start", [-0.05]),
            # Input timestamps are ends; recover onset: 0.0 - 1.0*0.1 = -0.1.
            ("end", "start", [-0.1]),
            # Input timestamps are centers; output center: 0.0 + (0.5*10 - 0.5)*0.1 = 0.45.
            ("center", "center", [0.45]),
        ],
    )
    def test_volume_time_reference_conversion(self, volume_ref, output_ref, expected):
        """volume_time_reference correctly shifts onset recovery."""
        iq = self._make_iq(
            np.arange(10) * 0.1,
            volume_acquisition_duration=0.1,
            volume_acquisition_reference=volume_ref,
        )
        result, _ = compute_processed_volume_timings(
            iq,
            clutter_window_width=10,
            clutter_window_stride=10,
            inner_window_width=10,
            inner_window_stride=10,
            processed_time_reference=output_ref,
        )
        assert_allclose(result, expected)

    @pytest.mark.parametrize(
        "bad_ref_kwarg", ["iq_time_reference", "processed_time_reference"]
    )
    def test_invalid_reference_raises(self, bad_ref_kwarg):
        """Invalid timing reference raises ValueError."""
        iq = self._make_iq(np.arange(10) * 0.1, volume_acquisition_duration=0.1)
        if bad_ref_kwarg == "iq_time_reference":
            iq = iq.assign_coords(
                time=xr.DataArray(
                    iq.coords["time"].values,
                    dims=("time",),
                    attrs={
                        **iq.coords["time"].attrs,
                        "volume_acquisition_reference": "invalid",
                    },
                )
            )
            kwargs = dict(processed_time_reference="start")
        else:
            kwargs = dict(processed_time_reference="invalid")

        kwargs.update(
            clutter_window_width=10,
            clutter_window_stride=10,
            inner_window_width=10,
            inner_window_stride=10,
        )
        with pytest.raises(ValueError, match=bad_ref_kwarg):
            compute_processed_volume_timings(iq, **kwargs)

    def test_matches_docstring_example(self):
        """Result matches the examples from the docstring."""
        iq = self._make_iq(np.arange(100) * 0.1, volume_acquisition_duration=0.1)
        kwargs = dict(
            clutter_window_width=50,
            clutter_window_stride=50,
            inner_window_width=50,
            inner_window_stride=50,
            processed_time_reference="start",
        )
        output_times, output_durations = compute_processed_volume_timings(iq, **kwargs)
        assert_allclose(
            output_times,
            [0.0, 5.0],
        )
        assert_allclose(output_durations, [5.0, 5.0])
        output_times, output_durations = compute_processed_volume_timings(
            iq, **{**kwargs, "processed_time_reference": "center"}
        )
        assert_allclose(
            output_times,
            [2.5, 7.5],
        )
        assert_allclose(output_durations, [5.0, 5.0])

    def test_uses_actual_window_span_when_there_is_dead_time(self):
        """Window timestamps use the observed first/last spacing, not n * duration."""
        iq = self._make_iq([0.0, 2.0, 4.0], volume_acquisition_duration=1.0)

        assert_allclose(
            compute_processed_volume_timings(
                iq,
                clutter_window_width=3,
                clutter_window_stride=3,
                inner_window_width=3,
                inner_window_stride=3,
                processed_time_reference="center",
            )[0],
            [2.5],
        )
        assert_allclose(
            compute_processed_volume_timings(
                iq,
                clutter_window_width=3,
                clutter_window_stride=3,
                inner_window_width=3,
                inner_window_stride=3,
                processed_time_reference="end",
            )[0],
            [5.0],
        )

    def test_returns_varying_window_durations(self):
        """Returned window durations reflect gaps between acquisitions."""
        iq = self._make_iq([0.0, 1.0, 3.0, 4.0, 7.0], volume_acquisition_duration=0.5)

        _, durations = compute_processed_volume_timings(
            iq,
            clutter_window_width=5,
            clutter_window_stride=5,
            inner_window_width=2,
            inner_window_stride=1,
            processed_time_reference="start",
        )

        assert_allclose(durations, [1.5, 2.5, 1.5, 3.5])


class TestComputePowerDopplerVolume:
    """Tests for compute_power_doppler_volume function."""

    def test_matches_reference_implementation(self, sample_iq_block_4d):
        """Result matches reference: mean(|filtered_block|^2)."""
        # No filtering (no cutoffs) so filtered_block = block.
        result = compute_power_doppler_volume(sample_iq_block_4d)

        expected = np.mean(np.abs(sample_iq_block_4d) ** 2, axis=0)
        # Result has shape (1, z, y, x) due to single window.
        assert_allclose(result[0], expected)

    def test_matches_reference_with_svd_filter(self, sample_iq_block_4d, spatial_mask):
        """Result with SVD filter matches reference implementation."""
        low_cutoff, high_cutoff = 2, 8
        time, z, y, x = sample_iq_block_4d.shape

        # Reference: apply SVD filter manually then compute power Doppler.
        signals = sample_iq_block_4d.reshape(time, -1)
        masked_signals = signals[:, spatial_mask.ravel()].astype(np.cdouble)
        u, _, _ = np.linalg.svd(masked_signals, full_matrices=False)
        clutter_vectors = np.concatenate(
            [u[:, :low_cutoff], u[:, high_cutoff:]], axis=1
        )
        filtered_signals = (
            signals - clutter_vectors @ clutter_vectors.conj().T @ signals
        )
        filtered_block = filtered_signals.reshape(sample_iq_block_4d.shape)
        expected = np.mean(np.abs(filtered_block) ** 2, axis=0)

        result = compute_power_doppler_volume(
            sample_iq_block_4d,
            filter_method="svd_indices",
            clutter_mask=spatial_mask,
            low_cutoff=low_cutoff,
            high_cutoff=high_cutoff,
        )

        assert_allclose(result[0], expected, rtol=1e-5, atol=1e-10)

    def test_butterworth_without_fs_raises(self, sample_iq_block_4d_long):
        """Butterworth filter without fs raises ValueError."""
        with pytest.raises(ValueError, match="sampling frequency must be provided"):
            compute_power_doppler_volume(
                sample_iq_block_4d_long,
                filter_method="butterworth",
                low_cutoff=5.0,
            )

    def test_invalid_filter_method_raises(self, sample_iq_block_4d):
        """Invalid filter_method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown clutter filter method"):
            compute_power_doppler_volume(
                sample_iq_block_4d,
                filter_method="invalid",  # type: ignore
            )

    def test_svd_indices_with_float_cutoffs_raises(self, sample_iq_block_4d):
        """SVD indices filter with float cutoffs raises ValueError."""
        with pytest.raises(ValueError, match="must both be integers"):
            compute_power_doppler_volume(
                sample_iq_block_4d,
                filter_method="svd_indices",
                low_cutoff=1.5,
            )


class TestComputeAxialVelocityVolume:
    """Tests for compute_axial_velocity_volume function."""

    def test_matches_reference_kasai_estimator(self, sample_iq_block_4d):
        """Result matches reference Kasai estimator implementation."""
        fs = 100.0
        transmit_frequency = 15.625e6
        beamforming_sound_velocity = 1540.0
        lag = 1

        # Reference Kasai estimator (average_angle method).
        block_rolled_conjugate = np.roll(sample_iq_block_4d, lag, axis=0).conj()
        block_rolled_conjugate[:lag, ...] = 0
        autocorrelation = sample_iq_block_4d * block_rolled_conjugate
        autocorrelation = autocorrelation[lag:]
        autocorrelation_phase = np.angle(autocorrelation)
        average_phase = autocorrelation_phase.mean(0)
        expected = (
            average_phase
            * fs
            * beamforming_sound_velocity
            / (4 * np.pi * transmit_frequency)
        )

        result = compute_axial_velocity_volume(
            sample_iq_block_4d,
            fs=fs,
            transmit_frequency=transmit_frequency,
            beamforming_sound_velocity=beamforming_sound_velocity,
            lag=lag,
            estimation_method="average_angle",
        )

        assert_allclose(result[0], expected, rtol=1e-5)

    def test_angle_average_method(self, sample_iq_block_4d):
        """Angle_average method computes angle of average autocorrelation."""
        fs = 100.0
        transmit_frequency = 15.625e6
        beamforming_sound_velocity = 1540.0
        lag = 1

        # Reference: angle of average autocorrelation.
        block_rolled_conjugate = np.roll(sample_iq_block_4d, lag, axis=0).conj()
        block_rolled_conjugate[:lag, ...] = 0
        autocorrelation = sample_iq_block_4d * block_rolled_conjugate
        autocorrelation = autocorrelation[lag:]
        average_phase = np.angle(autocorrelation.mean(0))
        expected = (
            average_phase
            * fs
            * beamforming_sound_velocity
            / (4 * np.pi * transmit_frequency)
        )

        result = compute_axial_velocity_volume(
            sample_iq_block_4d,
            fs=fs,
            transmit_frequency=transmit_frequency,
            beamforming_sound_velocity=beamforming_sound_velocity,
            lag=lag,
            estimation_method="angle_average",
        )

        assert_allclose(result[0], expected, rtol=1e-5)

    def test_invalid_estimation_method_raises(self, sample_iq_block_4d):
        """Invalid estimation_method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown estimation method"):
            compute_axial_velocity_volume(
                sample_iq_block_4d,
                fs=100.0,
                estimation_method="invalid",  # type: ignore
            )


class TestProcessIqBlocks:
    """Tests for process_iq_blocks function."""

    def test_defaults_window_width_and_stride_to_chunk_size(self, sample_iq_dataarray):
        """Missing window params default to the time chunk size."""
        iq = da.from_array(sample_iq_dataarray.values, chunks=(5, 4, 6, 8))

        def process_func(block: np.ndarray, **kwargs) -> np.ndarray:
            return block.mean(axis=0, keepdims=True)

        result = process_iq_blocks(iq, process_func=process_func)

        assert result.shape == (4, 4, 6, 8)

    def test_stride_greater_than_width_raises(self, sample_iq_dataarray):
        """window_stride > window_width raises ValueError."""
        iq = sample_iq_dataarray
        dask_iq = da.from_array(iq.data)

        with pytest.raises(ValueError, match="must be less than or equal"):
            process_iq_blocks(
                dask_iq,
                process_func=compute_power_doppler_volume,
                window_width=5,
                window_stride=10,
            )

    def test_warns_when_frames_dropped(self, sample_iq_dataarray):
        """Warns when input volumes don't fit into complete windows."""
        iq = sample_iq_dataarray
        dask_iq = da.from_array(iq.data)

        with pytest.warns(UserWarning, match="input volumes will be dropped"):
            # 20 volumes, width=7, stride=7 -> 2 windows (14 used), 6 dropped.
            process_iq_blocks(
                dask_iq,
                process_func=compute_power_doppler_volume,
                window_width=7,
                window_stride=7,
            )


class TestProcessIqToPowerDoppler:
    """Tests for process_iq_to_power_doppler function."""

    def test_wrong_dimensions_raises(self, rng):
        """DataArray with wrong dimensions raises ValueError."""
        data = rng.random((10, 4, 6)) + 1j * rng.random((10, 4, 6))
        iq = xr.DataArray(
            data,
            dims=("time", "z", "y"),  # Missing x dimension.
            coords={
                "time": np.arange(10),
                "z": np.arange(4),
                "y": np.arange(6),
            },
        )
        with pytest.raises(ValueError, match="Expected dimensions"):
            process_iq_to_power_doppler(iq)

    def test_non_complex_data_raises(self, rng):
        """Non-complex data raises TypeError."""
        data = rng.random((10, 4, 6, 8))  # Real, not complex.
        iq = xr.DataArray(
            data,
            dims=("time", "z", "y", "x"),
            coords={
                "time": np.arange(10),
                "z": np.arange(4),
                "y": np.arange(6),
                "x": np.arange(8),
            },
        )
        with pytest.raises(TypeError, match="complex-valued"):
            process_iq_to_power_doppler(iq)

    def test_output_has_correct_attributes(self, sample_iq_dataarray):
        """Output DataArray has expected attributes."""
        result = process_iq_to_power_doppler(
            sample_iq_dataarray,
            clutter_window_width=10,
            clutter_window_stride=5,
            doppler_window_width=2,
            doppler_window_stride=1,
            low_cutoff=1,
            high_cutoff=8,
        )

        assert result.name == "power_doppler"
        assert result.attrs["units"] == "a.u."
        assert result.attrs["clutter_filters"] == "Index-based SVD [1, 8["
        assert result.attrs["clutter_filter_window_duration"] == pytest.approx(1.0)
        assert result.attrs["clutter_filter_window_stride"] == pytest.approx(0.5)
        assert result.coords["time"].attrs[
            "volume_acquisition_duration"
        ] == pytest.approx(0.2)
        assert result.attrs["power_doppler_integration_stride"] == pytest.approx(0.1)
        assert result.coords["time"].attrs["volume_acquisition_reference"] == "start"

    def test_butterworth_uses_time_coord_step_as_fs(self, sample_iq_dataarray):
        """Butterworth filter design uses the time coordinate step, not scanner provenance."""
        iq = sample_iq_dataarray.copy()
        iq.attrs["compound_sampling_frequency"] = 100.0

        # Time coord step is 0.1 s → fs = 10 Hz, so Nyquist is 5 Hz. If the code used
        # compound_sampling_frequency instead, 30 Hz would incorrectly look valid.
        with pytest.raises(ValueError, match="must be in range"):
            process_iq_to_power_doppler(
                iq,
                clutter_window_width=20,
                filter_method="butterworth",
                low_cutoff=30.0,
            )

    def test_duration_metadata_from_actual_timestamps(self, sample_iq_dataarray):
        """volume_acquisition_duration is computed from actual window timestamps."""
        result = process_iq_to_power_doppler(
            sample_iq_dataarray,
            clutter_window_width=20,
            clutter_window_stride=5,
            doppler_window_width=4,
            doppler_window_stride=2,
        )

        assert result.coords["time"].attrs[
            "volume_acquisition_duration"
        ] == pytest.approx(0.4)

    def test_duration_metadata_set_without_time_coord_units(self, sample_iq_dataarray):
        """volume_acquisition_duration is always set even when time coord has no units."""
        iq = sample_iq_dataarray.assign_coords(
            time=xr.DataArray(np.arange(20) * 0.1, dims=("time",))
        )

        with pytest.warns(
            UserWarning, match="no `units` attribute|compound_sampling_frequency"
        ):
            result = process_iq_to_bmode(
                iq, bmode_window_width=10, bmode_window_stride=5
            )

        assert result.coords["time"].attrs[
            "volume_acquisition_duration"
        ] == pytest.approx(1.0)
        assert result.coords["time"].attrs["volume_acquisition_reference"] == "start"

    def test_duration_metadata_preserves_time_coordinate_units(
        self, sample_iq_dataarray
    ):
        """Window timing metadata stays in the native time-coordinate units."""
        iq = sample_iq_dataarray.assign_coords(
            time=xr.DataArray(
                np.arange(20) * 100.0,
                dims=("time",),
                attrs={"units": "ms"},
            )
        )

        with pytest.warns(UserWarning, match="compound_sampling_frequency"):
            result = process_iq_to_bmode(
                iq, bmode_window_width=10, bmode_window_stride=5
            )

        assert_allclose(result.coords["time"].values, [0.0, 500.0, 1000.0])
        assert result.coords["time"].attrs[
            "volume_acquisition_duration"
        ] == pytest.approx(1000.0)
        assert result.attrs["bmode_integration_duration"] == pytest.approx(1000.0)

    def test_varying_window_durations_warn_and_store_median(self, sample_iq_dataarray):
        """Variable output-window durations warn and store the median metadata value."""
        iq = sample_iq_dataarray.isel(time=slice(0, 5)).assign_coords(
            time=xr.DataArray([0.0, 1.0, 3.0, 4.0, 7.0], dims=("time",))
        )

        with pytest.warns(
            UserWarning,
            match=(
                "no `units` attribute|compound_sampling_frequency|"
                "B-mode integration (duration|stride) varies"
            ),
        ):
            result = process_iq_to_bmode(
                iq, bmode_window_width=2, bmode_window_stride=1
            )

        assert result.coords["time"].attrs[
            "volume_acquisition_duration"
        ] == pytest.approx(1.6)
        assert result.attrs["bmode_integration_duration"] == pytest.approx(1.6)

    def test_butterworth_non_uniform_time_raises(self, sample_iq_dataarray):
        """Butterworth filtering rejects non-uniform time coordinates."""
        iq = sample_iq_dataarray.assign_coords(
            time=xr.DataArray(
                [0.0, 0.1, 0.25, 0.35, 0.5] + list(np.arange(5, 20) * 0.1),
                dims=("time",),
                attrs={"units": "s"},
            )
        )

        with pytest.raises(ValueError, match="regularly sampled `time` coordinate"):
            process_iq_to_power_doppler(
                iq,
                clutter_window_width=4,
                clutter_window_stride=2,
                filter_method="butterworth",
                doppler_window_width=2,
                doppler_window_stride=1,
            )

    def test_single_time_point_has_volume_acquisition_duration(
        self, sample_iq_dataarray
    ) -> None:
        """Single-volume inputs still emit volume_acquisition_duration on the time coordinate."""
        iq = sample_iq_dataarray.isel(time=slice(0, 1))

        result = process_iq_to_bmode(iq, bmode_window_width=1, bmode_window_stride=1)

        assert result.coords["time"].attrs[
            "volume_acquisition_duration"
        ] == pytest.approx(1.0 / iq.attrs["compound_sampling_frequency"])
        assert result.coords["time"].attrs["volume_acquisition_reference"] == "start"

    def test_duration_falls_back_to_compound_sampling_frequency_with_warning(
        self, sample_iq_dataarray
    ) -> None:
        """Missing explicit duration falls back to compound_sampling_frequency with warning."""
        iq = sample_iq_dataarray.copy()
        iq.coords["time"].attrs.pop("volume_acquisition_duration", None)

        with pytest.warns(UserWarning, match="compound_sampling_frequency"):
            result = process_iq_to_bmode(
                iq, bmode_window_width=1, bmode_window_stride=1
            )

        assert result.coords["time"].attrs[
            "volume_acquisition_duration"
        ] == pytest.approx(1.0 / iq.attrs["compound_sampling_frequency"])

    def test_duration_falls_back_to_time_spacing_with_warning(
        self, sample_iq_dataarray
    ) -> None:
        """Missing explicit duration and scanner rate falls back to time spacing."""
        iq = sample_iq_dataarray.copy()
        iq.coords["time"].attrs.pop("volume_acquisition_duration", None)
        iq.attrs.pop("compound_sampling_frequency", None)

        with pytest.warns(
            UserWarning, match="representative `time` coordinate spacing"
        ):
            result = process_iq_to_bmode(
                iq, bmode_window_width=1, bmode_window_stride=1
            )

        assert result.coords["time"].attrs[
            "volume_acquisition_duration"
        ] == pytest.approx(0.1)

    def test_duration_irregular_time_spacing_warns_about_median_approximation(
        self, sample_iq_dataarray
    ) -> None:
        """Irregular timing emits warnings about approximate timing metadata."""
        iq = sample_iq_dataarray.copy()
        iq.coords["time"].attrs.pop("volume_acquisition_duration", None)
        iq.attrs.pop("compound_sampling_frequency", None)
        iq = iq.assign_coords(
            time=xr.DataArray(
                [0.0, 0.1, 0.25, 0.45] + list(np.arange(4, 20) * 0.1),
                dims=("time",),
                attrs={"units": "s"},
            )
        )

        with pytest.warns(
            UserWarning,
            match="median approximation|B-mode integration stride varies",
        ):
            result = process_iq_to_bmode(
                iq, bmode_window_width=1, bmode_window_stride=1
            )

        assert result.coords["time"].attrs[
            "volume_acquisition_duration"
        ] == pytest.approx(np.median(np.diff(iq.coords["time"].values)))

    def test_single_time_point_without_duration_metadata_raises(
        self, sample_iq_dataarray
    ) -> None:
        """A single time point without duration provenance cannot infer acquisition duration."""
        iq = sample_iq_dataarray.isel(time=slice(0, 1)).copy()
        iq.coords["time"].attrs.pop("volume_acquisition_duration", None)
        iq.attrs.pop("compound_sampling_frequency", None)

        with pytest.raises(
            ValueError, match="Cannot determine volume acquisition duration"
        ):
            process_iq_to_bmode(iq, bmode_window_width=1, bmode_window_stride=1)

    def test_accessor_delegates_to_process_iq_to_power_doppler(
        self, sample_iq_dataarray
    ):
        """Xarray accessor calls process_iq_to_power_doppler with the correct arguments."""
        from unittest.mock import patch

        import confusius  # noqa: F401 — registers the fusi accessor.

        iq = sample_iq_dataarray

        with patch("confusius.xarray.iq.process_iq_to_power_doppler") as mock_fn:
            iq.fusi.iq.process_to_power_doppler(
                clutter_window_width=20,
                clutter_window_stride=10,
                filter_method="svd_indices",
                clutter_mask=None,
                low_cutoff=2,
                high_cutoff=8,
                butterworth_order=4,
                doppler_window_width=10,
                doppler_window_stride=5,
            )

        mock_fn.assert_called_once_with(
            iq,
            clutter_window_width=20,
            clutter_window_stride=10,
            filter_method="svd_indices",
            clutter_mask=None,
            low_cutoff=2,
            high_cutoff=8,
            butterworth_order=4,
            doppler_window_width=10,
            doppler_window_stride=5,
        )


class TestProcessIqToAxialVelocity:
    """Tests for process_iq_to_axial_velocity function."""

    def test_missing_required_attributes_raises(self, sample_iq_dataarray):
        """Raises ValueError when required attributes are missing."""
        # Remove required attribute from the iq DataArray.
        sample_iq_dataarray.attrs.pop("transmit_frequency", None)

        with pytest.raises(ValueError, match="Missing required DataArray attributes"):
            process_iq_to_axial_velocity(sample_iq_dataarray)

    def test_output_has_correct_attributes(self, sample_iq_dataarray):
        """Output DataArray has expected attributes."""
        result = process_iq_to_axial_velocity(
            sample_iq_dataarray,
            clutter_window_width=10,
            clutter_window_stride=5,
            velocity_window_width=3,
            velocity_window_stride=1,
            lag=2,
            absolute_velocity=True,
        )

        assert result.name == "axial_velocity"
        assert result.attrs["units"] == "m/s"
        assert result.attrs["clutter_filters"] == "Index-based SVD"
        assert result.attrs["clutter_filter_window_duration"] == pytest.approx(1.0)
        assert result.attrs["clutter_filter_window_stride"] == pytest.approx(0.5)
        assert result.coords["time"].attrs[
            "volume_acquisition_duration"
        ] == pytest.approx(0.3)
        assert result.attrs["axial_velocity_integration_duration"] == pytest.approx(0.3)
        assert result.attrs["axial_velocity_integration_stride"] == pytest.approx(0.1)
        assert result.coords["time"].attrs["volume_acquisition_reference"] == "start"
        assert result.attrs["axial_velocity_lag"] == 2
        assert result.attrs["axial_velocity_absolute"] is True

    def test_uses_attrs_for_parameters(self, sample_iq_dataarray):
        """Uses DataArray attributes for transmit frequency and sound velocity."""
        # Set specific attribute values on the iq DataArray to verify they're used.
        sample_iq_dataarray.attrs["transmit_frequency"] = 10e6
        sample_iq_dataarray.attrs["beamforming_sound_velocity"] = 1500.0

        result = process_iq_to_axial_velocity(sample_iq_dataarray)

        assert result.attrs["transmit_frequency"] == 10e6
        assert result.attrs["beamforming_sound_velocity"] == 1500.0

    def test_axial_velocity_uses_time_coord_without_compound_sampling_frequency(
        self, sample_iq_dataarray
    ) -> None:
        """Axial velocity works without compound_sampling_frequency when timing lives on the time coord."""
        iq = sample_iq_dataarray.copy()
        iq.attrs.pop("compound_sampling_frequency", None)
        iq = iq.assign_coords(
            time=xr.DataArray(
                iq.coords["time"].values,
                dims=("time",),
                attrs={
                    **iq.coords["time"].attrs,
                    "volume_acquisition_duration": 0.1,
                },
            )
        )

        result = process_iq_to_axial_velocity(iq)

        assert result.coords["time"].attrs[
            "volume_acquisition_duration"
        ] == pytest.approx(2.0)

    def test_svd_energy_filter_method(self, sample_iq_dataarray) -> None:
        """Energy-based SVD filtering produces an axial velocity output."""
        result = process_iq_to_axial_velocity(
            sample_iq_dataarray,
            clutter_window_width=10,
            filter_method="svd_energy",
            low_cutoff=0.1,
            high_cutoff=0.9,
        )

        assert result.name == "axial_velocity"

    def test_svd_cumulative_energy_filter_method(self, sample_iq_dataarray) -> None:
        """Cumulative-energy SVD filtering produces a power Doppler output."""
        result = process_iq_to_power_doppler(
            sample_iq_dataarray,
            clutter_window_width=10,
            filter_method="svd_cumulative_energy",
            low_cutoff=0.1,
            high_cutoff=0.9,
        )

        assert result.name == "power_doppler"

    def test_default_clutter_window_width_uses_time_chunk_size(
        self, sample_iq_dataarray
    ) -> None:
        """Missing clutter window width defaults to the time chunk size."""
        iq = sample_iq_dataarray.copy(
            data=da.from_array(sample_iq_dataarray.values, chunks=(5, 4, 6, 8))
        )

        result = process_iq_to_power_doppler(iq)

        assert result is not None

    def test_accessor_delegates_to_process_iq_to_axial_velocity(
        self, sample_iq_dataarray
    ):
        """Xarray accessor calls process_iq_to_axial_velocity with the correct arguments."""
        from unittest.mock import patch

        import confusius  # noqa: F401 — registers the fusi accessor.

        iq = sample_iq_dataarray

        with patch("confusius.xarray.iq.process_iq_to_axial_velocity") as mock_fn:
            iq.fusi.iq.process_to_axial_velocity(
                clutter_window_width=20,
                clutter_window_stride=10,
                filter_method="svd_indices",
                clutter_mask=None,
                low_cutoff=2,
                high_cutoff=8,
                butterworth_order=4,
                velocity_window_width=10,
                velocity_window_stride=5,
                lag=2,
                absolute_velocity=True,
                spatial_kernel=3,
                estimation_method="angle_average",
            )

        mock_fn.assert_called_once_with(
            iq,
            clutter_window_width=20,
            clutter_window_stride=10,
            filter_method="svd_indices",
            clutter_mask=None,
            low_cutoff=2,
            high_cutoff=8,
            butterworth_order=4,
            velocity_window_width=10,
            velocity_window_stride=5,
            lag=2,
            absolute_velocity=True,
            spatial_kernel=3,
            estimation_method="angle_average",
        )

    def test_axial_velocity_output_uses_prefixed_metadata_names(
        self, sample_iq_dataarray
    ):
        """Axial-velocity-specific attrs use explicit `axial_velocity_` prefixes."""
        result = process_iq_to_axial_velocity(
            sample_iq_dataarray,
            lag=2,
            absolute_velocity=True,
            spatial_kernel=3,
            estimation_method="angle_average",
        )

        assert result.attrs["axial_velocity_lag"] == 2
        assert result.attrs["axial_velocity_absolute"] is True
        assert result.attrs["axial_velocity_spatial_kernel"] == 3
        assert result.attrs["axial_velocity_estimation_method"] == "angle_average"
        assert "lag" not in result.attrs
        assert "absolute_velocity" not in result.attrs
        assert "spatial_kernel" not in result.attrs
        assert "estimation_method" not in result.attrs


class TestDataArrayClutterMask:
    """Tests for DataArray clutter mask support in wrapper functions."""

    def test_dataarray_mask_matches_reference_power_doppler(
        self, sample_iq_dataarray, spatial_mask
    ):
        """DataArray mask matches numpy reference for power Doppler."""
        iq = sample_iq_dataarray
        n_time = iq.sizes["time"]

        # Create DataArray mask with matching coordinates.
        mask_dataarray = xr.DataArray(
            spatial_mask,
            dims=("z", "y", "x"),
            coords={
                "z": iq.coords["z"],
                "y": iq.coords["y"],
                "x": iq.coords["x"],
            },
        )

        result = process_iq_to_power_doppler(
            iq,
            clutter_window_width=n_time,
            clutter_window_stride=n_time,
            doppler_window_width=n_time,
            doppler_window_stride=n_time,
            clutter_mask=mask_dataarray,
            low_cutoff=2,
            high_cutoff=8,
        )

        expected = compute_power_doppler_volume(
            iq.values,
            filter_method="svd_indices",
            clutter_mask=mask_dataarray.values,
            low_cutoff=2,
            high_cutoff=8,
        )

        assert_allclose(result.values[0], expected[0])

    def test_dataarray_mask_matches_reference_axial_velocity(
        self, sample_iq_dataarray, spatial_mask
    ):
        """DataArray mask matches numpy reference for axial velocity."""
        iq = sample_iq_dataarray
        n_time = iq.sizes["time"]

        # Create DataArray mask with matching coordinates.
        mask_dataarray = xr.DataArray(
            spatial_mask,
            dims=("z", "y", "x"),
            coords={
                "z": iq.coords["z"],
                "y": iq.coords["y"],
                "x": iq.coords["x"],
            },
        )

        result = process_iq_to_axial_velocity(
            iq,
            clutter_window_width=n_time,
            clutter_window_stride=n_time,
            velocity_window_width=n_time,
            velocity_window_stride=n_time,
            clutter_mask=mask_dataarray,
            low_cutoff=2,
            high_cutoff=8,
        )

        expected = compute_axial_velocity_volume(
            iq.values,
            fs=iq.attrs["compound_sampling_frequency"],
            filter_method="svd_indices",
            clutter_mask=mask_dataarray.values,
            low_cutoff=2,
            high_cutoff=8,
            transmit_frequency=iq.attrs["transmit_frequency"],
            beamforming_sound_velocity=iq.attrs["beamforming_sound_velocity"],
        )

        assert_allclose(result.values[0], expected[0])

    def test_dataarray_mask_coordinate_mismatch_raises(
        self, sample_iq_dataarray, spatial_mask
    ):
        """DataArray mask with mismatched coordinates raises ValueError."""
        iq = sample_iq_dataarray

        # Create mask with different coordinates.
        mask_dataarray = xr.DataArray(
            spatial_mask,
            dims=("z", "y", "x"),
            coords={
                "z": iq.coords["z"] + 1.0,  # Shifted coordinates.
                "y": iq.coords["y"],
                "x": iq.coords["x"],
            },
        )

        with pytest.raises(
            ValueError,
            match=r"does not match between clutter_mask and data",
        ):
            process_iq_to_power_doppler(
                iq,
                clutter_mask=mask_dataarray,
                low_cutoff=2,
                high_cutoff=8,
            )

    def test_dataarray_mask_missing_coordinate_raises(
        self, sample_iq_dataarray, spatial_mask
    ):
        """DataArray mask missing a coordinate raises ValueError."""
        iq = sample_iq_dataarray

        # Create mask missing 'z' coordinate.
        mask_dataarray = xr.DataArray(
            spatial_mask,
            dims=("z", "y", "x"),
            coords={
                "y": iq.coords["y"],
                "x": iq.coords["x"],
            },
        )

        with pytest.raises(ValueError, match=r"is missing from clutter_mask"):
            process_iq_to_power_doppler(
                iq,
                clutter_mask=mask_dataarray,
                low_cutoff=2,
                high_cutoff=8,
            )

    def test_dataarray_mask_shape_mismatch_raises(self, sample_iq_dataarray):
        """DataArray mask with wrong shape raises ValueError."""
        iq = sample_iq_dataarray

        # Create mask with wrong shape.
        wrong_mask = np.ones((2, 2, 2), dtype=bool)
        mask_dataarray = xr.DataArray(
            wrong_mask,
            dims=("z", "y", "x"),
            coords={
                "z": np.arange(2),
                "y": np.arange(2),
                "x": np.arange(2),
            },
        )

        with pytest.raises(
            ValueError,
            match=r"does not match between clutter_mask and data",
        ):
            process_iq_to_power_doppler(
                iq,
                clutter_mask=mask_dataarray,
                low_cutoff=2,
                high_cutoff=8,
            )


class TestComputeBmodeVolume:
    """Tests for compute_bmode_volume function."""

    def test_matches_reference_implementation(self, sample_iq_block_4d):
        """Result matches reference: mean(|block|)."""
        result = compute_bmode_volume(sample_iq_block_4d)

        expected = np.mean(np.abs(sample_iq_block_4d), axis=0)
        # Result has shape (1, z, y, x) due to single window.
        assert_allclose(result[0], expected)

    def test_output_shape(self, sample_iq_block_4d):
        """Output has shape (1, z, y, x)."""
        time, z, y, x = sample_iq_block_4d.shape
        result = compute_bmode_volume(sample_iq_block_4d)

        assert result.shape == (1, z, y, x)

    def test_output_is_real(self, sample_iq_block_4d):
        """Output is real-valued (magnitude, not complex)."""
        result = compute_bmode_volume(sample_iq_block_4d)

        assert np.isrealobj(result)


class TestProcessIqToBmode:
    """Tests for process_iq_to_bmode function."""

    def test_wrong_dimensions_raises(self, rng):
        """DataArray with wrong dimensions raises ValueError."""
        data = rng.random((10, 4, 6)) + 1j * rng.random((10, 4, 6))
        iq = xr.DataArray(
            data,
            dims=("time", "z", "y"),  # Missing x dimension.
            coords={
                "time": np.arange(10),
                "z": np.arange(4),
                "y": np.arange(6),
            },
        )
        with pytest.raises(ValueError, match="Expected dimensions"):
            process_iq_to_bmode(iq)

    def test_non_complex_data_raises(self, rng):
        """Non-complex data raises TypeError."""
        data = rng.random((10, 4, 6, 8))  # Real, not complex.
        iq = xr.DataArray(
            data,
            dims=("time", "z", "y", "x"),
            coords={
                "time": np.arange(10),
                "z": np.arange(4),
                "y": np.arange(6),
                "x": np.arange(8),
            },
        )
        with pytest.raises(TypeError, match="complex-valued"):
            process_iq_to_bmode(iq)

    def test_output_has_correct_attributes(self, sample_iq_dataarray):
        """Output DataArray has expected attributes."""
        result = process_iq_to_bmode(
            sample_iq_dataarray,
            bmode_window_width=10,
            bmode_window_stride=5,
        )

        assert result.name == "bmode"
        assert result.attrs["units"] == "a.u."
        assert result.attrs["bmode_integration_duration"] == pytest.approx(1.0)
        assert result.attrs["bmode_integration_stride"] == pytest.approx(0.5)
        assert result.coords["time"].attrs[
            "volume_acquisition_duration"
        ] == pytest.approx(1.0)
        assert result.coords["time"].attrs["volume_acquisition_reference"] == "start"

    def test_matches_reference_implementation(self, sample_iq_dataarray):
        """Output matches reference mean magnitude computation."""
        iq = sample_iq_dataarray
        n_time = iq.sizes["time"]

        result = process_iq_to_bmode(
            iq,
            bmode_window_width=n_time,
            bmode_window_stride=n_time,
        )

        expected = np.mean(np.abs(iq.values), axis=0)
        assert_allclose(result.values[0], expected)

    def test_default_window_uses_chunk_size(self, sample_iq_dataarray):
        """Default window width falls back to the IQ chunk size."""
        iq = sample_iq_dataarray
        n_time = iq.sizes["time"]

        # Chunk along time so chunksize[0] is known.
        dask_iq = da.from_array(iq.values, chunks=(n_time, -1, -1, -1))
        iq_chunked = iq.copy(data=dask_iq)

        result = process_iq_to_bmode(iq_chunked)

        assert result.sizes["time"] == 1

    def test_accessor_delegates_to_process_iq_to_bmode(self, sample_iq_dataarray):
        """Xarray accessor calls process_iq_to_bmode with the correct arguments."""
        from unittest.mock import patch

        import confusius  # noqa: F401 — registers the fusi accessor.

        iq = sample_iq_dataarray

        with patch("confusius.xarray.iq.process_iq_to_bmode") as mock_fn:
            iq.fusi.iq.process_to_bmode(bmode_window_width=10, bmode_window_stride=5)

        mock_fn.assert_called_once_with(
            iq, bmode_window_width=10, bmode_window_stride=5
        )

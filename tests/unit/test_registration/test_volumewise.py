"""Unit tests for volumewise registration functions."""

from threading import Event

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from confusius.registration.diagnostics import RegistrationDiagnostics
from confusius.registration.volumewise import register_volumewise
from confusius.xarray import create_voxeldata


class _FakeVolumewiseProgressReporter:
    def __init__(self) -> None:
        self.completed_frames: list[int] = []
        self.closed = False

    def frame_completed(
        self,
        frame_index: int,
        registered_frame: xr.DataArray,
        diagnostics: RegistrationDiagnostics,
    ) -> None:
        self.completed_frames.append(frame_index)

    def close(self) -> None:
        self.closed = True


class TestRegisterVolumewise:
    """Tests for register_volumewise function."""

    def test_missing_time_dimension_raises(self):
        """Data without 'time' dimension raises ValueError."""
        data = xr.DataArray(np.zeros((10, 10)), dims=("y", "x"))
        with pytest.raises(ValueError, match="Time dimension 'time' not found"):
            register_volumewise(data)

    def test_h5py_backed_raises_with_parallel_jobs(self, scan_2d):
        """h5py-backed DataArray (from a .scan file) raises TypeError when n_jobs != 1."""
        with pytest.raises(TypeError, match="h5py dataset"):
            register_volumewise(scan_2d, n_jobs=2)

    def test_h5py_backed_works_with_n_jobs_1(self, scan_2d):
        """h5py-backed DataArray (from a .scan file) with n_jobs=1 does not raise."""
        # n_jobs=1 (serial) should not raise for h5py-backed data.
        result = register_volumewise(scan_2d, n_jobs=1, transform="translation")
        assert result.shape == scan_2d.shape

    def test_non_h5py_dask_backed_does_not_raise(self, sample_fusi_2dt_registration):
        """Dask-backed (non-h5py) DataArray with n_jobs != 1 does not raise TypeError."""
        import dask.array as da

        # Build a dask-backed DataArray that is NOT backed by h5py; is_h5py_backed
        # should return False and registration should proceed normally.
        dask_data = xr.DataArray(
            da.from_array(sample_fusi_2dt_registration.values),
            dims=sample_fusi_2dt_registration.dims,
            coords=sample_fusi_2dt_registration.coords,
            attrs=sample_fusi_2dt_registration.attrs,
        )
        result = register_volumewise(dask_data, n_jobs=2, transform="translation")
        assert result.shape == sample_fusi_2dt_registration.shape

    def test_show_progress_false_skips_joblib_progress_import(
        self, sample_fusi_2dt_registration, monkeypatch
    ):
        """show_progress=False does not import joblib_progress."""
        import builtins

        original_import = builtins.__import__

        def _guarded_import(name, *args, **kwargs):
            if name == "joblib_progress":
                raise AssertionError(
                    "joblib_progress should not be imported when show_progress=False"
                )
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _guarded_import)

        result = register_volumewise(
            sample_fusi_2dt_registration,
            n_jobs=1,
            transform="translation",
            show_progress=False,
        )

        assert result.shape == sample_fusi_2dt_registration.shape

    def test_abort_event_returns_partial_dataset(self, sample_fusi_2dt_registration):
        """A pre-set abort event returns an aborted partial dataset."""
        abort_event = Event()
        abort_event.set()

        result = register_volumewise(
            sample_fusi_2dt_registration,
            n_jobs=2,
            transform="translation",
            abort_event=abort_event,
        )

        assert result.shape == sample_fusi_2dt_registration.shape
        assert set(result.attrs["motion_params"]["status"]) == {"aborted"}
        assert_allclose(
            result.values,
            np.full_like(
                sample_fusi_2dt_registration.values,
                sample_fusi_2dt_registration.values.min(),
            ),
        )

    def test_progress_reporter_receives_frame_updates(
        self, sample_fusi_2dt_registration, monkeypatch
    ):
        reporter = _FakeVolumewiseProgressReporter()

        def _fake_register_volume(_volume, _ref_da, **kwargs):
            diagnostics = RegistrationDiagnostics(
                metric="correlation",
                metric_values=np.asarray([-1.0, -0.5]),
                final_metric_value=-0.5,
                n_iterations=2,
                stop_condition="done",
                status="completed",
            )
            return _volume.copy(), np.eye(4), diagnostics

        monkeypatch.setattr(
            "confusius.registration.volumewise.register_volume",
            _fake_register_volume,
        )

        result = register_volumewise(
            sample_fusi_2dt_registration,
            n_jobs=1,
            transform="translation",
            show_progress=False,
            progress_reporter=reporter,
        )

        assert result.shape == sample_fusi_2dt_registration.shape
        assert sorted(reporter.completed_frames) == list(
            range(sample_fusi_2dt_registration.sizes["time"])
        )
        assert reporter.closed

    def test_abort_during_run_skips_not_yet_started_frames(
        self, sample_fusi_2dt_registration, monkeypatch
    ):
        """Already-scheduled frames hit the cheap aborted-frame fast path."""
        import joblib

        abort_event = Event()
        calls = {"count": 0}

        def _fake_register_volume(volume, _ref_da, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                abort_event.set()
            diagnostics = RegistrationDiagnostics(
                metric="correlation",
                metric_values=np.asarray([-1.0]),
                final_metric_value=-1.0,
                n_iterations=1,
                stop_condition="done",
                status="completed",
            )
            return volume.copy(), np.eye(4), diagnostics

        class _FakeParallel:
            def __init__(self, *args, **kwargs):
                del args, kwargs

            def __call__(self, tasks):
                scheduled = list(tasks)

                def _run():
                    for task in scheduled:
                        yield task()

                return _run()

        def _fake_delayed(func):
            def _wrap(*args, **kwargs):
                return lambda: func(*args, **kwargs)

            return _wrap

        monkeypatch.setattr(
            "confusius.registration.volumewise.register_volume",
            _fake_register_volume,
        )
        monkeypatch.setattr(joblib, "Parallel", _FakeParallel)
        monkeypatch.setattr(joblib, "delayed", _fake_delayed)

        result = register_volumewise(
            sample_fusi_2dt_registration,
            n_jobs=2,
            transform="translation",
            show_progress=False,
            abort_event=abort_event,
        )

        statuses = list(result.attrs["motion_params"]["status"])
        assert statuses[0] == "completed"
        assert all(status == "aborted" for status in statuses[1:])
        assert calls["count"] == 1

        background = sample_fusi_2dt_registration.values.min()
        assert np.all(result.values[1:] == background)

    def test_wrong_dimensionality_raises(self):
        """Data that is neither 2D+t nor 3D+t raises ValueError."""
        # 1D+time = 2D total.
        data = xr.DataArray(np.zeros((5, 10)), dims=("time", "i"))
        with pytest.raises(
            ValueError,
            match="at least 2 spatial dimensions|native voxel|missing voxel dimension",
        ):
            register_volumewise(data)

    @pytest.mark.parametrize(
        ("data_fixture", "dims"),
        [
            ("sample_fusi_2dt_registration", ("time", "k", "j", "i")),
            ("sample_fusi_3dt_registration", ("time", "k", "j", "i")),
        ],
    )
    def test_identical_frames_unchanged(self, data_fixture, dims, request):
        """Identical frames remain unchanged after registration (2D and 3D)."""
        data = request.getfixturevalue(data_fixture)
        result = register_volumewise(data, n_jobs=1, transform="translation")

        assert result.dims == dims
        assert result.shape == data.shape
        # Identical frames should produce nearly identical output.
        assert_allclose(result.values, data.values, atol=1e-3)

    def test_2d_recovers_known_shift(self, sample_fusi_2d_registration):
        """Registration of a singleton-k volume recovers a known translation."""
        # Create data with a shifted frame.
        n_frames = 3
        shift_x, shift_y = 2, 3

        frames = [sample_fusi_2d_registration.values.copy() for _ in range(n_frames)]
        # Shift frame 1 by rolling (simulates translation).
        frames[1] = np.roll(np.roll(frames[1], shift_y, axis=1), shift_x, axis=2)

        data = create_voxeldata(
            np.stack(frames, axis=0),
            dims=("time", "k", "j", "i"),
            time=np.arange(n_frames) * 0.1,
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            volume_acquisition_duration=0.1,
        )

        result = register_volumewise(
            data, reference_time=0, n_jobs=1, transform="translation"
        )

        # Check motion parameters recovered the shift.
        motion_df = result.attrs["motion_params"]
        # Frame 1 should have approximately the opposite translation.
        assert abs(motion_df.loc[motion_df.index[1], "trans_x"]) < shift_x + 1
        assert abs(motion_df.loc[motion_df.index[1], "trans_y"]) < shift_y + 1

    def test_output_has_motion_metadata_attributes(self, sample_fusi_2dt_registration):
        """Output has motion metadata attributes."""
        result = register_volumewise(
            sample_fusi_2dt_registration, reference_time=2, n_jobs=1
        )

        assert "registration" not in result.attrs
        assert result.attrs["reference_time"] == 2
        assert "motion_params" in result.attrs

    def test_preserves_input_attributes(self, sample_fusi_2dt_registration):
        """Input attributes are preserved in output."""
        sample_fusi_2dt_registration.attrs["custom_attr"] = "test_value"

        result = register_volumewise(sample_fusi_2dt_registration, n_jobs=1)

        assert result.attrs["custom_attr"] == "test_value"

    def test_preserves_coordinates(self, sample_fusi_2dt_registration):
        """Coordinates are preserved in output."""
        result = register_volumewise(sample_fusi_2dt_registration, n_jobs=1)

        assert_allclose(
            result.coords["time"].values,
            sample_fusi_2dt_registration.coords["time"].values,
        )
        assert_allclose(
            result.coords["y"].values, sample_fusi_2dt_registration.coords["y"].values
        )
        assert_allclose(
            result.coords["x"].values, sample_fusi_2dt_registration.coords["x"].values
        )

    def test_different_reference_time(self, sample_fusi_2dt_registration):
        """Can use different reference time indices."""
        result = register_volumewise(
            sample_fusi_2dt_registration, reference_time=2, n_jobs=1
        )

        assert result.attrs["reference_time"] == 2

    def test_transform_option(self, sample_fusi_2dt_registration):
        """transform parameter changes registration behavior."""
        # Both should work without error.
        result_no_rot = register_volumewise(
            sample_fusi_2dt_registration, n_jobs=1, transform="translation"
        )
        result_with_rot = register_volumewise(
            sample_fusi_2dt_registration, n_jobs=1, transform="rigid"
        )

        # Motion params should have rotation columns in both cases.
        assert "rot_x" in result_no_rot.attrs["motion_params"].columns
        assert "rot_x" in result_with_rot.attrs["motion_params"].columns

    def test_singleton_dimension_handling(self, sample_fusi_2d_registration):
        """Singleton spatial dimensions are handled correctly."""
        # Create data with a singleton k dimension (2D slice in 3D array).
        data = create_voxeldata(
            sample_fusi_2d_registration.values[np.newaxis, :, :, :].repeat(3, axis=0),
            dims=("time", "k", "j", "i"),
            time=np.arange(3) * 0.1,
            spacing=(0.2, 0.1, 0.1),
            origin=(0.0, 0.0, 0.0),
            volume_acquisition_duration=0.1,
        )

        result = register_volumewise(data, n_jobs=1)

        # Should preserve the singleton dimension.
        assert result.dims == data.dims
        assert result.shape == data.shape
        assert result.sizes["k"] == 1
        # Identical frames should produce nearly identical output.
        assert_allclose(result.values, data.values, atol=1e-3)

    def test_output_dimension_order_matches_input(self, sample_fusi_2d_registration):
        """Output dimension order matches input regardless of internal transposition."""
        # Create data with non-standard dimension order.
        data = create_voxeldata(
            np.stack([sample_fusi_2d_registration.values] * 3, axis=0),
            dims=("time", "k", "j", "i"),
            time=np.arange(3) * 0.1,
            spacing=(0.2, 0.1, 0.1),
            origin=(0.0, 0.0, 0.0),
            volume_acquisition_duration=0.1,
        ).transpose("k", "j", "i", "time")

        result = register_volumewise(data, n_jobs=1)

        assert result.dims == ("k", "j", "i", "time")
        # Identical frames should produce nearly identical output.
        assert_allclose(result.values, data.values, atol=1e-3)

    def test_multi_resolution_does_not_crash(self, sample_fusi_3dt_registration):
        """Multi-resolution pyramid completes without error."""
        result = register_volumewise(
            sample_fusi_3dt_registration,
            n_jobs=1,
            transform="translation",
            use_multi_resolution=True,
        )
        assert result.shape == sample_fusi_3dt_registration.shape
        # Identical frames should produce nearly identical output.
        assert_allclose(result.values, sample_fusi_3dt_registration.values, atol=1e-3)

    def test_keep_diagnostics_toggles_full_trace(self, sample_fusi_2dt_registration):
        """`keep_diagnostics` gates only the full diagnostics list.

        The cheap per-frame summaries (`final_metric_value`, `n_iterations`)
        are always attached to `motion_params`; only the
        memory-hungry trace list is opt-in.
        """
        # Default (False): summary columns yes, full diagnostics list no.
        result_off = register_volumewise(sample_fusi_2dt_registration, n_jobs=1)
        assert "registration_diagnostics" not in result_off.attrs
        motion_df_off = result_off.attrs["motion_params"]
        assert "final_metric_value" in motion_df_off.columns
        assert "n_iterations" in motion_df_off.columns

        # Opt-in: full diagnostics list is also attached.
        result_on = register_volumewise(
            sample_fusi_2dt_registration, n_jobs=1, keep_diagnostics=True
        )
        diagnostics = result_on.attrs["registration_diagnostics"]
        assert len(diagnostics) == sample_fusi_2dt_registration.sizes["time"]
        assert all(isinstance(d, RegistrationDiagnostics) for d in diagnostics)

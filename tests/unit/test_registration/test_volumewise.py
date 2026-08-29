"""Unit tests for volumewise registration functions."""

from threading import Event

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from confusius.registration.diagnostics import RegistrationDiagnostics
from confusius.registration.volumewise import register_volumewise
from confusius.validation import ensure_voxeldata
from confusius.xarray import create_voxeldata

SHIFT_X, SHIFT_Y = 2, 3
"""Voxel shift applied to frame 1 of the synthetic shifted-frame recording."""


def _create_shifted_frames_data(volume: xr.DataArray) -> xr.DataArray:
    """Stack three copies of `volume` in time, rolling frame 1 by a known shift.

    Parameters
    ----------
    volume : xarray.DataArray
        Singleton-k VoxelData volume to replicate.

    Returns
    -------
    xarray.DataArray
        `(3, 1, J, I)` VoxelData recording with unit spacing whose frame 1 is
        translated by `(SHIFT_Y, SHIFT_X)` voxels relative to frames 0 and 2.
    """
    frames = [volume.values.copy() for _ in range(3)]
    frames[1] = np.roll(np.roll(frames[1], SHIFT_Y, axis=1), SHIFT_X, axis=2)
    return create_voxeldata(
        np.stack(frames, axis=0),
        dims=("time", "k", "j", "i"),
        time=np.arange(3) * 0.1,
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        volume_acquisition_duration=0.1,
    )


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

    @pytest.mark.parametrize("intensity_scaling", ["gamma", 0.0, -1.0])
    def test_invalid_intensity_scaling_raises(
        self, sample_voxeldata_2dt_registration, intensity_scaling
    ):
        """An unknown mode or non-positive exponent raises ValueError."""
        with pytest.raises(ValueError, match="Invalid intensity_scaling"):
            register_volumewise(
                sample_voxeldata_2dt_registration, intensity_scaling=intensity_scaling
            )

    def test_float_intensity_scaling_is_forwarded_to_register_volume(
        self, sample_voxeldata_2dt_registration, monkeypatch
    ):
        """intensity_scaling is forwarded as both fixed and moving scaling."""
        import confusius.registration.volumewise as volumewise_module

        calls: list[tuple[float | str, float | str]] = []
        original_register_volume = volumewise_module.register_volume

        def spy_register_volume(
            *args,
            fixed_intensity_scaling="none",
            moving_intensity_scaling="none",
            **kwargs,
        ):
            calls.append((fixed_intensity_scaling, moving_intensity_scaling))
            return original_register_volume(
                *args,
                fixed_intensity_scaling=fixed_intensity_scaling,
                moving_intensity_scaling=moving_intensity_scaling,
                **kwargs,
            )

        monkeypatch.setattr(volumewise_module, "register_volume", spy_register_volume)

        register_volumewise(
            sample_voxeldata_2dt_registration,
            n_jobs=1,
            transform="translation",
            intensity_scaling=2.0,
        )

        assert calls and all(scaling == (2.0, 2.0) for scaling in calls)

    def test_h5py_backed_raises_with_parallel_jobs(self, scan_2d):
        """h5py-backed DataArray (from a .scan file) raises TypeError when n_jobs != 1."""
        with pytest.raises(TypeError, match="h5py dataset"):
            register_volumewise(scan_2d, n_jobs=2)

    def test_h5py_backed_works_with_n_jobs_1(self, scan_2d):
        """h5py-backed DataArray (from a .scan file) with n_jobs=1 does not raise."""
        # n_jobs=1 (serial) should not raise for h5py-backed data.
        result = register_volumewise(scan_2d, n_jobs=1, transform="translation")
        assert result.shape == scan_2d.shape

    def test_non_h5py_dask_backed_does_not_raise(
        self, sample_voxeldata_2dt_registration
    ):
        """Dask-backed (non-h5py) DataArray with n_jobs != 1 does not raise TypeError."""
        import dask.array as da

        # Build a dask-backed DataArray that is NOT backed by h5py; is_h5py_backed
        # should return False and registration should proceed normally.
        dask_data = xr.DataArray(
            da.from_array(sample_voxeldata_2dt_registration.values),
            dims=sample_voxeldata_2dt_registration.dims,
            coords=sample_voxeldata_2dt_registration.coords,
            attrs=sample_voxeldata_2dt_registration.attrs,
        )
        result = register_volumewise(dask_data, n_jobs=2, transform="translation")
        assert result.shape == sample_voxeldata_2dt_registration.shape

    def test_show_progress_false_skips_joblib_progress_import(
        self, sample_voxeldata_2dt_registration, monkeypatch
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
            sample_voxeldata_2dt_registration,
            n_jobs=1,
            transform="translation",
            show_progress=False,
        )

        assert result.shape == sample_voxeldata_2dt_registration.shape

    def test_abort_event_returns_partial_dataset(
        self, sample_voxeldata_2dt_registration
    ):
        """A pre-set abort event returns an aborted partial dataset."""
        abort_event = Event()
        abort_event.set()

        result = register_volumewise(
            sample_voxeldata_2dt_registration,
            n_jobs=2,
            transform="translation",
            abort_event=abort_event,
        )

        assert result.shape == sample_voxeldata_2dt_registration.shape
        assert set(result.attrs["motion_params"]["status"]) == {"aborted"}
        assert_allclose(
            result.values,
            np.full_like(
                sample_voxeldata_2dt_registration.values,
                sample_voxeldata_2dt_registration.values.min(),
            ),
        )

    def test_progress_reporter_receives_frame_updates(
        self, sample_voxeldata_2dt_registration, monkeypatch
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
            sample_voxeldata_2dt_registration,
            n_jobs=1,
            transform="translation",
            show_progress=False,
            progress_reporter=reporter,
        )

        assert result.shape == sample_voxeldata_2dt_registration.shape
        assert sorted(reporter.completed_frames) == list(
            range(sample_voxeldata_2dt_registration.sizes["time"])
        )
        assert reporter.closed

    def test_abort_during_run_skips_not_yet_started_frames(
        self, sample_voxeldata_2dt_registration, monkeypatch
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
            sample_voxeldata_2dt_registration,
            n_jobs=2,
            transform="translation",
            show_progress=False,
            abort_event=abort_event,
        )

        statuses = list(result.attrs["motion_params"]["status"])
        assert statuses[0] == "completed"
        assert all(status == "aborted" for status in statuses[1:])
        assert calls["count"] == 1

        background = sample_voxeldata_2dt_registration.values.min()
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
            ("sample_voxeldata_2dt_registration", ("time", "k", "j", "i")),
            ("sample_voxeldata_3dt_registration", ("time", "k", "j", "i")),
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

    def test_2d_recovers_known_shift(self, sample_voxeldata_2d_registration):
        """Registration of a singleton-k volume recovers a known translation."""
        data = _create_shifted_frames_data(sample_voxeldata_2d_registration)

        result = register_volumewise(
            data,
            reference_time=0,
            n_jobs=1,
            transform="translation",
            learning_rate=1.0,
            show_progress=False,
        )

        motion_df = result.attrs["motion_params"]
        assert motion_df.iloc[1]["trans_x"] == pytest.approx(SHIFT_X, abs=0.1)
        assert motion_df.iloc[1]["trans_y"] == pytest.approx(SHIFT_Y, abs=0.1)

    def test_output_has_motion_metadata_attributes(
        self, sample_voxeldata_2dt_registration
    ):
        """Output has motion metadata attributes."""
        result = register_volumewise(
            sample_voxeldata_2dt_registration, reference_time=2, n_jobs=1
        )

        assert "registration" not in result.attrs
        assert result.attrs["reference_time"] == 2
        assert "motion_params" in result.attrs

    def test_preserves_input_attributes(self, sample_voxeldata_2dt_registration):
        """Input attributes are preserved in output."""
        sample_voxeldata_2dt_registration.attrs["custom_attr"] = "test_value"

        result = register_volumewise(sample_voxeldata_2dt_registration, n_jobs=1)

        assert result.attrs["custom_attr"] == "test_value"

    def test_preserves_coordinates(self, sample_voxeldata_2dt_registration):
        """Coordinates and VoxelData geometry are preserved in output."""
        result = register_volumewise(sample_voxeldata_2dt_registration, n_jobs=1)

        ensure_voxeldata(result)
        assert result.dims == sample_voxeldata_2dt_registration.dims
        for coord in ("time", "k", "j", "i", "z", "y", "x"):
            assert_allclose(
                result.coords[coord].values,
                sample_voxeldata_2dt_registration.coords[coord].values,
            )
        assert_allclose(
            result.fusi.affine.voxel_to_world,
            sample_voxeldata_2dt_registration.fusi.affine.voxel_to_world,
        )

    def test_different_reference_time(self, sample_voxeldata_2dt_registration):
        """Can use different reference time indices."""
        result = register_volumewise(
            sample_voxeldata_2dt_registration, reference_time=2, n_jobs=1
        )

        assert result.attrs["reference_time"] == 2

    def test_fixed_frame_matches_reference_time(self, sample_voxeldata_2d_registration):
        """Passing a frame of `data` as `fixed` matches `reference_time` on that frame."""
        data = _create_shifted_frames_data(sample_voxeldata_2d_registration)

        by_index = register_volumewise(
            data,
            reference_time=2,
            n_jobs=1,
            transform="translation",
            learning_rate=1.0,
            show_progress=False,
        )
        by_volume = register_volumewise(
            data,
            fixed=data.isel(time=2),
            n_jobs=1,
            transform="translation",
            learning_rate=1.0,
            show_progress=False,
        )

        assert_allclose(by_volume.values, by_index.values)
        pd.testing.assert_frame_equal(
            by_volume.attrs["motion_params"], by_index.attrs["motion_params"]
        )
        assert by_index.attrs["reference_time"] == 2
        assert by_volume.attrs["reference_time"] is None

    def test_custom_fixed_volume_recovers_known_shift(
        self, sample_voxeldata_2d_registration
    ):
        """A custom `fixed` volume (not a frame index) is the registration target."""
        data = _create_shifted_frames_data(sample_voxeldata_2d_registration)
        fixed = data.mean("time")

        result = register_volumewise(
            data,
            fixed=fixed,
            n_jobs=1,
            transform="translation",
            learning_rate=1.0,
            show_progress=False,
        )

        ensure_voxeldata(result)
        assert result.dims == data.dims
        assert_allclose(result.coords["time"].values, data.coords["time"].values)
        assert result.attrs["reference_time"] is None

        # The target blends the shifted and unshifted squares, so frame 1 is
        # pulled most of the way back rather than exactly by the shift.
        motion_df = result.attrs["motion_params"]
        assert motion_df.iloc[1]["trans_x"] > SHIFT_X / 2
        assert motion_df.iloc[1]["trans_y"] > SHIFT_Y / 2

        # The shifted frame must end up closer to `fixed` than it started.
        def _corr_with_fixed(frame):
            return np.corrcoef(frame.ravel(), fixed.values.ravel())[0, 1]

        assert _corr_with_fixed(result.values[1]) > _corr_with_fixed(data.values[1])

    def test_h5py_backed_fixed_works_with_parallel_jobs(self, scan_2d):
        """A lazily loaded h5py-backed `fixed` works with joblib workers."""
        result = register_volumewise(
            scan_2d.compute(),
            fixed=scan_2d.isel(time=0),
            n_jobs=2,
            transform="translation",
            show_progress=False,
        )

        assert result.attrs["reference_time"] is None
        assert set(result.attrs["motion_params"]["status"]) == {"completed"}

    def test_fixed_and_reference_time_raises(self, sample_voxeldata_2dt_registration):
        """Passing both `reference_time` and `fixed` raises ValueError."""
        data = sample_voxeldata_2dt_registration
        with pytest.raises(ValueError, match="not both"):
            register_volumewise(data, reference_time=0, fixed=data.isel(time=0))

    def test_fixed_with_time_dimension_raises(self, sample_voxeldata_2dt_registration):
        """A `fixed` volume that still has a `time` dimension raises ValueError."""
        data = sample_voxeldata_2dt_registration
        with pytest.raises(ValueError, match="time dimension"):
            register_volumewise(data, fixed=data.isel(time=slice(0, 2)))

    @pytest.mark.parametrize("mismatch", ["spacing", "shape"])
    def test_fixed_on_different_grid_raises(
        self,
        sample_voxeldata_2d_registration,
        sample_voxeldata_2dt_registration,
        mismatch,
    ):
        """A `fixed` volume on another voxel grid raises ValueError."""
        if mismatch == "spacing":
            fixed = create_voxeldata(
                sample_voxeldata_2d_registration.values,
                dims=("k", "j", "i"),
                spacing=(1.0, 0.2, 0.1),
                origin=(0.0, 0.0, 0.0),
            )
        else:
            fixed = sample_voxeldata_2d_registration.isel(j=slice(0, 16))

        with pytest.raises(ValueError, match="voxel grid"):
            register_volumewise(sample_voxeldata_2dt_registration, fixed=fixed)

    def test_fixed_non_voxeldata_raises(
        self, sample_voxeldata_2d_registration, sample_voxeldata_2dt_registration
    ):
        """A plain DataArray without VoxelData geometry as `fixed` raises ValueError."""
        fixed = xr.DataArray(
            sample_voxeldata_2d_registration.values, dims=("k", "j", "i")
        )
        with pytest.raises(ValueError, match="VoxelToWorldIndex|native voxel"):
            register_volumewise(sample_voxeldata_2dt_registration, fixed=fixed)

    def test_transform_option(self, sample_voxeldata_2dt_registration):
        """transform parameter changes registration behavior."""
        # Both should work without error.
        result_no_rot = register_volumewise(
            sample_voxeldata_2dt_registration, n_jobs=1, transform="translation"
        )
        result_with_rot = register_volumewise(
            sample_voxeldata_2dt_registration, n_jobs=1, transform="rigid"
        )

        # Motion params should have rotation columns in both cases.
        assert "rot_x" in result_no_rot.attrs["motion_params"].columns
        assert "rot_x" in result_with_rot.attrs["motion_params"].columns

    def test_singleton_dimension_handling(self, sample_voxeldata_2d_registration):
        """Singleton spatial dimensions are handled correctly."""
        # Create data with a singleton k dimension (2D slice in 3D array).
        data = create_voxeldata(
            sample_voxeldata_2d_registration.values[np.newaxis, :, :, :].repeat(
                3, axis=0
            ),
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

    def test_output_dimension_order_matches_input(
        self, sample_voxeldata_2d_registration
    ):
        """Output dimension order matches input regardless of internal transposition."""
        # Create data with non-standard dimension order.
        data = create_voxeldata(
            np.stack([sample_voxeldata_2d_registration.values] * 3, axis=0),
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

    def test_multi_resolution_does_not_crash(self, sample_voxeldata_3dt_registration):
        """Multi-resolution pyramid completes without error."""
        result = register_volumewise(
            sample_voxeldata_3dt_registration,
            n_jobs=1,
            transform="translation",
            use_multi_resolution=True,
        )
        assert result.shape == sample_voxeldata_3dt_registration.shape
        # Identical frames should produce nearly identical output.
        assert_allclose(
            result.values, sample_voxeldata_3dt_registration.values, atol=1e-3
        )

    def test_keep_diagnostics_toggles_full_trace(
        self, sample_voxeldata_2dt_registration
    ):
        """`keep_diagnostics` gates only the full diagnostics list.

        The cheap per-frame summaries (`final_metric_value`, `n_iterations`)
        are always attached to `motion_params`; only the
        memory-hungry trace list is opt-in.
        """
        # Default (False): summary columns yes, full diagnostics list no.
        result_off = register_volumewise(sample_voxeldata_2dt_registration, n_jobs=1)
        assert "registration_diagnostics" not in result_off.attrs
        motion_df_off = result_off.attrs["motion_params"]
        assert "final_metric_value" in motion_df_off.columns
        assert "n_iterations" in motion_df_off.columns

        # Opt-in: full diagnostics list is also attached.
        result_on = register_volumewise(
            sample_voxeldata_2dt_registration, n_jobs=1, keep_diagnostics=True
        )
        diagnostics = result_on.attrs["registration_diagnostics"]
        assert len(diagnostics) == sample_voxeldata_2dt_registration.sizes["time"]
        assert all(isinstance(d, RegistrationDiagnostics) for d in diagnostics)

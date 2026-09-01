"""Unit tests for volumewise registration functions."""

import numpy as np
import pytest
import xarray as xr
from distributed import Client, Event
from numpy.testing import assert_allclose

from confusius.registration.diagnostics import RegistrationDiagnostics
from confusius.registration.volumewise import register_volumewise
from confusius.validation import ensure_voxeldata
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


@pytest.fixture(scope="class")
def ambient_client():
    """A small shared distributed Client, reused across one test class.

    Most tests below exercise `register_volumewise`'s default (no `abort_event`)
    code path, which uses whatever `distributed.Client` is active
    (`distributed.get_client()`) rather than creating -- and tearing down -- its
    own for every call. Sharing one small Client across a class keeps the suite
    fast. Class-scoped (not module-scoped) so `TestRegisterVolumewiseClientManagement`
    below, which never requests this fixture, is guaranteed no ambient client is
    active -- covering the auto-creation path too.
    """
    with Client(
        n_workers=2, threads_per_worker=1, processes=True, dashboard_address=":0"
    ) as client:
        yield client


@pytest.fixture
def thread_client():
    """A thread-based (in-process) distributed Client, for monkeypatch-visible tests.

    `distributed.Client.submit` always pickles the submitted call, even for a
    thread-based/in-process worker -- but a thread-based worker unpickles it back
    into the *same* Python process, so a name lookup inside the task (e.g. the
    `register_volume` reference `_register_one` resolves from
    `confusius.registration.volumewise`'s module globals) sees this process's
    already-imported (and monkeypatched) module object. A process-based worker
    would import its own fresh, unpatched copy instead. Function-scoped (not
    shared) since tests using this monkeypatch module state that should not leak
    to other tests.
    """
    with Client(
        n_workers=1, threads_per_worker=2, processes=False, dashboard_address=":0"
    ) as client:
        yield client


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
        self, sample_voxeldata_2dt_registration, monkeypatch, thread_client
    ):
        """intensity_scaling is forwarded as both fixed and moving scaling."""
        import confusius.registration.volumewise as volumewise_module

        original_register_volume = volumewise_module.register_volume

        def spy_register_volume(
            *args,
            fixed_intensity_scaling="none",
            moving_intensity_scaling="none",
            **kwargs,
        ):
            # distributed.Client.submit always pickles the submitted call -- even
            # for a thread-based/in-process worker -- so a spy can't report back
            # through ordinary closure-captured state (e.g. appending to a list):
            # each submission gets an independently deep-copied reconstruction of
            # that state. Stash the observed values in the real return value
            # instead (stop_condition), which *does* correctly flow back through
            # future.result().
            registered_da, affine, diagnostics = original_register_volume(
                *args,
                fixed_intensity_scaling=fixed_intensity_scaling,
                moving_intensity_scaling=moving_intensity_scaling,
                **kwargs,
            )
            from dataclasses import replace

            diagnostics = replace(
                diagnostics,
                stop_condition=f"{fixed_intensity_scaling},{moving_intensity_scaling}",
            )
            return registered_da, affine, diagnostics

        monkeypatch.setattr(volumewise_module, "register_volume", spy_register_volume)

        # thread_client: the monkeypatch (applied to this process's module object)
        # is only visible to code that runs in this same process -- a process-based
        # distributed.Client (the default) runs frames on separate worker processes,
        # which import their own fresh, unpatched copy of this module.
        result = register_volumewise(
            sample_voxeldata_2dt_registration,
            transform="translation",
            intensity_scaling=2.0,
            keep_diagnostics=True,
        )

        observed = [d.stop_condition for d in result.attrs["registration_diagnostics"]]
        assert observed and all(scaling == "2.0,2.0" for scaling in observed)

    def test_h5py_backed_raises(self, scan_2d, ambient_client):
        """h5py-backed DataArray raises TypeError.

        Registration always runs through the ambient/auto-created
        distributed.Client, which cannot pickle an h5py dataset to send it to a
        worker.
        """
        with pytest.raises(TypeError, match="h5py dataset"):
            register_volumewise(scan_2d, transform="translation")

    def test_non_h5py_dask_backed_does_not_raise(
        self, sample_voxeldata_2dt_registration, ambient_client
    ):
        """Dask-backed (non-h5py) DataArray does not raise TypeError."""
        import dask.array as da

        # Build a dask-backed DataArray that is NOT backed by h5py; is_h5py_backed
        # should return False and registration should proceed normally.
        dask_data = xr.DataArray(
            da.from_array(sample_voxeldata_2dt_registration.values),
            dims=sample_voxeldata_2dt_registration.dims,
            coords=sample_voxeldata_2dt_registration.coords,
            attrs=sample_voxeldata_2dt_registration.attrs,
        )
        result = register_volumewise(dask_data, transform="translation")
        assert result.shape == sample_voxeldata_2dt_registration.shape

    def test_show_progress_false_skips_progress_bar(
        self, sample_voxeldata_2dt_registration, monkeypatch, ambient_client
    ):
        """show_progress=False never constructs a progress bar."""
        import confusius.registration.volumewise as volumewise_module

        def _guarded_progress(*args, **kwargs):
            raise AssertionError(
                "Progress should not be created when show_progress=False"
            )

        monkeypatch.setattr(volumewise_module, "Progress", _guarded_progress)

        result = register_volumewise(
            sample_voxeldata_2dt_registration,
            transform="translation",
            show_progress=False,
        )

        assert result.shape == sample_voxeldata_2dt_registration.shape

    def test_abort_event_returns_partial_dataset(
        self, sample_voxeldata_2dt_registration, ambient_client
    ):
        """A pre-set abort event returns an aborted partial dataset."""
        abort_event = Event()
        abort_event.set()

        result = register_volumewise(
            sample_voxeldata_2dt_registration,
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
        self, sample_voxeldata_2dt_registration, monkeypatch, thread_client
    ):
        """progress_reporter is notified once per completed frame, then closed."""
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

        # See test_float_intensity_scaling_is_forwarded_to_register_volume for why
        # thread_client is needed for this monkeypatch to take effect.
        result = register_volumewise(
            sample_voxeldata_2dt_registration,
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
        import confusius.registration.volumewise as volumewise_module

        # A single thread (not the shared thread_client fixture's 2) makes frame
        # execution strictly sequential, so the first frame's call to
        # abort_event.set() is guaranteed to be visible to every subsequent
        # (not-yet-started) frame's cheap short-circuit check.
        #
        # abort_event is a distributed.Event, so it stays live (backed by the
        # scheduler) across the pickle boundary -- but ordinary closure-captured
        # state (e.g. a plain call counter) does not: distributed.Client.submit
        # always pickles the submitted call, even for a thread-based/in-process
        # worker, so any such state is deep-copied per submission rather than
        # shared. "exactly one real call happened" is therefore verified below
        # through the actual registration output (n_iterations/status), not a
        # counter.
        with Client(
            n_workers=1, threads_per_worker=1, processes=False, dashboard_address=":0"
        ):
            abort_event = Event()

            def _fake_register_volume(volume, _ref_da, **kwargs):
                # Single worker thread + this abort_event.set() on the only call
                # that actually runs (frame 0; every later frame's _register_one
                # sees abort_event already set and short-circuits before reaching
                # this function at all) is what makes "exactly one real call"
                # deterministic here.
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

            monkeypatch.setattr(
                volumewise_module, "register_volume", _fake_register_volume
            )

            result = register_volumewise(
                sample_voxeldata_2dt_registration,
                transform="translation",
                show_progress=False,
                abort_event=abort_event,
            )

        statuses = list(result.attrs["motion_params"]["status"])
        n_iterations = list(result.attrs["motion_params"]["n_iterations"])
        assert statuses[0] == "completed"
        assert n_iterations[0] == 1
        assert all(status == "aborted" for status in statuses[1:])
        assert all(n == 0 for n in n_iterations[1:])

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
    def test_identical_frames_unchanged(
        self, data_fixture, dims, request, ambient_client
    ):
        """Identical frames remain unchanged after registration (2D and 3D)."""
        data = request.getfixturevalue(data_fixture)
        result = register_volumewise(data, transform="translation")

        assert result.dims == dims
        assert result.shape == data.shape
        # Identical frames should produce nearly identical output.
        assert_allclose(result.values, data.values, atol=1e-3)

    def test_2d_recovers_known_shift(
        self, sample_voxeldata_2d_registration, ambient_client
    ):
        """Registration of a singleton-k volume recovers a known translation."""
        # Create data with a shifted frame.
        n_frames = 3
        shift_x, shift_y = 2, 3

        frames = [
            sample_voxeldata_2d_registration.values.copy() for _ in range(n_frames)
        ]
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

        result = register_volumewise(data, reference_time=0, transform="translation")

        # Check motion parameters recovered the shift.
        motion_df = result.attrs["motion_params"]
        # Frame 1 should have approximately the opposite translation.
        assert abs(motion_df.loc[motion_df.index[1], "trans_x"]) < shift_x + 1
        assert abs(motion_df.loc[motion_df.index[1], "trans_y"]) < shift_y + 1

    def test_output_has_motion_metadata_attributes(
        self, sample_voxeldata_2dt_registration, ambient_client
    ):
        """Output has motion metadata attributes."""
        result = register_volumewise(
            sample_voxeldata_2dt_registration, reference_time=2
        )

        assert "registration" not in result.attrs
        assert result.attrs["reference_time"] == 2
        assert "motion_params" in result.attrs

    def test_preserves_input_attributes(
        self, sample_voxeldata_2dt_registration, ambient_client
    ):
        """Input attributes are preserved in output."""
        sample_voxeldata_2dt_registration.attrs["custom_attr"] = "test_value"

        result = register_volumewise(sample_voxeldata_2dt_registration)

        assert result.attrs["custom_attr"] == "test_value"

    def test_preserves_coordinates(
        self, sample_voxeldata_2dt_registration, ambient_client
    ):
        """Coordinates and VoxelData geometry are preserved in output."""
        result = register_volumewise(sample_voxeldata_2dt_registration)

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

    def test_different_reference_time(
        self, sample_voxeldata_2dt_registration, ambient_client
    ):
        """Can use different reference time indices."""
        result = register_volumewise(
            sample_voxeldata_2dt_registration, reference_time=2
        )

        assert result.attrs["reference_time"] == 2

    def test_transform_option(self, sample_voxeldata_2dt_registration, ambient_client):
        """transform parameter changes registration behavior."""
        # Both should work without error.
        result_no_rot = register_volumewise(
            sample_voxeldata_2dt_registration, transform="translation"
        )
        result_with_rot = register_volumewise(
            sample_voxeldata_2dt_registration, transform="rigid"
        )

        # Motion params should have rotation columns in both cases.
        assert "rot_x" in result_no_rot.attrs["motion_params"].columns
        assert "rot_x" in result_with_rot.attrs["motion_params"].columns

    def test_singleton_dimension_handling(
        self, sample_voxeldata_2d_registration, ambient_client
    ):
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

        result = register_volumewise(data)

        # Should preserve the singleton dimension.
        assert result.dims == data.dims
        assert result.shape == data.shape
        assert result.sizes["k"] == 1
        # Identical frames should produce nearly identical output.
        assert_allclose(result.values, data.values, atol=1e-3)

    def test_output_dimension_order_matches_input(
        self, sample_voxeldata_2d_registration, ambient_client
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

        result = register_volumewise(data)

        assert result.dims == ("k", "j", "i", "time")
        # Identical frames should produce nearly identical output.
        assert_allclose(result.values, data.values, atol=1e-3)

    def test_multi_resolution_does_not_crash(
        self, sample_voxeldata_3dt_registration, ambient_client
    ):
        """Multi-resolution pyramid completes without error."""
        result = register_volumewise(
            sample_voxeldata_3dt_registration,
            transform="translation",
            use_multi_resolution=True,
        )
        assert result.shape == sample_voxeldata_3dt_registration.shape
        # Identical frames should produce nearly identical output.
        assert_allclose(
            result.values, sample_voxeldata_3dt_registration.values, atol=1e-3
        )

    def test_keep_diagnostics_toggles_full_trace(
        self, sample_voxeldata_2dt_registration, ambient_client
    ):
        """`keep_diagnostics` gates only the full diagnostics list.

        The cheap per-frame summaries (`final_metric_value`, `n_iterations`)
        are always attached to `motion_params`; only the
        memory-hungry trace list is opt-in.
        """
        # Default (False): summary columns yes, full diagnostics list no.
        result_off = register_volumewise(sample_voxeldata_2dt_registration)
        assert "registration_diagnostics" not in result_off.attrs
        motion_df_off = result_off.attrs["motion_params"]
        assert "final_metric_value" in motion_df_off.columns
        assert "n_iterations" in motion_df_off.columns

        # Opt-in: full diagnostics list is also attached.
        result_on = register_volumewise(
            sample_voxeldata_2dt_registration, keep_diagnostics=True
        )
        diagnostics = result_on.attrs["registration_diagnostics"]
        assert len(diagnostics) == sample_voxeldata_2dt_registration.sizes["time"]
        assert all(isinstance(d, RegistrationDiagnostics) for d in diagnostics)


class TestRegisterVolumewiseClientManagement:
    """Tests that specifically require no ambient distributed.Client."""

    def test_creates_local_client_when_none_active(
        self, sample_voxeldata_2dt_registration
    ):
        """Without any active distributed.Client, one is created and torn down."""
        from distributed import get_client

        with pytest.raises(ValueError):
            get_client()

        result = register_volumewise(
            sample_voxeldata_2dt_registration, transform="translation"
        )
        assert result.shape == sample_voxeldata_2dt_registration.shape

        # The auto-created Client is closed again once register_volumewise returns.
        with pytest.raises(ValueError):
            get_client()

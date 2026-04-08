"""Unit tests for slice timing correction."""

import numpy as np
import pytest
import xarray as xr
from scipy.interpolate import interp1d

from confusius.multipose import correct_slice_timings


def _make_consolidated_da(
    ntime: int = 20,
    nz: int = 5,
    ny: int = 2,
    nx: int = 3,
    tr: float = 0.2,
    slice_offsets: np.ndarray | None = None,
    volume_acquisition_reference: str = "start",
    seed: int = 42,
) -> xr.DataArray:
    """Create a synthetic consolidated DataArray for testing."""
    rng = np.random.default_rng(seed)
    data = rng.random((ntime, nz, ny, nx), dtype=np.float64)

    time_vals = np.arange(ntime) * tr
    time_coord = xr.DataArray(
        time_vals,
        dims=["time"],
        attrs={
            "units": "s",
            "volume_acquisition_reference": volume_acquisition_reference,
        },
    )

    z_vals = np.arange(nz) * 0.2
    z_coord = xr.DataArray(z_vals, dims=["z"], attrs={"units": "mm"})

    y_vals = np.arange(ny) * 0.3
    y_coord = xr.DataArray(y_vals, dims=["y"], attrs={"units": "mm"})

    x_vals = np.arange(nx) * 0.4
    x_coord = xr.DataArray(x_vals, dims=["x"], attrs={"units": "mm"})

    if slice_offsets is None:
        slice_offsets = np.linspace(0, tr * 0.8, nz)

    slice_time_vals = np.zeros((ntime, nz))
    for t in range(ntime):
        slice_time_vals[t, :] = time_vals[t] + slice_offsets

    slice_time_coord = xr.DataArray(
        slice_time_vals,
        dims=["time", "z"],
        attrs={"units": "s", "volume_acquisition_reference": "start"},
    )

    return xr.DataArray(
        data,
        dims=("time", "z", "y", "x"),
        coords={
            "time": time_coord,
            "z": z_coord,
            "y": y_coord,
            "x": x_coord,
            "slice_time": slice_time_coord,
        },
        attrs={"affines": {"physical_to_lab": np.eye(4)}},
        name="scan_data",
    )


def _naive_correct_slice_timings(
    da: xr.DataArray, timing_coord_name: str
) -> np.ndarray:
    """Reference: naive per-voxel interp1d loop, equivalent to the pre-apply_ufunc impl."""
    target_times = da.coords["time"].values
    timing_values = da.coords[timing_coord_name].values
    sweep_dim = da.coords[timing_coord_name].dims[1]
    sweep_dim_idx = da.dims.index(sweep_dim)
    other_dims = [d for d in da.dims if d not in ("time", sweep_dim)]
    result = np.empty_like(da.values)
    for s in range(timing_values.shape[1]):
        acq_times = timing_values[:, s]
        for idx in np.ndindex(tuple(da.sizes[d] for d in other_dims)):
            sel: list[int | slice] = [slice(None)] * len(da.dims)
            sel[sweep_dim_idx] = s
            for i, d in enumerate(other_dims):
                sel[da.dims.index(d)] = idx[i]
            sel_t = tuple(sel)
            result[sel_t] = interp1d(
                acq_times,
                da.values[sel_t],
                bounds_error=False,
                fill_value="extrapolate",
            )(target_times)
    return result


class TestCorrectSliceTiming:
    """Tests for correct_slice_timings."""

    # ------------------------------------------------------------------
    # Error paths
    # ------------------------------------------------------------------

    def test_raises_missing_time_dim(self) -> None:
        """Raises ValueError if DataArray has no time dimension."""
        da = xr.DataArray(
            np.random.random((5, 3, 2)),
            dims=("z", "y", "x"),
            coords={"z": np.arange(5), "y": np.arange(3), "x": np.arange(2)},
        )
        with pytest.raises(ValueError, match="'time' dimension"):
            correct_slice_timings(da)

    def test_raises_chunked_time(self, consolidated_scan_4d: xr.DataArray) -> None:
        """Raises ValueError if the time dimension is chunked."""
        pytest.importorskip("dask.array")
        da_chunked = consolidated_scan_4d.chunk({"time": 1})
        with pytest.raises(ValueError, match="chunked along the 'time' dimension"):
            correct_slice_timings(da_chunked)

    def test_raises_missing_timing_coord(self) -> None:
        """Raises ValueError if DataArray has no slice_time or pose_time coordinate."""
        da = xr.DataArray(
            np.random.random((5, 3, 2, 2)),
            dims=("time", "z", "y", "x"),
            coords={
                "time": np.arange(5),
                "z": np.arange(3),
                "y": np.arange(2),
                "x": np.arange(2),
            },
        )
        with pytest.raises(
            ValueError, match="neither 'slice_time' nor 'pose_time' coordinate"
        ):
            correct_slice_timings(da)

    def test_raises_invalid_slice_time_dims(self) -> None:
        """Raises ValueError if slice_time has wrong dimensions."""
        da = xr.DataArray(
            np.random.random((5, 3, 2, 2)),
            dims=("time", "z", "y", "x"),
            coords={
                "time": [0, 1, 2, 3, 4],
                "z": [0, 1, 2],
                "y": [0, 1],
                "x": [0, 1],
                "slice_time": xr.DataArray(
                    np.random.random((3, 5)),
                    dims=("z", "time"),
                ),
            },
        )
        with pytest.raises(ValueError, match="dims \\('time', <sweep_dim>\\)"):
            correct_slice_timings(da)

    # ------------------------------------------------------------------
    # Correctness: analytical references
    # ------------------------------------------------------------------

    def test_zero_shift_returns_input(self) -> None:
        """When all slices are acquired at the volume reference time, output equals input."""
        ntime, nz, tr = 20, 4, 0.2
        da = _make_consolidated_da(
            ntime=ntime, nz=nz, tr=tr, slice_offsets=np.zeros(nz)
        )
        result = correct_slice_timings(da)
        np.testing.assert_allclose(result.values, da.values, atol=1e-12)

    def test_sinusoid_correction_accuracy(self) -> None:
        """Correction resamples shifted sinusoids to the volume reference time.

        Each z-slice contains the same sinusoid sampled at a known offset from the
        volume onset. After correction all slices should match the signal evaluated at
        the volume onset (the `time` coordinate), within linear interpolation error.
        """
        rng = np.random.default_rng(0)
        ntime = 100
        nz = 5
        tr = 0.2  # s
        freq = 0.5  # Hz, well below Nyquist (2.5 Hz)

        slice_offsets = np.array([0.0, tr / 4, tr / 2, 3 * tr / 4, 0.0])
        time_vals = np.arange(ntime) * tr
        ref_signal = np.sin(2 * np.pi * freq * time_vals)

        data = np.zeros((ntime, nz, 1, 1))
        slice_time_vals = np.zeros((ntime, nz))
        for s in range(nz):
            acq_times = time_vals + slice_offsets[s]
            data[:, s, 0, 0] = np.sin(2 * np.pi * freq * acq_times)
            slice_time_vals[:, s] = acq_times

        da = xr.DataArray(
            data,
            dims=("time", "z", "y", "x"),
            coords={
                "time": xr.DataArray(
                    time_vals,
                    dims=["time"],
                    attrs={"units": "s", "volume_acquisition_reference": "start"},
                ),
                "z": xr.DataArray(np.arange(nz) * 0.2, dims=["z"]),
                "y": xr.DataArray([0.0], dims=["y"]),
                "x": xr.DataArray([0.0], dims=["x"]),
                "slice_time": xr.DataArray(
                    slice_time_vals, dims=["time", "z"], attrs={"units": "s"}
                ),
            },
            attrs={"affines": {"physical_to_lab": np.eye(4)}},
        )

        result = correct_slice_timings(da, method="linear")

        # Interior time points avoid boundary extrapolation artefacts.
        interior = slice(5, ntime - 5)
        for s in range(nz):
            np.testing.assert_allclose(
                result.values[interior, s, 0, 0],
                ref_signal[interior],
                atol=0.05,
                err_msg=f"Slice {s} not corrected within tolerance",
            )

        # Random noise should not be corrected to match the sinusoid.
        noise_da = da.copy(data=rng.standard_normal((ntime, nz, 1, 1)))
        noise_result = correct_slice_timings(noise_da)
        assert not np.allclose(noise_result.values[:, 1, 0, 0], ref_signal, atol=0.05)

    # ------------------------------------------------------------------
    # Correctness: reference implementation, both timing coord paths
    # ------------------------------------------------------------------

    def test_slice_time_path_matches_reference(
        self, consolidated_scan_4d: xr.DataArray
    ) -> None:
        """slice_time path matches naive per-voxel interp1d reference.

        Also verifies that slice_time is dropped and all other coords are preserved.
        """
        da = consolidated_scan_4d
        result = correct_slice_timings(da)

        np.testing.assert_allclose(
            result.values, _naive_correct_slice_timings(da, "slice_time"), atol=1e-12
        )
        assert "slice_time" not in result.coords
        assert result.dims == da.dims
        for coord in da.coords:
            if coord != "slice_time":
                np.testing.assert_array_equal(
                    result.coords[coord].values, da.coords[coord].values
                )

    def test_pose_time_path_matches_reference(self, scan_4d: xr.DataArray) -> None:
        """pose_time path matches naive per-voxel interp1d reference.

        Also verifies that pose_time is dropped and all other coords are preserved.
        """
        da = scan_4d
        result = correct_slice_timings(da)

        np.testing.assert_allclose(
            result.values, _naive_correct_slice_timings(da, "pose_time"), atol=1e-12
        )
        assert "pose_time" not in result.coords
        assert result.dims == da.dims
        for coord in da.coords:
            if coord != "pose_time":
                np.testing.assert_array_equal(
                    result.coords[coord].values, da.coords[coord].values
                )

    @pytest.mark.parametrize("reference", ["start", "center", "end"])
    def test_pose_time_and_slice_time_equivalent(self, reference: str) -> None:
        """pose_time and slice_time paths give identical corrections for any reference.

        Both paths use the same actual per-pose acquisition times (pose_time_vals) and
        the same `time` coordinate, so the interpolation target and source are identical
        regardless of which reference point the `time` coordinate uses.
        """
        ntime, npose = 20, 4
        tr = 0.2
        rng = np.random.default_rng(7)
        data = rng.random((ntime, npose, 2, 3))

        # Volume onset times (always start-referenced internally).
        onset_vals = np.arange(ntime) * tr
        # Shift the `time` coordinate to the chosen reference point.
        ref_offsets = {"start": 0.0, "center": 0.5, "end": 1.0}
        time_vals = onset_vals + ref_offsets[reference] * tr
        # Acquisition time for pose p at volume t is the onset plus p * (TR / npose).
        pose_time_vals = onset_vals[:, None] + np.arange(npose) * (tr / npose)

        time_coord = xr.DataArray(
            time_vals,
            dims=["time"],
            attrs={"units": "s", "volume_acquisition_reference": reference},
        )
        timing_attrs = {"units": "s"}

        da_unconsolidated = xr.DataArray(
            data[:, :, None],
            dims=("time", "pose", "z", "y", "x"),
            coords={
                "time": time_coord,
                "pose": np.arange(npose),
                "z": np.array([0.0]),
                "y": np.arange(2) * 0.3,
                "x": np.arange(3) * 0.4,
                "pose_time": xr.DataArray(
                    pose_time_vals, dims=["time", "pose"], attrs=timing_attrs
                ),
            },
        )

        da_consolidated = xr.DataArray(
            data,
            dims=("time", "z", "y", "x"),
            coords={
                "time": time_coord,
                "z": np.arange(npose) * 0.5,
                "y": np.arange(2) * 0.3,
                "x": np.arange(3) * 0.4,
                "slice_time": xr.DataArray(
                    pose_time_vals, dims=["time", "z"], attrs=timing_attrs
                ),
            },
        )

        result_unconsolidated = correct_slice_timings(da_unconsolidated)
        result_consolidated = correct_slice_timings(da_consolidated)

        np.testing.assert_allclose(
            result_unconsolidated.squeeze("z").values,
            result_consolidated.values,
            atol=1e-12,
        )

    # ------------------------------------------------------------------
    # Laziness
    # ------------------------------------------------------------------

    def test_lazy_with_dask_input(self, consolidated_scan_4d: xr.DataArray) -> None:
        """Output is dask-backed and numerically identical to the eager result."""
        pytest.importorskip("dask.array")
        da = consolidated_scan_4d
        da_dask = da.chunk({"z": 1})
        eager_result = correct_slice_timings(da)
        lazy_result = correct_slice_timings(da_dask)
        assert hasattr(lazy_result.data, "dask"), "Output should be dask-backed."
        np.testing.assert_allclose(lazy_result.values, eager_result.values, atol=1e-12)

    # ------------------------------------------------------------------
    # Method fallback
    # ------------------------------------------------------------------

    def test_cubic_method_fallback(self, consolidated_scan_4d: xr.DataArray) -> None:
        """Cubic falls back to linear when there are too few points."""
        # Load to eager so the warning fires immediately rather than at compute time.
        da = consolidated_scan_4d.isel(time=slice(0, 3)).load()
        with pytest.warns(UserWarning, match="falling back to 'linear'"):
            result = correct_slice_timings(da, method="cubic")
        expected = correct_slice_timings(da, method="linear")
        np.testing.assert_allclose(result.values, expected.values, atol=1e-12)

    def test_reraises_unrecognised_interp_error(
        self, consolidated_scan_4d: xr.DataArray
    ) -> None:
        """Non-boundary ValueError from interp1d is re-raised unchanged."""
        # Load to eager so the error fires immediately rather than at compute time.
        # fill_value=(1.0, 2.0, 3.0) makes interp1d raise a ValueError whose message
        # does not contain "derivatives at boundaries", so _interpolate_timeseries
        # should re-raise rather than fall back to linear.
        with pytest.raises(ValueError, match="broadcast"):
            correct_slice_timings(
                consolidated_scan_4d.load(),
                fill_value=(1.0, 2.0, 3.0),  # type: ignore[arg-type]
            )

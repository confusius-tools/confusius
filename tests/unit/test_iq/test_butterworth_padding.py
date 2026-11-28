"""Shared Butterworth design and padding behavior through public filtering APIs."""

from functools import partial
from typing import Any

import dask.array as da
import numpy as np
import pytest
import scipy.signal as sp_signal
from dask.callbacks import Callback
from numpy.testing import assert_allclose

from confusius.iq import (
    clutter_filter_butterworth,
    clutter_filter_sosfiltfilt,
    compute_axial_velocity_volume,
    compute_power_doppler_volume,
    process_iq_to_axial_velocity,
    process_iq_to_power_doppler,
)
from confusius.signal import filter_butterworth


@pytest.mark.parametrize(
    "padtype,padlen", [("odd", 2), ("even", 2), ("constant", 0), (None, 100)]
)
def test_custom_padding_matches_scipy(sample_voxeldata_iq_dataarray, padtype, padlen):
    """Signal and IQ filters agree with SciPy, including disabled padding."""
    iq = sample_voxeldata_iq_dataarray.isel(time=slice(0, 8))
    sos = sp_signal.butter(4, 1.0, btype="highpass", fs=10.0, output="sos")
    expected = sp_signal.sosfiltfilt(
        sos, iq.values, axis=0, padtype=padtype, padlen=padlen
    )
    assert_allclose(
        clutter_filter_butterworth(
            iq.values, fs=10.0, low_cutoff=1.0, padtype=padtype, padlen=padlen
        ),
        expected,
    )
    assert_allclose(
        clutter_filter_sosfiltfilt(iq.values, sos, padtype=padtype, padlen=padlen),
        expected,
    )
    result = filter_butterworth(
        iq.chunk({"time": -1, "i": 2}),
        low_cutoff=1.0,
        order=4,
        padtype=padtype,
        padlen=padlen,
    )
    assert_allclose(result.compute().values, expected)


@pytest.mark.parametrize("order", [3, 4])
@pytest.mark.parametrize("low,high", [(1.0, None), (None, 3.0), (1.0, 3.0)])
def test_signal_padding_boundary_matches_scipy(
    sample_voxeldata_iq_dataarray, order, low, high
):
    """Signal filtering accepts exactly SciPy's default minimum sample count."""
    iq = sample_voxeldata_iq_dataarray
    signals = iq.isel(time=np.arange(40) % iq.sizes["time"]).assign_coords(
        time=("time", np.arange(40) / 10.0, iq.time.attrs)
    )
    bandpass = low is not None and high is not None
    padlen = 3 * (2 * order + 1) if bandpass else 3 * (order + 1)
    sos = sp_signal.butter(
        order,
        [low, high] if bandpass else (low if low is not None else high),
        btype="bandpass"
        if bandpass
        else ("highpass" if low is not None else "lowpass"),
        fs=10.0,
        output="sos",
    )
    with pytest.raises(ValueError, match="padlen"):
        sp_signal.sosfiltfilt(sos, signals.values[:padlen], axis=0)
    with pytest.raises(ValueError, match="padlen"):
        filter_butterworth(
            signals.isel(time=slice(0, padlen)),
            low_cutoff=low,
            high_cutoff=high,
            order=order,
        )
    signals = signals.isel(time=slice(0, padlen + 1))
    expected = sp_signal.sosfiltfilt(sos, signals.values, axis=0)
    actual = filter_butterworth(signals, low_cutoff=low, high_cutoff=high, order=order)
    assert_allclose(actual.values, expected)


@pytest.mark.parametrize("velocity", [False, True])
@pytest.mark.parametrize("accessor", [False, True])
@pytest.mark.parametrize("padtype,padlen", [("even", 2), ("constant", 0), (None, 100)])
def test_iq_wrappers_forward_custom_padding(
    sample_voxeldata_iq_dataarray, velocity, accessor, padtype, padlen
):
    """Lazy wrappers and accessors forward padding through both IQ reduction kernels."""
    iq = sample_voxeldata_iq_dataarray.isel(time=slice(0, 16)).chunk({"time": 8})
    kwargs: dict[str, Any] = {
        "filter_method": "butterworth",
        "low_cutoff": 1.0,
        "butterworth_padtype": padtype,
        "butterworth_padlen": padlen,
    }
    if velocity:
        process = (
            iq.fusi.iq.process_to_axial_velocity
            if accessor
            else partial(process_iq_to_axial_velocity, iq)
        )
        kernel = compute_axial_velocity_volume
    else:
        process = (
            iq.fusi.iq.process_to_power_doppler
            if accessor
            else partial(process_iq_to_power_doppler, iq)
        )
        kernel = compute_power_doppler_volume
    expected = np.concatenate(
        [kernel(iq.values[start : start + 8], fs=10.0, **kwargs) for start in (0, 8)]
    )
    with Callback(pretask=lambda *args: pytest.fail("IQ read during validation")):
        result = process(clutter_window_width=8, **kwargs)
    assert isinstance(result.data, da.Array)
    assert_allclose(result.compute().values, expected)


@pytest.mark.parametrize("padtype,padlen", [("invalid", None), ("odd", 8)])
@pytest.mark.parametrize(
    "process",
    [filter_butterworth, process_iq_to_power_doppler, process_iq_to_axial_velocity],
)
def test_invalid_padding_fails_before_execution(
    sample_voxeldata_iq_dataarray, process, padtype, padlen
):
    """Bad padding is rejected before running lazy IQ or signal tasks."""
    iq = sample_voxeldata_iq_dataarray.isel(time=slice(0, 8)).chunk({"time": -1})
    kwargs: dict[str, Any] = {"padtype": padtype, "padlen": padlen}
    if process is not filter_butterworth:
        kwargs = {f"butterworth_{key}": value for key, value in kwargs.items()}
        kwargs["filter_method"] = "butterworth"
    with (
        Callback(pretask=lambda *args: pytest.fail("IQ read for invalid padding")),
        pytest.raises(ValueError, match="padtype|padlen"),
    ):
        process(iq, low_cutoff=1.0, **kwargs)

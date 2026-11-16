"""Regression tests for lazy, fixed-output IQ processing contracts."""

from collections.abc import Callable
from contextlib import nullcontext
from functools import partial
from math import ceil
from typing import Any

import dask.array as da
import numpy as np
import pytest
from dask import delayed
from dask.callbacks import Callback
from numpy.testing import assert_allclose

from confusius.iq import (
    clutter_filter_svd_from_cumulative_energy,
    clutter_filter_svd_from_energy,
    compute_axial_velocity_volume,
    compute_bmode_volume,
    compute_power_doppler_volume,
    compute_svd_cumulative_energy_threshold,
    process_iq_blocks,
    process_iq_to_axial_velocity,
    process_iq_to_bmode,
    process_iq_to_power_doppler,
)


@pytest.mark.parametrize("width,stride", [(20, 20), (10, 10), (10, 5), (10, 1), (8, 4)])
@pytest.mark.parametrize("windows_per_task", [None, 1, 2, 4, 8, 32])
def test_block_graph_is_lazy_and_executes_each_window_once(
    sample_voxeldata_iq_dataarray, width, stride, windows_per_task
):
    """Source reads and callbacks wait until compute, including one/two batches."""
    values = sample_voxeldata_iq_dataarray.values.astype(np.complex64)
    loads, calls = [], []

    @delayed
    def load():
        loads.append(True)
        return values

    def process(block):
        calls.append(True)
        midpoint = len(block) // 2
        return np.stack(
            [
                np.abs(block[:midpoint]).mean(axis=0),
                np.abs(block[midpoint:]).mean(axis=0),
            ]
        )

    iq = da.from_delayed(load(), shape=values.shape, dtype=values.dtype).rechunk(
        (5, 2, 3, 4)
    )
    with Callback(
        pretask=lambda *args: pytest.fail("IQ graph executed during construction")
    ):
        result = process_iq_blocks(
            iq,
            process,
            window_width=width,
            window_stride=stride,
            output_shape=(2,) + values.shape[1:],
            output_dtype=np.float32,
            windows_per_task=windows_per_task,
        )
    assert not loads and not calls
    actual = result.compute(scheduler="synchronous")
    starts = range(0, len(values) - width + 1, stride)
    expected = np.concatenate(
        [
            np.stack(
                [
                    np.abs(values[start : start + width // 2]).mean(axis=0),
                    np.abs(values[start + width // 2 : start + width]).mean(axis=0),
                ]
            )
            for start in starts
        ]
    )
    assert len(calls) == len(starts)
    batch_size = max(1, ceil(width / stride) - 1, windows_per_task or 1)
    assert result.numblocks[0] == ceil(len(starts) / batch_size)
    assert result.chunks[0][0] == 2 * (len(starts) % batch_size or batch_size)
    assert len(loads) == 1
    assert actual.dtype == result.dtype == np.float32
    assert_allclose(actual, expected, rtol=1e-6)


@pytest.mark.parametrize("helper", ["bmode", "power", "velocity", "threshold"])
@pytest.mark.parametrize("stride", [10, 5, 1])
def test_public_helpers_do_not_execute_iq_during_construction(
    sample_voxeldata_iq_dataarray, helper, stride
):
    """All built-in helpers, including threshold calibration, are genuinely lazy."""
    values = sample_voxeldata_iq_dataarray.values.astype(np.complex64)
    loads = []

    @delayed
    def load():
        loads.append(True)
        return values

    iq = sample_voxeldata_iq_dataarray.copy(
        data=da.from_delayed(load(), shape=values.shape, dtype=values.dtype)
    )
    width = 10
    kwargs: dict[str, Any] = {}
    wrapper: Callable[..., Any]
    kernel: Callable[..., Any] | None
    if helper == "bmode":
        wrapper, kernel = process_iq_to_bmode, compute_bmode_volume
    elif helper == "power":
        wrapper, kernel = process_iq_to_power_doppler, compute_power_doppler_volume
        kwargs = {"low_cutoff": 1}
    elif helper == "velocity":
        wrapper, kernel = process_iq_to_axial_velocity, compute_axial_velocity_volume
        kwargs = {"low_cutoff": 1}
    else:
        wrapper, kernel = compute_svd_cumulative_energy_threshold, None
        kwargs = {"singular_value_index": 1}
    window_kwargs: dict[str, Any]
    if helper == "threshold":
        window_kwargs = {"window_width": width, "window_stride": stride, **kwargs}
    elif helper == "bmode":
        window_kwargs = {"bmode_window_width": width, "bmode_window_stride": stride}
    else:
        window_kwargs = {
            "clutter_window_width": width,
            "clutter_window_stride": stride,
            **kwargs,
        }
    with Callback(
        pretask=lambda *args: pytest.fail("IQ graph executed during construction")
    ):
        result = wrapper(iq, **window_kwargs)
    assert not loads
    actual = result.compute(scheduler="synchronous")
    windows = [
        values[start : start + width]
        for start in range(0, len(values) - width + 1, stride)
    ]
    if helper == "threshold":
        expected = min(
            np.linalg.eigvalsh(
                window.reshape(width, -1).astype(np.complex128)
                @ window.reshape(width, -1).astype(np.complex128).conj().T
            )[:-1].sum()
            for window in windows
        )
    else:
        assert kernel is not None
        kernel_kwargs = {**kwargs, "fs": 10.0} if helper == "velocity" else kwargs
        expected = np.concatenate(
            [kernel(window, **kernel_kwargs) for window in windows]
        )
        actual = actual.values
    assert loads
    assert_allclose(actual, expected, rtol=1e-6, atol=1e-10)


@pytest.mark.parametrize("stride", [8, 4, 1])
@pytest.mark.parametrize("windows_per_task", [None, 8])
def test_added_map_axis_preserves_window_mapping(
    sample_voxeldata_iq_dataarray, stride, windows_per_task
):
    """An added component axis is independent of the temporal batch axis."""
    values = sample_voxeldata_iq_dataarray.values.astype(np.complex64)

    def process(block):
        return np.stack([block.real.mean(axis=0), block.imag.mean(axis=0)])[None]

    with pytest.warns(UserWarning) if stride == 8 else nullcontext():
        result = process_iq_blocks(
            da.from_array(values, chunks=(5, 2, 3, 4)),
            process,
            window_width=8,
            window_stride=stride,
            output_shape=(1, 2) + values.shape[1:],
            output_dtype=np.float32,
            windows_per_task=windows_per_task,
        )
    expected = np.concatenate(
        [
            process(values[start : start + 8])
            for start in range(0, len(values) - 8 + 1, stride)
        ]
    )
    assert_allclose(result.compute(), expected, rtol=1e-6)


@pytest.mark.parametrize("stride", [10, 5, 1])
@pytest.mark.parametrize("windows_per_task", [None, 8])
def test_spatially_reduced_outputs_and_multiple_times(
    sample_voxeldata_iq_dataarray, stride, windows_per_task
):
    """Spatially reduced outputs concatenate on output time, not input time."""
    values = sample_voxeldata_iq_dataarray.values.astype(np.complex64)

    def process(block):
        return np.array([block.real.min(), block.real.max()])

    result = process_iq_blocks(
        da.from_array(values, chunks=(5, 2, 3, 4)),
        process,
        window_width=10,
        window_stride=stride,
        output_shape=(2,),
        output_dtype=np.float32,
        windows_per_task=windows_per_task,
    )
    expected = np.concatenate(
        [
            process(values[start : start + 10])
            for start in range(0, len(values) - 10 + 1, stride)
        ]
    )
    assert_allclose(result.compute(), expected)


@pytest.mark.parametrize(
    "width,stride", [(0, 1), (1, 0), (-1, 1), (21, 1), (2, 3), (2.5, 1), (True, 1)]
)
def test_invalid_windows_fail_without_running_callback(
    sample_voxeldata_iq_dataarray, width, stride
):
    """Window validation is metadata-only and reports a public ValueError."""
    with pytest.raises(ValueError, match="window"):
        process_iq_blocks(
            da.from_array(sample_voxeldata_iq_dataarray.values),
            lambda block: pytest.fail("callback ran"),
            window_width=width,
            window_stride=stride,
            output_shape=(1, 4, 6, 8),
            output_dtype=np.float64,
        )


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True])
def test_invalid_batch_sizes_raise(sample_voxeldata_iq_dataarray, batch_size):
    """Execution batching accepts only positive integral window counts."""
    with pytest.raises(ValueError, match="windows_per_task"):
        process_iq_blocks(
            da.from_array(sample_voxeldata_iq_dataarray.values),
            compute_bmode_volume,
            output_shape=(1, 4, 6, 8),
            output_dtype=np.float64,
            windows_per_task=batch_size,
        )


@pytest.mark.parametrize(
    "shape", [(), (0, 4, 6, 8), (1, -1, 6, 8), (1.0, 4, 6, 8), (True, 4, 6, 8)]
)
def test_invalid_output_shape_raises(sample_voxeldata_iq_dataarray, shape):
    """An output declaration must contain concrete positive integer sizes."""
    with pytest.raises(ValueError, match="output_shape"):
        process_iq_blocks(
            da.from_array(sample_voxeldata_iq_dataarray.values),
            compute_bmode_volume,
            output_shape=shape,
            output_dtype=np.float64,
        )


@pytest.mark.parametrize(
    "shape",
    [
        (2,),
        (2, 5, 7),
        (2, 4, 6, 8),
        (1, 2, 4, 6, 8),
        (1, 4, 2, 6, 8),
        (2, 3, 4, 5, 6, 8),
    ],
)
@pytest.mark.parametrize("stride", [10, 5, 1])
@pytest.mark.parametrize("windows_per_task", [None, 8])
def test_output_shape_alone_defines_axis_mapping(
    sample_voxeldata_iq_dataarray, shape, stride, windows_per_task
):
    """Arbitrary output ranks and sizes retain every time/map value in order."""
    values = sample_voxeldata_iq_dataarray.values.astype(np.complex64)

    def process(block):
        return (
            np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
            + block.real.mean()
        )

    result = process_iq_blocks(
        da.from_array(values, chunks=(5, 2, 3, 4)),
        process,
        output_shape=shape,
        output_dtype=np.float32,
        window_width=10,
        window_stride=stride,
        windows_per_task=windows_per_task,
    )
    expected = np.concatenate(
        [
            process(values[start : start + 10])
            for start in range(0, len(values) - 10 + 1, stride)
        ]
    )
    assert_allclose(result.compute(), expected, rtol=1e-6)


@pytest.mark.parametrize("stride", [10, 5, 1])
@pytest.mark.parametrize("windows_per_task", [None, 8])
def test_partial_keeps_callback_settings_separate(
    sample_voxeldata_iq_dataarray, stride, windows_per_task
):
    """A partial can bind callback settings named like orchestration settings."""
    values = sample_voxeldata_iq_dataarray.values.astype(np.complex64)

    def process(block, *, window_width, output_dtype):
        return (
            block[:window_width].real.mean(axis=0, keepdims=True).astype(output_dtype)
        )

    processor = partial(process, window_width=2, output_dtype=np.float32)
    result = process_iq_blocks(
        da.from_array(values, chunks=(5, 2, 3, 4)),
        processor,
        output_shape=(1,) + values.shape[1:],
        output_dtype=np.float32,
        window_width=10,
        window_stride=stride,
        windows_per_task=windows_per_task,
    )
    expected = np.stack(
        [
            values[start : start + 2].real.mean(axis=0)
            for start in range(0, len(values) - 9, stride)
        ]
    )
    assert_allclose(result.compute(), expected, rtol=1e-6)


@pytest.mark.parametrize("stride", [10, 1])
@pytest.mark.parametrize("violation", ["shape", "dtype", "array"])
def test_callback_contract_is_checked_at_execution(
    sample_voxeldata_iq_dataarray, stride, violation
):
    """Declarations never silently reshape or cast an invalid callback output."""

    def process(block):
        if violation == "shape":
            return np.zeros((2, 4, 6, 8))
        if violation == "dtype":
            return np.zeros((1, 4, 6, 8), dtype=np.float32)
        return [0]

    result = process_iq_blocks(
        da.from_array(sample_voxeldata_iq_dataarray.values),
        process,
        window_width=10,
        window_stride=stride,
        output_shape=(1, 4, 6, 8),
        output_dtype=np.float64,
    )
    error = ValueError if violation == "shape" else TypeError
    with pytest.raises(error, match="shape|dtype|NumPy"):
        result.compute()


@pytest.mark.parametrize(
    "process", [process_iq_to_power_doppler, process_iq_to_axial_velocity]
)
def test_unknown_clutter_filter_raises_before_iq_execution(
    sample_voxeldata_iq_dataarray, process
):
    """Unknown filter names are rejected without executing IQ tasks."""
    iq = sample_voxeldata_iq_dataarray.chunk({"time": 10})
    with (
        Callback(
            pretask=lambda *args: pytest.fail("IQ executed for an invalid filter")
        ),
        pytest.raises(ValueError, match="Unknown clutter filter method: unsupported"),
    ):
        process(iq, filter_method="unsupported")


@pytest.mark.parametrize(
    "process", [process_iq_to_power_doppler, process_iq_to_axial_velocity]
)
@pytest.mark.parametrize("high_cutoff", [1.0, 2.0])
def test_butterworth_cutoff_order_raises_before_iq_execution(
    sample_voxeldata_iq_dataarray, process, high_cutoff
):
    """Reversed and equal band cutoffs fail before any IQ processing."""
    iq = sample_voxeldata_iq_dataarray.chunk({"time": 10})
    with (
        Callback(pretask=lambda *args: pytest.fail("IQ executed for invalid cutoffs")),
        pytest.raises(ValueError, match="must be greater than low cutoff frequency"),
    ):
        process(
            iq, filter_method="butterworth", low_cutoff=2.0, high_cutoff=high_cutoff
        )


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize(
    "method", ["svd_indices", "svd_energy", "svd_cumulative_energy", "butterworth"]
)
@pytest.mark.parametrize("active", [False, True])
def test_wrapper_dtype_matches_kernel(
    sample_voxeldata_iq_dataarray, dtype, method, active
):
    """No-op filters preserve precision, while potentially active filters promote."""
    iq = sample_voxeldata_iq_dataarray.astype(dtype)
    kwargs: dict[str, Any] = {
        "filter_method": method,
        "low_cutoff": 1 if active else None,
    }
    expected_dtype = np.float64 if dtype == np.complex128 or active else np.float32
    for wrapper, kernel in [
        (process_iq_to_power_doppler, compute_power_doppler_volume),
        (process_iq_to_axial_velocity, compute_axial_velocity_volume),
    ]:
        result = wrapper(iq, clutter_window_width=20, **kwargs)
        actual = result.compute()
        expected = kernel(iq.values, fs=10.0, **kwargs)
        assert result.dtype == actual.dtype == expected.dtype == expected_dtype
        assert_allclose(actual.values, expected, rtol=1e-6, atol=1e-10)


@pytest.mark.parametrize(
    "method,kernel",
    [
        ("svd_energy", clutter_filter_svd_from_energy),
        ("svd_cumulative_energy", clutter_filter_svd_from_cumulative_energy),
    ],
)
def test_adaptive_filter_precision_does_not_depend_on_selected_vectors(
    sample_voxeldata_iq_dataarray, method, kernel
):
    """Zero and nonzero selected clutter vectors produce the same declared precision."""
    values = sample_voxeldata_iq_dataarray.values.astype(np.complex64)
    values[:10] *= 0.01
    iq = sample_voxeldata_iq_dataarray.copy(
        data=da.from_array(values, chunks=(10, 4, 6, 8))
    )
    first = kernel(values[:10], high_cutoff=100)
    assert first.dtype == np.complex128
    assert_allclose(first, values[:10], rtol=0, atol=0)
    expected = np.concatenate(
        [
            compute_power_doppler_volume(
                values[start : start + 10], filter_method=method, high_cutoff=100
            )
            for start in (0, 10)
        ]
    )
    result = process_iq_to_power_doppler(
        iq, clutter_window_width=10, filter_method=method, high_cutoff=100
    )
    actual = result.compute()
    assert result.dtype == actual.dtype == expected.dtype == np.float64
    assert_allclose(actual.values, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_bmode_preserves_real_precision(sample_voxeldata_iq_dataarray, dtype):
    """B-mode declarations and computed values retain the input's real precision."""
    iq = sample_voxeldata_iq_dataarray.astype(dtype)
    result = process_iq_to_bmode(iq, bmode_window_width=10, bmode_window_stride=5)
    expected = np.stack(
        [np.abs(iq.values[start : start + 10]).mean(axis=0) for start in (0, 5, 10)]
    )
    actual = result.compute()
    assert result.dtype == actual.dtype == expected.dtype
    assert_allclose(actual.values, expected, rtol=1e-6)


def test_masked_full_index_range_is_a_noop(sample_voxeldata_iq_dataarray):
    """No-op precision is determined from the masked component count."""
    iq = sample_voxeldata_iq_dataarray.astype(np.complex64)
    mask = iq.isel(time=0, drop=True).copy(data=np.zeros(iq.shape[1:], dtype=bool))
    mask.data.ravel()[:2] = True
    result = process_iq_to_power_doppler(
        iq, clutter_window_width=10, clutter_mask=mask, low_cutoff=0, high_cutoff=2
    )
    expected = np.stack(
        [
            np.mean(np.abs(iq.values[start : start + 10]) ** 2, axis=0)
            for start in (0, 10)
        ]
    )
    actual = result.compute()
    assert result.dtype == actual.dtype == np.float32
    assert_allclose(actual.values, expected, rtol=1e-6)


def test_unknown_input_sizes_do_not_trigger_computation():
    """Unknown Dask shapes must be resolved explicitly rather than eagerly sampled."""
    iq = da.arange(20)
    unknown = iq[iq > 0]
    with (
        Callback(
            pretask=lambda *args: pytest.fail("unknown sizes triggered computation")
        ),
        pytest.raises(ValueError, match="known dimension sizes"),
    ):
        process_iq_blocks(
            unknown, lambda block: block, output_shape=(1,), output_dtype=int
        )


@pytest.mark.parametrize("iq", [np.ones((2, 1, 1, 1))])
def test_numpy_input_is_rejected_by_low_level_api(iq):
    """Only wrappers convert eager input to Dask arrays."""
    with pytest.raises(TypeError, match="Dask array"):
        process_iq_blocks(
            iq,
            compute_bmode_volume,
            output_shape=(1, 1, 1, 1),
            output_dtype=float,
        )


def test_numpy_scalar_velocity_parameters_match_declared_dtype(
    sample_voxeldata_iq_dataarray,
):
    """NumPy scalar promotion agrees with the velocity graph's dtype declaration."""
    iq = sample_voxeldata_iq_dataarray.astype(np.complex64)
    iq.attrs["transmit_frequency"] = np.float64(15.625e6)
    iq.attrs["beamforming_sound_velocity"] = np.float64(1540)
    result = process_iq_to_axial_velocity(iq, clutter_window_width=20, lag=1)
    expected = compute_axial_velocity_volume(
        iq.values,
        fs=10,
        lag=1,
        transmit_frequency=iq.attrs["transmit_frequency"],
        beamforming_sound_velocity=iq.attrs["beamforming_sound_velocity"],
    )
    actual = result.compute()
    assert result.dtype == actual.dtype == expected.dtype == np.float64
    assert_allclose(actual.values, expected, rtol=1e-6)


@pytest.mark.parametrize("lag", [0, -1, 10, 1.5, True])
def test_velocity_lag_validation_is_shared(sample_voxeldata_iq_dataarray, lag):
    """Kernels and wrappers reject lags that cannot estimate the requested window."""
    with pytest.raises(ValueError, match="lag"):
        compute_axial_velocity_volume(
            sample_voxeldata_iq_dataarray.values,
            fs=10,
            velocity_window_width=10,
            lag=lag,
        )
    with pytest.raises(ValueError, match="lag"):
        process_iq_to_axial_velocity(
            sample_voxeldata_iq_dataarray, clutter_window_width=10, lag=lag
        )

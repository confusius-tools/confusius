"""Benchmark production IQ helpers across execution batch sizes.

Run `uv run python tools/benchmarks/iq_blocks.py --output PATH`.
The wrappers retain their public signatures: the benchmark injects the low-level
`windows_per_task` option while constructing each helper's real production graph.
"""

import argparse
import gc
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import dask
import dask.array as da
import numpy as np
import numpy.typing as npt
import psutil
import scipy
import scipy.linalg
import scipy.signal
import xarray as xr
from threadpoolctl import threadpool_info, threadpool_limits

import confusius.iq.process as iq_process
from confusius.iq import (
    compute_axial_velocity_volume,
    compute_bmode_volume,
    compute_power_doppler_volume,
    compute_svd_cumulative_energy_threshold,
    process_iq_to_axial_velocity,
    process_iq_to_bmode,
    process_iq_to_power_doppler,
)
from confusius.xarray import create_voxeldata

HELPERS = (
    "bmode",
    "power_svd",
    "power_butterworth",
    "velocity_svd",
    "velocity_butterworth",
    "threshold",
)
"""Public helper and filter configurations included in the timing sweep."""


def make_values(shape: tuple[int, ...]) -> npt.NDArray[np.complex64]:
    """Create deterministic complex64 input without float64-sized temporaries.

    Parameters
    ----------
    shape : tuple[int, ...]
        Input dimensions.

    Returns
    -------
    numpy.ndarray
        Seeded complex IQ samples.
    """
    rng = np.random.default_rng(2026)
    values = np.empty(shape, dtype=np.complex64)
    values.real = rng.standard_normal(shape, dtype=np.float32)
    values.imag = rng.standard_normal(shape, dtype=np.float32)
    return values


def make_iq(values: npt.NDArray, chunk: int) -> xr.DataArray:
    """Attach VoxelData geometry and 100 Hz sampling to chunked IQ samples.

    Parameters
    ----------
    values : numpy.ndarray
        Complex input samples.
    chunk : int
        Stored/source temporal chunk size; spatial chunks span the full volume.

    Returns
    -------
    xarray.DataArray
        Dask-backed IQ VoxelData.
    """
    return create_voxeldata(
        da.from_array(values, chunks=(chunk,) + values.shape[1:]),
        dims=("time", "k", "j", "i"),
        dt=0.01,
        spacing=(0.1, 0.1, 0.1),
        origin=(0, 0, 0),
        attrs={"transmit_frequency": 15.625e6, "beamforming_sound_velocity": 1540.0},
    )


def filter_kwargs(helper: str, width: int) -> dict[str, Any]:
    """Describe the filtering and nested integration used by a helper.

    Parameters
    ----------
    helper : str
        Helper configuration name.
    width : int
        Outer window width.

    Returns
    -------
    dict
        Shared wrapper/kernel keyword arguments.
    """
    kwargs: dict[str, Any] = {
        "filter_method": "butterworth"
        if helper.endswith("butterworth")
        else "svd_indices",
        "low_cutoff": 5.0 if helper.endswith("butterworth") else 2,
    }
    prefix = "velocity" if helper.startswith("velocity") else "doppler"
    kwargs[f"{prefix}_window_width"] = width // 2
    kwargs[f"{prefix}_window_stride"] = width // 4
    return kwargs


def build_helper(
    helper: str,
    iq: xr.DataArray,
    width: int,
    stride: int,
    batch: int | None,
) -> tuple[da.Array, dict[str, Any]]:
    """Construct a real public helper graph with an injected batching request.

    Parameters
    ----------
    helper : str
        Public helper/filter configuration.
    iq : xarray.DataArray
        IQ VoxelData.
    width, stride : int
        Scientific outer window width and stride.
    batch : int, optional
        Requested windows per task. If not provided, retains production defaults.

    Returns
    -------
    result : dask.array.Array
        Complete helper result, including threshold reduction where applicable.
    layout : dict
        Processing graph sizes and effective per-task window counts.
    """
    original = iq_process.process_iq_blocks
    processing_graphs = []

    def capture(*args: Any, **kwargs: Any) -> da.Array:
        """Construct and record the production processing graph.

        Parameters
        ----------
        *args, **kwargs
            Arguments supplied by the public helper.

        Returns
        -------
        dask.array.Array
            Unchanged production result with the requested batching option.
        """
        result = original(*args, windows_per_task=batch, **kwargs)
        processing_graphs.append(result)
        return result

    with patch.object(iq_process, "process_iq_blocks", capture):
        if helper == "bmode":
            output = process_iq_to_bmode(iq, width, stride)
        elif helper == "threshold":
            output = compute_svd_cumulative_energy_threshold(
                iq,
                2,
                window_width=width,
                window_stride=stride,
            )
        elif helper.startswith("velocity"):
            output = process_iq_to_axial_velocity(
                iq,
                width,
                stride,
                **filter_kwargs(helper, width),
            )
        else:
            output = process_iq_to_power_doppler(
                iq,
                width,
                stride,
                **filter_kwargs(helper, width),
            )
    assert len(processing_graphs) == 1
    processing = processing_graphs[0]
    outputs_per_window = 1 if helper in ("bmode", "threshold") else 3
    counts = [int(size // outputs_per_window) for size in processing.chunks[0]]
    result = output.data if isinstance(output, xr.DataArray) else output
    assert isinstance(result, da.Array)
    return result, {
        "graph_tasks": len(result.dask),
        "processing_tasks": int(np.prod(processing.numblocks)),
        "windows_per_processing_task": counts,
        "max_input_frames_per_task": width + (max(counts) - 1) * stride,
    }


def reference(helper: str, values: npt.NDArray, width: int, stride: int) -> npt.NDArray:
    """Evaluate each scientific window serially, without Dask batching.

    Parameters
    ----------
    helper : str
        Helper/filter configuration.
    values : numpy.ndarray
        Complete input samples.
    width, stride : int
        Scientific window sizes.

    Returns
    -------
    numpy.ndarray
        Serial reference measurements, or a scalar minimum energy threshold.
    """
    results = []
    for start in range(0, len(values) - width + 1, stride):
        window = values[start : start + width]
        if helper == "threshold":
            signals = window.reshape(width, -1).astype(np.complex128)
            eigenvalues = np.linalg.eigvalsh(signals @ signals.conj().T)
            results.append(np.cumsum(eigenvalues)[-3].real)
        elif helper == "bmode":
            results.append(compute_bmode_volume(window))
        elif helper.startswith("velocity"):
            results.append(
                compute_axial_velocity_volume(
                    window, fs=100.0, **filter_kwargs(helper, width)
                )
            )
        else:
            results.append(
                compute_power_doppler_volume(
                    window, fs=100.0, **filter_kwargs(helper, width)
                )
            )
    return (
        np.asarray(min(results)) if helper == "threshold" else np.concatenate(results)
    )


def summarize(durations: list[float]) -> dict[str, Any]:
    """Summarize measured durations in milliseconds.

    Parameters
    ----------
    durations : list[float]
        Durations in seconds.

    Returns
    -------
    dict
        Median, range, and individual measurements.
    """
    samples = [duration * 1000 for duration in durations]
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def benchmark_case(
    helper: str,
    iq: xr.DataArray,
    values: npt.NDArray,
    width: int,
    stride: int,
    batches: list[int | None],
    repeats: int,
) -> dict[str, Any]:
    """Time shuffled batching variants and check every result against a reference.

    Parameters
    ----------
    helper : str
        Helper/filter configuration.
    iq : xarray.DataArray
        Chunked input with geometry.
    values : numpy.ndarray
        Eager samples for the independent window-loop reference.
    width, stride : int
        Scientific window sizes.
    batches : list[int or None]
        Requested task batch sizes, including the unchanged default.
    repeats : int
        Measured repetitions after one warm-up per variant.

    Returns
    -------
    dict
        Case parameters, graph layouts, and timing summaries.

    Raises
    ------
    AssertionError
        If computed values or dtype do not match the serial reference.
    """
    expected = reference(helper, values, width, stride)
    records: dict[str, Any] = {}
    rng = np.random.default_rng(42)
    for repetition in range(repeats + 1):
        for index in rng.permutation(len(batches)):
            batch = batches[index]
            label = "default" if batch is None else str(batch)
            gc.collect()
            start = time.perf_counter()
            output, layout = build_helper(helper, iq, width, stride, batch)
            built = time.perf_counter()
            actual = output.compute()
            finished = time.perf_counter()
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-10)
            assert actual.dtype == output.dtype == expected.dtype
            if label not in records:
                records[label] = {"build": [], "compute": [], "total": [], **layout}
            if repetition:
                records[label]["build"].append(built - start)
                records[label]["compute"].append(finished - built)
                records[label]["total"].append(finished - start)
    for record in records.values():
        for phase in ("build", "compute", "total"):
            record[phase] = summarize(record[phase])
    return {
        "helper": helper,
        "input_shape": list(values.shape),
        "source_chunks": list(iq.data.chunks[0]),
        "width": width,
        "stride": stride,
        "outer_windows": (len(values) - width) // stride + 1,
        "dtype": str(expected.dtype),
        "repeats": repeats,
        "variants": records,
    }


def memory_worker(helper: str, batch: int | None) -> dict[str, Any]:
    """Sample resident memory for a larger helper workload in an isolated process.

    Parameters
    ----------
    helper : str
        Helper/filter configuration.
    batch : int, optional
        Requested batch size; an omitted value retains the default.

    Returns
    -------
    dict
        Baseline RSS, graph-construction increase, and peak increase in MiB.
    """
    values = make_values((2048, 1, 64, 64))
    iq = make_iq(values, 128)
    pilot, _ = build_helper(helper, iq.isel(time=slice(0, 128)), 128, 16, batch)
    pilot.compute()
    del pilot
    gc.collect()
    process = psutil.Process()
    baseline = process.memory_info().rss
    samples = [baseline]
    stop = threading.Event()

    def sample() -> None:
        """Sample process RSS until computation finishes."""
        while not stop.wait(0.001):
            samples.append(process.memory_info().rss)

    sampler = threading.Thread(target=sample)
    sampler.start()
    try:
        output, layout = build_helper(helper, iq, 128, 16, batch)
        build_peak = max(samples + [process.memory_info().rss])
        actual = output.compute()
        assert np.isfinite(actual).all()
        samples.append(process.memory_info().rss)
    finally:
        stop.set()
        sampler.join()
    return {
        "baseline_mib": baseline / 2**20,
        "peak_mib": max(samples) / 2**20,
        "incremental_mib": (max(samples) - baseline) / 2**20,
        "build_incremental_mib": (build_peak - baseline) / 2**20,
        **layout,
    }


def main() -> None:
    """Run the timing and isolated-memory sweeps, then write JSON results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--memory-repeats", type=int, default=2)
    parser.add_argument("--memory-helper", choices=HELPERS)
    parser.add_argument("--memory-batch", type=int, default=0)
    args = parser.parse_args()
    if args.repeats < 1 or args.workers < 1 or args.memory_repeats < 0:
        parser.error(
            "repeats/workers must be positive; memory-repeats must be nonnegative"
        )
    with (
        threadpool_limits(limits=1),
        dask.config.set(scheduler="threads", num_workers=args.workers),
    ):
        if args.memory_helper:
            print(
                json.dumps(memory_worker(args.memory_helper, args.memory_batch or None))
            )
            return
        if args.output is None:
            parser.error("--output is required")
        results: dict[str, Any] = {
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "cpu_count": os.cpu_count(),
                "ram_gib": psutil.virtual_memory().total / 2**30,
                "versions": {
                    module.__name__: module.__version__
                    for module in (np, scipy, dask, xr)
                },
                "threadpools": threadpool_info(),
            },
            "method": {
                "workers": args.workers,
                "blas_threads": 1,
                "repeats": args.repeats,
                "warmups_per_variant": 1,
                "batch_sizes": [None, 1, 2, 4, 8, 16, 32],
                "storage": "in-memory NumPy source chunks; no persistence",
            },
            "cases": {},
            "memory": {},
        }
        batches: list[int | None] = [None, 1, 2, 4, 8, 16, 32]
        values = make_values((1024, 1, 32, 32))
        for chunk in (64, 100):
            iq = make_iq(values, chunk)
            for stride in (64, 32, 8):
                for helper in HELPERS:
                    name = f"small_chunk{chunk}_{helper}_s{stride}"
                    print(f"Benchmarking {name}", flush=True)
                    results["cases"][name] = benchmark_case(
                        helper, iq, values, 64, stride, batches, args.repeats
                    )
        values = make_values((2048, 1, 64, 64))
        iq = make_iq(values, 128)
        for helper in ("bmode", "power_svd", "velocity_svd", "threshold"):
            name = f"large_{helper}_s16"
            print(f"Benchmarking {name}", flush=True)
            results["cases"][name] = benchmark_case(
                helper, iq, values, 128, 16, batches, min(3, args.repeats)
            )
    for helper in ("bmode", "power_svd", "velocity_svd", "threshold"):
        results["memory"][helper] = {}
        for batch in (None, 8, 32):
            label = "default" if batch is None else str(batch)
            print(f"Sampling memory: {helper}, batch={label}", flush=True)
            results["memory"][helper][label] = [
                json.loads(
                    subprocess.check_output(
                        [
                            sys.executable,
                            __file__,
                            "--memory-helper",
                            helper,
                            "--memory-batch",
                            str(batch or 0),
                            "--workers",
                            str(args.workers),
                        ],
                        text=True,
                    )
                )
                for _ in range(args.memory_repeats)
            ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Results: {args.output}", flush=True)


if __name__ == "__main__":
    main()

"""Benchmark complete power Doppler VoxelData pipelines with Dask schedulers.

Run from the repository root with:
`uv run python -m tools.benchmark_iq_power_doppler_dask --output /tmp/pd-dask.json`.
See `tools/benchmark_iq_power_doppler_dask.md` for methodology and larger runs.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from contextlib import ExitStack
from functools import partial
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any

import dask
import dask.array as da
import distributed
import numpy as np
import xarray as xr
from distributed import Client, LocalCluster, wait
from rich.console import Console
from rich.table import Table
from threadpoolctl import threadpool_info, threadpool_limits

from confusius.iq import process_iq_to_power_doppler as process_current
from confusius.validation import validate_voxeldata
from confusius.xarray import create_voxeldata
from tools.benchmark_iq_power_doppler import errors, make_signals
from tools.iq_to_power_doppler import process_iq_to_power_doppler as process_fast


def configure_worker(blas_threads: int, dask_worker: Any) -> None:
    """Limit native thread pools for the lifetime of a Distributed worker.

    Parameters
    ----------
    blas_threads : int
        Native threads allowed per BLAS call.
    dask_worker : distributed.Worker
        Worker supplied by `Client.register_worker_callbacks`.

    Returns
    -------
    None
        The controller is kept alive on the worker.
    """
    dask_worker.pd_thread_limiter = threadpool_limits(limits=blas_threads)


def measure_pipeline(
    iq: xr.DataArray, method: str, options: dict[str, Any]
) -> tuple[xr.DataArray, dict[str, float | int]]:
    """Measure fresh graph construction and complete computation separately.

    Parameters
    ----------
    iq : xarray.DataArray
        Persisted IQ VoxelData input.
    method : str
        `current`, `subtract`, or `retained`.
    options : dict
        Shared sliding-window and index-cutoff arguments.

    Returns
    -------
    output : xarray.DataArray
        Fully computed power Doppler output on the driver.
    sample : dict
        Graph-build, execution/gather, total wall time, and graph task count.
    """
    start = perf_counter()
    if method == "current":
        lazy = process_current(iq, **options)
    else:
        lazy = process_fast(iq, method=method, **options)
    built = perf_counter()
    tasks = len(lazy.data.__dask_graph__())
    execute_start = perf_counter()
    output = lazy.compute()
    end = perf_counter()
    return output, {
        "build_s": built - start,
        "compute_s": end - execute_start,
        "total_s": (built - start) + (end - execute_start),
        "graph_tasks": tasks,
    }


def benchmark_scheduler(args: argparse.Namespace, scheduler: str) -> dict[str, Any]:
    """Benchmark all methods under one local Dask scheduler configuration.

    Parameters
    ----------
    args : argparse.Namespace
        Validated command-line settings.
    scheduler : str
        `threads`, `distributed-threads`, or `distributed-processes`.

    Returns
    -------
    dict
        Setup timings, warmed samples, precision metrics, and machine settings.
    """
    methods = ["current", "subtract", "retained"]
    options = {
        "clutter_window_width": args.frames,
        "clutter_window_stride": args.stride,
        "low_cutoff": args.low,
        "high_cutoff": args.high,
    }
    with ExitStack() as stack:
        stack.enter_context(threadpool_limits(limits=args.blas_threads))
        start = perf_counter()
        client = None
        if scheduler == "threads":
            stack.enter_context(
                dask.config.set(
                    scheduler="threads",
                    num_workers=args.workers * args.threads_per_worker,
                )
            )
        else:
            cluster = stack.enter_context(
                LocalCluster(
                    n_workers=args.workers,
                    threads_per_worker=args.threads_per_worker,
                    processes=scheduler == "distributed-processes",
                    memory_limit=args.worker_memory,
                    dashboard_address=None,
                )
            )
            client = stack.enter_context(Client(cluster))
            client.register_worker_callbacks(
                setup=partial(configure_worker, args.blas_threads)
            )
            stack.enter_context(dask.config.set(scheduler=client))
        setup_s = perf_counter() - start
        frames = args.frames + (args.windows - 1) * args.stride
        start = perf_counter()
        values = make_signals(
            args.voxels, frames, args.tissue_rank, args.db, args.dtype
        )
        iq = create_voxeldata(
            da.from_array(
                values.reshape(frames, 1, 1, args.voxels),
                chunks=(args.time_chunk, 1, 1, args.voxels),
            ),
            dims=("time", "k", "j", "i"),
            dt=0.001,
            spacing=(0.1, 0.1, 0.1),
        )
        generation_s = perf_counter() - start
        start = perf_counter()
        iq = iq.persist()
        if client is not None:
            wait(iq.data)
        persist_s = perf_counter() - start
        del values
        reference, _ = measure_pipeline(iq, "current", options)
        validate_voxeldata(reference)
        # Warm every method once; subsequent samples rebuild the graph each time.
        for method in methods[1:]:
            measure_pipeline(iq, method, options)
        samples: dict[str, list] = {method: [] for method in methods}
        rng = np.random.default_rng(2026)
        for repeat in range(args.repeats):
            for method in rng.permutation(methods).tolist():
                output, sample = measure_pipeline(iq, method, options)
                validate_voxeldata(output)
                np.testing.assert_array_equal(output.time, reference.time)
                sample["error_vs_current"] = errors(output.values, reference.values)
                if not np.isfinite(output.values).all():
                    raise RuntimeError(
                        f"Non-finite power from {method} under {scheduler}."
                    )
                samples[method].append(sample)
                print(
                    f"{scheduler} repeat={repeat + 1} {method}: "
                    f"build={sample['build_s']:.3f}s "
                    f"compute={sample['compute_s']:.3f}s "
                    f"total={sample['total_s']:.3f}s",
                    flush=True,
                )
        summary = {}
        for method in methods:
            summary[method] = {
                key: float(np.median([sample[key] for sample in samples[method]]))
                for key in ("build_s", "compute_s", "total_s")
            }
            summary[method]["speedup_vs_current"] = (
                np.median([sample["total_s"] for sample in samples["current"]])
                / summary[method]["total_s"]
            )
        return {
            "scheduler": scheduler,
            "scheduler_setup_s": setup_s,
            "generation_s": generation_s,
            "input_persist_s": persist_s,
            "input_shape": list(iq.shape),
            "input_chunks": iq.data.chunks,
            "input_bytes": int(iq.nbytes),
            "driver_threadpools": threadpool_info(),
            "workers": client.run(threadpool_info) if client is not None else None,
            "worker_memory": {
                address: {
                    "limit_bytes": info["memory_limit"],
                    "metrics": {
                        key: info.get("metrics", {}).get(key)
                        for key in ("memory", "managed_bytes", "spilled_bytes")
                    },
                }
                for address, info in client.scheduler_info()["workers"].items()
            }
            if client is not None
            else None,
            "summary": summary,
            "samples": samples,
        }


def print_results(runs: list[dict[str, Any]]) -> None:
    """Display scheduler comparisons using median timings and worst observed error.

    Parameters
    ----------
    runs : list of dict
        Completed scheduler benchmarks, including summaries and raw samples.

    Returns
    -------
    None
        A Rich table is printed to the console.
    """
    table = Table(
        title="Power Doppler results",
        caption=(
            "Median times in seconds; speedup versus current under the same scheduler. "
            "Rel. L2 and negative voxel counts are maxima across repetitions."
        ),
    )
    table.add_column("Scheduler")
    table.add_column("Method")
    for column in (
        "Build (s)",
        "Compute (s)",
        "Total (s)",
        "Speedup",
        "Rel. L2",
        "Negative",
    ):
        table.add_column(column, justify="right")
    for run in runs:
        for method, summary in run["summary"].items():
            errors = [sample["error_vs_current"] for sample in run["samples"][method]]
            speedup = summary["speedup_vs_current"]
            table.add_row(
                run["scheduler"],
                method,
                f"{summary['build_s']:.3f}",
                f"{summary['compute_s']:.3f}",
                f"{summary['total_s']:.3f}",
                f"{speedup:.2f}x",
                f"{max(error['relative_l2'] for error in errors):.2e}",
                str(max(error["negative_voxels"] for error in errors)),
            )
        table.add_section()
    Console().print(table)


def main() -> None:
    """Parse settings, run scheduler comparisons, and save incremental JSON results.

    Returns
    -------
    None
        Results are written after each completed scheduler benchmark.

    Raises
    ------
    ValueError
        If output geometry is invalid.
    RuntimeError
        If output power contains non-finite values.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--schedulers",
        nargs="+",
        default=["threads", "distributed-processes"],
        choices=["threads", "distributed-threads", "distributed-processes"],
    )
    parser.add_argument("--voxels", type=int, default=10_000)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--windows", type=int, default=4)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--time-chunk", type=int)
    parser.add_argument("--low", type=int, default=20)
    parser.add_argument("--high", type=int)
    parser.add_argument("--tissue-rank", type=int, default=20)
    parser.add_argument("--db", type=float, default=40)
    parser.add_argument(
        "--dtype", choices=["complex64", "complex128"], default="complex64"
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads-per-worker", type=int, default=1)
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument(
        "--worker-memory",
        default="auto",
        help="Distributed limit per worker, e.g. 4GiB",
    )
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    args.stride = args.frames if args.stride is None else args.stride
    args.time_chunk = args.frames if args.time_chunk is None else args.time_chunk
    high = args.frames if args.high is None else args.high
    if not (2 <= args.frames <= args.voxels and 0 <= args.low < high <= args.frames):
        parser.error("Require 2 <= frames <= voxels and 0 <= low < high <= frames.")
    if (
        min(
            args.windows,
            args.repeats,
            args.workers,
            args.threads_per_worker,
            args.blas_threads,
            args.time_chunk,
            args.tissue_rank,
        )
        < 1
    ):
        parser.error("Counts and thread settings must be positive.")
    if not 1 <= args.stride <= args.frames or args.tissue_rank > args.frames:
        parser.error("Require 1 <= stride <= frames and tissue-rank <= frames.")
    cpu = platform.processor() or platform.machine()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        cpu = next(
            (
                line.split(":", 1)[1].strip()
                for line in cpuinfo.read_text().splitlines()
                if line.startswith("model name")
            ),
            cpu,
        )
    metadata = {
        "command": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "numpy": np.__version__,
        "scipy": version("scipy"),
        "confusius": version("confusius"),
        "dask": dask.__version__,
        "distributed": distributed.__version__,
        "settings": {**vars(args), "output": str(args.output)},
        "runs": [],
    }
    for scheduler in args.schedulers:
        run = benchmark_scheduler(args, scheduler)
        metadata["runs"].append(run)
        args.output.write_text(json.dumps(metadata, indent=2) + "\n")
    print_results(metadata["runs"])
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()

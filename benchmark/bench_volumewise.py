"""Benchmark joblib-backed vs Dask-backed volumewise registration.

Usage
-----
uv run python benchmark/bench_volumewise.py
"""

import tempfile
import time
from pathlib import Path

import dask
import dask.array as da
import h5py
import numpy as np
from distributed import Client

from confusius.registration.volumewise import register_volumewise
from confusius.registration.volumewise_dask import register_volumewise_dask
from confusius.xarray import create_voxeldata

N_FRAMES = 40
SHAPE = (24, 24, 24)
N_WORKERS = 8
COMMON_KWARGS = dict(
    transform="rigid",
    metric="correlation",
    number_of_iterations=50,
    use_multi_resolution=False,
)


def make_recording(rng: np.random.Generator) -> np.ndarray:
    base = rng.random(SHAPE).astype(np.float32)
    shifts = rng.integers(-2, 3, size=(N_FRAMES, 3))
    return np.stack(
        [np.roll(base, tuple(s), axis=(0, 1, 2)) for s in shifts]
    )


def bench(label: str, fn) -> float:
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    n_converged = (result.attrs["motion_params"]["status"] == "completed").sum()
    print(f"{label}: {elapsed:.2f}s ({n_converged}/{N_FRAMES} converged)")
    return elapsed


def main() -> None:
    rng = np.random.default_rng(0)
    recording = make_recording(rng)
    data = create_voxeldata(
        recording, dims=("time", "k", "j", "i"), spacing=(1, 1, 1), dt=1.0
    )

    print(f"n_frames={N_FRAMES}, shape={SHAPE}, n_workers={N_WORKERS}\n")

    print("-- in-memory numpy input --")
    t_joblib = bench(
        "joblib (n_jobs=%d, processes)" % N_WORKERS,
        lambda: register_volumewise(
            data, n_jobs=N_WORKERS, show_progress=False, **COMMON_KWARGS
        ),
    )
    with Client(n_workers=N_WORKERS, threads_per_worker=1, processes=True) as _client:
        t_dask_proc = bench(
            "dask (%d worker processes)" % N_WORKERS,
            lambda: register_volumewise_dask(data, **COMMON_KWARGS),
        )
    # Local threaded scheduler (no distributed.Client): tasks run as real threads
    # sharing the parent process's memory, so nothing needs pickling.
    with dask.config.set(scheduler="threads", num_workers=N_WORKERS):
        t_dask_thread = bench(
            "dask (local threaded scheduler, %d threads)" % N_WORKERS,
            lambda: register_volumewise_dask(data, **COMMON_KWARGS),
        )

    print("\n-- h5py-backed lazy input (never materialized before the call) --")
    with tempfile.TemporaryDirectory() as tmp_dir:
        h5_path = Path(tmp_dir) / "recording.h5"
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("data", data=recording)
        with h5py.File(h5_path, "r") as f:
            lazy_array = da.from_array(f["data"], chunks=(1, *SHAPE))
            lazy_data = create_voxeldata(
                lazy_array,
                dims=("time", "k", "j", "i"),
                spacing=(1, 1, 1),
                dt=1.0,
            )

            try:
                register_volumewise(
                    lazy_data, n_jobs=N_WORKERS, show_progress=False, **COMMON_KWARGS
                )
                print("joblib (n_jobs=%d): unexpectedly succeeded" % N_WORKERS)
            except TypeError as exc:
                print(f"joblib (n_jobs={N_WORKERS}): raises TypeError ({exc})")

            with dask.config.set(scheduler="threads", num_workers=N_WORKERS):
                t_dask_lazy = bench(
                    "dask (local threaded scheduler, %d threads, h5py-lazy)"
                    % N_WORKERS,
                    lambda: register_volumewise_dask(lazy_data, **COMMON_KWARGS),
                )

    print("\n-- summary (in-memory input) --")
    print(f"joblib processes:  {t_joblib:.2f}s")
    print(f"dask processes:    {t_dask_proc:.2f}s")
    print(f"dask threads:      {t_dask_thread:.2f}s")
    print(f"dask threads lazy: {t_dask_lazy:.2f}s")


if __name__ == "__main__":
    main()

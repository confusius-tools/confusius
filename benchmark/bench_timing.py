"""Timing comparison on real fUSI data, randomized/interleaved trial order.

Uses the same recording/window as
`docs/examples/03_registration/02_volumewise_motion_correction.py` (Cybis
Pereira 2026 dataset, rat75/20220523/slice32, frames 220:340).

Avoids the fixed-order bias of the first benchmark pass (joblib always
cold-started first, dask always run later/warmer): each of N_ROUNDS rounds
runs all configurations in a fresh random order, and round 0 is discarded as
warm-up.

Usage
-----
uv run python benchmark/bench_timing.py
"""

import random
import statistics
import time
from pathlib import Path

from distributed import Client

import confusius as cf
from confusius.registration.volumewise import register_volumewise

SUBJECT = "rat75"
SESSION = "20220523"
ACQ = "slice32"
START_FRAME = 220
N_FRAMES = 120
N_WORKERS = 8
N_ROUNDS = 5  # round 0 discarded as warm-up
COMMON_KWARGS = {
    "transform": "rigid",
    "metric": "correlation",
    "learning_rate": 1.0,
}


def load_recording():
    bids_root = cf.datasets.fetch_cybis_pereira_2026(
        datasets="rawdata", subjects=SUBJECT, sessions=SESSION, acqs=ACQ
    )
    pwd_path = (
        Path(bids_root)
        / f"sub-{SUBJECT}"
        / f"ses-{SESSION}"
        / "fusi"
        / f"sub-{SUBJECT}_ses-{SESSION}_task-openfield_acq-{ACQ}_pwd.nii.gz"
    )
    return (
        cf.load(pwd_path)
        .isel(time=slice(START_FRAME, START_FRAME + N_FRAMES))
        .compute()
    )


def main() -> None:
    data = load_recording()
    print(f"data: {dict(data.sizes)}, dtype={data.dtype}\n")

    client = Client(n_workers=N_WORKERS, threads_per_worker=1, processes=True)

    configs = {
        "joblib (n_jobs=8, loky processes)": lambda: register_volumewise(
            data, n_jobs=N_WORKERS, show_progress=False, **COMMON_KWARGS
        ),
    }
    # Imported after the distributed Client is created so it's the ambient
    # scheduler when register_volumewise_dask calls dask.compute(). Threads
    # are deliberately excluded here: bench_memory.py showed SimpleITK barely
    # releases the GIL for this real registration workload (~110% CPU out of
    # 8 requested threads), so a threaded scheduler would not be a fair
    # parallel-throughput comparison against process-based joblib.
    from confusius.registration.volumewise_dask import register_volumewise_dask

    configs["dask (8 worker processes via distributed.Client)"] = (
        lambda: register_volumewise_dask(data, **COMMON_KWARGS)
    )

    order = random.Random(0)
    timings: dict[str, list[float]] = {name: [] for name in configs}

    print(
        f"n_frames={N_FRAMES}, n_workers={N_WORKERS}, {N_ROUNDS} rounds "
        "(round 0 = warm-up, discarded)\n"
    )

    for round_idx in range(N_ROUNDS):
        names = list(configs)
        order.shuffle(names)
        round_times = {}
        for name in names:
            t0 = time.perf_counter()
            configs[name]()
            elapsed = time.perf_counter() - t0
            round_times[name] = elapsed
            if round_idx > 0:
                timings[name].append(elapsed)
        print(f"round {round_idx} order={names}")
        for name in names:
            tag = " (warm-up, discarded)" if round_idx == 0 else ""
            print(f"  {name}: {round_times[name]:.2f}s{tag}")

    client.close()

    print("\n-- median over rounds 1.. --")
    for name, values in timings.items():
        formatted = [f"{v:.2f}" for v in values]
        print(f"{name}: median={statistics.median(values):.2f}s  all={formatted}")


if __name__ == "__main__":
    main()

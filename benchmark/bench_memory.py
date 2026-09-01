"""Peak-RSS comparison: eager `.values` materialization vs lazy Dask reads.

Each configuration runs in its own subprocess (`resource.getrusage(...).
ru_maxrss` is a process-lifetime high-water mark, so it must be measured in a
fresh process per case to be meaningful). Input is the same window used in
`docs/examples/03_registration/02_volumewise_motion_correction.py` and in
bench_timing.py (Cybis Pereira 2026 dataset, rat75/20220523/slice32, frames
220:340), sliced but never `.compute()`-d.

Not the full 1800-frame recording: a direct probe (`data.isel(time=t).values`
for several `t`) showed gzip-compressed NIfTI pays a near-constant ~0.4-0.6s
per single-frame read *regardless of index* -- i.e. each independent
`isel(time=t)` re-decompresses close to the whole file rather than seeking,
since gzip has no random access. That cost is comparable to or larger than
the SimpleITK registration compute itself, so reading 1800 frames
independently (as the lazy per-frame `isel` in register_volumewise_dask
does) took over 12 minutes and was killed rather than let finish -- this is
a real limitation of the lazy-read strategy for gzip-compressed formats
specifically, not a bug in the benchmark. h5py/SCAN files (chunked, real
random access) would not pay this repeated cost; this dataset does.

Note: `ru_maxrss` measures the process's own resident memory, not OS page
cache, so OS-level disk caching does not affect this measurement -- it only
affects wall-clock read time (see bench_timing.py / README).

The lazy config uses a process-based `distributed.Client`, not a threaded
scheduler: bench_timing.py found SimpleITK barely releases the GIL for this
real registration workload, so threads are both much slower and an unfair
comparison. This means the RSS measured here is only the *driver* process's
memory -- each dask worker process has its own separate address space not
counted in this number, same as joblib's loky workers aren't counted in the
joblib driver's RSS either. Both configurations are compared on the same
basis: driver-process memory only.

Usage
-----
uv run python benchmark/bench_memory.py
"""

import resource
import subprocess
import sys
import textwrap

START_FRAME = 220
N_FRAMES = 120
N_WORKERS = 8

LOAD_SNIPPET = f"""
from pathlib import Path
import confusius as cf

bids_root = cf.datasets.fetch_cybis_pereira_2026(
    datasets="rawdata", subjects="rat75", sessions="20220523", acqs="slice32"
)
pwd_path = (
    Path(bids_root) / "sub-rat75" / "ses-20220523" / "fusi"
    / "sub-rat75_ses-20220523_task-openfield_acq-slice32_pwd.nii.gz"
)
data = cf.load(pwd_path).isel(time=slice({START_FRAME}, {START_FRAME + N_FRAMES}))
# lazy, NEVER .compute()-d
"""

EAGER_SCRIPT = LOAD_SNIPPET + f"""
from confusius.registration.volumewise import register_volumewise

register_volumewise(
    data,
    n_jobs={N_WORKERS},
    transform="rigid",
    metric="correlation",
    learning_rate=1.0,
    show_progress=False,
)
"""

LAZY_SCRIPT = LOAD_SNIPPET + f"""
from distributed import Client
from confusius.registration.volumewise_dask import register_volumewise_dask

with Client(n_workers={N_WORKERS}, threads_per_worker=1, processes=True):
    register_volumewise_dask(
        data,
        transform="rigid",
        metric="correlation",
        learning_rate=1.0,
    )
"""

RSS_WRAPPER = """
import resource
{body}
print("PEAK_RSS_KB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
"""


def run(label: str, script_body: str) -> int:
    full_script = RSS_WRAPPER.format(body=textwrap.dedent(script_body))
    proc = subprocess.run(
        [sys.executable, "-c", full_script],
        capture_output=True,
        text=True,
        check=True,
    )
    peak_kb = None
    for line in proc.stdout.splitlines():
        if line.startswith("PEAK_RSS_KB"):
            peak_kb = int(line.split()[1])
    if peak_kb is None:
        print(proc.stdout)
        print(proc.stderr)
        raise RuntimeError(f"{label}: did not report peak RSS")
    print(f"{label}: peak RSS = {peak_kb / 1024:.0f} MiB")
    return peak_kb


def main() -> None:
    raw_mb = N_FRAMES * 112 * 128 * 8 / 1e6
    print(
        f"window: {N_FRAMES} frames, (1, 112, 128) float64 = {raw_mb:.0f} MB raw. "
        f"n_workers={N_WORKERS}\n"
    )
    eager_kb = run(
        "register_volumewise (n_jobs=8, joblib, eager .values)", EAGER_SCRIPT
    )
    lazy_kb = run(
        "register_volumewise_dask (distributed, 8 processes, lazy isel)",
        LAZY_SCRIPT,
    )
    print(f"\ndelta: {(eager_kb - lazy_kb) / 1024:.0f} MiB (expected ~{raw_mb / 1e3 * 1024:.0f} MiB, i.e. the materialized window size)")


if __name__ == "__main__":
    main()

# register_volumewise: joblib vs. Dask prototype

`src/confusius/registration/volumewise_dask.py` is a prototype
`register_volumewise_dask` that replaces joblib with `dask.delayed` +
`dask.compute`. Behavior/return value matches `register_volumewise` (same
`motion_params`/diagnostics attrs); `n_jobs`, `show_progress`,
`progress_reporter`, and `abort_event` were dropped for this prototype —
parallelism is controlled by whichever Dask scheduler/`Client` is active when
`dask.compute()` runs, and progress/cancellation would need a Dask-native
mechanism (e.g. `distributed`'s progress/future cancellation) if this gets
productionized.

All numbers below use the real recording/window from
`docs/examples/03_registration/02_volumewise_motion_correction.py` (Cybis
Pereira 2026 dataset, rat75/20220523/slice32, frames 220:340 — 120 frames,
`(1, 112, 128)` float64), 8 workers.

## What was actually broken

Not joblib parallelization itself — `for t, volume in enumerate(data_moved)`
already indexes lazily, one frame at a time. The materialization happened at
`arr = data_moved.values` (`volumewise.py:278`), which pulls the *entire*
recording into RAM up front just to build the blank-frame output buffer,
regardless of `n_jobs`. Separately, `n_jobs != 1` requires the whole array to
be process-pickle-able, so h5py-backed (lazy SCAN) data is rejected outright
unless already `.compute()`-d — though NIfTI-backed lazy data (this dataset)
isn't caught by that specific guard, since it only checks for h5py.

The Dask version drops the eager `.values` call — each frame is read via
`data_moved.isel(time=t)` inside its own `dask.delayed` task, so nothing is
pulled into memory before the scheduler decides to run that frame's task.

## Timing: joblib vs. dask, real data, randomized trial order

`benchmark/bench_timing.py`. First pass (not included here) ran joblib and
dask in a fixed order every round and found dask ~1.8x faster — that result
was a warm-cache/order artifact from a synthetic-data run, not a real effect;
seeded here from a review question about cache/order control. Re-run with
randomized per-round ordering, round 0 discarded as warm-up, materialized
in-memory input for both (process-based only — see GIL note below):

```
joblib (n_jobs=8, loky processes):               median=7.86s  all=[7.46, 7.99, 8.18, 7.73]
dask (8 worker processes via distributed.Client): median=7.76s  all=[7.70, 7.82, 7.88, 7.56]
```

**Essentially tied.** The two backends have comparable per-task scheduling
overhead at this scale; neither is a meaningful speed win over the other.

**GIL note:** an earlier attempt also benchmarked a threaded Dask scheduler.
On real data, SimpleITK's registration loop barely releases the GIL for this
transform/metric/data combination — 8 requested threads pulled only ~110%
CPU total, so the run never finished in reasonable time and was killed. A
synthetic-data run earlier had suggested threads were viable (and even
fastest); that was almost certainly the same warm-cache/small-sample
artifact as the 1.8x claim above, not a real property of the workload.
Process-based backends only should be used for real throughput comparisons
here.

## Memory: peak driver-process RSS, real data

`benchmark/bench_memory.py`. Each config runs in its own subprocess
(`resource.getrusage(...).ru_maxrss` is a process-lifetime high-water mark,
so it needs a fresh process per case). Same 120-frame window, lazy
(never-`.compute()`-d) NIfTI input for both:

```
register_volumewise (n_jobs=8, joblib, eager .values):          peak RSS = 457 MiB
register_volumewise_dask (distributed, 8 processes, lazy isel):  peak RSS = 391 MiB
delta: 66 MiB (raw materialized window is only ~14 MiB)
```

Direction is right — the eager path holds more in the driver process — but
the delta is bigger than the raw window size alone predicts, likely from
joblib/loky's own import/serialization overhead plus `register_volumewise`
briefly holding both the materialized input and the output buffer at once.
Not chased further; the qualitative point (eager forces full materialization
into one process regardless of parallelism, lazy doesn't) is what mattered
here — see the gzip caveat below for why we didn't push this measurement to
the full 1800-frame recording.

Note: `ru_maxrss` measures the process's own resident memory, not OS page
cache, so OS-level disk caching does not affect this measurement — only
wall-clock read time (below) is sensitive to that.

## Page cache and the timing numbers

The timing numbers above were run with a warm OS page cache (this session
has no `sudo`, so caches weren't dropped). That only affects wall-clock
*read* time, not the RSS numbers above. For a true cold-disk-read timing
run:

```
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

then re-run `bench_timing.py`. Given per-frame registration compute
(~0.3–0.5s) dominates wall time far more than a single-digit-MB gzip read
once decompressed, we don't expect this to change the joblib-vs-dask
comparison materially — but it wasn't verified.

## gzip-compressed NIfTI: a real lazy-read pathology (separate from this PR)

Investigating why a full 1800-frame lazy run hung for 12+ minutes surfaced an
issue independent of joblib vs. dask: gzip has no random access, so any read
of a gzip-backed NIfTI must decompress sequentially from the start. Verified
directly:

```
chunks: (1170, 630)                     # this file's default dask chunking along time
frame 0   (cold):              0.89s
frame 1   (cold, same chunk):  0.69s    # same chunk as frame 0, same cost -- no reuse
frame 0   (after .persist()):  0.002s
frame 500 (after .persist()):  0.002s
frame 1799(after .persist()):  0.002s
```

Dask does not cache results across independent `.compute()`/`.values()`
calls by default, so a loop doing many independent `isel(time=t).values()`
reads against an *unpersisted* lazy array — exactly what
`register_volumewise_dask`'s per-frame `dask.delayed` tasks were doing —
re-pays that full decompression cost on every single frame. That's what
made the full-recording lazy run take far longer than the eager (joblib)
path instead of the intended win.

**Considered and reverted**: `register_volumewise_dask` briefly called
`data_moved.persist()` once before building the per-frame delayed tasks.
This does fix the redundant-decompression pathology (verified: 300 real,
never-materialized frames completed in ~16.5s with it, vs. 12+ minutes
without for the full 1800-frame recording) — but `.persist()` forces the
*entire* array to be computed and held in memory (distributed across workers
with a `distributed.Client`, not concentrated in the driver, but the same
total memory as fully materializing it). That trades away the RAM benefit
that was the actual point of the lazy path, for every backing — including
h5py/SCAN data, which never had this problem in the first place (HDF5
supports genuine chunked random access, confirmed by the earlier h5py
benchmark completing fine with no persist). So the fix was reverted; this is
now documented as a caveat (see below and the docstring) rather than baked
into the function as a behavior change.

This is a general gotcha for any Dask-based per-frame pipeline over
`cf.load()`-ed gzip NIfTI data, not specific to this function — see
[confusius-tools/confusius#439](https://github.com/confusius-tools/confusius/issues/439)
for measurements. If a caller knows their input is gzip-backed and has the
RAM/cluster capacity, they can call `.persist()` themselves before passing
data in; the function doesn't do it for them.

## Reproduce

```bash
uv run python benchmark/bench_timing.py
uv run python benchmark/bench_memory.py
```

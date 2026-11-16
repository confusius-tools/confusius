# Lazy IQ processing and batch-size benchmark

## Roadmap and implementation

See [the roadmap](iq_blocks_roadmap.md). This work is stacked on
`feat/sample-beamformed-iq` (parent PR #576).

Implemented in order:

1. Deterministic precision for potentially active filters, including adaptive
   windows selecting zero clutter vectors. Identifiable no-op filters and B-mode
   preserve input precision; wider input types are not intentionally narrowed.
2. Required keyword-only `output_shape` and `output_dtype` declarations in
   `process_iq_blocks`, describing one callback on one scientific window. Both eager
   inference sites are gone. Callback shape/dtype errors are checked during execution.
   All three image wrappers and the cumulative-energy threshold helper supply
   metadata internally; wrapper/accessor signatures are unchanged.
3. Public-API regression tests for zero IQ work during construction, one callback
   per scientific window, dtype/shape mismatches, added/dropped axes, one/two/partial
   batches, nested output volumes, invalid windows/lags, and adaptive dtype stability.
4. Opt-in low-level `windows_per_task` and a benchmark sweep of actual production
   helper graphs. Defaults and scientific window semantics remain unchanged.

There is no eager compatibility fallback for the low-level API. Existing custom
calls must declare their per-window shape/dtype. Those names, plus
`windows_per_task`, are now reserved rather than forwarded to callbacks. Bind
callback arguments with `functools.partial` if they use the same names.

Supplied masks are still materialized for validation; NumPy inputs are still
converted to Dask arrays by the wrappers. This is a guarantee of no speculative
**IQ processing**, not a promise of zero work for every wrapper argument.

### Why this change comes first

The original assessment found real-data probes ran one or two whole execution
batches, then discarded their results and processed those windows again. With
256 input frames, width 64, stride 8, construction invoked the callback 11 times;
final execution invoked it another 25 times. Separate probes also reread source
chunks, and distributed probes run the callback in the client after transferring
the computed input batch there.

Original 64 MiB SVD measurements at parent revision `7400965` showed construction
falling from about 127 ms to 1.4 ms with explicit metadata and unchanged low-level
batching; end-to-end time fell about 13%. Those were the earlier investigation's
measurements, **not** the wrapper-level batch sweep below. The current sweep also
includes wrapper geometry/timing construction and three inner measurements per
power/velocity window, so its absolute times should not be compared directly.

The original adaptive dtype reproduction produced float32 then float64 callback
outputs, but Dask assembled a float32 result. The new shared projection behavior
promotes even an empty adaptive selection, eliminating this first-window dependence.

## Reproduce the batch sweep

```bash
uv run python tools/benchmarks/iq_blocks.py \
    --output tools/benchmarks/iq_blocks_results.json
```

[Benchmark script](iq_blocks.py). Its JSON output is a local generated artifact,
ignored by Git and intentionally excluded from the PR.

For a shorter run, use `--repeats 3 --memory-repeats 0`. The full recorded run used
the defaults: five measured small-case repetitions, three large-case repetitions,
one warm-up per variant, and two isolated subprocesses per memory configuration.

### Method

- Intel Core Ultra 9 275HX, 24 logical CPUs, about 30.8 GiB RAM; Linux.
- Python 3.14.10, NumPy 2.4.6, SciPy 1.18.0, Dask 2026.7.0,
  xarray 2026.9.0. Versions and loaded BLAS libraries are stored in JSON.
- Local threaded scheduler: four workers, one thread in both loaded BLAS libraries.
- Seeded complex64 input; full-spatial source chunks.
- Small input: `(1024, 1, 32, 32)`, 8 MiB, temporal source chunks 64 or 100.
- Small scientific windows: width 64, stride 64, 32, or 8.
- Large input: `(2048, 1, 64, 64)`, 64 MiB, chunks 128, windows 128 / 16.
- Power/velocity use inner width half the outer width and inner stride one quarter
  of it, yielding three measurements per outer window.
- Active index SVD removes the first two components. Butterworth uses a 5 Hz
  high-pass at 100 Hz sampling. Threshold calibration uses singular-value index 2.
- Requested batches: unchanged default, 1, 2, 4, 8, 16, and 32. Requests below the
  overlap minimum are clamped; actual window counts are recorded.
- Public helpers are called normally. A temporary benchmark-only patch injects
  `windows_per_task` into their call to the actual production `process_iq_blocks`.
  There is no alternate benchmark orchestration implementation.
- Every timed output is checked against a serial scientific-window loop, including
  actual dtype. Threshold reference spectra use NumPy eigendecomposition.
- Strategy order is shuffled per repetition. Input allocation, reference computation,
  garbage collection before trials, and correctness checks are outside timing.
- Build includes wrapper metadata construction; compute executes the complete helper,
  including the scalar minimum for threshold calibration. Total includes both.
- Reported timings are medians of total durations in milliseconds. Raw JSON contains
  build/compute breakdowns, ranges, individual samples, and unoptimized graph sizes.

There are 40 workload configurations and 280 case/batch variants. The full run
checked 1,344 measured executions plus 280 warm-ups and 24 isolated memory runs.

## Small aligned input: half-overlap windows

Width 64, stride 32; 31 scientific windows. The minimum batch is one window.

| Helper | Default | 1 | 2 | 4 | 8 | 16 | 32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B-mode | 22.31 | 24.08 | 15.08 | 13.62 | 11.17 | 8.47 | 9.70 |
| Power, SVD | 40.71 | 39.76 | 32.46 | 31.67 | 29.82 | 35.09 | 45.59 |
| Power, Butterworth | 51.66 | 50.44 | 45.41 | 43.73 | 41.58 | 51.92 | 63.62 |
| Velocity, SVD | 43.10 | 41.64 | 36.72 | 33.51 | 33.80 | 36.59 | 48.11 |
| Velocity, Butterworth | 44.62 | 42.76 | 39.20 | 36.63 | 34.35 | 43.03 | 65.23 |
| Threshold | 31.63 | 34.17 | 25.85 | 21.69 | 17.31 | 18.77 | 27.18 |

Moderate batching helps every helper in this case. B-mode benefits most: 16-window
batches reduce the measured total about 2.6-fold. Power/velocity/threshold tend to
prefer four or eight windows, rather than the largest possible batch.

## Small aligned input: disjoint windows

Width/stride 64; only 16 scientific windows.

| Helper | Default | 4 | 8 | 16 | 32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| B-mode | 11.01 | 9.66 | 9.17 | 7.09 | 8.02 |
| Power, SVD | 23.43 | 23.75 | 30.21 | 39.91 | 27.06 |
| Power, Butterworth | 26.78 | 26.81 | 30.53 | 36.16 | 35.62 |
| Velocity, SVD | 22.32 | 22.07 | 26.45 | 28.35 | 27.75 |
| Velocity, Butterworth | 27.73 | 26.96 | 30.22 | 44.94 | 36.92 |
| Threshold | 18.36 | 16.88 | 17.83 | 17.53 | 25.96 |

At eight windows per task there are only two processing tasks; at 16 or 32 there
is only one. Cheap B-mode can still benefit from reduced overhead, while expensive
SVD/Butterworth kernels lose useful parallelism. Differences among tiny B-mode
measurements are noisy; the repeated default and batch-one runs use the same layout
and illustrate this variability.

## Small aligned input: strong overlap

Width 64, stride 8; 121 windows. The overlap minimum is **seven**, so requests
1, 2, and 4 do not actually produce those small batches.

| Helper | Default (7) | 8 | 16 | 32 |
| --- | ---: | ---: | ---: | ---: |
| B-mode | 18.74 | 14.60 | 13.29 | 14.66 |
| Power, SVD | 108.96 | 102.56 | 103.44 | 101.78 |
| Power, Butterworth | 134.93 | 134.99 | 130.92 | 125.83 |
| Velocity, SVD | 122.95 | 117.54 | 110.52 | 115.74 |
| Velocity, Butterworth | 141.52 | 136.78 | 134.78 | 132.07 |
| Threshold | 65.77 | 62.02 | 61.58 | 59.20 |

The existing minimum batching already amortizes much of the overhead. Additional
improvements are smaller for expensive kernels than in the half-overlap case.

## Chunk alignment changes the result

Selected misaligned-source results: temporal source chunks 100, windows 64 / 32.

| Helper | Default | 8 | 16 | 32 |
| --- | ---: | ---: | ---: | ---: |
| B-mode | 28.60 | 12.32 | 11.65 | 13.11 |
| Power, SVD | 42.79 | 35.38 | 35.38 | 69.81 |
| Velocity, SVD | 40.34 | 31.01 | 36.20 | 48.17 |
| Threshold | 34.89 | 18.15 | 19.25 | 27.16 |

All source-chunk/stride/filter combinations are in the JSON. Alignment and task
count are both relevant; the best setting is not an intrinsic constant of a helper.
The particularly slow individual combinations should be interpreted alongside
raw ranges, not as a stable universal slowdown factor.

## Larger strong-overlap workload

Width 128, stride 16; 121 windows, source chunks 128. Median total milliseconds:

| Helper | Default (7) | 8 | 16 | 32 |
| --- | ---: | ---: | ---: | ---: |
| B-mode | 38.62 | 34.65 | 35.05 | 38.30 |
| Power, SVD | 885.31 | 846.46 | 853.19 | 845.95 |
| Velocity, SVD | 964.54 | 918.68 | 927.99 | 921.14 |
| Threshold | 531.08 | 495.73 | 492.01 | 518.24 |

Eight-window batches give modest improvements across all four helpers here.
Power's eight- and 32-window totals are effectively tied at this measurement scale;
the memory costs are not tied.

For the image helpers, graph keys / processing tasks are:

| Requested batch | Graph keys | Processing tasks | Actual window counts |
| --- | ---: | ---: | --- |
| Default / 1 / 2 / 4 | 135 | 18 | 2, then seventeen batches of 7 |
| 8 | 79 | 16 | 1, then fifteen batches of 8 |
| 16 | 55 | 8 | 9, then seven batches of 16 |
| 32 | 35 | 4 | 25, then three batches of 32 |

Threshold adds reduction tasks: its corresponding graph sizes are 161, 100, 66,
and 40. Graph-key counts are unoptimized and are not scheduler-executed task counts.

Eight windows advance the temporal core by exactly one source chunk (`8 * 16 =
128`), improving alignment over seven-window cores. This explains why the graph
shrinks considerably even though processing tasks only decrease from 18 to 16.

## Memory

Two fresh subprocesses per configuration, same 64 MiB large workload, four workers.
RSS sampled every millisecond after input creation and a single-window warm-up.
Values are median peak increases over each process's baseline, in MiB:

| Helper | Default | 8 | 32 |
| --- | ---: | ---: | ---: |
| B-mode | 75.55 | 45.52 | 153.23 |
| Power, SVD | 149.74 | 128.92 | 233.85 |
| Velocity, SVD | 165.24 | 140.73 | 244.29 |
| Threshold | 165.48 | 155.06 | 303.83 |

Construction increases were only about 0.05–0.17 MiB across these configurations:
no initial filtering batch is processed to obtain metadata.

Memory baselines include imports, eager input, Dask's retained source allocation,
and warmed allocator state. RSS sampling can miss short-lived allocations, and two
runs are not a precise allocation profile. Nevertheless, the large batch-32 memory
increase is consistent across helpers and is not justified by a consistent speedup.
Bigger temporal cores and halos must be materialized together; output lists and
filter temporaries add to that working set.

## Recommendations

1. **Keep existing defaults.** Explicit metadata and stable dtype are the clear
   unconditional improvements. A new universal batch size is not established.
2. **Offer opt-in low-level tuning only.** `windows_per_task` allows actual tuning
   without adding options to all wrappers or changing scientific parameters.
3. For cheap reductions with many small tasks, try moderate batches. For expensive
   kernels, keep enough processing tasks to occupy workers; avoid making one or two
   enormous tasks from a short recording.
4. Prefer source-aligned temporal cores when feasible, but respect memory and
   parallelism. The minimum halo-safe batch remains a lower bound.
5. Benchmark on representative real recordings before introducing an automatic
   policy or exposing batching in every wrapper. Eight was a sensible compromise
   for this larger workload, not a universal optimum.

## Validation

- 246 IQ tests passed, including the new laziness/dtype/batching regressions.
- 462 focused IQ, xarray, skills, and packaging tests passed.
- 2,919 non-napari unit tests passed with visual regression enabled.
- Repository-wide `ty`, focused Ruff/format checks, and all source pre-commit hooks
  passed (`uv run prek run --files ...`).
- Strict documentation build passed after rendering a fresh gallery with
  `tools/build_gallery.py --no-run`; gallery examples were not executed/downloaded.
  Existing generated gallery artifacts were restored afterwards.

The broad test command was:

```bash
env -u MPLBACKEND QT_QPA_PLATFORM=offscreen uv run pytest \
    tests/unit --ignore=tests/unit/test_napari --mpl -q --no-cov
```

An initial forced `MPLBACKEND=Agg` suppressed inline notebook output; removing that
extra environment override resolved the gallery test. The initial strict docs
failure likewise came from obsolete generated gallery pages, not changed sources.

## Scope and limitations

This sweep measures CPU NumPy processing of synthetic complex64 IQ, local threaded
Dask, two in-memory temporal chunk layouts, and a larger workload. It does not test
real scanner data, cold/remote storage, multiple processes/machines, GPU arrays,
worker spill/retries, or arbitrary spatially distributed SVD. The original study's
local Zarr measurements motivated the laziness change but are not a storage sweep
for these batch sizes.

No scheduler rewrite, native B-mode replacement, scientific filtering change,
automatic batch tuner, wrapper parameter, or new dependency was added.

# Dask power Doppler benchmark

Compare the **actual ConfUSIus VoxelData pipeline** against the experimental
reconstruction-free `subtract` and `retained` pipelines, using:

- Dask's local threaded scheduler (`threads`).
- Distributed with in-process threaded workers (`distributed-threads`).
- Distributed with separate worker processes (`distributed-processes`).

Run from the repository root. No extra dependency is needed: the project already
includes `dask[complete]`. The experiment does not change production code or APIs.

## Run on another computer

```bash
git fetch origin
git switch --track origin/benchmark/iq-power-doppler-dask
uv sync

uv run python -m tools.benchmark_iq_power_doppler_dask \
  --output /tmp/pd-remove20.json \
  --schedulers threads distributed-threads distributed-processes \
  --workers 2 --threads-per-worker 1 --blas-threads 1 \
  --voxels 10000 --frames 300 --windows 4 --repeats 5 \
  --worker-memory 2GiB

# Same workload, but retain only components [20, 60).
uv run python -m tools.benchmark_iq_power_doppler_dask \
  --output /tmp/pd-retain40.json \
  --schedulers threads distributed-threads distributed-processes \
  --workers 2 --threads-per-worker 1 --blas-threads 1 \
  --voxels 10000 --frames 300 --windows 4 --repeats 5 \
  --high 60 --worker-memory 2GiB
```

Save both JSON files; they contain the configuration, package/BLAS versions,
worker thread settings, raw samples, precision metrics, and median summaries.
An interactive Rich progress bar shows the current scheduler, setup/warmup stage,
repetition, and method instead of printing every sample. Progress counts completed
warmups and timed samples, not individual Dask tasks. Rendering happens only between
measured operations, without a background refresh thread. When output is not an
interactive terminal, the progress display is disabled; warnings remain visible.

Results are saved after each completed scheduler. A final Rich table compares
median build, compute, and total time, speedup, and worst observed relative L2
error and negative voxel count across schedulers. Output files are overwritten,
so use distinct names per run.

For a larger workload, change `--voxels 100000 --windows 8` and increase
`--worker-memory` appropriately, e.g. `8GiB` per worker if the machine has enough
RAM. Input alone occupies 1.92 GB with those settings and `complex64`; synthetic
generation and concurrent eigendecompositions need substantially more memory.
Start small and avoid worker spilling if you want an in-memory comparison.

For overlapping windows and a different input chunking:

```bash
uv run python -m tools.benchmark_iq_power_doppler_dask \
  --output /tmp/pd-overlap.json --frames 300 --stride 150 \
  --time-chunk 100 --windows 8 --worker-memory 2GiB
```

To explore parallelism, vary `--workers`, `--threads-per-worker`, and
`--blas-threads` independently. The local threaded scheduler gets
`workers * threads-per-worker` task threads; Distributed gets the specified number
of workers and threads per worker. Start with **one BLAS thread per task** to avoid
oversubscription. BLAS limits are applied in the driver and separately in every
Distributed worker, and the observed worker settings are recorded.

Additional useful parameters:

- `--low 20 --high 60`: descending, half-open retained component interval.
- `--db 80`: expected tissue-to-blood/noise power ratio for cancellation tests.
- `--dtype complex128`: double-precision stored input.
- `--tissue-rank 20`: actual rank of the generated coherent tissue term.
- `--schedulers threads`: run a single scheduler to reduce runtime.

## What is measured

Each scheduler receives the same seeded synthetic IQ mixture: coherent low-rank
complex tissue plus independent complex Gaussian blood/noise. The data is wrapped
as VoxelData with a real voxel-to-world index and acquisition timing metadata.
There are `windows` output images; input length is
`frames + (windows - 1) * stride`. Input spatial shape is `(1, 1, voxels)`.
All processing sees the complete spatial grid per temporal window.

1. Start the scheduler/cluster; record setup time separately.
2. Generate and **persist input** under that scheduler; wait for Distributed
   futures. Generation and initial persistence/transport are measured separately,
   outside pipeline timings. A large-graph warning during initial persistence is
   expected for large synthetic arrays embedded in the graph; that transport is
   not included in the pipeline comparison.
3. Warm all three pipelines once. The current pipeline's warmup also supplies the
   reference power images. This is a numerical-equivalence test against the
   existing method, not a test of biological recovery accuracy.
4. For each repetition, use a reproducibly shuffled method order, build a **fresh
   output graph**, then fully compute and gather the result onto the driver.
   Input remains persisted, but output is never persisted between samples.
5. Record graph-build time, execution/gather time, their sum, and graph task count.
   Correctness checks and graph task counting are outside these timings.

**Use `total_s` as the primary comparison.** The existing `process_iq_blocks`
executes initial blocks while constructing its graph to determine output metadata.
Timing only `.compute()` would omit real filtering work. We retain this behavior
in all pipelines rather than giving an experimental implementation a different
execution structure. `total_s` includes those initial computations, scheduling,
rechunking, reconstruction (where applicable), power reduction, and output gather.
It excludes startup, input setup, task counting, and correctness checks.

`summary` contains the median of each timing field and the ratio of current to
candidate median total time. Median build and compute times need not sum to the
median total. Each sample also includes relative L2 power-image error, maximum
pointwise relative error, and the count of negative voxels against the warmed
current-pipeline reference. Non-finite power fails the run; finite numerical errors
are reported rather than hidden by clipping or a tolerance gate. This allows high
dynamic-range cancellation experiments to complete.

Worker memory limits and sampled memory/spill metrics are recorded for Distributed.
These are **not peak RSS measurements**, and worker metrics may lag computations.
The driver still participates in metadata computations; it needs adequate RAM.

## Local results

Intel Core Ultra 9 275HX, two concurrent task threads/processes, one BLAS thread per
call, 10,000 voxels, four non-overlapping 300-frame windows, `complex64`, 40 dB,
five repetitions. Times below are median **total seconds**, not just `.compute()`.

| Retain | Scheduler | Current | Subtract | Retained | Best speedup |
|:---:|---|---:|---:|---:|---:|
| `[20,300)` | threads | 0.607 | **0.506** | 0.822 | 1.20× |
| `[20,300)` | distributed-threads | 0.626 | **0.524** | 0.839 | 1.19× |
| `[20,300)` | distributed-processes | 0.645 | **0.548** | 0.843 | 1.18× |
| `[20,60)` | threads | 1.143 | 0.802 | **0.506** | 2.26× |
| `[20,60)` | distributed-threads | 1.188 | 0.863 | **0.540** | 2.20× |
| `[20,60)` | distributed-processes | 1.186 | 0.838 | **0.541** | 2.19× |

Raw samples for these two runs are in
`tools/benchmark_iq_power_doppler_dask_results.json`. On this workload, Distributed
added some overhead rather than improving throughput. That is not a general
scheduler recommendation: the input is small and task concurrency was held equal.

Relative L2 image error versus the current pipeline was identical across schedulers:
1.78e-11 for subtraction and 9.32e-15 for retained accumulation in the removal-only
case; 1.06e-10 and 2.44e-14 respectively when retaining 40 components. There were
no negative power values in these 40 dB tests.

## Checks and limits

```bash
uv run pytest tests/unit/test_iq/test_power_doppler_experiment.py --no-cov
```

Tests compare candidate power values, timings, geometry, masks, cutoffs, and
no-filter behavior with the public production API. A CLI test runs all three
schedulers with process spawning, overlapping windows, and temporal rechunking,
checks numerical error, and verifies worker BLAS limits.

The experimental implementation only supports index-based SVD and uniform
integration over the **entire decomposition window**. It requires at least as many
basis-estimation voxels as frames. It is not a replacement for adaptive cutoffs,
short-window integration, or filtered IQ output. Energy subtraction still has the
cancellation risks documented in `tools/benchmark_iq_power_doppler.md`.

Not measured: storage I/O, cold caches, remote/multi-node clusters, real recordings,
GPU execution, measured peak memory, CPU affinity, or statistical confidence
intervals. Run on an otherwise idle machine for meaningful cross-machine results.

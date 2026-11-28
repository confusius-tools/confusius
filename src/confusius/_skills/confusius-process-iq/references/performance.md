# Efficient IQ processing with Dask

Start with a **bounded threaded scheduler** and one BLAS thread per task. Benchmark
complete representative windows before increasing concurrency: IQ processing uses full
spatial blocks plus intermediate arrays, and concurrent SVD windows multiply memory.

## Local execution

```python
import dask
from threadpoolctl import threadpool_limits

import confusius as cf

with threadpool_limits(limits=1, user_api="blas"):
    with dask.config.set(scheduler="threads", num_workers=2):
        iq = cf.load("recording.zarr")
        pwd = cf.iq.process_iq_to_power_doppler(
            iq,
            clutter_window_width=300,
            low_cutoff=40,
        )
        cf.save(pwd, "power_doppler.zarr")
```

Window/cutoff values are illustrative. See [temporal windows](temporal-windows.md)
and [clutter filtering](clutter-filtering.md) for scientific parameter selection.

- **Memory/throughput:** measure peak RAM and total runtime, including graph creation;
  increase `num_workers` only when beneficial. Do not assume all cores fit in memory.
- **Batching:** custom `process_iq_blocks` calls can set `windows_per_task` without
  changing scientific windows. The overlap minimum still applies. Larger batches
  may reduce overhead, but can increase RAM and leave too few tasks for workers;
  built-in wrappers retain their minimum-batch default.
- **Large outputs:** keep input/output Dask-backed and save directly to Zarr.
- **Remote data:** materialize only a small pilot while tuning; consider a local Zarr
  copy for repeated processing. Rechunking does not rewrite stored chunk sizes.
- **Multiple outputs:** `dask.compute(bmode, pwd, velocity)` may share input tasks, not
  necessarily filters. Decibel scaling also triggers a global maximum.

IQ processing graphs are constructed without reading IQ samples or running filtering
windows. A supplied clutter mask is still materialized during validation. Keep
scheduler/BLAS settings active during compute/save; graph construction can stay inside
the same context for clarity.

## Distributed execution

Use `dask.distributed.Client` for multiple machines, worker-level memory/resource
controls, or its dashboard. A local Client is not inherently faster; compare measured
throughput, memory, and transfer overhead on representative data.

Set worker count, `threads_per_worker`, and `memory_limit` deliberately. Dask's
nanny-managed process workers normally set `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
and `OPENBLAS_NUM_THREADS` to `1` before startup, so the default process-based
`Client()` setup usually needs no additional BLAS limits. Existing environment
variables can override these defaults; check worker settings for overrides or custom
deployments. Connecting a client to an existing cluster does not configure its workers.
Monitor spilling/transfers and close the client when finished.

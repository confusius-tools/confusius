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
- **Large outputs:** keep input/output Dask-backed and save directly to Zarr.
- **Remote data:** materialize only a small pilot while tuning; consider a local Zarr
  copy for repeated processing. Rechunking does not rewrite stored chunk sizes.
- **Multiple outputs:** `dask.compute(bmode, pwd, velocity)` may share input tasks, not
  necessarily filters or eager probes. Decibel scaling also triggers a global maximum.

Keep scheduler/BLAS settings active during graph creation **and** compute/save:
ConfUSIus eagerly processes initial windows to infer output shape/dtype.

## Distributed execution

Use `dask.distributed.Client` for multiple machines, worker-level memory/resource
controls, or its dashboard. A local Client is not inherently faster; compare measured
throughput, memory, and transfer overhead on representative data.

Set worker count, `threads_per_worker`, and `memory_limit` deliberately. Limit BLAS
inside worker processes (environment before startup or worker initialization); the
parent's `threadpool_limits` alone is insufficient. Keep parent limits for eager
probes too. Monitor spilling/transfers and close the client when finished.

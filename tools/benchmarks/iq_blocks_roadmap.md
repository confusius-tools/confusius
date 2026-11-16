# Lazy IQ processing roadmap

Branch: `feat/lazy-iq-blocks`, based on `feat/sample-beamformed-iq` (parent PR #576).
Stacked PR: [#586](https://github.com/confusius-tools/confusius/pull/586), targeting
that parent branch, not `main`. Implementation, regression checks, and the batch
sweep are complete; see [results and recommendations](iq_blocks_report.md).

## 1. Establish deterministic output contracts

- Preserve input precision for B-mode and statically identifiable no-op filters.
- Promote potentially active filters consistently, including adaptive windows that
  select no clutter components; do not narrow wider input dtypes.
- Require per-window `output_shape` and `output_dtype` in `process_iq_blocks`.
- Remove real-data metadata probes; validate declared results inside worker tasks.
- Supply declarations from power Doppler, B-mode, velocity, and threshold helpers.
- Keep wrapper/accessor signatures, scientific windows, geometry, and timestamps.

## 2. Prove laziness and compatibility

- Verify no IQ source tasks or callback calls occur during graph construction.
- Compare numerical outputs and execution counts with independent window loops.
- Cover dtype/shape mismatches, adaptive dtype consistency, nested outputs,
  added/dropped axes, partial batches, and invalid windows.
- Preserve early validation that does not require IQ samples, including empty masks.
- Update API docs, custom-processing skill, and development changelog.
- Run focused tests, typing/lint/docstring checks, and broader relevant tests.

## 3. Benchmark batching separately

- Retain current default/minimum batching and scientific window semantics.
- Sweep requested batch sizes 1, 2, 4, 8, 16, and 32, subject to halo constraints.
- Exercise B-mode, power Doppler, axial velocity, and cumulative-energy threshold,
  including disjoint, overlapping, and strongly overlapping windows.
- Record construction/execution timings, graph sizes, effective batch sizes,
  output correctness, and representative memory use.
- Compare aligned/misaligned source chunks and a larger workload.
- Add only the low-level opt-in batching control needed to exercise actual
  production orchestration; do not add wrapper options or change default batching
  based on a synthetic benchmark.
- Publish reproducible results and recommendations, not a universal batch-size claim.

## 4. Ship the stacked PR

- Review the diff against `feat/sample-beamformed-iq` and run final quality gates.
- Push the new branch and use `gh pr create --base feat/sample-beamformed-iq`.
- Add the resulting PR link to changelog entries and push the final update.

## Deliberately out of scope

No scheduler rewrite, automatic tuning policy, native B-mode replacement,
scientific filtering changes, spatially distributed SVD, or new dependency.
Real scanner/remote-storage validation remains a follow-up unless suitable data
are already available locally.

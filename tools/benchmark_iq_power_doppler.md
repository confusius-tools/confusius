# Reconstruction-free power Doppler experiment

## Conclusion

Yes: full-window power Doppler can be computed without reconstructing filtered IQ.
On this CPU, the prototype was **1.37–1.65× faster end-to-end** than the current
public block pipeline in the 100,000-voxel tests when subtracting removed-component
energy with double-precision arithmetic. When retaining only 40 of 300 components,
direct retained-energy accumulation was **2.46× faster**.

These are measured CPU results, not a GPU prediction or a general speed guarantee.
The mathematical identity is exact; energy subtraction is **not equally stable**
as reconstruction or direct retained-energy accumulation.

No production code or API was changed.

## Reproduce

```bash
uv run python tools/benchmark_iq_power_doppler.py \
  --threads 4 --repeats 5 --output /tmp/pd-4threads.json
uv run python tools/benchmark_iq_power_doppler.py \
  --threads 1 --repeats 3 --sizes 10000 --output /tmp/pd-1thread.json
```

`tools/benchmark_iq_power_doppler_results.json` contains both completed runs,
including individual timing samples, BLAS versions/thread counts, and precision
metrics. The executable includes correctness assertions for the timing cases,
the subset-estimated basis, and the short-window quadratic form.

Hardware: Intel Core Ultra 9 275HX, Linux, OpenBLAS. Threads are limited through
`threadpoolctl` for both NumPy and SciPy. Each operation has one untimed warmup;
tables show medians of five repetitions for four threads. Data generation,
independent reference SVD, and correctness checks are outside timing. All array
allocations, dtype conversions, Gram construction, and EVD are included in the
end-to-end candidate measurements. No disk I/O is timed.

## What the existing implementation actually does

- `clutter_filter_svd_from_indices` already computes a temporal Gram EVD, casting
  the basis-estimation data to `complex128`, even for `complex64` input.
- It regresses out the **removed** vectors, using two matrix multiplications:
  `signals - removed @ (removed.conj().T @ signals)`.
- `compute_power_doppler_volume` then selects integration windows and averages
  squared magnitudes. Its full-window selection also makes an advanced-indexing
  copy of the filtered block.

Therefore, comparing against a full SVD or reconstructing every retained component
would overstate the benefit relative to ConfUSIus. The benchmark uses the actual
public `compute_power_doppler_volume` as its primary baseline, and also measures
an equivalent bare reconstruction kernel to separate wrapper/copy overhead.
The candidate explicitly reproduces the production Gram EVD rather than importing
private functions.

With the repository's `(time, voxels)` orientation, and descending temporal
vectors `Q`, the two candidates are:

```python
# Retained components.
a = Q[:, low:high].conj().T @ signals
pd = np.sum(np.abs(a) ** 2, axis=0) / n_frames

# Removed components (both low-index tissue and any high-index rejection).
q_removed = np.concatenate((Q[:, :low], Q[:, high:]), axis=1)
a_removed = q_removed.conj().T @ signals
energy = np.einsum("tn,tn->n", signals.real, signals.real, dtype=np.float64)
energy += np.einsum("tn,tn->n", signals.imag, signals.imag, dtype=np.float64)
pd = (energy - np.sum(np.abs(a_removed) ** 2, axis=0)) / n_frames
```

The two `einsum` reductions multiply **and** accumulate in double precision without
allocating a full double-precision copy of IQ. Merely specifying a double-precision
sum after `np.abs(complex64_iq) ** 2` would leave single-precision magnitude/squaring
errors in place. A preliminary implementation with a full cast before squaring
was more memory-intensive and had less consistent large-array speedups; the saved
results and tables use the reductions above.

## Synthetic data

Timing uses seeded `complex64` IQ with 300 frames and 10,000 or 100,000 voxels.
The signal is a rank-20 coherent complex tissue term plus independent complex
Gaussian blood/noise, with expected tissue-to-blood/noise power ratio 40 dB.
Temporal tissue vectors are orthonormalized random vectors; spatial coefficients
are independent complex Gaussian variables. This is an algebra/performance test,
not a physiological vascular or motion simulator.

Cases retain `[20, 300)` or `[20, 60)` in descending energy order. A third case
estimates the basis using every fourth voxel, but filters/evaluates all voxels.
The estimation subset always has at least as many voxels as frames. We do not test
the underdetermined-mask case or extrapolate the retained-only formula to the
production API's component-limit semantics in that case.

## Speed: full pipeline

All entries below are milliseconds, four BLAS threads. Both candidate columns
include Gram construction and full EVD, not just processing with a cached basis.

| Voxels | Retain | Basis voxels | Current public API | Direct retained energy | Double energy subtraction | Best speedup |
|---:|:---:|---:|---:|---:|---:|---:|
| 10,000 | `[20,300)` | 100% | 166.0 | 165.1 | **113.0** | 1.47× |
| 10,000 | `[20,60)` | 100% | 237.9 | **114.2** | 175.2 | 2.08× |
| 10,000 | `[20,300)` | 25% | 99.0 | 115.8 | **60.0** | 1.65× |
| 100,000 | `[20,300)` | 100% | 1373.2 | 1397.9 | **1004.8** | 1.37× |
| 100,000 | `[20,60)` | 100% | 2236.3 | **909.7** | 1573.2 | 2.46× |
| 100,000 | `[20,300)` | 25% | 777.9 | 1003.9 | **473.2** | 1.64× |

With one thread and 10,000 voxels, the best corresponding speedups were 1.12×,
2.37×, and 1.48×. The benefit is sensitive to BLAS threading and system load.
Early large-array runs showed substantial timing variability, so do not interpret
small differences as statistically significant. These sequential microbenchmarks
are not randomized, CPU-pinned, or run on an otherwise isolated machine.

### Isolating reconstruction savings

For 100,000 voxels, **after the same basis has already been computed**:

| Retain | Bare reconstruction + integration | Direct retained energy | Double subtraction |
|:---:|---:|---:|---:|
| `[20,300)` | 488.9 ms | 752.1 ms | **228.2 ms** |
| `[20,60)` | 1374.8 ms | **188.3 ms** | 805.8 ms |

Removing 20 components: subtraction speeds up the post-basis stage 2.14×,
but Gram/EVD remains a major cost, limiting end-to-end gains. Retaining 280
components instead is not a useful default: its projection is much larger than
the current rank-20 removal projection. Retaining 40 components is the opposite
case, making direct accumulation substantially better.

At 100,000 voxels × 300 frames, one reconstructed `complex128` IQ array occupies
480 MB (458 MiB). Subtraction avoids that array, with a 20-component coefficient
array occupying 32 MB instead. These are array-size calculations, **not measured
peak RSS**; Gram construction and BLAS still allocate working memory, and direct
retained accumulation can itself require a large coefficient array.

## Precision

A separate experiment uses 1,024 voxels, 64 frames, rank-4 tissue, and expected
tissue-to-blood/noise ratios of 0, 40, 80, and 120 dB. Both `complex64` and
`complex128` storage are tested, with double-precision Gram EVD in every case.

Reference: a direct double-precision SVD of the **same stored input**, followed by
rank-4 removal and explicit reconstruction. This avoids assuming that Gram EVD is
itself an exact reference. Errors are also recorded against reconstruction from
the identical Gram basis to isolate the power-estimation identity.

Relative L2 image error against direct SVD for `complex64` input:

| Tissue/blood power ratio | Current-style Gram reconstruction | Direct retained energy | Double subtraction | Native float32 energy subtraction |
|---:|---:|---:|---:|---:|
| 0 dB | 3.40e-16 | 1.60e-15 | 1.04e-15 | 2.59e-7 |
| 40 dB | 1.69e-14 | 1.73e-14 | 4.04e-11 | 1.35e-3 |
| 80 dB | 1.46e-12 | 1.47e-12 | 1.73e-7 | 1.29e1 |
| 120 dB | 1.70e-10 | 1.62e-10 | 5.52e-3 | 1.36e5 |

- At 80 dB, native-energy subtraction produced **472 negative voxels out of
  1,024**. At 120 dB, 509 were negative. The projection still used double-precision
  eigenvectors: that alone is insufficient to make subtraction safe.
- Even double subtraction lost precision at 120 dB: 0.55% relative L2 error and
  2.73% maximum pointwise relative error for `complex64` input.
- `complex128` storage did not eliminate cancellation: double subtraction had
  0.40% relative L2 error at 120 dB, versus about 1.5e-10 for reconstruction and
  retained accumulation. At 80 dB, double subtraction's relative L2 error was
  1.06e-6. Finite-precision basis orthogonality contributes to this sensitivity.
- For the 40 dB timing cases, direct retained accumulation agreed with the actual
  public API within roughly 1e-13 maximum pointwise relative error. Double
  subtraction stayed within roughly 1e-8, without negative values.

These errors measure numerical equivalence, not biological recovery accuracy.
Quantization already lost when storing the original IQ in `complex64` cannot be
recovered by any candidate; the reference intentionally uses that same input.
Clipping negative powers to zero would hide failures, not solve cancellation.

## Window and subset checks

A separate 60-frame test estimates its basis from 25% of voxels, retains 56
components, and checks power over the first 10 frames:

- The correct window-dependent quadratic form agrees with explicit reconstruction
  to relative L2 error 3.45e-16.
- Incorrectly replacing short-window power with full-window component energy has
  relative L2 error **0.276** (27.6%); maximum pointwise relative error is 228%.
- Full-window subtraction and retained accumulation agree for the subset basis.

Thus the simple energy identity is suitable only when the integration spans the
entire decomposition window with uniform weights. A subset-estimated basis does
not invalidate it; changing the time window or weighting does. The quadratic form
was correctness-checked, **not speed-benchmarked**.

## Recommendation

1. A fused SVD-to-power path is worthwhile for full-window, uniformly integrated
   output. Keep IQ reconstruction for temporal outputs and shorter windows.
2. When few components are retained, direct retained-energy accumulation is both
   fast and numerically robust in these tests.
3. When few components are removed, double-precision energy subtraction can be
   faster and avoid a full reconstructed IQ array, but it should not silently
   replace the robust existing calculation for arbitrary dynamic range.
4. An eventual implementation needs a documented precision policy or a robust
   fallback; choosing solely by retained/removed component counts is insufficient.
   A safe cancellation threshold has not been established by this experiment.

Not checked: real recordings, GPU execution, distributed/Dask scheduling, adaptive
cutoff performance, weighted-window speed, or measured peak memory. No production
optimization, automatic precision policy, or new user-facing options were added.

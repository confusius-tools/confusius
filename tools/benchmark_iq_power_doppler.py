"""Benchmark reconstruction-free power Doppler against the public IQ block API.

Run with `uv run python tools/benchmark_iq_power_doppler.py --output /tmp/pd.json`.
This is an experiment, not a supported filtering implementation. All timings are CPU
wall times; synthetic generation, reference SVD, and correctness checks are untimed.
"""

from __future__ import annotations

import argparse
import json
import platform
from collections.abc import Callable
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import numpy.typing as npt
from scipy import linalg
from threadpoolctl import threadpool_info, threadpool_limits

from confusius.iq import compute_power_doppler_volume


def make_signals(n: int, t: int, rank: int, db: float, dtype: str) -> npt.NDArray:
    """Generate low-rank coherent tissue plus independent complex blood/noise.

    Parameters
    ----------
    n, t, rank : int
        Voxel count, frame count, and tissue rank.
    db : float
        Expected tissue-to-blood/noise power ratio in decibels.
    dtype : str
        Storage dtype, applied after generating the mixture in double precision.

    Returns
    -------
    numpy.ndarray
        C-contiguous `(time, voxels)` IQ matrix, seeded reproducibly.
    """
    rng = np.random.default_rng(42)
    basis, _ = np.linalg.qr(
        rng.standard_normal((t, rank)) + 1j * rng.standard_normal((t, rank))
    )
    coefficients = (
        rng.standard_normal((rank, n)) + 1j * rng.standard_normal((rank, n))
    ) * np.sqrt(t / (2 * rank) * 10 ** (db / 10))
    signals = basis @ coefficients
    signals += (
        rng.standard_normal((t, n)) + 1j * rng.standard_normal((t, n))
    ) / np.sqrt(2)
    return np.asarray(signals, dtype=dtype, order="C")


def get_basis(signals: npt.NDArray, subset: npt.NDArray | None) -> npt.NDArray:
    """Reproduce the production double-precision temporal Gram decomposition.

    Parameters
    ----------
    signals : numpy.ndarray
        `(time, voxels)` IQ matrix.
    subset : numpy.ndarray, optional
        Boolean voxel selection for basis estimation only.

    Returns
    -------
    numpy.ndarray
        Temporal eigenvectors, ordered by decreasing eigenvalue.
    """
    masked = signals if subset is None else signals[:, subset]
    masked = masked.astype(np.complex128, copy=False)
    _, vectors = linalg.eigh(masked @ masked.conj().T)
    return vectors[:, ::-1]


def power(
    signals: npt.NDArray,
    vectors: npt.NDArray | None,
    low: int,
    high: int,
    method: str,
    subset: npt.NDArray | None = None,
) -> npt.NDArray:
    """Evaluate full-window power using a fixed temporal basis.

    Parameters
    ----------
    signals : numpy.ndarray
        IQ signals.
    vectors : numpy.ndarray, optional
        Descending temporal eigenvectors. If not provided, compute from signals.
    low, high : int
        Retained half-open component interval.
    method : str
        `reconstruct`, `retained`, `subtract64`, or `subtract_native`.
    subset : numpy.ndarray, optional
        Boolean voxel selection for basis estimation only.

    Returns
    -------
    numpy.ndarray
        Per-voxel power. Negative subtraction results are deliberately not clipped.
    """
    if vectors is None:
        vectors = get_basis(signals, subset)
    t = signals.shape[0]
    if method == "retained":
        coefficients = vectors[:, low:high].conj().T @ signals
        return np.sum(np.abs(coefficients) ** 2, axis=0) / t
    removed = np.concatenate((vectors[:, :low], vectors[:, high:]), axis=1)
    coefficients = removed.conj().T @ signals
    if method == "reconstruct":
        filtered = signals - removed @ coefficients
        return np.mean(np.abs(filtered) ** 2, axis=0)
    # Multiply and accumulate in double precision without copying the full IQ array.
    if method == "subtract64":
        total = np.einsum("tn,tn->n", signals.real, signals.real, dtype=np.float64)
        total += np.einsum("tn,tn->n", signals.imag, signals.imag, dtype=np.float64)
    else:
        total = np.sum(np.abs(signals) ** 2, axis=0)
    return (total - np.sum(np.abs(coefficients) ** 2, axis=0)) / t


def measure(func: Callable[[], npt.NDArray], repeats: int) -> dict:
    """Warm up and measure repeated wall-clock durations.

    Parameters
    ----------
    func : callable
        Operation to benchmark, including all allocations.
    repeats : int
        Number of timed calls after one untimed warmup.

    Returns
    -------
    dict
        Median and all samples in milliseconds.
    """
    func()
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        func()
        samples.append(1000 * (perf_counter() - start))
    return {"median_ms": float(np.median(samples)), "samples_ms": samples}


def errors(actual: npt.NDArray, reference: npt.NDArray) -> dict:
    """Summarize power error without hiding negative values or outliers.

    Parameters
    ----------
    actual, reference : numpy.ndarray
        Candidate and reference per-voxel power.

    Returns
    -------
    dict
        Relative L2 error, maximum pointwise relative error, and negative count.
    """
    relative = np.abs(actual - reference) / np.maximum(reference, np.finfo(float).tiny)
    return {
        "relative_l2": float(
            np.linalg.norm(actual - reference) / np.linalg.norm(reference)
        ),
        "max_relative": float(relative.max()),
        "negative_voxels": int(np.count_nonzero(actual < 0)),
    }


def run(repeats: int, sizes: list[int]) -> dict:
    """Run timing, cancellation, masking, and window-identity experiments.

    Parameters
    ----------
    repeats : int
        Timed repetitions per method.
    sizes : list of int
        Voxel counts for the timing experiment (300 frames).

    Returns
    -------
    dict
        Machine metadata and all measured results.
    """
    methods = ("reconstruct", "retained", "subtract64", "subtract_native")
    result = {
        "platform": platform.platform(),
        "numpy": np.__version__,
        "threadpools": threadpool_info(),
        "timing": [],
        "precision": [],
    }
    # Cover low-rank tissue removal, narrow retention, and subset-estimated bases.
    for n in sizes:
        signals = make_signals(n, 300, 20, 40, "complex64")
        for low, high, masked in ((20, 300, False), (20, 60, False), (20, 300, True)):
            subset = np.arange(n) % 4 == 0 if masked else None
            block = signals.reshape(300, 1, 1, n)
            mask = subset.reshape(1, 1, n) if masked else None
            vectors = get_basis(signals, subset)
            public = partial(
                compute_power_doppler_volume,
                block,
                low_cutoff=low,
                high_cutoff=high,
                clutter_mask=mask,
            )
            reference = public().ravel()
            row = {
                "voxels": n,
                "frames": 300,
                "low": low,
                "high": high,
                "subset_fraction": 0.25 if masked else 1.0,
                "public_pipeline": measure(public, repeats),
                "gram_evd": measure(partial(get_basis, signals, subset), repeats),
                "post_basis": {},
                "end_to_end": {},
                "errors": {},
            }
            for method in methods:
                row["post_basis"][method] = measure(
                    partial(power, signals, vectors, low, high, method), repeats
                )
                row["errors"][method] = errors(
                    power(signals, vectors, low, high, method), reference
                )
                if method != "subtract_native":
                    row["end_to_end"][method] = measure(
                        partial(power, signals, None, low, high, method, subset),
                        repeats,
                    )
                    np.testing.assert_allclose(
                        power(signals, vectors, low, high, method),
                        reference,
                        rtol=1e-8,
                        atol=1e-9,
                    )
            result["timing"].append(row)
            print(
                f"timing n={n}, keep=[{low},{high}), masked={masked}: "
                f"public={row['public_pipeline']['median_ms']:.1f} ms",
                flush=True,
            )
    # Independent direct-SVD reference avoids conflating cancellation with Gram errors.
    for dtype in ("complex64", "complex128"):
        for db in (0, 40, 80, 120):
            signals = make_signals(1024, 64, 4, db, dtype)
            vectors = get_basis(signals, None)
            svd_vectors, _, _ = linalg.svd(
                signals.astype(np.complex128), full_matrices=False
            )
            reference = power(signals, svd_vectors, 4, 64, "reconstruct")
            gram_reference = power(signals, vectors, 4, 64, "reconstruct")
            row = {"dtype": dtype, "tissue_to_blood_db": db, "methods": {}}
            for method in methods:
                value = power(signals, vectors, 4, 64, method)
                row["methods"][method] = {
                    "vs_direct_svd": errors(value, reference),
                    "vs_same_basis_reconstruction": errors(value, gram_reference),
                }
            result["precision"].append(row)
    # Full-window identity does not hold on arbitrary subwindows.
    signals = make_signals(256, 60, 4, 20, "complex128")
    vectors = get_basis(signals, np.arange(256) % 4 == 0)
    retained = vectors[:, 4:]
    coefficients = retained.conj().T @ signals
    filtered = retained @ coefficients
    reference = np.mean(np.abs(filtered[:10]) ** 2, axis=0)
    metric = retained[:10].conj().T @ retained[:10] / 10
    quadratic = np.sum(coefficients.conj() * (metric @ coefficients), axis=0).real
    np.testing.assert_allclose(quadratic, reference, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        power(signals, vectors, 4, 60, "subtract64"),
        power(signals, vectors, 4, 60, "retained"),
        rtol=1e-10,
        atol=1e-10,
    )
    result["short_window"] = {
        "quadratic_form": errors(quadratic, reference),
        "incorrect_full_window_identity": errors(
            np.sum(np.abs(coefficients) ** 2, axis=0) / 60, reference
        ),
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--sizes", type=int, nargs="+", default=[10_000, 100_000])
    args = parser.parse_args()
    if args.threads < 1 or args.repeats < 1 or min(args.sizes) < 1200:
        parser.error("threads/repeats must be positive; sizes must be >= 1200")
    with threadpool_limits(limits=args.threads):
        results = run(args.repeats, args.sizes)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Results written to {args.output}")

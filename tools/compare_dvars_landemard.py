"""Compare three DVARS baselines on a cached recording.

Defaults to Landemard; use --source, --output, and --title for another recording.
Run with the repository's src directory on PYTHONPATH. Outputs are exploratory,
not a validation of artifact-detection thresholds for fUSI.
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import xarray as xr
from scipy.stats import iqr

import confusius as cf
from confusius.glm._utils import estimate_ar_coeffs
from confusius.qc import compute_dvars

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path.home() / ".cache/confusius/landemard-2026-bids/1.0.0"
SOURCE = ROOT / "sub-ALD019/fusi/sub-ALD019_task-awake_acq-ref04_run-1_pwd.nii.gz"
OUTPUT = Path("analysis/dvars-landemard")


def main() -> None:
    """Compute baselines, verify agreement with the API, and save results.

    Returns
    -------
    None
        Writes JSON, CSV, and a comparison plot.

    Raises
    ------
    AssertionError
        If the reference calculation disagrees with the library or invariants fail.
    ValueError
        If signals or estimated denominators are invalid.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--title", default="Landemard ALD019 ref04 run-1")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = cf.load(args.source).compute()
    signals = data.transpose("time", ...).values.reshape(data.sizes["time"], -1)
    signals = signals.astype(np.float64)
    finite = np.isfinite(signals).all(axis=0)
    sd = np.zeros(signals.shape[1])
    sd[finite] = iqr(signals[:, finite], axis=0, scale="normal", interpolation="lower")
    valid = finite & (sd > 0)
    if not valid.any():
        raise ValueError("No finite signals with positive robust variance.")
    intensity = np.median(signals[:, valid], axis=0)
    bright = np.zeros_like(valid)
    bright[valid] = intensity >= np.median(intensity)
    masks = {"all_valid": valid, "bright_half": bright}
    report = {
        "source": str(args.source),
        "confusius_source": cf.__file__,
        "dims": list(data.dims),
        "shape": list(data.shape),
        "frames": data.sizes["time"],
        "time_units": data.time.attrs.get("units"),
        "time_start_end": data.time.values[[0, -1]].tolist(),
        "median_time_step": float(np.median(np.diff(data.time.values))),
        "arithmetic_dtype": str(signals.dtype),
        "spatial_voxels": signals.shape[1],
        "nonfinite_voxels": int((~finite).sum()),
        "finite_zero_iqr_voxels": int((finite & ~valid).sum()),
        "preprocessing": "No detrending, filtering, spatial resampling, or intensity scaling.",
        "mask_note": "No anatomical brain mask; bright_half selects the upper half of temporal median intensity among valid voxels.",
        "masks": {},
    }
    fig, axes = plt.subplots(2, 2, figsize=(14, 7), constrained_layout=True)
    for row, (name, mask) in enumerate(masks.items()):
        values = signals[:, mask]
        rho, _ = estimate_ar_coeffs(values, order=1)
        diff_sd = np.sqrt(2 * (1 - rho.squeeze())) * sd[mask]
        raw = np.sqrt(np.mean(np.diff(values, axis=0) ** 2, axis=1))
        denominators = {
            "current_mean_sd": float(np.mean(diff_sd)),
            "ar1_rms_sd": float(np.sqrt(np.mean(diff_sd**2))),
            "median_squared": float(np.sqrt(np.median(raw**2))),
        }
        if not all(np.isfinite(d) and d > 0 for d in denominators.values()):
            raise ValueError("Nonfinite or nonpositive DVARS denominator.")
        traces = {key: raw / denominator for key, denominator in denominators.items()}
        table = xr.DataArray(values, dims=("time", "space"), coords={"time": data.time})
        reference = compute_dvars(table, baseline_estimator="legacy").values[1:]
        np.testing.assert_allclose(traces["current_mean_sd"], reference, rtol=1e-12)
        for estimator, label in (("ar1", "ar1_rms_sd"), ("median", "median_squared")):
            np.testing.assert_allclose(
                compute_dvars(table, baseline_estimator=estimator).values[1:],
                traces[label],
                rtol=1e-12,
            )
        np.testing.assert_allclose(
            compute_dvars(table, variant="raw", normalization_factor=None).values[1:],
            raw,
            rtol=1e-12,
        )
        assert denominators["ar1_rms_sd"] >= denominators["current_mean_sd"]
        np.testing.assert_allclose(np.median(traces["median_squared"] ** 2), 1)
        summaries = {}
        for label, trace in traces.items():
            np.testing.assert_array_equal(np.argsort(trace), np.argsort(raw))
            summaries[label] = {
                "denominator": denominators[label],
                "quantiles_min_p50_p95_p99_max": np.quantile(
                    trace, [0, 0.5, 0.95, 0.99, 1]
                ).tolist(),
                "frames_above_threshold": {
                    str(t): int((trace > t).sum()) for t in [1.5, 2, 2.5, 3]
                },
            }
            axes[row, 0].plot(
                data.time.values[1:], trace, label=label, alpha=0.8, lw=0.7
            )
        report["masks"][name] = {
            "voxels": int(mask.sum()),
            "valid_differences": len(raw),
            "diff_sd_coefficient_of_variation": float(
                np.std(diff_sd) / np.mean(diff_sd)
            ),
            "current_over_ar1_rms": denominators["ar1_rms_sd"]
            / denominators["current_mean_sd"],
            "current_over_median_squared": denominators["median_squared"]
            / denominators["current_mean_sd"],
            "methods": summaries,
            "largest_difference_frame_indices": (
                np.argsort(raw)[-10:][::-1] + 1
            ).tolist(),
        }
        np.savetxt(
            args.output / f"{name}.csv",
            np.column_stack(
                [
                    np.arange(1, len(raw) + 1),
                    data.time.values[1:],
                    raw,
                    *traces.values(),
                ]
            ),
            delimiter=",",
            header="frame,time,raw," + ",".join(traces),
            comments="",
        )
        axes[row, 0].set(
            title=f"{name}: {mask.sum():,} voxels",
            xlabel="Time (s)",
            ylabel="Normalized DVARS",
        )
        axes[row, 0].axhline(2.5, color="gray", ls="--", lw=0.8)
        axes[row, 0].legend(fontsize=8)
        axes[row, 1].hist(diff_sd, bins=80)
        for index, (label, denominator) in enumerate(denominators.items()):
            axes[row, 1].axvline(denominator, label=label, color=f"C{index}", ls="--")
        axes[row, 1].set(
            xlabel="Estimated difference SD / baseline (native intensity units)",
            ylabel="Voxels",
        )
        axes[row, 1].legend(fontsize=8)
    fig.suptitle(f"{args.title} — first synthetic DVARS sample excluded")
    fig.savefig(args.output / "comparison.png", dpi=160)
    plt.close(fig)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

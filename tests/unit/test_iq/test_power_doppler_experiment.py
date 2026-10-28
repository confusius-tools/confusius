"""End-to-end checks for the standalone Dask power Doppler experiment."""

import argparse
import json
import subprocess
import sys

import numpy as np
import pytest
from rich.progress import Progress

from confusius.iq import process_iq_to_power_doppler as process_reference
from confusius.validation import validate_voxeldata
from tools.benchmark_iq_power_doppler_dask import benchmark_scheduler, print_results
from tools.iq_to_power_doppler import process_iq_to_power_doppler


def test_results_table(capsys, monkeypatch):
    """Show measured medians and the worst observed error, rather than the last sample."""
    monkeypatch.setenv("COLUMNS", "180")
    print_results(
        [
            {
                "scheduler": "threads",
                "summary": {
                    "subtract": {
                        "build_s": 0.123,
                        "compute_s": 0.456,
                        "total_s": 0.579,
                        "speedup_vs_current": 1.42,
                    }
                },
                "samples": {
                    "subtract": [
                        {
                            "error_vs_current": {
                                "relative_l2": 0.025,
                                "negative_voxels": 7,
                            }
                        },
                        {
                            "error_vs_current": {
                                "relative_l2": 0.001,
                                "negative_voxels": 0,
                            }
                        },
                    ]
                },
            }
        ]
    )
    rendered = capsys.readouterr().out
    for value in (
        "threads",
        "subtract",
        "0.123",
        "0.456",
        "0.579",
        "1.42x",
        "2.50e-02",
        "7",
    ):
        assert value in rendered


@pytest.mark.parametrize("method", ["subtract", "retained"])
@pytest.mark.parametrize("cutoffs", [(0, None), (2, None), (2, 5)])
def test_voxeldata_matches_reference(
    sample_voxeldata_iq_dataarray, sample_spatial_mask_xarray, method, cutoffs
):
    """Preserve power, time metadata, and geometry with masked overlapping windows."""
    iq = sample_voxeldata_iq_dataarray.chunk({"time": 5, "k": 2, "j": 3, "i": 4})
    options = {
        "clutter_window_width": 8,
        "clutter_window_stride": 4,
        "clutter_mask": sample_spatial_mask_xarray,
        "low_cutoff": cutoffs[0],
        "high_cutoff": cutoffs[1],
    }
    reference = process_reference(iq, **options).compute()
    actual = process_iq_to_power_doppler(iq, method=method, **options).compute()
    validate_voxeldata(actual, require_time=True)
    np.testing.assert_allclose(actual.values, reference.values, rtol=1e-11, atol=1e-12)
    assert actual.time.identical(reference.time)
    for name in ("k", "j", "i", "z", "y", "x"):
        np.testing.assert_array_equal(actual.coords[name], reference.coords[name])


def test_progress_counts_warmups_and_samples(capsys):
    """Advance for every warmup and sample without printing per-sample timings."""
    args = argparse.Namespace(
        frames=8,
        stride=8,
        low=2,
        high=None,
        workers=1,
        threads_per_worker=1,
        blas_threads=1,
        windows=2,
        voxels=128,
        tissue_rank=2,
        db=40,
        dtype="complex64",
        time_chunk=8,
        repeats=2,
    )
    with Progress(disable=True, auto_refresh=False) as progress:
        task = progress.add_task("test", total=9)
        run = benchmark_scheduler(args, "threads", progress, task)
        assert progress.tasks[task].completed == 9
        assert progress.tasks[task].finished
    assert all(len(samples) == 2 for samples in run["samples"].values())
    assert "repeat=" not in capsys.readouterr().out


def test_benchmark_all_schedulers(tmp_path):
    """Exercise CLI, worker spawning, overlap, rechunking, precision, and JSON output."""
    output = tmp_path / "results.json"
    schedulers = ["threads", "distributed-threads", "distributed-processes"]
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.benchmark_iq_power_doppler_dask",
            "--output",
            str(output),
            "--schedulers",
            *schedulers,
            "--voxels",
            "128",
            "--frames",
            "8",
            "--windows",
            "3",
            "--stride",
            "4",
            "--time-chunk",
            "3",
            "--low",
            "2",
            "--high",
            "5",
            "--tissue-rank",
            "2",
            "--repeats",
            "1",
            "--worker-memory",
            "1GiB",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert "Power Doppler results" in completed.stdout
    assert "repeat=" not in completed.stdout
    assert "build=" not in completed.stdout
    result = json.loads(output.read_text())
    assert [run["scheduler"] for run in result["runs"]] == schedulers
    for run in result["runs"]:
        assert run["input_shape"] == [16, 1, 1, 128]
        for method in ("current", "subtract", "retained"):
            sample = run["samples"][method][0]
            assert sample["error_vs_current"]["relative_l2"] < 1e-7
            assert sample["error_vs_current"]["negative_voxels"] == 0
            assert sample["total_s"] >= sample["compute_s"]
        if run["workers"] is not None:
            assert len(run["workers"]) == 2
            for pools in run["workers"].values():
                assert pools
                assert all(pool["num_threads"] == 1 for pool in pools)

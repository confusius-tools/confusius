"""Reproduce issue #70 with the downloaded scan.

This script demonstrates the original coordinate-matching failure reported in
issue #70: extracting signals from a single selected mask can leave an
unrelated scalar `mask` coordinate attached to the output, which makes
`signals.coords["time"].equals(sample_mask.coords["time"])` return `False`
even though the time index itself matches.

Run with:

`uv run python scripts/repro_issue_70.py`
"""

from pathlib import Path

import numpy as np
import xarray as xr
from scipy.ndimage import uniform_filter1d
from scipy.stats import iqr

import confusius as cf

SCAN_PATH = Path("/home/sdiebolt/Downloads/4Dscan_4_RS16_20_fus3D.source.scan")
SCRUB_KERNEL = 15
DVARS_PERCENTILE = 75
DVARS_MULTIPLIER = 1.5

fusi = cf.io.load(SCAN_PATH)
fusi = cf.multipose.consolidate_poses(fusi)

brain = (fusi.isel(time=0) > np.percentile(fusi.isel(time=0).values, 75)).astype(int)
brain_voxel_signal = fusi.fusi.extract.with_mask(brain)

dvars = cf.qc.compute_dvars(brain_voxel_signal)
dvars = xr.DataArray(
    uniform_filter1d(dvars.values, SCRUB_KERNEL, mode="nearest"),
    dims=dvars.dims,
    coords=dvars.coords,
)
dvars_threshold = np.percentile(dvars, DVARS_PERCENTILE) + iqr(dvars) * DVARS_MULTIPLIER
sample_mask = xr.DataArray(
    dvars.values < dvars_threshold,
    dims=["time"],
    coords={"time": fusi.time},
)

mask_data = np.zeros((2, *fusi.shape[1:]), dtype=int)
mask_data[0, 0, :, :] = 1
mask_data[1, 1, :, :] = 2
labels = xr.DataArray(
    mask_data,
    dims=["mask", "z", "y", "x"],
    coords={"mask": ["A", "B"], "z": fusi.z, "y": fusi.y, "x": fusi.x},
)

single_mask = labels.isel(mask=0)
roi_signal_raw = fusi.fusi.extract.with_labels(single_mask)

print(f"Loaded scan: {SCAN_PATH}")
print(f"Signal coords: {list(roi_signal_raw.coords)}")
print(f"Time coord attached coords: {list(roi_signal_raw.coords['time'].coords)}")
print(
    "signals.coords['time'].equals(sample_mask.coords['time']):",
    roi_signal_raw.coords["time"].equals(sample_mask.coords["time"]),
)
print(
    "signals.indexes['time'].equals(sample_mask.indexes['time']):",
    roi_signal_raw.indexes["time"].equals(sample_mask.indexes["time"]),
)

try:
    result = cf.signal.clean(
        roi_signal_raw,
        standardize_method="zscore",
        detrend_order=None,
        low_cutoff=None,
        high_cutoff=None,
        confounds=None,
        sample_mask=sample_mask,
    )
except Exception as exc:
    print(f"clean(...) raised: {type(exc).__name__}: {exc}")
else:
    print("clean(...) succeeded.")
    print(f"Result shape: {result.shape}")

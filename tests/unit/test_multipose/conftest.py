"""Fixtures for multipose unit tests."""

import pytest
import xarray as xr

from confusius.multipose import consolidate_poses


@pytest.fixture
def consolidated_scan_4d(scan_4d: xr.DataArray) -> xr.DataArray:
    """Consolidated 4D scan with dims (time, z, y, x) and slice_time coordinate."""
    return consolidate_poses(scan_4d)

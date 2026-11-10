"""Tests for the lazy beamformed IQ sample."""

from unittest.mock import patch

import dask.array as da
import numpy.testing as npt
import xarray as xr

from confusius.datasets import fetch_sample_beamformed_iq
from confusius.io import save


def test_sample_is_loaded_lazily_with_geometry(tmp_path, sample_voxeldata_3d):
    """Resolve the store URL and preserve lazy data and geometry."""
    path = tmp_path / "input.zarr"
    save(sample_voxeldata_3d.rename("iq"), path)
    original_open = xr.open_zarr
    url = "https://example.org/sample.zarr"
    index = {
        "sub-ALD019_task-awake_acq-ref25_iq.zarr/zarr.json": {"url": url + "/zarr.json"}
    }
    with (
        patch("confusius.datasets._sample_beamformed_iq.get_index", return_value=index),
        patch(
            "xarray.open_zarr",
            side_effect=lambda path, **kw: original_open(tmp_path / "input.zarr", **kw),
        ) as opener,
    ):
        data = fetch_sample_beamformed_iq(print_citation=False)
    opener.assert_called_once_with(url, chunks={})
    assert isinstance(data.data, da.Array)
    npt.assert_allclose(data.compute().values, sample_voxeldata_3d.values)
    npt.assert_allclose(
        data.fusi.affine.voxel_to_world, sample_voxeldata_3d.fusi.affine.voxel_to_world
    )
    assert "Landemard" in data.attrs["citation"]

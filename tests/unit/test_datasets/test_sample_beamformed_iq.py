"""Tests for the lazy beamformed IQ sample."""

from unittest.mock import Mock, patch

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
        data = fetch_sample_beamformed_iq(data_dir=tmp_path, print_citation=False)
    opener.assert_called_once_with(url, chunks={})
    assert isinstance(data.data, da.Array)
    npt.assert_allclose(data.compute().values, sample_voxeldata_3d.values)
    npt.assert_allclose(
        data.fusi.affine.voxel_to_world, sample_voxeldata_3d.fusi.affine.voxel_to_world
    )
    assert "Landemard" in data.attrs["citation"]


def test_release_metadata_cached_until_refresh(tmp_path, sample_voxeldata_3d):
    """Reuse the release index without catalog requests unless refreshed."""
    catalog = Mock(text="[datasets]\nsample-beamformed-iq = 1.0.0\n")
    manifest = Mock()
    manifest.json.return_value = {
        "sub-ALD019_task-awake_acq-ref25_iq.zarr/zarr.json": {
            "size": 1,
            "sha256": "0" * 64,
        }
    }
    with (
        patch(
            "confusius.datasets._s3.requests.get",
            side_effect=[catalog, manifest, catalog, manifest],
        ) as request,
        patch(
            "confusius.datasets._sample_beamformed_iq.load",
            side_effect=lambda *args, **kwargs: sample_voxeldata_3d.copy(),
        ),
    ):
        fetch_sample_beamformed_iq(data_dir=tmp_path, print_citation=False)
        assert request.call_count == 2
        fetch_sample_beamformed_iq(data_dir=tmp_path, print_citation=False)
        assert request.call_count == 2
        fetch_sample_beamformed_iq(
            data_dir=tmp_path, refresh=True, print_citation=False
        )
        assert request.call_count == 4

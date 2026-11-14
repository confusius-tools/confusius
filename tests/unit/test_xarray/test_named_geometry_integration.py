"""General named geometry at ConfUSIus construction, timing, extraction, and I/O boundaries."""

import numpy as np
import pytest
import xarray as xr

from confusius import io
from confusius.validation import canonicalize_voxeldata, validate_voxeldata
from confusius.xarray import create_voxeldata


def test_constructor_forwards_ordered_geometry_dependencies(
    sample_named_geometry_voxeldata,
):
    """Geometry depends on time/view, not channel or the parent dimension order."""
    data = sample_named_geometry_voxeldata
    assert data.vd.geometry_dims == ("time", "view")
    assert data.z.dims == ("time", "view", "k", "j", "i")
    assert data.time.attrs["volume_acquisition_duration"] == 1.0
    validate_voxeldata(data)
    canonical = canonicalize_voxeldata(data.isel(j=1))
    assert canonical.vd.geometry_dims == ("time", "view")
    np.testing.assert_array_equal(canonical.vd.affines["lab"], data.vd.affines["lab"])
    np.testing.assert_allclose(
        canonical.isel(time=2, view=1).z.squeeze("j"), data.isel(time=2, view=1, j=1).z
    )


def test_named_timestamps_independent_of_shared_geometry():
    """fUSI timing defaults follow named timestamp axes, not affine dependencies."""
    time = xr.DataArray(
        [[0.0, 0.2], [1.0, 1.2], [2.0, 2.2]],
        dims=("time", "view"),
        attrs={"units": "s"},
    )
    data = create_voxeldata(
        np.zeros((3, 2, 1, 2, 3)),
        dims=("time", "view", "k", "j", "i"),
        extra_coords={"view": [11, 41]},
        time=time.transpose("view", "time"),
        voxel_to_world=np.eye(4),
        geometry_dims=(),
    )
    assert data.vd.geometry_dims == ()
    np.testing.assert_allclose(data.time, time)
    assert data.time.attrs["volume_acquisition_duration"] == pytest.approx(0.2)
    assert data.time.attrs["volume_acquisition_reference"] == "start"
    validate_voxeldata(data)


@pytest.mark.parametrize("dim", ["view", "pose"])
def test_zarr_roundtrip_preserves_named_geometry_axes(
    tmp_path, sample_named_geometry_voxeldata, dim
):
    """Persist dependency names/order, full frame stacks, and fUSI timing metadata."""
    data = sample_named_geometry_voxeldata
    if dim != "view":
        data = data.rename(view=dim)
    path = tmp_path / "geometry.zarr"
    io.save(data, path)
    stored = xr.open_zarr(path)
    assert stored[next(iter(stored.data_vars))].attrs[
        "voxel_to_world_geometry_dims"
    ] == ["time", dim]
    assert not {"z", "y", "x"} & set(stored.coords)
    stored.close()
    restored = io.load(path)
    assert restored.vd.geometry_dims == ("time", dim)
    np.testing.assert_array_equal(restored.transpose(*data.dims).values, data.values)
    np.testing.assert_allclose(restored.vd.voxel_to_world, data.vd.voxel_to_world)
    np.testing.assert_array_equal(restored.vd.affines["lab"], data.vd.affines["lab"])
    assert restored.time.attrs == data.time.attrs
    assert "voxel_to_world_geometry_dims" not in restored.attrs
    np.testing.assert_allclose(
        restored.isel({"time": 2, dim: 1}).z, data.isel({"time": 2, dim: 1}).z
    )


@pytest.mark.parametrize("duration", [None, 0.05])
def test_named_multiaxis_timestamps_roundtrip(
    tmp_path, sample_named_geometry_voxeldata, duration
):
    """Timestamp dependencies remain separate from geometry and slice timing."""
    source = sample_named_geometry_voxeldata
    values = (
        np.arange(3.0)[:, None, None]
        + np.array([0.0, 0.1])[None, :, None]
        + np.array([0.0, 0.3])[None, None, :]
    )
    timestamps = xr.DataArray(
        values,
        dims=("time", "view", "channel"),
        attrs={"units": "s", "volume_acquisition_reference": "end"},
    )
    data = create_voxeldata(
        source.data,
        dims=source.dims,
        time=timestamps.transpose("view", "channel", "time"),
        extra_coords={"view": source.view, "channel": source.channel},
        k=source.k,
        j=source.j,
        i=source.i,
        voxel_to_world=source.vd.voxel_to_world,
        geometry_dims=source.vd.geometry_dims,
        affines=source.vd.affines,
        volume_acquisition_duration=duration,
    )
    assert data.vd.geometry_dims == ("time", "view")
    np.testing.assert_array_equal(
        data.time.transpose("time", "view", "channel"), values
    )
    assert data.time.attrs["volume_acquisition_reference"] == "end"
    assert data.time.attrs["volume_acquisition_duration"] == pytest.approx(
        0.1 if duration is None else duration
    )
    validate_voxeldata(data)
    path = tmp_path / "multiaxis-time.zarr"
    io.save(data, path)
    restored = io.load(path)
    np.testing.assert_array_equal(
        restored.time.transpose("time", "view", "channel"), values
    )
    assert restored.time.attrs == data.time.attrs
    assert restored.vd.geometry_dims == data.vd.geometry_dims
    np.testing.assert_array_equal(restored.vd.affines["lab"], data.vd.affines["lab"])
    np.testing.assert_array_equal(restored.transpose(*data.dims).values, data.values)


def test_zarr_shared_pose_affine_does_not_infer_dependency(tmp_path):
    """Shared geometry remains shared when a pose dimension is present."""
    data = create_voxeldata(
        np.arange(12).reshape(2, 1, 2, 3),
        dims=("pose", "k", "j", "i"),
        voxel_to_world=np.eye(4),
    )
    path = tmp_path / "shared.zarr"
    io.save(data, path)
    restored = io.load(path)
    assert restored.vd.geometry_dims == ()
    np.testing.assert_array_equal(restored.values, data.values)
    np.testing.assert_allclose(restored.vd.voxel_to_world, np.eye(4))


def test_zarr_affine_stack_requires_dependency_metadata(
    tmp_path, sample_named_geometry_voxeldata
):
    """Serialized affine stacks must name their geometry dependencies."""
    data = sample_named_geometry_voxeldata.isel(time=0, drop=True).rename(view="pose")
    raw = data.drop_vars(["z", "y", "x"])
    raw.attrs = {
        "voxel_to_world": data.vd.voxel_to_world.tolist(),
        "voxel_to_world_named_affines": {
            name: value.tolist() for name, value in data.vd.affines.items()
        },
    }
    path = tmp_path / "missing-dependencies.zarr"
    raw.to_dataset(name="data").to_zarr(path)
    with pytest.raises(ValueError, match="explicit geometry_dims"):
        io.load(path)


def test_geometry_axis_reindex_is_documented_as_unsupported(
    sample_named_geometry_voxeldata,
):
    """Use label selection for reordering; reject stale geometry on VoxelData access."""
    data = sample_named_geometry_voxeldata
    safe = data.sel(view=[41, 11])
    np.testing.assert_array_equal(
        safe.vd.voxel_to_world, data.vd.voxel_to_world[:, [1, 0]]
    )
    stale = data.reindex(view=[41, 11])
    with pytest.raises(ValueError, match="no longer matches the affine index"):
        validate_voxeldata(stale)

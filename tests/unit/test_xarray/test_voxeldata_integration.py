"""Integration boundaries between VoxelData geometry and fUSI acquisition semantics."""

import numpy as np
import pytest
import voxeldata
import xarray as xr

from confusius import io
from confusius._utils.geometry import VoxelToWorldIndex, attach_voxel_to_world_index
from confusius.multipose import stack_poses
from confusius.validation import ensure_voxeldata, validate_voxeldata
from confusius.xarray import apply_affine, create_voxeldata


def test_geometry_owned_by_external_package(sample_voxeldata_3d):
    """ConfUSIus and external consumers use the same index and accessor classes."""
    assert VoxelToWorldIndex is voxeldata.VoxelToWorldIndex
    assert attach_voxel_to_world_index is voxeldata.attach_voxel_to_world_index
    assert apply_affine is voxeldata.apply_affine
    assert isinstance(sample_voxeldata_3d.vd, voxeldata.VoxelDataAccessor)
    for name in ("affine", "spacing", "origin", "direction"):
        assert not hasattr(sample_voxeldata_3d.fusi, name)


@pytest.mark.parametrize(
    "dims",
    [
        ("time", "component", "grid", "k", "j", "i"),
        ("grid", "time", "component", "k", "j", "i"),
    ],
)
def test_leading_order_preserved_and_canonicalization_optional(dims):
    """Creation and validation preserve order; canonicalization is explicit."""
    sizes = {"time": 3, "component": 2, "grid": 2, "k": 1, "j": 2, "i": 3}
    values = np.arange(np.prod([sizes[dim] for dim in dims])).reshape(
        tuple(sizes[dim] for dim in dims)
    )
    geometry = np.broadcast_to(np.eye(4), (2, 4, 4)).copy()
    data = create_voxeldata(
        values, dims=dims, voxel_to_world=geometry, dt=1.0, t0=[0.0, 0.2]
    )
    assert data.dims == dims
    np.testing.assert_array_equal(data.values, values)
    validate_voxeldata(data)
    validate_voxeldata(data.assign_coords(time=data.time.transpose("grid", "time")))

    canonical = ensure_voxeldata(data)
    assert canonical.dims == ("component", "grid", "time", "k", "j", "i")
    np.testing.assert_array_equal(canonical.transpose(*dims).values, values)
    assert canonical.time.attrs["volume_acquisition_duration"] == pytest.approx(0.2)


def test_generic_validation_does_not_require_fusi_acquisition_metadata():
    """The fUSI layer adds acquisition checks without changing generic VoxelData."""
    data = voxeldata.create_voxeldata(
        np.zeros((3, 1, 2, 3)),
        dims=("time", "k", "j", "i"),
        spacing=(1.0, 1.0, 1.0),
        dt=0.5,
    )
    data.vd.validate()
    with pytest.raises(ValueError, match="volume_acquisition_reference"):
        validate_voxeldata(data)
    with pytest.warns(UserWarning, match="defaulting"):
        fusi_data = ensure_voxeldata(data)
    assert fusi_data.time.attrs["volume_acquisition_duration"] == 0.5
    assert "volume_acquisition_duration" not in data.time.attrs


def test_scalar_selection_canonicalized_before_geometry_mutation(sample_voxeldata_3d):
    """Restore scalar-selected voxel axes explicitly before using VoxelData mutators."""
    selected = sample_voxeldata_3d.isel(k=1)
    result = ensure_voxeldata(selected).vd.apply(np.eye(4))
    assert result.sizes["k"] == 1
    np.testing.assert_array_equal(result.squeeze("k").values, selected.values)
    for dim in ("z", "y", "x"):
        np.testing.assert_allclose(result[dim].squeeze("k"), selected[dim])


def test_stack_probe_poses_preserves_distinct_named_frames():
    """Distinct source frames become a grid-dependent stack, not stale attrs."""
    first_frame = np.eye(4)
    second_frame = np.eye(4)
    second_frame[0, 3] = 5.0
    poses = [
        create_voxeldata(
            np.full((1, 2, 3), value),
            dims=("k", "j", "i"),
            spacing=(1.0, 1.0, 1.0),
            affines={"world_to_lab": frame},
        )
        for value, frame in enumerate((first_frame, second_frame))
    ]
    result = stack_poses(poses, grid=[10, 20])
    np.testing.assert_array_equal(
        result.vd.get_affine("world_to_lab"), np.stack([first_frame, second_frame])
    )
    np.testing.assert_array_equal(result.isel(grid=1).values, np.ones((1, 2, 3)))
    assert "affines" not in result.attrs


def test_zarr_roundtrip_preserves_grid_geometry_and_named_frames(tmp_path):
    """Both affine stacks are serialized before the custom index is dropped."""
    geometry = np.broadcast_to(np.eye(4), (2, 4, 4)).copy()
    geometry[1, 0, 3] = 10.0
    frames = np.broadcast_to(np.eye(4), (2, 4, 4)).copy()
    frames[1, 1, 3] = 5.0
    data = create_voxeldata(
        np.arange(12).reshape(2, 1, 2, 3),
        dims=("grid", "k", "j", "i"),
        grid=[10, 20],
        voxel_to_world=geometry,
        affines={"world_to_lab": frames},
    )
    path = tmp_path / "grids.zarr"
    io.save(data, path)
    restored = io.load(path)
    np.testing.assert_array_equal(restored.values, data.values)
    np.testing.assert_array_equal(restored.vd.voxel_to_world, geometry)
    np.testing.assert_array_equal(restored.vd.get_affine("world_to_lab"), frames)
    assert "affines" not in restored.attrs
    assert "voxel_to_world_named_affines" not in restored.attrs
    assert isinstance(restored.vd.index, voxeldata.VoxelToWorldIndex)


@pytest.mark.parametrize("legacy_key", ["pose", "affines"])
@pytest.mark.parametrize("loader", [io.load, io.load_atlas])
def test_legacy_zarr_geometry_rejected_without_silently_losing_frames(
    tmp_path, legacy_key, loader
):
    """Old geometry requires explicit migration rather than silently dropping frames."""
    attrs = {"voxel_to_world": np.eye(4).tolist()}
    if legacy_key == "affines":
        attrs["affines"] = {"world_to_lab": np.eye(4).tolist()}
    data = xr.DataArray(np.zeros((1, 2, 3)), dims=("k", "j", "i"), attrs=attrs)
    if legacy_key == "pose":
        data = data.expand_dims(pose=[0])
    path = tmp_path / "legacy.zarr"
    data.to_dataset(name="data").to_zarr(path, consolidated=False)
    with pytest.raises(ValueError, match="Legacy ConfUSIus"):
        loader(path, consolidated=False)

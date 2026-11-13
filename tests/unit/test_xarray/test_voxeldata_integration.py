"""Integration boundaries between VoxelData geometry and fUSI acquisition semantics."""

import numpy as np
import pytest
import voxeldata

from confusius import io
from confusius._utils.geometry import VoxelToWorldIndex, attach_voxel_to_world_index
from confusius.multipose import stack_poses
from confusius.validation import ensure_voxeldata, validate_voxeldata
from confusius.xarray import create_voxeldata


def test_geometry_owned_by_external_package(sample_voxeldata_3d):
    """ConfUSIus and external consumers use the same index and accessor classes."""
    assert VoxelToWorldIndex is voxeldata.VoxelToWorldIndex
    assert attach_voxel_to_world_index is voxeldata.attach_voxel_to_world_index
    assert isinstance(sample_voxeldata_3d.vd, voxeldata.VoxelDataAccessor)
    for name in (
        "affine",
        "spacing",
        "origin",
        "direction",
        "translate",
        "rotate",
        "zoom",
        "flip",
    ):
        assert not hasattr(sample_voxeldata_3d.fusi, name)


@pytest.mark.parametrize(
    "dims",
    [
        ("time", "component", "pose", "k", "j", "i"),
        ("pose", "time", "component", "k", "j", "i"),
    ],
)
def test_leading_order_preserved_and_canonicalization_optional(dims):
    """Creation and validation preserve order; canonicalization is explicit."""
    sizes = {"time": 3, "component": 2, "pose": 2, "k": 1, "j": 2, "i": 3}
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
    validate_voxeldata(data.assign_coords(time=data.time.transpose("pose", "time")))

    canonical = ensure_voxeldata(data)
    assert canonical.dims == ("component", "pose", "time", "k", "j", "i")
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
    """Distinct source frames become a pose-dependent stack, not stale attrs."""
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
    result = stack_poses(poses, pose=[10, 20])
    np.testing.assert_array_equal(
        result.vd.get_affine("world_to_lab"), np.stack([first_frame, second_frame])
    )
    np.testing.assert_array_equal(result.isel(pose=1).values, np.ones((1, 2, 3)))
    assert "affines" not in result.attrs


def test_zarr_roundtrip_preserves_grid_geometry_and_named_frames(tmp_path):
    """Both affine stacks are serialized before the custom index is dropped."""
    geometry = np.broadcast_to(np.eye(4), (2, 4, 4)).copy()
    geometry[1, 0, 3] = 10.0
    frames = np.broadcast_to(np.eye(4), (2, 4, 4)).copy()
    frames[1, 1, 3] = 5.0
    data = create_voxeldata(
        np.arange(12).reshape(2, 1, 2, 3),
        dims=("pose", "k", "j", "i"),
        pose=[10, 20],
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


@pytest.mark.parametrize(
    "name",
    [
        "TransformCenter",
        "apply_affine",
        "apply_translation",
        "apply_rotation",
        "apply_zoom",
        "apply_flip",
        "get_bounding_box",
        "get_geometry_center",
        "get_relative_affine",
        "reindex_voxels",
        "reindex_voxels_like",
    ],
)
def test_public_affine_exports_are_voxeldata_functions(name):
    """The public xarray namespace exposes VoxelData's affine implementations."""
    from confusius import xarray as extensions

    assert getattr(extensions, name) is getattr(voxeldata, name)


def test_world_transform_preserves_acquisition_timing(sample_voxeldata_3dt):
    """VoxelData transforms retain ConfUSIus timing metadata and sample ordering."""
    source = sample_voxeldata_3dt
    shifted = source.vd.translate((1.0, -2.0, 3.0))
    np.testing.assert_array_equal(shifted.values, source.values)
    np.testing.assert_array_equal(shifted.time, source.time)
    assert shifted.time.attrs == source.time.attrs
    assert shifted.vd.dt == pytest.approx(source.vd.dt)
    for dim, offset in zip(("z", "y", "x"), (1, -2, 3), strict=True):
        np.testing.assert_allclose(shifted.coords[dim], source.coords[dim] + offset)
    validate_voxeldata(shifted)


def test_internal_geometry_exports_are_voxeldata_functions():
    """Internal consumers share the same geometry classes and helper functions."""
    from voxeldata import geometry as generic

    from confusius._utils import geometry

    for name in geometry.__all__:
        assert getattr(geometry, name) is getattr(generic, name)

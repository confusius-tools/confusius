"""Tests for canonical xarray constructor helpers."""

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from confusius.validation import validate_fusi, validate_iq
from confusius.xarray import create_fusi_dataarray, create_iq_dataarray

_VOXEL_DIM_BY_WORLD_NAME = {"z": "k", "y": "j", "x": "i"}


def _world_coord_1d(da: xr.DataArray, name: str) -> np.ndarray:
    """Return a world coordinate's 1D values, reducing other axis-aligned dims."""
    coord = da.coords[name]
    dim = _VOXEL_DIM_BY_WORLD_NAME[name]
    if coord.dims == (dim,):
        return coord.values
    others = {d: 0 for d in coord.dims if d != dim}
    return coord.isel(others).values


def test_create_fusi_dataarray_builds_canonical_volume():
    """Spatial input dims are canonicalized to native voxel dims."""
    data = np.zeros((5, 1, 8, 12))

    result = create_fusi_dataarray(
        data,
        dims=("time", "k", "j", "i"),
        dt=0.5,
        t0=1.0,
        spacing=(0.4, 0.1, 0.2),
        origin=(2.0, 0.05, 0.1),
    )

    assert result.dims == ("time", "k", "j", "i")
    assert_allclose(result.coords["time"], 1.0 + np.arange(5) * 0.5)
    assert_allclose(_world_coord_1d(result, "z"), [2.0])
    assert_allclose(_world_coord_1d(result, "y"), 0.05 + np.arange(8) * 0.1)
    assert_allclose(_world_coord_1d(result, "x"), 0.1 + np.arange(12) * 0.2)
    assert result.coords["z"].attrs == {"units": "mm"}
    validate_fusi(result, require_time=True)


def test_create_fusi_dataarray_does_not_copy_numpy_input():
    """A numpy array passed straight through must not be copied."""
    data = np.zeros((4, 8, 12))

    result = create_fusi_dataarray(
        data, dims=("k", "j", "i"), spacing=(0.4, 0.1, 0.2), origin=(0.0, 0.0, 0.0)
    )

    assert np.shares_memory(result.values, data)


def test_create_fusi_dataarray_keeps_dask_input_lazy():
    """A dask array must stay a dask array, never get eagerly computed."""
    data = da.zeros((4, 8, 12), chunks=(2, 4, 6))

    result = create_fusi_dataarray(
        data, dims=("k", "j", "i"), spacing=(0.4, 0.1, 0.2), origin=(0.0, 0.0, 0.0)
    )

    assert isinstance(result.data, da.Array)


def test_create_iq_dataarray_keeps_dask_input_lazy():
    """create_iq_dataarray must preserve dask laziness for large IQ volumes."""
    data = da.zeros((5, 4, 8, 12), chunks=(5, 2, 4, 6), dtype=np.complex64)

    result = create_iq_dataarray(
        data,
        dims=("time", "k", "j", "i"),
        dt=0.5,
        spacing=(0.4, 0.1, 0.2),
        origin=(0.0, 0.0, 0.0),
    )

    assert isinstance(result.data, da.Array)


def test_create_fusi_dataarray_uses_default_probe_origins():
    """Default origins match the probe-centered z/x and surface-referenced y model."""
    result = create_fusi_dataarray(
        np.zeros((3, 8, 4)),
        dims=("k", "j", "i"),
        spacing=(0.4, 0.1, 0.2),
    )

    assert_allclose(_world_coord_1d(result, "z"), [-0.4, 0.0, 0.4])
    assert_allclose(_world_coord_1d(result, "y"), 0.05 + np.arange(8) * 0.1)
    assert_allclose(_world_coord_1d(result, "x"), [-0.3, -0.1, 0.1, 0.3])


def test_create_fusi_dataarray_world_coord_attrs_overrides_units():
    """world_coord_attrs overrides given keys, keeps auto-computed defaults for others."""
    result = create_fusi_dataarray(
        np.zeros((4, 8, 12)),
        dims=("k", "j", "i"),
        spacing=(0.4, 0.1, 0.2),
        origin=(0.0, 0.0, 0.0),
        world_coord_attrs={"z": {"units": "um"}},
    )

    assert result.coords["z"].attrs["units"] == "um"
    assert result.coords["y"].attrs["units"] == "mm"


def test_create_fusi_dataarray_pads_missing_spatial_dim():
    """A 2D input gets singleton axes for missing spatial dims."""
    result = create_fusi_dataarray(
        np.zeros((5, 8, 12)),
        dims=("time", "j", "i"),
        time=np.arange(5) * 0.5,
        spacing=(0.4, 0.1, 0.2),
        origin=(2.0, 0.0, 0.0),
    )

    assert result.dims == ("time", "k", "j", "i")
    assert result.shape == (5, 1, 8, 12)
    assert "z" in result.coords
    assert_allclose(_world_coord_1d(result, "y"), np.arange(8) * 0.1)


def test_create_fusi_dataarray_accepts_direction_matrix():
    """Direction is folded into the voxel-to-world affine."""
    result = create_fusi_dataarray(
        np.zeros((2, 3, 4)),
        dims=("k", "j", "i"),
        spacing=(1.0, 2.0, 3.0),
        origin=(4.0, 5.0, 6.0),
        direction=np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
    )

    assert_allclose(
        result.fusi.affine.voxel_to_world,
        [
            [0.0, 2.0, 0.0, 4.0],
            [1.0, 0.0, 0.0, 5.0],
            [0.0, 0.0, 3.0, 6.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    )


def test_create_fusi_dataarray_accepts_voxel_to_world_affine():
    """A full affine is an alternative to origin/spacing/direction."""
    affine = np.array(
        [
            [0.4, 0.0, 0.0, 2.0],
            [0.0, 0.1, 0.0, 3.0],
            [0.0, 0.0, 0.2, 4.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    result = create_fusi_dataarray(
        np.zeros((1, 8, 12)),
        dims=("k", "j", "i"),
        voxel_to_world=affine,
    )

    assert_allclose(result.fusi.affine.voxel_to_world, affine)


def test_create_fusi_dataarray_accepts_pose_stacked_voxel_to_world():
    """A pose-stacked affine wires one voxel-to-world affine per pose."""
    affine = np.stack(
        [
            np.eye(4),
            np.array(
                [
                    [1.0, 0.0, 0.0, 100.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        ]
    )

    result = create_fusi_dataarray(
        np.zeros((2, 2, 3, 4)),
        dims=("pose", "k", "j", "i"),
        voxel_to_world=affine,
    )

    assert result.dims == ("pose", "k", "j", "i")
    assert_allclose(result.coords["pose"].values, [0, 1])
    assert_allclose(
        result.coords["z"].isel(pose=1, j=0, i=0).values, [100.0, 101.0]
    )
    validate_fusi(result)


def test_create_fusi_dataarray_pose_stacked_voxel_to_world_requires_pose_dim():
    """A pose-stacked affine without a `pose` dim in `dims` raises clearly."""
    affine = np.stack([np.eye(4), np.eye(4)])

    with pytest.raises(ValueError, match="requires a 'pose' dimension"):
        create_fusi_dataarray(
            np.zeros((2, 3, 4)),
            dims=("k", "j", "i"),
            voxel_to_world=affine,
        )


def test_create_fusi_dataarray_pose_stacked_voxel_to_world_rejects_wrong_shape():
    """A pose-stacked affine whose per-pose blocks aren't (4, 4) raises clearly."""
    affine = np.stack([np.eye(3), np.eye(3)])

    with pytest.raises(ValueError, match="must have shape \\(npose, 4, 4\\)"):
        create_fusi_dataarray(
            np.zeros((2, 2, 3, 4)),
            dims=("pose", "k", "j", "i"),
            voxel_to_world=affine,
        )


def test_create_fusi_dataarray_pose_stacked_voxel_to_world_rejects_non_homogeneous():
    """A pose-stacked affine whose last row isn't [0, 0, 0, 1] raises clearly."""
    affine = np.stack([np.eye(4), np.eye(4)])
    affine[1, -1] = [0.0, 0.0, 0.0, 2.0]  # pose 1's last row is not homogeneous.

    with pytest.raises(ValueError, match="must be a homogeneous affine"):
        create_fusi_dataarray(
            np.zeros((2, 2, 3, 4)),
            dims=("pose", "k", "j", "i"),
            voxel_to_world=affine,
        )


def test_create_fusi_dataarray_pose_stacked_voxel_to_world_rejects_mixed_geometry():
    """A pose-stacked affine is still mutually exclusive with spacing/origin/direction."""
    affine = np.stack([np.eye(4), np.eye(4)])

    with pytest.raises(ValueError, match="mutually exclusive"):
        create_fusi_dataarray(
            np.zeros((2, 2, 3, 4)),
            dims=("pose", "k", "j", "i"),
            voxel_to_world=affine,
            spacing=(1.0, 1.0, 1.0),
        )


def test_create_fusi_dataarray_pose_stacked_voxel_to_world_rejects_wrong_length():
    """A pose stack length must match the `pose` dimension size."""
    affine = np.stack([np.eye(4)])

    with pytest.raises(ValueError, match="does not match the 'pose' dimension size"):
        create_fusi_dataarray(
            np.zeros((2, 2, 3, 4)),
            dims=("pose", "k", "j", "i"),
            voxel_to_world=affine,
        )


def test_create_fusi_dataarray_accepts_2d_per_pose_time():
    """A 2D (n_time, npose) time array gives each pose its own real timestamps."""
    npose = 3
    n_time = 4
    time_2d = np.stack(
        [np.arange(n_time) * 2.4 + p * 0.6 for p in range(npose)], axis=1
    )
    affine = np.stack([np.eye(4) for _ in range(npose)])

    result = create_fusi_dataarray(
        np.zeros((n_time, npose, 2, 3, 4)),
        dims=("time", "pose", "k", "j", "i"),
        time=xr.DataArray(time_2d, attrs={"units": "s"}),
        pose=np.arange(npose),
        voxel_to_world=affine,
    )

    assert result.coords["time"].dims == ("time", "pose")
    assert_allclose(result.coords["time"].values, time_2d)
    assert result.coords["time"].attrs["units"] == "s"
    assert "time" not in result.xindexes
    # spacing along "time" specifically (not the cross-pose 0.6 offset).
    assert result.fusi.spacing["time"] == pytest.approx(2.4)


def test_create_fusi_dataarray_2d_time_requires_pose_dim():
    """A 2D time array without a `pose` dim in `dims` raises clearly."""
    time_2d = np.zeros((4, 2))

    with pytest.raises(ValueError, match="requires a 'pose' dimension"):
        create_fusi_dataarray(
            np.zeros((4, 2, 3, 4)),
            dims=("time", "k", "j", "i"),
            time=time_2d,
            spacing=(1.0, 1.0, 1.0),
        )


def test_create_fusi_dataarray_2d_time_rejects_wrong_pose_count():
    """A 2D time array's pose column count must match the `pose` dimension size."""
    npose = 3
    time_2d = np.zeros((4, 2))
    affine = np.stack([np.eye(4) for _ in range(npose)])

    with pytest.raises(ValueError, match="pose columns"):
        create_fusi_dataarray(
            np.zeros((4, npose, 2, 3, 4)),
            dims=("time", "pose", "k", "j", "i"),
            time=time_2d,
            pose=np.arange(npose),
            voxel_to_world=affine,
        )


def test_create_fusi_dataarray_rejects_spatial_extra_coords():
    """Spatial coordinates must come from voxel-to-world geometry, not constructor coord arrays."""
    with pytest.raises(ValueError, match="extra_coords must not include core"):
        create_fusi_dataarray(
            np.zeros((2, 3)),
            dims=("j", "i"),
            extra_coords={"x": np.arange(3)},
            spacing=(1.0, 0.1, 0.2),
        )


def test_create_fusi_dataarray_rejects_mixed_geometry_inputs():
    """The affine and decomposed geometry inputs are mutually exclusive."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        create_fusi_dataarray(
            np.zeros((2, 3)),
            dims=("j", "i"),
            spacing=(1.0, 0.1, 0.2),
            voxel_to_world=np.eye(4),
        )


def test_create_fusi_dataarray_rejects_missing_geometry():
    """Spatial geometry is mandatory."""
    with pytest.raises(ValueError, match="spacing must be provided"):
        create_fusi_dataarray(np.zeros((2, 3)), dims=("j", "i"))


def test_create_fusi_dataarray_rejects_wrong_length_explicit_coord():
    """An explicit coordinate whose length disagrees with the axis size is rejected."""
    with pytest.raises(ValueError, match=r"Coordinate 'time' must be 1D with length 5"):
        create_fusi_dataarray(
            np.zeros((5, 1, 2, 3)),
            dims=("time", "k", "j", "i"),
            time=np.arange(3) * 0.1,
            spacing=(0.4, 0.1, 0.2),
        )


def test_create_fusi_dataarray_rejects_none_t0_with_dt():
    """An explicit `t0=None` cannot serve as the time-coordinate origin."""
    with pytest.raises(
        ValueError, match="Origin for dimension 'time' must be provided"
    ):
        create_fusi_dataarray(
            np.zeros((5, 1, 2, 3)),
            dims=("time", "k", "j", "i"),
            dt=0.5,
            t0=None,  # ty: ignore[invalid-argument-type]
            spacing=(0.4, 0.1, 0.2),
        )


def test_create_fusi_dataarray_singleton_time_uses_t0_and_dt():
    """A singleton time dimension uses `t0`, and still requires `dt`."""
    result = create_fusi_dataarray(
        np.zeros((1, 1, 2, 3)),
        dims=("time", "k", "j", "i"),
        t0=3.0,
        dt=0.6,
        spacing=(0.4, 0.1, 0.2),
    )

    assert_allclose(result.coords["time"], [3.0])
    assert result.coords["time"].attrs["volume_acquisition_duration"] == 0.6


def test_create_fusi_dataarray_singleton_time_requires_dt():
    """A singleton time dimension without `dt` or an explicit coordinate raises."""
    with pytest.raises(ValueError, match="Spacing for dimension 'time' is required"):
        create_fusi_dataarray(
            np.zeros((1, 1, 2, 3)),
            dims=("time", "k", "j", "i"),
            t0=3.0,
            spacing=(0.4, 0.1, 0.2),
        )


def test_create_fusi_dataarray_rejects_wrong_length_spacing():
    """`spacing` must contain exactly 3 values, one per z/y/x axis."""
    with pytest.raises(ValueError, match="spacing must have length 3 in z/y/x order"):
        create_fusi_dataarray(
            np.zeros((2, 3)),
            dims=("j", "i"),
            spacing=(1.0, 0.2),
        )


def test_create_fusi_dataarray_rejects_wrong_shape_voxel_to_world():
    """`voxel_to_world` must be a 4x4 matrix."""
    with pytest.raises(ValueError, match=r"voxel_to_world must have shape \(4, 4\)"):
        create_fusi_dataarray(
            np.zeros((2, 3)),
            dims=("j", "i"),
            voxel_to_world=np.eye(3),
        )


def test_create_fusi_dataarray_rejects_non_homogeneous_voxel_to_world():
    """`voxel_to_world`'s last row must be `[0, 0, 0, 1]`."""
    affine = np.eye(4)
    affine[3, 3] = 2.0
    with pytest.raises(ValueError, match="voxel_to_world must be a homogeneous affine"):
        create_fusi_dataarray(
            np.zeros((2, 3)),
            dims=("j", "i"),
            voxel_to_world=affine,
        )


def test_create_fusi_dataarray_rejects_wrong_length_origin():
    """`origin` must contain exactly 3 values, one per z/y/x axis."""
    with pytest.raises(ValueError, match="origin must have length 3 in z/y/x order"):
        create_fusi_dataarray(
            np.zeros((2, 3)),
            dims=("j", "i"),
            spacing=(1.0, 0.1, 0.2),
            origin=(0.0, 0.0),
        )


def test_create_fusi_dataarray_rejects_non_finite_origin():
    """`origin` values must all be finite."""
    with pytest.raises(ValueError, match="origin must contain finite values"):
        create_fusi_dataarray(
            np.zeros((2, 3)),
            dims=("j", "i"),
            spacing=(1.0, 0.1, 0.2),
            origin=(np.nan, 0.0, 0.0),
        )


def test_create_fusi_dataarray_rejects_wrong_shape_direction():
    """`direction` must be a 3x3 matrix."""
    with pytest.raises(ValueError, match=r"direction must have shape \(3, 3\)"):
        create_fusi_dataarray(
            np.zeros((2, 3)),
            dims=("j", "i"),
            spacing=(1.0, 0.1, 0.2),
            direction=np.eye(2),
        )


def test_create_fusi_dataarray_rejects_non_finite_direction():
    """`direction` values must all be finite."""
    direction = np.eye(3)
    direction[0, 1] = np.inf
    with pytest.raises(ValueError, match="direction must contain finite values"):
        create_fusi_dataarray(
            np.zeros((2, 3)),
            dims=("j", "i"),
            spacing=(1.0, 0.1, 0.2),
            direction=direction,
        )


def test_create_fusi_dataarray_rejects_invalid_volume_acquisition_reference():
    """`volume_acquisition_reference` must be one of the recognized timing references."""
    with pytest.raises(ValueError, match="volume_acquisition_reference must be one of"):
        create_fusi_dataarray(
            np.zeros((5, 1, 2, 3)),
            dims=("time", "k", "j", "i"),
            dt=0.5,
            spacing=(0.4, 0.1, 0.2),
            volume_acquisition_reference="middle",  # ty: ignore[invalid-argument-type]
        )


def test_create_fusi_dataarray_rejects_duration_without_time_dim():
    """`volume_acquisition_duration` requires a `time` dimension to attach to."""
    with pytest.raises(
        ValueError,
        match="time and volume_acquisition_duration require a 'time' dimension",
    ):
        create_fusi_dataarray(
            np.zeros((2, 3)),
            dims=("j", "i"),
            spacing=(1.0, 0.1, 0.2),
            volume_acquisition_duration=1.0,
        )


def test_create_fusi_dataarray_rejects_world_dim_names():
    """`dims` must use native voxel names; world z/y/x names are rejected."""
    with pytest.raises(ValueError, match="dims must use native voxel names"):
        create_fusi_dataarray(
            np.zeros((2, 3, 4)),
            dims=("z", "y", "x"),
            spacing=(0.4, 0.1, 0.2),
        )


def test_create_iq_dataarray_validates_complex_input():
    """IQ constructor delegates geometry and enforces complex data."""
    result = create_iq_dataarray(
        np.ones((5, 1, 2, 3), dtype=np.complex64),
        dims=("time", "k", "j", "i"),
        time=xr.DataArray(np.arange(5) * 0.1, dims=("time",), attrs={"units": "s"}),
        spacing=(0.4, 0.1, 0.2),
    )

    assert result.name == "iq"
    validate_iq(result)

    with pytest.raises(TypeError, match="complex"):
        create_iq_dataarray(
            np.ones((5, 1, 2, 3)),
            dims=("time", "k", "j", "i"),
            time=np.arange(5) * 0.1,
            spacing=(0.4, 0.1, 0.2),
        )

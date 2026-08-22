"""Tests for canonical xarray constructor helpers (`create_voxeldata`).

Grouped by input shape, roughly in order of increasing complexity:

1. Padding missing spatial dims (1D/2D data -> dense k/j/i).
2. Regular 3D volumes: geometry via voxel_to_world vs spacing/origin/direction,
   default vs custom k/j/i.
3. 3D volumes + time: explicit time array vs dt/t0.
4. 3D volumes + pose (voxel_to_world only; no per-pose spacing/origin API exists).
5. 3D volumes + pose + time: explicit (time, pose) array vs per-pose t0 + shared dt.
6. Extra (non-core) dims and their coordinates.
7. Geometry/coordinate/timing input validation (error paths).
"""

from collections.abc import Callable

import dask.array as da
import numpy as np
import numpy.typing as npt
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from confusius.validation import validate_voxeldata
from confusius.xarray import create_voxeldata

_VOXEL_DIM_BY_WORLD_NAME = {"z": "k", "y": "j", "x": "i"}


def _world_coord_1d(da: xr.DataArray, name: str) -> np.ndarray:
    """Return a world coordinate's 1D values, reducing other axis-aligned dims."""
    coord = da.coords[name]
    dim = _VOXEL_DIM_BY_WORLD_NAME[name]
    if coord.dims == (dim,):
        return coord.values
    others = {d: 0 for d in coord.dims if d != dim}
    return coord.isel(others).values


# ---------------------------------------------------------------------------
# 1. Padding missing spatial dims
# ---------------------------------------------------------------------------


def test_create_voxeldata_pads_missing_spatial_dims_from_2d():
    """A 2D spatial input gets singleton axes for the two missing voxel dims."""
    result = create_voxeldata(
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


def test_create_voxeldata_pads_missing_spatial_dims_from_1d():
    """A 1D spatial input gets singleton axes for the two missing voxel dims."""
    result = create_voxeldata(
        np.zeros(12),
        dims=("i",),
        spacing=(0.4, 0.1, 0.2),
        origin=(2.0, 0.0, 0.0),
    )

    assert result.dims == ("k", "j", "i")
    assert result.shape == (1, 1, 12)
    assert_allclose(_world_coord_1d(result, "z"), [2.0])
    assert_allclose(_world_coord_1d(result, "y"), [0.0])
    assert_allclose(_world_coord_1d(result, "x"), np.arange(12) * 0.2)
    validate_voxeldata(result)


# ---------------------------------------------------------------------------
# 2. Regular 3D volumes: geometry inputs
# ---------------------------------------------------------------------------


def test_create_voxeldata_spacing_origin_default_voxel_coords():
    """spacing/origin geometry with default (dense zero-based) k/j/i coordinates."""
    data = np.zeros((1, 8, 12))

    result = create_voxeldata(
        data,
        dims=("k", "j", "i"),
        spacing=(0.4, 0.1, 0.2),
        origin=(2.0, 0.05, 0.1),
    )

    assert result.dims == ("k", "j", "i")
    np.testing.assert_array_equal(result.coords["k"], [0])
    np.testing.assert_array_equal(result.coords["j"], np.arange(8))
    np.testing.assert_array_equal(result.coords["i"], np.arange(12))
    assert_allclose(_world_coord_1d(result, "z"), [2.0])
    assert_allclose(_world_coord_1d(result, "y"), 0.05 + np.arange(8) * 0.1)
    assert_allclose(_world_coord_1d(result, "x"), 0.1 + np.arange(12) * 0.2)
    assert result.coords["z"].attrs == {"units": "mm"}
    validate_voxeldata(result)


def test_create_voxeldata_does_not_copy_numpy_input():
    """A numpy array passed straight through must not be copied."""
    data = np.zeros((4, 8, 12))

    result = create_voxeldata(
        data, dims=("k", "j", "i"), spacing=(0.4, 0.1, 0.2), origin=(0.0, 0.0, 0.0)
    )

    assert np.shares_memory(result.values, data)


def test_create_voxeldata_keeps_dask_input_lazy():
    """A dask array must stay a dask array, never get eagerly computed."""
    data = da.zeros((4, 8, 12), chunks=(2, 4, 6))

    result = create_voxeldata(
        data, dims=("k", "j", "i"), spacing=(0.4, 0.1, 0.2), origin=(0.0, 0.0, 0.0)
    )

    assert isinstance(result.data, da.Array)


def test_create_voxeldata_accepts_explicit_voxel_coords_with_spacing():
    """Explicit k/j/i coordinates may be sparse, combined with spacing/origin."""
    result = create_voxeldata(
        np.zeros((3, 2, 2)),
        dims=("k", "j", "i"),
        k=[0, 2, 5],
        j=[10, 12],
        i=[20, 23],
        spacing=(0.4, 0.1, 0.2),
        origin=(0.0, 0.0, 0.0),
    )

    np.testing.assert_array_equal(result.coords["k"], [0, 2, 5])
    assert_allclose(_world_coord_1d(result, "z"), np.array([0, 2, 5]) * 0.4)
    validate_voxeldata(result)


def test_create_voxeldata_accepts_explicit_voxel_coords_with_voxel_to_world():
    """Explicit k/j/i coordinates may be sparse, combined with voxel_to_world."""
    result = create_voxeldata(
        np.zeros((3, 2, 2)),
        dims=("k", "j", "i"),
        k=[0, 2, 5],
        j=[10, 12],
        i=[20, 23],
        voxel_to_world=np.eye(4),
    )

    np.testing.assert_array_equal(result.coords["k"], [0, 2, 5])
    np.testing.assert_array_equal(_world_coord_1d(result, "z"), [0, 2, 5])
    validate_voxeldata(result)


def test_create_voxeldata_uses_default_probe_origins():
    """Default origins match the probe-centered z/x and surface-referenced y model."""
    result = create_voxeldata(
        np.zeros((3, 8, 4)),
        dims=("k", "j", "i"),
        spacing=(0.4, 0.1, 0.2),
    )

    assert_allclose(_world_coord_1d(result, "z"), [-0.4, 0.0, 0.4])
    assert_allclose(_world_coord_1d(result, "y"), 0.05 + np.arange(8) * 0.1)
    assert_allclose(_world_coord_1d(result, "x"), [-0.3, -0.1, 0.1, 0.3])


def test_create_voxeldata_sets_name_and_attrs():
    """`name` and arbitrary `attrs` (e.g. IQ velocity metadata) round-trip and
    satisfy the validation that requires them."""
    result = create_voxeldata(
        np.ones((4, 8, 12), dtype=np.complex64),
        dims=("k", "j", "i"),
        spacing=(0.4, 0.1, 0.2),
        origin=(0.0, 0.0, 0.0),
        name="iq",
        attrs={
            "description": "test recording",
            "transmit_frequency": 15.625e6,
            "beamforming_sound_velocity": 1540.0,
        },
    )

    assert result.name == "iq"
    assert result.attrs["description"] == "test recording"
    assert result.attrs["transmit_frequency"] == 15.625e6
    assert result.attrs["beamforming_sound_velocity"] == 1540.0
    validate_voxeldata(result, require_velocity_attrs=True)


def test_create_voxeldata_accepts_dims_in_any_order(
    identity_pose_affines: Callable[[int], npt.NDArray[np.float64]],
):
    """`dims` may list core *and* extra dims in any order; the result is
    canonicalized to `(extra_dims, time, pose, k, j, i)` regardless."""
    npose = 2
    n_time = 3
    n_component = 2
    # Deliberately scrambled: extra dim in the middle, core dims reversed.
    result = create_voxeldata(
        np.zeros((4, n_component, 3, npose, 2, n_time)),
        dims=("i", "component", "j", "pose", "k", "time"),
        extra_coords={"component": ["a", "b"]},
        dt=0.5,
        pose=np.arange(npose),
        voxel_to_world=identity_pose_affines(npose),
    )

    assert result.dims == ("component", "time", "pose", "k", "j", "i")
    assert result.shape == (n_component, n_time, npose, 2, 3, 4)
    np.testing.assert_array_equal(result.coords["component"], ["a", "b"])
    validate_voxeldata(result, require_time=True)


def test_create_voxeldata_accepts_direction_matrix():
    """Direction is folded into the voxel-to-world affine."""
    result = create_voxeldata(
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


def test_create_voxeldata_accepts_voxel_to_world_affine():
    """A full affine is an alternative to origin/spacing/direction, default k/j/i."""
    affine = np.array(
        [
            [0.4, 0.0, 0.0, 2.0],
            [0.0, 0.1, 0.0, 3.0],
            [0.0, 0.0, 0.2, 4.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    result = create_voxeldata(
        np.zeros((1, 8, 12)),
        dims=("k", "j", "i"),
        voxel_to_world=affine,
    )

    np.testing.assert_array_equal(result.coords["i"], np.arange(12))
    assert_allclose(result.fusi.affine.voxel_to_world, affine)
    assert result.coords["z"].attrs == {"units": "mm"}
    validate_voxeldata(result)


# ---------------------------------------------------------------------------
# 3. 3D volumes + time (no pose)
# ---------------------------------------------------------------------------


def test_create_voxeldata_time_from_dt_t0():
    """`dt`/`t0` build a regular `time` coordinate with inferred timing metadata."""
    data = np.zeros((5, 1, 8, 12))

    result = create_voxeldata(
        data,
        dims=("time", "k", "j", "i"),
        dt=0.5,
        t0=1.0,
        spacing=(0.4, 0.1, 0.2),
        origin=(2.0, 0.05, 0.1),
    )

    assert result.dims == ("time", "k", "j", "i")
    assert_allclose(result.coords["time"], 1.0 + np.arange(5) * 0.5)
    assert result.coords["time"].attrs["units"] == "s"
    assert result.coords["time"].attrs["volume_acquisition_reference"] == "start"
    assert result.coords["time"].attrs["volume_acquisition_duration"] == 0.5
    validate_voxeldata(result, require_time=True)


def test_create_voxeldata_time_explicit_plain_array():
    """A plain, regularly-spaced explicit `time` array gets inferred timing metadata."""
    result = create_voxeldata(
        np.zeros((5, 1, 2, 3)),
        dims=("time", "k", "j", "i"),
        time=np.arange(5) * 0.5,
        spacing=(0.4, 0.1, 0.2),
        origin=(0.0, 0.0, 0.0),
    )

    assert_allclose(result.coords["time"], np.arange(5) * 0.5)
    assert result.coords["time"].attrs["units"] == "s"
    assert result.coords["time"].attrs["volume_acquisition_reference"] == "start"
    assert result.coords["time"].attrs["volume_acquisition_duration"] == pytest.approx(
        0.5
    )
    validate_voxeldata(result, require_time=True)


def test_create_voxeldata_time_explicit_dataarray_attrs_take_priority():
    """An explicit `time` DataArray's own attrs override the constructor defaults."""
    result = create_voxeldata(
        np.ones((5, 1, 2, 3), dtype=np.complex64),
        dims=("time", "k", "j", "i"),
        time=xr.DataArray(
            np.arange(5) * 0.1,
            dims=("time",),
            attrs={
                "units": "s",
                "volume_acquisition_reference": "end",
                "volume_acquisition_duration": 0.05,
            },
        ),
        spacing=(0.4, 0.1, 0.2),
    )

    assert result.coords["time"].attrs["volume_acquisition_reference"] == "end"
    assert result.coords["time"].attrs["volume_acquisition_duration"] == 0.05


def test_create_voxeldata_preserves_iq_velocity_attrs():
    """IQ metadata is plain VoxelData attrs and can still be validated."""
    result = create_voxeldata(
        np.ones((5, 1, 2, 3), dtype=np.complex64),
        dims=("time", "k", "j", "i"),
        time=xr.DataArray(np.arange(5) * 0.1, dims=("time",), attrs={"units": "s"}),
        attrs={
            "transmit_frequency": 15.625e6,
            "beamforming_sound_velocity": 1540.0,
        },
        spacing=(0.4, 0.1, 0.2),
    )

    validate_voxeldata(result, require_velocity_attrs=True)


def test_create_voxeldata_singleton_time_uses_t0_and_dt():
    """A singleton time dimension uses `t0`, and still requires `dt`."""
    result = create_voxeldata(
        np.zeros((1, 1, 2, 3)),
        dims=("time", "k", "j", "i"),
        t0=3.0,
        dt=0.6,
        spacing=(0.4, 0.1, 0.2),
    )

    assert_allclose(result.coords["time"], [3.0])
    assert result.coords["time"].attrs["volume_acquisition_duration"] == 0.6


# ---------------------------------------------------------------------------
# 4. 3D volumes + pose (voxel_to_world only)
# ---------------------------------------------------------------------------


def test_create_voxeldata_pose_stacked_voxel_to_world(
    identity_pose_affines: Callable[[int], npt.NDArray[np.float64]],
):
    """A pose-stacked affine wires one voxel-to-world affine per pose."""
    affine = identity_pose_affines(2)
    affine[1, :3, 3] = [100.0, 0.0, 0.0]

    result = create_voxeldata(
        np.zeros((2, 2, 3, 4)),
        dims=("pose", "k", "j", "i"),
        voxel_to_world=affine,
    )

    assert result.dims == ("pose", "k", "j", "i")
    assert_allclose(result.coords["pose"].values, [0, 1])
    assert_allclose(result.coords["z"].isel(pose=1, j=0, i=0).values, [100.0, 101.0])
    validate_voxeldata(result)


def test_create_voxeldata_pose_stacked_voxel_to_world_requires_pose_dim():
    """A pose-stacked affine without a `pose` dim in `dims` raises clearly."""
    affine = np.stack([np.eye(4), np.eye(4)])

    with pytest.raises(ValueError, match="requires a 'pose' dimension"):
        create_voxeldata(
            np.zeros((2, 3, 4)),
            dims=("k", "j", "i"),
            voxel_to_world=affine,
        )


def test_create_voxeldata_pose_stacked_voxel_to_world_rejects_wrong_shape():
    """A pose-stacked affine whose per-pose blocks aren't (4, 4) raises clearly."""
    affine = np.stack([np.eye(3), np.eye(3)])

    with pytest.raises(ValueError, match="must have shape \\(npose, 4, 4\\)"):
        create_voxeldata(
            np.zeros((2, 2, 3, 4)),
            dims=("pose", "k", "j", "i"),
            voxel_to_world=affine,
        )


def test_create_voxeldata_pose_stacked_voxel_to_world_rejects_non_homogeneous():
    """A pose-stacked affine whose last row isn't [0, 0, 0, 1] raises clearly."""
    affine = np.stack([np.eye(4), np.eye(4)])
    affine[1, -1] = [0.0, 0.0, 0.0, 2.0]  # pose 1's last row is not homogeneous.

    with pytest.raises(ValueError, match="must be a homogeneous affine"):
        create_voxeldata(
            np.zeros((2, 2, 3, 4)),
            dims=("pose", "k", "j", "i"),
            voxel_to_world=affine,
        )


def test_create_voxeldata_pose_stacked_voxel_to_world_rejects_mixed_geometry():
    """A pose-stacked affine is still mutually exclusive with spacing/origin/direction."""
    affine = np.stack([np.eye(4), np.eye(4)])

    with pytest.raises(ValueError, match="mutually exclusive"):
        create_voxeldata(
            np.zeros((2, 2, 3, 4)),
            dims=("pose", "k", "j", "i"),
            voxel_to_world=affine,
            spacing=(1.0, 1.0, 1.0),
        )


def test_create_voxeldata_pose_stacked_voxel_to_world_rejects_wrong_length():
    """A pose stack length must match the `pose` dimension size."""
    affine = np.stack([np.eye(4)])

    with pytest.raises(ValueError, match="does not match the 'pose' dimension size"):
        create_voxeldata(
            np.zeros((2, 2, 3, 4)),
            dims=("pose", "k", "j", "i"),
            voxel_to_world=affine,
        )


# ---------------------------------------------------------------------------
# 5. 3D volumes + pose + time
# ---------------------------------------------------------------------------


def test_create_voxeldata_accepts_2d_per_pose_time(
    identity_pose_affines: Callable[[int], npt.NDArray[np.float64]],
):
    """A 2D (n_time, npose) time array gives each pose its own real timestamps."""
    npose = 3
    n_time = 4
    time_2d = np.stack(
        [np.arange(n_time) * 2.4 + p * 0.6 for p in range(npose)], axis=1
    )

    result = create_voxeldata(
        np.zeros((n_time, npose, 2, 3, 4)),
        dims=("time", "pose", "k", "j", "i"),
        time=xr.DataArray(time_2d, attrs={"units": "s"}),
        pose=np.arange(npose),
        voxel_to_world=identity_pose_affines(npose),
    )

    assert result.coords["time"].dims == ("time", "pose")
    assert_allclose(result.coords["time"].values, time_2d)
    assert result.coords["time"].attrs["units"] == "s"
    assert "time" not in result.xindexes
    # spacing along "time" specifically (not the cross-pose 0.6 offset).
    assert result.fusi.spacing["time"] == pytest.approx(2.4)


def test_create_voxeldata_2d_time_plain_array_gets_valid_attrs(
    identity_pose_affines: Callable[[int], npt.NDArray[np.float64]],
):
    """A plain (non-DataArray) 2D `time` array still gets required timing attrs."""
    npose = 4
    # Poses 0 and 1 simultaneous (tie); the rest 0.15s apart.
    time_2d = np.array(
        [
            [0.0, 0.0, 0.15, 0.3],
            [0.6, 0.6, 0.75, 0.9],
        ]
    )

    result = create_voxeldata(
        np.zeros((2, npose, 2, 3, 4)),
        dims=("time", "pose", "k", "j", "i"),
        time=time_2d,
        pose=np.arange(npose),
        voxel_to_world=identity_pose_affines(npose),
    )

    assert result.coords["time"].attrs["units"] == "s"
    assert result.coords["time"].attrs["volume_acquisition_reference"] == "start"
    assert result.coords["time"].attrs["volume_acquisition_duration"] == 0.15
    validate_voxeldata(result, allow_pose=True)


def test_create_voxeldata_2d_time_dataarray_attrs_take_priority(
    identity_pose_affines: Callable[[int], npt.NDArray[np.float64]],
):
    """An `xr.DataArray` `time`'s own attrs override the constructor defaults."""
    npose = 2
    time_2d = xr.DataArray(
        np.array([[0.0, 0.1], [0.6, 0.7]]),
        dims=("time", "pose"),
        attrs={
            "units": "s",
            "volume_acquisition_reference": "end",
            "volume_acquisition_duration": 0.05,
        },
    )

    result = create_voxeldata(
        np.zeros((2, npose, 2, 3, 4)),
        dims=("time", "pose", "k", "j", "i"),
        time=time_2d,
        pose=np.arange(npose),
        voxel_to_world=identity_pose_affines(npose),
    )

    assert result.coords["time"].attrs["volume_acquisition_reference"] == "end"
    assert result.coords["time"].attrs["volume_acquisition_duration"] == 0.05


def test_create_voxeldata_2d_time_requires_pose_dim():
    """A 2D time array without a `pose` dim in `dims` raises clearly."""
    time_2d = np.zeros((4, 2))

    with pytest.raises(ValueError, match="requires a 'pose' dimension"):
        create_voxeldata(
            np.zeros((4, 2, 3, 4)),
            dims=("time", "k", "j", "i"),
            time=time_2d,
            spacing=(1.0, 1.0, 1.0),
        )


def test_create_voxeldata_2d_time_rejects_wrong_pose_count(
    identity_pose_affines: Callable[[int], npt.NDArray[np.float64]],
):
    """A 2D time array's pose column count must match the `pose` dimension size."""
    npose = 3
    time_2d = np.zeros((4, 2))

    with pytest.raises(ValueError, match="pose columns"):
        create_voxeldata(
            np.zeros((4, npose, 2, 3, 4)),
            dims=("time", "pose", "k", "j", "i"),
            time=time_2d,
            pose=np.arange(npose),
            voxel_to_world=identity_pose_affines(npose),
        )


def test_create_voxeldata_accepts_per_pose_t0_with_shared_dt(
    identity_pose_affines: Callable[[int], npt.NDArray[np.float64]],
):
    """A 1D `t0` array generates pose-dependent `(time, pose)` timestamps."""
    npose = 3
    n_time = 4
    t0 = np.array([0.0, 0.6, 1.2])

    result = create_voxeldata(
        np.zeros((n_time, npose, 2, 3, 4)),
        dims=("time", "pose", "k", "j", "i"),
        dt=2.4,
        t0=t0,
        pose=np.arange(npose),
        voxel_to_world=identity_pose_affines(npose),
    )

    expected = np.arange(n_time)[:, None] * 2.4 + t0[None, :]
    assert result.coords["time"].dims == ("time", "pose")
    assert_allclose(result.coords["time"].values, expected)
    assert result.coords["time"].attrs["units"] == "s"
    # Defaults from the spacing between poses' own onsets (0.6), not `dt` (2.4,
    # the repetition period between successive samples of the *same* pose).
    assert result.coords["time"].attrs["volume_acquisition_duration"] == 0.6
    assert "time" not in result.xindexes
    assert result.fusi.spacing["time"] == pytest.approx(2.4)


def test_create_voxeldata_per_pose_t0_ignores_simultaneous_ties(
    identity_pose_affines: Callable[[int], npt.NDArray[np.float64]],
):
    """Poses sharing the same `t0` don't collapse the default duration to zero."""
    npose = 4

    result = create_voxeldata(
        np.zeros((4, npose, 2, 3, 4)),
        dims=("time", "pose", "k", "j", "i"),
        dt=0.6,
        t0=np.array([0.0, 0.0, 0.2, 0.4]),
        pose=np.arange(npose),
        voxel_to_world=identity_pose_affines(npose),
    )

    assert result.coords["time"].attrs["volume_acquisition_duration"] == 0.2


def test_create_voxeldata_per_pose_t0_honors_explicit_duration(
    identity_pose_affines: Callable[[int], npt.NDArray[np.float64]],
):
    """An explicit `volume_acquisition_duration` overrides the per-pose default."""
    npose = 3

    result = create_voxeldata(
        np.zeros((4, npose, 2, 3, 4)),
        dims=("time", "pose", "k", "j", "i"),
        dt=2.4,
        t0=np.array([0.0, 0.6, 1.2]),
        volume_acquisition_duration=0.1,
        pose=np.arange(npose),
        voxel_to_world=identity_pose_affines(npose),
    )

    assert result.coords["time"].attrs["volume_acquisition_duration"] == 0.1


def test_create_voxeldata_per_pose_t0_requires_pose_dim():
    """A 1D `t0` array without a `pose` dim in `dims` raises clearly."""
    with pytest.raises(ValueError, match="requires a 'pose' dimension"):
        create_voxeldata(
            np.zeros((4, 2, 3, 4)),
            dims=("time", "k", "j", "i"),
            dt=2.4,
            t0=np.array([0.0, 0.6]),
            spacing=(1.0, 1.0, 1.0),
        )


def test_create_voxeldata_per_pose_t0_requires_matching_pose_count():
    """A 1D `t0` array length must match the `pose` dimension size."""
    with pytest.raises(ValueError, match="t0 has length 2"):
        create_voxeldata(
            np.zeros((4, 3, 2, 3, 4)),
            dims=("time", "pose", "k", "j", "i"),
            dt=2.4,
            t0=np.array([0.0, 0.6]),
            spacing=(1.0, 1.0, 1.0),
        )


def test_create_voxeldata_per_pose_t0_requires_dt():
    """A 1D `t0` array still needs a shared `dt` to build `time`."""
    with pytest.raises(ValueError, match="Spacing for dimension 'time' is required"):
        create_voxeldata(
            np.zeros((4, 2, 2, 3, 4)),
            dims=("time", "pose", "k", "j", "i"),
            t0=np.array([0.0, 0.6]),
            spacing=(1.0, 1.0, 1.0),
        )


# ---------------------------------------------------------------------------
# 6. Extra (non-core) dims
# ---------------------------------------------------------------------------


def test_create_voxeldata_accepts_extra_dim_with_coords():
    """A non-core dim (e.g. PCA `component`) keeps its own coordinate, reordered first."""
    result = create_voxeldata(
        np.zeros((5, 3, 1, 8, 12)),
        dims=("time", "component", "k", "j", "i"),
        extra_coords={"component": ["a", "b", "c"]},
        dt=0.5,
        spacing=(0.4, 0.1, 0.2),
    )

    assert result.dims == ("component", "time", "k", "j", "i")
    np.testing.assert_array_equal(result.coords["component"], ["a", "b", "c"])
    validate_voxeldata(result, require_time=True)


def test_create_voxeldata_accepts_extra_dim_without_explicit_coords():
    """A non-core dim without an explicit coordinate still gets dense indices."""
    result = create_voxeldata(
        np.zeros((4, 1, 2, 3)),
        dims=("channel", "k", "j", "i"),
        spacing=(1.0, 1.0, 1.0),
    )

    assert result.dims == ("channel", "k", "j", "i")
    np.testing.assert_array_equal(result.coords["channel"], np.arange(4))


def test_create_voxeldata_accepts_extra_dim_with_pose_and_time(
    identity_pose_affines: Callable[[int], npt.NDArray[np.float64]],
):
    """An extra dim combines with `pose`, `time`, and custom native voxel
    coordinates all at once."""
    npose = 2
    n_time = 3
    n_component = 2

    result = create_voxeldata(
        np.zeros((n_component, n_time, npose, 1, 2, 3)),
        dims=("component", "time", "pose", "k", "j", "i"),
        extra_coords={"component": ["a", "b"]},
        dt=0.5,
        t0=[0.0, 0.2],
        pose=np.arange(npose),
        k=[5],
        j=[10, 12],
        i=[20, 23, 26],
        voxel_to_world=identity_pose_affines(npose),
    )

    assert result.dims == ("component", "time", "pose", "k", "j", "i")
    np.testing.assert_array_equal(result.coords["component"], ["a", "b"])
    assert result.coords["time"].dims == ("time", "pose")
    np.testing.assert_array_equal(result.coords["k"], [5])
    np.testing.assert_array_equal(result.coords["j"], [10, 12])
    np.testing.assert_array_equal(result.coords["i"], [20, 23, 26])
    validate_voxeldata(result, require_time=True)


def test_create_voxeldata_rejects_spatial_extra_coords():
    """Spatial coordinates must come from voxel-to-world geometry, not constructor coord arrays."""
    with pytest.raises(ValueError, match="extra_coords must not include core"):
        create_voxeldata(
            np.zeros((2, 3)),
            dims=("j", "i"),
            extra_coords={"x": np.arange(3)},
            spacing=(1.0, 0.1, 0.2),
        )


# ---------------------------------------------------------------------------
# 7. Geometry/coordinate/timing input validation (error paths)
# ---------------------------------------------------------------------------


def test_create_voxeldata_rejects_mixed_geometry_inputs():
    """The affine and decomposed geometry inputs are mutually exclusive."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        create_voxeldata(
            np.zeros((2, 3)),
            dims=("j", "i"),
            spacing=(1.0, 0.1, 0.2),
            voxel_to_world=np.eye(4),
        )


def test_create_voxeldata_rejects_missing_geometry():
    """Spatial geometry is mandatory."""
    with pytest.raises(ValueError, match="spacing or voxel_to_world must be provided"):
        create_voxeldata(np.zeros((2, 3)), dims=("j", "i"))


def test_create_voxeldata_rejects_wrong_length_explicit_coord():
    """An explicit coordinate whose length disagrees with the axis size is rejected."""
    with pytest.raises(ValueError, match=r"Coordinate 'time' must be 1D with length 5"):
        create_voxeldata(
            np.zeros((5, 1, 2, 3)),
            dims=("time", "k", "j", "i"),
            time=np.arange(3) * 0.1,
            spacing=(0.4, 0.1, 0.2),
        )


def test_create_voxeldata_rejects_none_t0_with_dt():
    """An explicit `t0=None` cannot serve as the time-coordinate origin."""
    with pytest.raises(
        ValueError, match="Origin for dimension 'time' must be provided"
    ):
        create_voxeldata(
            np.zeros((5, 1, 2, 3)),
            dims=("time", "k", "j", "i"),
            dt=0.5,
            t0=None,  # ty: ignore[invalid-argument-type]
            spacing=(0.4, 0.1, 0.2),
        )


def test_create_voxeldata_singleton_time_requires_dt():
    """A singleton time dimension without `dt` or an explicit coordinate raises."""
    with pytest.raises(ValueError, match="Spacing for dimension 'time' is required"):
        create_voxeldata(
            np.zeros((1, 1, 2, 3)),
            dims=("time", "k", "j", "i"),
            t0=3.0,
            spacing=(0.4, 0.1, 0.2),
        )


def test_create_voxeldata_rejects_wrong_length_spacing():
    """`spacing` must contain exactly 3 values, one per z/y/x axis."""
    with pytest.raises(ValueError, match="spacing must have length 3 in z/y/x order"):
        create_voxeldata(
            np.zeros((2, 3)),
            dims=("j", "i"),
            spacing=(1.0, 0.2),
        )


def test_create_voxeldata_rejects_wrong_shape_voxel_to_world():
    """`voxel_to_world` must be a 4x4 matrix."""
    with pytest.raises(ValueError, match=r"voxel_to_world must have shape \(4, 4\)"):
        create_voxeldata(
            np.zeros((2, 3)),
            dims=("j", "i"),
            voxel_to_world=np.eye(3),
        )


def test_create_voxeldata_rejects_non_homogeneous_voxel_to_world():
    """`voxel_to_world`'s last row must be `[0, 0, 0, 1]`."""
    affine = np.eye(4)
    affine[3, 3] = 2.0
    with pytest.raises(ValueError, match="voxel_to_world must be a homogeneous affine"):
        create_voxeldata(
            np.zeros((2, 3)),
            dims=("j", "i"),
            voxel_to_world=affine,
        )


def test_create_voxeldata_rejects_wrong_length_origin():
    """`origin` must contain exactly 3 values, one per z/y/x axis."""
    with pytest.raises(ValueError, match="origin must have length 3 in z/y/x order"):
        create_voxeldata(
            np.zeros((2, 3)),
            dims=("j", "i"),
            spacing=(1.0, 0.1, 0.2),
            origin=(0.0, 0.0),
        )


def test_create_voxeldata_rejects_non_finite_origin():
    """`origin` values must all be finite."""
    with pytest.raises(ValueError, match="origin must contain finite values"):
        create_voxeldata(
            np.zeros((2, 3)),
            dims=("j", "i"),
            spacing=(1.0, 0.1, 0.2),
            origin=(np.nan, 0.0, 0.0),
        )


def test_create_voxeldata_rejects_wrong_shape_direction():
    """`direction` must be a 3x3 matrix."""
    with pytest.raises(ValueError, match=r"direction must have shape \(3, 3\)"):
        create_voxeldata(
            np.zeros((2, 3)),
            dims=("j", "i"),
            spacing=(1.0, 0.1, 0.2),
            direction=np.eye(2),
        )


def test_create_voxeldata_rejects_non_finite_direction():
    """`direction` values must all be finite."""
    direction = np.eye(3)
    direction[0, 1] = np.inf
    with pytest.raises(ValueError, match="direction must contain finite values"):
        create_voxeldata(
            np.zeros((2, 3)),
            dims=("j", "i"),
            spacing=(1.0, 0.1, 0.2),
            direction=direction,
        )


def test_create_voxeldata_rejects_invalid_volume_acquisition_reference():
    """`volume_acquisition_reference` must be one of the recognized timing references."""
    with pytest.raises(ValueError, match="volume_acquisition_reference must be one of"):
        create_voxeldata(
            np.zeros((5, 1, 2, 3)),
            dims=("time", "k", "j", "i"),
            dt=0.5,
            spacing=(0.4, 0.1, 0.2),
            volume_acquisition_reference="middle",  # ty: ignore[invalid-argument-type]
        )


def test_create_voxeldata_rejects_duration_without_time_dim():
    """`volume_acquisition_duration` requires a `time` dimension to attach to."""
    with pytest.raises(
        ValueError,
        match="time and volume_acquisition_duration require a 'time' dimension",
    ):
        create_voxeldata(
            np.zeros((2, 3)),
            dims=("j", "i"),
            spacing=(1.0, 0.1, 0.2),
            volume_acquisition_duration=1.0,
        )


def test_create_voxeldata_rejects_world_dim_names():
    """`dims` must use native voxel names; world z/y/x names are rejected."""
    with pytest.raises(ValueError, match="dims must use native voxel names"):
        create_voxeldata(
            np.zeros((2, 3, 4)),
            dims=("z", "y", "x"),
            spacing=(0.4, 0.1, 0.2),
        )

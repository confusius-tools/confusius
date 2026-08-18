"""Unit tests for confusius.multipose.stack_poses."""

import numpy as np
import pytest
import xarray as xr

from confusius._utils.geometry import get_voxel_to_world_affine
from confusius.multipose import stack_poses
from confusius.xarray import create_fusi_dataarray


def _make_pose(affine: np.ndarray, *, time: np.ndarray | None = None) -> xr.DataArray:
    """Build one single-pose VoxelData DataArray."""
    shape = (2, 3, 4) if time is None else (len(time), 2, 3, 4)
    dims = ("k", "j", "i") if time is None else ("time", "k", "j", "i")
    return create_fusi_dataarray(
        np.zeros(shape),
        dims=dims,
        time=time,
        voxel_to_world=affine,
    )


class TestStackPoses:
    """Tests for stack_poses."""

    def test_stacks_geometry_and_data(self):
        """Each pose's affine and data are preserved as one entry of the stack."""
        rng = np.random.default_rng(0)
        affines = [
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
        poses = []
        for affine in affines:
            da = _make_pose(affine)
            da.values[:] = rng.random(da.shape)
            poses.append(da)

        result = stack_poses(poses)

        assert result.dims == ("pose", "k", "j", "i")
        np.testing.assert_array_equal(result.coords["pose"].values, [0, 1])
        np.testing.assert_allclose(
            get_voxel_to_world_affine(result), np.stack(affines)
        )
        for i, da in enumerate(poses):
            np.testing.assert_array_equal(result.isel(pose=i).values, da.values)

    def test_custom_pose_labels(self):
        """Explicit `pose` labels are used instead of the default range."""
        poses = [_make_pose(np.eye(4)), _make_pose(np.eye(4))]

        result = stack_poses(poses, pose=[10, 20])

        np.testing.assert_array_equal(result.coords["pose"].values, [10, 20])

    def test_shared_time_stays_one_dimensional(self):
        """Identical per-pose time values keep `time` as a plain 1D coordinate."""
        time = np.arange(5) * 2.4
        poses = [
            _make_pose(np.eye(4), time=time),
            _make_pose(np.eye(4), time=time),
        ]

        result = stack_poses(poses)

        assert result.coords["time"].dims == ("time",)
        assert "time" in result.xindexes
        np.testing.assert_allclose(result.coords["time"].values, time)

    def test_differing_time_becomes_2d_and_requires_pose_selection(self):
        """Per-pose time differences produce a (time, pose) `time` coordinate.

        `.sel(time=...)` is unavailable until a pose is selected (no PandasIndex on a
        2D coordinate); after `.set_xindex("time")` on a pose-selected result, it
        becomes selectable again with that pose's own real timestamps.
        """
        time0 = np.arange(5) * 2.4
        time1 = time0 + 0.6
        poses = [
            _make_pose(np.eye(4), time=time0),
            _make_pose(
                np.array(
                    [
                        [1.0, 0.0, 0.0, 100.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
                time=time1,
            ),
        ]

        result = stack_poses(poses)

        assert result.coords["time"].dims == ("time", "pose")
        assert "time" not in result.xindexes
        np.testing.assert_allclose(result.coords["time"].isel(pose=0).values, time0)
        np.testing.assert_allclose(result.coords["time"].isel(pose=1).values, time1)

        promoted = result.isel(pose=1).set_xindex("time")
        assert "time" in promoted.xindexes
        selected = promoted.sel(time=time1[2], method="nearest")
        assert selected.coords["time"].item() == pytest.approx(time1[2])

    def test_empty_poses_raises(self):
        """An empty poses sequence raises ValueError."""
        with pytest.raises(ValueError, match="at least one"):
            stack_poses([])

    def test_pose_label_length_mismatch_raises(self):
        """A mismatched pose label count raises ValueError."""
        poses = [_make_pose(np.eye(4)), _make_pose(np.eye(4))]
        with pytest.raises(ValueError, match="one label per entry"):
            stack_poses(poses, pose=[0])

    def test_mismatched_time_length_raises(self):
        """Poses with different time lengths raise ValueError."""
        poses = [
            _make_pose(np.eye(4), time=np.arange(5) * 2.4),
            _make_pose(np.eye(4), time=np.arange(3) * 2.4),
        ]
        with pytest.raises(ValueError, match="same 'time' length"):
            stack_poses(poses)

    def test_rejects_pose_already_present(self):
        """A pose already carrying a `pose` dimension raises via allow_pose=False."""
        already_stacked = stack_poses(
            [_make_pose(np.eye(4)), _make_pose(np.eye(4))]
        )
        with pytest.raises(ValueError, match="pose"):
            stack_poses([already_stacked])

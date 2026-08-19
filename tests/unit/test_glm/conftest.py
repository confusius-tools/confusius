"""Shared fixtures for GLM tests.

Mirrors the conventions of the project-wide
[`sample_fusi_3dt`][tests.unit.conftest.sample_fusi_3dt] fixture (mm spatial
coordinates, units/attrs metadata) but uses the longer time series GLM model
fitting needs to estimate conditions plus drift cleanly.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from confusius._utils.geometry import (
    attach_voxel_to_world_index,
    get_voxel_to_world_affine,
)
from confusius.xarray import create_voxeldata


@pytest.fixture
def frame_times():
    """200 uniformly spaced volume times at 2 Hz (dt=0.5 s)."""
    return np.arange(200) * 0.5


@pytest.fixture
def events():
    """Two-condition event table interleaving A and B every 10 s."""
    onsets_a = np.arange(5) * 20.0
    onsets_b = np.arange(5) * 20.0 + 10.0
    return pd.DataFrame(
        {
            "trial_type": ["A"] * 5 + ["B"] * 5,
            "onset": np.concatenate([onsets_a, onsets_b]),
            "duration": [2.0] * 10,
        }
    )


@pytest.fixture
def make_glm_test_dataarray():
    """Build valid fUSI arrays for GLM tests without constructor helper."""

    def factory(data, dims, *, time=None):
        shape = np.shape(data)
        coord_time = None
        if "time" in dims:
            n_time = shape[dims.index("time")]
            coord_time = xr.DataArray(
                np.arange(n_time) if time is None else time,
                dims=("time",),
                attrs={"units": "s"},
            )
        return create_voxeldata(
            data,
            dims=dims,
            time=coord_time,
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
        )

    return factory


@pytest.fixture
def fusi_data(rng, frame_times):
    """Small `(time, k, j, i)` VoxelData array."""
    n_time = len(frame_times)
    return create_voxeldata(
        rng.standard_normal((n_time, 2, 3, 4)),
        dims=("time", "k", "j", "i"),
        time=frame_times,
        spacing=(0.5, 0.1, 0.1),
        origin=(0.0, 0.0, 0.0),
    )


@pytest.fixture
def fusi_data_2d(rng, frame_times):
    """Small `(time, 1, j, i)` VoxelData array."""
    n_time = len(frame_times)
    return create_voxeldata(
        rng.standard_normal((n_time, 5, 6)),
        dims=("time", "j", "i"),
        time=frame_times,
        spacing=(1.0, 0.1, 0.1),
        origin=(0.0, 0.0, 0.0),
    )


@pytest.fixture
def fusi_data_pose(rng, frame_times):
    """Small `(time, pose, k, j, i)` DataArray with two poses.

    Spatial shape matches `fusi_data`; poses share the same voxel-to-world
    affine, so results should match fitting each pose slice independently.
    """
    n_time = len(frame_times)
    n_pose = 2
    affine = np.stack([np.eye(4) for _ in range(n_pose)])
    return create_voxeldata(
        rng.standard_normal((n_time, n_pose, 2, 3, 4)),
        dims=("time", "pose", "k", "j", "i"),
        time=frame_times,
        pose=np.arange(n_pose),
        voxel_to_world=affine,
    )


@pytest.fixture
def spatial_maps(rng):
    """10 spatial maps of shape `(2, 3, 4)` for group-level tests."""
    maps = []
    for _ in range(10):
        maps.append(
            create_voxeldata(
                rng.standard_normal((2, 3, 4)),
                dims=("k", "j", "i"),
                spacing=(0.5, 0.1, 0.1),
                origin=(0.0, 0.0, 0.0),
            )
        )
    return maps


@pytest.fixture
def spatial_maps_with_mismatched_k(spatial_maps):
    """Spatial maps where the second map has mismatched native `k` coordinates."""
    second = spatial_maps[1]
    bad = attach_voxel_to_world_index(
        xr.DataArray(
            second.data,
            dims=second.dims,
            coords={
                "k": second.coords["k"].values + 10,
                "j": second.coords["j"].values,
                "i": second.coords["i"].values,
            },
        ),
        get_voxel_to_world_affine(second),
    )
    return [spatial_maps[0], bad]


@pytest.fixture
def spatial_maps_2d(rng):
    """8 spatial maps of shape `(5, 6)` (no `z` axis)."""
    maps = []
    for _ in range(8):
        maps.append(
            create_voxeldata(
                rng.standard_normal((5, 6)),
                dims=("j", "i"),
                spacing=(1.0, 0.1, 0.1),
                origin=(0.0, 0.0, 0.0),
            )
        )
    return maps

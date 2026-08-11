"""Shared fixtures for GLM tests.

Mirrors the conventions of the project-wide
[`sample_3dt_volume`][tests.unit.conftest.sample_3dt_volume] fixture (mm spatial
coordinates, units/attrs metadata) but uses the longer time series GLM model
fitting needs to estimate conditions plus drift cleanly.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from confusius._utils.geometry import add_physical_coords_from_voxel_affine


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
    """Build valid CTI-backed fUSI arrays for GLM tests without constructor helper."""

    def factory(data, dims, *, time=None):
        shape = np.shape(data)
        coords = {}
        if "time" in dims:
            n_time = shape[dims.index("time")]
            coords["time"] = xr.DataArray(
                np.arange(n_time) if time is None else time,
                dims=("time",),
                attrs={"units": "s"},
            )
        for dim in ("k", "j", "i"):
            if dim in dims:
                coords[dim] = np.arange(shape[dims.index(dim)])
        da = xr.DataArray(data, dims=dims, coords=coords)
        voxel_dims = tuple(dim for dim in dims if dim in {"k", "j", "i"})
        physical_names = tuple(
            {"k": "z", "j": "y", "i": "x"}[dim] for dim in voxel_dims
        )
        return add_physical_coords_from_voxel_affine(
            da,
            np.eye(len(voxel_dims) + 1),
            voxel_dims=voxel_dims,
            physical_coord_names=physical_names,
            physical_coord_attrs={
                name: {"units": "mm", "voxdim": 1.0} for name in physical_names
            },
        )

    return factory


@pytest.fixture
def fusi_data(rng, frame_times):
    """Small `(time, k, j, i)` DataArray with mm spatial coordinates."""
    n_time = len(frame_times)
    data = xr.DataArray(
        rng.standard_normal((n_time, 2, 3, 4)),
        dims=["time", "k", "j", "i"],
        coords={
            "time": frame_times,
            "k": np.arange(2),
            "j": np.arange(3),
            "i": np.arange(4),
        },
    )
    return add_physical_coords_from_voxel_affine(
        data,
        np.diag([0.5, 0.1, 0.1, 1.0]),
        voxel_dims=("k", "j", "i"),
        physical_coord_names=("z", "y", "x"),
    )


@pytest.fixture
def fusi_data_2d(rng, frame_times):
    """Small `(time, j, i)` DataArray (no `k` axis)."""
    n_time = len(frame_times)
    data = xr.DataArray(
        rng.standard_normal((n_time, 5, 6)),
        dims=["time", "j", "i"],
        coords={
            "time": frame_times,
            "j": np.arange(5),
            "i": np.arange(6),
        },
    )
    return add_physical_coords_from_voxel_affine(
        data,
        np.diag([0.1, 0.1, 1.0]),
        voxel_dims=("j", "i"),
        physical_coord_names=("y", "x"),
    )


@pytest.fixture
def spatial_maps(rng):
    """10 spatial maps of shape `(2, 3, 4)` for group-level tests."""
    maps = []
    for _ in range(10):
        da = xr.DataArray(
            rng.standard_normal((2, 3, 4)),
            dims=["k", "j", "i"],
            coords={"k": np.arange(2), "j": np.arange(3), "i": np.arange(4)},
        )
        maps.append(
            add_physical_coords_from_voxel_affine(
                da,
                np.diag([0.5, 0.1, 0.1, 1.0]),
                voxel_dims=("k", "j", "i"),
                physical_coord_names=("z", "y", "x"),
            )
        )
    return maps


@pytest.fixture
def spatial_maps_2d(rng):
    """8 spatial maps of shape `(5, 6)` (no `z` axis)."""
    maps = []
    for _ in range(8):
        da = xr.DataArray(
            rng.standard_normal((5, 6)),
            dims=["j", "i"],
            coords={"j": np.arange(5), "i": np.arange(6)},
        )
        maps.append(
            add_physical_coords_from_voxel_affine(
                da,
                np.diag([0.1, 0.1, 1.0]),
                voxel_dims=("j", "i"),
                physical_coord_names=("y", "x"),
            )
        )
    return maps

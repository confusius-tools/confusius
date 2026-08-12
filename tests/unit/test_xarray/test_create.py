"""Tests for canonical xarray constructor helpers."""

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
        dims=("time", "z", "y", "x"),
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
    assert result.coords["z"].attrs == {"units": "mm", "voxdim": 0.4}
    validate_fusi(result, require_time=True)


def test_create_fusi_dataarray_uses_default_probe_origins():
    """Default origins match the probe-centered z/x and surface-referenced y model."""
    result = create_fusi_dataarray(
        np.zeros((3, 8, 4)),
        dims=("z", "y", "x"),
        spacing=(0.4, 0.1, 0.2),
    )

    assert_allclose(_world_coord_1d(result, "z"), [-0.4, 0.0, 0.4])
    assert_allclose(_world_coord_1d(result, "y"), 0.05 + np.arange(8) * 0.1)
    assert_allclose(_world_coord_1d(result, "x"), [-0.3, -0.1, 0.1, 0.3])


def test_create_fusi_dataarray_pads_missing_spatial_dim():
    """A 2D input gets singleton axes for missing spatial dims."""
    result = create_fusi_dataarray(
        np.zeros((5, 8, 12)),
        dims=("time", "y", "x"),
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
        dims=("z", "y", "x"),
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
        dims=("z", "y", "x"),
        voxel_to_world=affine,
    )

    assert_allclose(result.fusi.affine.voxel_to_world, affine)
    assert result.coords["x"].attrs["voxdim"] == pytest.approx(0.2)


def test_create_fusi_dataarray_voxdim_overrides_metadata_only():
    """`voxdim` sets coordinate attrs without changing the affine."""
    result = create_fusi_dataarray(
        np.zeros((1, 2, 3)),
        dims=("z", "y", "x"),
        spacing=(0.4, 0.1, 0.2),
        voxdim=(1.0, 1.0, 1.0),
    )

    assert result.coords["z"].attrs["voxdim"] == pytest.approx(1.0)
    assert_allclose(result.fusi.affine.voxel_to_world[:3, :3], np.diag([0.4, 0.1, 0.2]))


def test_create_fusi_dataarray_rejects_spatial_extra_coords():
    """Spatial coordinates must come from voxel-to-world geometry, not constructor coord arrays."""
    with pytest.raises(ValueError, match="extra_coords must not include core"):
        create_fusi_dataarray(
            np.zeros((2, 3)),
            dims=("y", "x"),
            extra_coords={"x": np.arange(3)},
            spacing=(1.0, 0.1, 0.2),
        )


def test_create_fusi_dataarray_rejects_mixed_geometry_inputs():
    """The affine and decomposed geometry inputs are mutually exclusive."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        create_fusi_dataarray(
            np.zeros((2, 3)),
            dims=("y", "x"),
            spacing=(1.0, 0.1, 0.2),
            voxel_to_world=np.eye(4),
        )


def test_create_fusi_dataarray_rejects_missing_geometry():
    """Spatial geometry is mandatory."""
    with pytest.raises(ValueError, match="spacing must be provided"):
        create_fusi_dataarray(np.zeros((2, 3)), dims=("y", "x"))


def test_create_iq_dataarray_validates_complex_input():
    """IQ constructor delegates geometry and enforces complex data."""
    result = create_iq_dataarray(
        np.ones((5, 1, 2, 3), dtype=np.complex64),
        dims=("time", "z", "y", "x"),
        time=xr.DataArray(np.arange(5) * 0.1, dims=("time",), attrs={"units": "s"}),
        spacing=(0.4, 0.1, 0.2),
    )

    assert result.name == "iq"
    validate_iq(result)

    with pytest.raises(TypeError, match="complex"):
        create_iq_dataarray(
            np.ones((5, 1, 2, 3)),
            dims=("time", "z", "y", "x"),
            time=np.arange(5) * 0.1,
            spacing=(0.4, 0.1, 0.2),
        )

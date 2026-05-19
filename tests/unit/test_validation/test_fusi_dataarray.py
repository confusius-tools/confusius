"""Tests for generic ConfUSIus DataArray validation."""

import numpy as np
import pytest
import xarray as xr

from confusius.validation import validate_fusi_dataarray


@pytest.fixture
def valid_2dt_dataarray() -> xr.DataArray:
    """Return a minimal valid 2D+t ConfUSIus DataArray."""
    return xr.DataArray(
        np.zeros((5, 3, 4), dtype=np.float32),
        dims=("time", "y", "x"),
        coords={
            "time": xr.DataArray(np.arange(5) * 0.5, dims=["time"], attrs={"units": "s"}),
            "y": xr.DataArray(
                np.array([1.0, 1.5, 2.0]),
                dims=["y"],
                attrs={"units": "mm", "voxdim": 0.5},
            ),
            "x": xr.DataArray(
                np.array([-1.0, -0.5, 0.0, 0.5]),
                dims=["x"],
                attrs={"units": "mm", "voxdim": 0.5},
            ),
        },
    )


def test_validate_fusi_dataarray_accepts_valid_2dt(
    valid_2dt_dataarray: xr.DataArray,
) -> None:
    """A canonical 2D+t DataArray validates successfully."""
    validate_fusi_dataarray(valid_2dt_dataarray)


def test_validate_fusi_dataarray_rejects_non_dataarray() -> None:
    """Non-DataArray inputs raise `TypeError`."""
    with pytest.raises(TypeError, match="xarray.DataArray"):
        validate_fusi_dataarray(np.zeros((2, 2)))  # type: ignore[arg-type]


def test_validate_fusi_dataarray_allows_extra_dims_by_default(
    valid_2dt_dataarray: xr.DataArray,
) -> None:
    """Extra non-core dimensions are allowed by default."""
    data = valid_2dt_dataarray.expand_dims(region=["roi"])

    validate_fusi_dataarray(data)


def test_validate_fusi_dataarray_can_forbid_extra_dims(
    valid_2dt_dataarray: xr.DataArray,
) -> None:
    """Extra non-core dimensions can be rejected explicitly."""
    data = valid_2dt_dataarray.expand_dims(region=["roi"])

    with pytest.raises(ValueError, match="Unexpected dimensions"):
        validate_fusi_dataarray(data, allow_extra_dims=False)


def test_validate_fusi_dataarray_requires_minimum_spatial_dims(
    valid_2dt_dataarray: xr.DataArray,
) -> None:
    """At least the configured number of spatial dimensions must be present."""
    bad = valid_2dt_dataarray.isel(y=0, drop=True)

    with pytest.raises(ValueError, match="at least 2 spatial dimensions"):
        validate_fusi_dataarray(bad)


def test_validate_fusi_dataarray_can_require_time(
    valid_2dt_dataarray: xr.DataArray,
) -> None:
    """`require_time=True` rejects arrays without a time dimension."""
    spatial = valid_2dt_dataarray.isel(time=0, drop=True)

    with pytest.raises(ValueError, match="must have a 'time' dimension"):
        validate_fusi_dataarray(spatial, require_time=True)


def test_validate_fusi_dataarray_can_forbid_pose() -> None:
    """`allow_pose=False` rejects multi-pose arrays."""
    data = xr.DataArray(
        np.zeros((2, 3, 4), dtype=np.float32),
        dims=("pose", "y", "x"),
        coords={
            "pose": [0, 1],
            "y": xr.DataArray(
                np.arange(3),
                dims=["y"],
                attrs={"units": "mm", "voxdim": 1.0},
            ),
            "x": xr.DataArray(
                np.arange(4),
                dims=["x"],
                attrs={"units": "mm", "voxdim": 1.0},
            ),
        },
    )

    with pytest.raises(ValueError, match="must not have a 'pose' dimension"):
        validate_fusi_dataarray(data, allow_pose=False)


def test_validate_fusi_dataarray_rejects_missing_dimension_coordinate(
    valid_2dt_dataarray: xr.DataArray,
) -> None:
    """Every dimension must have a same-named coordinate."""
    bad = valid_2dt_dataarray.drop_vars("x")

    with pytest.raises(ValueError, match="Missing required coordinate"):
        validate_fusi_dataarray(bad)


def test_validate_fusi_dataarray_rejects_non_monotonic_numeric_coordinate(
    valid_2dt_dataarray: xr.DataArray,
) -> None:
    """Numeric dimension coordinates must be strictly increasing."""
    bad = valid_2dt_dataarray.assign_coords(y=[1.0, 2.0, 1.5])

    with pytest.raises(ValueError, match="strictly monotonic-increasing"):
        validate_fusi_dataarray(bad)


def test_validate_fusi_dataarray_can_require_canonical_dim_order(
    valid_2dt_dataarray: xr.DataArray,
) -> None:
    """Canonical core dimension order can be enforced explicitly."""
    reordered = valid_2dt_dataarray.transpose("y", "x", "time")

    with pytest.raises(ValueError, match="not in canonical ConfUSIus order"):
        validate_fusi_dataarray(reordered, require_canonical_dim_order=True)


def test_validate_fusi_dataarray_can_require_regular_spacing(
    valid_2dt_dataarray: xr.DataArray,
) -> None:
    """Regular-spacing mode rejects non-uniform numeric coordinates."""
    bad = valid_2dt_dataarray.assign_coords(time=[0.0, 0.5, 1.0, 1.7, 2.2])

    with pytest.raises(ValueError, match="must have regular spacing"):
        validate_fusi_dataarray(bad, require_regular_spacing=True)


def test_validate_fusi_dataarray_can_require_spatial_voxdim(
    valid_2dt_dataarray: xr.DataArray,
) -> None:
    """Spatial `voxdim` metadata can be enforced explicitly."""
    bad = valid_2dt_dataarray.copy(deep=True)
    del bad.coords["y"].attrs["voxdim"]

    with pytest.raises(ValueError, match="missing required 'voxdim' metadata"):
        validate_fusi_dataarray(bad, require_spatial_voxdim=True)


def test_validate_fusi_dataarray_can_require_coordinate_units(
    valid_2dt_dataarray: xr.DataArray,
) -> None:
    """Coordinate `units` metadata can be enforced explicitly."""
    bad = valid_2dt_dataarray.copy(deep=True)
    del bad.coords["time"].attrs["units"]

    with pytest.raises(ValueError, match="missing required 'units' metadata"):
        validate_fusi_dataarray(bad, require_time_units=True)

"""Tests for generic ConfUSIus DataArray validation."""

from typing import Any

import numpy as np
import pytest
import xarray as xr

from confusius._utils.geometry import add_physical_coords_from_voxel_affine
from confusius.validation import validate_fusi_dataarray


def _make_voxel_affine_volume() -> xr.DataArray:
    """Create a small ConfUSIus-style 3D volume."""
    base = xr.DataArray(
        np.zeros((2, 3, 4), dtype=np.float32),
        dims=("k", "j", "i"),
        coords={
            "k": xr.DataArray([0.0, 1.0], dims=("k",), attrs={"voxdim": 1.0}),
            "j": xr.DataArray([0.0, 2.0, 4.0], dims=("j",), attrs={"voxdim": 2.0}),
            "i": xr.DataArray(
                [0.0, 1.0, 2.0, 3.0], dims=("i",), attrs={"voxdim": 1.0}
            ),
        },
    )
    return add_physical_coords_from_voxel_affine(
        base,
        np.array(
            [
                [2.0, 0.0, 0.0, 10.0],
                [0.0, 3.0, 0.0, 20.0],
                [0.0, 0.0, 4.0, 30.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        voxel_dims=("k", "j", "i"),
        physical_coord_names=("z", "y", "x"),
        physical_coord_attrs={
            "z": {"units": "mm", "voxdim": 2.0},
            "y": {"units": "mm", "voxdim": 6.0},
            "x": {"units": "mm", "voxdim": 4.0},
        },
    )


def _make_voxel_affine_time_series() -> xr.DataArray:
    """Create a small ConfUSIus-style 3D+t volume."""
    volume = _make_voxel_affine_volume().expand_dims(time=6).copy()
    volume = volume.assign_coords(
        time=xr.DataArray(
            np.arange(6, dtype=float) * 0.5,
            dims=("time",),
            attrs={"units": "s"},
        )
    )
    return volume.transpose("time", "k", "j", "i")


def test_validate_fusi_dataarray_accepts_valid_3d() -> None:
    """A canonical 3D ConfUSIus volume validates successfully."""
    validate_fusi_dataarray(_make_voxel_affine_volume())


def test_validate_fusi_dataarray_accepts_valid_3dt() -> None:
    """A canonical 3D+t ConfUSIus volume validates successfully."""
    validate_fusi_dataarray(_make_voxel_affine_time_series())


def test_validate_fusi_dataarray_rejects_non_dataarray() -> None:
    """Non-DataArray inputs raise `TypeError`."""
    bad_data: Any = np.zeros((2, 2))

    with pytest.raises(TypeError, match="xarray.DataArray"):
        validate_fusi_dataarray(bad_data)


def test_validate_fusi_dataarray_rejects_plain_physical_grid() -> None:
    """Plain z/y/x dimension arrays are no longer a valid ConfUSIus model."""
    data = xr.DataArray(
        np.zeros((2, 3, 4), dtype=np.float32),
        dims=("z", "y", "x"),
        coords={"z": [0.0, 1.0], "y": [0.0, 1.0, 2.0], "x": [0.0, 1.0, 2.0, 3.0]},
    )

    with pytest.raises(ValueError, match="native voxel dimensions `k/j/i`"):
        validate_fusi_dataarray(data)


def test_validate_fusi_dataarray_rejects_voxel_affine_missing_physical_coord() -> None:
    """Voxel-to-physical geometry requires linked physical coordinates."""
    good = _make_voxel_affine_volume()
    bad = xr.DataArray(
        good.values,
        dims=good.dims,
        coords={dim: good.coords[dim] for dim in ("k", "j", "i")},
        attrs={"voxel_to_physical": good.attrs["voxel_to_physical"]},
    )

    with pytest.raises(ValueError, match="missing physical coordinate 'z'"):
        validate_fusi_dataarray(bad)


def test_validate_fusi_dataarray_allows_extra_dims_by_default() -> None:
    """Extra non-core dimensions are allowed by default."""
    data = _make_voxel_affine_time_series().expand_dims(region=["roi"])
    validate_fusi_dataarray(data)


def test_validate_fusi_dataarray_can_forbid_extra_dims() -> None:
    """Extra non-core dimensions can be rejected explicitly."""
    data = _make_voxel_affine_time_series().expand_dims(region=["roi"])

    with pytest.raises(ValueError, match="Unexpected dimensions"):
        validate_fusi_dataarray(data, allow_extra_dims=False)


def test_validate_fusi_dataarray_requires_minimum_spatial_dims() -> None:
    """At least the configured number of voxel spatial dimensions must be present."""
    base = xr.DataArray(
        np.zeros((3, 4), dtype=np.float32),
        dims=("j", "i"),
        coords={"j": [0.0, 2.0, 4.0], "i": [0.0, 1.0, 2.0, 3.0]},
    )
    bad = add_physical_coords_from_voxel_affine(
        base,
        np.array([[3.0, 0.0, 20.0], [0.0, 4.0, 30.0], [0.0, 0.0, 1.0]]),
        voxel_dims=("j", "i"),
        physical_coord_names=("y", "x"),
        physical_coord_attrs={"y": {"units": "mm"}, "x": {"units": "mm"}},
    )

    with pytest.raises(ValueError, match="at least 3 spatial dimensions"):
        validate_fusi_dataarray(bad, minimum_spatial_dims=3)


def test_validate_fusi_dataarray_can_require_time() -> None:
    """`require_time=True` rejects arrays without a time dimension."""
    spatial = _make_voxel_affine_time_series().isel(time=0, drop=True)

    with pytest.raises(ValueError, match="must have a 'time' dimension"):
        validate_fusi_dataarray(spatial, require_time=True)


def test_validate_fusi_dataarray_can_forbid_pose() -> None:
    """`allow_pose=False` rejects multi-pose arrays."""
    data = _make_voxel_affine_volume().expand_dims(pose=[0, 1])

    with pytest.raises(ValueError, match="must not have a 'pose' dimension"):
        validate_fusi_dataarray(data, allow_pose=False)


def test_validate_fusi_dataarray_rejects_missing_dimension_coordinate() -> None:
    """Every core dimension must have a same-named coordinate."""
    bad = _make_voxel_affine_time_series().drop_vars("i")

    with pytest.raises(ValueError, match="Missing required coordinate"):
        validate_fusi_dataarray(bad)


def test_validate_fusi_dataarray_allows_missing_extra_dimension_coordinate() -> None:
    """Missing extra-dimension coordinates are allowed."""
    bad = _make_voxel_affine_time_series().expand_dims(region=["roi"]).drop_vars("region")
    validate_fusi_dataarray(bad)


def test_validate_fusi_dataarray_rejects_non_numeric_core_coordinate() -> None:
    """Core dimension coordinates must be numeric."""
    n_i = _make_voxel_affine_time_series().sizes["i"]
    labels = np.array([f"v{i}" for i in range(n_i)], dtype=object)
    bad = _make_voxel_affine_time_series().assign_coords(i=xr.DataArray(labels, dims=("i",)))

    with pytest.raises(ValueError, match="must be numeric"):
        validate_fusi_dataarray(bad)


def test_validate_fusi_dataarray_can_require_canonical_dim_order() -> None:
    """Canonical core dimension order can be enforced explicitly."""
    reordered = _make_voxel_affine_time_series().transpose("j", "i", "time", "k")

    with pytest.raises(ValueError, match="not in canonical ConfUSIus order"):
        validate_fusi_dataarray(reordered, require_canonical_dim_order=True)


def test_validate_fusi_dataarray_can_require_regular_spacing() -> None:
    """Regular-spacing mode rejects non-uniform voxel coordinates."""
    bad = _make_voxel_affine_time_series().assign_coords(
        j=np.array([0.0, 2.5, 4.0], dtype=float)
    )

    with pytest.raises(ValueError, match="must have regular spacing"):
        validate_fusi_dataarray(bad, require_regular_spacing=True)


def test_validate_fusi_dataarray_regular_spacing_can_target_space_dims_only() -> None:
    """Space-only regular-spacing checks ignore irregular time sampling."""
    bad_time = _make_voxel_affine_time_series().assign_coords(
        time=np.array([0.0, 0.5, 1.0, 1.7, 2.2, 2.8], dtype=float)
    )

    validate_fusi_dataarray(
        bad_time,
        require_regular_spacing=True,
        regular_spacing_dims="space",
    )


def test_validate_fusi_dataarray_regular_spacing_core_checks_time_when_present() -> None:
    """`core` mode includes `time` and rejects irregular time spacing."""
    bad_time = _make_voxel_affine_time_series().assign_coords(
        time=np.array([0.0, 0.5, 1.0, 1.7, 2.2, 2.8], dtype=float)
    )

    with pytest.raises(ValueError, match="must have regular spacing"):
        validate_fusi_dataarray(
            bad_time,
            require_regular_spacing=True,
            regular_spacing_dims="core",
        )


def test_validate_fusi_dataarray_can_require_spatial_voxdim() -> None:
    """Spatial `voxdim` metadata can be enforced explicitly."""
    bad = _make_voxel_affine_time_series().copy(deep=True)
    del bad.coords["k"].attrs["voxdim"]

    with pytest.raises(ValueError, match="missing required 'voxdim' metadata"):
        validate_fusi_dataarray(bad, require_spatial_voxdim=True)


def test_validate_fusi_dataarray_can_require_coordinate_units() -> None:
    """Coordinate `units` metadata can be enforced explicitly."""
    bad = _make_voxel_affine_time_series().copy(deep=True)
    del bad.coords["time"].attrs["units"]

    with pytest.raises(ValueError, match="missing required 'units' metadata"):
        validate_fusi_dataarray(bad, require_time_units=True)


def test_validate_fusi_dataarray_can_require_spatial_units() -> None:
    """Spatial `units` metadata can be enforced explicitly."""
    bad = _make_voxel_affine_time_series().copy(deep=True)
    del bad.coords["x"].attrs["units"]

    with pytest.raises(ValueError, match="missing required 'units' metadata"):
        validate_fusi_dataarray(bad, require_spatial_units=True)


def test_validate_fusi_dataarray_rejects_non_finite_numeric_coordinate() -> None:
    """Numeric coordinates must be finite."""
    bad = _make_voxel_affine_time_series().assign_coords(
        time=np.array([0.0, 0.5, np.nan, 1.5, 2.0, 2.5], dtype=float)
    )

    with pytest.raises(ValueError, match="non-finite numeric values"):
        validate_fusi_dataarray(bad)


def test_validate_fusi_dataarray_rejects_non_string_dimension_names() -> None:
    """Dimension names must be strings."""
    bad = xr.DataArray(np.zeros((2, 3, 4), dtype=np.float32), dims=("time", "j", 1))

    with pytest.raises(ValueError, match="All dimensions must be strings"):
        validate_fusi_dataarray(bad)


def test_validate_fusi_dataarray_rejects_non_dimension_coordinate() -> None:
    """Dimension coordinates must be 1D along their own dimension."""
    data = _make_voxel_affine_time_series()
    bad = data.assign_coords(i=xr.DataArray(np.arange(data.sizes["j"]), dims=("j",)))

    with pytest.raises(ValueError, match="must be a 1D dimension coordinate"):
        validate_fusi_dataarray(bad)

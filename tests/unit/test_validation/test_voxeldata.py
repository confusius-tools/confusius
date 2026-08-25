"""Tests for generic ConfUSIus DataArray validation."""

from typing import Any

import numpy as np
import pytest
import xarray as xr

from confusius._utils.geometry import attach_voxel_to_world_index
from confusius.validation import canonicalize_voxeldata, validate_voxeldata
from confusius.xarray import create_voxeldata

_SLICE_TIME_ATTRS = {
    "units": "s",
    "volume_acquisition_reference": "start",
    "volume_acquisition_duration": 0.1,
}


def _make_voxel_to_world_volume() -> xr.DataArray:
    """Create a small ConfUSIus-style 3D volume."""
    base = xr.DataArray(
        np.zeros((2, 3, 4), dtype=np.float32),
        dims=("k", "j", "i"),
        coords={
            "k": xr.DataArray([0, 1], dims=("k",)),
            "j": xr.DataArray([0, 2, 4], dims=("j",)),
            "i": xr.DataArray([0, 1, 2, 3], dims=("i",)),
        },
    )
    return attach_voxel_to_world_index(
        base,
        np.array(
            [
                [2.0, 0.0, 0.0, 10.0],
                [0.0, 3.0, 0.0, 20.0],
                [0.0, 0.0, 4.0, 30.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )


def _make_voxel_to_world_time_series() -> xr.DataArray:
    """Create a small ConfUSIus-style 3D+t volume."""
    volume = _make_voxel_to_world_volume().expand_dims(time=6).copy()
    volume = volume.assign_coords(
        time=xr.DataArray(
            np.arange(6, dtype=float) * 0.5,
            dims=("time",),
            attrs={
                "units": "s",
                "volume_acquisition_reference": "start",
                "volume_acquisition_duration": 0.5,
            },
        )
    )
    return volume.transpose("time", "k", "j", "i")


def test_validate_voxeldata_accepts_valid_3d() -> None:
    """A canonical 3D ConfUSIus volume validates successfully."""
    validate_voxeldata(_make_voxel_to_world_volume())


def test_validate_voxeldata_accepts_valid_3dt() -> None:
    """A canonical 3D+t ConfUSIus volume validates successfully."""
    validate_voxeldata(_make_voxel_to_world_time_series())


def test_validate_voxeldata_accepts_time_first_slice_time() -> None:
    """`slice_time` is a valid optional VoxelData timing coordinate."""
    time_values = _make_voxel_to_world_time_series().coords["time"].values
    data = _make_voxel_to_world_time_series().assign_coords(
        slice_time=xr.DataArray(
            time_values[:, np.newaxis] + np.array([0.0, 0.2]),
            dims=("time", "k"),
            attrs=_SLICE_TIME_ATTRS,
        )
    )

    validate_voxeldata(data)


def test_validate_voxeldata_rejects_non_time_first_slice_time() -> None:
    """Time-series `slice_time` coordinates must be time-first."""
    data = _make_voxel_to_world_time_series().assign_coords(
        slice_time=xr.DataArray(
            np.zeros((2, 6)), dims=("k", "time"), attrs=_SLICE_TIME_ATTRS
        )
    )

    with pytest.raises(ValueError, match="must have dims"):
        validate_voxeldata(data)


def test_validate_voxeldata_rejects_slice_time_without_units() -> None:
    """`slice_time` coordinates carry physical time units."""
    data = _make_voxel_to_world_time_series().assign_coords(
        slice_time=xr.DataArray(np.zeros((6, 2)), dims=("time", "k"))
    )

    with pytest.raises(ValueError, match="missing required 'units'"):
        validate_voxeldata(data)


def test_validate_voxeldata_rejects_nonfinite_slice_time() -> None:
    """`slice_time` coordinates must be finite."""
    values = np.zeros((6, 2))
    values[0, 0] = np.nan
    data = _make_voxel_to_world_time_series().assign_coords(
        slice_time=xr.DataArray(values, dims=("time", "k"), attrs={"units": "s"})
    )

    with pytest.raises(ValueError, match="non-finite"):
        validate_voxeldata(data)


def test_validate_voxeldata_accepts_1d_slice_time_with_scalar_time() -> None:
    """Single-volume snapshots can keep 1D `slice_time` metadata."""
    data = (
        _make_voxel_to_world_time_series()
        .isel(time=0)
        .assign_coords(
            slice_time=xr.DataArray(
                np.array([0.0, 0.2]), dims=("k",), attrs=_SLICE_TIME_ATTRS
            )
        )
    )

    validate_voxeldata(data)


def test_validate_voxeldata_rejects_1d_slice_time_on_time_series() -> None:
    """Time-series `slice_time` needs an explicit time axis."""
    data = _make_voxel_to_world_time_series().assign_coords(
        slice_time=xr.DataArray(np.zeros(2), dims=("k",), attrs=_SLICE_TIME_ATTRS)
    )

    with pytest.raises(ValueError, match="must have dims"):
        validate_voxeldata(data)


def test_validate_voxeldata_rejects_3d_slice_time() -> None:
    """`slice_time` names one acquisition sweep dimension, not a volume grid."""
    data = _make_voxel_to_world_time_series().assign_coords(
        slice_time=xr.DataArray(
            np.zeros((6, 2, 3)), dims=("time", "k", "j"), attrs=_SLICE_TIME_ATTRS
        )
    )

    with pytest.raises(ValueError, match="must have dims"):
        validate_voxeldata(data)


def test_validate_voxeldata_rejects_non_dataarray() -> None:
    """Non-DataArray inputs raise `TypeError`."""
    bad_data: Any = np.zeros((2, 2))

    with pytest.raises(TypeError, match="xarray.DataArray"):
        validate_voxeldata(bad_data)


def test_validate_voxeldata_rejects_plain_world_grid() -> None:
    """Plain z/y/x dimension arrays are not valid fUSI data."""
    data = xr.DataArray(
        np.zeros((2, 3, 4), dtype=np.float32),
        dims=("z", "y", "x"),
        coords={"z": [0.0, 1.0], "y": [0.0, 1.0, 2.0], "x": [0.0, 1.0, 2.0, 3.0]},
    )

    with pytest.raises(ValueError, match="must include all native voxel dimensions"):
        validate_voxeldata(data)


def test_validate_voxeldata_rejects_zero_length_core_dim() -> None:
    """A zero-length core voxel dimension means the array has no data at all."""
    data = _make_voxel_to_world_volume().isel(k=slice(0, 0))

    with pytest.raises(ValueError, match="zero-length dimensions.*'k'"):
        validate_voxeldata(data)


def test_validate_voxeldata_rejects_zero_length_extra_dim() -> None:
    """A zero-length non-core dimension is rejected too, not only core dims."""
    data = _make_voxel_to_world_time_series().isel(time=slice(0, 0))

    with pytest.raises(ValueError, match="zero-length dimensions.*'time'"):
        validate_voxeldata(data)


def test_validate_voxeldata_rejects_voxel_to_world_missing_world_coord() -> None:
    """Voxel-to-world geometry requires linked world coordinates."""
    good = _make_voxel_to_world_volume()
    bad = xr.DataArray(
        good.values,
        dims=good.dims,
        coords={dim: good.coords[dim] for dim in ("k", "j", "i")},
    )

    with pytest.raises(ValueError, match="VoxelToWorldIndex-backed"):
        validate_voxeldata(bad)


def test_validate_voxeldata_rejects_scalar_indexed_voxel_dim() -> None:
    """Canonical fUSI data must keep all native voxel dimensions active."""
    base = xr.DataArray(
        np.zeros((2, 3, 4), dtype=np.float32),
        dims=("k", "j", "i"),
        coords={
            "k": xr.DataArray([0, 1], dims=("k",)),
            "j": xr.DataArray([0, 1, 2], dims=("j",)),
            "i": xr.DataArray([0, 1, 2, 3], dims=("i",)),
        },
    )
    oblique = np.array(
        [
            [0.0, 1.0, 0.0, 10.0],
            [1.0, 0.0, 0.0, 20.0],
            [0.0, 0.0, 1.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    data = attach_voxel_to_world_index(base, oblique)
    fixed = data.isel(k=0)

    with pytest.raises(ValueError, match="must include all native voxel dimensions"):
        validate_voxeldata(fixed)


def test_validate_voxeldata_allows_extra_dims_by_default() -> None:
    """Extra non-core dimensions are allowed by default."""
    data = _make_voxel_to_world_time_series().expand_dims(region=["roi"])
    validate_voxeldata(data)


def test_validate_voxeldata_can_forbid_extra_dims() -> None:
    """Extra non-core dimensions can be rejected explicitly."""
    data = _make_voxel_to_world_time_series().expand_dims(region=["roi"])

    with pytest.raises(ValueError, match="Unexpected dimensions"):
        validate_voxeldata(data, allow_extra_dims=False)


def test_validate_voxeldata_rejects_2d_voxel_to_world_data() -> None:
    """Canonical fUSI data must include all `k/j/i` voxel dimensions."""
    base = xr.DataArray(
        np.zeros((3, 4), dtype=np.float32),
        dims=("j", "i"),
        coords={"j": [0, 2, 4], "i": [0, 1, 2, 3]},
    )
    with pytest.raises(ValueError, match="must have all native voxel dims"):
        bad = attach_voxel_to_world_index(
            base,
            np.array([[3.0, 0.0, 20.0], [0.0, 4.0, 30.0], [0.0, 0.0, 1.0]]),
        )
        validate_voxeldata(bad)


def test_validate_voxeldata_can_require_time() -> None:
    """`require_time=True` rejects arrays without a time dimension."""
    spatial = _make_voxel_to_world_time_series().isel(time=0, drop=True)

    with pytest.raises(ValueError, match="must have a 'time' dimension"):
        validate_voxeldata(spatial, require_time=True)


def test_validate_voxeldata_can_forbid_pose() -> None:
    """`allow_pose=False` rejects multi-pose arrays."""
    data = _make_voxel_to_world_volume().expand_dims(pose=[0, 1])

    with pytest.raises(ValueError, match="must not have a 'pose' dimension"):
        validate_voxeldata(data, allow_pose=False)


def test_validate_voxeldata_rejects_missing_dimension_coordinate() -> None:
    """Every core dimension must have a same-named coordinate."""
    bad = _make_voxel_to_world_time_series().drop_vars("i")

    with pytest.raises(ValueError, match="Missing required coordinate"):
        validate_voxeldata(bad)


def test_validate_voxeldata_allows_missing_extra_dimension_coordinate() -> None:
    """Missing extra-dimension coordinates are allowed."""
    bad = (
        _make_voxel_to_world_time_series()
        .expand_dims(region=["roi"])
        .drop_vars("region")
    )
    validate_voxeldata(bad)


def test_validate_voxeldata_rejects_non_numeric_core_coordinate() -> None:
    """Core dimension coordinates must be numeric."""
    n_i = _make_voxel_to_world_time_series().sizes["i"]
    labels = np.array([f"v{i}" for i in range(n_i)], dtype=object)
    bad = _make_voxel_to_world_time_series().assign_coords(
        i=xr.DataArray(labels, dims=("i",))
    )

    with pytest.raises(ValueError, match="must be numeric"):
        validate_voxeldata(bad)


def test_validate_voxeldata_rejects_noncanonical_dim_order() -> None:
    """VoxelData dimensions must follow the canonical order."""
    reordered = _make_voxel_to_world_time_series().transpose("j", "i", "time", "k")

    with pytest.raises(ValueError, match="not in canonical ConfUSIus order"):
        validate_voxeldata(reordered)


def test_canonicalize_voxeldata_reorders_dimensions() -> None:
    """Canonicalization moves extra and core dimensions into model order."""
    data = _make_voxel_to_world_time_series().expand_dims(component=[0, 1])
    reordered = data.transpose("j", "component", "i", "time", "k")

    result = canonicalize_voxeldata(reordered)

    assert result.dims == ("component", "time", "k", "j", "i")


def test_validate_voxeldata_can_require_regular_spacing() -> None:
    """Regular-spacing mode rejects non-uniform voxel coordinates."""
    bad = _make_voxel_to_world_time_series().assign_coords(
        j=np.array([0.0, 2.5, 4.0], dtype=float)
    )

    with pytest.raises(ValueError, match="must have regular spacing"):
        validate_voxeldata(bad, require_regular_spacing=True)


def test_validate_voxeldata_regular_spacing_can_target_space_dims_only() -> None:
    """Space-only regular-spacing checks ignore irregular time sampling."""
    base = _make_voxel_to_world_time_series()
    bad_time = base.assign_coords(
        time=xr.DataArray(
            np.array([0.0, 0.5, 1.0, 1.7, 2.2, 2.8], dtype=float),
            dims=("time",),
            attrs=base.coords["time"].attrs,
        )
    )

    validate_voxeldata(
        bad_time,
        require_regular_spacing=True,
        regular_spacing_dims="space",
    )


def test_validate_voxeldata_regular_spacing_core_checks_time_when_present() -> None:
    """`core` mode includes `time` and rejects irregular time spacing."""
    base = _make_voxel_to_world_time_series()
    bad_time = base.assign_coords(
        time=xr.DataArray(
            np.array([0.0, 0.5, 1.0, 1.7, 2.2, 2.8], dtype=float),
            dims=("time",),
            attrs=base.coords["time"].attrs,
        )
    )

    with pytest.raises(ValueError, match="must have regular spacing"):
        validate_voxeldata(
            bad_time,
            require_regular_spacing=True,
            regular_spacing_dims="core",
        )


def test_validate_voxeldata_regular_spacing_all_skips_non_numeric_extra_dim() -> None:
    """`all` mode skips a non-numeric extra dim coordinate instead of rejecting it.

    Regular-spacing checks only make sense for numeric coordinates; a string-labeled
    extra dim (e.g. a region name) has nothing to check for uniformity.
    """
    data = _make_voxel_to_world_time_series().expand_dims(region=["roi"])
    data = data.assign_coords(region=("region", np.array(["roi"], dtype=object)))

    validate_voxeldata(
        data,
        require_regular_spacing=True,
        regular_spacing_dims="all",
    )


def test_validate_voxeldata_rejects_missing_time_units() -> None:
    """`time` coordinate `units` metadata is always required."""
    bad = _make_voxel_to_world_time_series().copy(deep=True)
    del bad.coords["time"].attrs["units"]

    with pytest.raises(ValueError, match="missing required 'units' metadata"):
        validate_voxeldata(bad)


def test_validate_voxeldata_rejects_invalid_volume_acquisition_reference() -> None:
    """`volume_acquisition_reference` must name a known timing anchor."""
    bad = _make_voxel_to_world_time_series().copy(deep=True)
    bad.coords["time"].attrs["volume_acquisition_reference"] = "middle"

    with pytest.raises(
        ValueError, match="volume_acquisition_reference.*start.*center.*end"
    ):
        validate_voxeldata(bad)


def test_validate_voxeldata_rejects_missing_volume_acquisition_reference() -> None:
    """`time` coordinate `volume_acquisition_reference` metadata is required."""
    bad = _make_voxel_to_world_time_series().copy(deep=True)
    del bad.coords["time"].attrs["volume_acquisition_reference"]

    with pytest.raises(ValueError, match="missing 'volume_acquisition_reference'"):
        validate_voxeldata(bad)


def test_validate_voxeldata_rejects_missing_volume_acquisition_duration() -> None:
    """`time` coordinate `volume_acquisition_duration` metadata is required."""
    bad = _make_voxel_to_world_time_series().copy(deep=True)
    del bad.coords["time"].attrs["volume_acquisition_duration"]

    with pytest.raises(ValueError, match="missing 'volume_acquisition_duration'"):
        validate_voxeldata(bad)


def test_validate_voxeldata_rejects_non_finite_numeric_coordinate() -> None:
    """Numeric coordinates must be finite."""
    base = _make_voxel_to_world_time_series()
    bad = base.assign_coords(
        time=xr.DataArray(
            np.array([0.0, 0.5, np.nan, 1.5, 2.0, 2.5], dtype=float),
            dims=("time",),
            attrs=base.coords["time"].attrs,
        )
    )

    with pytest.raises(ValueError, match="non-finite numeric values"):
        validate_voxeldata(bad)


def test_validate_voxeldata_rejects_non_string_dimension_names() -> None:
    """Dimension names must be strings."""
    bad = xr.DataArray(np.zeros((2, 3, 4), dtype=np.float32), dims=("time", "j", 1))

    with pytest.raises(ValueError, match="All dimensions must be strings"):
        validate_voxeldata(bad)


def test_validate_voxeldata_rejects_non_monotonic_voxel_coordinate() -> None:
    """Voxel dim coordinates may run in either direction, but must be monotonic."""
    bad = _make_voxel_to_world_time_series().assign_coords(
        i=xr.DataArray([0.0, 2.0, 1.0, 3.0], dims=("i",))
    )

    with pytest.raises(ValueError, match="must be strictly monotonic"):
        validate_voxeldata(bad)


def test_validate_voxeldata_accepts_descending_voxel_coordinate() -> None:
    """A voxel dim coordinate may be strictly decreasing (e.g. a flipped axis)."""
    flipped = _make_voxel_to_world_time_series().assign_coords(
        i=xr.DataArray([3.0, 2.0, 1.0, 0.0], dims=("i",))
    )

    validate_voxeldata(flipped)


def test_validate_voxeldata_rejects_descending_time_coordinate() -> None:
    """`time`/`pose` coordinates must be strictly increasing, unlike voxel dims."""
    bad = _make_voxel_to_world_time_series().isel(time=slice(None, None, -1))

    with pytest.raises(ValueError, match="must be strictly monotonic-increasing"):
        validate_voxeldata(bad)


def test_validate_voxeldata_rejects_non_dimension_coordinate() -> None:
    """Dimension coordinates must be 1D along their own dimension."""
    data = _make_voxel_to_world_time_series()
    bad = data.assign_coords(i=xr.DataArray(np.arange(data.sizes["j"]), dims=("j",)))

    with pytest.raises(ValueError, match="must be a 1D dimension coordinate"):
        validate_voxeldata(bad)


def _make_pose_dependent_time_volume() -> xr.DataArray:
    """Create a small pose-dependent VoxelData volume with a (time, pose) time coord."""
    npose = 2
    time_values = xr.DataArray(
        np.stack(
            [np.arange(3, dtype=float) * 0.5, np.arange(3, dtype=float) * 0.5 + 0.1],
            axis=-1,
        ),
        dims=("time", "pose"),
        attrs={
            "units": "s",
            "volume_acquisition_reference": "start",
            "volume_acquisition_duration": 0.5,
        },
    )
    return create_voxeldata(
        np.zeros((3, npose, 2, 3, 4), dtype=np.float32),
        dims=("time", "pose", "k", "j", "i"),
        time=time_values,
        pose=np.arange(npose),
        voxel_to_world=np.broadcast_to(
            np.diag([2.0, 3.0, 4.0, 1.0]), (npose, 4, 4)
        ).copy(),
    )


def test_validate_voxeldata_accepts_pose_dependent_time() -> None:
    """A valid (time, pose)-shaped time coordinate passes validation."""
    validate_voxeldata(_make_pose_dependent_time_volume())


def test_validate_voxeldata_rejects_non_numeric_pose_dependent_time() -> None:
    """A non-numeric (time, pose) time coordinate is rejected."""
    data = _make_pose_dependent_time_volume()
    bad_time = xr.DataArray(
        data.coords["time"].values.astype(str),
        dims=("time", "pose"),
        attrs=dict(data.coords["time"].attrs),
    )
    bad = data.drop_vars("time").assign_coords(time=bad_time)

    with pytest.raises(ValueError, match="must be numeric"):
        validate_voxeldata(bad)


def test_validate_voxeldata_rejects_non_finite_pose_dependent_time() -> None:
    """A (time, pose) time coordinate with a NaN is rejected."""
    data = _make_pose_dependent_time_volume()
    values = data.coords["time"].values.copy()
    values[0, 0] = np.nan
    bad = data.drop_vars("time").assign_coords(
        time=xr.DataArray(
            values, dims=("time", "pose"), attrs=dict(data.coords["time"].attrs)
        )
    )

    with pytest.raises(ValueError, match="non-finite"):
        validate_voxeldata(bad)


def test_validate_voxeldata_accepts_single_timepoint_pose_dependent_time() -> None:
    """A single-timepoint (time, pose) time coordinate has nothing to check."""
    data = _make_pose_dependent_time_volume().isel(time=[0])

    validate_voxeldata(data)


def test_validate_voxeldata_rejects_non_ascending_pose_dependent_time() -> None:
    """A pose whose timestamps aren't strictly increasing is rejected."""
    data = _make_pose_dependent_time_volume()
    values = data.coords["time"].values.copy()
    values[:, 0] = values[::-1, 0]  # reverse pose 0's timestamps.
    bad = data.drop_vars("time").assign_coords(
        time=xr.DataArray(
            values, dims=("time", "pose"), attrs=dict(data.coords["time"].attrs)
        )
    )

    with pytest.raises(ValueError, match="strictly monotonic-increasing"):
        validate_voxeldata(bad)


def test_canonicalize_voxeldata_restores_scalar_indexed_voxel_dim() -> None:
    """A voxel dim collapsed to a scalar coordinate (e.g. by `.isel`) is restored.

    Selecting a single index along `j` without dropping it leaves `j` as a scalar
    coordinate rather than a dimension. `canonicalize_voxeldata` must reinstate it as a
    size-1 dimension in its original position among the other voxel dims, carrying
    over its original coordinate value and attributes.
    """
    reduced = _make_voxel_to_world_volume().isel(j=1)
    assert "j" not in reduced.dims
    assert reduced.coords["j"].shape == ()

    restored = canonicalize_voxeldata(reduced)

    assert restored.dims == ("k", "j", "i")
    assert restored.sizes["j"] == 1
    np.testing.assert_array_equal(restored.coords["j"].values, [2.0])
    assert restored.coords["j"].attrs == {}


def test_canonicalize_voxeldata_rejects_non_scalar_coordinate_for_missing_dim() -> None:
    """A missing voxel dim with a non-scalar same-named coordinate is rejected.

    `canonicalize_voxeldata` only knows how to restore a voxel dimension from a scalar
    coordinate value; a coordinate that merely shares the dimension's name but
    varies along a different dimension cannot be interpreted as that dimension's
    dropped value.
    """
    bad = xr.DataArray(
        np.zeros((2, 4), dtype=np.float32),
        dims=("k", "i"),
        coords={
            "k": xr.DataArray([0, 1], dims=("k",)),
            "i": xr.DataArray([0, 1, 2, 3], dims=("i",)),
            "j": xr.DataArray([0, 1], dims=("k",)),
        },
    )

    with pytest.raises(
        ValueError,
        match="missing voxel dimension 'j', but coordinate 'j' is not scalar",
    ):
        canonicalize_voxeldata(bad)


def test_canonicalize_voxeldata_rejects_dim_missing_entirely() -> None:
    """A voxel dim missing from both dims and coords is rejected, not skipped.

    Regression test: unlike a scalar-indexed dim (which `canonicalize_voxeldata`
    restores) or a non-scalar same-named coordinate (rejected above), a voxel
    dimension entirely absent from `data` has no scalar coordinate to restore it
    from, and used to be silently skipped instead of raising the documented
    `ValueError`.
    """
    bad = xr.DataArray(
        np.zeros((2, 4), dtype=np.float32),
        dims=("k", "i"),
        coords={
            "k": xr.DataArray([0, 1], dims=("k",)),
            "i": xr.DataArray([0, 1, 2, 3], dims=("i",)),
        },
    )

    with pytest.raises(
        ValueError,
        match="missing voxel dimension 'j', and has no scalar coordinate 'j'",
    ):
        canonicalize_voxeldata(bad)

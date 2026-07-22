"""Tests for the create_fusi_dataarray constructor helper."""

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose
from xarray.testing import assert_identical

from confusius.validation import validate_fusi_dataarray
from confusius.xarray import create_fusi_dataarray


def test_create_fusi_dataarray_builds_canonical_time_varying_single_slice():
    """A single-slice recording is built as canonical `(time, z, y, x)` data."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal((5, 1, 8, 12))

    result = create_fusi_dataarray(
        data,
        dims=("time", "z", "y", "x"),
        dt=0.5,
        dz=0.4,
        dy=0.1,
        dx=0.2,
        t0=1.0,
        z0=2.0,
    )

    assert result.dims == ("time", "z", "y", "x")
    assert_allclose(result.coords["time"].values, 1.0 + np.arange(5) * 0.5)
    assert_allclose(result.coords["z"].values, 2.0 + np.arange(1) * 0.4)
    assert_allclose(result.coords["y"].values, np.arange(8) * 0.1)
    assert result.coords["time"].attrs["units"] == "s"
    assert result.coords["z"].attrs == {"units": "mm", "voxdim": 0.4}
    assert result.coords["x"].attrs == {"units": "mm", "voxdim": 0.2}
    validate_fusi_dataarray(result, require_time=True)


def test_create_fusi_dataarray_adds_missing_singleton_spatial_dim():
    """A missing spatial dim is added from its required spacing/origin."""
    result = create_fusi_dataarray(
        np.zeros((5, 8, 12)),
        dims=("time", "y", "x"),
        dt=0.5,
        dz=0.4,
        dy=0.1,
        dx=0.2,
        z0=2.0,
    )

    assert result.dims == ("time", "z", "y", "x")
    assert result.shape == (5, 1, 8, 12)
    assert_allclose(result.coords["z"].values, [2.0])
    assert result.coords["z"].attrs == {"units": "mm", "voxdim": 0.4}
    validate_fusi_dataarray(result, require_time=True)


def test_create_fusi_dataarray_reorders_input_dims_by_default():
    """Input dims may be any order and are canonicalized by default."""
    data = np.arange(4 * 2 * 3).reshape(4, 2, 3)

    result = create_fusi_dataarray(
        data,
        dims=("x", "z", "time"),
        dt=0.5,
        dz=0.4,
        dy=0.1,
        dx=0.2,
    )

    assert result.dims == ("time", "z", "y", "x")
    assert result.shape == (3, 2, 1, 4)
    assert_allclose(result.isel(y=0).transpose("x", "z", "time"), data)


def test_create_fusi_dataarray_can_preserve_input_dim_order():
    """`canonical_order=False` preserves supplied order apart from added singleton dims."""
    result = create_fusi_dataarray(
        np.zeros((4, 2, 3)),
        dims=("x", "z", "time"),
        dt=0.5,
        dz=0.4,
        dy=0.1,
        dx=0.2,
        canonical_order=False,
    )

    assert result.dims == ("y", "x", "z", "time")
    validate_fusi_dataarray(result, require_time=True)


def test_create_fusi_dataarray_accepts_explicit_coordinates():
    """Explicit coordinates can replace spacing arguments when spacing is inferable."""
    coords = {
        "time": xr.DataArray(
            np.arange(5) * 0.5,
            dims=("time",),
            attrs={"units": "s"},
        ),
        "z": xr.DataArray(
            [2.0],
            dims=("z",),
            attrs={"units": "mm", "voxdim": 0.4},
        ),
        "y": np.arange(8) * 0.1,
        "x": np.arange(12) * 0.2,
    }

    result = create_fusi_dataarray(
        np.zeros((5, 1, 8, 12)),
        dims=("time", "z", "y", "x"),
        coords=coords,
    )

    assert_allclose(result.coords["time"], coords["time"])
    assert result.coords["z"].attrs == {"units": "mm", "voxdim": 0.4}
    assert result.coords["y"].attrs["units"] == "mm"
    assert result.coords["x"].attrs["units"] == "mm"
    assert result.coords["y"].attrs["voxdim"] == pytest.approx(0.1)
    assert result.coords["x"].attrs["voxdim"] == pytest.approx(0.2)


def test_create_fusi_dataarray_accepts_nonuniform_time_coordinate():
    """Exact timestamps can represent recordings with nonuniform frame timing."""
    timestamps = np.array([0.0, 1.0, 2.1, 3.0])

    result = create_fusi_dataarray(
        np.zeros((4, 2, 3)),
        dims=("time", "y", "x"),
        coords={"time": timestamps},
        dz=0.4,
        dy=0.1,
        dx=0.2,
    )

    assert_allclose(result.coords["time"], timestamps)
    assert result.coords["time"].attrs["volume_acquisition_duration"] == pytest.approx(
        1.0
    )


def test_create_fusi_dataarray_accepts_explicit_pose_coordinate():
    """Coordinates can be provided for the `pose` dimension."""
    result = create_fusi_dataarray(
        np.zeros((2, 4, 1, 3, 5)),
        dims=("pose", "time", "z", "y", "x"),
        coords={"pose": [0.0, 15.0]},
        dt=0.5,
        dz=0.4,
        dy=0.1,
        dx=0.2,
    )

    assert result.dims == ("time", "pose", "z", "y", "x")
    assert_allclose(result.coords["pose"].values, [0.0, 15.0])


def test_create_fusi_dataarray_rejects_missing_spacing():
    """ConfUSIus must not guess physical spacing."""
    with pytest.raises(ValueError, match="Spacing for dimension 'z' is required"):
        create_fusi_dataarray(
            np.zeros((1, 4, 6)),
            dims=("z", "y", "x"),
            dy=0.1,
            dx=0.2,
        )


@pytest.mark.parametrize("bad_dx", [0.0, np.inf])
def test_create_fusi_dataarray_rejects_invalid_spacing(bad_dx: float):
    """Spacing values must be positive finite physical distances."""
    with pytest.raises(ValueError, match="positive and finite"):
        create_fusi_dataarray(
            np.zeros((1, 4, 6)),
            dims=("z", "y", "x"),
            dz=0.4,
            dy=0.1,
            dx=bad_dx,
        )


@pytest.mark.parametrize("bad_voxdim", [0.0, -0.1, np.nan])
def test_create_fusi_dataarray_rejects_invalid_explicit_voxdim(bad_voxdim: float):
    """Explicit `voxdim` metadata must be positive and finite."""
    with pytest.raises(ValueError, match="voxdim for dimension 'z'"):
        create_fusi_dataarray(
            np.zeros((1, 4, 6)),
            dims=("z", "y", "x"),
            coords={"z": xr.DataArray([0.0], dims="z", attrs={"voxdim": bad_voxdim})},
            dy=0.1,
            dx=0.2,
        )


def test_create_fusi_dataarray_rejects_singleton_explicit_coord_without_voxdim():
    """A length-1 coordinate needs spacing metadata or a spacing argument."""
    with pytest.raises(ValueError, match="Spacing for dimension 'z' is required"):
        create_fusi_dataarray(
            np.zeros((1, 4, 6)),
            dims=("z", "y", "x"),
            coords={"z": [0.0], "y": np.arange(4) * 0.1, "x": np.arange(6) * 0.2},
        )


def test_create_fusi_dataarray_rejects_nonuniform_explicit_coord():
    """Explicit core coordinates must be regularly spaced."""
    with pytest.raises(ValueError, match="Spacing for dimension 'y' is required"):
        create_fusi_dataarray(
            np.zeros((1, 4, 6)),
            dims=("z", "y", "x"),
            coords={"y": [0.0, 0.1, 0.3, 0.4], "x": np.arange(6) * 0.2},
            dz=0.4,
        )


def test_create_fusi_dataarray_rejects_decreasing_explicit_coord():
    """Explicit coordinates must not imply negative spacing."""
    with pytest.raises(ValueError, match="Spacing for dimension 'x' is required"):
        create_fusi_dataarray(
            np.zeros((1, 4, 3)),
            dims=("z", "y", "x"),
            coords={"x": [0.2, 0.1, 0.0]},
            dz=0.4,
            dy=0.1,
        )


def test_create_fusi_dataarray_rejects_explicit_coord_shape_mismatch():
    """Explicit coordinates must match the named data axis length."""
    with pytest.raises(ValueError, match="Coordinate 'x' must be 1D with length 6"):
        create_fusi_dataarray(
            np.zeros((1, 4, 6)),
            dims=("z", "y", "x"),
            coords={"x": np.zeros((1, 6))},
            dz=0.4,
            dy=0.1,
        )


def test_create_fusi_dataarray_allows_extra_dim_without_coordinate():
    """Extra non-core dimensions get index coordinates when none are provided."""
    result = create_fusi_dataarray(
        np.zeros((2, 1, 4, 6)),
        dims=("condition", "z", "y", "x"),
        dz=0.4,
        dy=0.1,
        dx=0.2,
    )

    assert result.dims == ("z", "y", "x", "condition")
    assert_allclose(result.coords["condition"].values, [0, 1])


def test_create_fusi_dataarray_uses_dt_for_singleton_time_coordinate():
    """A singleton explicit time coordinate needs `dt` for duration metadata."""
    result = create_fusi_dataarray(
        np.zeros((1, 1, 4, 6)),
        dims=("time", "z", "y", "x"),
        coords={"time": [2.0]},
        dt=0.5,
        dz=0.4,
        dy=0.1,
        dx=0.2,
    )

    assert result.coords["time"].attrs["volume_acquisition_duration"] == 0.5


def test_create_fusi_dataarray_rejects_dims_length_mismatch():
    """The length of `dims` must match the array rank."""
    with pytest.raises(ValueError, match=r"must match the number of array dimensions"):
        create_fusi_dataarray(
            np.zeros((1, 4, 6)),
            dims=("z", "y", "x", "time"),
            dt=0.5,
            dz=0.4,
            dy=0.1,
            dx=0.2,
        )


def test_create_fusi_dataarray_rejects_duplicate_dims():
    """Duplicate dimension names are rejected."""
    with pytest.raises(ValueError, match=r"must not contain duplicate names"):
        create_fusi_dataarray(
            np.zeros((1, 4, 4)),
            dims=("z", "y", "y"),
            dz=0.4,
            dy=0.1,
            dx=0.2,
        )


def test_create_fusi_dataarray_stores_acquisition_metadata_on_time_coord():
    """Acquisition window metadata is attached to the `time` coordinate."""
    result = create_fusi_dataarray(
        np.zeros((5, 1, 4, 6)),
        dims=("time", "z", "y", "x"),
        dt=0.5,
        dz=0.4,
        dy=0.1,
        dx=0.2,
        volume_acquisition_reference="center",
        volume_acquisition_duration=0.4,
    )

    time_attrs = result.coords["time"].attrs
    assert time_attrs["volume_acquisition_reference"] == "center"
    assert time_attrs["volume_acquisition_duration"] == 0.4
    assert time_attrs["units"] == "s"


def test_create_fusi_dataarray_acquisition_metadata_defaults_to_time_spacing():
    """`time` duration defaults to the provided or inferred time spacing."""
    result = create_fusi_dataarray(
        np.zeros((5, 1, 4, 6)),
        dims=("time", "z", "y", "x"),
        dt=0.5,
        dz=0.4,
        dy=0.1,
        dx=0.2,
    )

    time_attrs = result.coords["time"].attrs
    assert time_attrs["volume_acquisition_reference"] == "start"
    assert time_attrs["volume_acquisition_duration"] == 0.5


@pytest.mark.parametrize("bad_duration", [0.0, -0.1, np.nan])
def test_create_fusi_dataarray_rejects_invalid_acquisition_duration(
    bad_duration: float,
):
    """Acquisition duration metadata must be positive and finite."""
    with pytest.raises(ValueError, match="volume_acquisition_duration"):
        create_fusi_dataarray(
            np.zeros((5, 1, 4, 6)),
            dims=("time", "z", "y", "x"),
            dt=0.5,
            dz=0.4,
            dy=0.1,
            dx=0.2,
            volume_acquisition_duration=bad_duration,
        )


def test_create_fusi_dataarray_rejects_invalid_reference():
    """An out-of-set `volume_acquisition_reference` is rejected."""
    with pytest.raises(
        ValueError, match=r"volume_acquisition_reference must be one of"
    ):
        create_fusi_dataarray(
            np.zeros((5, 1, 4, 6)),
            dims=("time", "z", "y", "x"),
            dt=0.5,
            dz=0.4,
            dy=0.1,
            dx=0.2,
            volume_acquisition_reference="middle",  # ty: ignore[invalid-argument-type]
        )


def test_create_fusi_dataarray_rejects_duration_without_time():
    """`volume_acquisition_duration` requires a `time` dimension."""
    with pytest.raises(ValueError, match=r"requires a 'time' dimension"):
        create_fusi_dataarray(
            np.zeros((1, 4, 6)),
            dims=("z", "y", "x"),
            dz=0.4,
            dy=0.1,
            dx=0.2,
            volume_acquisition_duration=0.4,
        )


def test_create_fusi_dataarray_passes_through_dataarray_attrs():
    """DataArray-level acquisition metadata flows through `attrs`."""
    result = create_fusi_dataarray(
        np.zeros((1, 4, 6)),
        dims=("z", "y", "x"),
        dz=0.4,
        dy=0.1,
        dx=0.2,
        attrs={"transmit_frequency": 15.625e6, "beamforming_sound_velocity": 1540.0},
    )

    assert result.attrs["transmit_frequency"] == 15.625e6
    assert result.attrs["beamforming_sound_velocity"] == 1540.0


def test_create_fusi_dataarray_matches_explicit_singleton_constructor():
    """Omitting singleton `z` is equivalent to passing it explicitly."""
    data = np.random.default_rng(0).standard_normal((5, 4, 6))

    omitted = create_fusi_dataarray(
        data,
        dims=("time", "y", "x"),
        dt=0.5,
        dz=0.4,
        dy=0.1,
        dx=0.2,
    )
    explicit = create_fusi_dataarray(
        data[:, None, :, :],
        dims=("time", "z", "y", "x"),
        dt=0.5,
        dz=0.4,
        dy=0.1,
        dx=0.2,
    )

    assert_identical(omitted, explicit)

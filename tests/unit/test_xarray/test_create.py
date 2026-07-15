"""Tests for the create_fusi_dataarray constructor helper."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

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
    # The result must pass fUSI validation.
    validate_fusi_dataarray(result, require_time=True)


def test_create_fusi_dataarray_defaults_spacing_to_unit():
    """Unspecified spacings default to 1.0 and are recorded as `voxdim`."""
    result = create_fusi_dataarray(np.zeros((1, 4, 6)), dims=("z", "y", "x"))

    assert result.coords["z"].attrs["voxdim"] == 1.0
    assert_allclose(result.coords["x"].values, np.arange(6) * 1.0)
    validate_fusi_dataarray(result)


def test_create_fusi_dataarray_rejects_dims_without_full_trio():
    """A `dims` sequence missing a spatial axis fails validation."""
    with pytest.raises(ValueError, match=r"must contain all spatial dimensions"):
        create_fusi_dataarray(np.zeros((4, 6)), dims=("y", "x"))


def test_create_fusi_dataarray_rejects_dims_length_mismatch():
    """The length of `dims` must match the array rank."""
    with pytest.raises(ValueError, match=r"must match the number of array dimensions"):
        create_fusi_dataarray(np.zeros((1, 4, 6)), dims=("z", "y", "x", "time"))


def test_create_fusi_dataarray_rejects_duplicate_dims():
    """Duplicate dimension names are rejected."""
    with pytest.raises(ValueError, match=r"must not contain duplicate names"):
        create_fusi_dataarray(np.zeros((1, 4, 4)), dims=("z", "y", "y"))


def test_create_fusi_dataarray_stores_acquisition_metadata_on_time_coord():
    """Acquisition window metadata is attached to the `time` coordinate."""
    result = create_fusi_dataarray(
        np.zeros((5, 1, 4, 6)),
        dims=("time", "z", "y", "x"),
        dt=0.5,
        volume_acquisition_reference="center",
        volume_acquisition_duration=0.4,
    )

    time_attrs = result.coords["time"].attrs
    assert time_attrs["volume_acquisition_reference"] == "center"
    assert time_attrs["volume_acquisition_duration"] == 0.4
    assert time_attrs["units"] == "s"


def test_create_fusi_dataarray_acquisition_metadata_defaults():
    """`time` gets a 'start' reference and duration defaulting to the time spacing."""
    result = create_fusi_dataarray(
        np.zeros((5, 1, 4, 6)), dims=("time", "z", "y", "x"), dt=0.5
    )

    time_attrs = result.coords["time"].attrs
    assert time_attrs["volume_acquisition_reference"] == "start"
    assert time_attrs["volume_acquisition_duration"] == 0.5


def test_create_fusi_dataarray_rejects_invalid_reference():
    """An out-of-set `volume_acquisition_reference` is rejected."""
    with pytest.raises(ValueError, match=r"volume_acquisition_reference must be one of"):
        create_fusi_dataarray(
            np.zeros((5, 1, 4, 6)),
            dims=("time", "z", "y", "x"),
            volume_acquisition_reference="middle",  # ty: ignore[invalid-argument-type]
        )


def test_create_fusi_dataarray_rejects_duration_without_time():
    """`volume_acquisition_duration` requires a `time` dimension."""
    with pytest.raises(ValueError, match=r"requires a 'time' dimension"):
        create_fusi_dataarray(
            np.zeros((1, 4, 6)),
            dims=("z", "y", "x"),
            volume_acquisition_duration=0.4,
        )


def test_create_fusi_dataarray_passes_through_dataarray_attrs():
    """DataArray-level acquisition metadata flows through `attrs`."""
    result = create_fusi_dataarray(
        np.zeros((1, 4, 6)),
        dims=("z", "y", "x"),
        attrs={"transmit_frequency": 15.625e6, "beamforming_sound_velocity": 1540.0},
    )

    assert result.attrs["transmit_frequency"] == 15.625e6
    assert result.attrs["beamforming_sound_velocity"] == 1540.0

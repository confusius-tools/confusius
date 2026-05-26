"""Tests for generic ConfUSIus DataArray validation."""

import numpy as np
import pytest
import xarray as xr

from confusius.validation import validate_fusi_dataarray


def test_validate_fusi_dataarray_accepts_valid_3d(
    sample_3d_volume: xr.DataArray,
) -> None:
    """A canonical 2D+t DataArray also validates successfully."""
    validate_fusi_dataarray(sample_3d_volume)


def test_validate_fusi_dataarray_accepts_valid_2dt(
    sample_3dt_volume: xr.DataArray,
) -> None:
    """A canonical 2D+t DataArray also validates successfully."""
    data_2dt = sample_3dt_volume.isel(z=0, drop=True)
    validate_fusi_dataarray(data_2dt)


def test_validate_fusi_dataarray_accepts_valid_3dt(
    sample_3dt_volume: xr.DataArray,
) -> None:
    """A canonical 3D+t DataArray validates successfully."""
    validate_fusi_dataarray(sample_3dt_volume)


def test_validate_fusi_dataarray_rejects_non_dataarray() -> None:
    """Non-DataArray inputs raise `TypeError`."""
    with pytest.raises(TypeError, match="xarray.DataArray"):
        validate_fusi_dataarray(np.zeros((2, 2)))  # type: ignore


def test_validate_fusi_dataarray_allows_extra_dims_by_default(
    sample_3dt_volume: xr.DataArray,
) -> None:
    """Extra non-core dimensions are allowed by default."""
    data = sample_3dt_volume.expand_dims(region=["roi"])

    validate_fusi_dataarray(data)


def test_validate_fusi_dataarray_rejects_non_monotonic_numeric_coordinate(
    sample_3dt_volume: xr.DataArray,
) -> None:
    """Core numeric dimension coordinates must be strictly increasing."""
    n_y = sample_3dt_volume.sizes["y"]
    bad = sample_3dt_volume.assign_coords(
        y=np.array([1.0, 2.0, 1.5, *range(3, n_y)], dtype=float)
    )

    with pytest.raises(ValueError, match="strictly monotonic-increasing"):
        validate_fusi_dataarray(bad)


def test_validate_fusi_dataarray_allows_non_monotonic_numeric_extra_dimension_coordinate(
    sample_3dt_volume: xr.DataArray,
) -> None:
    """Extra-dimension coordinates are not constrained by monotonicity checks."""
    data = sample_3dt_volume.expand_dims(region=[0.0, 2.0, 1.0])

    validate_fusi_dataarray(data)


def test_validate_fusi_dataarray_allows_non_dimension_coordinates(
    sample_3dt_volume: xr.DataArray,
) -> None:
    """Auxiliary non-dimension coordinates are allowed."""
    data = sample_3dt_volume.assign_coords(
        quality=xr.DataArray(
            np.array(["ok"] * sample_3dt_volume.size, dtype=object).reshape(
                sample_3dt_volume.shape
            ),
            dims=sample_3dt_volume.dims,
        )
    )

    validate_fusi_dataarray(data)


def test_validate_fusi_dataarray_can_forbid_extra_dims(
    sample_3dt_volume: xr.DataArray,
) -> None:
    """Extra non-core dimensions can be rejected explicitly."""
    data = sample_3dt_volume.expand_dims(region=["roi"])

    with pytest.raises(ValueError, match="Unexpected dimensions"):
        validate_fusi_dataarray(data, allow_extra_dims=False)


def test_validate_fusi_dataarray_requires_minimum_spatial_dims(
    sample_3dt_volume: xr.DataArray,
) -> None:
    """At least the configured number of spatial dimensions must be present."""
    bad = sample_3dt_volume.isel(z=0, y=0, drop=True)

    with pytest.raises(ValueError, match="at least 2 spatial dimensions"):
        validate_fusi_dataarray(bad)


def test_validate_fusi_dataarray_can_require_time(
    sample_3dt_volume: xr.DataArray,
) -> None:
    """`require_time=True` rejects arrays without a time dimension."""
    spatial = sample_3dt_volume.isel(time=0, drop=True)

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
    sample_3dt_volume: xr.DataArray,
) -> None:
    """Every dimension must have a same-named coordinate."""
    bad = sample_3dt_volume.drop_vars("x")

    with pytest.raises(ValueError, match="Missing required coordinate"):
        validate_fusi_dataarray(bad)


def test_validate_fusi_dataarray_allows_missing_extra_dimension_coordinate(
    sample_3dt_volume: xr.DataArray,
) -> None:
    """Missing extra-dimension coordinates are allowed."""
    bad = sample_3dt_volume.expand_dims(region=["roi"]).drop_vars("region")

    validate_fusi_dataarray(bad)


def test_validate_fusi_dataarray_still_requires_core_dimension_coordinate(
    sample_3dt_volume: xr.DataArray,
) -> None:
    """Core-dimension coordinates are always required."""
    bad = sample_3dt_volume.drop_vars("x")

    with pytest.raises(ValueError, match="Missing required coordinate"):
        validate_fusi_dataarray(bad)


def test_validate_fusi_dataarray_rejects_non_numeric_core_coordinate(
    sample_3dt_volume: xr.DataArray,
) -> None:
    """Core dimension coordinates must be numeric."""
    n_x = sample_3dt_volume.sizes["x"]
    labels = np.array([f"v{i}" for i in range(n_x)], dtype=object)
    bad = sample_3dt_volume.assign_coords(x=xr.DataArray(labels, dims=("x",)))

    with pytest.raises(ValueError, match="must be numeric"):
        validate_fusi_dataarray(bad)


def test_validate_fusi_dataarray_can_require_canonical_dim_order(
    sample_3dt_volume: xr.DataArray,
) -> None:
    """Canonical core dimension order can be enforced explicitly."""
    reordered = sample_3dt_volume.transpose("y", "x", "time", "z")

    with pytest.raises(ValueError, match="not in canonical ConfUSIus order"):
        validate_fusi_dataarray(reordered, require_canonical_dim_order=True)


def test_validate_fusi_dataarray_can_require_regular_spacing(
    sample_3dt_volume: xr.DataArray,
) -> None:
    """Regular-spacing mode rejects non-uniform spatial coordinates by default."""
    n_y = sample_3dt_volume.sizes["y"]
    bad = sample_3dt_volume.assign_coords(
        y=np.array([1.0, 1.6, 2.0, *range(3, n_y)], dtype=float)
    )

    with pytest.raises(ValueError, match="must have regular spacing"):
        validate_fusi_dataarray(bad, require_regular_spacing=True)


def test_validate_fusi_dataarray_regular_spacing_can_target_space_dims_only(
    sample_3dt_volume: xr.DataArray,
) -> None:
    """Space-only regular-spacing checks ignore irregular time sampling."""
    n_t = sample_3dt_volume.sizes["time"]
    bad_time = sample_3dt_volume.assign_coords(
        time=np.array([0.0, 0.5, 1.0, 1.7, 2.2, *range(5, n_t)], dtype=float)
    )

    validate_fusi_dataarray(
        bad_time,
        require_regular_spacing=True,
        regular_spacing_dims="space",
    )


def test_validate_fusi_dataarray_regular_spacing_rejects_missing_selected_dim(
    sample_3dt_volume: xr.DataArray,
) -> None:
    """Explicit regular-spacing dims must exist in the DataArray."""
    with pytest.raises(ValueError, match="regular_spacing_dims contains dimensions"):
        validate_fusi_dataarray(
            sample_3dt_volume,
            require_regular_spacing=True,
            regular_spacing_dims=["w"],
        )


def test_validate_fusi_dataarray_space_regular_spacing_excludes_pose() -> None:
    """`space` regular-spacing mode does not include the `pose` dimension."""
    data = xr.DataArray(
        np.zeros((3, 4, 5), dtype=np.float32),
        dims=("pose", "y", "x"),
        coords={
            "pose": [0.0, 1.0, 3.0],
            "y": np.arange(4, dtype=float),
            "x": np.arange(5, dtype=float),
        },
    )

    validate_fusi_dataarray(data, require_regular_spacing=True)


def test_validate_fusi_dataarray_regular_spacing_skips_non_numeric_selected_dims(
    sample_3dt_volume: xr.DataArray,
) -> None:
    """Explicit non-numeric selected dims are ignored."""
    data = sample_3dt_volume.expand_dims(region=["roi_a", "roi_b"])

    validate_fusi_dataarray(
        data,
        require_regular_spacing=True,
        regular_spacing_dims=["region"],
    )


def test_validate_fusi_dataarray_regular_spacing_all_checks_all_numeric_dims(
    sample_3dt_volume: xr.DataArray,
) -> None:
    """`all` mode checks all numeric dimensions."""
    validate_fusi_dataarray(
        sample_3dt_volume,
        require_regular_spacing=True,
        regular_spacing_dims="all",
    )


def test_validate_fusi_dataarray_regular_spacing_all_skips_non_numeric_dims(
    sample_3dt_volume: xr.DataArray,
) -> None:
    """`all` mode ignores non-numeric dimension coordinates."""
    data = sample_3dt_volume.expand_dims(region=["roi_a", "roi_b"])

    validate_fusi_dataarray(
        data,
        require_regular_spacing=True,
        regular_spacing_dims="all",
    )


def test_validate_fusi_dataarray_regular_spacing_core_checks_time_when_present(
    sample_3dt_volume: xr.DataArray,
) -> None:
    """`core` mode includes `time` and rejects irregular time spacing."""
    n_t = sample_3dt_volume.sizes["time"]
    bad_time = sample_3dt_volume.assign_coords(
        time=np.array([0.0, 0.5, 1.0, 1.7, 2.2, *range(5, n_t)], dtype=float)
    )

    with pytest.raises(ValueError, match="must have regular spacing"):
        validate_fusi_dataarray(
            bad_time,
            require_regular_spacing=True,
            regular_spacing_dims="core",
        )


def test_validate_fusi_dataarray_can_require_spatial_voxdim(
    sample_3dt_volume: xr.DataArray,
) -> None:
    """Spatial `voxdim` metadata can be enforced explicitly."""
    bad = sample_3dt_volume.copy(deep=True)
    del bad.coords["y"].attrs["voxdim"]

    with pytest.raises(ValueError, match="missing required 'voxdim' metadata"):
        validate_fusi_dataarray(bad, require_spatial_voxdim=True)


def test_validate_fusi_dataarray_can_require_coordinate_units(
    sample_3dt_volume: xr.DataArray,
) -> None:
    """Coordinate `units` metadata can be enforced explicitly."""
    bad = sample_3dt_volume.copy(deep=True)
    del bad.coords["time"].attrs["units"]

    with pytest.raises(ValueError, match="missing required 'units' metadata"):
        validate_fusi_dataarray(bad, require_time_units=True)


def test_validate_fusi_dataarray_rejects_non_finite_numeric_coordinate(
    sample_3dt_volume: xr.DataArray,
) -> None:
    """Numeric coordinates must be finite."""
    n_t = sample_3dt_volume.sizes["time"]
    bad = sample_3dt_volume.assign_coords(
        time=np.array([0.0, 0.5, np.nan, 1.5, 2.0, *range(5, n_t)], dtype=float)
    )

    with pytest.raises(ValueError, match="non-finite numeric values"):
        validate_fusi_dataarray(bad)


def test_validate_fusi_dataarray_rejects_non_string_dimension_names() -> None:
    """Dimension names must be strings."""
    bad = xr.DataArray(
        np.zeros((2, 3, 4), dtype=np.float32),
        dims=("time", "y", 1),
    )

    with pytest.raises(ValueError, match="All dimensions must be strings"):
        validate_fusi_dataarray(bad)


def test_validate_fusi_dataarray_validates_minimum_spatial_dims_bounds(
    sample_3dt_volume: xr.DataArray,
) -> None:
    """minimum_spatial_dims must be in [0, 3]."""
    with pytest.raises(ValueError, match="between 0 and 3 inclusive"):
        validate_fusi_dataarray(sample_3dt_volume, minimum_spatial_dims=4)


def test_validate_fusi_dataarray_can_require_spatial_units(
    sample_3dt_volume: xr.DataArray,
) -> None:
    """Spatial `units` metadata can be enforced explicitly."""
    bad = sample_3dt_volume.copy(deep=True)
    del bad.coords["x"].attrs["units"]

    with pytest.raises(ValueError, match="missing required 'units' metadata"):
        validate_fusi_dataarray(bad, require_spatial_units=True)


def test_validate_fusi_dataarray_rejects_non_dimension_coordinate(
    sample_3dt_volume: xr.DataArray,
) -> None:
    """Dimension coordinates must be 1D along their own dimension."""
    bad = sample_3dt_volume.assign_coords(
        x=xr.DataArray(np.arange(sample_3dt_volume.sizes["y"]), dims=("y",))
    )

    with pytest.raises(ValueError, match="must be a 1D dimension coordinate"):
        validate_fusi_dataarray(bad)


def test_validate_fusi_dataarray_regular_spacing_still_requires_numeric_core_coords(
    sample_3dt_volume: xr.DataArray,
) -> None:
    """Core coords must remain numeric even when regular spacing is requested."""
    n_x = sample_3dt_volume.sizes["x"]
    labels = np.array([f"v{i}" for i in range(n_x)], dtype=object)
    data = sample_3dt_volume.assign_coords(x=xr.DataArray(labels, dims=("x",)))

    with pytest.raises(ValueError, match="must be numeric"):
        validate_fusi_dataarray(data, require_regular_spacing=True)

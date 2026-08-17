"""Tests for voxel-space geometry helpers."""

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal

from confusius._utils.geometry import (
    VoxelToWorldIndex,
    VoxelToWorldTransform,
    attach_voxel_to_world_index,
    get_affine_axis_scalings,
    get_affine_axis_vectors,
    get_affine_direction_matrix,
    get_voxel_to_world_affine,
    get_voxel_to_world_coord_names,
    get_voxel_to_world_index_origin,
    get_voxel_to_world_spacings_from_coords,
    restore_voxel_to_world_index,
)


def test_axis_aligned_voxel_to_world_computes_correct_world_coords() -> None:
    """Axis-aligned voxel-to-world geometry computes correct world coords.

    World coordinates are backed by a single joint index spanning all voxel
    dimensions (see [VoxelToWorldIndex][confusius._utils.geometry.VoxelToWorldIndex]
    for why), so each is `(k, j, i)`-shaped even though, for an axis-aligned affine,
    its value only actually varies along its own paired voxel dimension.
    """
    data = xr.DataArray(
        np.arange(24).reshape(2, 3, 4),
        dims=("k", "j", "i"),
        coords={
            "k": [0, 2],
            "j": [0, 1, 3],
            "i": [0, 2, 3, 7],
        },
    )
    voxel_to_world = np.array(
        [
            [10.0, 0.0, 0.0, 100.0],
            [0.0, 2.0, 0.0, 200.0],
            [0.0, 0.0, 3.0, 300.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    result = attach_voxel_to_world_index(
        data,
        voxel_to_world,
    )

    assert result.coords["z"].dims == ("k", "j", "i")
    assert result.coords["y"].dims == ("k", "j", "i")
    assert result.coords["x"].dims == ("k", "j", "i")
    # Each coordinate only varies along its own paired voxel dimension.
    z_by_k = result.coords["z"].isel(j=0, i=0).values
    y_by_j = result.coords["y"].isel(k=0, i=0).values
    x_by_i = result.coords["x"].isel(k=0, j=0).values
    assert_array_equal(z_by_k, [100.0, 120.0])
    assert_array_equal(y_by_j, [200.0, 202.0, 206.0])
    assert_array_equal(x_by_i, [300.0, 306.0, 309.0, 321.0])
    assert type(result.xindexes["x"]).__name__ == "VoxelToWorldIndex"


def test_axis_aligned_voxel_to_world_uses_voxel_to_world_index() -> None:
    """Axis-aligned world coords are owned by a VoxelToWorldIndex."""
    data = xr.DataArray(
        np.arange(24).reshape(2, 3, 4),
        dims=("k", "j", "i"),
        coords={
            "k": [0, 2],
            "j": [0, 1, 3],
            "i": [0, 2, 3, 7],
        },
    )
    voxel_to_world = np.array(
        [
            [1.0, 0.0, 0.0, 10.0],
            [0.0, 2.0, 0.0, 20.0],
            [0.0, 0.0, 3.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    result = attach_voxel_to_world_index(
        data,
        voxel_to_world,
    )

    assert list(result.xindexes) == ["k", "j", "i", "z", "y", "x"]
    assert type(result.xindexes["z"]).__name__ == "VoxelToWorldIndex"
    result.stack(space=("k", "j", "i"))


def test_oblique_coordinate_transform_index_selection_uses_world_coords() -> None:
    """Oblique voxel-to-world geometry still uses pointwise world selection."""
    data = xr.DataArray(
        np.arange(24).reshape(2, 3, 4),
        dims=("k", "j", "i"),
        coords={
            "k": [0, 2],
            "j": [0, 1, 3],
            "i": [0, 2, 3, 7],
        },
    )
    voxel_to_world = np.array(
        [
            [1.0, 0.1, 0.0, 10.0],
            [0.0, 2.0, 0.0, 20.0],
            [0.0, 0.0, 3.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    result = attach_voxel_to_world_index(
        data,
        voxel_to_world,
    )

    selected = result.sel(
        z=xr.Variable("point", [12.1]),
        y=xr.Variable("point", [26.2]),
        x=xr.Variable("point", [39.4]),
        method="nearest",
    )

    assert type(result.xindexes["x"]).__name__ == "VoxelToWorldIndex"
    assert result.coords["x"].dims == ("k", "j", "i")
    assert selected.item() == data.sel(k=2.0, j=3.0, i=3.0).item()


def test_affine_geometry_helpers_extract_vectors_scalings_and_orientation() -> None:
    """Affine geometry helpers expose the linear part in world-space form."""
    voxel_to_world = np.array(
        [
            [2.0, 1.0, 0.0, 10.0],
            [0.0, 3.0, 0.0, 20.0],
            [0.0, 0.0, 4.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    assert_allclose(
        get_affine_axis_vectors(voxel_to_world, ("k", "j", "i"))["k"],
        [2.0, 0.0, 0.0],
    )
    assert_allclose(
        get_affine_axis_vectors(voxel_to_world, ("k", "j", "i"))["j"],
        [1.0, 3.0, 0.0],
    )
    scalings = get_affine_axis_scalings(voxel_to_world, ("k", "j", "i"))
    assert scalings.keys() == {"k", "j", "i"}
    assert_allclose(scalings["k"], 2.0)
    assert_allclose(scalings["j"], np.sqrt(10.0))
    assert_allclose(scalings["i"], 4.0)
    expected_orientation = np.array(
        [
            [1.0, 1.0 / np.sqrt(10.0), 0.0],
            [0.0, 3.0 / np.sqrt(10.0), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    assert_allclose(get_affine_direction_matrix(voxel_to_world), expected_orientation)


def test_get_world_spacings_singleton_axis_uses_affine_column_norm() -> None:
    """Singleton voxel axes still have a world per-voxel spacing from the affine."""
    voxel_coords = {
        "k": [0],
        "j": [0, 2, 4],
        "i": [0, 1, 2, 3],
    }
    voxel_to_world = np.array(
        [
            [0.4, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    spacing = get_voxel_to_world_spacings_from_coords(voxel_coords, voxel_to_world)

    assert spacing == {"k": 0.4, "j": 6.0, "i": 5.0}


def test_get_world_spacings_returns_none_for_irregular_voxel_axes() -> None:
    """World spacing is undefined when voxel-space sampling is irregular."""
    voxel_coords = {
        "k": [0, 1, 2],
        "j": [0, 2, 4],
        "i": [0, 1, 3, 4],
    }
    voxel_to_world = np.array(
        [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    spacing = get_voxel_to_world_spacings_from_coords(voxel_coords, voxel_to_world)

    assert spacing == {"k": 2.0, "j": 6.0, "i": None}


def test_get_voxel_to_world_origin_uses_first_sampled_voxel() -> None:
    """Voxel-to-world origin is the world location of array index zero."""
    data = xr.DataArray(
        np.zeros((2, 3, 4)),
        dims=("k", "j", "i"),
        coords={
            "k": [10, 11],
            "j": [5, 7, 9],
            "i": [100, 101, 102, 103],
        },
    )
    data = attach_voxel_to_world_index(
        data,
        np.array(
            [
                [2.0, 0.0, 0.0, 10.0],
                [0.0, 3.0, 0.0, 20.0],
                [0.0, 0.0, 4.0, 30.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )

    assert get_voxel_to_world_index_origin(data) == {"z": 30.0, "y": 35.0, "x": 430.0}


def _axis_aligned_result() -> xr.DataArray:
    """Build a 3D axis-aligned voxel-to-world DataArray shared by several tests."""
    data = xr.DataArray(
        np.arange(24).reshape(2, 3, 4),
        dims=("k", "j", "i"),
        coords={
            "k": [0, 2],
            "j": [0, 1, 3],
            "i": [0, 2, 3, 7],
        },
    )
    return attach_voxel_to_world_index(
        data,
        np.diag([10.0, 2.0, 3.0, 1.0]),
    )


def test_alignment_between_different_world_grids_raises_clear_error() -> None:
    """Arithmetic alignment tells users to resample mismatched world grids."""
    left = _axis_aligned_result()
    right = attach_voxel_to_world_index(
        xr.DataArray(
            np.ones_like(left.values),
            dims=left.dims,
            coords={dim: left.coords[dim].values for dim in left.dims},
        ),
        np.array(
            [
                [10.0, 0.0, 0.0, 1.0],
                [0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 3.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )

    with pytest.raises(ValueError, match="Cannot automatically align .*resample_like"):
        left + right


def test_reindex_between_different_world_grids_raises_clear_error() -> None:
    """Reindexing alignment tells users to resample mismatched world grids."""
    left = _axis_aligned_result()
    right = attach_voxel_to_world_index(
        xr.DataArray(
            np.ones_like(left.values),
            dims=left.dims,
            coords={dim: left.coords[dim].values for dim in left.dims},
        ),
        np.array(
            [
                [10.0, 0.0, 0.0, 1.0],
                [0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 3.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )

    with pytest.raises(
        ValueError, match="Cannot automatically reindex .*resample_like"
    ):
        left.reindex_like(right)


def test_add_world_coords_defaults_to_yx_names_for_2d_voxel_dims() -> None:
    """2D voxel geometry defaults to `y`/`x` world coordinate names."""
    data = xr.DataArray(
        np.zeros((3, 4)),
        dims=("j", "i"),
        coords={"j": np.arange(3), "i": np.arange(4)},
    )

    result = attach_voxel_to_world_index(data, np.eye(3))

    assert set(result.coords) == {"j", "i", "y", "x"}


def test_voxel_to_world_index_from_affine_defaults_to_yx_names_for_2d() -> None:
    """`VoxelToWorldIndex.from_affine` itself defaults to `y`/`x` for 2D geometry.

    `attach_voxel_to_world_index` always resolves `world_coord_names` before
    delegating to `from_affine`, so `from_affine`'s own default only fires when it is
    called directly.
    """
    index = VoxelToWorldIndex.from_affine(
        {"j": np.arange(3), "i": np.arange(4)}, np.eye(3)
    )

    assert index.world_coord_names == ("y", "x")


def test_sel_resolves_descending_and_nonmonotonic_axis_aligned_axes() -> None:
    """Point selection resolves correctly for descending or non-monotonic axes.

    `VoxelToWorldIndex.sel` reverse-looks-up world labels into voxel positions via
    `_reverse_lookup_positions`, which special-cases strictly-descending axes
    (interpolation on the reversed axis) and falls back to exact-match lookup for
    genuinely non-monotonic axes.
    """
    data = xr.DataArray(
        np.arange(24).reshape(2, 3, 4),
        dims=("k", "j", "i"),
        coords={
            "k": [4, 0],  # Strictly descending.
            "j": [0, 3, 1],  # Non-monotonic.
            "i": [0, 1, 2, 3],
        },
    )
    result = attach_voxel_to_world_index(data, np.eye(4))

    descending = result.sel(z=0.0)
    assert descending.coords["k"].item() == 0.0
    assert_array_equal(descending.values, data.isel(k=1).values)

    nonmonotonic = result.sel(y=1.0)
    assert nonmonotonic.coords["j"].item() == 1.0
    assert_array_equal(nonmonotonic.values, data.isel(j=2).values)


def test_sel_slice_out_of_range_selects_nothing_and_step_subsamples() -> None:
    """Axis-aligned slice selection handles an out-of-range slice and a step."""
    result = _axis_aligned_result()

    empty = result.sel(z=slice(1000.0, 2000.0))
    assert empty.sizes["k"] == 0

    stepped = result.sel(x=slice(0.0, 21.0, 2))
    assert_array_equal(stepped.coords["i"].values, [0.0, 3.0])


def test_sel_plain_slice_without_step_selects_contiguous_range() -> None:
    """A plain (no-step) slice with hits selects the contiguous covered range."""
    result = _axis_aligned_result()

    selected = result.sel(z=slice(0.0, 15.0))

    assert_array_equal(selected.coords["k"].values, [0.0])


def test_voxel_to_world_index_sel_with_unrelated_labels_selects_nothing() -> None:
    """Calling `.sel` with no labels belonging to the index selects nothing."""
    result = _axis_aligned_result()
    index = result.xindexes["z"]

    selection = index.sel({})

    assert selection.dim_indexers == {}


def test_reverse_skips_dimension_fixed_by_a_prior_scalar_isel() -> None:
    """Oblique point selection after a scalar isel ignores the now-fixed dimension.

    A scalar `isel` on one voxel dimension pins it via `fixed_voxel_coords` rather
    than dropping the geometry; a subsequent oblique `.sel` on the remaining
    dimensions must not try to resolve a position for the fixed one.
    """
    data = xr.DataArray(
        np.arange(24).reshape(2, 3, 4),
        dims=("k", "j", "i"),
        coords={
            "k": [0, 2],
            "j": [0, 1, 3],
            "i": [0, 2, 3, 7],
        },
    )
    voxel_to_world = np.array(
        [
            [1.0, 0.1, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    result = attach_voxel_to_world_index(
        data,
        voxel_to_world,
    )
    fixed = result.isel(k=0)

    selected = fixed.sel(
        z=xr.Variable("point", [0.1]),
        y=xr.Variable("point", [2.0]),
        x=xr.Variable("point", [6.0]),
        method="nearest",
    )

    assert selected.item() == data.isel(k=0, j=1, i=1).item()


def test_voxel_to_world_transform_rejects_invalid_construction() -> None:
    """`VoxelToWorldTransform` validates its constructor arguments."""
    valid_coords = {"j": np.arange(3), "i": np.arange(4)}

    with pytest.raises(ValueError, match="must exactly cover active dims"):
        VoxelToWorldTransform(valid_coords, np.eye(3), all_dims=("j", "i", "k"))

    with pytest.raises(ValueError, match="at least one active dim"):
        VoxelToWorldTransform({}, np.eye(3))

    with pytest.raises(ValueError, match="only supports 2D or 3D"):
        VoxelToWorldTransform(
            {
                "k": np.arange(2),
                "j": np.arange(3),
                "i": np.arange(4),
                "t": np.arange(2.0),
            },
            np.eye(5),
        )

    with pytest.raises(ValueError, match="must be 1D"):
        VoxelToWorldTransform({"j": np.arange(3), "i": np.zeros((2, 2))}, np.eye(3))

    with pytest.raises(ValueError, match="one entry per voxel dimension"):
        VoxelToWorldTransform(valid_coords, np.eye(3), world_coord_names=("y",))

    with pytest.raises(ValueError, match=r"must have shape \(3, 3\)"):
        VoxelToWorldTransform(valid_coords, np.eye(4))


def test_voxel_to_world_transform_defaults_to_yx_names_for_2d() -> None:
    """A `VoxelToWorldTransform` built without explicit names defaults to `y`/`x`."""
    transform = VoxelToWorldTransform({"j": np.arange(3), "i": np.arange(4)}, np.eye(3))

    assert transform.coord_names == ("y", "x")


def test_voxel_to_world_transform_equals_rejects_other_types() -> None:
    """`VoxelToWorldTransform.equals` returns `False` for non-transform values."""
    transform = VoxelToWorldTransform({"j": np.arange(3), "i": np.arange(4)}, np.eye(3))

    assert transform.equals(object()) is False  # ty: ignore[invalid-argument-type]


def test_voxel_to_world_transform_repr_reports_dims_and_coord_names() -> None:
    """`repr` reports the transform's active dims and world coordinate names."""
    transform = VoxelToWorldTransform({"j": np.arange(3), "i": np.arange(4)}, np.eye(3))

    assert repr(transform) == (
        "VoxelToWorldTransform(dims=('j', 'i'), coord_names=('y', 'x'))"
    )


def test_voxel_to_world_transform_isel_unsupported_indexers_return_none() -> None:
    """`VoxelToWorldTransform.isel` returns `None` for unsupported indexers.

    Covers: an indexer for a dimension the transform doesn't own (skipped), a
    multi-dimensional `Variable`, a non-slice/array/scalar indexer (also exercising
    `_is_scalar_indexer`'s final `False` fallback), and a fancy multi-dimensional
    array index that would produce a non-1D result.
    """
    transform = VoxelToWorldTransform({"j": np.arange(3), "i": np.arange(4)}, np.eye(3))

    unrelated = transform.isel({"nonexistent": 0})
    assert unrelated is not None
    assert unrelated.dims == transform.dims

    fixed = transform.isel({"j": xr.Variable((), 1)})
    assert fixed is not None
    assert fixed.fixed_voxel_coords == {"j": 1.0}

    assert transform.isel({"j": xr.Variable(("a", "b"), np.zeros((2, 2)))}) is None
    assert transform.isel({"j": "bogus"}) is None
    assert transform.isel({"j": np.zeros((2, 2), dtype=int)}) is None


def test_add_world_coords_validates_voxel_dims_and_coordinates() -> None:
    """`attach_voxel_to_world_index` validates dims and their coordinates."""
    data = xr.DataArray(
        np.zeros((3, 4)),
        dims=("j", "i"),
        coords={"j": np.arange(3), "i": np.arange(4)},
    )

    with pytest.raises(ValueError, match="must have a matching 1D coordinate"):
        attach_voxel_to_world_index(data.drop_vars("i"), np.eye(3))

    non_dim_coord = xr.DataArray(
        np.zeros((3, 4)),
        dims=("j", "i"),
        coords={"j": (("j", "i"), np.zeros((3, 4))), "i": np.arange(4)},
    )
    with pytest.raises(ValueError, match="must be a 1D dimension coordinate"):
        attach_voxel_to_world_index(non_dim_coord, np.eye(3))

    no_voxel_dims = xr.DataArray(np.zeros((3, 4)), dims=("a", "b"))
    with pytest.raises(ValueError, match="must have at least one native voxel dim"):
        attach_voxel_to_world_index(no_voxel_dims, np.eye(3))


def test_get_voxel_to_world_affine_raises_without_voxel_to_world_geometry() -> None:
    """`get_voxel_to_world_affine` raises for a DataArray without a voxel-to-world index."""
    data = xr.DataArray(np.zeros((3, 4)), dims=("y", "x"))

    with pytest.raises(ValueError, match="must have a voxel-to-world index"):
        get_voxel_to_world_affine(data)


def test_restore_world_coords_rebuilds_geometry_after_expand_dims() -> None:
    """Restoring a dimension fixed by isel rebuilds identical world coordinates."""
    result = _axis_aligned_result()
    fixed = result.isel(k=1)

    assert restore_voxel_to_world_index(fixed) is fixed

    expanded = fixed.expand_dims(k=[2])
    restored = restore_voxel_to_world_index(expanded)

    assert restored.coords["z"].dims == ("k", "j", "i")
    assert_allclose(restored.coords["z"].values, result.isel(k=[1]).coords["z"].values)


def test_get_voxel_to_world_world_coord_names_falls_back_to_plain_coords() -> None:
    """World coordinate names are inferred from plain coords without an index.

    This covers restoring a DataArray that carries dense world coordinates matching
    the voxel dims (e.g. after an operation that drops the `VoxelToWorldIndex` but
    keeps the coordinate arrays) rather than a live `VoxelToWorldIndex`.
    """
    plain = xr.DataArray(np.zeros((2, 3, 4)), dims=("k", "j", "i")).assign_coords(
        z=(("k", "j", "i"), np.zeros((2, 3, 4))),
        y=(("k", "j", "i"), np.zeros((2, 3, 4))),
        x=(("k", "j", "i"), np.zeros((2, 3, 4))),
    )

    assert get_voxel_to_world_coord_names(plain) == ("z", "y", "x")


def test_get_voxel_to_world_world_coord_names_defaults_when_coords_are_incomplete() -> (
    None
):
    """Default `z`/`y`/`x` names are returned when plain world coords are incomplete.

    Only a matching subset of the default-named world coordinates is present (`y`
    and `x` are missing), so the fallback cannot confirm all of them and returns the
    plain default names instead.
    """
    plain = xr.DataArray(np.zeros((2, 3, 4)), dims=("k", "j", "i")).assign_coords(
        z=(("k", "j", "i"), np.zeros((2, 3, 4))),
    )

    assert get_voxel_to_world_coord_names(plain) == ("z", "y", "x")

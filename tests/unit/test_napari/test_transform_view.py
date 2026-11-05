"""Tests for the lazily resampled layer view, against the `plot_napari` path."""

import dask.array as da
import numpy as np
import pytest

from confusius._napari._transform._model import TransformState
from confusius._napari._transform._view import (
    ResampledView,
    compute_sweep_grid,
    compute_tight_grid,
)
from confusius._utils.napari import build_napari_layer_data
from confusius.xarray import create_voxeldata


class CountingArray:
    """Array-like counting reads, standing in for a slow on-disk source."""

    def __init__(self, values):
        self.values = values
        self.shape = values.shape
        self.dtype = values.dtype
        self.ndim = values.ndim
        self.reads = 0

    def __getitem__(self, key):
        self.reads += 1
        return self.values[key]


def _as_counted_dask(data):
    """Return `data` backed by one Dask chunk over a read-counting array."""
    counting = CountingArray(data.values)
    lazy = da.from_array(
        counting,
        chunks=counting.shape,
        asarray=False,
        fancy=False,
        lock=False,
        name="counted",
        meta=np.empty((0,) * counting.ndim, dtype=counting.dtype),
    )
    return data.copy(data=lazy), counting


def _rotation(data, axis="x", angle=17.0):
    """Return a rotation of `data` about its geometry center."""
    center = data.fusi.affine.geometry_center.to_numpy()
    return TransformState.identity(center=center).rotated(axis, angle, frame="global")


def _view(source, transform, *, layer_type="image", fill_value=0.0):
    """Return a view of `source` on the tight grid of `transform`."""
    view = ResampledView(source, layer_type=layer_type, fill_value=fill_value)
    view.set_transform(transform, compute_tight_grid(source, transform))
    return view


def _plot_napari_layer(source, transform, *, layer_type="image", fill_value=0.0):
    """Return the array, scale and translate `plot_napari` would display."""
    array, kwargs, _ = build_napari_layer_data(
        source.fusi.affine.apply(transform),
        layer_type=layer_type,
        resample_fill_value=fill_value,
    )
    return np.asarray(array), kwargs["scale"], kwargs["translate"]


def _assert_matches_plot_napari(source, transform):
    """Assert the view's voxels and grid are the ones `plot_napari` displays."""
    expected, scale, translate = _plot_napari_layer(source, transform)

    view = _view(source, transform)
    grid = compute_tight_grid(source, transform)

    assert view.shape == expected.shape
    np.testing.assert_allclose(np.asarray(view), expected, atol=1e-10)
    np.testing.assert_allclose(grid.spacing, scale[-3:])
    np.testing.assert_allclose(grid.origin, translate[-3:])


def test_rotation_matches_plot_napari(sample_voxeldata_3d):
    _assert_matches_plot_napari(sample_voxeldata_3d, _rotation(sample_voxeldata_3d).matrix())


def test_oblique_source_matches_plot_napari(sample_voxeldata_3d_oblique):
    data = sample_voxeldata_3d_oblique

    _assert_matches_plot_napari(data, _rotation(data, "z").matrix())


def test_rotation_plus_unequal_zoom_matches_plot_napari(sample_voxeldata_3d_oblique):
    data = sample_voxeldata_3d_oblique
    state = _rotation(data).zoomed((1.5, 1.0, 0.5))

    _assert_matches_plot_napari(data, state.matrix())


def test_flip_of_oblique_source_matches_plot_napari(sample_voxeldata_3d_oblique):
    # An oblique source takes the resampling path, away from the deferred placement bug
    # for negative-diagonal affines.
    data = sample_voxeldata_3d_oblique
    state = _rotation(data).flipped("y", frame="local")

    _assert_matches_plot_napari(data, state.matrix())


@pytest.mark.parametrize(
    "selection",
    [
        {"k": slice(1, None)},
        {"i": slice(None, None, 2)},
        {"j": slice(None, None, -1)},
    ],
    ids=["cropped", "strided", "reversed"],
)
def test_non_dense_voxel_coordinates_match_plot_napari(sample_voxeldata_3d, selection):
    data = sample_voxeldata_3d.isel(selection)

    _assert_matches_plot_napari(data, _rotation(data).matrix())


def test_time_series_matches_plot_napari(sample_voxeldata_3dt):
    data = sample_voxeldata_3dt

    _assert_matches_plot_napari(data, _rotation(data).matrix())


def test_extra_leading_dims_match_plot_napari(sample_voxeldata_3dt):
    stacked = np.stack([sample_voxeldata_3dt.values, -sample_voxeldata_3dt.values])
    data = create_voxeldata(
        stacked,
        dims=("component", "time", "k", "j", "i"),
        extra_coords={"component": [0, 1]},
        time=sample_voxeldata_3dt.coords["time"],
        voxel_to_world=sample_voxeldata_3dt.fusi.affine.voxel_to_world,
    )

    _assert_matches_plot_napari(data, _rotation(data).matrix())


def test_unsorted_extra_dim_matches_plot_napari(sample_voxeldata_3d):
    # `build_napari_layer_data` sorts leading dims other than time, so a slider index
    # must select the same element in the preview as on the committed layer.
    stacked = np.stack([sample_voxeldata_3d.values * scale for scale in (3.0, 2.0, 1.0)])
    data = create_voxeldata(
        stacked,
        dims=("component", "k", "j", "i"),
        extra_coords={"component": ["z", "y", "x"]},
        voxel_to_world=sample_voxeldata_3d.fusi.affine.voxel_to_world,
    )

    _assert_matches_plot_napari(data, _rotation(data).matrix())


@pytest.mark.parametrize("byte_order", [">f4", ">f8"])
def test_big_endian_source_is_displayed(sample_voxeldata_3d, byte_order):
    # SimpleITK rejects non-native byte orders, and NIfTI files can be big-endian.
    native = sample_voxeldata_3d.copy(data=sample_voxeldata_3d.values.astype(byte_order[1:]))
    source = sample_voxeldata_3d.copy(data=sample_voxeldata_3d.values.astype(byte_order))
    transform = _rotation(source).matrix()

    view = _view(source, transform)

    assert view.dtype.isnative
    np.testing.assert_allclose(
        np.asarray(view), np.asarray(_view(native, transform)), atol=1e-6
    )


def test_big_endian_labels_are_displayed(sample_voxeldata_3d):
    values = (sample_voxeldata_3d.values * 40).astype(">i2")
    source = sample_voxeldata_3d.copy(data=values)

    view = _view(source, np.eye(4), layer_type="labels")

    assert view.dtype.isnative
    np.testing.assert_array_equal(np.asarray(view), values)


def test_size_counts_the_displayed_voxels(sample_voxeldata_3dt):
    view = _view(sample_voxeldata_3dt, _rotation(sample_voxeldata_3dt).matrix())

    assert view.size == np.prod(view.shape)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_plane_equals_volume_slice(sample_voxeldata_3d, axis):
    view = _view(sample_voxeldata_3d, _rotation(sample_voxeldata_3d).matrix())
    volume = np.asarray(view)
    index = view.shape[axis] // 2
    key = [slice(None)] * 3
    key[axis] = index

    plane = view[tuple(key)]

    np.testing.assert_allclose(plane, volume[tuple(key)], atol=1e-10)


def test_time_series_plane_equals_volume_slice(sample_voxeldata_3dt):
    view = _view(sample_voxeldata_3dt, _rotation(sample_voxeldata_3dt).matrix())

    plane = view[3, 1, :, :]

    np.testing.assert_allclose(plane, np.asarray(view)[3, 1], atol=1e-10)


@pytest.mark.parametrize("layer_type", ["image", "labels"])
def test_time_course_matches_planes(sample_voxeldata_3dt, layer_type):
    source = sample_voxeldata_3dt
    if layer_type == "labels":
        source = source.copy(data=(source.values * 50).astype(np.int32))
    view = _view(source, _rotation(source).matrix(), layer_type=layer_type)
    n_time, n_k, n_j, n_i = view.shape

    for voxel in [(n_k // 2, n_j // 2, n_i // 2), (0, 0, 0), (n_k - 1, n_j - 1, 0)]:
        course = view[(slice(None), *voxel)]

        expected = [view[(frame, *voxel)] for frame in range(n_time)]
        np.testing.assert_allclose(course, expected, atol=1e-10)


def test_time_course_outside_the_field_of_view_is_the_fill_value(sample_voxeldata_3dt):
    source = sample_voxeldata_3dt
    view = _view(source, _rotation(source, angle=40.0).matrix(), fill_value=-7.0)

    # A rotated box does not reach the corners of the grid enclosing it.
    course = view[:, 0, 0, 0]

    np.testing.assert_array_equal(course, np.full(source.sizes["time"], -7.0))


def test_time_course_reads_source_once(sample_voxeldata_3dt):
    source, counting = _as_counted_dask(sample_voxeldata_3dt)
    view = _view(source, _rotation(sample_voxeldata_3dt).matrix())
    n_k, n_j, n_i = view.shape[1:]

    course = view[:, n_k // 2, n_j // 2, n_i // 2]

    assert course.shape == (source.sizes["time"],)
    assert counting.reads == 1


def test_whole_array_reads_source_once(sample_voxeldata_3dt):
    source, counting = _as_counted_dask(sample_voxeldata_3dt)
    view = _view(source, _rotation(sample_voxeldata_3dt).matrix())

    volume = np.asarray(view)

    assert volume.shape == view.shape
    assert counting.reads == 1


def test_changing_transform_does_not_read_source_again(sample_voxeldata_3dt):
    source, counting = _as_counted_dask(sample_voxeldata_3dt)
    state = _rotation(sample_voxeldata_3dt)
    view = _view(source, state.matrix())
    view[2, 0, :, :]

    for angle in (1.0, 2.0, 3.0):
        view.set_transform(state.rotated("x", angle, frame="global").matrix())
        view[2, 0, :, :]

    assert counting.reads == 1


def test_length_one_slices_reuse_the_cached_volume(sample_voxeldata_3dt):
    # What napari 0.9 sends for the sliders' position, rather than integers.
    source, counting = _as_counted_dask(sample_voxeldata_3dt)
    state = _rotation(sample_voxeldata_3dt)
    view = _view(source, state.matrix())
    key = (slice(2, 3), slice(1, 2), slice(None), slice(None))
    view[key]

    for angle in (1.0, 2.0, 3.0):
        view.set_transform(state.rotated("x", angle, frame="global").matrix())
        view[key]

    assert counting.reads == 1


def test_length_one_slices_keep_their_dimensions(sample_voxeldata_3dt):
    view = _view(sample_voxeldata_3dt, _rotation(sample_voxeldata_3dt).matrix())

    result = view[2:3, 1:2, :, :]

    assert result.shape == (1, 1, view.shape[2], view.shape[3])
    np.testing.assert_allclose(result[0, 0], view[2, 1, :, :], atol=1e-10)


def test_length_one_time_slice_around_a_volume_keeps_its_dimension(sample_voxeldata_3dt):
    view = _view(sample_voxeldata_3dt, _rotation(sample_voxeldata_3dt).matrix())

    result = view[4:5]

    assert result.shape == (1, *view.shape[1:])
    np.testing.assert_allclose(result[0], view[4], atol=1e-10)


def test_time_window_of_several_frames_reads_source_once(sample_voxeldata_3dt):
    source, counting = _as_counted_dask(sample_voxeldata_3dt)
    view = _view(source, _rotation(sample_voxeldata_3dt).matrix())

    result = view[2:5, 1:2, :, :]

    assert result.shape == (3, 1, view.shape[2], view.shape[3])
    assert counting.reads == 1


def test_set_transform_changes_what_is_displayed(sample_voxeldata_3d):
    source = sample_voxeldata_3d
    state = _rotation(source)
    moved = state.rotated("x", 20.0, frame="global").matrix()
    grid = compute_tight_grid(source, state.matrix())
    view = ResampledView(source, fill_value=0.0)
    view.set_transform(state.matrix(), grid)
    before = view[:, :, :]

    view.set_transform(moved)
    after = view[:, :, :]

    assert view.shape == before.shape
    assert not np.allclose(before, after)
    # Same grid, new transform: a fresh view must agree.
    fresh = ResampledView(source, fill_value=0.0)
    fresh.set_transform(moved, grid)
    np.testing.assert_array_equal(after, fresh[:, :, :])


def test_plane_then_volume_then_time_shows_right_data(sample_voxeldata_3dt):
    view = _view(sample_voxeldata_3dt, _rotation(sample_voxeldata_3dt).matrix())
    expected = np.asarray(view).copy()

    np.testing.assert_allclose(view[2, 1, :, :], expected[2, 1], atol=1e-10)
    np.testing.assert_allclose(view[2, :, :, :], expected[2], atol=1e-10)
    np.testing.assert_allclose(view[7, 1, :, :], expected[7, 1], atol=1e-10)
    np.testing.assert_allclose(view[2, 1, :, :], expected[2, 1], atol=1e-10)


def test_labels_stay_labels(sample_voxeldata_3d):
    values = (sample_voxeldata_3d.values * 40).astype(np.int32) + 1
    source = sample_voxeldata_3d.copy(data=values)

    view = _view(source, _rotation(source, angle=30.0).matrix(), layer_type="labels")

    result = np.asarray(view)
    assert view.dtype == np.int32
    assert result.dtype == np.int32
    assert set(np.unique(result)) <= set(np.unique(values)) | {0}
    # Nearest interpolation with a zero fill, as `plot_napari` does for labels.
    expected, _, _ = _plot_napari_layer(
        source, _rotation(source, angle=30.0).matrix(), layer_type="labels"
    )
    np.testing.assert_array_equal(result, expected)


def test_labels_default_to_zero_fill(sample_voxeldata_3d):
    source = sample_voxeldata_3d.copy(data=np.full((4, 6, 8), 5, dtype=np.int32))

    view = ResampledView(source, layer_type="labels")

    assert view.fill_value == 0.0


def test_bool_labels_become_int32(sample_voxeldata_3d):
    source = sample_voxeldata_3d.copy(data=sample_voxeldata_3d.values > 0.5)

    view = _view(source, _rotation(source).matrix(), layer_type="labels")

    assert view.dtype == np.int32
    assert set(np.unique(np.asarray(view))) <= {0, 1}


def test_float_labels_become_int32(sample_voxeldata_3d):
    source = sample_voxeldata_3d.copy(data=np.floor(sample_voxeldata_3d.values * 5))

    view = _view(source, _rotation(source).matrix(), layer_type="labels")

    assert view.dtype == np.int32
    assert np.asarray(view).dtype == np.int32


def test_complex_source_shows_magnitude(sample_voxeldata_3d):
    values = sample_voxeldata_3d.values
    source = sample_voxeldata_3d.copy(data=values * np.exp(1j * values))

    view = _view(source, _rotation(source).matrix())

    assert not np.iscomplexobj(np.asarray(view))
    expected = _view(sample_voxeldata_3d, _rotation(source).matrix())
    np.testing.assert_allclose(np.asarray(view), np.asarray(expected), atol=1e-10)


def test_default_fill_is_the_displayed_volume_minimum(sample_voxeldata_3d):
    source = sample_voxeldata_3d
    view = ResampledView(source)

    assert view.fill_value == pytest.approx(float(source.min()))


def test_default_fill_ignores_nan(sample_voxeldata_3d):
    values = sample_voxeldata_3d.values.copy()
    values[0, 0, 0] = np.nan

    view = ResampledView(sample_voxeldata_3d.copy(data=values))

    assert view.fill_value == pytest.approx(float(np.nanmin(values)))


def test_ellipsis_key(sample_voxeldata_3dt):
    view = _view(sample_voxeldata_3dt, _rotation(sample_voxeldata_3dt).matrix())

    np.testing.assert_array_equal(view[2, ...], view[2, :, :, :])


def test_short_key_fills_with_full_slices(sample_voxeldata_3dt):
    view = _view(sample_voxeldata_3dt, _rotation(sample_voxeldata_3dt).matrix())

    np.testing.assert_array_equal(view[2], view[2, :, :, :])


def test_negative_index_counts_from_the_end(sample_voxeldata_3d):
    view = _view(sample_voxeldata_3d, _rotation(sample_voxeldata_3d).matrix())

    np.testing.assert_array_equal(view[-1, :, :], view[view.shape[0] - 1, :, :])


def test_positive_step_matches_numpy(sample_voxeldata_3d):
    view = _view(sample_voxeldata_3d, _rotation(sample_voxeldata_3d).matrix())

    result = view[::2, 1:, ::3]

    np.testing.assert_allclose(result, np.asarray(view)[::2, 1:, ::3], atol=1e-10)


def test_empty_slice_has_a_matching_shape(sample_voxeldata_3d):
    view = _view(sample_voxeldata_3d, _rotation(sample_voxeldata_3d).matrix())

    result = view[0, 0:0, :]

    assert result.shape == (0, view.shape[2])
    assert result.dtype == view.dtype


def test_negative_step_raises(sample_voxeldata_3d):
    view = _view(sample_voxeldata_3d, _rotation(sample_voxeldata_3d).matrix())

    with pytest.raises(IndexError, match="step"):
        view[::-1, :, :]


def test_integer_out_of_range_raises(sample_voxeldata_3d):
    view = _view(sample_voxeldata_3d, _rotation(sample_voxeldata_3d).matrix())

    with pytest.raises(IndexError, match="out of bounds"):
        view[view.shape[0], :, :]


def test_too_many_indices_raises(sample_voxeldata_3d):
    view = _view(sample_voxeldata_3d, _rotation(sample_voxeldata_3d).matrix())

    with pytest.raises(IndexError, match="too many"):
        view[0, 0, 0, 0]


@pytest.mark.parametrize("axis_vector", [(0.0, 0.0, 1.0), (0.6, 0.0, 0.8)])
def test_sweep_grid_encloses_every_rotation(sample_voxeldata_3d, axis_vector):
    source = sample_voxeldata_3d
    state = _rotation(source)
    pivot = state.pivot()

    sweep = compute_sweep_grid(source, state.matrix(), np.array(axis_vector), pivot)

    sweep_end = sweep.origin + (sweep.shape - 1) * sweep.spacing
    for angle in np.linspace(-45.0, 45.0, 91):
        turned = _turned(state.matrix(), np.array(axis_vector), pivot, angle)
        tight = compute_tight_grid(source, turned)
        tight_end = tight.origin + (tight.shape - 1) * tight.spacing
        assert np.all(tight.origin >= sweep.origin - 1e-9)
        assert np.all(tight_end <= sweep_end + 1e-9)


def _turned(matrix, axis, pivot, angle):
    """Return `matrix` after a rotation of `angle` about `axis` through `pivot`."""
    from scipy.spatial.transform import Rotation

    step = np.eye(4)
    step[:3, :3] = Rotation.from_rotvec(np.deg2rad(angle) * axis).as_matrix()
    step[:3, 3] = pivot - step[:3, :3] @ pivot
    return step @ matrix


def test_sweep_grid_coarsens_to_stay_under_max_voxels():
    source = create_voxeldata(
        np.zeros((40, 64, 64), dtype=np.float32),
        dims=("k", "j", "i"),
        spacing=(0.1, 0.1, 0.1),
        origin=(0.0, 0.0, 0.0),
    )
    state = TransformState.identity()
    axis = state.rotation_axis("x", frame="global")
    full = compute_sweep_grid(source, state.matrix(), axis, state.pivot())

    capped = compute_sweep_grid(
        source, state.matrix(), axis, state.pivot(), max_voxels=100_000
    )

    assert np.prod(full.shape) > 100_000
    assert np.prod(capped.shape) <= 120_000
    np.testing.assert_allclose(capped.spacing, capped.spacing[0])
    assert capped.spacing[0] > full.spacing[0]
    # The coarser grid still covers the same sweep.
    full_end = full.origin + (full.shape - 1) * full.spacing
    capped_end = capped.origin + (capped.shape - 1) * capped.spacing
    np.testing.assert_allclose(capped.origin, full.origin)
    # The fine grid overshoots the sweep by up to one of its own voxels when it rounds.
    assert np.all(capped_end >= full_end - full.spacing)


def test_sweep_grid_under_max_voxels_stays_at_full_resolution(sample_voxeldata_3d):
    source = sample_voxeldata_3d
    state = _rotation(source)
    axis = state.rotation_axis("x", frame="global")

    full = compute_sweep_grid(source, state.matrix(), axis, state.pivot())
    capped = compute_sweep_grid(
        source, state.matrix(), axis, state.pivot(), max_voxels=10_000_000
    )

    np.testing.assert_array_equal(capped.shape, full.shape)
    np.testing.assert_array_equal(capped.spacing, full.spacing)


def test_sweep_grid_keeps_the_spacing_of_the_drag_start(sample_voxeldata_3d):
    source = sample_voxeldata_3d
    state = _rotation(source)

    sweep = compute_sweep_grid(
        source, state.matrix(), np.array([0.0, 0.0, 1.0]), state.pivot()
    )

    tight = compute_tight_grid(source, state.matrix())
    np.testing.assert_allclose(sweep.spacing, tight.spacing)


def test_sweep_grid_shape_stays_constant_during_a_drag(sample_voxeldata_3d):
    source = sample_voxeldata_3d
    state = _rotation(source)
    axis = state.rotation_axis("x", frame="global")
    sweep = compute_sweep_grid(source, state.matrix(), axis, state.pivot())
    view = ResampledView(source, fill_value=0.0)
    view.set_transform(state.matrix(), sweep)

    for angle in (-40.0, -5.0, 30.0):
        view.set_transform(state.rotated("x", angle, frame="global").matrix())

        assert view.shape == tuple(sweep.shape)


def test_sweep_grid_encloses_a_full_turn(sample_voxeldata_3d):
    source = sample_voxeldata_3d
    state = _rotation(source)
    pivot = state.pivot()
    axis = np.array([0.0, 0.6, 0.8])

    sweep = compute_sweep_grid(source, state.matrix(), axis, pivot, max_angle=180.0)

    sweep_end = sweep.origin + (sweep.shape - 1) * sweep.spacing
    for angle in np.linspace(-180.0, 180.0, 145):
        turned = _turned(state.matrix(), axis, pivot, angle)
        tight = compute_tight_grid(source, turned)
        tight_end = tight.origin + (tight.shape - 1) * tight.spacing
        assert np.all(tight.origin >= sweep.origin - 1e-9)
        assert np.all(tight_end <= sweep_end + 1e-9)


def test_sweep_grid_encloses_a_full_turn_of_a_large_volume_about_its_corner():
    source = create_voxeldata(
        np.zeros((200, 300, 400), dtype=np.float32),
        dims=("k", "j", "i"),
        spacing=(0.02, 0.02, 0.02),
        origin=(0.0, 0.0, 0.0),
    )
    state = TransformState.identity()
    axis = state.rotation_axis("x", frame="global")
    pivot = state.pivot()

    sweep = compute_sweep_grid(source, state.matrix(), axis, pivot, max_angle=180.0)

    sweep_end = sweep.origin + (sweep.shape - 1) * sweep.spacing
    for angle in np.arange(-175.0, 180.0, 10.0):
        turned = _turned(state.matrix(), axis, pivot, angle)
        tight = compute_tight_grid(source, turned)
        tight_end = tight.origin + (tight.shape - 1) * tight.spacing
        assert np.all(tight.origin >= sweep.origin - 1e-9), angle
        assert np.all(tight_end <= sweep_end + 1e-9), angle

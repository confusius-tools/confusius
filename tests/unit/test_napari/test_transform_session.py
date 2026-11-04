"""Tests for the transform preview session on real napari layers."""

import dask.array as da
import numpy as np
import pytest

from confusius._napari._transform._model import TransformState
from confusius._napari._transform._session import (
    PreviewSession,
    build_applied_layer,
    build_resampled_layer,
)
from confusius._napari._transform._view import ResampledView
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


@pytest.fixture
def volume():
    """Return a smooth, anisotropic, offset 3D volume, large enough to slice around."""
    grids = np.meshgrid(*[np.linspace(0, 3, n) for n in (8, 16, 20)], indexing="ij")
    return create_voxeldata(
        np.sin(grids[0]) + np.cos(grids[1]) * np.sin(grids[2]) + 3.0,
        dims=("k", "j", "i"),
        spacing=(0.4, 0.1, 0.1),
        origin=(1.0, 2.0, 3.0),
        name="volume",
    )


@pytest.fixture
def series(volume):
    """Return a short time series whose frames differ."""
    stack = np.stack([volume.values * (1.0 + 0.1 * t) for t in range(6)])
    return create_voxeldata(
        stack,
        dims=("time", "k", "j", "i"),
        time=np.arange(6, dtype=float) * 0.5,
        voxel_to_world=volume.fusi.affine.voxel_to_world,
        name="series",
    )


def _add_layer(viewer, data, **kwargs):
    """Add `data` to `viewer` the way the plugin readers and `plot_napari` do."""
    array, layer_kwargs, layer_type = build_napari_layer_data(data, **kwargs)
    return getattr(viewer, f"add_{layer_type}")(array, **layer_kwargs)


def _rotation(data, axis="x", angle=20.0):
    """Return a state rotating `data` about its geometry center."""
    center = data.fusi.affine.geometry_center.to_numpy()
    return TransformState.identity(center=center).rotated(axis, angle, frame="global")


def test_show_positions_layer_like_plot_napari(make_napari_viewer, volume):
    viewer = make_napari_viewer()
    layer = _add_layer(viewer, volume)
    session = PreviewSession(layer)
    transform = _rotation(volume).matrix()

    session.show(transform)

    expected, kwargs, _ = build_napari_layer_data(
        volume.fusi.affine.apply(transform),
        resample_fill_value=session.fill_value,
    )
    assert layer.data.shape == expected.shape
    np.testing.assert_allclose(layer.scale, kwargs["scale"])
    np.testing.assert_allclose(layer.translate, kwargs["translate"])
    plane = int(viewer.dims.current_step[0])
    np.testing.assert_allclose(
        layer._slice.image.raw, np.asarray(expected)[plane], atol=1e-10
    )


def test_metadata_unchanged_during_preview(make_napari_viewer, volume):
    viewer = make_napari_viewer()
    layer = _add_layer(viewer, volume)
    displayed = layer.metadata["xarray"]
    source = layer.metadata["source_xarray"]
    session = PreviewSession(layer)

    session.show(_rotation(volume).matrix())

    assert layer.metadata["xarray"] is displayed
    assert layer.metadata["source_xarray"] is source


def test_restore_returns_the_original_layer(make_napari_viewer, volume):
    viewer = make_napari_viewer()
    layer = _add_layer(viewer, volume)
    original = layer.data
    scale, translate = layer.scale.copy(), layer.translate.copy()
    session = PreviewSession(layer)
    session.show(_rotation(volume).matrix())

    session.restore()

    assert layer.data is original
    np.testing.assert_array_equal(layer.scale, scale)
    np.testing.assert_array_equal(layer.translate, translate)


def test_show_identity_restores(make_napari_viewer, volume):
    viewer = make_napari_viewer()
    layer = _add_layer(viewer, volume)
    original = layer.data
    session = PreviewSession(layer)
    session.show(_rotation(volume).matrix())

    session.show(np.eye(4))

    assert layer.data is original


def test_restore_without_a_preview_changes_nothing(make_napari_viewer, volume):
    viewer = make_napari_viewer()
    layer = _add_layer(viewer, volume)
    original = layer.data

    PreviewSession(layer).restore()

    assert layer.data is original


def test_preview_can_restart_after_restore(make_napari_viewer, volume):
    viewer = make_napari_viewer()
    layer = _add_layer(viewer, volume)
    original = layer.data
    session = PreviewSession(layer)
    transform = _rotation(volume).matrix()
    session.show(transform)
    session.restore()

    session.show(transform)
    assert layer.data is not original

    session.restore()
    assert layer.data is original


def test_time_dim_keeps_its_scale_and_translate(make_napari_viewer, series):
    viewer = make_napari_viewer()
    layer = _add_layer(viewer, series)
    time_scale, time_translate = layer.scale[0], layer.translate[0]
    session = PreviewSession(layer)

    session.show(_rotation(series).matrix())

    assert layer.data.shape[0] == series.sizes["time"]
    assert layer.scale[0] == time_scale
    assert layer.translate[0] == time_translate


def test_drag_keeps_shape_and_grid(make_napari_viewer, volume):
    viewer = make_napari_viewer()
    layer = _add_layer(viewer, volume)
    session = PreviewSession(layer)
    state = _rotation(volume)
    axis = state.rotation_axis("x", frame="global")

    session.begin_rotation_drag(state.matrix(), axis, state.pivot())
    shape = layer.data.shape
    scale, translate = layer.scale.copy(), layer.translate.copy()

    for angle in (-30.0, -2.0, 11.0, 40.0):
        session.drag_to(state.rotated("x", angle, frame="global").matrix())

        assert layer.data.shape == shape
        np.testing.assert_array_equal(layer.scale, scale)
        np.testing.assert_array_equal(layer.translate, translate)


def test_drag_displays_the_transform_it_was_given(make_napari_viewer, volume):
    viewer = make_napari_viewer()
    layer = _add_layer(viewer, volume)
    session = PreviewSession(layer)
    state = _rotation(volume)
    session.begin_rotation_drag(
        state.matrix(), state.rotation_axis("x", frame="global"), state.pivot()
    )
    before = layer._slice.image.raw.copy()

    session.drag_to(state.rotated("x", 25.0, frame="global").matrix())

    # A rotation about x moves voxels in the displayed plane.
    assert not np.allclose(layer._slice.image.raw, before)
    plane = int(viewer.dims.current_step[0])
    np.testing.assert_array_equal(layer._slice.image.raw, layer.data[plane, :, :])


def test_show_after_drag_returns_to_the_tight_grid(make_napari_viewer, volume):
    viewer = make_napari_viewer()
    layer = _add_layer(viewer, volume)
    session = PreviewSession(layer)
    state = _rotation(volume)
    session.begin_rotation_drag(
        state.matrix(), state.rotation_axis("x", frame="global"), state.pivot()
    )
    moved = state.rotated("x", 10.0, frame="global").matrix()
    session.drag_to(moved)

    session.show(moved)

    expected, kwargs, _ = build_napari_layer_data(
        volume.fusi.affine.apply(moved), resample_fill_value=session.fill_value
    )
    assert layer.data.shape == expected.shape
    np.testing.assert_allclose(layer.translate, kwargs["translate"])


def test_world_plane_stays_within_a_step_while_the_grid_changes(
    make_napari_viewer, volume
):
    viewer = make_napari_viewer()
    layer = _add_layer(viewer, volume)
    viewer.dims.set_current_step(0, 4)
    point = viewer.dims.point[0]
    session = PreviewSession(layer)
    state = TransformState.identity(center=volume.fusi.affine.geometry_center.to_numpy())

    for angle in (5.0, 10.0, 15.0, 20.0):
        session.show(state.rotated("x", angle, frame="global").matrix())

        assert abs(viewer.dims.point[0] - point) <= layer.scale[0]


def test_cursor_time_course_matches_the_preview(make_napari_viewer, series):
    viewer = make_napari_viewer()
    layer = _add_layer(viewer, series)
    session = PreviewSession(layer)
    session.show(_rotation(series).matrix())
    world = [0.0, *(layer.translate[1:] + layer.scale[1:] * (np.array(layer.data.shape[1:]) // 2))]

    # The read `SignalPlotter._extract_signals` does.
    index = [round(x) for x in layer.world_to_data(world)]
    index[0] = slice(None)
    course = layer.data[tuple(index)]

    expected = [layer.data[(frame, *index[1:])] for frame in range(series.sizes["time"])]
    np.testing.assert_allclose(course, expected, atol=1e-10)


def test_drag_does_not_read_the_source_again(make_napari_viewer, series):
    counting = CountingArray(series.values)
    lazy = da.from_array(
        counting,
        chunks=counting.shape,
        asarray=False,
        fancy=False,
        lock=False,
        name="counted-session",
        meta=np.empty((0,) * counting.ndim, dtype=counting.dtype),
    )
    source = series.copy(data=lazy)
    viewer = make_napari_viewer()
    layer = viewer.add_image(
        series.values, scale=(0.5, 0.4, 0.1, 0.1), metadata={"xarray": source}
    )
    session = PreviewSession(layer)
    state = _rotation(series)
    session.begin_rotation_drag(
        state.matrix(), state.rotation_axis("x", frame="global"), state.pivot()
    )
    reads = counting.reads

    for angle in (5.0, 10.0, 15.0, 20.0):
        session.drag_to(state.rotated("x", angle, frame="global").matrix())

    assert counting.reads == reads


def test_restarting_a_preview_does_not_read_the_source_again(make_napari_viewer, series):
    counting = CountingArray(series.values)
    lazy = da.from_array(
        counting,
        chunks=counting.shape,
        asarray=False,
        fancy=False,
        lock=False,
        name="counted-restart",
        meta=np.empty((0,) * counting.ndim, dtype=counting.dtype),
    )
    viewer = make_napari_viewer()
    layer = viewer.add_image(
        series.values,
        scale=(0.5, 0.4, 0.1, 0.1),
        metadata={"xarray": series.copy(data=lazy)},
    )
    session = PreviewSession(layer)
    transform = _rotation(series).matrix()
    session.show(transform)
    reads = counting.reads
    session.restore()

    session.show(transform)

    assert counting.reads == reads


def test_fill_value_survives_a_restore(make_napari_viewer, volume):
    viewer = make_napari_viewer()
    layer = _add_layer(viewer, volume)
    session = PreviewSession(layer)
    session.show(_rotation(volume).matrix())
    fill_value = session.fill_value

    session.restore()

    assert session.fill_value == fill_value


def test_translation_only_slices_once(make_napari_viewer, volume):
    viewer = make_napari_viewer()
    layer = _add_layer(viewer, volume)
    session = PreviewSession(layer)
    state = _rotation(volume)
    session.show(state.matrix())
    calls = []
    set_view_slice = layer.set_view_slice

    def spy():
        calls.append(1)
        set_view_slice()

    layer.set_view_slice = spy

    session.show(state.translated((0.5, 0.0, 0.0), frame="global").matrix())

    assert len(calls) == 1


def test_same_shape_update_still_displays_the_new_transform(make_napari_viewer, volume):
    viewer = make_napari_viewer()
    layer = _add_layer(viewer, volume)
    session = PreviewSession(layer)
    state = _rotation(volume)
    session.show(state.matrix())
    moved = state.translated((0.0, 0.3, 0.0), frame="global").matrix()

    session.show(moved)

    expected, kwargs, _ = build_napari_layer_data(
        volume.fusi.affine.apply(moved), resample_fill_value=session.fill_value
    )
    plane = int(viewer.dims.current_step[0])
    np.testing.assert_allclose(layer.translate, kwargs["translate"])
    np.testing.assert_allclose(
        layer._slice.image.raw, np.asarray(expected)[plane], atol=1e-10
    )


def test_drag_grid_respects_max_voxels(make_napari_viewer, volume):
    viewer = make_napari_viewer()
    layer = _add_layer(viewer, volume)
    session = PreviewSession(layer)
    state = _rotation(volume)

    session.begin_rotation_drag(
        state.matrix(),
        state.rotation_axis("x", frame="global"),
        state.pivot(),
        max_voxels=500,
    )

    assert np.prod(layer.data.shape) <= 1000


def _labels_of(volume):
    """Return integer labels on `volume`'s grid."""
    return volume.copy(data=(volume.values > 3.0).astype(np.int32))


def _commit_args(session, transform):
    """Return the arguments the panel passes to the layer builders."""
    return {
        "layer_type": session.layer_type,
        "fill_value": session.fill_value,
        "name": "committed",
        "committed_labels": session.committed_labels,
    }


def test_restore_rereads_labels_painted_since(make_napari_viewer, volume):
    viewer = make_napari_viewer()
    layer = _add_layer(viewer, _labels_of(volume))
    session = PreviewSession(layer)
    state = TransformState.identity().translated((0.4, 0.0, 0.0), frame="global")
    session.show(state.matrix())
    session.restore()
    layer.data[2, 3, 4] = 9

    session.show(state.matrix())

    assert np.count_nonzero(np.asarray(layer.data) == 9) == 1


def test_labels_cannot_be_painted_during_a_preview(make_napari_viewer, volume):
    viewer = make_napari_viewer()
    layer = _add_layer(viewer, _labels_of(volume))
    session = PreviewSession(layer)
    state = _rotation(volume)
    assert layer.editable

    session.show(state.matrix())

    assert not layer.editable

    session.show(state.rotated("x", 5.0, frame="global").matrix())

    assert not layer.editable

    session.restore()

    assert layer.editable


def test_paint_on_integer_labels_survives_apply(make_napari_viewer, volume):
    # The layer holds the source's own array, so paint lands in the source too.
    viewer = make_napari_viewer()
    layer = _add_layer(viewer, _labels_of(volume))
    layer.data[2, 3, 4] = 9
    session = PreviewSession(layer)
    transform = TransformState.identity().translated(
        (0.4, 0.0, 0.0), frame="global"
    ).matrix()
    session.show(transform)

    data, kwargs, _ = build_applied_layer(
        session.source, transform, **_commit_args(session, transform)
    )
    session.adopt(data, kwargs)

    assert np.count_nonzero(np.asarray(layer.data) == 9) == 1


def test_apply_refuses_to_lose_paint_on_cast_labels(make_napari_viewer, volume):
    # Float labels are cast for the layer, so the layer holds a copy of the source.
    viewer = make_napari_viewer()
    labels = volume.copy(data=(volume.values > 3.0).astype(np.float64))
    layer = _add_layer(viewer, labels, layer_type="labels")
    session = PreviewSession(layer)
    layer.data[2, 3, 4] = 9
    transform = _rotation(volume).matrix()
    session.show(transform)

    with pytest.raises(ValueError, match="paint"):
        build_applied_layer(
            session.source, transform, **_commit_args(session, transform)
        )


@pytest.mark.parametrize("builder", [build_applied_layer, build_resampled_layer])
def test_commit_refuses_to_lose_paint_on_oblique_labels(make_napari_viewer, volume, builder):
    # The layer displays a resampled copy of an oblique source.
    viewer = make_napari_viewer()
    state = _rotation(volume)
    oblique = _labels_of(volume).fusi.affine.apply(state.matrix())
    layer = _add_layer(viewer, oblique)
    session = PreviewSession(layer)
    layer.data[2, 3, 4] = 9
    transform = state.rotated("y", 5.0, frame="global").matrix()

    with pytest.raises(ValueError, match="paint"):
        builder(session.source, transform, **_commit_args(session, transform))


def test_commit_passes_on_unpainted_oblique_labels(make_napari_viewer, volume):
    viewer = make_napari_viewer()
    state = _rotation(volume)
    oblique = _labels_of(volume).fusi.affine.apply(state.matrix())
    layer = _add_layer(viewer, oblique)
    session = PreviewSession(layer)
    transform = state.rotated("y", 5.0, frame="global").matrix()
    session.show(transform)

    data, kwargs, _ = build_applied_layer(
        session.source, transform, **_commit_args(session, transform)
    )
    session.adopt(data, kwargs)

    # A second commit compares against the layer the first one built.
    session.show(transform)
    build_applied_layer(session.source, transform, **_commit_args(session, transform))


def test_images_are_not_checked_for_paint(make_napari_viewer, volume):
    viewer = make_napari_viewer()
    layer = _add_layer(viewer, volume)

    assert PreviewSession(layer).committed_labels is None


def test_resampled_labels_do_not_share_memory_with_the_source(make_napari_viewer, volume):
    # A translation is not resampled, so the new layer could alias the original.
    labels = _labels_of(volume)
    transform = TransformState.identity().translated(
        (0.4, 0.0, 0.0), frame="global"
    ).matrix()

    array, kwargs, _ = build_resampled_layer(
        labels, transform, layer_type="labels", fill_value=0.0, name="resampled"
    )

    assert not np.shares_memory(np.asarray(array), labels.values)
    assert not np.shares_memory(kwargs["metadata"]["source_xarray"].values, labels.values)


def test_restore_works_when_the_view_cannot_be_read(
    make_napari_viewer, volume, monkeypatch
):
    viewer = make_napari_viewer()
    layer = _add_layer(viewer, volume)
    original = layer.data
    scale = layer.scale.copy()
    session = PreviewSession(layer)
    session.show(_rotation(volume).matrix())

    def fail(self, key):
        raise TypeError("cannot read")

    monkeypatch.setattr(ResampledView, "__getitem__", fail)

    session.restore()

    assert layer.data is original
    np.testing.assert_array_equal(layer.scale, scale)


def test_contrast_full_range_works_during_a_preview(make_napari_viewer, volume):
    # What napari's "full range" contrast button calls, and it reads `layer.data.size`.
    viewer = make_napari_viewer()
    layer = _add_layer(viewer, volume)
    session = PreviewSession(layer)
    session.show(_rotation(volume).matrix())

    layer.reset_contrast_limits_range()

    # napari samples a few planes here, so the range sits within the displayed values.
    low, high = layer.contrast_limits_range
    displayed = np.asarray(layer.data)
    assert displayed.min() <= low < high <= displayed.max()


def test_scrubbing_time_during_a_preview_shows_that_frame(make_napari_viewer, series):
    viewer = make_napari_viewer()
    layer = _add_layer(viewer, series)
    viewer.dims.set_current_step(0, 0)
    session = PreviewSession(layer)
    session.show(_rotation(series).matrix())
    first = layer._slice.image.raw.copy()

    viewer.dims.set_current_step(0, 5)

    # Each frame multiplies the first by (1 + 0.1 * t), so a stale frame would not match.
    inside = first != session.fill_value
    np.testing.assert_allclose(layer._slice.image.raw[inside], first[inside] * 1.5)


def test_switching_to_3d_during_a_preview_shows_the_volume(make_napari_viewer, volume):
    viewer = make_napari_viewer()
    layer = _add_layer(viewer, volume)
    session = PreviewSession(layer)
    session.show(_rotation(volume).matrix())

    viewer.dims.ndisplay = 3

    assert layer._slice.image.raw.shape == layer.data.shape


def test_contrast_limits_stay_during_a_preview(make_napari_viewer, volume):
    viewer = make_napari_viewer()
    layer = _add_layer(viewer, volume, name="volume")
    layer.contrast_limits = (3.1, 3.2)
    session = PreviewSession(layer)

    session.show(_rotation(volume).matrix())

    assert tuple(layer.contrast_limits) == (3.1, 3.2)


def test_refuses_layer_without_metadata(make_napari_viewer):
    viewer = make_napari_viewer()
    layer = viewer.add_image(np.zeros((3, 4, 5)))

    with pytest.raises(ValueError, match="ConfUSIus"):
        PreviewSession(layer)


def test_refuses_pose_stack(make_napari_viewer, sample_voxeldata_3dt_pose):
    viewer = make_napari_viewer()
    layer = viewer.add_image(
        np.zeros(sample_voxeldata_3dt_pose.shape),
        metadata={"xarray": sample_voxeldata_3dt_pose},
    )

    with pytest.raises(ValueError, match="pose"):
        PreviewSession(layer)


def test_refuses_painted_labels(make_napari_viewer, volume):
    viewer = make_napari_viewer()
    labels = volume.copy(data=(volume.values > 3.0).astype(np.int32))
    layer = _add_layer(viewer, labels)
    layer.data = layer.data.copy()
    layer.data[0, 0, 0] = 7

    with pytest.raises(ValueError, match="paint"):
        PreviewSession(layer)


def test_refuses_labels_painted_over_another_array(make_napari_viewer, series):
    # What `draw_napari_labels` builds: a 3D paint layer described by a 4D image.
    viewer = make_napari_viewer()
    layer = viewer.add_labels(
        np.zeros(series.shape[1:], dtype=np.int32), metadata={"xarray": series}
    )

    with pytest.raises(ValueError, match="paint"):
        PreviewSession(layer)


def test_accepts_unpainted_labels(make_napari_viewer, volume):
    viewer = make_napari_viewer()
    labels = volume.copy(data=(volume.values > 3.0).astype(np.int32))
    layer = _add_layer(viewer, labels)
    layer.data = layer.data.copy()

    PreviewSession(layer)


def test_labels_preview_uses_nearest_and_zero_fill(make_napari_viewer, volume):
    viewer = make_napari_viewer()
    values = (volume.values * 10).astype(np.int32)
    labels = volume.copy(data=values)
    layer = _add_layer(viewer, labels)
    session = PreviewSession(layer)

    session.show(_rotation(volume, angle=30.0).matrix())

    displayed = np.asarray(layer.data)
    assert session.fill_value == 0.0
    assert set(np.unique(displayed)) <= set(np.unique(values)) | {0}


def test_build_applied_layer_matches_affine_apply(volume):
    transform = _rotation(volume).matrix()

    _, kwargs, _ = build_applied_layer(
        volume, transform, layer_type="image", fill_value=0.0, name="moved"
    )

    source = kwargs["metadata"]["source_xarray"]
    expected = volume.fusi.affine.apply(transform)
    np.testing.assert_array_equal(
        source.fusi.affine.voxel_to_world, expected.fusi.affine.voxel_to_world
    )
    np.testing.assert_array_equal(source.values, volume.values)
    assert kwargs["name"] == "moved"


def test_build_applied_layer_keeps_stored_affines(volume):
    stored = np.eye(4)
    stored[0, 3] = 5.0
    data = volume.assign_attrs(affines={"to_lab": stored})
    transform = _rotation(volume).matrix()

    _, kwargs, _ = build_applied_layer(
        data, transform, layer_type="image", fill_value=0.0, name="moved"
    )

    np.testing.assert_allclose(
        kwargs["metadata"]["source_xarray"].attrs["affines"]["to_lab"],
        stored @ np.linalg.inv(transform),
    )


def test_adopt_commits_the_transform(make_napari_viewer, volume):
    viewer = make_napari_viewer()
    layer = _add_layer(viewer, volume)
    session = PreviewSession(layer)
    transform = _rotation(volume).matrix()
    session.show(transform)
    fill_value = session.fill_value
    previewed = np.asarray(layer.data)

    data, kwargs, _ = build_applied_layer(
        session.source, transform, layer_type="image", fill_value=fill_value, name="v"
    )
    session.adopt(data, kwargs)

    assert isinstance(layer.data, np.ndarray)
    np.testing.assert_allclose(layer.data, previewed, atol=1e-10)
    np.testing.assert_allclose(layer.scale, kwargs["scale"])
    np.testing.assert_allclose(layer.translate, kwargs["translate"])
    assert layer.metadata["source_xarray"] is kwargs["metadata"]["source_xarray"]
    assert layer.metadata["xarray"] is kwargs["metadata"]["xarray"]
    assert session.source is kwargs["metadata"]["source_xarray"]


def test_restore_after_adopt_keeps_the_committed_layer(make_napari_viewer, volume):
    viewer = make_napari_viewer()
    layer = _add_layer(viewer, volume)
    session = PreviewSession(layer)
    transform = _rotation(volume).matrix()
    session.show(transform)
    data, kwargs, _ = build_applied_layer(
        session.source,
        transform,
        layer_type="image",
        fill_value=session.fill_value,
        name="v",
    )
    session.adopt(data, kwargs)

    session.restore()

    assert layer.data is data


def test_build_resampled_layer_source_holds_the_resampled_voxels(volume):
    transform = _rotation(volume).matrix()

    array, kwargs, _ = build_resampled_layer(
        volume, transform, layer_type="image", fill_value=0.0, name="resampled"
    )

    source = kwargs["metadata"]["source_xarray"]
    assert source.shape == array.shape
    np.testing.assert_array_equal(source.values, np.asarray(array))
    assert source.fusi.affine.voxel_to_world[0, 1] == 0.0


def test_build_resampled_layer_on_a_reference_grid(volume, series):
    reference = create_voxeldata(
        np.zeros((2, 5, 6, 7)),
        dims=("time", "k", "j", "i"),
        time=np.arange(2, dtype=float),
        spacing=(0.5, 0.2, 0.3),
        origin=(0.0, 1.0, 2.0),
    )

    array, kwargs, _ = build_resampled_layer(
        volume,
        _rotation(volume).matrix(),
        layer_type="image",
        fill_value=0.0,
        name="resampled",
        reference=reference,
    )

    assert array.shape == (5, 6, 7)
    np.testing.assert_allclose(kwargs["scale"], (0.5, 0.2, 0.3))
    np.testing.assert_allclose(kwargs["translate"], (0.0, 1.0, 2.0))
    assert kwargs["metadata"]["source_xarray"].shape == (5, 6, 7)


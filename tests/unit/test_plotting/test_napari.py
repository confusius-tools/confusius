"""Tests for napari-based plotting functions."""

import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr

from confusius._utils.geometry import (
    attach_voxel_to_world_index,
    get_voxel_to_world_affine,
    has_axis_aligned_voxel_to_world_index,
    has_voxel_to_world_index,
)
from confusius.plotting import draw_napari_labels, labels_from_layer, plot_napari
from confusius.xarray import create_voxeldata

_VOXEL_DIM_BY_WORLD_NAME = {"z": "k", "y": "j", "x": "i"}


def _world_coord_1d(da: xr.DataArray, name: str) -> np.ndarray:
    """Return a world coordinate's 1D values, reducing other axis-aligned dims."""
    coord = da.coords[name]
    dim = name if name in coord.dims else _VOXEL_DIM_BY_WORLD_NAME[name]
    if coord.dims == (dim,):
        return coord.values
    others = {d: 0 for d in coord.dims if d != dim}
    return coord.isel(others).values


class TestPlotNapari:
    """Tests for plot_napari scale and translate parameters."""

    def test_3d_scale_and_translate(self, sample_voxeldata_3d, make_napari_viewer):
        """3D layer scale matches fusi.spacing; translate matches fusi.origin."""
        viewer = make_napari_viewer()
        _, layer = plot_napari(sample_voxeldata_3d, viewer=viewer)

        # z: origin=1.0 spacing=0.2; y: origin=2.0 spacing=0.1; x: origin=3.0 spacing=0.05
        npt.assert_allclose(layer.scale, [0.2, 0.1, 0.05], rtol=1e-5)
        npt.assert_allclose(layer.translate, [1.0, 2.0, 3.0], rtol=1e-5)
        viewer.close()

    def test_length_three_spatial_axis_not_treated_as_rgb(
        self, sample_voxeldata_3d, make_napari_viewer
    ):
        """A spatial axis of length 3 is not auto-interpreted as RGB channels."""
        data = sample_voxeldata_3d.isel(i=slice(0, 3))
        viewer = make_napari_viewer()
        _, layer = plot_napari(
            data, viewer=viewer, show_colorbar=False, show_scale_bar=False
        )

        assert not layer.rgb
        npt.assert_allclose(layer.scale, [0.2, 0.1, 0.05], rtol=1e-5)
        npt.assert_allclose(layer.translate, [1.0, 2.0, 3.0], rtol=1e-5)
        viewer.close()

    def test_4d_scale_uses_time_spacing(self, sample_voxeldata_3dt, make_napari_viewer):
        """4D layer scale uses fusi.spacing for all dims, including time."""
        viewer = make_napari_viewer()
        _, layer = plot_napari(
            sample_voxeldata_3dt,
            viewer=viewer,
            show_colorbar=False,
            show_scale_bar=False,
        )

        # time: origin=10.0 spacing=0.5; z: origin=1.0 spacing=0.2;
        # y: origin=2.0 spacing=0.1; x: origin=3.0 spacing=0.05
        npt.assert_allclose(layer.scale, [0.5, 0.2, 0.1, 0.05], rtol=1e-5)
        npt.assert_allclose(layer.translate, [10.0, 1.0, 2.0, 3.0], rtol=1e-5)
        viewer.close()

    def test_voxel_to_world_resamples_to_world_grid(
        self, make_napari_viewer, sample_voxeldata_3d_oblique
    ):
        """Oblique volumes are displayed on an axis-aligned world grid in napari.

        The resampled grid stays on native voxel dims with its voxel-to-world index
        intact (like the axis-aligned case); layers are always positioned in world
        space, so axis_labels shows world names regardless.
        """
        data = sample_voxeldata_3d_oblique
        viewer = make_napari_viewer()
        _, layer = plot_napari(
            data, viewer=viewer, show_colorbar=False, show_scale_bar=False
        )

        assert layer.metadata["xarray"].dims == ("k", "j", "i")
        assert has_voxel_to_world_index(layer.metadata["xarray"])
        # `source_xarray` is the canonicalized copy of `data` (plot_napari always
        # canonicalizes its input), not the original object -- compare by value.
        xr.testing.assert_identical(layer.metadata["source_xarray"], data)
        assert tuple(layer.axis_labels) == ("z", "y", "x")
        npt.assert_allclose(layer.translate, [10.0, 20.0, 30.0], rtol=1e-5)
        # The displayed layer is actually resampled onto an axis-aligned grid,
        # unlike the sheared source data.
        assert not has_axis_aligned_voxel_to_world_index(data)
        assert has_axis_aligned_voxel_to_world_index(layer.metadata["xarray"])
        viewer.close()

    def test_axis_aligned_voxel_to_world_uses_world_display_by_default(
        self, make_napari_viewer
    ):
        """Axis-aligned data keeps native voxel dims but shows world display labels."""
        data = xr.DataArray(
            np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4),
            dims=["k", "j", "i"],
            coords={"k": [0, 1], "j": [0, 1, 2], "i": [0, 1, 2, 3]},
        )
        data = attach_voxel_to_world_index(
            data,
            np.diag([0.4, 0.3, 0.25, 1.0]),
        )
        viewer = make_napari_viewer()
        _, layer = plot_napari(
            data, viewer=viewer, show_colorbar=False, show_scale_bar=False
        )

        # Both are canonicalized copies of `data`, not the original object.
        xr.testing.assert_identical(layer.metadata["xarray"], data)
        xr.testing.assert_identical(layer.metadata["source_xarray"], data)
        assert tuple(layer.axis_labels) == ("z", "y", "x")
        npt.assert_allclose(layer.scale, [0.4, 0.3, 0.25], rtol=1e-5)
        npt.assert_allclose(layer.translate, [0.0, 0.0, 0.0], rtol=1e-5)
        viewer.close()

    def test_scale_falls_back_to_1_when_no_coords(
        self, sample_voxeldata_3d, make_napari_viewer
    ):
        """A dim without a coordinate uses scale=1.0 and translate=0.0."""
        da = sample_voxeldata_3d.expand_dims({"channel": 2})
        viewer = make_napari_viewer()
        with pytest.warns(UserWarning):
            _, layer = plot_napari(
                da, viewer=viewer, show_colorbar=False, show_scale_bar=False
            )

        assert layer.scale[0] == pytest.approx(1.0)
        assert layer.translate[0] == pytest.approx(0.0)
        viewer.close()

    def test_dim_order_reorders_4d_display_axes(
        self, sample_voxeldata_3dt, make_napari_viewer
    ):
        """`dim_order` reorders the spatial display axes; time is kept first."""
        viewer = make_napari_viewer()
        plot_napari(
            sample_voxeldata_3dt,
            viewer=viewer,
            dim_order=("j", "k", "i"),
            show_colorbar=False,
            show_scale_bar=False,
        )
        # all_dims = (time, k, j, i); requested = (j, k, i) → indices (2, 1, 3),
        # prepended with the time-dim index (0).
        assert tuple(viewer.dims.order) == (0, 2, 1, 3)
        viewer.close()

    def test_dim_order_mismatch_raises(self, sample_voxeldata_3dt, make_napari_viewer):
        """`dim_order` must list every spatial dim by name."""
        viewer = make_napari_viewer()
        with pytest.raises(ValueError, match="dim_order"):
            plot_napari(
                sample_voxeldata_3dt,
                viewer=viewer,
                dim_order=("k", "j", "foo"),
                show_colorbar=False,
                show_scale_bar=False,
            )
        viewer.close()

    def test_default_dim_order_puts_singleton_planar_axis_first(
        self, make_napari_viewer
    ):
        """A singleton spatial axis defaults to a slider, not the canvas.

        Reproduces the napari-side issue reported on #407: a permuted
        voxel-to-world affine can map a data array's singleton native dim (the
        planar acquisition's elevation axis) onto any world axis after resampling
        to an axis-aligned grid, not necessarily the one that ends up last in
        native dim order. Without inferring this, napari's default "last two axes
        are the canvas" convention could put the singleton axis in the canvas.
        """
        # Permutation affine: world z <- native i, world y <- native j,
        # world x <- native k. Native k is the singleton axis, so after resampling
        # to an axis-aligned grid it ends up mapped to world x, and (being native
        # k) it lands first rather than last in native dim order.
        voxel_to_world = np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        data = create_voxeldata(
            np.arange(1 * 3 * 4, dtype=float).reshape(1, 3, 4),
            dims=("k", "j", "i"),
            voxel_to_world=voxel_to_world,
        )
        viewer = make_napari_viewer()
        _, layer = plot_napari(
            data, viewer=viewer, show_colorbar=False, show_scale_bar=False
        )

        singleton_dim = next(
            i
            for i, size in enumerate(layer.metadata["xarray"].sizes.values())
            if size == 1
        )
        assert singleton_dim not in viewer.dims.order[-2:]
        viewer.close()

    def test_labels_layer_preserves_xarray_metadata(
        self, sample_voxeldata_3dt, make_napari_viewer
    ):
        """Labels layers keep the source DataArray in napari metadata."""
        labels = xr.DataArray(
            (sample_voxeldata_3dt > 0.5).astype(np.int32),
            dims=sample_voxeldata_3dt.dims,
            coords=sample_voxeldata_3dt.coords,
            attrs=sample_voxeldata_3dt.attrs,
        )
        viewer = make_napari_viewer()
        _, layer = plot_napari(
            labels,
            viewer=viewer,
            layer_type="labels",
            show_colorbar=False,
            show_scale_bar=False,
        )

        # `xarray` is the canonicalized copy of `labels`, not the original object.
        xr.testing.assert_identical(layer.metadata["xarray"], labels)
        viewer.close()

    def test_invalid_layer_type_raises(self, sample_voxeldata_3d) -> None:
        with pytest.raises(ValueError, match="Unknown layer_type"):
            plot_napari(sample_voxeldata_3d, layer_type="bogus")  # ty: ignore[invalid-argument-type]

    def test_without_voxel_to_world_index_raises(self) -> None:
        """plot_napari raises clearly without a voxel-to-world index."""
        data = xr.DataArray(np.zeros((2, 3, 4)), dims=("k", "j", "i"))

        with pytest.raises(ValueError, match="VoxelToWorldIndex"):
            plot_napari(data)

    def test_non_uniform_spatial_coords_warn(
        self, sample_voxeldata_3d, make_napari_viewer
    ):
        data = attach_voxel_to_world_index(
            sample_voxeldata_3d.assign_coords(j=[2, 3, 5, 6, 7, 9]),
            get_voxel_to_world_affine(sample_voxeldata_3d),
        )
        viewer = make_napari_viewer()
        with pytest.warns(UserWarning, match="non-uniform spacing"):
            _, _ = plot_napari(
                data,
                viewer=viewer,
                show_colorbar=False,
                show_scale_bar=False,
            )
        viewer.close()

    def test_image_attrs_cmap_is_forwarded(
        self, sample_voxeldata_3d, make_napari_viewer
    ):
        data = sample_voxeldata_3d.copy()
        data.attrs["cmap"] = "magma"
        viewer = make_napari_viewer()
        _, layer = plot_napari(
            data,
            viewer=viewer,
            show_colorbar=False,
            show_scale_bar=False,
        )

        assert layer.colormap.name == "magma"
        viewer.close()

    def test_labels_without_viewer_create_one_and_cast_to_int(
        self, make_napari_viewer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A single 2D slice: singleton `k` restored as a scalar coordinate, matching
        # what `.isel(k=0)` produces from a real 3D VoxelData array.
        labels = create_voxeldata(
            np.array([[[0.0, 1.0], [2.0, 0.0]]], dtype=np.float32),
            dims=("k", "j", "i"),
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
        ).isel(k=0)
        viewer = make_napari_viewer()
        monkeypatch.setattr("confusius.plotting.napari.napari.Viewer", lambda: viewer)

        created_viewer, layer = plot_napari(
            labels,
            viewer=None,
            layer_type="labels",
            show_colorbar=False,
            show_scale_bar=False,
        )

        assert created_viewer is viewer
        assert np.issubdtype(np.asarray(layer.data).dtype, np.integer)
        viewer.close()

    def test_complex_data_warns_and_plots_magnitude(
        self, sample_voxeldata_iq_3dt, make_napari_viewer
    ):
        """Complex-valued image data is converted to magnitude with a warning."""
        viewer = make_napari_viewer()
        with pytest.warns(UserWarning, match="Complex-valued data"):
            _, layer = plot_napari(
                sample_voxeldata_iq_3dt,
                viewer=viewer,
                show_colorbar=False,
                show_scale_bar=False,
            )

        assert np.issubdtype(np.asarray(layer.data).dtype, np.floating)
        npt.assert_allclose(
            np.asarray(layer.data), np.abs(sample_voxeldata_iq_3dt.data)
        )
        viewer.close()

    def test_non_monotonic_coords_are_sorted_before_napari(
        self, sample_voxeldata_3d, make_napari_viewer
    ):
        """plot_napari sorts non-monotonic spatial coordinates before display.

        Voxel dims may run in either direction and still be valid VoxelData (the
        affine, not coordinate direction, encodes orientation), so reversing `j`/`i`
        keeps `data` valid while still requiring the pre-display sort.
        """
        data = sample_voxeldata_3d.copy().isel(
            j=slice(None, None, -1), i=slice(None, None, -1)
        )

        viewer = make_napari_viewer()
        _, layer = plot_napari(
            data,
            viewer=viewer,
            show_colorbar=False,
            show_scale_bar=False,
        )

        y_sorted = np.sort(_world_coord_1d(data, "y").astype(float))
        x_sorted = np.sort(_world_coord_1d(data, "x").astype(float))
        npt.assert_allclose(
            layer.translate, [1.0, float(y_sorted[0]), float(x_sorted[0])], rtol=1e-5
        )
        result_da = layer.metadata["xarray"]
        assert np.all(np.diff(_world_coord_1d(result_da, "y")) > 0)
        assert np.all(np.diff(_world_coord_1d(result_da, "x")) > 0)
        viewer.close()

    # Image comparison tests with pytest-mpl
    # These generate baseline images for visual regression testing

    def test_napari_labels_hover_shows_roi_name(self, make_napari_viewer):
        """`plot_napari(layer_type='labels')` makes napari's status bar show ROI names.

        Sets `attrs["roi_labels"]` on a tiny integer label map; calls
        `plot_napari(..., layer_type="labels")`; then directly invokes
        `Labels.get_status` (the function napari calls to populate the status
        bar) at one labelled and one background voxel.
        """
        roi_labels = {7: "somatosensory", 42: "visual"}
        # A single 2D slice: singleton `k` restored as a scalar coordinate, matching
        # what `.isel(k=0)` produces from a real 3D VoxelData array.
        labels = create_voxeldata(
            np.array([[[0, 0, 0, 0], [0, 7, 7, 0], [0, 7, 42, 0], [0, 0, 42, 0]]]),
            dims=("k", "j", "i"),
            spacing=(1.0, 0.5, 0.5),
            origin=(0.0, 0.0, 0.0),
            attrs={"roi_labels": roi_labels},
        ).isel(k=0)

        viewer = make_napari_viewer()
        _, layer = plot_napari(
            labels,
            viewer=viewer,
            layer_type="labels",
            show_scale_bar=False,
        )

        # `world=True` means positions are in world coordinates (the same
        # space the user hovers in the canvas).
        # Voxel (y=0.5, x=0.5) holds label 7.
        roi_status = layer.get_status(
            (0.5, 0.5),
            view_direction=np.array([1.0, 0.0]),
            dims_displayed=[0, 1],
            world=True,
        )
        assert "name: somatosensory" in roi_status["coordinates"]

        # Background voxel: NaN row hides the default `[No Properties]` placeholder.
        bg_status = layer.get_status(
            (0.0, 0.0),
            view_direction=np.array([1.0, 0.0]),
            dims_displayed=[0, 1],
            world=True,
        )
        assert "[No Properties]" not in bg_status["coordinates"]
        viewer.close()


class TestDrawNapariLabels:
    """Tests for draw_napari_labels."""

    def test_without_voxel_to_world_index_raises(self) -> None:
        """draw_napari_labels raises clearly without a voxel-to-world index."""
        data = xr.DataArray(np.zeros((2, 3, 4)), dims=("k", "j", "i"))

        with pytest.raises(ValueError, match="VoxelToWorldIndex"):
            draw_napari_labels(data)

    def test_labels_scale_translate_match_image(
        self, sample_voxeldata_3d, make_napari_viewer
    ):
        """Labels overlay shares the image layer's world frame."""
        viewer = make_napari_viewer()
        _, labels_layer = draw_napari_labels(
            sample_voxeldata_3d,
            viewer=viewer,
            show_colorbar=False,
            show_scale_bar=False,
        )
        # sample_voxeldata_3d: z spacing 0.2, y 0.1, x 0.05; origins 1.0, 2.0, 3.0.
        npt.assert_allclose(labels_layer.scale, [0.2, 0.1, 0.05], rtol=1e-5)
        npt.assert_allclose(labels_layer.translate, [1.0, 2.0, 3.0], rtol=1e-5)
        viewer.close()

    def test_oblique_labels_match_resampled_image_layer(
        self, sample_voxeldata_3d_oblique, make_napari_viewer
    ):
        """Labels overlay uses the same displayed grid as the image layer."""
        viewer = make_napari_viewer()
        _, labels_layer = draw_napari_labels(
            sample_voxeldata_3d_oblique,
            viewer=viewer,
            show_colorbar=False,
            show_scale_bar=False,
        )
        image_layer = viewer.layers[0]

        assert labels_layer.data.shape == image_layer.data.shape
        npt.assert_allclose(labels_layer.scale, image_layer.scale, rtol=1e-5)
        npt.assert_allclose(labels_layer.translate, image_layer.translate, rtol=1e-5)
        assert tuple(labels_layer.axis_labels) == tuple(image_layer.axis_labels)
        painted = np.asarray(labels_layer.data)
        painted[0, 0, 0] = 1
        label_map = labels_from_layer(labels_layer, sample_voxeldata_3d_oblique)
        assert label_map.shape[1:] == labels_layer.data.shape
        assert has_voxel_to_world_index(label_map)
        viewer.close()

    def test_strips_time_dim_from_labels_shape(
        self, sample_voxeldata_3dt, make_napari_viewer
    ):
        """A reference with a `time` dim produces a spatial-only labels
        layer."""
        viewer = make_napari_viewer()
        _, labels_layer = draw_napari_labels(
            sample_voxeldata_3dt,
            viewer=viewer,
            show_colorbar=False,
            show_scale_bar=False,
        )
        assert labels_layer.data.shape == (4, 6, 8)
        viewer.close()


class TestLabelsFromLayer:
    """Tests for labels_from_layer."""

    def test_no_labels_raises(self, sample_voxeldata_3d, make_napari_viewer) -> None:
        viewer = make_napari_viewer()
        _, labels_layer = draw_napari_labels(
            sample_voxeldata_3d,
            viewer=viewer,
            show_colorbar=False,
            show_scale_bar=False,
        )
        with pytest.raises(ValueError, match="non-zero labels"):
            labels_from_layer(labels_layer, sample_voxeldata_3d)
        viewer.close()

    def test_multiple_labels_stack_as_sorted_disjoint_slices(
        self, sample_roi_labels, make_napari_viewer
    ) -> None:
        """Each unique non-zero label becomes its own mask slice, sorted ascending."""
        viewer = make_napari_viewer()
        _, labels_layer = plot_napari(
            sample_roi_labels,
            viewer=viewer,
            layer_type="labels",
            show_scale_bar=False,
        )

        result = labels_from_layer(labels_layer, sample_roi_labels)

        # np.unique sorts ascending: motor=3, somatosensory=7, visual=42.
        npt.assert_array_equal(result.coords["mask"].values, [3, 7, 42])
        for label in (3, 7, 42):
            npt.assert_array_equal(
                result.sel(mask=label).values,
                np.where(sample_roi_labels.values == label, label, 0).astype(np.int32),
            )
        viewer.close()

    def test_preserves_spatial_coordinates(
        self, sample_roi_labels, make_napari_viewer
    ) -> None:
        viewer = make_napari_viewer()
        _, labels_layer = plot_napari(
            sample_roi_labels,
            viewer=viewer,
            layer_type="labels",
            show_scale_bar=False,
        )

        result = labels_from_layer(labels_layer, sample_roi_labels)

        # World coordinates must still be index-derived, not materialized plain
        # arrays (see AGENTS.md: world coordinates are never stored directly).
        assert has_voxel_to_world_index(result)
        for dim in ("z", "y", "x"):
            npt.assert_array_equal(
                result.coords[dim].values, sample_roi_labels.coords[dim].values
            )
        viewer.close()

    def test_drops_time_from_reference(
        self, sample_voxeldata_3dt, sample_roi_labels, make_napari_viewer
    ) -> None:
        """A 4D reference array produces a purely spatial output."""
        viewer = make_napari_viewer()
        _, labels_layer = plot_napari(
            sample_roi_labels,
            viewer=viewer,
            layer_type="labels",
            show_scale_bar=False,
        )

        result = labels_from_layer(labels_layer, sample_voxeldata_3dt)

        assert result.dims == ("mask", "k", "j", "i")
        viewer.close()

    def test_attrs_round_trip_label_metadata(
        self, sample_roi_labels, make_napari_viewer
    ) -> None:
        """Layer name and per-label colors round-trip through napari exactly."""
        viewer = make_napari_viewer()
        _, labels_layer = plot_napari(
            sample_roi_labels,
            viewer=viewer,
            layer_type="labels",
            name="hand_drawn",
            show_scale_bar=False,
        )

        result = labels_from_layer(labels_layer, sample_roi_labels)

        assert result.attrs["long_name"] == "Drawn label map"
        assert result.attrs["labels_layer_name"] == "hand_drawn"
        # Fixture's rgb_lookup must come back exactly via napari's
        # DirectLabelColormap built by build_atlas_cmap_and_norm.
        for label, expected_rgb in sample_roi_labels.attrs["rgb_lookup"].items():
            assert result.attrs["rgb_lookup"][label] == expected_rgb
        viewer.close()

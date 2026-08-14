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

    def test_3d_scale_and_translate(self, sample_fusi_3d, make_napari_viewer):
        """3D layer scale matches fusi.spacing; translate matches fusi.origin."""
        viewer = make_napari_viewer()
        _, layer = plot_napari(sample_fusi_3d, viewer=viewer)

        # z: origin=1.0 spacing=0.2; y: origin=2.0 spacing=0.1; x: origin=3.0 spacing=0.05
        npt.assert_allclose(layer.scale, [0.2, 0.1, 0.05], rtol=1e-5)
        npt.assert_allclose(layer.translate, [1.0, 2.0, 3.0], rtol=1e-5)
        viewer.close()

    def test_length_three_spatial_axis_not_treated_as_rgb(
        self, sample_fusi_3d, make_napari_viewer
    ):
        """A spatial axis of length 3 is not auto-interpreted as RGB channels."""
        data = sample_fusi_3d.isel(i=slice(0, 3))
        viewer = make_napari_viewer()
        _, layer = plot_napari(
            data, viewer=viewer, show_colorbar=False, show_scale_bar=False
        )

        assert not layer.rgb
        npt.assert_allclose(layer.scale, [0.2, 0.1, 0.05], rtol=1e-5)
        npt.assert_allclose(layer.translate, [1.0, 2.0, 3.0], rtol=1e-5)
        viewer.close()

    def test_4d_scale_uses_time_spacing(self, sample_fusi_3dt, make_napari_viewer):
        """4D layer scale uses fusi.spacing for all dims, including time."""
        viewer = make_napari_viewer()
        _, layer = plot_napari(
            sample_fusi_3dt, viewer=viewer, show_colorbar=False, show_scale_bar=False
        )

        # time: origin=10.0 spacing=0.5; z: origin=1.0 spacing=0.2;
        # y: origin=2.0 spacing=0.1; x: origin=3.0 spacing=0.05
        npt.assert_allclose(layer.scale, [0.5, 0.2, 0.1, 0.05], rtol=1e-5)
        npt.assert_allclose(layer.translate, [10.0, 1.0, 2.0, 3.0], rtol=1e-5)
        viewer.close()

    def test_voxel_to_world_resamples_to_world_grid(
        self, make_napari_viewer, sample_fusi_3d_oblique
    ):
        """Oblique volumes are displayed on an axis-aligned world grid in napari.

        The resampled grid stays on native voxel dims with its voxel-to-world index
        intact (like the axis-aligned case); layers are always positioned in world
        space, so axis_labels shows world names regardless.
        """
        data = sample_fusi_3d_oblique
        viewer = make_napari_viewer()
        _, layer = plot_napari(
            data, viewer=viewer, show_colorbar=False, show_scale_bar=False
        )

        assert layer.metadata["xarray"].dims == ("k", "j", "i")
        assert has_voxel_to_world_index(layer.metadata["xarray"])
        assert layer.metadata["source_xarray"] is data
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
            coords={"k": [0.0, 1.0], "j": [0.0, 1.0, 2.0], "i": [0.0, 1.0, 2.0, 3.0]},
        )
        data = attach_voxel_to_world_index(
            data,
            np.diag([0.4, 0.3, 0.25, 1.0]),
            world_coord_attrs={
                "z": {"units": "mm"},
                "y": {"units": "mm"},
                "x": {"units": "mm"},
            },
        )
        viewer = make_napari_viewer()
        _, layer = plot_napari(
            data, viewer=viewer, show_colorbar=False, show_scale_bar=False
        )

        assert layer.metadata["xarray"] is data
        assert layer.metadata["source_xarray"] is data
        assert tuple(layer.axis_labels) == ("z", "y", "x")
        npt.assert_allclose(layer.scale, [0.4, 0.3, 0.25], rtol=1e-5)
        npt.assert_allclose(layer.translate, [0.0, 0.0, 0.0], rtol=1e-5)
        viewer.close()

    def test_scale_falls_back_to_1_when_no_coords(
        self, sample_fusi_3d, make_napari_viewer
    ):
        """A dim without a coordinate uses scale=1.0 and translate=0.0."""
        da = sample_fusi_3d.expand_dims({"channel": 2})
        viewer = make_napari_viewer()
        with pytest.warns(UserWarning):
            _, layer = plot_napari(
                da, viewer=viewer, show_colorbar=False, show_scale_bar=False
            )

        assert layer.scale[0] == pytest.approx(1.0)
        assert layer.translate[0] == pytest.approx(0.0)
        viewer.close()

    def test_dim_order_reorders_4d_display_axes(
        self, sample_fusi_3dt, make_napari_viewer
    ):
        """`dim_order` reorders the spatial display axes; time is kept first."""
        viewer = make_napari_viewer()
        plot_napari(
            sample_fusi_3dt,
            viewer=viewer,
            dim_order=("j", "k", "i"),
            show_colorbar=False,
            show_scale_bar=False,
        )
        # all_dims = (time, k, j, i); requested = (j, k, i) → indices (2, 1, 3),
        # prepended with the time-dim index (0).
        assert tuple(viewer.dims.order) == (0, 2, 1, 3)
        viewer.close()

    def test_dim_order_mismatch_raises(self, sample_fusi_3dt, make_napari_viewer):
        """`dim_order` must list every spatial dim by name."""
        viewer = make_napari_viewer()
        with pytest.raises(ValueError, match="dim_order"):
            plot_napari(
                sample_fusi_3dt,
                viewer=viewer,
                dim_order=("k", "j", "foo"),
                show_colorbar=False,
                show_scale_bar=False,
            )
        viewer.close()

    def test_labels_layer_preserves_xarray_metadata(
        self, sample_fusi_3dt, make_napari_viewer
    ):
        """Labels layers keep the source DataArray in napari metadata."""
        labels = xr.DataArray(
            (sample_fusi_3dt > 0.5).astype(np.int32),
            dims=sample_fusi_3dt.dims,
            coords=sample_fusi_3dt.coords,
            attrs=sample_fusi_3dt.attrs,
        )
        viewer = make_napari_viewer()
        _, layer = plot_napari(
            labels,
            viewer=viewer,
            layer_type="labels",
            show_colorbar=False,
            show_scale_bar=False,
        )

        assert layer.metadata["xarray"] is labels
        viewer.close()

    def test_invalid_layer_type_raises(self, sample_fusi_3d) -> None:
        with pytest.raises(ValueError, match="Unknown layer_type"):
            plot_napari(sample_fusi_3d, layer_type="bogus")  # ty: ignore[invalid-argument-type]

    def test_non_uniform_spatial_coords_warn(self, sample_fusi_3d, make_napari_viewer):
        data = attach_voxel_to_world_index(
            sample_fusi_3d.assign_coords(j=[2.0, 2.1, 2.4, 2.6, 2.7, 2.9]),
            get_voxel_to_world_affine(sample_fusi_3d),
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

    def test_image_attrs_cmap_is_forwarded(self, sample_fusi_3d, make_napari_viewer):
        data = sample_fusi_3d.copy()
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
        labels = attach_voxel_to_world_index(
            xr.DataArray(
                np.array([[0.0, 1.0], [2.0, 0.0]], dtype=np.float32),
                dims=["j", "i"],
                coords={"j": [0.0, 1.0], "i": [0.0, 1.0]},
            ),
            np.eye(3),
        )
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
        self, sample_iq_3dt, make_napari_viewer
    ):
        """Complex-valued image data is converted to magnitude with a warning."""
        viewer = make_napari_viewer()
        with pytest.warns(UserWarning, match="Complex-valued data"):
            _, layer = plot_napari(
                sample_iq_3dt,
                viewer=viewer,
                show_colorbar=False,
                show_scale_bar=False,
            )

        assert np.issubdtype(np.asarray(layer.data).dtype, np.floating)
        npt.assert_allclose(np.asarray(layer.data), np.abs(sample_iq_3dt.data))
        viewer.close()

    def test_non_monotonic_coords_are_sorted_before_napari(
        self, sample_fusi_3d, make_napari_viewer
    ):
        """plot_napari sorts non-monotonic spatial coordinates before display."""
        data = sample_fusi_3d.copy().isel(j=[2, 0, 1], i=[3, 1, 2, 0])

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
        labels = attach_voxel_to_world_index(
            xr.DataArray(
                np.array([[0, 0, 0, 0], [0, 7, 7, 0], [0, 7, 42, 0], [0, 0, 42, 0]]),
                dims=["j", "i"],
                coords={"j": [0.0, 1.0, 2.0, 3.0], "i": [0.0, 1.0, 2.0, 3.0]},
                attrs={"roi_labels": roi_labels},
            ),
            np.diag([0.5, 0.5, 1.0]),
        )

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

    def test_labels_scale_translate_match_image(
        self, sample_fusi_3d, make_napari_viewer
    ):
        """Labels overlay shares the image layer's world frame."""
        viewer = make_napari_viewer()
        _, labels_layer = draw_napari_labels(
            sample_fusi_3d,
            viewer=viewer,
            show_colorbar=False,
            show_scale_bar=False,
        )
        # sample_fusi_3d: z spacing 0.2, y 0.1, x 0.05; origins 1.0, 2.0, 3.0.
        npt.assert_allclose(labels_layer.scale, [0.2, 0.1, 0.05], rtol=1e-5)
        npt.assert_allclose(labels_layer.translate, [1.0, 2.0, 3.0], rtol=1e-5)
        viewer.close()

    def test_strips_time_dim_from_labels_shape(
        self, sample_fusi_3dt, make_napari_viewer
    ):
        """A reference with a `time` dim produces a spatial-only labels
        layer."""
        viewer = make_napari_viewer()
        _, labels_layer = draw_napari_labels(
            sample_fusi_3dt,
            viewer=viewer,
            show_colorbar=False,
            show_scale_bar=False,
        )
        assert labels_layer.data.shape == (4, 6, 8)
        viewer.close()


class TestLabelsFromLayer:
    """Tests for labels_from_layer."""

    def test_no_labels_returns_empty_mask_stack(
        self, sample_fusi_3d, make_napari_viewer
    ) -> None:
        viewer = make_napari_viewer()
        _, labels_layer = draw_napari_labels(
            sample_fusi_3d,
            viewer=viewer,
            show_colorbar=False,
            show_scale_bar=False,
        )
        result = labels_from_layer(labels_layer, sample_fusi_3d)
        assert result.dims == ("mask", "k", "j", "i")
        assert result.sizes["mask"] == 0
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

        for dim in ("z", "y", "x"):
            npt.assert_array_equal(
                result.coords[dim].values, sample_roi_labels.coords[dim].values
            )
        viewer.close()

    def test_drops_time_from_reference(
        self, sample_fusi_3dt, sample_roi_labels, make_napari_viewer
    ) -> None:
        """A 4D reference array produces a purely spatial output."""
        viewer = make_napari_viewer()
        _, labels_layer = plot_napari(
            sample_roi_labels,
            viewer=viewer,
            layer_type="labels",
            show_scale_bar=False,
        )

        result = labels_from_layer(labels_layer, sample_fusi_3dt)

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

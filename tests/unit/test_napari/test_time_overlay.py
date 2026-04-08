"""Unit tests for the _TimeOverlay class."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import xarray as xr

from confusius._napari._time_overlay import _TimeOverlay
from confusius.plotting import plot_napari


@pytest.fixture
def nifti_4d_ms(tmp_path: Path) -> Path:
    """4D NIfTI file with time units set to milliseconds."""
    shape = (4, 6, 8, 10)  # x, y, z, time (NIfTI convention)
    data = np.random.default_rng(0).random(shape).astype(np.float32)
    affine = np.diag([0.1, 0.1, 0.2, 1.0])
    img = nib.Nifti1Image(data, affine)
    # NIfTI xyzt_units encodes spatial + time units in a single byte.
    # spatial=mm (0x02), time=msec (0x10) → 0x12 = 18.
    img.header.set_xyzt_units(xyz="mm", t="msec")
    img.header["pixdim"][4] = 50.0  # 50 ms per frame
    path = tmp_path / "vol_ms.nii.gz"
    img.to_filename(path)
    return path


def _make_4d_da(
    rng,
    time_coords,
    *,
    time_units="s",
    spatial_translate=(0.0, 0.0, 0.0),
):
    """Create a minimal 4D DataArray with the given time coordinates."""
    shape = (len(time_coords), 4, 6, 8)
    data = rng.random(shape).astype(np.float32)
    z0, y0, x0 = spatial_translate
    return xr.DataArray(
        data,
        dims=["time", "z", "y", "x"],
        coords={
            "time": xr.DataArray(
                time_coords, dims=["time"], attrs={"units": time_units}
            ),
            "z": xr.DataArray(z0 + np.arange(4) * 0.2, dims=["z"]),
            "y": xr.DataArray(y0 + np.arange(6) * 0.1, dims=["y"]),
            "x": xr.DataArray(x0 + np.arange(8) * 0.05, dims=["x"]),
        },
    )


class TestTimeOverlay:
    """_TimeOverlay reads time units from layer xarray metadata."""

    def test_reads_units_from_nifti_layer(self, nifti_4d_ms: Path, make_napari_viewer):
        """Units are correctly read as 'ms' from a 4D NIfTI loaded via the reader."""
        from confusius._napari._io._readers import read_nifti

        viewer = make_napari_viewer()
        overlay = _TimeOverlay(viewer)

        reader = read_nifti(str(nifti_4d_ms))
        assert reader is not None
        layer_data_list = reader(str(nifti_4d_ms))
        data, kwargs, layer_type = layer_data_list[0]
        viewer._add_layer_from_data(data, kwargs, layer_type)

        # After adding a 4D layer, time should be a slider (not displayed).
        overlay.check()
        assert overlay._active
        assert overlay._time_idx is not None
        assert overlay._units == "ms"
        assert "ms" in viewer.text_overlay.text

    def test_updates_text_when_time_step_changes(
        self, sample_4d_volume, make_napari_viewer
    ) -> None:
        """Changing the time slider updates the overlay text."""
        viewer = make_napari_viewer()
        _, _ = plot_napari(
            sample_4d_volume,
            viewer=viewer,
            show_colorbar=False,
            show_scale_bar=False,
        )
        overlay = _TimeOverlay(viewer)

        overlay.check()
        assert overlay._time_idx is not None

        viewer.dims.set_current_step(overlay._time_idx, 3)

        expected_time = float(sample_4d_volume.coords["time"].values[3])
        expected = f"{expected_time:.2f} s"
        assert viewer.text_overlay.text == expected
        assert viewer.text_overlay.visible

    def test_nonuniform_time_matches_xarray_coords(
        self, rng, make_napari_viewer
    ) -> None:
        """Overlay must show the actual xarray coordinate, not the linear approximation.

        When time spacing is non-uniform, napari's ``dims.point`` uses a linear
        scale/translate approximation (median spacing) that diverges from the
        true coordinate values. The overlay should display the real coordinate
        so it stays consistent with the signal plotter's x-axis cursor.
        """
        # Non-uniform time: large gaps followed by small ones.
        time_coords = np.array([0.0, 0.5, 2.0, 2.1, 5.0])
        da = _make_4d_da(rng, time_coords)

        viewer = make_napari_viewer()
        with pytest.warns(UserWarning, match="non-uniform spacing"):
            _, _ = plot_napari(
                da, viewer=viewer, show_colorbar=False, show_scale_bar=False
            )
        overlay = _TimeOverlay(viewer)
        overlay.check()
        assert overlay._active

        # Step through every frame and verify the overlay shows the true
        # xarray coordinate, not the linear scale*step + translate value.
        for step, expected_time in enumerate(time_coords):
            viewer.dims.set_current_step(overlay._time_idx, step)
            expected_text = f"{expected_time:.2f} s"
            assert viewer.text_overlay.text == expected_text, (
                f"step {step}: overlay shows {viewer.text_overlay.text!r}, "
                f"expected {expected_text!r}"
            )

    def test_ref_layer_defaults_to_first_time_layer(
        self, rng, make_napari_viewer
    ) -> None:
        """When no layer is selected, the first layer with time is used."""
        da = _make_4d_da(rng, np.arange(5) * 0.5)

        viewer = make_napari_viewer()
        _, layer = plot_napari(
            da, viewer=viewer, show_colorbar=False, show_scale_bar=False
        )
        overlay = _TimeOverlay(viewer)
        overlay.check()

        assert overlay._ref_layer is layer

    def test_selecting_single_time_layer_changes_ref(
        self, rng, make_napari_viewer
    ) -> None:
        """Selecting a single time-aware layer switches the reference."""
        da_a = _make_4d_da(rng, 0.0 + np.arange(5) * 0.5, time_units="s")
        da_b = _make_4d_da(rng, 10.0 + np.arange(5) * 0.5, time_units="s")

        viewer = make_napari_viewer()
        _, layer_a = plot_napari(
            da_a, viewer=viewer, show_colorbar=False, show_scale_bar=False
        )
        _, layer_b = plot_napari(
            da_b, viewer=viewer, show_colorbar=False, show_scale_bar=False
        )
        overlay = _TimeOverlay(viewer)
        overlay.check()
        assert overlay._ref_layer is layer_a

        # Ensure layer_a is selected first so the switch to layer_b fires
        # a changed event (napari auto-selects the last added layer).
        viewer.layers.selection = {layer_a}
        assert overlay._ref_layer is layer_a

        # Now switch to layer_b.
        viewer.layers.selection = {layer_b}
        assert overlay._ref_layer is layer_b

        # Move the slider to a world coordinate inside layer_b's time range.
        # layer_b time origin is 10.0 with 0.5 spacing, so frame 2 = 11.0.
        # The global step for world=11.0 depends on the dims range; compute it
        # from the scale and origin napari resolved.
        time_idx = overlay._time_idx
        dims_range = viewer.dims.range[time_idx]
        step_for_11 = round((11.0 - dims_range[0]) / dims_range[2])
        viewer.dims.set_current_step(time_idx, step_for_11)

        expected = f"{float(da_b.coords['time'].values[2]):.2f} s"
        assert viewer.text_overlay.text == expected

    def test_selecting_non_time_layer_keeps_ref(self, rng, make_napari_viewer) -> None:
        """Selecting a layer without time does not change the reference."""
        da = _make_4d_da(rng, np.arange(5) * 0.5)

        viewer = make_napari_viewer()
        _, time_layer = plot_napari(
            da, viewer=viewer, show_colorbar=False, show_scale_bar=False
        )
        overlay = _TimeOverlay(viewer)
        overlay.check()
        assert overlay._ref_layer is time_layer

        # Add a 3D layer (no time) and select it.
        points_layer = viewer.add_points(np.empty((0, 3)), ndim=3)
        viewer.layers.selection = {points_layer}

        # Reference should remain unchanged.
        assert overlay._ref_layer is time_layer

    def test_selecting_multiple_time_layers_keeps_ref(
        self, rng, make_napari_viewer
    ) -> None:
        """Selecting multiple time-aware layers does not change the reference."""
        da_a = _make_4d_da(rng, np.arange(5) * 0.5)
        da_b = _make_4d_da(rng, 10.0 + np.arange(5) * 0.5)

        viewer = make_napari_viewer()
        _, layer_a = plot_napari(
            da_a, viewer=viewer, show_colorbar=False, show_scale_bar=False
        )
        _, layer_b = plot_napari(
            da_b, viewer=viewer, show_colorbar=False, show_scale_bar=False
        )
        overlay = _TimeOverlay(viewer)
        overlay.check()
        assert overlay._ref_layer is layer_a

        # First select only layer_a so we start from a known state.
        viewer.layers.selection = {layer_a}
        assert overlay._ref_layer is layer_a

        # Select both time layers — reference should not change.
        viewer.layers.selection = {layer_a, layer_b}
        assert overlay._ref_layer is layer_a

    def test_deactivates_when_time_layer_is_removed(
        self, sample_4d_volume, make_napari_viewer
    ) -> None:
        """Removing the only time-aware layer hides and clears the overlay."""
        viewer = make_napari_viewer()
        _, layer = plot_napari(
            sample_4d_volume,
            viewer=viewer,
            show_colorbar=False,
            show_scale_bar=False,
        )
        overlay = _TimeOverlay(viewer)

        overlay.check()
        assert overlay._active

        viewer.layers.remove(layer)

        assert not overlay._active
        assert overlay._time_idx is None
        assert not viewer.text_overlay.visible
        assert viewer.text_overlay.text == ""

    def test_removing_ref_layer_resets_reference(self, rng, make_napari_viewer) -> None:
        """Removing the reference layer resets it; a new one is picked."""
        da_a = _make_4d_da(rng, np.arange(5) * 0.5)
        da_b = _make_4d_da(rng, 10.0 + np.arange(5) * 0.5)

        viewer = make_napari_viewer()
        _, layer_a = plot_napari(
            da_a, viewer=viewer, show_colorbar=False, show_scale_bar=False
        )
        _, layer_b = plot_napari(
            da_b, viewer=viewer, show_colorbar=False, show_scale_bar=False
        )
        overlay = _TimeOverlay(viewer)
        overlay.check()
        assert overlay._ref_layer is layer_a

        viewer.layers.remove(layer_a)

        # Reference should have been reset and re-picked as layer_b.
        assert overlay._ref_layer is layer_b

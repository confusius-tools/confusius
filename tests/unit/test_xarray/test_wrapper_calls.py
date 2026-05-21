"""Unit tests for thin xarray wrapper methods using monkeypatch."""

import numpy as np
import xarray as xr

import confusius  # noqa: F401  # Registers accessor.


def test_scale_wrappers_forward_calls(monkeypatch, sample_4d_volume):
    """Scale accessor methods forward arguments to module functions."""
    expected = xr.DataArray(np.array([1.0]), dims=["k"])
    calls: dict[str, tuple] = {}

    def _db(data, factor=10):
        calls["db"] = (data, factor)
        return expected

    def _log(data):
        calls["log"] = (data,)
        return expected

    def _power(data, exponent=0.5):
        calls["power"] = (data, exponent)
        return expected

    monkeypatch.setattr("confusius.xarray.scale.db_scale", _db)
    monkeypatch.setattr("confusius.xarray.scale.log_scale", _log)
    monkeypatch.setattr("confusius.xarray.scale.power_scale", _power)

    assert sample_4d_volume.fusi.scale.db(factor=20) is expected
    assert calls["db"] == (sample_4d_volume, 20)

    assert sample_4d_volume.fusi.scale.log() is expected
    assert calls["log"] == (sample_4d_volume,)

    assert sample_4d_volume.fusi.scale.power(exponent=2.0) is expected
    assert calls["power"] == (sample_4d_volume, 2.0)


def test_extract_wrappers_forward_calls(monkeypatch, sample_4d_volume, sample_roi_labels):
    """Extract accessor methods forward arguments to extraction functions."""
    expected = xr.DataArray(np.array([1.0]), dims=["k"])
    mask = sample_roi_labels > 0
    calls: dict[str, tuple] = {}

    def _with_labels(data, labels, reduction="mean"):
        calls["labels"] = (data, labels, reduction)
        return expected

    def _with_mask(data, mask_):
        calls["mask"] = (data, mask_)
        return expected

    def _unmask(data, mask, fill_value=0.0):
        calls["unmask"] = (data, mask, fill_value)
        return expected

    monkeypatch.setattr("confusius.extract.labels.extract_with_labels", _with_labels)
    monkeypatch.setattr("confusius.extract.mask.extract_with_mask", _with_mask)
    monkeypatch.setattr("confusius.extract.reconstruction.unmask", _unmask)

    assert sample_4d_volume.fusi.extract.with_labels(sample_roi_labels, reduction="sum") is expected
    assert calls["labels"] == (sample_4d_volume, sample_roi_labels, "sum")

    assert sample_4d_volume.fusi.extract.with_mask(mask) is expected
    assert calls["mask"] == (sample_4d_volume, mask)

    assert sample_4d_volume.fusi.extract.unmask(mask, fill_value=-1.0) is expected
    assert calls["unmask"] == (sample_4d_volume, mask, -1.0)


def test_affine_wrappers_forward_calls(monkeypatch, sample_4d_volume):
    """Affine accessor methods forward arguments to helper functions."""
    expected_to = np.eye(4)
    expected_apply = sample_4d_volume
    other = sample_4d_volume.copy()
    affine = np.diag([1.0, 2.0, 3.0, 1.0])
    calls: dict[str, tuple] = {}

    def _to(data, other_, via):
        calls["to"] = (data, other_, via)
        return expected_to

    def _apply(data, affine_, inplace=False):
        calls["apply"] = (data, affine_, inplace)
        return expected_apply

    monkeypatch.setattr("confusius.xarray.affine.affine_to", _to)
    monkeypatch.setattr("confusius.xarray.affine.apply_affine", _apply)

    assert sample_4d_volume.fusi.affine.to(other, via="physical_to_lab") is expected_to
    assert calls["to"] == (sample_4d_volume, other, "physical_to_lab")

    assert sample_4d_volume.fusi.affine.apply(affine, inplace=True) is expected_apply
    assert calls["apply"] == (sample_4d_volume, affine, True)


def test_iq_wrappers_forward_calls(monkeypatch, sample_4d_volume_complex):
    """IQ accessor methods forward all arguments to processing functions."""
    expected = xr.DataArray(np.array([1.0]), dims=["k"])
    calls: dict[str, tuple] = {}

    def _pwd(data, **kwargs):
        calls["pwd"] = (data, kwargs)
        return expected

    def _vel(data, **kwargs):
        calls["vel"] = (data, kwargs)
        return expected

    def _bmode(data, **kwargs):
        calls["bmode"] = (data, kwargs)
        return expected

    monkeypatch.setattr("confusius.xarray.iq.process_iq_to_power_doppler", _pwd)
    monkeypatch.setattr("confusius.xarray.iq.process_iq_to_axial_velocity", _vel)
    monkeypatch.setattr("confusius.xarray.iq.process_iq_to_bmode", _bmode)

    mask = xr.DataArray(
        np.ones(sample_4d_volume_complex.shape[1:], dtype=bool),
        dims=["z", "y", "x"],
        coords={k: sample_4d_volume_complex.coords[k] for k in ("z", "y", "x")},
    )

    assert sample_4d_volume_complex.fusi.iq.process_to_power_doppler(
        clutter_window_width=11,
        clutter_window_stride=7,
        filter_method="butterworth",
        clutter_mask=mask,
        low_cutoff=30.5,
        high_cutoff=80.0,
        butterworth_order=6,
        doppler_window_width=9,
        doppler_window_stride=3,
    ) is expected
    assert calls["pwd"] == (
        sample_4d_volume_complex,
        {
            "clutter_window_width": 11,
            "clutter_window_stride": 7,
            "filter_method": "butterworth",
            "clutter_mask": mask,
            "low_cutoff": 30.5,
            "high_cutoff": 80.0,
            "butterworth_order": 6,
            "doppler_window_width": 9,
            "doppler_window_stride": 3,
        },
    )

    assert sample_4d_volume_complex.fusi.iq.process_to_axial_velocity(
        clutter_window_width=13,
        clutter_window_stride=5,
        filter_method="svd_energy",
        clutter_mask=mask,
        low_cutoff=2,
        high_cutoff=12,
        butterworth_order=2,
        velocity_window_width=8,
        velocity_window_stride=4,
        lag=2,
        absolute_velocity=True,
        spatial_kernel=3,
        estimation_method="angle_average",
    ) is expected
    assert calls["vel"] == (
        sample_4d_volume_complex,
        {
            "clutter_window_width": 13,
            "clutter_window_stride": 5,
            "filter_method": "svd_energy",
            "clutter_mask": mask,
            "low_cutoff": 2,
            "high_cutoff": 12,
            "butterworth_order": 2,
            "velocity_window_width": 8,
            "velocity_window_stride": 4,
            "lag": 2,
            "absolute_velocity": True,
            "spatial_kernel": 3,
            "estimation_method": "angle_average",
        },
    )

    assert (
        sample_4d_volume_complex.fusi.iq.process_to_bmode(
            bmode_window_width=5,
            bmode_window_stride=2,
        )
        is expected
    )
    assert calls["bmode"] == (
        sample_4d_volume_complex,
        {"bmode_window_width": 5, "bmode_window_stride": 2},
    )


def test_registration_wrappers_forward_calls(monkeypatch, sample_3d_volume):
    """Registration accessor methods forward all arguments and return results."""
    reg_result = (sample_3d_volume, np.eye(4), object())
    volumewise_result = sample_3d_volume
    calls: dict[str, tuple] = {}

    def _to_volume(data, fixed, **kwargs):
        calls["to_volume"] = (data, fixed, kwargs)
        return reg_result

    def _volumewise(data, **kwargs):
        calls["volumewise"] = (data, kwargs)
        return volumewise_result

    monkeypatch.setattr("confusius.xarray.registration.register_volume", _to_volume)
    monkeypatch.setattr("confusius.xarray.registration.register_volumewise", _volumewise)

    fixed = sample_3d_volume.copy()
    assert sample_3d_volume.fusi.register.to_volume(
        fixed,
        transform="affine",
        metric="mattes_mi",
        number_of_histogram_bins=40,
        learning_rate=0.2,
        number_of_iterations=25,
        convergence_minimum_value=1e-4,
        convergence_window_size=4,
        initialization="moments",
        optimizer_weights=[0, 0, 1, 1, 1, 1],
        mesh_size=(3, 4, 5),
        use_multi_resolution=True,
        shrink_factors=(4, 2, 1),
        smoothing_sigmas=(3, 1, 0),
        resample=True,
        resample_interpolation="bspline",
        show_progress=True,
        plot_metric=False,
        plot_composite=False,
        fill_value=-1.0,
    ) is reg_result
    assert calls["to_volume"] == (
        sample_3d_volume,
        fixed,
        {
            "transform_type": "affine",
            "metric": "mattes_mi",
            "number_of_histogram_bins": 40,
            "learning_rate": 0.2,
            "number_of_iterations": 25,
            "convergence_minimum_value": 1e-4,
            "convergence_window_size": 4,
            "centering_initialization": "moments",
            "optimizer_weights": [0, 0, 1, 1, 1, 1],
            "mesh_size": (3, 4, 5),
            "use_multi_resolution": True,
            "shrink_factors": (4, 2, 1),
            "smoothing_sigmas": (3, 1, 0),
            "resample": True,
            "resample_interpolation": "bspline",
            "show_progress": True,
            "plot_metric": False,
            "plot_composite": False,
            "fill_value": -1.0,
        },
    )

    assert sample_3d_volume.fusi.register.volumewise(
        reference_time=2,
        n_jobs=1,
        transform="translation",
        metric="mattes_mi",
        number_of_histogram_bins=20,
        learning_rate=0.15,
        number_of_iterations=30,
        convergence_minimum_value=1e-5,
        convergence_window_size=5,
        initialization="none",
        optimizer_weights=[1, 1, 1, 0, 0, 1],
        use_multi_resolution=True,
        shrink_factors=(3, 1),
        smoothing_sigmas=(2, 0),
        resample_interpolation="bspline",
        show_progress=False,
        keep_diagnostics=True,
    ) is volumewise_result
    assert calls["volumewise"] == (
        sample_3d_volume,
        {
            "reference_time": 2,
            "n_jobs": 1,
            "transform": "translation",
            "metric": "mattes_mi",
            "number_of_histogram_bins": 20,
            "learning_rate": 0.15,
            "number_of_iterations": 30,
            "convergence_minimum_value": 1e-5,
            "convergence_window_size": 5,
            "initialization": "none",
            "optimizer_weights": [1, 1, 1, 0, 0, 1],
            "use_multi_resolution": True,
            "shrink_factors": (3, 1),
            "smoothing_sigmas": (2, 0),
            "resample_interpolation": "bspline",
            "show_progress": False,
            "keep_diagnostics": True,
        },
    )


def test_connectivity_seed_map_constructs_and_fits(sample_4d_volume, sample_roi_labels, monkeypatch):
    """Connectivity wrapper creates SeedBasedMaps with kwargs and calls fit."""
    calls: dict[str, object] = {}

    class DummySeedBasedMaps:
        def __init__(self, **kwargs):
            calls["init_kwargs"] = kwargs

        def fit(self, data):
            calls["fit_data"] = data
            return self

    monkeypatch.setattr("confusius.connectivity.SeedBasedMaps", DummySeedBasedMaps)

    seed_signals = xr.DataArray(np.ones((3, 2)), dims=["time", "region"])
    result = sample_4d_volume.fusi.connectivity.seed_map(
        seed_masks=sample_roi_labels,
        seed_signals=seed_signals,
        labels_reduction="median",
        clean_kwargs={"detrend_order": 1},
    )

    assert isinstance(result, DummySeedBasedMaps)
    assert calls["fit_data"] is sample_4d_volume
    assert calls["init_kwargs"] == {
        "seed_masks": sample_roi_labels,
        "seed_signals": seed_signals,
        "labels_reduction": "median",
        "clean_kwargs": {"detrend_order": 1},
    }


def test_plot_wrappers_forward_calls(monkeypatch, sample_3d_volume, sample_roi_labels):
    """Plot accessor methods forward all arguments to plotting helpers."""
    calls: dict[str, tuple] = {}

    def _napari(data, **kwargs):
        calls["napari"] = (data, kwargs)
        return "viewer", "layer"

    def _draw(data, **kwargs):
        calls["draw"] = (data, kwargs)
        return "viewer", "labels"

    def _labels(layer, data):
        calls["labels"] = (layer, data)
        return sample_roi_labels

    def _carpet(data, **kwargs):
        calls["carpet"] = (data, kwargs)
        return "fig", "ax"

    def _volume(data, **kwargs):
        calls["volume"] = (data, kwargs)
        return "plotter"

    def _contours(data, **kwargs):
        calls["contours"] = (data, kwargs)
        return "plotter"

    def _composite(data, other, **kwargs):
        calls["composite"] = (data, other, kwargs)
        return "plotter"

    monkeypatch.setattr("confusius.xarray.plotting.plot_napari", _napari)
    monkeypatch.setattr("confusius.xarray.plotting.draw_napari_labels", _draw)
    monkeypatch.setattr("confusius.xarray.plotting.labels_from_layer", _labels)
    monkeypatch.setattr("confusius.xarray.plotting.plot_carpet", _carpet)
    monkeypatch.setattr("confusius.xarray.plotting.plot_volume", _volume)
    monkeypatch.setattr("confusius.xarray.plotting.plot_contours", _contours)
    monkeypatch.setattr("confusius.xarray.plotting.plot_composite", _composite)

    viewer = object()
    assert sample_3d_volume.fusi.plot(show_scale_bar=False) == ("viewer", "layer")
    assert calls["napari"] == (
        sample_3d_volume,
        {
            "show_colorbar": True,
            "show_scale_bar": False,
            "dim_order": None,
            "viewer": None,
            "layer_type": "image",
        },
    )

    assert sample_3d_volume.fusi.plot.napari(
        show_colorbar=False,
        show_scale_bar=False,
        dim_order=("y", "z", "x"),
        viewer=viewer,
        layer_type="labels",
        opacity=0.4,
    ) == ("viewer", "layer")
    assert calls["napari"] == (
        sample_3d_volume,
        {
            "show_colorbar": False,
            "show_scale_bar": False,
            "dim_order": ("y", "z", "x"),
            "viewer": viewer,
            "layer_type": "labels",
            "opacity": 0.4,
        },
    )

    assert sample_3d_volume.fusi.plot.draw_napari_labels(
        labels_layer_name="roi",
        viewer=viewer,
        colormap="hot",
    ) == ("viewer", "labels")
    assert calls["draw"] == (
        sample_3d_volume,
        {"labels_layer_name": "roi", "viewer": viewer, "colormap": "hot"},
    )

    assert sample_3d_volume.fusi.plot.labels_from_layer("layer") is sample_roi_labels
    assert calls["labels"] == ("layer", sample_3d_volume)

    assert (
        sample_3d_volume.fusi.plot.carpet(
            mask=sample_roi_labels > 0,
            detrend_order=1,
            standardize=False,
            cmap="viridis",
            vmin=-1.0,
            vmax=2.0,
            decimation_threshold=None,
            figsize=(6, 4),
            title="carpet",
            fontsize=12,
            bg_color="black",
            fg_color="white",
            ax="existing_ax",
        )
        == ("fig", "ax")
    )
    assert calls["carpet"][0] is sample_3d_volume
    assert calls["carpet"][1]["detrend_order"] == 1

    assert (
        sample_3d_volume.fusi.plot.volume(
            slice_coords=[1.0],
            slice_mode="y",
            nrows=1,
            ncols=1,
            threshold=0.2,
            threshold_mode="upper",
            cmap="magma",
            norm="norm",
            vmin=0.0,
            vmax=1.0,
            alpha=0.8,
            show_colorbar=False,
            cbar_label="u",
            show_titles=False,
            show_axis_labels=False,
            show_axis_ticks=False,
            show_axes=False,
            fontsize=10,
            yincrease=True,
            xincrease=False,
            bg_color="white",
            fg_color="black",
            figure="fig",
            axes="axes",
            dpi=100,
        )
        == "plotter"
    )
    assert calls["volume"][0] is sample_3d_volume
    assert calls["volume"][1]["threshold_mode"] == "upper"

    assert (
        sample_3d_volume.fusi.plot.contours(
            colors={1: "red"},
            linewidths=2.0,
            linestyles="dashed",
            slice_mode="x",
            slice_coords=[3.0],
            fontsize=11,
            yincrease=True,
            xincrease=False,
            bg_color="white",
            fg_color="black",
            figure="fig",
            axes="axes",
            zorder=3,
        )
        == "plotter"
    )
    assert calls["contours"][0] is sample_3d_volume
    assert calls["contours"][1]["linestyles"] == "dashed"

    other = sample_3d_volume.copy()
    assert (
        sample_3d_volume.fusi.plot.composite(
            other,
            resample=False,
            resample_kwargs={"fill_value": 0},
            rtol=1e-3,
            atol=1e-4,
            normalize_strategy="shared",
            slice_coords=[1.0],
            slice_mode="z",
            alpha=0.7,
            show_titles=False,
            show_axis_labels=False,
            show_axis_ticks=False,
            show_axes=False,
            fontsize=9,
            yincrease=True,
            xincrease=False,
            bg_color="white",
            fg_color="black",
            figure="fig",
            axes="axes",
            nrows=1,
            ncols=1,
            dpi=120,
        )
        == "plotter"
    )
    assert calls["composite"] == (
        sample_3d_volume,
        other,
        {
            "resample": False,
            "resample_kwargs": {"fill_value": 0},
            "rtol": 1e-3,
            "atol": 1e-4,
            "normalize_strategy": "shared",
            "slice_coords": [1.0],
            "slice_mode": "z",
            "alpha": 0.7,
            "show_titles": False,
            "show_axis_labels": False,
            "show_axis_ticks": False,
            "show_axes": False,
            "fontsize": 9,
            "yincrease": True,
            "xincrease": False,
            "bg_color": "white",
            "fg_color": "black",
            "figure": "fig",
            "axes": "axes",
            "nrows": 1,
            "ncols": 1,
            "dpi": 120,
        },
    )

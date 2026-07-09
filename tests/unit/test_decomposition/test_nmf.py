"""Tests for confusius.decomposition.NMF."""

import warnings

import numpy as np
import pytest
import xarray as xr
from sklearn.decomposition import NMF as SklearnNMF
from sklearn.utils.validation import check_is_fitted

from confusius.decomposition import NMF


@pytest.fixture
def nmf_3dt_volume():
    """3D+t volume with low-rank structure that NMF can factorize quickly.

    Generated as `W @ H` with `W` temporal and `H` spatial, scaled so the
    rank-`k` signal is well separated from numerical noise. The CD solver
    converges in a handful of iterations, keeping the test suite fast. Uses
    a locally seeded RNG so the data is independent of test ordering.
    """
    local_rng = np.random.default_rng(0)
    n_t, n_z, n_y, n_x, k = 40, 4, 6, 8, 3
    W = local_rng.random((n_t, k))
    H = local_rng.random((k, n_z * n_y * n_x))
    data = (10.0 * (W @ H) + 1.0).reshape(n_t, n_z, n_y, n_x)
    return xr.DataArray(
        data,
        name="power_doppler",
        dims=["time", "z", "y", "x"],
        coords={
            "time": xr.DataArray(
                10.0 + np.arange(n_t) * 0.5,
                dims=["time"],
                attrs={"units": "s"},
            ),
            "z": xr.DataArray(
                1.0 + np.arange(n_z) * 0.2,
                dims=["z"],
                attrs={"units": "mm", "voxdim": 0.2},
            ),
            "y": xr.DataArray(
                2.0 + np.arange(n_y) * 0.1,
                dims=["y"],
                attrs={"units": "mm", "voxdim": 0.1},
            ),
            "x": xr.DataArray(
                3.0 + np.arange(n_x) * 0.05,
                dims=["x"],
                attrs={"units": "mm", "voxdim": 0.05},
            ),
        },
        attrs={"long_name": "Intensity", "units": "a.u."},
    )


def test_feature_names_in_for_string_feature_labels(nmf_3dt_volume):
    """feature_names_in_ is defined when flattened feature labels are strings."""
    data = (
        nmf_3dt_volume.isel(z=0, x=0, drop=True)
        .rename({"y": "region"})
        .assign_coords(region=["A", "B", "C", "D", "E", "F"])
    )

    # sklearn's NMF iteration can hit a transient `RuntimeWarning: invalid value
    # encountered in dot` on BLAS implementations that flush denormals differently
    # (notably Accelerate on macOS). The test only inspects `feature_names_in_`,
    # which is independent of the iteration's numerical result.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in dot")
        model = NMF(n_components=2, random_state=0).fit(data)

    np.testing.assert_array_equal(
        model.feature_names_in_,
        np.array(["A", "B", "C", "D", "E", "F"]),
    )


@pytest.mark.parametrize("mode", ["temporal", "spatial"])
def test_fit_transform_matches_fit_then_transform(nmf_3dt_volume, mode):
    """fit_transform matches calling fit followed by transform."""
    model_direct = NMF(n_components=3, random_state=0, mode=mode)
    direct = model_direct.fit_transform(nmf_3dt_volume)

    model_two_step = NMF(n_components=3, random_state=0, mode=mode)
    two_step = model_two_step.fit(nmf_3dt_volume).transform(nmf_3dt_volume)

    xr.testing.assert_identical(direct, two_step)


def test_wrapper_matches_sklearn_attributes(nmf_3dt_volume):
    """Temporal wrapper exposes the same learned quantities as sklearn NMF."""
    stacked = nmf_3dt_volume.transpose("time", "z", "y", "x").stack(
        feature=["z", "y", "x"]
    )
    X = np.asarray(stacked.values, dtype=np.float64)

    model = NMF(n_components=3, random_state=0).fit(nmf_3dt_volume)
    sklearn_model = SklearnNMF(n_components=3, random_state=0).fit(X)

    np.testing.assert_allclose(
        model.transform(nmf_3dt_volume).values,
        sklearn_model.transform(X),
    )
    np.testing.assert_allclose(
        model.maps_.stack(feature=["z", "y", "x"]).values,
        sklearn_model.components_,
    )
    assert model.n_components_ == sklearn_model.n_components_
    assert model.n_iter_ == sklearn_model.n_iter_
    np.testing.assert_allclose(
        model.reconstruction_err_, sklearn_model.reconstruction_err_
    )


def test_fit_raises_for_negative_input():
    """fit raises ValueError when input data contains negative values."""
    rng = np.random.default_rng(0)
    data = xr.DataArray(
        rng.standard_normal((20, 4, 5, 6)),
        dims=["time", "z", "y", "x"],
    )

    with pytest.raises(ValueError, match="non-negative input data"):
        NMF(n_components=3, random_state=0).fit(data)


def test_inverse_transform_matches_sklearn_temporal(nmf_3dt_volume):
    """inverse_transform matches sklearn NMF reconstruction in temporal mode."""
    stacked = nmf_3dt_volume.transpose("time", "z", "y", "x").stack(
        feature=["z", "y", "x"]
    )
    X = np.asarray(stacked.values, dtype=np.float64)

    model = NMF(n_components=3, random_state=0, mode="temporal")
    reconstructed = model.inverse_transform(model.fit_transform(nmf_3dt_volume))

    sklearn_model = SklearnNMF(n_components=3, random_state=0).fit(X)
    sklearn_reconstructed = sklearn_model.inverse_transform(sklearn_model.transform(X))

    np.testing.assert_allclose(
        reconstructed.stack(feature=["z", "y", "x"]).values,
        sklearn_reconstructed,
    )


def test_inverse_transform_runs_in_fitted_geometry(nmf_3dt_volume):
    """NMF inverse_transform reconstructs in fitted spatial geometry with metadata."""
    model = NMF(n_components=3, random_state=0)

    signals = model.fit_transform(nmf_3dt_volume)
    reconstructed = model.inverse_transform(signals)

    assert reconstructed.dims == nmf_3dt_volume.dims
    np.testing.assert_allclose(
        reconstructed.coords["time"], nmf_3dt_volume.coords["time"]
    )
    assert reconstructed.name == nmf_3dt_volume.name
    assert reconstructed.attrs == nmf_3dt_volume.attrs


def test_temporal_mode_signals_and_maps_are_non_negative(nmf_3dt_volume):
    """Temporal-mode signals and maps are non-negative — the defining NMF guarantee."""
    model = NMF(n_components=3, random_state=0, mode="temporal")
    signals = model.fit_transform(nmf_3dt_volume)

    assert float(signals.min()) >= 0.0
    assert float(model.maps_.min()) >= 0.0


def test_inverse_transform_from_numpy_returns_dataarray(nmf_3dt_volume):
    """inverse_transform accepts ndarray input and returns DataArray."""
    model = NMF(n_components=3, random_state=0)
    scores = model.fit_transform(nmf_3dt_volume).values

    reconstructed = model.inverse_transform(scores)

    assert isinstance(reconstructed, xr.DataArray)
    assert reconstructed.dims == nmf_3dt_volume.dims
    np.testing.assert_array_equal(
        reconstructed.coords["time"], np.arange(nmf_3dt_volume.sizes["time"])
    )


def test_inverse_transform_raises_for_invalid_dataarray_dims(nmf_3dt_volume):
    """inverse_transform raises when DataArray dims are not time/component."""
    model = NMF(n_components=3, random_state=0).fit(nmf_3dt_volume)
    bad = xr.DataArray(
        np.zeros((nmf_3dt_volume.sizes["time"], 3)),
        dims=["time", "region"],
    )

    with pytest.raises(ValueError, match="exactly the dimensions"):
        model.inverse_transform(bad)


def test_inverse_transform_raises_for_component_count_mismatch(nmf_3dt_volume):
    """inverse_transform raises when component count differs from fitted NMF."""
    model = NMF(n_components=3, random_state=0).fit(nmf_3dt_volume)
    scores = model.transform(nmf_3dt_volume)
    bad = scores.isel(component=slice(0, 2))

    with pytest.raises(ValueError, match="but NMF was fitted with"):
        model.inverse_transform(bad)


def test_inverse_transform_raises_for_invalid_numpy_shape(nmf_3dt_volume):
    """inverse_transform raises when ndarray input is not 2D."""
    model = NMF(n_components=3, random_state=0).fit(nmf_3dt_volume)

    with pytest.raises(ValueError, match="must be 2D"):
        model.inverse_transform(np.zeros((nmf_3dt_volume.sizes["time"], 3, 1)))


def test_inverse_transform_raises_for_invalid_input_type(nmf_3dt_volume):
    """inverse_transform raises TypeError for unsupported input types."""
    model = NMF(n_components=3, random_state=0).fit(nmf_3dt_volume)

    with pytest.raises(TypeError, match="DataArray or ndarray"):
        model.inverse_transform([1, 2, 3])


def test_fit_requires_time_dimension(nmf_3dt_volume):
    """fit raises when the input has no `time` dimension."""
    no_time = nmf_3dt_volume.isel(time=0, drop=True)

    with pytest.raises(ValueError, match="must have a 'time' dimension"):
        NMF().fit(no_time)


def test_fit_requires_more_than_one_timepoint(nmf_3dt_volume):
    """fit raises when only one timepoint is provided."""
    single_timepoint = nmf_3dt_volume.isel(time=[0])

    with pytest.raises(ValueError, match="requires more than 1 timepoint"):
        NMF().fit(single_timepoint)


def test_fit_requires_spatial_dimension():
    """fit raises when input has no spatial dimensions."""
    only_time = xr.DataArray(np.arange(30.0), dims=["time"])

    with pytest.raises(ValueError, match="at least one spatial dimension"):
        NMF().fit(only_time)


def test_mask_must_match_full_spatial_dims_in_order(nmf_3dt_volume):
    """Mask must span all spatial dims in the stacked feature order."""
    mask = xr.DataArray(
        np.ones(
            (
                nmf_3dt_volume.sizes["y"],
                nmf_3dt_volume.sizes["z"],
                nmf_3dt_volume.sizes["x"],
            ),
            dtype=bool,
        ),
        dims=["y", "z", "x"],
        coords={
            "y": nmf_3dt_volume.coords["y"],
            "z": nmf_3dt_volume.coords["z"],
            "x": nmf_3dt_volume.coords["x"],
        },
    )

    with pytest.raises(ValueError, match="must match all non-time dimensions"):
        NMF(mask=mask).fit(nmf_3dt_volume)


def test_fit_rejects_unexpected_fit_params(nmf_3dt_volume):
    """fit raises when unexpected sklearn-style fit params are provided."""
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        NMF().fit(
            nmf_3dt_volume,
            sample_weight=np.ones(nmf_3dt_volume.sizes["time"]),
        )


def test_fit_transform_rejects_unexpected_fit_params(nmf_3dt_volume):
    """fit_transform raises when unexpected sklearn-style fit params are provided."""
    with pytest.raises(TypeError, match="Unexpected fit parameters"):
        NMF().fit_transform(
            nmf_3dt_volume,
            sample_weight=np.ones(nmf_3dt_volume.sizes["time"]),
        )


def test_transform_checks_spatial_layout(nmf_3dt_volume):
    """transform raises if spatial layout differs from fit."""
    model = NMF(n_components=3, random_state=0).fit(nmf_3dt_volume)
    bad = nmf_3dt_volume.isel(x=slice(0, 4))

    with pytest.raises(ValueError, match="Spatial dimension 'x' has size"):
        model.transform(bad)


def test_transform_checks_spatial_dimension_names(nmf_3dt_volume):
    """transform raises if spatial dimension names differ from fit."""
    model = NMF(n_components=3, random_state=0).fit(nmf_3dt_volume)
    bad = nmf_3dt_volume.rename({"x": "region"})

    with pytest.raises(ValueError, match="spatial dimensions do not match"):
        model.transform(bad)


def test_transform_without_time_coordinate_uses_index(nmf_3dt_volume):
    """transform falls back to integer time coordinate when absent."""
    model = NMF(n_components=3, random_state=0).fit(nmf_3dt_volume)
    no_time_coord = xr.DataArray(
        nmf_3dt_volume.values,
        dims=nmf_3dt_volume.dims,
        coords={
            "z": nmf_3dt_volume.coords["z"],
            "y": nmf_3dt_volume.coords["y"],
            "x": nmf_3dt_volume.coords["x"],
        },
    )

    transformed = model.transform(no_time_coord)

    np.testing.assert_array_equal(
        transformed.coords["time"].values,
        np.arange(nmf_3dt_volume.sizes["time"]),
    )


def test_transform_chunked_time_reports_transform_operation(nmf_3dt_volume):
    """transform chunking error message identifies NMF.transform."""
    model = NMF(n_components=3, random_state=0).fit(nmf_3dt_volume)
    chunked = nmf_3dt_volume.chunk({"time": 5})

    with pytest.raises(ValueError, match="NMF.transform requires the full time series"):
        model.transform(chunked)


def test_sklearn_interface_fitted_state(nmf_3dt_volume):
    """Estimator exposes sklearn fitted-state behavior."""
    model = NMF(n_components=3, random_state=0)
    with pytest.raises(Exception):
        check_is_fitted(model)

    check_is_fitted(model.fit(nmf_3dt_volume))


def test_fit_failure_does_not_mark_estimator_fitted(nmf_3dt_volume, monkeypatch):
    """Estimator remains unfitted when underlying sklearn NMF fit fails."""
    import confusius.decomposition.nmf as nmf_module

    def _raise_fit(self, X, y=None):
        raise RuntimeError("fit failed")

    monkeypatch.setattr(nmf_module._SklearnNMF, "fit", _raise_fit)

    model = NMF(n_components=3, random_state=0)
    with pytest.raises(RuntimeError, match="fit failed"):
        model.fit(nmf_3dt_volume)

    assert not hasattr(model, "_estimator")
    assert not model.__sklearn_is_fitted__()
    with pytest.raises(Exception):
        check_is_fitted(model)


def test_get_params_includes_constructor_arguments():
    """get_params includes all constructor arguments."""
    model = NMF(
        n_components=3,
        init="nndsvda",
        solver="mu",
        beta_loss="kullback-leibler",
        tol=1e-3,
        max_iter=500,
        random_state=42,
        alpha_W=0.1,
        alpha_H=0.1,
        l1_ratio=0.5,
        shuffle=True,
        mode="spatial",
    )
    params = model.get_params()

    assert params["n_components"] == 3
    assert params["init"] == "nndsvda"
    assert params["solver"] == "mu"
    assert params["beta_loss"] == "kullback-leibler"
    assert params["tol"] == 1e-3
    assert params["max_iter"] == 500
    assert params["random_state"] == 42
    assert params["alpha_W"] == 0.1
    assert params["alpha_H"] == 0.1
    assert params["l1_ratio"] == 0.5
    assert params["shuffle"] is True
    assert params["mode"] == "spatial"


def test_set_params_updates_values():
    """set_params updates constructor parameters."""
    model = NMF()
    model.set_params(
        n_components=2,
        init="nndsvdar",
        random_state=7,
        shuffle=True,
        mode="spatial",
    )

    assert model.n_components == 2
    assert model.init == "nndsvdar"
    assert model.random_state == 7
    assert model.shuffle is True
    assert model.mode == "spatial"


def test_spatial_mode_matches_reference_implementation(nmf_3dt_volume):
    """Spatial mode matches sklearn NMF fitted on transposed data."""
    stacked = nmf_3dt_volume.transpose("time", "z", "y", "x").stack(
        feature=["z", "y", "x"]
    )
    X = np.asarray(stacked.values, dtype=np.float64)

    model = NMF(n_components=3, random_state=0, mode="spatial").fit(nmf_3dt_volume)
    sklearn_model = SklearnNMF(n_components=3, random_state=0).fit(X.T)

    spatial_maps = sklearn_model.transform(X.T).T
    voxel_mean = X.mean(axis=0)
    reference_signals = (X - voxel_mean) @ spatial_maps.T
    reference_reconstructed = reference_signals @ spatial_maps + voxel_mean

    np.testing.assert_allclose(
        model.transform(nmf_3dt_volume).values,
        reference_signals,
    )
    np.testing.assert_allclose(
        model.maps_.stack(feature=["z", "y", "x"]).values,
        spatial_maps,
    )
    np.testing.assert_allclose(
        model.inverse_transform(model.transform(nmf_3dt_volume))
        .stack(feature=["z", "y", "x"])
        .values,
        reference_reconstructed,
    )


def test_fit_rejects_invalid_mode(nmf_3dt_volume):
    """fit raises for unsupported NMF mode."""
    with pytest.raises(ValueError, match="mode must be 'temporal' or 'spatial'"):
        NMF(mode="invalid").fit(nmf_3dt_volume)  # type: ignore[arg-type]


def test_mask_restricts_features(nmf_3dt_volume):
    """mask restricts fitted feature count to selected voxels."""
    mask = xr.DataArray(
        np.zeros(
            (
                nmf_3dt_volume.sizes["z"],
                nmf_3dt_volume.sizes["y"],
                nmf_3dt_volume.sizes["x"],
            ),
            dtype=bool,
        ),
        dims=["z", "y", "x"],
        coords={
            "z": nmf_3dt_volume.coords["z"],
            "y": nmf_3dt_volume.coords["y"],
            "x": nmf_3dt_volume.coords["x"],
        },
    )
    mask.values[:2, :, :] = True

    model = NMF(n_components=3, random_state=0, mask=mask).fit(nmf_3dt_volume)

    assert model.n_features_in_ == int(mask.values.sum())


def test_masked_fit_reconstructs_full_geometry_with_zero_fill(nmf_3dt_volume):
    """Masked NMF keeps full geometry and fills outside-mask voxels with zero."""
    mask = xr.DataArray(
        np.zeros(
            (
                nmf_3dt_volume.sizes["z"],
                nmf_3dt_volume.sizes["y"],
                nmf_3dt_volume.sizes["x"],
            ),
            dtype=bool,
        ),
        dims=["z", "y", "x"],
        coords={
            "z": nmf_3dt_volume.coords["z"],
            "y": nmf_3dt_volume.coords["y"],
            "x": nmf_3dt_volume.coords["x"],
        },
    )
    mask.values[:2, :, :] = True

    model = NMF(n_components=3, random_state=0, mask=mask).fit(nmf_3dt_volume)
    reconstructed = model.inverse_transform(model.transform(nmf_3dt_volume))

    assert reconstructed.dims == nmf_3dt_volume.dims
    np.testing.assert_array_equal(
        reconstructed.where(~mask, other=np.nan).fillna(0.0).values,
        0.0,
    )
    np.testing.assert_array_equal(
        model.maps_.where(~mask, other=np.nan).fillna(0.0).values,
        0.0,
    )


def test_mask_mismatch_raises(nmf_3dt_volume):
    """fit raises when mask does not match spatial dimensions."""
    bad_mask = xr.DataArray(np.ones((3, 3), dtype=bool), dims=["y", "x"])

    with pytest.raises(ValueError, match="missing from mask"):
        NMF(mask=bad_mask).fit(nmf_3dt_volume)

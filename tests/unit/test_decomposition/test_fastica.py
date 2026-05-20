"""Tests for confusius.decomposition.FastICA."""

import numpy as np
import pytest
import xarray as xr
from sklearn.decomposition import FastICA as SklearnFastICA
from sklearn.utils.validation import check_is_fitted

from confusius.decomposition import FastICA


FASTICA_TEST_KWARGS = {
    "n_components": 2,
    "random_state": 0,
    "max_iter": 1000,
    "tol": 1e-3,
    "fun": "cube",
}


def test_feature_names_in_for_string_feature_labels():
    """feature_names_in_ is defined when flattened feature labels are strings."""
    data = xr.DataArray(
        np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 2.0],
                [2.0, 1.0, 0.0],
                [1.0, 2.0, 1.0],
                [2.0, 0.0, 2.0],
            ]
        ),
        dims=["time", "region"],
        coords={"region": ["A", "B", "C"]},
    )

    model = FastICA(**FASTICA_TEST_KWARGS).fit(data)

    np.testing.assert_array_equal(model.feature_names_in_, np.array(["A", "B", "C"]))


@pytest.mark.parametrize("mode", ["spatial", "temporal"])
def test_fit_transform_matches_fit_then_transform(sample_4d_volume, mode):
    """fit_transform matches calling fit followed by transform."""
    model_direct = FastICA(**FASTICA_TEST_KWARGS, mode=mode)
    direct = model_direct.fit_transform(sample_4d_volume)

    model_two_step = FastICA(**FASTICA_TEST_KWARGS, mode=mode)
    two_step = model_two_step.fit(sample_4d_volume).transform(sample_4d_volume)

    xr.testing.assert_identical(direct, two_step)


@pytest.mark.parametrize("mode", ["spatial", "temporal"])
def test_inverse_transform_matches_sklearn(sample_4d_volume, mode):
    """inverse_transform matches sklearn FastICA reconstruction for both modes."""
    stacked = sample_4d_volume.transpose("time", "z", "y", "x").stack(
        feature=["z", "y", "x"]
    )
    X = np.asarray(stacked.values, dtype=np.float64)

    model = FastICA(**FASTICA_TEST_KWARGS, mode=mode)
    reconstructed = model.inverse_transform(model.fit_transform(sample_4d_volume))

    if mode == "temporal":
        sklearn_model = SklearnFastICA(**FASTICA_TEST_KWARGS).fit(X)
        sklearn_reconstructed = sklearn_model.inverse_transform(
            sklearn_model.transform(X)
        )
    else:
        sklearn_model = SklearnFastICA(**FASTICA_TEST_KWARGS).fit(X.T)
        spatial_maps = sklearn_model.transform(X.T).T
        voxel_mean = X.mean(axis=0)
        time_courses = (X - voxel_mean) @ spatial_maps.T
        sklearn_reconstructed = time_courses @ spatial_maps + voxel_mean

    np.testing.assert_allclose(
        reconstructed.stack(feature=["z", "y", "x"]).values,
        sklearn_reconstructed,
    )
    assert reconstructed.name == sample_4d_volume.name
    assert reconstructed.attrs == sample_4d_volume.attrs


@pytest.mark.parametrize("mode", ["spatial", "temporal"])
def test_wrapper_matches_sklearn_attributes(sample_4d_volume, mode):
    """Wrapper exposes the same learned matrices as sklearn FastICA for both modes."""
    stacked = sample_4d_volume.transpose("time", "z", "y", "x").stack(
        feature=["z", "y", "x"]
    )
    X = np.asarray(stacked.values, dtype=np.float64)

    model = FastICA(**FASTICA_TEST_KWARGS, mode=mode).fit(sample_4d_volume)

    if mode == "temporal":
        sklearn_model = SklearnFastICA(**FASTICA_TEST_KWARGS).fit(X)
        np.testing.assert_allclose(
            model.transform(sample_4d_volume).values,
            sklearn_model.transform(X),
        )
        np.testing.assert_allclose(
            model.maps_.stack(feature=["z", "y", "x"]).values,
            sklearn_model.components_,
        )
        np.testing.assert_allclose(
            model.mean_.stack(feature=["z", "y", "x"]).values, sklearn_model.mean_
        )
        np.testing.assert_allclose(
            model.whitening_.stack(feature=["z", "y", "x"]).values,
            sklearn_model.whitening_,
        )
        assert model.n_iter_ == sklearn_model.n_iter_
    else:
        # Spatial ICA fits on (n_voxels, n_time); sources are spatial maps.
        sklearn_model = SklearnFastICA(**FASTICA_TEST_KWARGS).fit(X.T)
        spatial_maps = sklearn_model.transform(X.T).T
        voxel_mean = X.mean(axis=0)
        np.testing.assert_allclose(
            model.transform(sample_4d_volume).values,
            (X - voxel_mean) @ spatial_maps.T,
        )
        np.testing.assert_allclose(
            model.maps_.stack(feature=["z", "y", "x"]).values,
            spatial_maps,
        )
        np.testing.assert_allclose(
            model.mean_.stack(feature=["z", "y", "x"]).values,
            voxel_mean,
        )
        assert not hasattr(model, "whitening_")
        assert model.n_iter_ == sklearn_model.n_iter_


@pytest.mark.parametrize("mode", ["spatial", "temporal"])
def test_inverse_transform_from_numpy_returns_dataarray(sample_4d_volume, mode):
    """inverse_transform accepts ndarray input and returns DataArray."""
    model = FastICA(**FASTICA_TEST_KWARGS, mode=mode)
    signals = model.fit_transform(sample_4d_volume).values

    reconstructed = model.inverse_transform(signals)

    assert isinstance(reconstructed, xr.DataArray)
    assert reconstructed.dims == sample_4d_volume.dims
    np.testing.assert_array_equal(
        reconstructed.coords["time"], np.arange(sample_4d_volume.sizes["time"])
    )


def test_inverse_transform_raises_for_invalid_dataarray_dims(sample_4d_volume):
    """inverse_transform raises when DataArray dims are not time/component."""
    model = FastICA(**FASTICA_TEST_KWARGS).fit(sample_4d_volume)
    bad = xr.DataArray(
        np.zeros((sample_4d_volume.sizes["time"], 2)),
        dims=["time", "region"],
    )

    with pytest.raises(ValueError, match="exactly the dimensions"):
        model.inverse_transform(bad)


def test_inverse_transform_raises_for_component_count_mismatch(sample_4d_volume):
    """inverse_transform raises when component count differs from fitted FastICA."""
    model = FastICA(**FASTICA_TEST_KWARGS).fit(sample_4d_volume)
    scores = model.transform(sample_4d_volume)
    bad = scores.isel(component=slice(0, 1))

    with pytest.raises(ValueError, match="but FastICA was fitted with"):
        model.inverse_transform(bad)


def test_inverse_transform_raises_for_invalid_numpy_shape(sample_4d_volume):
    """inverse_transform raises when ndarray input is not 2D."""
    model = FastICA(**FASTICA_TEST_KWARGS).fit(sample_4d_volume)

    with pytest.raises(ValueError, match="must be 2D"):
        model.inverse_transform(np.zeros((sample_4d_volume.sizes["time"], 2, 1)))


def test_inverse_transform_raises_for_invalid_input_type(sample_4d_volume):
    """inverse_transform raises TypeError for unsupported input types."""
    model = FastICA(**FASTICA_TEST_KWARGS).fit(sample_4d_volume)

    with pytest.raises(TypeError, match="DataArray or ndarray"):
        model.inverse_transform([1, 2, 3])


def test_fit_rejects_invalid_mode(sample_4d_volume):
    """fit raises ValueError for unknown mode values."""
    with pytest.raises(ValueError, match="mode must be"):
        FastICA(mode="invalid").fit(sample_4d_volume)  # type: ignore[arg-type]


def test_fit_requires_time_dimension(sample_4d_volume):
    """fit raises when the input has no `time` dimension."""
    no_time = sample_4d_volume.isel(time=0, drop=True)

    with pytest.raises(ValueError, match="must have a 'time' dimension"):
        FastICA().fit(no_time)


def test_fit_requires_more_than_one_timepoint(sample_4d_volume):
    """fit raises when only one timepoint is provided."""
    single_timepoint = sample_4d_volume.isel(time=[0])

    with pytest.raises(ValueError, match="requires more than 1 timepoint"):
        FastICA().fit(single_timepoint)


def test_fit_requires_spatial_dimension():
    """fit raises when input has no spatial dimensions."""
    only_time = xr.DataArray(np.arange(30.0), dims=["time"])

    with pytest.raises(ValueError, match="at least one spatial dimension"):
        FastICA().fit(only_time)


def test_fit_rejects_unexpected_fit_params(sample_4d_volume):
    """fit raises when unexpected sklearn-style fit params are provided."""
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        FastICA().fit(
            sample_4d_volume,
            sample_weight=np.ones(sample_4d_volume.sizes["time"]),
        )


def test_fit_transform_rejects_unexpected_fit_params(sample_4d_volume):
    """fit_transform raises when unexpected sklearn-style fit params are provided."""
    with pytest.raises(TypeError, match="Unexpected fit parameters"):
        FastICA().fit_transform(
            sample_4d_volume,
            sample_weight=np.ones(sample_4d_volume.sizes["time"]),
        )


def test_transform_checks_spatial_layout(sample_4d_volume):
    """transform raises if spatial layout differs from fit."""
    model = FastICA(**FASTICA_TEST_KWARGS).fit(sample_4d_volume)
    bad = sample_4d_volume.isel(x=slice(0, 4))

    with pytest.raises(ValueError, match="Spatial dimension 'x' has size"):
        model.transform(bad)


def test_transform_checks_spatial_dimension_names(sample_4d_volume):
    """transform raises if spatial dimension names differ from fit."""
    model = FastICA(**FASTICA_TEST_KWARGS).fit(sample_4d_volume)
    bad = sample_4d_volume.rename({"x": "region"})

    with pytest.raises(ValueError, match="spatial dimensions do not match"):
        model.transform(bad)


def test_transform_without_time_coordinate_uses_index(sample_4d_volume):
    """transform falls back to integer time coordinate when absent."""
    model = FastICA(**FASTICA_TEST_KWARGS).fit(sample_4d_volume)
    no_time_coord = xr.DataArray(
        sample_4d_volume.values,
        dims=sample_4d_volume.dims,
        coords={
            "z": sample_4d_volume.coords["z"],
            "y": sample_4d_volume.coords["y"],
            "x": sample_4d_volume.coords["x"],
        },
    )

    transformed = model.transform(no_time_coord)

    np.testing.assert_array_equal(
        transformed.coords["time"].values,
        np.arange(sample_4d_volume.sizes["time"]),
    )


def test_transform_chunked_time_reports_transform_operation(sample_4d_volume):
    """transform chunking error message identifies FastICA.transform."""
    model = FastICA(**FASTICA_TEST_KWARGS).fit(sample_4d_volume)
    chunked = sample_4d_volume.chunk({"time": 5})

    with pytest.raises(
        ValueError, match="FastICA.transform requires the full time series"
    ):
        model.transform(chunked)


def test_sklearn_interface_fitted_state(sample_4d_volume):
    """Estimator exposes sklearn fitted-state behavior."""
    model = FastICA(**FASTICA_TEST_KWARGS)
    with pytest.raises(Exception):
        check_is_fitted(model)

    check_is_fitted(model.fit(sample_4d_volume))


def test_fit_failure_does_not_mark_estimator_fitted(sample_4d_volume, monkeypatch):
    """Estimator remains unfitted when underlying sklearn FastICA fit fails."""
    import confusius.decomposition.fastica as fastica_module

    def _raise_fit(self, X, y=None):
        raise RuntimeError("fit failed")

    monkeypatch.setattr(fastica_module._SklearnFastICA, "fit", _raise_fit)

    model = FastICA(**FASTICA_TEST_KWARGS)
    with pytest.raises(RuntimeError, match="fit failed"):
        model.fit(sample_4d_volume)

    assert not hasattr(model, "_estimator")
    assert not model.__sklearn_is_fitted__()
    with pytest.raises(Exception):
        check_is_fitted(model)


def test_get_params_includes_constructor_arguments():
    """get_params includes all constructor arguments."""
    w_init = np.eye(3)
    model = FastICA(
        n_components=3,
        mode="temporal",
        algorithm="deflation",
        whiten="arbitrary-variance",
        fun="cube",
        fun_args={"alpha": 1.0},
        max_iter=300,
        tol=1e-3,
        w_init=w_init,
        whiten_solver="eigh",
        random_state=42,
    )
    params = model.get_params()

    assert params["n_components"] == 3
    assert params["mode"] == "temporal"
    assert params["algorithm"] == "deflation"
    assert params["whiten"] == "arbitrary-variance"
    assert params["fun"] == "cube"
    assert params["fun_args"] == {"alpha": 1.0}
    assert params["max_iter"] == 300
    assert params["tol"] == 1e-3
    assert params["w_init"] is w_init
    assert params["whiten_solver"] == "eigh"
    assert params["random_state"] == 42


def test_set_params_updates_values():
    """set_params updates constructor parameters."""
    model = FastICA()
    model.set_params(n_components=2, algorithm="deflation", whiten_solver="eigh")

    assert model.n_components == 2
    assert model.algorithm == "deflation"
    assert model.whiten_solver == "eigh"


def test_reproducible_with_random_state():
    """FastICA gives reproducible results with fixed random_state."""
    data = xr.DataArray(
        np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 2.0],
                [2.0, 1.0, 0.0],
                [1.0, 2.0, 1.0],
                [2.0, 0.0, 2.0],
            ]
        ),
        dims=["time", "region"],
        coords={"region": ["A", "B", "C"]},
    )
    model_1 = FastICA(**FASTICA_TEST_KWARGS)
    model_2 = FastICA(**FASTICA_TEST_KWARGS)

    signals_1 = model_1.fit_transform(data)
    signals_2 = model_2.fit_transform(data)

    np.testing.assert_allclose(signals_1.values, signals_2.values)
    np.testing.assert_allclose(model_1.maps_.values, model_2.maps_.values)

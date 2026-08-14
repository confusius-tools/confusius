"""Tests for confusius.decomposition.FastICA."""

from typing import Literal, TypedDict

import numpy as np
import pytest
import xarray as xr
from sklearn.decomposition import FastICA as SklearnFastICA
from sklearn.utils.validation import check_is_fitted

from confusius._utils.geometry import attach_voxel_to_world_index
from confusius.decomposition import FastICA


class _FasticaTestKwargs(TypedDict):
    n_components: int
    random_state: int
    max_iter: int
    tol: float
    fun: Literal["cube"]


FASTICA_TEST_KWARGS: _FasticaTestKwargs = {
    "n_components": 2,
    "random_state": 0,
    "max_iter": 1000,
    "tol": 1e-3,
    "fun": "cube",
}


@pytest.fixture
def sample_fusi_3dt():
    """Stable 3D+t fUSI input for FastICA convergence tests."""
    rng = np.random.default_rng(42)
    da = xr.DataArray(
        rng.random((10, 4, 6, 8)),
        name="power_doppler",
        dims=["time", "k", "j", "i"],
        coords={
            "time": xr.DataArray(
                10.0 + np.arange(10) * 0.5,
                dims=["time"],
                attrs={"units": "s"},
            ),
            "k": np.arange(4),
            "j": np.arange(6),
            "i": np.arange(8),
        },
        attrs={"long_name": "Intensity", "units": "a.u."},
    )
    return attach_voxel_to_world_index(
        da,
        np.diag([0.2, 0.1, 0.05, 1.0]),
        world_coord_attrs={
            "z": {"units": "mm", "voxdim": 0.2},
            "y": {"units": "mm", "voxdim": 0.1},
            "x": {"units": "mm", "voxdim": 0.05},
        },
    )


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
def test_fit_transform_matches_fit_then_transform(sample_fusi_3dt, mode):
    """fit_transform matches calling fit followed by transform."""
    model_direct = FastICA(**FASTICA_TEST_KWARGS, mode=mode)
    direct = model_direct.fit_transform(sample_fusi_3dt)

    model_two_step = FastICA(**FASTICA_TEST_KWARGS, mode=mode)
    two_step = model_two_step.fit(sample_fusi_3dt).transform(sample_fusi_3dt)

    xr.testing.assert_identical(direct, two_step)


@pytest.mark.parametrize("mode", ["spatial", "temporal"])
def test_inverse_transform_matches_sklearn(sample_fusi_3dt, mode):
    """inverse_transform matches sklearn FastICA reconstruction for both modes."""
    stacked = sample_fusi_3dt.transpose("time", "k", "j", "i").stack(
        feature=["k", "j", "i"]
    )
    X = np.asarray(stacked.values, dtype=np.float64)

    model = FastICA(**FASTICA_TEST_KWARGS, mode=mode)
    reconstructed = model.inverse_transform(model.fit_transform(sample_fusi_3dt))

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
        reconstructed.stack(feature=["k", "j", "i"]).values,
        sklearn_reconstructed,
    )
    assert reconstructed.name == sample_fusi_3dt.name
    assert reconstructed.attrs == sample_fusi_3dt.attrs


@pytest.mark.parametrize("mode", ["spatial", "temporal"])
def test_wrapper_matches_sklearn_attributes(sample_fusi_3dt, mode):
    """Wrapper exposes the same learned matrices as sklearn FastICA for both modes."""
    stacked = sample_fusi_3dt.transpose("time", "k", "j", "i").stack(
        feature=["k", "j", "i"]
    )
    X = np.asarray(stacked.values, dtype=np.float64)

    model = FastICA(**FASTICA_TEST_KWARGS, mode=mode).fit(sample_fusi_3dt)

    if mode == "temporal":
        sklearn_model = SklearnFastICA(**FASTICA_TEST_KWARGS).fit(X)
        np.testing.assert_allclose(
            model.transform(sample_fusi_3dt).values,
            sklearn_model.transform(X),
        )
        np.testing.assert_allclose(
            model.maps_.stack(feature=["k", "j", "i"]).values,
            sklearn_model.components_,
        )
        np.testing.assert_allclose(
            model.mean_.stack(feature=["k", "j", "i"]).values, sklearn_model.mean_
        )
        np.testing.assert_allclose(
            model.whitening_.stack(feature=["k", "j", "i"]).values,
            sklearn_model.whitening_,
        )
        assert model.n_iter_ == sklearn_model.n_iter_
    else:
        # Spatial ICA fits on (n_voxels, n_time); sources are spatial maps.
        sklearn_model = SklearnFastICA(**FASTICA_TEST_KWARGS).fit(X.T)
        spatial_maps = sklearn_model.transform(X.T).T
        voxel_mean = X.mean(axis=0)
        np.testing.assert_allclose(
            model.transform(sample_fusi_3dt).values,
            (X - voxel_mean) @ spatial_maps.T,
        )
        np.testing.assert_allclose(
            model.maps_.stack(feature=["k", "j", "i"]).values,
            spatial_maps,
        )
        np.testing.assert_allclose(
            model.mean_.stack(feature=["k", "j", "i"]).values,
            voxel_mean,
        )
        assert not hasattr(model, "whitening_")
        assert model.n_iter_ == sklearn_model.n_iter_


@pytest.mark.parametrize("mode", ["spatial", "temporal"])
def test_inverse_transform_from_numpy_returns_dataarray(sample_fusi_3dt, mode):
    """inverse_transform accepts ndarray input and returns DataArray."""
    model = FastICA(**FASTICA_TEST_KWARGS, mode=mode)
    signals = model.fit_transform(sample_fusi_3dt).values

    reconstructed = model.inverse_transform(signals)

    assert isinstance(reconstructed, xr.DataArray)
    assert reconstructed.dims == sample_fusi_3dt.dims
    np.testing.assert_array_equal(
        reconstructed.coords["time"], np.arange(sample_fusi_3dt.sizes["time"])
    )


def test_inverse_transform_raises_for_invalid_dataarray_dims(sample_fusi_3dt):
    """inverse_transform raises when DataArray dims are not time/component."""
    model = FastICA(**FASTICA_TEST_KWARGS).fit(sample_fusi_3dt)
    bad = xr.DataArray(
        np.zeros((sample_fusi_3dt.sizes["time"], 2)),
        dims=["time", "region"],
    )

    with pytest.raises(ValueError, match="exactly the dimensions"):
        model.inverse_transform(bad)


def test_inverse_transform_raises_for_component_count_mismatch(sample_fusi_3dt):
    """inverse_transform raises when component count differs from fitted FastICA."""
    model = FastICA(**FASTICA_TEST_KWARGS).fit(sample_fusi_3dt)
    scores = model.transform(sample_fusi_3dt)
    bad = scores.isel(component=slice(0, 1))

    with pytest.raises(ValueError, match="but FastICA was fitted with"):
        model.inverse_transform(bad)


def test_inverse_transform_raises_for_invalid_numpy_shape(sample_fusi_3dt):
    """inverse_transform raises when ndarray input is not 2D."""
    model = FastICA(**FASTICA_TEST_KWARGS).fit(sample_fusi_3dt)

    with pytest.raises(ValueError, match="must be 2D"):
        model.inverse_transform(np.zeros((sample_fusi_3dt.sizes["time"], 2, 1)))


def test_inverse_transform_raises_for_invalid_input_type(sample_fusi_3dt):
    """inverse_transform raises TypeError for unsupported input types."""
    model = FastICA(**FASTICA_TEST_KWARGS).fit(sample_fusi_3dt)

    with pytest.raises(TypeError, match="DataArray or ndarray"):
        model.inverse_transform([1, 2, 3])  # ty: ignore[invalid-argument-type]


def test_fit_rejects_invalid_mode(sample_fusi_3dt):
    """fit raises ValueError for unknown mode values."""
    with pytest.raises(ValueError, match="mode must be"):
        FastICA(mode="invalid").fit(sample_fusi_3dt)  # ty: ignore[invalid-argument-type]


def test_fit_requires_time_dimension(sample_fusi_3dt):
    """fit raises when the input has no `time` dimension."""
    no_time = sample_fusi_3dt.isel(time=0, drop=True)

    with pytest.raises(ValueError, match="must have a 'time' dimension"):
        FastICA().fit(no_time)


def test_fit_requires_more_than_one_timepoint(sample_fusi_3dt):
    """fit raises when only one timepoint is provided."""
    single_timepoint = sample_fusi_3dt.isel(time=[0])

    with pytest.raises(ValueError, match="requires more than 1 timepoint"):
        FastICA().fit(single_timepoint)


def test_fit_requires_spatial_dimension():
    """fit raises when input has no spatial dimensions."""
    only_time = xr.DataArray(np.arange(30.0), dims=["time"])

    with pytest.raises(ValueError, match="at least one spatial dimension"):
        FastICA().fit(only_time)


def test_fit_rejects_unexpected_fit_params(sample_fusi_3dt):
    """fit raises when unexpected sklearn-style fit params are provided."""
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        FastICA().fit(
            sample_fusi_3dt,
            sample_weight=np.ones(sample_fusi_3dt.sizes["time"]),  # ty: ignore[unknown-argument]
        )


def test_fit_transform_rejects_unexpected_fit_params(sample_fusi_3dt):
    """fit_transform raises when unexpected sklearn-style fit params are provided."""
    with pytest.raises(TypeError, match="Unexpected fit parameters"):
        FastICA().fit_transform(
            sample_fusi_3dt,
            sample_weight=np.ones(sample_fusi_3dt.sizes["time"]),
        )


def test_transform_checks_spatial_layout(sample_fusi_3dt):
    """transform raises if spatial layout differs from fit."""
    model = FastICA(**FASTICA_TEST_KWARGS).fit(sample_fusi_3dt)
    bad = sample_fusi_3dt.isel(i=slice(0, 4))

    with pytest.raises(ValueError, match="Spatial dimension 'i' has size"):
        model.transform(bad)


def test_transform_checks_spatial_dimension_names(sample_fusi_3dt):
    """transform raises if spatial dimension names differ from fit."""
    model = FastICA(**FASTICA_TEST_KWARGS).fit(sample_fusi_3dt)
    bad = sample_fusi_3dt.rename({"i": "region"})

    with pytest.raises(ValueError, match="spatial dimensions do not match"):
        model.transform(bad)


def test_transform_without_time_coordinate_uses_index(sample_fusi_3dt):
    """transform falls back to integer time coordinate when absent."""
    model = FastICA(**FASTICA_TEST_KWARGS).fit(sample_fusi_3dt)
    no_time_coord = xr.DataArray(
        sample_fusi_3dt.values,
        dims=sample_fusi_3dt.dims,
        coords={
            "k": sample_fusi_3dt.coords["k"],
            "j": sample_fusi_3dt.coords["j"],
            "i": sample_fusi_3dt.coords["i"],
        },
    )

    transformed = model.transform(no_time_coord)

    np.testing.assert_array_equal(
        transformed.coords["time"].values,
        np.arange(sample_fusi_3dt.sizes["time"]),
    )


def test_transform_chunked_time_reports_transform_operation(sample_fusi_3dt):
    """transform chunking error message identifies FastICA.transform."""
    model = FastICA(**FASTICA_TEST_KWARGS).fit(sample_fusi_3dt)
    chunked = sample_fusi_3dt.chunk({"time": 5})

    with pytest.raises(
        ValueError, match="FastICA.transform requires the full time series"
    ):
        model.transform(chunked)


def test_sklearn_interface_fitted_state(sample_fusi_3dt):
    """Estimator exposes sklearn fitted-state behavior."""
    model = FastICA(**FASTICA_TEST_KWARGS)
    with pytest.raises(Exception):
        check_is_fitted(model)

    check_is_fitted(model.fit(sample_fusi_3dt))


def test_fit_failure_does_not_mark_estimator_fitted(sample_fusi_3dt, monkeypatch):
    """Estimator remains unfitted when underlying sklearn FastICA fit fails."""
    import confusius.decomposition.fastica as fastica_module

    def _raise_fit(self, X, j=None):
        raise RuntimeError("fit failed")

    monkeypatch.setattr(fastica_module._SklearnFastICA, "fit", _raise_fit)

    model = FastICA(**FASTICA_TEST_KWARGS)
    with pytest.raises(RuntimeError, match="fit failed"):
        model.fit(sample_fusi_3dt)

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


def test_mask_restricts_features(sample_fusi_3dt):
    """mask restricts fitted feature count to selected voxels."""
    mask = xr.DataArray(
        np.zeros(
            (
                sample_fusi_3dt.sizes["k"],
                sample_fusi_3dt.sizes["j"],
                sample_fusi_3dt.sizes["i"],
            ),
            dtype=bool,
        ),
        dims=["k", "j", "i"],
        coords={
            "k": sample_fusi_3dt.coords["k"],
            "j": sample_fusi_3dt.coords["j"],
            "i": sample_fusi_3dt.coords["i"],
        },
    )
    mask.values[:, :2, :] = True

    model = FastICA(**FASTICA_TEST_KWARGS, mask=mask).fit(sample_fusi_3dt)

    assert model.n_features_in_ == int(mask.values.sum())


def test_masked_fit_reconstructs_full_geometry_with_zero_fill(sample_fusi_3dt):
    """Masked FastICA keeps full geometry and fills outside-mask voxels with zero."""
    mask = xr.DataArray(
        np.zeros(
            (
                sample_fusi_3dt.sizes["k"],
                sample_fusi_3dt.sizes["j"],
                sample_fusi_3dt.sizes["i"],
            ),
            dtype=bool,
        ),
        dims=["k", "j", "i"],
        coords={
            "k": sample_fusi_3dt.coords["k"],
            "j": sample_fusi_3dt.coords["j"],
            "i": sample_fusi_3dt.coords["i"],
        },
    )
    mask.values[:, :2, :] = True

    model = FastICA(**FASTICA_TEST_KWARGS, mask=mask).fit(sample_fusi_3dt)
    reconstructed = model.inverse_transform(model.transform(sample_fusi_3dt))

    assert reconstructed.dims == sample_fusi_3dt.dims
    np.testing.assert_array_equal(
        reconstructed.where(~mask, other=np.nan).fillna(0.0).values,
        0.0,
    )
    np.testing.assert_array_equal(
        model.maps_.where(~mask, other=np.nan).fillna(0.0).values,
        0.0,
    )
    np.testing.assert_array_equal(
        model.mean_.where(~mask, other=np.nan).fillna(0.0).values,
        0.0,
    )


def test_mask_mismatch_raises(sample_fusi_3dt):
    """fit raises when mask does not match spatial dimensions."""
    bad_mask = xr.DataArray(np.ones((3, 3), dtype=bool), dims=["j", "i"])

    with pytest.raises(ValueError, match="missing from mask"):
        FastICA(mask=bad_mask).fit(sample_fusi_3dt)

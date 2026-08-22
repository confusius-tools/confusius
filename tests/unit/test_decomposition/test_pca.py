"""Tests for confusius.decomposition.PCA."""

import numpy as np
import pytest
import xarray as xr
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.utils.validation import check_is_fitted

from confusius._utils.geometry import (
    attach_voxel_to_world_index,
    get_voxel_to_world_affine,
    get_voxel_to_world_units,
)
from confusius.decomposition import PCA


def _make_mask(
    reference: xr.DataArray, dims: tuple[str, ...] = ("k", "j", "i")
) -> xr.DataArray:
    """Create an empty boolean mask sharing `reference`'s voxel grid.

    Parameters
    ----------
    reference : xarray.DataArray
        Canonical DataArray supplying the voxel grid.
    dims : tuple[str, ...], default: ("k", "j", "i")
        Dimension order for the mask.

    Returns
    -------
    xarray.DataArray
        Zero-valued mask carrying `reference`'s `VoxelToWorldIndex`.
    """
    mask = xr.DataArray(
        np.zeros(tuple(reference.sizes[dim] for dim in dims), dtype=bool),
        dims=dims,
        coords={dim: reference.coords[dim] for dim in dims},
    )
    return attach_voxel_to_world_index(
        mask,
        get_voxel_to_world_affine(reference),
        units=get_voxel_to_world_units(reference),
    )


def test_rejects_non_voxeldata_input():
    """Fitting on a non-spatial (time, region) table is rejected, not silently used.

    PCA is VoxelData-only: decomposing an already-reduced signals table has no
    spatial structure to track, so it offers nothing over calling scikit-learn
    directly on the array's values.
    """
    data = xr.DataArray(
        np.arange(18.0).reshape(6, 3),
        dims=["time", "region"],
        coords={"region": ["A", "B", "C"]},
    )

    with pytest.raises(ValueError, match="missing voxel dimension"):
        PCA(n_components=2, random_state=0).fit(data)


@pytest.mark.parametrize("mode", ["temporal", "spatial"])
def test_fit_transform_matches_fit_then_transform(sample_voxeldata_3dt, mode):
    """fit_transform matches calling fit followed by transform."""
    model_direct = PCA(n_components=6, random_state=0, mode=mode)
    direct = model_direct.fit_transform(sample_voxeldata_3dt)

    model_two_step = PCA(n_components=6, random_state=0, mode=mode)
    two_step = model_two_step.fit(sample_voxeldata_3dt).transform(sample_voxeldata_3dt)

    xr.testing.assert_identical(direct, two_step)


def test_wrapper_matches_sklearn_attributes(sample_voxeldata_3dt):
    """Temporal wrapper exposes the same learned quantities as sklearn PCA."""
    stacked = sample_voxeldata_3dt.transpose("time", "k", "j", "i").stack(
        feature=["k", "j", "i"]
    )
    X = np.asarray(stacked.values, dtype=np.float64)

    model = PCA(n_components=4, random_state=0).fit(sample_voxeldata_3dt)
    sklearn_model = SklearnPCA(n_components=4, random_state=0).fit(X)

    np.testing.assert_allclose(
        model.transform(sample_voxeldata_3dt).values,
        sklearn_model.transform(X),
    )
    np.testing.assert_allclose(
        model.maps_.stack(feature=["k", "j", "i"]).values,
        sklearn_model.components_,
    )
    np.testing.assert_allclose(
        model.mean_.stack(feature=["k", "j", "i"]).values,
        sklearn_model.mean_,
    )
    np.testing.assert_allclose(
        model.explained_variance_.values,
        sklearn_model.explained_variance_,
    )
    np.testing.assert_allclose(
        model.explained_variance_ratio_.values,
        sklearn_model.explained_variance_ratio_,
    )
    np.testing.assert_allclose(
        model.singular_values_.values,
        sklearn_model.singular_values_,
    )
    assert model.n_components_ == sklearn_model.n_components_
    np.testing.assert_allclose(model.noise_variance_, sklearn_model.noise_variance_)


def test_inverse_transform_reconstructs_with_all_components(sample_voxeldata_3dt):
    """Using all components reconstructs the original data."""
    model = PCA(random_state=0)

    signals = model.fit_transform(sample_voxeldata_3dt)
    reconstructed = model.inverse_transform(signals)

    assert reconstructed.dims == sample_voxeldata_3dt.dims
    np.testing.assert_allclose(
        reconstructed.coords["time"], sample_voxeldata_3dt.coords["time"]
    )
    np.testing.assert_allclose(
        reconstructed.values, sample_voxeldata_3dt.values, atol=1e-10
    )
    assert reconstructed.name == sample_voxeldata_3dt.name
    assert reconstructed.attrs == sample_voxeldata_3dt.attrs


def test_inverse_transform_from_numpy_returns_dataarray(sample_voxeldata_3dt):
    """inverse_transform accepts ndarray input and returns DataArray."""
    model = PCA(n_components=5, random_state=0)
    scores = model.fit_transform(sample_voxeldata_3dt).values

    reconstructed = model.inverse_transform(scores)

    assert isinstance(reconstructed, xr.DataArray)
    assert reconstructed.dims == sample_voxeldata_3dt.dims
    np.testing.assert_array_equal(
        reconstructed.coords["time"], np.arange(sample_voxeldata_3dt.sizes["time"])
    )


def test_inverse_transform_raises_for_invalid_dataarray_dims(sample_voxeldata_3dt):
    """inverse_transform raises when DataArray dims are not time/component."""
    model = PCA(n_components=4, random_state=0).fit(sample_voxeldata_3dt)
    bad = xr.DataArray(
        np.zeros((sample_voxeldata_3dt.sizes["time"], 4)),
        dims=["time", "region"],
    )

    with pytest.raises(ValueError, match="exactly the dimensions"):
        model.inverse_transform(bad)


def test_inverse_transform_raises_for_component_count_mismatch(sample_voxeldata_3dt):
    """inverse_transform raises when component count differs from fitted PCA."""
    model = PCA(n_components=4, random_state=0).fit(sample_voxeldata_3dt)
    scores = model.transform(sample_voxeldata_3dt)
    bad = scores.isel(component=slice(0, 3))

    with pytest.raises(ValueError, match="but PCA was fitted with"):
        model.inverse_transform(bad)


def test_inverse_transform_raises_for_invalid_numpy_shape(sample_voxeldata_3dt):
    """inverse_transform raises when ndarray input is not 2D."""
    model = PCA(n_components=4, random_state=0).fit(sample_voxeldata_3dt)

    with pytest.raises(ValueError, match="must be 2D"):
        model.inverse_transform(np.zeros((sample_voxeldata_3dt.sizes["time"], 4, 1)))


def test_inverse_transform_raises_for_invalid_input_type(sample_voxeldata_3dt):
    """inverse_transform raises TypeError for unsupported input types."""
    model = PCA(n_components=4, random_state=0).fit(sample_voxeldata_3dt)

    with pytest.raises(TypeError, match="DataArray or ndarray"):
        model.inverse_transform([1, 2, 3])  # ty: ignore[invalid-argument-type]


def test_fit_requires_time_dimension(sample_voxeldata_3dt):
    """fit raises when the input has no `time` dimension."""
    no_time = sample_voxeldata_3dt.isel(time=0, drop=True)

    with pytest.raises(ValueError, match="must have a 'time' dimension"):
        PCA().fit(no_time)


def test_fit_requires_more_than_one_timepoint(sample_voxeldata_3dt):
    """fit raises when only one timepoint is provided."""
    single_timepoint = sample_voxeldata_3dt.isel(time=[0])

    with pytest.raises(ValueError, match="requires more than 1 timepoint"):
        PCA().fit(single_timepoint)


def test_fit_requires_spatial_dimension():
    """fit raises when input has no spatial (native voxel) dimensions."""
    only_time = xr.DataArray(np.arange(30.0), dims=["time"])

    with pytest.raises(ValueError, match="missing voxel dimension"):
        PCA().fit(only_time)


def test_mask_dim_order_is_canonicalized(sample_voxeldata_3dt):
    """Wrong-order VoxelData masks are canonicalized before fitting."""
    mask = _make_mask(sample_voxeldata_3dt, dims=("j", "k", "i"))
    mask.values[:] = True

    result = PCA(n_components=2, mask=mask).fit(sample_voxeldata_3dt)

    assert result.spatial_dims_ == ("k", "j", "i")


def test_fit_rejects_unexpected_fit_params(sample_voxeldata_3dt):
    """fit raises when unexpected sklearn-style fit params are provided."""
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        PCA().fit(
            sample_voxeldata_3dt,
            sample_weight=np.ones(sample_voxeldata_3dt.sizes["time"]),  # ty: ignore[unknown-argument]
        )


def test_fit_transform_rejects_unexpected_fit_params(sample_voxeldata_3dt):
    """fit_transform raises when unexpected sklearn-style fit params are provided."""
    with pytest.raises(TypeError, match="Unexpected fit parameters"):
        PCA().fit_transform(
            sample_voxeldata_3dt,
            sample_weight=np.ones(sample_voxeldata_3dt.sizes["time"]),
        )


def test_transform_checks_spatial_layout(sample_voxeldata_3dt):
    """transform raises if spatial layout differs from fit."""
    model = PCA(n_components=4, random_state=0).fit(sample_voxeldata_3dt)
    bad = sample_voxeldata_3dt.isel(i=slice(0, 4))

    with pytest.raises(ValueError, match="Spatial dimension 'i' has size"):
        model.transform(bad)


def test_transform_checks_spatial_dimension_names(sample_voxeldata_3dt):
    """transform rejects data missing a native voxel dimension."""
    model = PCA(n_components=4, random_state=0).fit(sample_voxeldata_3dt)
    bad = sample_voxeldata_3dt.rename({"i": "region"})

    with pytest.raises(ValueError, match="missing voxel dimension"):
        model.transform(bad)


def test_transform_rejects_missing_time_coordinate(sample_voxeldata_3dt):
    """transform rejects data with a time dim but no time coordinate.

    A VoxelData array's time dim must always carry a time coordinate; this isn't
    a fallback case to paper over.
    """
    model = PCA(n_components=4, random_state=0).fit(sample_voxeldata_3dt)
    no_time_coord = sample_voxeldata_3dt.drop_vars("time")

    with pytest.raises(ValueError, match="Missing required coordinate"):
        model.transform(no_time_coord)


def test_transform_chunked_time_reports_transform_operation(sample_voxeldata_3dt):
    """transform chunking error message identifies PCA.transform."""
    model = PCA(n_components=4, random_state=0).fit(sample_voxeldata_3dt)
    chunked = sample_voxeldata_3dt.chunk({"time": 5})

    with pytest.raises(ValueError, match="PCA.transform requires the full time series"):
        model.transform(chunked)


def test_sklearn_interface_fitted_state(sample_voxeldata_3dt):
    """Estimator exposes sklearn fitted-state behavior."""
    model = PCA(n_components=3, random_state=0)
    with pytest.raises(Exception):
        check_is_fitted(model)

    check_is_fitted(model.fit(sample_voxeldata_3dt))


def test_fit_failure_does_not_mark_estimator_fitted(sample_voxeldata_3dt, monkeypatch):
    """Estimator remains unfitted when underlying sklearn PCA fit fails."""
    import confusius.decomposition.pca as pca_module

    def _raise_fit(self, X, j=None):
        raise RuntimeError("fit failed")

    monkeypatch.setattr(pca_module._SklearnPCA, "fit", _raise_fit)

    model = PCA(n_components=3, random_state=0)
    with pytest.raises(RuntimeError, match="fit failed"):
        model.fit(sample_voxeldata_3dt)

    assert not hasattr(model, "_estimator")
    assert not model.__sklearn_is_fitted__()
    with pytest.raises(Exception):
        check_is_fitted(model)


def test_get_params_includes_constructor_arguments():
    """get_params includes all constructor arguments."""
    model = PCA(
        n_components=3,
        whiten=True,
        svd_solver="full",
        tol=1e-3,
        iterated_power=4,
        n_oversamples=12,
        power_iteration_normalizer="LU",
        random_state=42,
        mode="spatial",
    )
    params = model.get_params()

    assert params["n_components"] == 3
    assert params["whiten"] is True
    assert params["svd_solver"] == "full"
    assert params["tol"] == 1e-3
    assert params["iterated_power"] == 4
    assert params["n_oversamples"] == 12
    assert params["power_iteration_normalizer"] == "LU"
    assert params["random_state"] == 42
    assert params["mode"] == "spatial"


def test_set_params_updates_values():
    """set_params updates constructor parameters."""
    model = PCA()
    model.set_params(n_components=2, svd_solver="full", whiten=True, mode="spatial")

    assert model.n_components == 2
    assert model.svd_solver == "full"
    assert model.whiten is True
    assert model.mode == "spatial"


def test_randomized_solver_reproducible_with_random_state(sample_voxeldata_3dt):
    """Randomized solver gives reproducible results with fixed random_state."""
    model_1 = PCA(n_components=3, svd_solver="randomized", random_state=0)
    model_2 = PCA(n_components=3, svd_solver="randomized", random_state=0)

    signals_1 = model_1.fit_transform(sample_voxeldata_3dt)
    signals_2 = model_2.fit_transform(sample_voxeldata_3dt)

    np.testing.assert_allclose(signals_1.values, signals_2.values)
    np.testing.assert_allclose(model_1.maps_.values, model_2.maps_.values)


def test_spatial_mode_matches_reference_implementation(sample_voxeldata_3dt):
    """Spatial mode matches sklearn PCA fitted on transposed data."""
    stacked = sample_voxeldata_3dt.transpose("time", "k", "j", "i").stack(
        feature=["k", "j", "i"]
    )
    X = np.asarray(stacked.values, dtype=np.float64)

    model = PCA(n_components=4, random_state=0, mode="spatial").fit(
        sample_voxeldata_3dt
    )
    sklearn_model = SklearnPCA(n_components=4, random_state=0).fit(X.T)

    spatial_maps = sklearn_model.transform(X.T).T
    voxel_mean = X.mean(axis=0)
    reference_signals = (X - voxel_mean) @ spatial_maps.T
    reference_reconstructed = reference_signals @ spatial_maps + voxel_mean

    np.testing.assert_allclose(
        model.transform(sample_voxeldata_3dt).values,
        reference_signals,
    )
    np.testing.assert_allclose(
        model.maps_.stack(feature=["k", "j", "i"]).values,
        spatial_maps,
    )
    np.testing.assert_allclose(
        model.mean_.stack(feature=["k", "j", "i"]).values,
        voxel_mean,
    )
    np.testing.assert_allclose(
        model.inverse_transform(model.transform(sample_voxeldata_3dt))
        .stack(feature=["k", "j", "i"])
        .values,
        reference_reconstructed,
    )


def test_fit_raises_for_invalid_mode(sample_voxeldata_3dt):
    """fit raises for unsupported PCA mode."""
    with pytest.raises(ValueError, match="mode must be 'temporal' or 'spatial'"):
        PCA(mode="invalid").fit(sample_voxeldata_3dt)  # ty: ignore[invalid-argument-type]


def test_mask_restricts_features(sample_voxeldata_3dt):
    """mask restricts fitted feature count to selected voxels."""
    mask = _make_mask(sample_voxeldata_3dt)
    mask.values[:2, :, :] = True

    model = PCA(n_components=3, random_state=0, mask=mask).fit(sample_voxeldata_3dt)

    assert model.n_features_in_ == int(mask.values.sum())


def test_masked_fit_reconstructs_full_geometry_with_zero_fill(sample_voxeldata_3dt):
    """Masked PCA keeps full geometry and fills outside-mask voxels with zero."""
    mask = _make_mask(sample_voxeldata_3dt)
    mask.values[:2, :, :] = True

    model = PCA(n_components=3, random_state=0, mask=mask).fit(sample_voxeldata_3dt)
    reconstructed = model.inverse_transform(model.transform(sample_voxeldata_3dt))

    assert reconstructed.dims == sample_voxeldata_3dt.dims
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


def test_mask_mismatch_raises(sample_voxeldata_3dt):
    """fit raises when mask does not match spatial dimensions."""
    bad_mask = xr.DataArray(np.ones((3, 3), dtype=bool), dims=["j", "i"])

    with pytest.raises(
        ValueError, match="native voxel dimensions|missing voxel dimension"
    ):
        PCA(mask=bad_mask).fit(sample_voxeldata_3dt)

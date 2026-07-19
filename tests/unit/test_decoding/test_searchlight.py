"""Tests for `confusius.decoding.SearchLight`."""

import numpy as np
import pytest
import xarray as xr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.utils.validation import check_is_fitted

from confusius.decoding import SearchLight


def _reference_scores(data, mask, y, radius, estimator, cv):
    """Brute-force searchlight, used as the correctness reference.

    Loops over every masked voxel in Python, computes Euclidean distances to every
    other masked voxel directly, and cross-validates on the resulting neighbourhood.

    Parameters
    ----------
    data : xarray.DataArray
        `(time, z, y, x)` volume.
    mask : xarray.DataArray
        Boolean spatial mask.
    y : numpy.ndarray
        Targets aligned with `data`'s time axis.
    radius : float
        Neighbourhood radius in coordinate units.
    estimator : sklearn.base.BaseEstimator
        Estimator cloned into each neighbourhood.
    cv : sklearn.model_selection.BaseCrossValidator
        Cross-validation splitter.

    Returns
    -------
    numpy.ndarray
        Flat `(n_masked,)` array of mean cross-validation scores.
    """
    from sklearn.base import clone

    dims = tuple(str(d) for d in mask.dims)
    grids = np.meshgrid(*[mask.coords[d].values for d in dims], indexing="ij")
    coords = np.stack([g.ravel() for g in grids], axis=-1)
    flat_mask = np.asarray(mask.values).ravel()
    coords = coords[flat_mask]

    features = np.asarray(
        data.transpose("time", *dims).values.reshape(data.sizes["time"], -1)
    )[:, flat_mask]

    scores = []
    for centre in coords:
        distances = np.sqrt(((coords - centre) ** 2).sum(axis=1))
        neighbours = np.flatnonzero(distances <= radius)
        fold_scores = cross_val_score(
            clone(estimator), features[:, neighbours], y, cv=cv, n_jobs=1
        )
        scores.append(float(np.mean(fold_scores)))
    return np.asarray(scores)


def test_matches_brute_force_reference(decoding_volume, full_mask, rng):
    """SearchLight reproduces an explicit voxel-by-voxel reference implementation."""
    y = rng.standard_normal(decoding_volume.sizes["time"])

    searchlight = SearchLight(
        mask=full_mask, estimator=Ridge(alpha=1.0), radius=0.25, cv=3
    )
    searchlight.fit(decoding_volume, y)

    expected = _reference_scores(
        decoding_volume,
        full_mask,
        y,
        radius=0.25,
        estimator=Ridge(alpha=1.0),
        cv=KFold(n_splits=3, shuffle=False),
    )
    actual = searchlight.scores_.values.ravel()

    np.testing.assert_allclose(actual, expected)


def test_recovers_planted_signal(decoding_volume, full_mask, rng):
    """The score map peaks where a decodable signal was planted."""
    y = rng.standard_normal(decoding_volume.sizes["time"])

    # Plant y into a 2x2 in-plane patch at z index 0, y indices 2:4, x indices 2:4.
    # Voxels centred on that patch see a strongly predictive neighbourhood; every
    # other voxel sees noise.
    #
    # Index positionally, not by label. The y and x coordinates are built with
    # `np.arange(n) * 0.2`, so the third value is 0.6000000000000001 and a label
    # lookup of 0.6 raises KeyError.
    planted = decoding_volume.copy(deep=True)
    planted.values[:, 0, 2:4, 2:4] += 3.0 * y[:, None, None]

    searchlight = SearchLight(
        mask=full_mask, estimator=Ridge(alpha=1.0), radius=0.25, cv=3
    )
    searchlight.fit(planted, y)

    z_index, y_index, x_index = np.unravel_index(
        int(np.argmax(searchlight.scores_.values)), searchlight.scores_.shape
    )
    assert z_index == 0
    assert y_index in (2, 3)
    assert x_index in (2, 3)


def test_scores_geometry_and_metadata(decoding_volume, full_mask, rng):
    """`scores_` is a spatial map carrying the mask geometry and a long_name."""
    y = rng.standard_normal(decoding_volume.sizes["time"])
    searchlight = SearchLight(
        mask=full_mask, estimator=Ridge(alpha=1.0), radius=0.25, cv=3
    )
    searchlight.fit(decoding_volume, y)

    assert searchlight.scores_.dims == ("z", "y", "x")
    assert searchlight.scores_.shape == (2, 5, 6)
    np.testing.assert_array_equal(
        searchlight.scores_.y.values, decoding_volume.y.values
    )
    assert searchlight.scores_.attrs["long_name"] == "Searchlight CV score"


def test_is_fitted_protocol(decoding_volume, full_mask, rng):
    """`check_is_fitted` fails before `fit` and passes after."""
    y = rng.standard_normal(decoding_volume.sizes["time"])
    searchlight = SearchLight(
        mask=full_mask, estimator=Ridge(alpha=1.0), radius=0.25, cv=3
    )
    with pytest.raises(Exception, match="not fitted"):
        check_is_fitted(searchlight)

    searchlight.fit(decoding_volume, y)
    check_is_fitted(searchlight)


def test_score_refuses_single_number(decoding_volume, full_mask, rng):
    """`score` raises, because a searchlight has one score per voxel."""
    y = rng.standard_normal(decoding_volume.sizes["time"])
    searchlight = SearchLight(
        mask=full_mask, estimator=Ridge(alpha=1.0), radius=0.25, cv=3
    ).fit(decoding_volume, y)

    with pytest.raises(NotImplementedError, match="no single score"):
        searchlight.score(decoding_volume, y)


def test_single_slice_volume(decoding_volume, rng):
    """A size-1 spatial dimension gives in-plane discs rather than raising.

    Single-plane fUSI acquisitions are common. Every voxel shares the same `z`
    coordinate, so `z` contributes nothing to any distance and the sphere degenerates
    to a disc in the remaining plane.
    """
    volume = decoding_volume.isel(z=slice(0, 1))
    mask = xr.ones_like(volume.isel(time=0, drop=True), dtype=bool)
    y = rng.standard_normal(volume.sizes["time"])

    searchlight = SearchLight(
        mask=mask, estimator=Ridge(alpha=1.0), radius=0.25, cv=3
    ).fit(volume, y)

    assert searchlight.scores_.dims == ("z", "y", "x")
    assert searchlight.scores_.shape == (1, 5, 6)
    assert np.isfinite(searchlight.scores_.values).all()

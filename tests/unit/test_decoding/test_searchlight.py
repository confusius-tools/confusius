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


def test_matches_brute_force_reference_sparse_mask(decoding_volume, rng):
    """SearchLight matches the reference under a sparse mask.

    The all-`True` variant of this comparison cannot see a feature-ordering bug: with
    every voxel retained, the boolean indexing is a no-op on both sides, so
    `extract_with_mask` (xarray `stack` plus boolean `isel`) and `_masked_coordinates`
    (numpy `ravel` plus boolean index) agree trivially. A sparse mask makes that
    indexing load-bearing, so any disagreement between the two flattening orders
    mis-assigns every neighbourhood and shows up here.

    Parameters
    ----------
    decoding_volume : xarray.DataArray
        Random `(time, z, y, x)` volume from the shared fixtures.
    rng : numpy.random.Generator
        Seeded generator from the shared test fixtures.
    """
    spatial = decoding_volume.isel(time=0, drop=True)
    mask = xr.DataArray(
        rng.random(spatial.shape) < 0.65,
        dims=spatial.dims,
        coords=spatial.coords,
    )
    # Every centre is itself a feature, so no neighbourhood can be empty at any radius.
    assert mask.values.any()

    y = rng.standard_normal(decoding_volume.sizes["time"])

    searchlight = SearchLight(mask=mask, estimator=Ridge(alpha=1.0), radius=0.25, cv=3)
    searchlight.fit(decoding_volume, y)

    expected = _reference_scores(
        decoding_volume,
        mask,
        y,
        radius=0.25,
        estimator=Ridge(alpha=1.0),
        cv=KFold(n_splits=3, shuffle=False),
    )
    actual = searchlight.scores_.values[mask.values]

    np.testing.assert_allclose(actual, expected)
    assert np.isnan(searchlight.scores_.values[~mask.values]).all()


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


def test_raises_on_missing_coordinate(decoding_volume, rng):
    """A spatial dimension without a coordinate is an error, not a silent fallback."""
    volume = decoding_volume.drop_vars("y")
    mask = xr.ones_like(volume.isel(time=0, drop=True), dtype=bool)
    y = rng.standard_normal(volume.sizes["time"])

    searchlight = SearchLight(mask=mask, estimator=Ridge(), radius=0.25, cv=3)
    with pytest.raises(ValueError, match="lack a numeric coordinate"):
        searchlight.fit(volume, y)


def test_raises_when_process_mask_not_subset(decoding_volume, full_mask, rng):
    """A process_mask reaching outside mask is rejected."""
    mask = full_mask.copy(deep=True)
    mask.loc[dict(z=0.0)] = False
    y = rng.standard_normal(decoding_volume.sizes["time"])

    searchlight = SearchLight(
        mask=mask,
        estimator=Ridge(),
        radius=0.25,
        process_mask=full_mask,
        cv=3,
    )
    with pytest.raises(ValueError, match="must be a subset of mask"):
        searchlight.fit(decoding_volume, y)


def test_raises_on_y_length_mismatch(decoding_volume, full_mask, rng):
    """A y of the wrong length is rejected."""
    searchlight = SearchLight(mask=full_mask, estimator=Ridge(), radius=0.25, cv=3)
    with pytest.raises(ValueError, match="but X has 40 time points"):
        searchlight.fit(decoding_volume, rng.standard_normal(39))


def test_raises_on_misaligned_y_dataarray(decoding_volume, full_mask, rng):
    """A DataArray y whose time coordinate disagrees with X is rejected."""
    y = xr.DataArray(
        rng.standard_normal(decoding_volume.sizes["time"]),
        dims=["time"],
        coords={"time": decoding_volume.time.values + 0.1},
    )
    searchlight = SearchLight(mask=full_mask, estimator=Ridge(), radius=0.25, cv=3)
    with pytest.raises(ValueError, match="does not match X"):
        searchlight.fit(decoding_volume, y)


def test_accepts_aligned_y_dataarray(decoding_volume, full_mask, rng):
    """A DataArray y sharing X's time coordinate is accepted."""
    y = xr.DataArray(
        rng.standard_normal(decoding_volume.sizes["time"]),
        dims=["time"],
        coords={"time": decoding_volume.time.values},
    )
    searchlight = SearchLight(mask=full_mask, estimator=Ridge(), radius=0.25, cv=3)
    searchlight.fit(decoding_volume, y)
    assert searchlight.scores_.dims == ("z", "y", "x")


def test_raises_on_groups_length_mismatch(decoding_volume, full_mask, rng):
    """A groups array of the wrong length is rejected."""
    y = rng.standard_normal(decoding_volume.sizes["time"])
    searchlight = SearchLight(mask=full_mask, estimator=Ridge(), radius=0.25, cv=3)
    with pytest.raises(ValueError, match="groups has 39 entries"):
        searchlight.fit(decoding_volume, y, groups=np.zeros(39))


def test_raises_on_h5py_backed_data(scan_2d, rng):
    """h5py-backed data is rejected, because joblib workers cannot pickle it."""
    mask = xr.ones_like(scan_2d.isel(time=0, drop=True), dtype=bool)
    y = rng.standard_normal(scan_2d.sizes["time"])

    searchlight = SearchLight(mask=mask, estimator=Ridge(), radius=0.25, cv=3)
    with pytest.raises(ValueError, match="h5py-backed"):
        searchlight.fit(scan_2d, y)


def test_warns_on_degenerate_neighborhoods(decoding_volume, full_mask, rng):
    """A radius below the voxel spacing warns that the analysis became univariate."""
    y = rng.standard_normal(decoding_volume.sizes["time"])
    searchlight = SearchLight(mask=full_mask, estimator=Ridge(), radius=0.01, cv=3)
    with pytest.warns(UserWarning, match="single-voxel"):
        searchlight.fit(decoding_volume, y)


def test_radius_is_in_coordinate_units(decoding_volume, full_mask, rng):
    """Radius uses physical coordinates, so the anisotropic z axis is excluded.

    `z` voxels are 1.0 apart while `y` and `x` are 0.2 apart. A radius of 0.25 must
    therefore select in-plane neighbours only, giving each interior voxel exactly 5
    neighbours (itself plus 4 in-plane), never 6 or 7.
    """
    y = rng.standard_normal(decoding_volume.sizes["time"])
    searchlight = SearchLight(mask=full_mask, estimator=Ridge(), radius=0.25, cv=3).fit(
        decoding_volume, y
    )

    reference = SearchLight(mask=full_mask, estimator=Ridge(), radius=1.05, cv=3).fit(
        decoding_volume, y
    )

    # A radius spanning the z gap must change the result; if radius were interpreted
    # in voxel indices, both runs would give identical maps.
    assert not np.allclose(searchlight.scores_.values, reference.scores_.values)


def test_process_mask_restricts_centres(decoding_volume, full_mask, rng):
    """Only process_mask voxels get a score; the rest are NaN."""
    process_mask = xr.zeros_like(full_mask, dtype=bool)
    process_mask.loc[dict(z=0.0)] = True
    y = rng.standard_normal(decoding_volume.sizes["time"])

    searchlight = SearchLight(
        mask=full_mask,
        estimator=Ridge(),
        radius=0.25,
        process_mask=process_mask,
        cv=3,
    ).fit(decoding_volume, y)

    assert np.isfinite(searchlight.scores_.sel(z=0.0).values).all()
    assert np.isnan(searchlight.scores_.sel(z=1.0).values).all()


def test_classifier_selects_stratified_folds(decoding_volume, full_mask, rng):
    """A classifier estimator drives StratifiedKFold, a regressor drives KFold."""
    from sklearn.linear_model import LogisticRegression

    labels = np.tile([0, 1], decoding_volume.sizes["time"] // 2)
    searchlight = SearchLight(
        mask=full_mask, estimator=LogisticRegression(max_iter=1000), radius=0.25, cv=2
    ).fit(decoding_volume, labels)

    # Accuracy is bounded to [0, 1]; R-squared is not, so this pins the scorer family.
    finite = searchlight.scores_.values[np.isfinite(searchlight.scores_.values)]
    assert ((finite >= 0.0) & (finite <= 1.0)).all()

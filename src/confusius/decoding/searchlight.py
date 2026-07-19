"""Searchlight decoding for `(time, ...)` fUSI DataArrays.

Portions of this module are inspired by `nilearn.decoding.searchlight`, which is
licensed under the BSD-3-Clause License. See `NOTICE` for details.
"""

import warnings
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import xarray as xr
from scipy.spatial import KDTree
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.model_selection import (
    BaseCrossValidator,
    KFold,
    StratifiedKFold,
    cross_val_score,
)

from confusius._utils.io import is_h5py_backed
from confusius._utils.stack import find_stack_level
from confusius.extract import extract_with_mask, unmask
from confusius.validation import validate_mask, validate_time_series


def _masked_coordinates(mask: xr.DataArray) -> npt.NDArray[np.float64]:
    """Physical coordinates of the `True` voxels of a mask.

    Parameters
    ----------
    mask : xarray.DataArray
        Boolean spatial mask. Every dimension must carry a numeric coordinate.

    Returns
    -------
    numpy.ndarray
        `(n_masked, n_dims)` array of coordinates, in the order given by `mask.dims`.

    Raises
    ------
    ValueError
        If any dimension of the mask lacks a coordinate, or carries a non-numeric one.
        SearchLight measures its radius in coordinate units, so a missing coordinate
        would silently make the radius mean voxel indices, which is anisotropic.
    """
    dims = tuple(str(dim) for dim in mask.dims)
    invalid = [
        dim
        for dim in dims
        if dim not in mask.coords
        or not np.issubdtype(mask.coords[dim].dtype, np.number)
    ]
    if invalid:
        raise ValueError(
            f"Mask dimensions {invalid} lack a numeric coordinate. SearchLight "
            "measures `radius` in coordinate units, so every spatial dimension must "
            "carry one."
        )

    grids = np.meshgrid(
        *[np.asarray(mask.coords[dim].values, dtype=np.float64) for dim in dims],
        indexing="ij",
    )
    flat = np.stack([grid.ravel() for grid in grids], axis=-1)
    return flat[np.asarray(mask.values).ravel()]


def _neighborhood_indices(
    mask: xr.DataArray, process_mask: xr.DataArray, radius: float
) -> list[npt.NDArray[np.intp]]:
    """Feature indices falling within `radius` of every centre voxel.

    Parameters
    ----------
    mask : xarray.DataArray
        Boolean mask defining which voxels may act as features.
    process_mask : xarray.DataArray
        Boolean mask defining which voxels act as neighbourhood centres.
    radius : float
        Neighbourhood radius, in the units of the mask coordinates.

    Returns
    -------
    list[numpy.ndarray]
        One integer index array per centre, indexing into the masked feature axis.
    """
    feature_coords = _masked_coordinates(mask)
    centre_coords = _masked_coordinates(process_mask)
    tree = KDTree(feature_coords)
    return [
        np.asarray(sorted(indices), dtype=np.intp)
        for indices in tree.query_ball_point(centre_coords, r=radius)
    ]


def _resolve_cv(
    cv: int | BaseCrossValidator, *, classifier: bool
) -> BaseCrossValidator:
    """Turn the `cv` argument into a scikit-learn splitter.

    An integer becomes a splitter with `shuffle=False`, so folds are contiguous blocks
    of time. Shuffling would place temporally adjacent, and therefore highly
    correlated, fUSI volumes in both the training and test sets.

    Parameters
    ----------
    cv : int or sklearn.model_selection.BaseCrossValidator
        Number of contiguous folds, or a ready-made splitter passed through unchanged.
    classifier : bool
        Whether the estimator is a classifier, which selects `StratifiedKFold` over
        `KFold`.

    Returns
    -------
    sklearn.model_selection.BaseCrossValidator
        The resolved splitter.
    """
    if isinstance(cv, int):
        splitter = StratifiedKFold if classifier else KFold
        return splitter(n_splits=cv, shuffle=False)
    return cv


def _check_targets(y: npt.ArrayLike | xr.DataArray, data: xr.DataArray) -> npt.NDArray:
    """Validate targets against the sample axis of the data.

    Parameters
    ----------
    y : array-like or xarray.DataArray
        Targets aligned with `data`'s `time` axis. When a DataArray carrying a `time`
        coordinate is given, that coordinate is checked against `data`'s.
    data : xarray.DataArray
        `(time, ...)` volume providing the sample axis.

    Returns
    -------
    numpy.ndarray
        Targets as a plain array.

    Raises
    ------
    ValueError
        If the lengths disagree, or if a DataArray `y` carries a `time` coordinate
        that does not match `data`'s.
    """
    if isinstance(y, xr.DataArray):
        if "time" in y.coords and "time" in data.coords:
            if not np.array_equal(
                np.asarray(y.coords["time"].values),
                np.asarray(data.coords["time"].values),
            ):
                raise ValueError(
                    "y has a 'time' coordinate that does not match X. Resample y onto "
                    "X's acquisition times before fitting, for example with "
                    "`y.interp(time=X.time)`."
                )
        y_array = np.asarray(y.values)
    else:
        y_array = np.asarray(y)

    n_samples = data.sizes["time"]
    if y_array.shape[0] != n_samples:
        raise ValueError(
            f"y has {y_array.shape[0]} samples but X has {n_samples} time points."
        )
    return y_array


def _score_batch(
    estimator: BaseEstimator,
    features: npt.NDArray[np.float64],
    y: npt.NDArray,
    neighborhoods: list[npt.NDArray[np.intp]],
    cv: BaseCrossValidator,
    scoring: str | Callable | None,
    groups: npt.NDArray | None,
) -> list[float]:
    """Mean cross-validation score for a batch of neighbourhoods.

    The estimator is cloned per neighbourhood so that each fit is independent and the
    joblib worker holds no shared state.

    Parameters
    ----------
    estimator : sklearn.base.BaseEstimator
        Estimator to clone into each neighbourhood.
    features : numpy.ndarray
        `(n_samples, n_features)` masked data.
    y : numpy.ndarray
        `(n_samples,)` targets.
    neighborhoods : list[numpy.ndarray]
        Feature index arrays, one per centre in this batch.
    cv : sklearn.model_selection.BaseCrossValidator
        Cross-validation splitter.
    scoring : str, callable, or None
        Scorer passed to
        [`cross_val_score`][sklearn.model_selection.cross_val_score].
    groups : numpy.ndarray, optional
        Group labels forwarded to the splitter.

    Returns
    -------
    list[float]
        One mean score per neighbourhood, in input order.
    """
    return [
        float(
            np.mean(
                cross_val_score(
                    clone(estimator),
                    features[:, indices],
                    y,
                    cv=cv,
                    scoring=scoring,
                    groups=groups,
                    n_jobs=1,
                )
            )
        )
        for indices in neighborhoods
    ]


class SearchLight(BaseEstimator):
    """Searchlight decoder for fUSI data.

    For every voxel of `process_mask`, `fit` gathers the `mask` voxels lying within
    `radius`, cross-validates `estimator` on that neighbourhood, and stores the mean
    score. The result is a brain map answering "which regions carry information about
    `y`?".

    This estimator wraps scikit-learn while keeping xarray metadata:

    - Input data are expected as `(time, ...)` where `...` are spatial dimensions.
    - The `time` dimension is the sample axis. It need not be temporally ordered. For
      trial-averaged data, rename the trial dimension with `.rename(trial="time")`.
    - `scores_` is returned in the spatial geometry of `process_mask`.

    Parameters
    ----------
    mask : xarray.DataArray
        Boolean spatial mask selecting the voxels that may act as *features*. Every
        spatial dimension must carry a numeric coordinate, because `radius` is measured
        in coordinate units.
    estimator : sklearn.base.BaseEstimator
        Estimator or [`Pipeline`][sklearn.pipeline.Pipeline] cloned into each
        neighbourhood. Required. Whether it is a classifier or a regressor is detected
        with [`is_classifier`][sklearn.base.is_classifier], and that choice drives both
        the default cross-validator and the default scorer.
    radius : float, default: 1.0
        Neighbourhood radius, in the units of the mask's spatial coordinates. Check
        `mask[dim].attrs.get("units")` if unsure. Radii are measured in physical
        coordinates rather than voxel indices, so anisotropic voxels behave correctly.
    process_mask : xarray.DataArray, optional
        Boolean mask selecting the voxels that act as neighbourhood *centres*. Must be
        a subset of `mask`. If not provided, a score is computed at every `mask` voxel.
        Use it to restrict the searchlight to a region of interest while still drawing
        features from the surrounding tissue.
    cv : int or sklearn.model_selection.BaseCrossValidator, default: 5
        Cross-validation strategy. An integer builds a
        [`StratifiedKFold`][sklearn.model_selection.StratifiedKFold] or
        [`KFold`][sklearn.model_selection.KFold] with `shuffle=False`, so folds are
        contiguous blocks of time. Any scikit-learn splitter is accepted.
    scoring : str or callable, optional
        Scorer passed to
        [`cross_val_score`][sklearn.model_selection.cross_val_score]. If not provided,
        the estimator's own `score` is used, which is accuracy for classifiers and the
        coefficient of determination for regressors.
    n_jobs : int, default: 1
        Number of joblib workers. Centres are dispatched in batches, not one task
        each.

    Attributes
    ----------
    scores_ : (...) xarray.DataArray
        Mean cross-validation score at each `process_mask` centre, in the spatial
        geometry of `process_mask`. Voxels outside `process_mask` are `numpy.nan`.

    Warns
    -----
    UserWarning
        If `radius` is small enough that the median neighbourhood holds a single voxel.
        The run still produces a valid map, but it has silently become a univariate
        analysis rather than a multivariate one.

    Notes
    -----
    Consecutive fUSI volumes are strongly autocorrelated. Passing a splitter with
    `shuffle=True` places near-duplicate neighbouring volumes in both the training and
    test sets, which inflates scores. This is why an integer `cv` builds contiguous
    folds. For data with a run or block structure, prefer
    [`LeaveOneGroupOut`][sklearn.model_selection.LeaveOneGroupOut] with `groups`.

    References
    ----------
    [^1]:
        Kriegeskorte, N., Goebel, R., and Bandettini, P. (2006). "Information-based
        functional brain mapping". PNAS, 103(10), 3863-3868.

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> from sklearn.linear_model import Ridge
    >>> from confusius.decoding import SearchLight
    >>>
    >>> rng = np.random.default_rng(0)
    >>> data = xr.DataArray(
    ...     rng.standard_normal((40, 1, 5, 5)),
    ...     dims=["time", "z", "y", "x"],
    ...     coords={
    ...         "time": np.arange(40) * 0.5,
    ...         "z": [0.0],
    ...         "y": np.arange(5) * 0.2,
    ...         "x": np.arange(5) * 0.2,
    ...     },
    ... )
    >>> mask = xr.ones_like(data.isel(time=0, drop=True), dtype=bool)
    >>> speed = rng.standard_normal(40)
    >>>
    >>> searchlight = SearchLight(mask=mask, estimator=Ridge(), radius=0.25, cv=3)
    >>> searchlight.fit(data, speed).scores_.dims
    ('z', 'y', 'x')
    """

    def __init__(
        self,
        *,
        mask: xr.DataArray,
        estimator: BaseEstimator,
        radius: float = 1.0,
        process_mask: xr.DataArray | None = None,
        cv: int | BaseCrossValidator = 5,
        scoring: str | Callable | None = None,
        n_jobs: int = 1,
    ) -> None:
        self.mask = mask
        self.estimator = estimator
        self.radius = radius
        self.process_mask = process_mask
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs

    def fit(
        self,
        X: xr.DataArray,
        y: npt.ArrayLike | xr.DataArray,
        groups: npt.ArrayLike | None = None,
    ) -> "SearchLight":
        """Run searchlight decoding.

        Parameters
        ----------
        X : (time, ...) xarray.DataArray
            Functional ultrasound data. The `time` dimension is the sample axis.
        y : (n_samples,) array-like or xarray.DataArray
            Targets aligned with `X`'s `time` axis.
        groups : (n_samples,) array-like, optional
            Group labels forwarded to the cross-validator. Required by
            [`LeaveOneGroupOut`][sklearn.model_selection.LeaveOneGroupOut].

        Returns
        -------
        SearchLight
            The fitted estimator.

        Raises
        ------
        ValueError
            If `X` is h5py-backed, if `process_mask` is not a subset of `mask`, if `y`
            or `groups` do not align with `X`, or if a spatial dimension lacks a
            numeric coordinate.
        """
        if is_h5py_backed(X):
            raise ValueError(
                "SearchLight cannot run on h5py-backed data, because joblib workers "
                "cannot pickle h5py datasets. Call `.compute()` on the data first."
            )

        validate_time_series(X, operation_name="SearchLight.fit")

        spatial_dims = tuple(str(dim) for dim in X.dims if dim != "time")
        X_ordered = X.transpose("time", *spatial_dims)

        mask = validate_mask(self.mask, X_ordered, "mask", require_exact_dims=True)
        if self.process_mask is None:
            process_mask = mask
        else:
            process_mask = validate_mask(
                self.process_mask, X_ordered, "process_mask", require_exact_dims=True
            )
            outside = int(
                np.count_nonzero(
                    np.asarray(process_mask.values) & ~np.asarray(mask.values)
                )
            )
            if outside:
                raise ValueError(
                    f"process_mask must be a subset of mask, but {outside} of its "
                    "voxels fall outside mask."
                )

        y_array = _check_targets(y, X_ordered)
        groups_array = None
        if groups is not None:
            groups_array = np.asarray(groups)
            if groups_array.shape[0] != y_array.shape[0]:
                raise ValueError(
                    f"groups has {groups_array.shape[0]} entries but y has "
                    f"{y_array.shape[0]} samples."
                )

        features = np.asarray(extract_with_mask(X_ordered, mask).values)
        neighborhoods = _neighborhood_indices(mask, process_mask, self.radius)

        sizes = np.array([len(indices) for indices in neighborhoods])
        if sizes.size and float(np.median(sizes)) <= 1.0:
            warnings.warn(
                f"radius={self.radius} produces single-voxel searchlight "
                f"neighbourhoods (median size {float(np.median(sizes)):.0f}). The "
                "result is a univariate analysis rather than a multivariate one. "
                "Increase `radius` past the voxel spacing.",
                UserWarning,
                stacklevel=find_stack_level(),
            )

        cv = _resolve_cv(self.cv, classifier=is_classifier(self.estimator))

        scores = np.asarray(
            _score_batch(
                self.estimator,
                features,
                y_array,
                neighborhoods,
                cv,
                self.scoring,
                groups_array,
            ),
            dtype=np.float64,
        )

        self.scores_: xr.DataArray = unmask(
            scores,
            process_mask,
            attrs={"long_name": "Searchlight CV score"},
            fill_value=np.nan,
        )
        return self

    def __sklearn_is_fitted__(self) -> bool:
        """Whether `fit` has completed.

        Returns
        -------
        bool
            Whether `scores_` has been assigned.
        """
        return hasattr(self, "scores_")

    def score(self, X: xr.DataArray, y: npt.ArrayLike) -> float:
        """Refuse to produce a single score.

        Parameters
        ----------
        X : xarray.DataArray
            Ignored. SearchLight does not refit a single model.
        y : array-like
            Ignored. SearchLight does not refit a single model.

        Returns
        -------
        float
            Never returns.

        Raises
        ------
        NotImplementedError
            Always. A searchlight produces one score per voxel, not one per model, so
            there is no single number to report. Read `scores_` instead.
        """
        raise NotImplementedError(
            "SearchLight has no single score. Each voxel has its own "
            "cross-validation score in `scores_`."
        )

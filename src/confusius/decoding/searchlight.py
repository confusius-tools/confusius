"""Searchlight decoding for VoxelData arrays.

Portions of this module are inspired by `nilearn.decoding.searchlight`, which is
licensed under the BSD-3-Clause License. See `NOTICE` for details.
"""

import warnings
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import xarray as xr
from joblib import Parallel, delayed, effective_n_jobs
from scipy.spatial import KDTree
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.model_selection import (
    BaseCrossValidator,
    KFold,
    StratifiedKFold,
    cross_val_score,
)

from confusius._dims import VOXEL_DIMS, WORLD_DIMS
from confusius._utils.io import is_h5py_backed
from confusius._utils.progress import progress_bar
from confusius._utils.stack import find_stack_level
from confusius.extract import extract_with_mask, unmask
from confusius.validation import ensure_mask, ensure_voxeldata


def _get_masked_coordinates(mask: xr.DataArray) -> npt.NDArray[np.float64]:
    """World coordinates of the `True` voxels of a mask.

    Parameters
    ----------
    mask : xarray.DataArray
        Boolean spatial mask carrying a `VoxelToWorldIndex` over its native voxel
        (`k`/`j`/`i`) dimensions.

    Returns
    -------
    numpy.ndarray
        `(n_masked, n_dims)` array of world coordinates, in
        `confusius._dims.WORLD_DIMS` order.

    Notes
    -----
    SearchLight measures `radius` in world coordinate units, so this relies on
    `mask` carrying a real `VoxelToWorldIndex` (numeric `z`/`y`/`x` by
    construction) -- callers must validate that themselves (e.g. via
    `ensure_mask`) before reaching this function; a mask without a real index
    would otherwise silently make `radius` mean voxel indices instead, which is
    anisotropic.
    """
    dims = tuple(str(dim) for dim in mask.dims)
    coord_arrays = [
        np.broadcast_to(
            np.asarray(mask.coords[name].transpose(*dims).values, dtype=np.float64),
            mask.shape,
        )
        for name in WORLD_DIMS
    ]

    flat = np.stack([arr.ravel() for arr in coord_arrays], axis=-1)
    return flat[np.asarray(mask.values).ravel()]


def _find_neighborhood_indices(
    mask: xr.DataArray, process_mask: xr.DataArray, radius: float
) -> list[npt.NDArray[np.intp]]:
    """Feature indices falling within `radius` of every center voxel.

    Parameters
    ----------
    mask : xarray.DataArray
        Boolean mask defining which voxels may act as features.
    process_mask : xarray.DataArray
        Boolean mask defining which voxels act as neighborhood centers.
    radius : float
        Neighborhood radius, in the units of the mask coordinates.

    Returns
    -------
    list[numpy.ndarray]
        One integer index array per center, indexing into the masked feature axis.
    """
    feature_coords = _get_masked_coordinates(mask)
    center_coords = _get_masked_coordinates(process_mask)
    tree = KDTree(feature_coords)
    return [
        np.asarray(sorted(indices), dtype=np.intp)
        for indices in tree.query_ball_point(center_coords, r=radius)
    ]


def _resolve_cv(
    cv: int | BaseCrossValidator, *, classifier: bool
) -> BaseCrossValidator:
    """Turn the `cv` argument into a scikit-learn splitter.

    An integer becomes a splitter with `shuffle=False`. For a regressor that is `KFold`,
    whose folds are contiguous blocks of time; shuffling would place temporally adjacent,
    and therefore highly correlated, fUSI volumes in both the training and test sets. For
    a classifier it is `StratifiedKFold`, which keeps each fold class-balanced but
    interleaves time rather than cutting contiguous blocks, so autocorrelated volumes can
    still straddle the split.

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


def _get_score_units(scoring: str | Callable | None, *, classifier: bool) -> str | None:
    """Name of the metric a searchlight score is measured in, from the scorer.

    Parameters
    ----------
    scoring : str, callable, or None
        The `scoring` passed to
        [`cross_val_score`][sklearn.model_selection.cross_val_score]. A string names the
        metric directly. If not provided, the estimator's own `score` is used, which is
        accuracy for classifiers and the coefficient of determination for regressors.
    classifier : bool
        Whether the estimator is a classifier, which selects the default metric when
        `scoring` is not provided.

    Returns
    -------
    str or None
        The metric name to store as the `units` attribute of `scores_`, or `None` when
        it cannot be determined, as for a callable scorer.
    """
    if scoring is None:
        return "accuracy" if classifier else "R²"
    if isinstance(scoring, str):
        return scoring
    return None


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
        if (
            "time" in y.coords
            and "time" in data.coords
            and not np.array_equal(
                np.asarray(y.coords["time"].values),
                np.asarray(data.coords["time"].values),
            )
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
    """Mean cross-validation score for a batch of neighborhoods.

    The estimator is cloned per neighborhood so that each fit is independent and the
    joblib worker holds no shared state.

    Parameters
    ----------
    estimator : sklearn.base.BaseEstimator
        Estimator to clone into each neighborhood.
    features : numpy.ndarray
        `(n_samples, n_features)` masked data.
    y : numpy.ndarray
        `(n_samples,)` targets.
    neighborhoods : list[numpy.ndarray]
        Feature index arrays, one per center in this batch.
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
        One mean score per neighborhood, in input order.
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


def _run_searchlight(
    estimator: BaseEstimator,
    features: npt.NDArray[np.float64],
    y: npt.NDArray,
    neighborhoods: list[npt.NDArray[np.intp]],
    cv: BaseCrossValidator,
    scoring: str | Callable | None,
    groups: npt.NDArray | None,
    n_jobs: int,
    show_progress: bool,
) -> npt.NDArray[np.float64]:
    """Score every neighborhood, parallelizing over batches of centers.

    Centers are split into contiguous batches rather than one joblib task each, because
    a whole-brain searchlight has far more centers than the dispatch overhead can
    absorb.

    Parameters
    ----------
    estimator : sklearn.base.BaseEstimator
        Estimator cloned into each neighborhood.
    features : numpy.ndarray
        `(n_samples, n_features)` masked data.
    y : numpy.ndarray
        `(n_samples,)` targets.
    neighborhoods : list[numpy.ndarray]
        Feature index arrays, one per center.
    cv : sklearn.model_selection.BaseCrossValidator
        Cross-validation splitter.
    scoring : str, callable, or None
        Scorer passed to
        [`cross_val_score`][sklearn.model_selection.cross_val_score].
    groups : numpy.ndarray, optional
        Group labels forwarded to the splitter.
    n_jobs : int
        Number of joblib workers.
    show_progress : bool
        Whether to display a progress bar counting completed center voxels.

    Returns
    -------
    numpy.ndarray
        `(n_centers,)` array of mean scores, in center order.
    """
    # One batch per worker would minimise dispatch overhead but leaves the progress bar
    # with only `n_jobs` steps, which tells the user nothing during a run that takes
    # minutes. Several batches per worker cost a handful of extra dispatches, negligible
    # next to a batch of cross-validations, and also balance the load better.
    n_batches = max(1, min(len(neighborhoods), effective_n_jobs(n_jobs) * 8))
    batches = [
        [neighborhoods[index] for index in batch_indices]
        for batch_indices in np.array_split(np.arange(len(neighborhoods)), n_batches)
    ]

    results: list[npt.NDArray[np.float64]] = []
    # The bar is advanced by hand rather than through `joblib_progress`, which patches
    # `Parallel.print_progress` process-wide. The inner `cross_val_score` builds its own
    # `Parallel`, so its folds would advance this bar too.
    with progress_bar(
        "Scoring searchlights...", total=len(neighborhoods), show=show_progress
    ) as advance:
        # `return_as="generator"` keeps results in batch order, so concatenating them
        # still maps scores back onto centers. The unordered variant would scramble the
        # map.
        stream = Parallel(n_jobs=n_jobs, return_as="generator")(
            delayed(_score_batch)(estimator, features, y, batch, cv, scoring, groups)
            for batch in batches
        )
        for batch, result in zip(batches, stream, strict=True):
            results.append(np.asarray(result, dtype=np.float64))
            advance(len(batch))

    return np.concatenate(results)


class SearchLight(BaseEstimator):
    """Searchlight decoder for fUSI data.

    For every voxel of `process_mask`, `fit` gathers the `mask` voxels lying within
    `radius`, cross-validates `estimator` on that neighborhood, and stores the mean
    score.

    This estimator wraps scikit-learn while keeping xarray metadata:

    - Input data must be a VoxelData array with dims `(time, k, j, i)`—no `pose` or
      extra dimensions.
    - The `time` dimension is the sample axis. It need not be temporally ordered. For
      trial-averaged data, rename the trial dimension with `.rename(trial="time")`.
    - `scores_` is returned in the spatial geometry of `process_mask`.

    Parameters
    ----------
    estimator : sklearn.base.BaseEstimator
        Estimator or [`Pipeline`][sklearn.pipeline.Pipeline].
    mask : xarray.DataArray, optional
        Boolean spatial mask selecting the voxels that may act as *features*. If not
        provided, every voxel of the input data is used as a feature voxel. Must be a
        VoxelData array on `X`'s grid, because `radius` is measured in world
        coordinates.
    radius : float, default: 1.0
        Neighborhood radius, in the units of the data's spatial coordinates. Check
        `X.coords[dim].attrs.get("units")` if unsure. Radii are measured in world
        coordinates rather than voxel indices, so anisotropic voxels behave correctly.
    process_mask : xarray.DataArray, optional
        Boolean mask selecting the voxels that act as neighborhood *centers*. Must be
        a subset of `mask`, and a VoxelData array on `X`'s grid. If not provided, a
        score is computed at every `mask` voxel. Use it to restrict the searchlight to
        a region of interest while still drawing features from the surrounding
        tissue.
    cv : int or sklearn.model_selection.BaseCrossValidator, default: 5
        Cross-validation strategy. An integer builds a
        [`KFold`][sklearn.model_selection.KFold] for regressors, whose folds are
        contiguous blocks of time, or a
        [`StratifiedKFold`][sklearn.model_selection.StratifiedKFold] for classifiers,
        which keeps folds class-balanced but interleaves time. Both use `shuffle=False`.
        Any scikit-learn splitter is accepted.
    scoring : str or callable, optional
        Scorer passed to
        [`cross_val_score`][sklearn.model_selection.cross_val_score]. If not provided,
        the estimator's own `score` is used, which is accuracy for classifiers and the
        coefficient of determination for regressors.
    n_jobs : int, default: 1
        Number of joblib workers. Centers are dispatched in batches, not one task
        each.
    show_progress : bool, default: True
        Whether to display a progress bar during fit.

    Attributes
    ----------
    scores_ : (k, j, i) xarray.DataArray
        Mean cross-validation score at each `process_mask` center, in the spatial
        geometry of `process_mask`. Voxels outside `process_mask` are `numpy.nan`. The
        input's attributes are carried over, including any `affines`, so the map stays in
        the same world space. `long_name` is set to `"Searchlight CV score"` and
        `units` to the metric implied by `scoring`: the scorer string when one is given,
        otherwise `"accuracy"` for a classifier or `"R²"` for a regressor. A callable
        scorer leaves `units` unset.

    Warns
    -----
    UserWarning
        If `radius` is small enough that the median neighborhood holds a single voxel.
        The run still produces a valid map, but it has silently become a univariate
        analysis rather than a multivariate one. Also if `process_mask` selects no
        voxels at all, which yields an entirely NaN map.

    Notes
    -----
    Consecutive fUSI volumes are strongly autocorrelated. Passing a splitter with
    `shuffle=True` places near-duplicate neighboring volumes in both the training and
    test sets, which inflates scores. This is why an integer `cv` builds `KFold` folds
    with `shuffle=False` for regressors. Classifiers instead get `StratifiedKFold`, which
    balances classes across folds but interleaves time rather than keeping contiguous
    blocks, so on its own it does not prevent this leakage. For data with a run or block
    structure, or to keep classification folds contiguous, prefer
    [`LeaveOneGroupOut`][sklearn.model_selection.LeaveOneGroupOut] with `groups`.

    References
    ----------
    [^1]:
        Kriegeskorte, N., Goebel, R., and Bandettini, P. (2006). "Information-based
        functional brain mapping". PNAS, 103(10), 3863-3868.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import Ridge
    >>> from confusius.decoding import SearchLight
    >>> from confusius.xarray import create_voxeldata
    >>>
    >>> rng = np.random.default_rng(0)
    >>> data = create_voxeldata(
    ...     rng.standard_normal((40, 1, 5, 5)),
    ...     dims=("time", "k", "j", "i"),
    ...     dt=0.5,
    ...     spacing=(1.0, 0.2, 0.2),
    ... )
    >>> speed = rng.standard_normal(40)
    >>>
    >>> searchlight = SearchLight(
    ...     estimator=Ridge(), radius=0.25, cv=3, show_progress=False
    ... )
    >>> searchlight.fit(data, speed).scores_.dims
    ('k', 'j', 'i')
    """

    def __init__(
        self,
        *,
        estimator: BaseEstimator,
        mask: xr.DataArray | None = None,
        radius: float = 1.0,
        process_mask: xr.DataArray | None = None,
        cv: int | BaseCrossValidator = 5,
        scoring: str | Callable | None = None,
        n_jobs: int = 1,
        show_progress: bool = True,
    ) -> None:
        self.estimator = estimator
        self.mask = mask
        self.radius = radius
        self.process_mask = process_mask
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.show_progress = show_progress

    def fit(
        self,
        X: xr.DataArray,
        y: npt.ArrayLike | xr.DataArray,
        groups: npt.ArrayLike | None = None,
    ) -> "SearchLight":
        """Run searchlight decoding.

        Parameters
        ----------
        X : (time, k, j, i) xarray.DataArray
            VoxelData array. The `time` dimension is the sample axis.
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
            If `X` is h5py-backed or is not a VoxelData array with exactly dims
            `(time, k, j, i)`; if `radius` is negative; if `mask` or `process_mask` is
            not a VoxelData array; if `process_mask` is not a subset of `mask`; if `y`
            or `groups` do not align with `X`; or if the masked data contains
            non-finite values.

        Warns
        -----
        UserWarning
            If the median neighborhood holds a single voxel, either because `radius`
            is below the voxel spacing or because `mask` is very sparse. The run still
            produces a valid map, but it has silently become a univariate analysis
            rather than a multivariate one. Also if `process_mask` selects no voxels at
            all, which yields an entirely NaN map.
        """
        if is_h5py_backed(X):
            raise ValueError(
                "SearchLight cannot run on h5py-backed data. Fitting materializes the "
                "whole masked time series into memory at once, which would silently "
                "pull the entire lazy recording into RAM. Call `.compute()` on the "
                "data first."
            )

        X_ordered = ensure_voxeldata(
            X,
            require_time=True,
            require_unchunked_time=True,
            allow_pose=False,
            allow_extra_dims=False,
        )

        if self.radius < 0:
            raise ValueError(f"radius must be non-negative, got {self.radius}.")

        X_ordered = X_ordered.transpose("time", *VOXEL_DIMS)

        if self.mask is None:
            mask = xr.ones_like(X_ordered.isel(time=0, drop=True), dtype=bool)
        else:
            mask = ensure_mask(self.mask, X_ordered, "mask", require_exact_dims=True)

        if self.process_mask is None:
            process_mask = mask
        else:
            process_mask = ensure_mask(
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
        if not np.isfinite(features).all():
            raise ValueError(
                "The masked data contains non-finite values (NaN or inf), so every "
                "cross-validation fit over a neighborhood that touches one fails and "
                "`scores_` would silently fill with NaN. This commonly comes from "
                "z-scoring background or zero-variance voxels. Restrict `mask` to the "
                "finite voxels, for example `mask=X.notnull().all('time')`."
            )
        neighborhoods = _find_neighborhood_indices(mask, process_mask, self.radius)

        sizes = np.array([len(indices) for indices in neighborhoods])
        median_size = float(np.median(sizes)) if sizes.size else 0.0
        if not sizes.size:
            warnings.warn(
                "process_mask selects no voxels, so the score map is entirely NaN. "
                "Check that `process_mask` overlaps `mask` and is not all `False`.",
                UserWarning,
                stacklevel=find_stack_level(),
            )
        elif median_size <= 1.0:
            warnings.warn(
                f"radius={self.radius} produces single-voxel searchlight "
                f"neighborhoods (median size {median_size:.0f}). The result is a "
                "univariate analysis rather than a multivariate one. Increase "
                "`radius` past the voxel spacing, or check that `mask` is not too "
                "sparse.",
                UserWarning,
                stacklevel=find_stack_level(),
            )

        classifier = is_classifier(self.estimator)
        cv = _resolve_cv(self.cv, classifier=classifier)

        scores = _run_searchlight(
            self.estimator,
            features,
            y_array,
            neighborhoods,
            cv,
            self.scoring,
            groups_array,
            self.n_jobs,
            self.show_progress,
        )

        # Carry the input's attributes over, notably any `affines`, so the score map
        # stays in the same world space. Override the semantic attributes to describe
        # the score itself rather than the input signal.
        attrs = {**dict(X_ordered.attrs), "long_name": "Searchlight CV score"}
        units = _get_score_units(self.scoring, classifier=classifier)
        if units is not None:
            attrs["units"] = units
        else:
            attrs.pop("units", None)

        self.scores_: xr.DataArray = unmask(
            scores,
            process_mask,
            attrs=attrs,
            fill_value=np.nan,
        )
        return self

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

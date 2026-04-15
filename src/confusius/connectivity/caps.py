"""Co-activation patterns (CAPs) analysis for fUSI data."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

from confusius._utils import find_stack_level

if TYPE_CHECKING:
    from rich.progress import Progress

import numpy as np
import numpy.typing as npt
import xarray as xr
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from confusius.signal import clean
from confusius.validation import validate_time_series

_ALLOWED_METRICS = ("correlation", "cosine", "euclidean")
_ALLOWED_UPDATE_RULES = ("mean", "weighted")
_ALLOWED_SELECTION_METHODS = ("elbow", "silhouette", "davies_bouldin", "variance_ratio")


def _resolve_n_init(n_init: int | Literal["auto"]) -> int:
    """Resolve sklearn-style `n_init` for k-means++ initialization.

    Parameters
    ----------
    n_init : int or {"auto"}
        Number of initializations or sklearn-style automatic choice.

    Returns
    -------
    int
        Effective number of restarts for k-means++.

    Raises
    ------
    ValueError
        If `n_init` is neither a positive integer nor `"auto"`.
    """
    if n_init == "auto":
        # Match sklearn's behavior for k-means++ initialization.
        return 1
    if (
        isinstance(n_init, (int, np.integer))
        and not isinstance(n_init, bool)
        and n_init > 0
    ):
        return n_init
    raise ValueError(f"n_init must be a positive int or 'auto', got {n_init!r}.")


def _cosine_kmeans_init(
    X: npt.NDArray[np.floating],
    n_clusters: int,
    n_local_trials: int | None,
    rng: np.random.Generator,
) -> npt.NDArray[np.floating]:
    """K-means++ seeding with cosine distance for unit-norm data.

    Implements the k-means++ seeding strategy with cosine distance (`1 - dot product`)
    instead of squared Euclidean distance. At each step, `n_local_trials` candidates are
    sampled with probability proportional to the cosine distance to the nearest existing
    center, and the one that minimizes the total potential is kept greedily.

    Parameters
    ----------
    X : (time, space) numpy.ndarray
        Unit-norm input data (each row has L2 norm ≈ 1).
    n_clusters : int
        Number of centers to initialize.
    n_local_trials : int or None
        Number of candidate centers evaluated greedily at each seeding step. If `None`,
        uses `2 + int(np.log(n_clusters))`, matching sklearn's default.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    (cluster, space) numpy.ndarray
        Initial cluster centers (rows of `X`).

    References
    ----------
    [^1]:
        Arthur, D. and Vassilvitskii, S. "k-means++: the advantages of careful
        seeding." ACM-SIAM Symposium on Discrete Algorithms (SODA), 2007.
    """
    n_samples = X.shape[0]
    if n_local_trials is None:
        n_local_trials = 2 + int(np.log(n_clusters))
    n_local_trials = min(n_samples - 1, n_local_trials)

    center_indices = [int(rng.integers(n_samples))]
    # Cosine distance to the first center. For unit-norm X: d(x, c) = 1 - x·c.
    nearest_distance = np.maximum(1.0 - X @ X[center_indices[0]], 0.0)

    for _ in range(n_clusters - 1):
        pot = float(nearest_distance.sum())
        if pot == 0.0:
            # All remaining points are coincident with an existing center.
            center_indices.append(int(rng.integers(n_samples)))
            nearest_distance[:] = 0.0
            continue

        rand_vals = rng.random(n_local_trials) * pot
        candidate_ids = np.searchsorted(
            np.cumsum(nearest_distance.astype(np.float64)), rand_vals
        )
        np.clip(candidate_ids, 0, n_samples - 1, out=candidate_ids)

        # Distance from every sample to each candidate.
        distance_to_candidates = np.maximum(1.0 - X @ X[candidate_ids].T, 0.0)
        # Updated nearest-distance if this candidate were added.
        new_nearest_distance = np.minimum(
            nearest_distance[:, np.newaxis], distance_to_candidates
        )
        best_index = int(np.argmin(new_nearest_distance.sum(axis=0)))

        center_indices.append(int(candidate_ids[best_index]))
        nearest_distance = new_nearest_distance[:, best_index]

    return X[center_indices]


def _run_single_cosine_kmeans(
    X: npt.NDArray[np.floating],
    n_clusters: int,
    max_iter: int,
    n_local_trials: int | None,
    update_rule: Literal["mean", "weighted"],
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.intp], float]:
    """Single run of cosine k-means. Returns centers, labels, and cosine inertia.

    Cosine inertia is the total cosine distance from each volume to its assigned center
    (lower is better): `n_samples - sum(max similarities)`.

    Parameters
    ----------
    X : (time, space) numpy.ndarray
        Unit-norm input data.
    n_clusters : int
        Number of clusters.
    max_iter : int
        Maximum number of iterations.
    n_local_trials : int or None
        Number of local trials in k-means++ initialization. If `None`, uses
        `2 + int(np.log(n_clusters))`.
    update_rule : {"mean", "weighted"}
        Center update rule.
    rng : numpy.random.Generator
        Random number generator (state advanced in place).

    Returns
    -------
    centers : (cluster, space) numpy.ndarray
        Unit-norm cluster centers. Zero-norm rows indicate empty clusters.
    labels : (n_samples,) numpy.ndarray
        Cluster index for each sample.
    inertia : float
        Total cosine distance from each volume to its assigned center.
    """
    n_samples, _ = X.shape

    centers = _cosine_kmeans_init(X, n_clusters, n_local_trials, rng)
    norms = np.linalg.norm(centers, axis=1, keepdims=True)
    centers = centers / np.where(norms == 0.0, 1.0, norms)

    # Initialize to -1 so the first iteration always triggers an update.
    labels = np.full(n_samples, -1, dtype=np.intp)

    for _ in range(max_iter):
        similarities = X @ centers.T  # (n_samples, n_clusters)

        # Hard assignment: nearest center = max cosine similarity.
        new_labels = similarities.argmax(axis=1).astype(np.intp)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        if update_rule == "weighted":
            # Similarity-weighted update: weight each volume by its cosine
            # similarity to its assigned center. Entries within 1e-6 of the
            # row maximum are kept to handle near-ties gracefully.
            weights = similarities * (
                similarities >= similarities.max(axis=1, keepdims=True) - 1e-6
            )
            new_centers = weights.T @ X
        else:
            # Standard k-means: unweighted mean of assigned volumes.
            # One-hot assignment matrix keeps the update as a single BLAS
            # matrix multiply instead of a Python-level scatter.
            assignment = np.eye(n_clusters, dtype=X.dtype)[labels]
            new_centers = assignment.T @ X
            counts = assignment.sum(axis=0)
            nonempty = counts > 0.0
            new_centers[nonempty] /= counts[nonempty, np.newaxis]

        norms = np.linalg.norm(new_centers, axis=1, keepdims=True)
        centers = new_centers / np.where(norms == 0.0, 1.0, norms)
        # Zero out empty cluster centers so they never win argmax.
        centers[norms.ravel() == 0.0] = 0.0

    # Final assignment and inertia.
    similarities = X @ centers.T
    labels = similarities.argmax(axis=1).astype(np.intp)
    inertia = float(n_samples - similarities.max(axis=1).sum())
    return centers, labels, inertia


def _run_multi_cosine_kmeans(
    X: npt.NDArray[np.floating],
    n_clusters: int,
    max_iter: int,
    n_local_trials: int | None,
    update_rule: Literal["mean", "weighted"],
    n_init: int,
    random_state: int | None,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.intp]]:
    """K-means clustering with cosine distance on unit-norm data.

    Runs `n_init` independent k-means++ initializations sequentially and returns the
    result with the lowest cosine inertia (total cosine distance from each volume to its
    assigned center).

    Parameters
    ----------
    X : (time, space) numpy.ndarray
        Unit-norm input data (each row has L2 norm ≈ 1).
    n_clusters : int
        Number of clusters.
    max_iter : int
        Maximum number of assignment–update iterations per run.
    n_local_trials : int or None
        Number of local trials per step in k-means++ initialization. If
        `None`, uses `2 + int(np.log(n_clusters))`.
    update_rule : {"mean", "weighted"}
        Center update rule.
    n_init : int
        Number of independent random initializations. The run with the lowest
        cosine inertia is returned.
    random_state : int or None
        Seed for the random number generator.

    Returns
    -------
    centers : (cap, space) numpy.ndarray
        Unit-norm cluster centers from the best run. `n_caps` may be less than
        `n_clusters` if some clusters are empty after convergence.
    labels : (n_samples,) numpy.ndarray
        Cluster index for each sample.

    References
    ----------
    [^1]:
        Arthur, D. and Vassilvitskii, S. "k-means++: the advantages of careful
        seeding." ACM-SIAM Symposium on Discrete Algorithms (SODA), 2007.
    """
    # Generate per-restart seeds from the master seed for reproducibility.
    seeds = np.random.default_rng(random_state).integers(
        0, np.iinfo(np.int64).max, size=n_init
    )

    best_centers: npt.NDArray[np.floating] | None = None
    best_labels: npt.NDArray[np.intp] | None = None
    best_inertia = np.inf

    for seed in seeds:
        centers, labels, inertia = _run_single_cosine_kmeans(
            X,
            n_clusters,
            max_iter,
            n_local_trials,
            update_rule,
            np.random.default_rng(int(seed)),
        )
        if inertia < best_inertia:
            best_centers = centers
            best_labels = labels
            best_inertia = inertia

    assert best_centers is not None and best_labels is not None

    valid = np.linalg.norm(best_centers, axis=1) > 0.0
    if not valid.all():
        n_empty = int((~valid).sum())
        warnings.warn(
            f"{n_empty} empty cluster(s) removed after k-means convergence. "
            f"'caps_' will have {int(valid.sum())} CAPs instead of "
            f"{n_clusters}. Consider reducing 'n_clusters'.",
            stacklevel=find_stack_level(),
        )
        best_centers = best_centers[valid]
        best_labels = (X @ best_centers.T).argmax(axis=1).astype(np.intp)

    return best_centers, best_labels


def _find_elbow(cluster_range: list[int], scores: list[float]) -> int:
    """Find the elbow of a score curve using maximum perpendicular distance.

    Normalizes the curve to the unit square and returns the index of the point
    furthest from the line connecting the first and last points (kneedle
    approach).

    Parameters
    ----------
    cluster_range : list[int]
        Cluster counts evaluated.
    scores : list[float]
        Score for each cluster count (lower-is-better curves such as inertia
        work directly; higher-is-better curves should be negated before
        calling).

    Returns
    -------
    int
        Cluster count at the elbow.
    """
    n = len(scores)
    if n == 1:
        return cluster_range[0]

    x = np.linspace(0.0, 1.0, n)
    y = np.asarray(scores, dtype=float)
    y_range = y.max() - y.min()
    y = (y - y.min()) / (y_range if y_range > 0.0 else 1.0)

    # Unit vector along the diagonal from (x[0], y[0]) to (x[-1], y[-1]).
    d = np.array([x[-1] - x[0], y[-1] - y[0]])
    d /= np.linalg.norm(d)
    # Perpendicular distance from each point to that diagonal.
    vecs = np.column_stack([x - x[0], y - y[0]])
    distances = np.abs(vecs @ np.array([-d[1], d[0]]))

    return cluster_range[int(np.argmax(distances))]


class CoactivationPatterns(BaseEstimator):
    """Estimate co-activation patterns (CAPs) by clustering volumes.

    CAP analysis consists in clustering all volumes in one or more recording using
    *k*-means. Note that classical k-means minimizes within-cluster deviations from
    cluster centers, which amounts to minimizing squared Euclidean distances.
    Convergence is not guaranteed when using standard *k*-means using other distances.

    To allow for other metrics, this estimator changes the geometry according to
    `metric`: Euclidean k-means for `"euclidean"`, and spherical (cosine-based) k-means
    for `"cosine"` and `"correlation"` after normalization preprocessing.

    For `"correlation"` and `"cosine"`, this estimator uses a custom Lloyd-style cosine
    k-means with k-means++ initialization. For `"euclidean"`, sklearn's
    [`KMeans`][sklearn.cluster.KMeans] is used.

    !!! warning "Preprocessing matters"
        Strong global structure (e.g., unfiltered fUSI) can produce very similar CAPs
        across clusters. Temporally standardizing each voxel via `clean_kwargs` is often
        helpful (e.g., `clean_kwargs={"standardize_method": "zscore"}`).

    Parameters
    ----------
    n_clusters : int, default: 10
        Number of CAPs to extract.
    metric : {"correlation", "cosine", "euclidean"}, default: "correlation"
        Clustering geometry:

        - `"correlation"`: center each volume (subtract spatial mean), then L2-normalize
          and cluster with cosine k-means. Equivalent to Pearson-correlation geometry
          and sign-sensitive (anti-correlated volumes are far apart).
        - `"cosine"`: L2-normalize each volume (without centering), then cluster with
          cosine k-means.
        - `"euclidean"`: cluster preprocessed volumes with Euclidean k-means (sklearn
          [`KMeans`][sklearn.cluster.KMeans].

    update_rule : {"weighted", "mean"}, default: "weighted"
        Center update rule for cosine/correlation clustering:

        - `"mean"`: standard k-means, where centers are updated as the unweighted mean
          of assigned volumes then L2-normalized.
        - `"weighted"`: centers updated as the cosine-similarity-weighted mean
          of assigned volumes, giving more influence to volumes strongly matching
          the current center.

    max_iter : int, default: 300
        Maximum assignment-update iterations per run. Stops early if labels
        no longer change.
    n_local_trials : int or None, default: None
        Number of candidate centers evaluated greedily at each k-means++
        seeding step. If `None`, uses `2 + int(np.log(n_clusters))`,
        matching sklearn's default. Only used when `metric` is `"correlation"`
        or `"cosine"`.
    n_init : int or {"auto"}, default: "auto"
        Number of independent random initializations. If `"auto"`, this
        follows sklearn's k-means++ behavior and runs a single initialization.
        Applies to all metrics.
    random_state : int or None, default: 0
        Seed for the random number generator.
    clean_kwargs : dict, optional
        Keyword arguments forwarded to [`clean`][confusius.signal.clean].
        Cleaning is applied to the full data array before clustering. If not
        provided, no cleaning is applied.

        !!! warning "Chunking along time"
            Any operation in `clean_kwargs` that involves detrending or
            filtering requires the `time` dimension to be un-chunked.
            Rechunk your data before calling `fit`: `data.chunk({'time': -1})`.

    Attributes
    ----------
    caps_ : (cap, ...) xarray.DataArray
        CAP spatial maps, one per cluster. `cap` is the leading dimension; the remaining
        dimensions match the spatial dimensions of the data passed to
        [`fit`][confusius.connectivity.CoactivationPatterns.fit]. For `"correlation"`
        and `"cosine"` metrics, maps are unit-norm vectors in the preprocessed space.
        `attrs["long_name"]` is set to `"Co-activation patterns"` and `attrs["cmap"]` to
        `"coolwarm"` so that plotting functions pick up sensible defaults automatically.
    labels_ : (time,) xarray.DataArray
        CAP index assigned to each volume (0-based integer). Time coordinates are
        preserved from the input data when present.

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> from confusius.connectivity import CoactivationPatterns
    >>>
    >>> rng = np.random.default_rng(0)
    >>> data = xr.DataArray(
    ...     rng.standard_normal((200, 10, 20)),
    ...     dims=["time", "y", "x"],
    ... )
    >>>
    >>> caps = CoactivationPatterns(n_clusters=5, random_state=0)
    >>> caps.fit(data)
    CoactivationPatterns(n_clusters=5, random_state=0)
    >>> caps.caps_.dims
    ('cap', 'y', 'x')
    >>> caps.caps_.sizes["cap"]
    5
    >>> caps.labels_.dims
    ('time',)
    >>> caps.labels_.sizes["time"]
    200

    References
    ----------
    [^1]:
        Arthur, D. and Vassilvitskii, S. "k-means++: the advantages of careful
        seeding." ACM-SIAM Symposium on Discrete Algorithms (SODA), 2007.
    """

    def __init__(
        self,
        *,
        n_clusters: int = 10,
        metric: Literal["correlation", "cosine", "euclidean"] = "correlation",
        update_rule: Literal["mean", "weighted"] = "mean",
        max_iter: int = 300,
        n_local_trials: int | None = None,
        n_init: int | Literal["auto"] = "auto",
        random_state: int | None = 0,
        clean_kwargs: dict | None = None,
    ) -> None:
        self.n_clusters = n_clusters
        self.metric = metric
        self.update_rule = update_rule
        self.max_iter = max_iter
        self.n_local_trials = n_local_trials
        self.n_init = n_init
        self.random_state = random_state
        self.clean_kwargs = clean_kwargs

    def fit(self, X: xr.DataArray, y: None = None) -> "CoactivationPatterns":
        """Fit co-activation patterns by clustering volumes.

        Parameters
        ----------
        X : (time, ...) xarray.DataArray
            A fUSI DataArray to extract CAPs from. Must have a `time` dimension with at
            least 2 timepoints. All remaining dimensions are treated as spatial and
            flattened into a feature vector per volume.

            !!! warning "Chunking along time"
                The `time` dimension must NOT be chunked when `clean_kwargs`
                includes detrending or filtering steps. Rechunk first:
                `X.chunk({'time': -1})`.

        y : None, optional
            Ignored. Present for sklearn API compatibility.

        Returns
        -------
        CoactivationPatterns
            Fitted estimator.

        Raises
        ------
        ValueError
            If `metric` or `update_rule` is invalid, or if `X` has no `time`
            dimension or fewer than 2 timepoints.
        """
        if self.metric not in _ALLOWED_METRICS:
            raise ValueError(
                f"metric must be one of {_ALLOWED_METRICS}, got {self.metric!r}."
            )
        if self.update_rule not in _ALLOWED_UPDATE_RULES:
            raise ValueError(
                f"update_rule must be one of {_ALLOWED_UPDATE_RULES}, "
                f"got {self.update_rule!r}."
            )

        validate_time_series(X, operation_name="CoactivationPatterns.fit")

        X_proc, X_stacked = self._prepare_data(X)

        if self.metric in ("correlation", "cosine"):
            n_init = _resolve_n_init(self.n_init)
            centers, labels = _run_multi_cosine_kmeans(
                X_proc,
                self.n_clusters,
                self.max_iter,
                self.n_local_trials,
                self.update_rule,
                n_init,
                self.random_state,
            )
        else:
            km = KMeans(
                n_clusters=self.n_clusters,
                max_iter=self.max_iter,
                n_init=self.n_init,
                random_state=self.random_state,
            )
            km.fit(X_proc)
            centers = km.cluster_centers_
            labels = km.labels_

        n_caps = len(centers)
        caps_stacked = xr.DataArray(
            centers,
            dims=["cap", "space"],
            coords={
                "cap": np.arange(n_caps),
                "space": X_stacked.coords["space"],
            },
        )
        caps = caps_stacked.unstack("space")
        caps.attrs.update({"long_name": "CAP", "cmap": "coolwarm"})
        self.caps_: xr.DataArray = caps

        time_coords = {"time": X.coords["time"]} if "time" in X.coords else {}
        self.labels_: xr.DataArray = xr.DataArray(
            labels,
            dims=["time"],
            coords=time_coords,
        )

        return self

    def _prepare_data(
        self, X: xr.DataArray
    ) -> tuple[npt.NDArray[np.floating], xr.DataArray]:
        """Apply cleaning and metric preprocessing.

        Parameters
        ----------
        X : (time, ...) xarray.DataArray
            Input data. Must already be validated by the caller.

        Returns
        -------
        X_proc : (time, space) numpy.ndarray
            Preprocessed volumes ready for clustering.
        X_stacked : (time, space) xarray.DataArray
            Stacked version of the (optionally cleaned) input, used to recover spatial
            coordinates when building `caps_`.
        """
        if self.clean_kwargs is not None:
            X = clean(X, **self.clean_kwargs)
        spatial_dims = [str(d) for d in X.dims if d != "time"]
        X_stacked = X.stack(space=spatial_dims)
        X_proc = X_stacked.values

        if self.metric == "correlation":
            X_proc = X_proc.copy()
            X_proc -= X_proc.mean(axis=1, keepdims=True)
            norms = np.linalg.norm(X_proc, axis=1, keepdims=True)
            X_proc /= np.where(norms == 0.0, 1.0, norms)
        elif self.metric == "cosine":
            X_proc = X_proc.copy()
            norms = np.linalg.norm(X_proc, axis=1, keepdims=True)
            X_proc /= np.where(norms == 0.0, 1.0, norms)

        return X_proc, X_stacked

    def select_n_clusters(
        self,
        X: xr.DataArray,
        cluster_range: range | list[int],
        method: Literal[
            "elbow", "silhouette", "davies_bouldin", "variance_ratio"
        ] = "silhouette",
        show_progress: bool = True,
        progress: "Progress | None" = None,
    ) -> int:
        """Select the optimal number of clusters.

        Fits k-means for each value in `cluster_range` (preprocessing runs
        only once) and returns the cluster count that optimizes `method`.

        Parameters
        ----------
        X : (time, ...) xarray.DataArray
            Same data that will later be passed to
            [`fit`][confusius.connectivity.CoactivationPatterns.fit].
        cluster_range : range or list[int]
            Values of `n_clusters` to evaluate. Must contain at least 2
            entries, each ≥ 2.
        method : {"elbow", "silhouette", "davies_bouldin", "variance_ratio"}, \
                default: "silhouette"
            Selection criterion:

            - `"elbow"`: minimize cosine inertia (or euclidean inertia for
              `metric="euclidean"`); the elbow is found as the point of maximum
              perpendicular distance from the diagonal of the inertia curve.
            - `"silhouette"`: maximize the silhouette score, computed with
              cosine distance for `metric="correlation"` or `"cosine"`, and
              Euclidean distance for `metric="euclidean"`.
            - `"davies_bouldin"`: minimize the Davies-Bouldin index (Euclidean,
              applied to the preprocessed volumes).
            - `"variance_ratio"`: maximize the Calinski-Harabasz index
              (Euclidean, applied to the preprocessed volumes).

        show_progress : bool, default: True
            Whether to display a progress bar while evaluating cluster counts.
        progress : rich.progress.Progress, optional
            External `rich.progress.Progress` instance to add tasks to. If
            provided and `show_progress` is `True`, a task is added to this
            instance instead of creating a new progress bar with
            `rich.progress.track`.

        Returns
        -------
        int
            Recommended number of clusters.

        Raises
        ------
        ValueError
            If `metric`, `update_rule`, or `method` is invalid, or if
            `cluster_range` has fewer than 2 entries or any entry is < 2.
        """
        if self.metric not in _ALLOWED_METRICS:
            raise ValueError(
                f"metric must be one of {_ALLOWED_METRICS}, got {self.metric!r}."
            )
        if self.update_rule not in _ALLOWED_UPDATE_RULES:
            raise ValueError(
                f"update_rule must be one of {_ALLOWED_UPDATE_RULES}, "
                f"got {self.update_rule!r}."
            )
        if method not in _ALLOWED_SELECTION_METHODS:
            raise ValueError(
                f"method must be one of {_ALLOWED_SELECTION_METHODS}, got {method!r}."
            )

        cluster_list = list(cluster_range)
        if len(cluster_list) < 2:
            raise ValueError(
                "cluster_range must contain at least 2 values to evaluate."
            )
        if any(k < 2 for k in cluster_list):
            raise ValueError("All values in cluster_range must be >= 2.")

        validate_time_series(X, operation_name="CoactivationPatterns.select_n_clusters")
        X_proc, _ = self._prepare_data(X)

        sil_metric = (
            "cosine" if self.metric in ("correlation", "cosine") else "euclidean"
        )
        cosine_n_init = (
            _resolve_n_init(self.n_init)
            if self.metric in ("correlation", "cosine")
            else 1
        )

        scores: list[float] = []

        task_id = None
        if not show_progress:
            iterable = cluster_list
        elif progress is not None:
            task_id = progress.add_task(
                "Evaluating cluster counts...", total=len(cluster_list)
            )
            iterable = cluster_list
        else:
            from rich.progress import track

            iterable = track(
                cluster_list,
                description="Evaluating cluster counts...",
                total=len(cluster_list),
            )

        for k in iterable:
            if self.metric in ("correlation", "cosine"):
                centers, labels = _run_multi_cosine_kmeans(
                    X_proc,
                    k,
                    self.max_iter,
                    self.n_local_trials,
                    self.update_rule,
                    cosine_n_init,
                    self.random_state,
                )
                # Cosine inertia: n_samples - sum of max similarities.
                inertia = float(
                    X_proc.shape[0] - (X_proc @ centers.T).max(axis=1).sum()
                )
            else:
                km = KMeans(
                    n_clusters=k,
                    max_iter=self.max_iter,
                    n_init=self.n_init,
                    random_state=self.random_state,
                )
                km.fit(X_proc)
                labels = km.labels_
                assert km.inertia_ is not None
                inertia = float(km.inertia_)

            if method == "elbow":
                scores.append(inertia)
            elif method == "silhouette":
                scores.append(silhouette_score(X_proc, labels, metric=sil_metric))
            elif method == "davies_bouldin":
                scores.append(davies_bouldin_score(X_proc, labels))
            else:  # "variance_ratio"
                scores.append(calinski_harabasz_score(X_proc, labels))

            if task_id is not None and progress is not None:
                progress.advance(task_id)

        if method == "elbow":
            return _find_elbow(cluster_list, scores)
        if method in ("davies_bouldin",):
            return cluster_list[int(np.argmin(scores))]
        # silhouette and variance_ratio: higher is better.
        return cluster_list[int(np.argmax(scores))]

    def __sklearn_is_fitted__(self) -> bool:
        """Check whether the estimator has been fitted."""
        return hasattr(self, "caps_")

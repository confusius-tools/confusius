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
from sklearn.utils.validation import check_is_fitted

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
    # Track which cluster centers are empty (zero-norm) from the previous update.
    # An empty center has raw similarity 0 with all unit-norm inputs, which can
    # beat valid centers whose similarity is negative — hence the masking below.
    empty: npt.NDArray[np.bool_] = np.zeros(n_clusters, dtype=bool)

    for _ in range(max_iter):
        similarities = X @ centers.T  # (n_samples, n_clusters)

        # Hard assignment: nearest center = max cosine similarity.
        # Compute argmax on a masked copy so empty centers cannot win.
        if empty.any():
            masked = similarities.copy()
            masked[:, empty] = -np.inf
            new_labels = masked.argmax(axis=1).astype(np.intp)
        else:
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
        empty = norms.ravel() == 0.0
        centers[empty] = 0.0

    # Final assignment and inertia.
    similarities = X @ centers.T
    if empty.any():
        similarities[:, empty] = -np.inf
    labels = similarities.argmax(axis=1).astype(np.intp)
    inertia = float(n_samples - similarities[np.arange(n_samples), labels].sum())
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

    if best_centers is None or best_labels is None:
        raise RuntimeError(
            "Cosine k-means produced no valid solution across all restarts. "
            "The input data likely contains NaN or Inf values."
        )

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


def _segments(
    labels: npt.NDArray[np.intp], cap_id: int
) -> tuple[npt.NDArray[np.bool_], int]:
    """Count contiguous episodes of `cap_id` in `labels`.

    Parameters
    ----------
    labels : (time,) numpy.ndarray
        Sequence of CAP indices.
    cap_id : int
        Target CAP index.

    Returns
    -------
    binary : (time,) numpy.ndarray
        Boolean mask where `labels == cap_id`.
    n_segments : int
        Number of contiguous episodes. Zero when the CAP never appears.
    """
    binary = labels == cap_id
    indices = np.where(binary)[0]
    if len(indices) == 0:
        return binary, 0
    n_segments = int(np.sum(np.diff(indices) > 1)) + 1
    return binary, n_segments


def _volume_durations(lbl: xr.DataArray) -> npt.NDArray[np.floating]:
    """Compute per-volume durations from the `time` coordinate of `lbl`.

    Parameters
    ----------
    lbl : xarray.DataArray
        Per-recording label sequence with optional `time` coordinate.

    Returns
    -------
    (time,) numpy.ndarray
        Volume durations. When a `time` coordinate is present, durations are
        derived from consecutive time differences; the last volume is assigned
        the same duration as the penultimate interval. Returns all-ones (in
        volumes) when no `time` coordinate is present or when the recording
        has fewer than 2 volumes.
    """
    n = lbl.sizes["time"]
    if "time" not in lbl.coords or n < 2:
        return np.ones(n)
    times = lbl.coords["time"].values.astype(float)
    diffs = np.diff(times)
    return np.append(diffs, diffs[-1])


class CAP(BaseEstimator):
    """Co-activation pattern (CAP) analysis for fUSI data.

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
        Strong global structure can produce very similar CAPs across clusters.
        Temporally standardizing each voxel via [`clean`][confusius.signal.clean] before
        calling [`fit`][confusius.connectivity.CAP.fit] is often helpful (e.g.,
        `standardize_method="zscore"`).

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

        - `"weighted"`: centers updated as the cosine-similarity-weighted mean
          of assigned volumes, giving more influence to volumes strongly matching
          the current center.
        - `"mean"`: standard k-means, where centers are updated as the unweighted mean
          of assigned volumes then L2-normalized.

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

    Attributes
    ----------
    caps_ : (cap, ...) xarray.DataArray
        CAP spatial maps, one per cluster. `cap` is the leading dimension; the remaining
        dimensions match the spatial dimensions of the data passed to
        [`fit`][confusius.connectivity.CAP.fit]. For `"correlation"`
        and `"cosine"` metrics, maps are unit-norm vectors in the preprocessed space.
        `attrs["long_name"]` is set to `"CAP"` and `attrs["cmap"]` to
        `"coolwarm"` so that plotting functions pick up sensible defaults automatically.
    labels_ : list[xarray.DataArray]
        Per-recording CAP index sequences (0-based integer). Each element has
        `dims=["time"]` and, when present, the time coordinates of the corresponding
        input recording. The list length equals the number of recordings passed to
        [`fit`][confusius.connectivity.CAP.fit].

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> from confusius.connectivity import CAP
    >>>
    >>> rng = np.random.default_rng(0)
    >>> data = xr.DataArray(
    ...     rng.standard_normal((200, 10, 20)),
    ...     dims=["time", "y", "x"],
    ... )
    >>>
    >>> caps = CAP(n_clusters=5, random_state=0)
    >>> caps.fit([data])
    CAP(n_clusters=5, random_state=0)
    >>> caps.caps_.dims
    ('cap', 'y', 'x')
    >>> caps.caps_.sizes["cap"]
    5
    >>> len(caps.labels_)
    1
    >>> caps.labels_[0].dims
    ('time',)
    >>> caps.labels_[0].sizes["time"]
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
        update_rule: Literal["weighted", "mean"] = "weighted",
        max_iter: int = 300,
        n_local_trials: int | None = None,
        n_init: int | Literal["auto"] = "auto",
        random_state: int | None = 0,
    ) -> None:
        self.n_clusters = n_clusters
        self.metric = metric
        self.update_rule = update_rule
        self.max_iter = max_iter
        self.n_local_trials = n_local_trials
        self.n_init = n_init
        self.random_state = random_state

    def fit(self, X: list[xr.DataArray] | xr.DataArray, y: None = None) -> "CAP":
        """Fit co-activation patterns by clustering volumes across all recordings.

        Parameters
        ----------
        X : list[xarray.DataArray] or xarray.DataArray
            One or more fUSI recordings to extract CAPs from. Each DataArray must have
            a `time` dimension with at least 2 timepoints. All remaining dimensions are
            treated as spatial and flattened into a feature vector per volume. A single
            DataArray is treated as a single recording.

        y : None, optional
            Ignored. Present for sklearn API compatibility.

        Returns
        -------
        CAP
            Fitted estimator.

        Raises
        ------
        ValueError
            If `metric` or `update_rule` is invalid, if `X` is an empty list, or if
            any recording has no `time` dimension or fewer than 2 timepoints.
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

        recordings = [X] if isinstance(X, xr.DataArray) else list(X)
        if not recordings:
            raise ValueError("X must contain at least one recording.")
        for rec in recordings:
            validate_time_series(rec, operation_name="CAP.fit")

        # Stack spatial dims to (time, space) and concatenate across recordings.
        # Using np.concatenate instead of xr.concat avoids a redundant full-data
        # copy: np.concatenate always allocates fresh memory, so the subsequent
        # in-place preprocessing step requires no additional allocation.
        spatial_dims = [str(d) for d in recordings[0].dims if d != "time"]
        stacks = [rec.stack(space=spatial_dims) for rec in recordings]
        space_coords = stacks[0].coords["space"]
        X_raw = np.concatenate([s.values for s in stacks], axis=0)
        del stacks

        if np.isnan(X_raw).any():
            raise ValueError(
                "Input data contains NaN values. A common cause is z-score "
                "standardization of constant (zero-variance) voxels outside a brain "
                "mask. Fill or mask background voxels before calling fit(), e.g. "
                "`data.fillna(0)` or `data.where(mask > 0, 0)`."
            )

        X_proc = self._preprocess(X_raw, in_place=True)

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
            assert km.labels_ is not None
            labels = km.labels_

        n_caps = len(centers)
        caps_stacked = xr.DataArray(
            centers,
            dims=["cap", "space"],
            coords={
                "cap": np.arange(n_caps),
                "space": space_coords,
            },
        )
        caps = caps_stacked.unstack("space")
        caps.attrs.update({"long_name": "CAP", "cmap": "coolwarm"})
        self.caps_: xr.DataArray = caps
        self._spatial_dims: tuple[str, ...] = tuple(
            str(d) for d in caps.dims if d != "cap"
        )

        # Split the flat label array back into per-recording sequences.
        self.labels_: list[xr.DataArray] = []
        start = 0
        for rec in recordings:
            size = rec.sizes["time"]
            lbl = labels[start : start + size]
            time_coords = {"time": rec.coords["time"]} if "time" in rec.coords else {}
            self.labels_.append(xr.DataArray(lbl, dims=["time"], coords=time_coords))
            start += size

        return self

    def predict(self, X: list[xr.DataArray] | xr.DataArray) -> list[xr.DataArray]:
        """Assign recordings to CAPs using the fitted cluster centers.

        Parameters
        ----------
        X : list[xarray.DataArray] or xarray.DataArray
            One or more fUSI recordings to assign. Each must have the same spatial
            dimensions as the data passed to [`fit`][confusius.connectivity.CAP.fit].
            A single DataArray is treated as a single recording.

        Returns
        -------
        list[xarray.DataArray]
            Per-recording CAP label sequences (0-based integer), one `(time,)`
            DataArray per input recording. Time coordinates are preserved when present.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the estimator has not been fitted yet.
        ValueError
            If any recording has no `time` dimension or fewer than 2 timepoints.
        """
        check_is_fitted(self)

        recordings = [X] if isinstance(X, xr.DataArray) else list(X)

        caps_flat = self.caps_.stack(
            space=list(self._spatial_dims)
        ).values  # (cap, space)

        result = []
        for rec in recordings:
            validate_time_series(rec, operation_name="CAP.predict")
            # Ensure spatial dims match the order used during fit so that
            # the flattened feature vectors align with caps_flat.
            rec = rec.transpose("time", *self._spatial_dims)
            X_proc, _ = self._prepare_data(rec)

            if self.metric in ("correlation", "cosine"):
                labels = (X_proc @ caps_flat.T).argmax(axis=1).astype(np.intp)
            else:
                # ||x - c||² = ||x||² + ||c||² - 2x·c avoids allocating an
                # (n_samples × n_caps × n_features) tensor.
                cross = X_proc @ caps_flat.T
                X_sq = np.einsum("ij,ij->i", X_proc, X_proc)
                caps_sq = np.einsum("ij,ij->i", caps_flat, caps_flat)
                labels = (
                    (X_sq[:, np.newaxis] + caps_sq[np.newaxis, :] - 2.0 * cross)
                    .argmin(axis=1)
                    .astype(np.intp)
                )

            time_coords = {"time": rec.coords["time"]} if "time" in rec.coords else {}
            result.append(xr.DataArray(labels, dims=["time"], coords=time_coords))

        return result

    def compute_temporal_metrics(self) -> xr.Dataset:
        """Compute temporal dynamics metrics for each recording.

        Persistence is expressed in the time units of the recording when the `labels_`
        DataArrays carry a `time` coordinate; otherwise in volumes. The temporal
        resolution need not be constant: volume durations are derived from consecutive
        differences of the time coordinate, so irregular sampling is handled correctly.

        Returns
        -------
        xarray.Dataset
            Dataset indexed by `recording` (0-based) and `cap` with variables:

            - `temporal_fraction` `(recording, cap)`: fraction of volumes assigned
              to each CAP.
            - `counts` `(recording, cap)`: number of contiguous episodes of each CAP.
            - `persistence` `(recording, cap)`: mean episode duration. Zero when the
              CAP never appears. Units are inherited from the `time` coordinate's
              `units` attribute, or `"time"` when no such attribute exists, or
              `"volumes"` when no `time` coordinate is present.
            - `transition_frequency` `(recording,)`: total number of CAP switches.
            - `transition_matrix` `(recording, cap_from, cap_to)`: row-normalized
              transition probability matrix. Rows sum to 1 when the corresponding
              CAP appears; zero rows indicate CAPs that never appear as the origin
              of a transition.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the estimator has not been fitted yet.
        """
        check_is_fitted(self)

        cap_ids = self.caps_.coords["cap"].values
        n_caps = len(cap_ids)
        n_recordings = len(self.labels_)

        tf = np.zeros((n_recordings, n_caps))
        cnt = np.zeros((n_recordings, n_caps), dtype=np.intp)
        pers = np.zeros((n_recordings, n_caps))
        trans_freq = np.zeros(n_recordings, dtype=np.intp)
        trans_mat = np.zeros((n_recordings, n_caps, n_caps))

        for i, lbl in enumerate(self.labels_):
            arr = lbl.values
            n_volumes = len(arr)
            durations = _volume_durations(lbl)

            for j, cap_id in enumerate(cap_ids):
                binary, n_segs = _segments(arr, int(cap_id))
                tf[i, j] = float(binary.sum()) / n_volumes
                cnt[i, j] = n_segs
                # Floor n_segs at 1 to avoid 0/0; the binary.sum()==0 case
                # yields 0.0, correctly reflecting a CAP that never appears.
                pers[i, j] = float((binary * durations).sum()) / max(n_segs, 1)

            trans_freq[i] = int(np.sum(np.diff(arr) != 0))

            if n_volumes > 1:
                trans_from = arr[:-1]
                trans_to = arr[1:]
                for fi, from_id in enumerate(cap_ids):
                    mask_from = trans_from == from_id
                    total = int(mask_from.sum())
                    if total == 0:
                        continue
                    for ti, to_id in enumerate(cap_ids):
                        trans_mat[i, fi, ti] = (
                            float((mask_from & (trans_to == to_id)).sum()) / total
                        )

        rec_coords = np.arange(n_recordings)

        # Infer persistence units from the time coordinate of the first recording.
        pers_units = "volumes"
        if self.labels_ and "time" in self.labels_[0].coords:
            time_units = self.labels_[0].coords["time"].attrs.get("units")
            pers_units = str(time_units) if time_units is not None else "time"

        return xr.Dataset(
            {
                "temporal_fraction": xr.DataArray(
                    tf,
                    dims=["recording", "cap"],
                    coords={"recording": rec_coords, "cap": cap_ids},
                    attrs={"long_name": "Temporal fraction"},
                ),
                "counts": xr.DataArray(
                    cnt,
                    dims=["recording", "cap"],
                    coords={"recording": rec_coords, "cap": cap_ids},
                    attrs={"long_name": "Episode counts"},
                ),
                "persistence": xr.DataArray(
                    pers,
                    dims=["recording", "cap"],
                    coords={"recording": rec_coords, "cap": cap_ids},
                    attrs={
                        "long_name": "Persistence",
                        "units": pers_units,
                    },
                ),
                "transition_frequency": xr.DataArray(
                    trans_freq,
                    dims=["recording"],
                    coords={"recording": rec_coords},
                    attrs={"long_name": "Transition frequency"},
                ),
                "transition_matrix": xr.DataArray(
                    trans_mat,
                    dims=["recording", "cap_from", "cap_to"],
                    coords={
                        "recording": rec_coords,
                        "cap_from": cap_ids,
                        "cap_to": cap_ids,
                    },
                    attrs={"long_name": "Transition probability matrix"},
                ),
            }
        )

    def _preprocess(
        self, X: npt.NDArray[np.floating], in_place: bool = False
    ) -> npt.NDArray[np.floating]:
        """Apply metric-specific normalization to a (n_samples, n_features) array.

        Parameters
        ----------
        X : (n_samples, n_features) numpy.ndarray
            Raw (stacked) volumes.
        in_place : bool, default: False
            Whether to modify `X` in place. When `True`, no extra copy is
            allocated; the caller must ensure `X` is already a fresh array (e.g.
            freshly allocated by `numpy.concatenate`).

        Returns
        -------
        (n_samples, n_features) numpy.ndarray
            Normalized volumes. Same object as `X` when `in_place=True`.
        """
        if self.metric == "correlation":
            if not in_place:
                X = X.copy()
            X -= X.mean(axis=1, keepdims=True)
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            X /= np.where(norms == 0.0, 1.0, norms)
        elif self.metric == "cosine":
            if not in_place:
                X = X.copy()
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            X /= np.where(norms == 0.0, 1.0, norms)
        return X

    def _prepare_data(
        self, X: xr.DataArray
    ) -> tuple[npt.NDArray[np.floating], xr.DataArray]:
        """Stack and preprocess a single DataArray for prediction.

        Parameters
        ----------
        X : (time, ...) xarray.DataArray
            Input data. Must already be validated by the caller.

        Returns
        -------
        X_proc : (time, space) numpy.ndarray
            Preprocessed volumes ready for clustering or prediction.
        X_stacked : (time, space) xarray.DataArray
            Stacked version of the input, used to recover spatial coordinates
            when building `caps_`.
        """
        spatial_dims = [str(d) for d in X.dims if d != "time"]
        X_stacked = X.stack(space=spatial_dims)
        X_raw = X_stacked.values

        if np.isnan(X_raw).any():
            raise ValueError(
                "Input data contains NaN values. A common cause is z-score "
                "standardization of constant (zero-variance) voxels outside a brain "
                "mask. Fill or mask background voxels before calling fit(), e.g. "
                "`data.fillna(0)` or `data.where(mask > 0, 0)`."
            )

        return self._preprocess(X_raw, in_place=False), X_stacked

    def select_n_clusters(
        self,
        X: list[xr.DataArray] | xr.DataArray,
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
        X : list[xarray.DataArray] or xarray.DataArray
            Same data that will later be passed to
            [`fit`][confusius.connectivity.CAP.fit].
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

        recordings = [X] if isinstance(X, xr.DataArray) else list(X)
        if not recordings:
            raise ValueError("X must contain at least one recording.")
        for rec in recordings:
            validate_time_series(rec, operation_name="CAP.select_n_clusters")

        spatial_dims = [str(d) for d in recordings[0].dims if d != "time"]
        stacks = [rec.stack(space=spatial_dims) for rec in recordings]
        X_raw = np.concatenate([s.values for s in stacks], axis=0)
        del stacks

        if np.isnan(X_raw).any():
            raise ValueError(
                "Input data contains NaN values. A common cause is z-score "
                "standardization of constant (zero-variance) voxels outside a brain "
                "mask. Fill or mask background voxels before calling fit(), e.g. "
                "`data.fillna(0)` or `data.where(mask > 0, 0)`."
            )

        X_proc = self._preprocess(X_raw, in_place=True)

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

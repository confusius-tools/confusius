"""Connectivity matrix estimation from time series data.

Adapted from `nilearn.connectome.connectivity_matrices` (BSD-3-Clause License; see
`NOTICE` for details).
"""

import warnings
from math import floor, sqrt
from typing import Callable, Literal

import numpy as np
import numpy.typing as npt
import xarray as xr
from scipy import linalg
from sklearn.base import BaseEstimator, clone
from sklearn.covariance import LedoitWolf
from sklearn.utils.validation import check_is_fitted

from confusius._utils import find_stack_level
from confusius.validation import validate_time_series

_ALLOWED_KINDS = (
    "covariance",
    "correlation",
    "partial correlation",
    "tangent",
    "precision",
)
"""Allowed values for the `kind` parameter of `ConnectivityMatrix`."""


def _check_square(matrix: npt.NDArray) -> None:
    """Raise a ValueError if `matrix` is not square."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[-1]:
        raise ValueError(
            f"Expected a square matrix, got array of shape {matrix.shape}."
        )


def _check_spd(matrix: npt.NDArray) -> None:
    """Raise a ValueError if `matrix` is not symmetric positive definite."""
    if not np.allclose(matrix, matrix.T, atol=1e-7):
        raise ValueError("Expected a symmetric positive definite matrix.")
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        raise ValueError("Expected a symmetric positive definite matrix.")


def _form_symmetric(
    function: Callable, eigenvalues: npt.NDArray, eigenvectors: npt.NDArray
) -> npt.NDArray:
    """Form the symmetric matrix from transformed eigenvalues and fixed eigenvectors.

    Parameters
    ----------
    function : callable
        Transform applied element-wise to `eigenvalues`.
    eigenvalues : (n_features,) numpy.ndarray
        Input eigenvalues.
    eigenvectors : (n_features, n_features) numpy.ndarray
        Unitary matrix of eigenvectors.

    Returns
    -------
    (n_features, n_features) numpy.ndarray
        Symmetric matrix `eigenvectors @ diag(function(eigenvalues)) @ eigenvectors.T`.
    """
    return np.dot(eigenvectors * function(eigenvalues), eigenvectors.T)


def _map_eigenvalues(function, symmetric: npt.NDArray) -> npt.NDArray:
    """Apply `function` to the eigenvalues of a real symmetric matrix.

    Parameters
    ----------
    function : callable
        Transform applied element-wise to the eigenvalues.
    symmetric : (n_features, n_features) numpy.ndarray
        Input symmetric matrix.

    Returns
    -------
    (n_features, n_features) numpy.ndarray
        Symmetric matrix with transformed eigenvalues and the same eigenvectors.

    Notes
    -----
    No error is raised if the input matrix is not symmetric, but the result will be
    incorrect in that case.
    """
    eigenvalues, eigenvectors = linalg.eigh(symmetric)
    return _form_symmetric(function, eigenvalues, eigenvectors)


def _geometric_mean(
    matrices: list[npt.NDArray],
    init: npt.NDArray | None = None,
    max_iter: int = 10,
    tol: float | None = 1e-7,
) -> npt.NDArray:
    """Compute the geometric mean of symmetric positive definite matrices.

    The geometric mean minimizes the sum of squared Riemannian distances to each input
    matrix. For positive scalars this reduces to the ordinary geometric mean.

    Parameters
    ----------
    matrices : list[(n_features, n_features) numpy.ndarray]
        Symmetric positive definite matrices.
    init : (n_features, n_features) numpy.ndarray, optional
        Initialization for the gradient descent. Defaults to the arithmetic mean.
    max_iter : int, default: 10
        Maximum number of gradient descent iterations.
    tol : float or None, default: 1e-7
        Convergence tolerance on the gradient norm (normalized by matrix size). If
        `None`, no convergence check is performed.

    Returns
    -------
    (n_features, n_features) numpy.ndarray
        Geometric mean of `matrices`.

    Raises
    ------
    ValueError
        If `matrices` are not all square, not all of the same shape, or not all
        symmetric positive definite, or if `init` has an incompatible shape.
    FloatingPointError
        If a `NaN` appears after the matrix logarithm step.

    References
    ----------
    [^1]:
        Fletcher, P. Thomas, and Sarang Joshi. "Riemannian Geometry for the Statistical
        Analysis of Diffusion Tensor Data." Signal Processing, vol. 87, no. 2, Feb.
        2007, pp. 250–62. DOI.org (Crossref),
        https://doi.org/10.1016/j.sigpro.2005.12.018.
    """
    n_features = matrices[0].shape[0]
    for matrix in matrices:
        _check_square(matrix)
        if matrix.shape[0] != n_features:
            raise ValueError("Matrices are not of the same shape.")
        _check_spd(matrix)

    matrices_arr = np.array(matrices)
    if init is None:
        gmean = np.mean(matrices_arr, axis=0)
    else:
        _check_square(init)
        if init.shape[0] != n_features:
            raise ValueError("Initialization has incorrect shape.")
        _check_spd(init)
        gmean = init

    norm_old = np.inf
    step = 1.0

    for _ in range(max_iter):
        vals_gmean, vecs_gmean = linalg.eigh(gmean)
        gmean_inv_sqrt = _form_symmetric(np.sqrt, 1.0 / vals_gmean, vecs_gmean)
        whitened = [gmean_inv_sqrt.dot(m).dot(gmean_inv_sqrt) for m in matrices_arr]
        logs = [_map_eigenvalues(np.log, w) for w in whitened]
        logs_mean = np.mean(logs, axis=0)
        if np.any(np.isnan(logs_mean)):
            raise FloatingPointError("Nan value after logarithm operation.")

        norm = np.linalg.norm(logs_mean)

        vals_log, vecs_log = linalg.eigh(logs_mean)
        gmean_sqrt = _form_symmetric(np.sqrt, vals_gmean, vecs_gmean)
        gmean = gmean_sqrt.dot(_form_symmetric(np.exp, vals_log * step, vecs_log)).dot(
            gmean_sqrt
        )

        if norm < norm_old:
            norm_old = norm
        elif norm > norm_old:
            step = step / 2.0
            norm = norm_old

        if tol is not None and norm / gmean.size < tol:
            break

    if tol is not None and norm / gmean.size >= tol:
        warnings.warn(
            f"Maximum number of iterations {max_iter} reached without "
            f"getting to the requested tolerance level {tol}.",
            stacklevel=find_stack_level(),
        )

    return gmean


def symmetric_matrix_to_vector(
    symmetric: npt.NDArray, discard_diagonal: bool = False
) -> npt.NDArray:
    """Return the flattened lower triangular part of a symmetric matrix.

    Diagonal elements are divided by `sqrt(2)` when `discard_diagonal` is `False`
    so that the Frobenius norm is preserved under the vectorization.

    Acts on the last two dimensions if the input is not 2D.

    Parameters
    ----------
    symmetric : (..., n_features, n_features) numpy.ndarray
        Input symmetric matrix or batch of symmetric matrices.
    discard_diagonal : bool, default: False
        Whether diagonal elements should be omitted from the output.

    Returns
    -------
    numpy.ndarray
        Flattened lower triangular part. Shape is `(..., n_features * (n_features + 1)
        / 2)` when `discard_diagonal` is `False` and `(..., (n_features - 1) *
        n_features / 2)` otherwise.

    Notes
    -----
    Adapted from
    [`nilearn.connectome.connectivity_matrices`](https://github.com/nilearn/nilearn)
    (BSD-3-Clause License; see `NOTICE` for attribution).
    """
    if discard_diagonal:
        tril_mask = np.tril(np.ones(symmetric.shape[-2:]), k=-1).astype(bool)
        return symmetric[..., tril_mask]
    scaling = np.ones(symmetric.shape[-2:])
    np.fill_diagonal(scaling, sqrt(2.0))
    tril_mask = np.tril(np.ones(symmetric.shape[-2:])).astype(bool)
    return symmetric[..., tril_mask] / scaling[tril_mask]


def vector_to_symmetric_matrix(
    vec: npt.NDArray, diagonal: npt.NDArray | None = None
) -> npt.NDArray:
    """Return the symmetric matrix given its flattened lower triangular part.

    This is the inverse of
    [`symmetric_matrix_to_vector`][confusius.connectivity.symmetric_matrix_to_vector].
    Diagonal elements are multiplied by `sqrt(2)` to invert the norm-preserving scaling
    applied during vectorization. Acts on the last dimension of the input if it is not
    1D.

    Parameters
    ----------
    vec : (..., n * (n + 1) / 2) numpy.ndarray or (..., (n - 1) * n / 2) numpy.ndarray
        Vectorized lower triangular part. The diagonal may be included in `vec` or
        supplied separately via `diagonal`.
    diagonal : numpy.ndarray, shape (..., n), optional
        Diagonal values to insert. When provided, `vec` is assumed to contain only the
        off-diagonal elements and `diagonal` supplies the main diagonal.

    Returns
    -------
    numpy.ndarray, shape (..., n, n)
        Reconstructed symmetric matrix.

    Raises
    ------
    ValueError
        If `vec` has a length that does not correspond to a valid triangular number, or
        if `diagonal` has an incompatible shape.

    Notes
    -----
    Adapted from
    [`nilearn.connectome.connectivity_matrices`](https://github.com/nilearn/nilearn)
    (BSD-3-Clause License; see `NOTICE` for attribution).
    """
    n = vec.shape[-1]
    n_columns = (sqrt(8 * n + 1) - 1.0) / 2
    if diagonal is not None:
        n_columns += 1

    if n_columns > floor(n_columns):
        raise ValueError(
            f"Vector of unsuitable shape {vec.shape} cannot be transformed to "
            "a symmetric matrix."
        )

    n_columns = int(n_columns)
    first_shape = vec.shape[:-1]
    if diagonal is not None and (
        diagonal.shape[:-1] != first_shape or diagonal.shape[-1] != n_columns
    ):
        raise ValueError(
            f"diagonal of shape {diagonal.shape} incompatible "
            f"with vector of shape {vec.shape}"
        )

    sym = np.zeros((*first_shape, n_columns, n_columns))

    skip_diagonal = diagonal is not None
    mask = np.tril(np.ones((n_columns, n_columns)), k=-skip_diagonal).astype(bool)
    sym[..., mask] = vec
    sym.swapaxes(-1, -2)[..., mask] = vec

    mask.fill(False)
    np.fill_diagonal(mask, True)
    if diagonal is not None:
        sym[..., mask] = diagonal
    sym[..., mask] *= sqrt(2)

    return sym


def covariance_to_correlation(covariance: npt.NDArray) -> npt.NDArray:
    """Return the correlation matrix for a given covariance matrix.

    Parameters
    ----------
    covariance : numpy.ndarray, shape (n_features, n_features)
        Input covariance matrix.

    Returns
    -------
    numpy.ndarray, shape (n_features, n_features)
        Correlation matrix. Diagonal elements are exactly `1`.

    Notes
    -----
    Adapted from
    [`nilearn.connectome.connectivity_matrices`](https://github.com/nilearn/nilearn)
    (BSD-3-Clause License; see `NOTICE` for attribution).
    """
    diagonal = np.atleast_2d(1.0 / np.sqrt(np.diag(covariance)))
    correlation = covariance * diagonal * diagonal.T
    np.fill_diagonal(correlation, 1.0)
    return correlation


def precision_to_partial_correlation(precision: npt.NDArray) -> npt.NDArray:
    """Return the partial correlation matrix for a given precision matrix.

    Parameters
    ----------
    precision : numpy.ndarray, shape (n_features, n_features)
        Input precision matrix (inverse of the covariance matrix).

    Returns
    -------
    numpy.ndarray, shape (n_features, n_features)
        Partial correlation matrix. Diagonal elements are exactly `1`.

    Notes
    -----
    Adapted from
    [`nilearn.connectome.connectivity_matrices`](https://github.com/nilearn/nilearn)
    (BSD-3-Clause License; see `NOTICE` for attribution).
    """
    partial = -covariance_to_correlation(precision)
    np.fill_diagonal(partial, 1.0)
    return partial


class ConnectivityMatrix(BaseEstimator):
    """Functional connectivity matrices from fUSI region time series.

    Computes pairwise connectivity matrices between brain regions from time series
    DataArrays using one of several estimators: covariance, correlation, partial
    correlation, precision, or tangent-space projection. Supports both single-subject
    and group-level analysis.

    Parameters
    ----------
    cov_estimator : sklearn covariance estimator, optional
        Estimator used to compute covariance matrices. Defaults to
        `LedoitWolf(store_precision=False)`, which applies a small shrinkage towards
        zero compared to the maximum-likelihood estimate.
    kind : {"covariance", "correlation", "partial correlation", "tangent", \
            "precision"}, default: "correlation"
        Type of connectivity matrix to compute.

        - `"covariance"`: raw covariance matrix.
        - `"correlation"`: Pearson correlation matrix.
        - `"partial correlation"`: partial correlation matrix, controlling for all
          other variables.
        - `"precision"`: inverse of the covariance matrix.
        - `"tangent"`: symmetric displacement in the tangent space at the group
          geometric mean. Requires at least two subjects in `fit_transform`.

    vectorize : bool, default: False
        Whether connectivity matrices should be flattened to 1D vectors containing only
        the lower triangular elements.
    discard_diagonal : bool, default: False
        Whether diagonal elements should be excluded from the vectorized output. Only
        used when `vectorize` is `True`.

    Attributes
    ----------
    cov_estimator_ : sklearn covariance estimator
        A copy of `cov_estimator` with the same parameters, used during fitting.
    mean_ : (n_features, n_features) numpy.ndarray
        Mean connectivity matrix across subjects. For `"tangent"` kind, this is the
        geometric mean of the covariance matrices. For other kinds, it is the arithmetic
        mean.
    whitening_ : (n_features, n_features) numpy.ndarray or None
        Inverse square-root of the geometric mean covariance. Only set for `"tangent"`
        kind; `None` otherwise.
    n_features_in_ : int
        Number of features seen during `fit`.
    features_dim_in_ : str
        Name of the features dimension in the input DataArrays.

    Notes
    -----
    Adapted from Nilearn's
    [`ConnectivityMeasure`][nilearn.connectome.ConnectivityMeasure] (BSD-3-Clause
    License; see `NOTICE` for attribution).

    References
    ----------
    [^1]:
        Varoquaux, G., Baronnet, F., Kleinschmidt, A., Fillard, P., Thirion, B. (2010).
        Detection of Brain Functional-Connectivity Difference in Post-stroke Patients
        Using Group-Level Covariance Modeling. In: Jiang, T., Navab, N., Pluim, J.P.W.,
        Viergever, M.A. (eds) Medical Image Computing and Computer-Assisted Intervention
        – MICCAI 2010. MICCAI 2010. Lecture Notes in Computer Science, vol 6361.
        Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-642-15705-9_25

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> from confusius.connectivity import ConnectivityMatrix
    >>>
    >>> rng = np.random.default_rng(0)
    >>> # Five subjects, each with 100 time points and 10 brain regions.
    >>> signals = [
    ...     xr.DataArray(
    ...         rng.standard_normal((100, 10)),
    ...         dims=["time", "regions"],
    ...     )
    ...     for _ in range(5)
    ... ]
    >>>
    >>> measure = ConnectivityMatrix(kind="correlation")
    >>> connectivities = measure.fit_transform(signals)
    >>> connectivities.shape
    (5, 10, 10)
    >>>
    >>> # Vectorized output.
    >>> measure_vec = ConnectivityMatrix(kind="correlation", vectorize=True)
    >>> vecs = measure_vec.fit_transform(signals)
    >>> vecs.shape
    (5, 55)
    """

    def __init__(
        self,
        *,
        cov_estimator=None,
        kind: Literal[
            "covariance", "correlation", "partial correlation", "tangent", "precision"
        ] = "correlation",
        vectorize: bool = False,
        discard_diagonal: bool = False,
    ) -> None:
        self.cov_estimator = cov_estimator
        self.kind = kind
        self.vectorize = vectorize
        self.discard_diagonal = discard_diagonal

    def _normalize_input(
        self, X: xr.DataArray | list[xr.DataArray]
    ) -> list[xr.DataArray]:
        """Accept a single DataArray or a list of DataArrays."""
        if isinstance(X, xr.DataArray):
            return [X]
        if not hasattr(X, "__iter__"):
            raise TypeError(
                "X must be a DataArray or a list of DataArrays. "
                f"Got {type(X).__name__}."
            )
        subjects = list(X)
        if not all(isinstance(s, xr.DataArray) for s in subjects):
            bad_types = [
                type(s).__name__ for s in subjects if not isinstance(s, xr.DataArray)
            ]
            raise TypeError(f"All subjects must be xarray.DataArray, got {bad_types}.")
        return subjects

    def _validate_subjects(self, subjects: list[xr.DataArray]) -> None:
        """Validate that all subjects are (time, features) DataArrays with consistent dims."""
        for i, s in enumerate(subjects):
            validate_time_series(
                s, operation_name=f"ConnectivityMatrix.fit (subject {i})"
            )
            non_time = [d for d in s.dims if d != "time"]
            if len(non_time) != 1:
                raise ValueError(
                    f"Subject {i} must have exactly one non-time dimension, "
                    f"got dims {s.dims}."
                )

        feat_dims = [str(next(d for d in s.dims if d != "time")) for s in subjects]
        if len(set(feat_dims)) > 1:
            raise ValueError(
                "All subjects must have the same features dimension name. "
                f"Got {feat_dims}."
            )

        feat_sizes = [s.sizes[feat_dims[i]] for i, s in enumerate(subjects)]
        if len(set(feat_sizes)) > 1:
            raise ValueError(
                f"All subjects must have the same number of features. Got {feat_sizes}."
            )

    def _to_numpy(self, subjects: list[xr.DataArray]) -> list[npt.NDArray]:
        """Extract `(time, features)` numpy arrays from DataArrays."""
        arrays = []
        for s in subjects:
            feat_dim = next(d for d in s.dims if d != "time")
            arrays.append(s.transpose("time", feat_dim).values)
        return arrays

    def _compute_covariances(self, arrays: list[npt.NDArray]) -> list[npt.NDArray]:
        """Compute covariance matrices, z-scoring first for the correlation kind."""
        if self.kind == "correlation":
            covariances = []
            for x in arrays:
                mean = x.mean(axis=0)
                std = x.std(axis=0)
                std[std == 0] = 1.0  # Avoid division by zero for constant features.
                x_std = (x - mean) / std
                covariances.append(self.cov_estimator_.fit(x_std).covariance_)
        else:
            covariances = [self.cov_estimator_.fit(x).covariance_ for x in arrays]
        return covariances

    def _covariances_to_connectivities(
        self, covariances: list[npt.NDArray]
    ) -> list[npt.NDArray]:
        """Convert covariance matrices to the requested connectivity kind."""
        if self.kind in ("covariance", "tangent"):
            return covariances
        if self.kind == "precision":
            return [linalg.inv(cov) for cov in covariances]
        if self.kind == "partial correlation":
            return [
                precision_to_partial_correlation(linalg.inv(cov)) for cov in covariances
            ]
        # "correlation" kind.
        return [covariance_to_correlation(cov) for cov in covariances]

    def _fit_transform_subjects(
        self,
        subjects: list[xr.DataArray],
        do_fit: bool,
        do_transform: bool,
    ) -> npt.NDArray | None:
        """Core fit/transform logic.

        Covariances are computed only once.

        Parameters
        ----------
        subjects : list[xarray.DataArray]
            Already-validated subject DataArrays.
        do_fit : bool
            Whether to update fitted attributes (`mean_`, `whitening_`, etc.).
        do_transform : bool
            Whether to compute and return connectivity matrices.

        Returns
        -------
        numpy.ndarray or None
            Connectivity matrices (or vectors) when `do_transform` is `True`;
            `None` otherwise.
        """
        if do_fit:
            self.features_dim_in_: str = str(
                next(d for d in subjects[0].dims if d != "time")
            )
            self.n_features_in_: int = subjects[0].sizes[self.features_dim_in_]
            cov_estimator = (
                LedoitWolf(store_precision=False)
                if self.cov_estimator is None
                else self.cov_estimator
            )
            self.cov_estimator_ = clone(cov_estimator)

        arrays = self._to_numpy(subjects)
        covariances = self._compute_covariances(arrays)
        connectivities = self._covariances_to_connectivities(covariances)

        if do_fit:
            if self.kind == "tangent":
                self.mean_: npt.NDArray = _geometric_mean(
                    covariances, max_iter=30, tol=1e-7
                )
                self.whitening_: npt.NDArray | None = _map_eigenvalues(
                    lambda x: 1.0 / np.sqrt(x), self.mean_
                )
            else:
                self.mean_ = np.mean(connectivities, axis=0)
                # Enforce exact symmetry to suppress numerical noise.
                self.mean_ = (self.mean_ + self.mean_.T) * 0.5
                self.whitening_ = None

        if not do_transform:
            return None

        if self.kind == "tangent":
            assert self.whitening_ is not None
            connectivities = [
                _map_eigenvalues(np.log, self.whitening_.dot(cov).dot(self.whitening_))
                for cov in connectivities
            ]

        connectivities_arr = np.array(connectivities)
        if self.vectorize:
            connectivities_arr = symmetric_matrix_to_vector(
                connectivities_arr, discard_diagonal=self.discard_diagonal
            )
        return connectivities_arr

    def fit(
        self, X: xr.DataArray | list[xr.DataArray], y: None = None
    ) -> "ConnectivityMatrix":
        """Fit the covariance estimator and compute group-level statistics.

        Parameters
        ----------
        X : xarray.DataArray or list[xarray.DataArray]
            Time series for each subject. Each DataArray must have a `time` dimension
            and exactly one additional dimension (the features/regions dimension). The
            number of timepoints may differ across subjects, but the features dimension
            must have the same name and size.
        y : None, optional
            Ignored. Present for sklearn API compatibility.

        Returns
        -------
        ConnectivityMatrix
            Fitted estimator.

        Raises
        ------
        TypeError
            If `X` is not a DataArray or list of DataArrays.
        ValueError
            If any subject is missing the `time` dimension, has an incorrect number of
            dimensions, has inconsistent feature sizes, or if `kind` is not one of the
            allowed values.

        Notes
        -----
        Dask-backed DataArrays are computed in memory during `fit` when covariance
        matrices are estimated. This class is inherently eager: covariance estimation
        requires the full time series.
        """
        del y
        if self.kind not in _ALLOWED_KINDS:
            raise ValueError(
                f"kind must be one of {_ALLOWED_KINDS}, got {self.kind!r}."
            )
        subjects = self._normalize_input(X)
        self._validate_subjects(subjects)
        self._fit_transform_subjects(subjects, do_fit=True, do_transform=False)
        return self

    def transform(self, X: xr.DataArray | list[xr.DataArray]) -> npt.NDArray:
        """Compute connectivity matrices for new subjects.

        Parameters
        ----------
        X : xarray.DataArray or list[xarray.DataArray]
            Time series for each subject. The features dimension name and size must
            match the values seen during `fit`.

        Returns
        -------
        (n_subjects, n_features, n_features) numpy.ndarray or
                (n_subjects, n_features * (n_features + 1) / 2) numpy.ndarray
            Connectivity matrices, or their vectorized lower triangular parts when
            `vectorize` is `True`.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the estimator has not been fitted yet.
        ValueError
            If any subject has a features dimension that does not match
            `features_dim_in_` or `n_features_in_`.
        """
        check_is_fitted(self)
        subjects = self._normalize_input(X)
        self._validate_subjects(subjects)
        for i, s in enumerate(subjects):
            feat_dim = str(next(d for d in s.dims if d != "time"))
            if feat_dim != self.features_dim_in_:
                raise ValueError(
                    f"Subject {i} has features dimension {feat_dim!r}, "
                    f"expected {self.features_dim_in_!r}."
                )
            if s.sizes[feat_dim] != self.n_features_in_:
                raise ValueError(
                    f"Subject {i} has {s.sizes[feat_dim]} features, "
                    f"expected {self.n_features_in_}."
                )
        result = self._fit_transform_subjects(subjects, do_fit=False, do_transform=True)
        assert result is not None
        return result

    def fit_transform(
        self, X: xr.DataArray | list[xr.DataArray], y: None = None
    ) -> npt.NDArray:
        """Fit and transform in one step, computing covariances only once.

        Parameters
        ----------
        X : xarray.DataArray or list[xarray.DataArray]
            Time series for each subject. Each DataArray must have a `time` dimension
            and exactly one additional features dimension. The number of timepoints may
            differ across subjects, but the features dimension must be consistent.
        y : None, optional
            Ignored. Present for sklearn API compatibility.

        Returns
        -------
        (n_subjects, n_features, n_features) numpy.ndarray or \
                (n_subjects, n_features * (n_features + 1) / 2) numpy.ndarray
            Connectivity matrices, or their vectorized lower triangular parts when
            `vectorize` is `True`.

        Raises
        ------
        TypeError
            If `X` is not a DataArray or list of DataArrays.
        ValueError
            If subjects have inconsistent features dimensions, if `kind` is not valid,
            or if `kind="tangent"` is used with a single subject (tangent space returns
            deviations from a group mean, which is trivially zero for a single subject).
        """
        del y
        if self.kind not in _ALLOWED_KINDS:
            raise ValueError(
                f"kind must be one of {_ALLOWED_KINDS}, got {self.kind!r}."
            )
        subjects = self._normalize_input(X)
        # Check tangent constraint before fitting, a single subject in fit_transform
        # always yields zero displacements (trivial deviation from a single-point mean).
        if self.kind == "tangent" and len(subjects) <= 1:
            raise ValueError(
                "Tangent space parametrization can only be applied to a group of "
                f"subjects, as it returns deviations to the mean. Got {subjects!r}."
            )
        self._validate_subjects(subjects)
        result = self._fit_transform_subjects(subjects, do_fit=True, do_transform=True)
        assert result is not None
        return result

    def inverse_transform(
        self,
        connectivities: npt.NDArray,
        diagonal: npt.NDArray | None = None,
    ) -> npt.NDArray:
        """Reconstruct connectivity matrices from vectorized or tangent-space forms.

        Parameters
        ----------
        connectivities : (n_subjects, n_features, n_features) numpy.ndarray or \
                (n_subjects, n_features * (n_features + 1) / 2) numpy.ndarray or \
                (n_subjects, (n_features - 1) * n_features / 2) numpy.ndarray
            Connectivity matrices or their vectorized forms. When `kind="tangent"`,
            these are tangent space displacements that are mapped back to covariance
            matrices.
        diagonal : numpy.ndarray, shape (n_subjects, n_features), optional
            Diagonal values to restore when `discard_diagonal` was `True`. Required for
            `"covariance"` and `"precision"` kinds when the diagonal was discarded; for
            `"correlation"` and `"partial correlation"`, a diagonal of ones is assumed
            automatically.

        Returns
        -------
        numpy.ndarray, shape (n_subjects, n_features, n_features)
            Reconstructed connectivity matrices. For `"tangent"` kind, these are the
            original covariance matrices.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the estimator has not been fitted yet.
        ValueError
            If the diagonal was discarded for an ambiguous kind (`"covariance"` or
            `"precision"`) and no `diagonal` is provided.
        """
        check_is_fitted(self)

        connectivities = np.array(connectivities)

        if self.vectorize:
            if self.discard_diagonal and diagonal is None:
                if self.kind in ("correlation", "partial correlation"):
                    # Correlation diagonal is always 1; divide by sqrt(2) to match the
                    # scaling applied in symmetric_matrix_to_vector.
                    diagonal = np.ones(
                        (connectivities.shape[0], self.mean_.shape[0])
                    ) / sqrt(2.0)
                else:
                    raise ValueError(
                        "Diagonal values were discarded and are unknown for "
                        f"{self.kind!r} kind; cannot reconstruct connectivity matrices."
                    )
            connectivities = vector_to_symmetric_matrix(
                connectivities, diagonal=diagonal
            )

        if self.kind == "tangent":
            mean_sqrt = _map_eigenvalues(np.sqrt, self.mean_)
            connectivities = np.array(
                [
                    mean_sqrt.dot(_map_eigenvalues(np.exp, d)).dot(mean_sqrt)
                    for d in connectivities
                ]
            )

        return connectivities

    def __sklearn_is_fitted__(self) -> bool:
        """Check whether the estimator has been fitted."""
        return hasattr(self, "cov_estimator_")

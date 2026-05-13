"""FastICA decomposition for `(time, ...)` fUSI DataArrays."""

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import xarray as xr
from sklearn.decomposition import FastICA as _SklearnFastICA

from confusius.decomposition._base import _BaseFUSIDecomposer


class FastICA(_BaseFUSIDecomposer):
    """Fast independent component analysis (ICA) for fUSI data.

    The FastICA algorithm is based on Hyvarinen *et al.* (2000).

    This estimator wraps
    [`sklearn.decomposition.FastICA`][sklearn.decomposition.FastICA] while keeping
    xarray metadata through [`transform`][confusius.decomposition.FastICA.transform] and
    [`inverse_transform`][confusius.decomposition.FastICA.inverse_transform]:

    - Input data are expected as `(time, ...)` where `...` are spatial dimensions.
    - `transform` returns a `(time, component)` DataArray.
    - `inverse_transform` reconstructs to `(time, ...)` using the fitted spatial
      coordinates.

    Parameters
    ----------
    n_components : int, optional
        Number of components to use. If `None`, all are used.
    algorithm : {"parallel", "deflation"}, default: "parallel"
        Specify which algorithm to use for FastICA.
    whiten : {"unit-variance", "arbitrary-variance"} or bool, default: "unit-variance"
        Specify the whitening strategy to use.

        - If `"arbitrary-variance"`, a whitening with arbitrary variance is used.
        - If `"unit-variance"`, the whitening matrix is rescaled to ensure that each
          recovered source has unit variance.
        - If `False`, the data are already considered whitened, and no whitening is
          performed.
    fun : {"logcosh", "exp", "cube"} or callable, default: "logcosh"
        The functional form of the `G` function used in the approximation to
        neg-entropy. Can be either `"logcosh"`, `"exp"`, or `"cube"`.

        You can also provide your own function. It should return a tuple containing the
        value of the function and its derivative at the point. The derivative should be
        averaged along its last dimension.
    fun_args : dict[str, Any], optional
        Arguments to send to the functional form. If empty or `None` and
        `fun="logcosh"`, `fun_args` defaults to `{"alpha": 1.0}`.
    max_iter : int, default: 200
        Maximum number of iterations during fit.
    tol : float, default: 1e-4
        A positive scalar giving the tolerance at which the un-mixing matrix is
        considered to have converged.
    w_init : (n_components, n_components) numpy.ndarray, optional
        Initial un-mixing array. If not provided, values drawn from a normal
        distribution are used.
    whiten_solver : {"svd", "eigh"}, default: "svd"
        The solver to use for whitening.

        - `"svd"` is more stable numerically if the problem is degenerate, and often
          faster when `n_samples <= n_features`.
        - `"eigh"` is generally more memory efficient when
          `n_samples >= n_features`, and can be faster when
          `n_samples >= 50 * n_features`.
    random_state : int, optional
        Used to initialize `w_init` when not provided, with a normal distribution. Pass
        an int for reproducible results across multiple function calls.

    Attributes
    ----------
    components_ : (n_components, ...) xarray.DataArray
        Linear operator to apply to the data to recover the independent sources,
        reshaped to the original spatial geometry.

        This is equal to the un-mixing matrix when `whiten` is `False`, and equal to
        `np.dot(unmixing_matrix, whitening_)` otherwise.
    mixing_ : (..., n_components) xarray.DataArray
        Pseudo-inverse of `components_`, reshaped to the fitted spatial geometry.

        It is the linear operator that maps independent sources back to the data.
    mean_ : (...) xarray.DataArray
        Per-feature empirical mean from the training set. Defined only when `whiten` is
        not `False`.
    whitening_ : (n_components, ...) xarray.DataArray
        Pre-whitening matrix reshaped to the original spatial geometry. Defined only
        when `whiten` is not `False`.

        This projects data onto the first `n_components` principal components.
    n_components_ : int
        Number of components estimated during fit.
    n_iter_ : int
        If `algorithm="deflation"`, the maximum number of iterations run across all
        components. Otherwise, the number of iterations taken to converge.
    n_features_in_ : int
        Number of features seen during fit.
    feature_names_in_ : (n_features_in_,) numpy.ndarray
        Feature names seen during `fit`. Defined only when flattened feature labels are
        all strings.

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> from confusius.decomposition import FastICA
    >>>
    >>> rng = np.random.default_rng(0)
    >>> data = xr.DataArray(
    ...     rng.standard_normal((200, 5, 10, 20)),
    ...     dims=["time", "z", "y", "x"],
    ... )
    >>>
    >>> ica = FastICA(n_components=5, random_state=0)
    >>> signals = ica.fit_transform(data)
    >>> signals.dims
    ('time', 'component')
    >>> reconstructed = ica.inverse_transform(signals)
    >>> reconstructed.dims
    ('time', 'z', 'y', 'x')

    References
    ----------
    [^1]:
        Hyvarinen, A., and Oja, E. (2000). "Independent component analysis: Algorithms
        and applications". Neural Networks, 13(4-5), 411-430.
    """

    _signals_long_name = "FastICA signals"

    def __init__(
        self,
        *,
        n_components: int | None = None,
        algorithm: Literal["parallel", "deflation"] = "parallel",
        whiten: Literal["arbitrary-variance", "unit-variance"] | bool = "unit-variance",
        fun: Literal["logcosh", "exp", "cube"] | Callable[..., Any] = "logcosh",
        fun_args: dict[str, Any] | None = None,
        max_iter: int = 200,
        tol: float = 1e-4,
        w_init: npt.NDArray[np.floating] | None = None,
        whiten_solver: Literal["svd", "eigh"] = "svd",
        random_state: int | None = None,
    ) -> None:
        self.n_components = n_components
        self.algorithm = algorithm
        self.whiten = whiten
        self.fun = fun
        self.fun_args = fun_args
        self.max_iter = max_iter
        self.tol = tol
        self.w_init = w_init
        self.whiten_solver = whiten_solver
        self.random_state = random_state

    def fit(self, X: xr.DataArray, y: None = None) -> "FastICA":
        """Fit FastICA on `(time, ...)` fUSI data.

        Parameters
        ----------
        X : (time, ...) xarray.DataArray
            Input fUSI data.
        y : None, optional
            Ignored. Present for scikit-learn API compatibility.

        Returns
        -------
        FastICA
            Fitted estimator.

        Raises
        ------
        ValueError
            If input has no `time` dimension, fewer than 2 timepoints, or no spatial
            dimensions.
        """
        del y

        X_proc, X_stacked, spatial_dims = self._prepare_data(
            X,
            check_layout=False,
            operation_name="FastICA.fit",
        )

        fastica = _SklearnFastICA(
            n_components=self.n_components,
            algorithm=self.algorithm,
            whiten=self.whiten,
            fun=self.fun,
            fun_args=self.fun_args,
            max_iter=self.max_iter,
            tol=self.tol,
            w_init=self.w_init,
            whiten_solver=self.whiten_solver,
            random_state=self.random_state,
        )
        fastica.fit(X_proc)

        self._store_fit_metadata(X, X_proc, X_stacked, spatial_dims)

        component_coord = np.arange(fastica.components_.shape[0], dtype=np.intp)
        self.components_ = self._reshape_component_matrix(
            fastica.components_,
            component_coord,
            long_name="Independent components",
        )
        self.mixing_ = self._reshape_feature_component_matrix(
            fastica.mixing_,
            component_coord,
        )

        if hasattr(fastica, "mean_"):
            self.mean_ = self._reshape_mean(fastica.mean_)

        if hasattr(fastica, "whitening_"):
            self.whitening_ = self._reshape_component_matrix(
                fastica.whitening_,
                component_coord,
                long_name="Whitening matrix",
            )

        self.n_components_ = int(fastica.components_.shape[0])
        self.n_iter_ = int(fastica.n_iter_)

        self._store_feature_names(X)
        self._estimator = fastica

        return self

    def _reshape_feature_component_matrix(
        self,
        matrix: npt.NDArray[np.floating],
        component_coord: npt.NDArray[np.intp],
    ) -> xr.DataArray:
        """Reshape a `(feature, component)` matrix to the fitted spatial geometry.

        Parameters
        ----------
        matrix : (n_features, n_components) numpy.ndarray
            Flat matrix to reshape.
        component_coord : (n_components,) numpy.ndarray
            Coordinate values for the component dimension.

        Returns
        -------
        (..., n_components) xarray.DataArray
            Matrix unstacked to the original spatial dimensions and transposed to
            `(*spatial_dims_, "component")`, with `attrs["long_name"]` and
            `attrs["cmap"]` set.
        """
        matrix_stacked = xr.DataArray(
            matrix,
            dims=["feature", "component"],
            coords={"feature": self._feature_coord_, "component": component_coord},
        )
        reshaped = matrix_stacked.unstack("feature").transpose(
            *self.spatial_dims_, "component"
        )
        reshaped.attrs.update({"long_name": "Mixing matrix", "cmap": "coolwarm"})
        return reshaped

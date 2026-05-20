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
    - `transform` returns a `(time, component)` DataArray of IC time courses.
    - `inverse_transform` reconstructs to `(time, ...)` using the fitted spatial
      coordinates.

    Two ICA modes are supported, selected via the `mode` parameter.

    **Spatial ICA** (`mode="spatial"`, default, matches FSL MELODIC):
    Maximises independence across the *spatial* dimension. The independent components
    are spatial maps; time courses are derived by projecting the data onto those maps.
    This is the appropriate choice when different brain regions are assumed to have
    independently fluctuating activity, which is the standard prior for network
    discovery. It is also better conditioned for fUSI data where the number of voxels
    greatly exceeds the number of time points.

    **Temporal ICA** (`mode="temporal"`):
    Maximises independence across the *time* dimension. The independent components
    are temporal signals; the spatial maps are the corresponding mixing weights.
    Suitable when temporal independence is the relevant prior.

    Parameters
    ----------
    n_components : int, optional
        Number of components to use. If not provided, all are used.
    mode : {"spatial", "temporal"}, default: "spatial"
        Whether to find spatially or temporally independent components:

        - `"spatial"`: fit on `(voxels, time)`. The independent components are
          spatial maps; the projected time courses are their temporal mixing weights.
          Matches FSL MELODIC's default for single-subject data.
        - `"temporal"`: fit on `(time, voxels)`. The independent components are time
          courses; the spatial maps are their voxel-wise mixing weights.

    algorithm : {"parallel", "deflation"}, default: "parallel"
        Specify which algorithm to use for FastICA.
    whiten : {"unit-variance", "arbitrary-variance"} or bool, default: "unit-variance"
        Specify the whitening strategy to use.

        - If `"arbitrary-variance"`, a whitening with arbitrary variance is used.
        - If `"unit-variance"`, the whitening matrix is rescaled to ensure that each
          recovered source has unit variance.
        - If `False`, the data are already considered whitened, and no whitening is
          performed.

    fun : {"logcosh", "exp", "cube"} or callable, default: "cube"
        The functional form of the `G` function used in the approximation to
        neg-entropy. Can be either `"logcosh"`, `"exp"`, or `"cube"`.

        Default is `"cube"` to align with FSL MELODIC (`pow3`).

        You can also provide your own function. It should return a tuple containing the
        value of the function and its derivative at the point. The derivative should be
        averaged along its last dimension.
    fun_args : dict[str, Any], optional
        Arguments to send to the functional form. If empty or `None` and
        `fun="logcosh"`, `fun_args` defaults to `{"alpha": 1.0}`.
    max_iter : int, default: 500
        Maximum number of iterations during fit. Default matches FSL MELODIC
        (`--maxit=500`).
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
    maps_ : (n_components, ...) xarray.DataArray
        Spatial maps reshaped to the original spatial geometry.

        - In `"spatial"` mode: the independent component sources themselves, that is,
          the spatially independent patterns that FastICA found. These are the ICs in
          the strict sense.
        - In `"temporal"` mode: the unmixing directions in voxel space, that is, the
          spatial weights applied to each volume to extract IC time courses. The actual
          temporal ICs are returned by `transform`.

    mean_ : (...) xarray.DataArray
        Per-voxel empirical mean from the training set.
    whitening_ : (n_components, ...) xarray.DataArray
        Pre-whitening matrix reshaped to the original spatial geometry. Only present in
        `"temporal"` mode when `whiten` is not `False`.
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
    Spatial ICA (default, matches FSL MELODIC):

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
        mode: Literal["spatial", "temporal"] = "spatial",
        algorithm: Literal["parallel", "deflation"] = "parallel",
        whiten: Literal["arbitrary-variance", "unit-variance"] | bool = "unit-variance",
        fun: Literal["logcosh", "exp", "cube"] | Callable[..., Any] = "cube",
        fun_args: dict[str, Any] | None = None,
        max_iter: int = 500,
        tol: float = 1e-4,
        w_init: npt.NDArray[np.floating] | None = None,
        whiten_solver: Literal["svd", "eigh"] = "svd",
        random_state: int | None = None,
    ) -> None:
        self.n_components = n_components
        self.mode = mode
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
        ValueError
            If `mode` is not `"spatial"` or `"temporal"`.
        """
        del y

        if self.mode not in {"spatial", "temporal"}:
            raise ValueError(
                f"mode must be 'spatial' or 'temporal', got '{self.mode}'."
            )

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

        self._store_fit_metadata(X, X_proc, X_stacked, spatial_dims)

        if self.mode == "spatial":
            self._fit_spatial(fastica, X_proc)
        else:
            self._fit_temporal(fastica, X_proc)

        self._store_feature_names(X)
        return self

    def _fit_spatial(
        self,
        fastica: _SklearnFastICA,
        X_proc: npt.NDArray[np.floating],
    ) -> None:
        """Fit spatial ICA: find spatially independent maps.

        Parameters
        ----------
        fastica : sklearn.decomposition.FastICA
            Configured but unfitted sklearn estimator.
        X_proc : (n_time, n_voxels) numpy.ndarray
            Stacked data matrix.
        """
        # Fit on transposed data: (n_voxels, n_time). Sklearn treats voxels as
        # samples and time points as features, so the resulting independent
        # components are spatial maps.
        fastica.fit(X_proc.T)

        # Spatial maps: (n_ica, n_voxels) — rows are independent spatial patterns.
        spatial_maps_flat: npt.NDArray[np.floating] = fastica.transform(X_proc.T).T

        voxel_mean: npt.NDArray[np.floating] = X_proc.mean(axis=0)

        component_coord = np.arange(spatial_maps_flat.shape[0], dtype=np.intp)
        self.maps_ = self._reshape_component_matrix(
            spatial_maps_flat,
            component_coord,
            long_name="IC spatial maps",
        )
        self.mean_ = self._reshape_mean(voxel_mean)

        self.n_components_ = int(spatial_maps_flat.shape[0])
        self.n_iter_ = int(fastica.n_iter_)
        self._spatial_components_flat_ = spatial_maps_flat
        self._spatial_feature_mean_ = voxel_mean
        self._estimator = fastica

    def _fit_temporal(
        self,
        fastica: _SklearnFastICA,
        X_proc: npt.NDArray[np.floating],
    ) -> None:
        """Fit temporal ICA: find temporally independent time courses.

        Parameters
        ----------
        fastica : sklearn.decomposition.FastICA
            Configured but unfitted sklearn estimator.
        X_proc : (n_time, n_voxels) numpy.ndarray
            Stacked data matrix.
        """
        fastica.fit(X_proc)

        component_coord = np.arange(fastica.components_.shape[0], dtype=np.intp)
        self.maps_ = self._reshape_component_matrix(
            fastica.components_,
            component_coord,
            long_name="IC unmixing maps",
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
        self._estimator = fastica

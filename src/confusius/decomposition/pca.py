"""Principal component decomposition for `(time, ...)` fUSI DataArrays."""

from typing import Literal

import numpy as np
import numpy.typing as npt
import xarray as xr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA as _SklearnPCA
from sklearn.utils.validation import check_is_fitted

from confusius.validation import validate_time_series


class PCA(BaseEstimator, TransformerMixin):
    """Principal component analysis for fUSI data.

    This estimator wraps [`sklearn.decomposition.PCA`][sklearn.decomposition.PCA] but
    keeps xarray metadata through `transform` and `inverse_transform`:

    - Input data are expected as `(time, ...)` where `...` are spatial dimensions.
    - `transform` returns a `(time, component)` DataArray.
    - `inverse_transform` reconstructs to `(time, ...)` using the fitted spatial
      coordinates.

    Parameters
    ----------
    n_components : int, float or "mle", default: None
        Number of components to keep.
        if `n_components` is not set all components are kept:

        `n_components == min(n_samples, n_features)`

        If `n_components == "mle"` and `svd_solver == "full"`, Minka's
        MLE is used to guess the dimension. Use of `n_components == "mle"`
        will interpret `svd_solver == "auto"` as `svd_solver == "full"`.

        If `0 < n_components < 1` and `svd_solver == "full"`, select the
        number of components such that the amount of variance that needs to be
        explained is greater than the percentage specified by `n_components`.

        If `svd_solver == "arpack"`, the number of components must be
        strictly less than the minimum of `n_features` and `n_samples`.

        Hence, the None case results in:

        `n_components == min(n_samples, n_features) - 1`.
    whiten : bool, default: False
        When `True` (`False` by default) the `components_` vectors are multiplied
        by the square root of `n_samples` and then divided by the singular values
        to ensure uncorrelated outputs with unit component-wise variances.

        Whitening will remove some information from the transformed signal
        (the relative variance scales of the components) but can sometime
        improve the predictive accuracy of the downstream estimators by
        making their data respect some hard-wired assumptions.
    svd_solver : {"auto", "full", "covariance_eigh", "arpack", "randomized"}, \
            default: "auto"
        "auto":
            The solver is selected by a default "auto" policy based on
            `X.shape` and `n_components`: if the input data has fewer than
            1000 features and more than 10 times as many samples, then the
            "covariance_eigh" solver is used. Otherwise, if the input data is
            larger than 500x500 and the number of components to extract is
            lower than 80% of the smallest dimension of the data, then the
            more efficient "randomized" method is selected. Otherwise the exact
            "full" SVD is computed and optionally truncated afterwards.

        "full":
            Run exact full SVD calling the standard LAPACK solver via
            `scipy.linalg.svd` and select the components by postprocessing.

        "covariance_eigh":
            Precompute the covariance matrix (on centered data), run a
            classical eigenvalue decomposition on the covariance matrix
            typically using LAPACK and select the components by postprocessing.
            This solver is very efficient for `n_samples >> n_features` and
            small `n_features`. It is, however, not tractable otherwise for
            large `n_features` (large memory footprint required to materialize
            the covariance matrix). Also note that compared to the "full"
            solver, this solver effectively doubles the condition number and is
            therefore less numerically stable (e.g. on input data with a large
            range of singular values).

        "arpack":
            Run SVD truncated to `n_components` calling ARPACK solver via
            `scipy.sparse.linalg.svds`. It requires strictly
            `0 < n_components < min(X.shape)`.

        "randomized":
            Run randomized SVD by the method of Halko et al.
    tol : float, default: 0.0
        Tolerance for singular values computed by `svd_solver == "arpack"`.
        Must be of range `[0.0, infinity)`.
    iterated_power : int or "auto", default: "auto"
        Number of iterations for the power method computed by
        `svd_solver == "randomized"`.
        Must be of range `[0, infinity)`.
    n_oversamples : int, default: 10
        This parameter is only relevant when `svd_solver="randomized"`.
        It corresponds to the additional number of random vectors to sample the
        range of `X` so as to ensure proper conditioning. See
        [`randomized_svd`][sklearn.utils.extmath.randomized_svd] for more details.
    power_iteration_normalizer : {"auto", "QR", "LU", "none"}, default: "auto"
        Power iteration normalizer for randomized SVD solver.
        Not used by ARPACK. See
        [`randomized_svd`][sklearn.utils.extmath.randomized_svd] for more details.
    random_state : int or None, default: None
        Used when the "arpack" or "randomized" solvers are used. Pass an int
        for reproducible results across multiple function calls.

    Attributes
    ----------
    components_ : (n_components, ...) xarray.DataArray
        Principal axes in feature space, representing the directions of maximum variance
        in the data. Equivalently, the right singular vectors of the centered input
        data, parallel to its eigenvectors. The components are sorted by decreasing
        `explained_variance_` and reshaped to the original spatial geometry.
    explained_variance_ : (n_components,) xarray.DataArray
        The amount of variance explained by each of the selected components. The
        variance estimation uses `n_samples - 1` degrees of freedom.

        Equal to `n_components` largest eigenvalues of the covariance matrix of `X`.
    explained_variance_ratio_ : (n_components,) xarray.DataArray
        Percentage of variance explained by each of the selected components.

        If `n_components` is not set then all components are stored and the sum of the
        ratios is equal to 1.0.
    singular_values_ : (n_components,) xarray.DataArray
        The singular values corresponding to each of the selected components. The
        singular values are equal to the 2-norms of the `n_components` variables in the
        lower-dimensional space.
    mean_ : (...) xarray.DataArray
        Per-feature empirical mean, estimated from the training set. Equal to
        `X.mean(axis=0)`.
    n_components_ : int
        The estimated number of components. When `n_components` is set to "mle" or a
        number between 0 and 1 (with `svd_solver == "full"`) this number is estimated
        from input data. Otherwise it equals the parameter n_components, or the lesser
        value of `n_features` and `n_samples` if `n_components` is `None`.
    n_samples_ : int
        Number of samples in the training data.
    noise_variance_ : float
        The estimated noise covariance following the Probabilistic PCA model from
        Tipping and Bishop 1999. See "Pattern Recognition and Machine Learning" by C.
        Bishop, 12.2.1 p. 574 or <http://www.miketipping.com/papers/met-mppca.pdf>. It
        is required to compute the estimated data covariance and score samples.

        Equal to the average of `min(n_features, n_samples) - n_components` smallest
        eigenvalues of the covariance matrix of `X`.
    n_features_in_ : int
        Number of features seen during fit.
    feature_names_in_ : (n_features_in_,) numpy.ndarray
        Feature names seen during `fit`. Defined only when flattened feature labels are
        all strings.

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> from confusius.decomposition import PCA
    >>>
    >>> rng = np.random.default_rng(0)
    >>> data = xr.DataArray(
    ...     rng.standard_normal((200, 10, 20)),
    ...     dims=["time", "y", "x"],
    ... )
    >>>
    >>> pca = PCA(n_components=5, random_state=0)
    >>> signals = pca.fit_transform(data)
    >>> signals.dims
    ('time', 'component')
    >>> reconstructed = pca.inverse_transform(signals)
    >>> reconstructed.dims
    ('time', 'y', 'x')
    """

    def __init__(
        self,
        *,
        n_components: int | float | Literal["mle"] | None = None,
        whiten: bool = False,
        svd_solver: Literal[
            "auto", "full", "covariance_eigh", "arpack", "randomized"
        ] = "auto",
        tol: float = 0.0,
        iterated_power: int | Literal["auto"] = "auto",
        n_oversamples: int = 10,
        power_iteration_normalizer: Literal["auto", "QR", "LU", "none"] = "auto",
        random_state: int | None = None,
    ) -> None:
        self.n_components = n_components
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.n_oversamples = n_oversamples
        self.power_iteration_normalizer = power_iteration_normalizer
        self.random_state = random_state

    def fit(self, X: xr.DataArray, y: None = None) -> "PCA":
        """Fit PCA on `(time, ...)` fUSI data.

        Parameters
        ----------
        X : (time, ...) xarray.DataArray
            Input fUSI data.
        y : None, optional
            Ignored. Present for scikit-learn API compatibility.

        Returns
        -------
        PCA
            Fitted estimator.

        Raises
        ------
        ValueError
            If input has no `time` dimension, fewer than 2 timepoints, or no spatial
            dimensions.
        TypeError
            If additional fit parameters are provided.
        """
        del y

        X_proc, X_stacked, spatial_dims = self._prepare_data(
            X,
            check_layout=False,
            operation_name="PCA.fit",
        )

        pca = _SklearnPCA(
            n_components=self.n_components,
            whiten=self.whiten,
            svd_solver=self.svd_solver,
            tol=self.tol,
            iterated_power=self.iterated_power,
            n_oversamples=self.n_oversamples,
            power_iteration_normalizer=self.power_iteration_normalizer,
            random_state=self.random_state,
        )
        pca.fit(X_proc)

        self.spatial_dims_ = spatial_dims
        self._spatial_sizes_ = {dim: int(X.sizes[dim]) for dim in self.spatial_dims_}
        self._feature_coord_ = X_stacked.coords["feature"]
        self._fit_attrs_ = dict(X.attrs)
        self._fit_name_ = X.name

        component_coord = np.arange(pca.components_.shape[0], dtype=np.intp)
        components_stacked = xr.DataArray(
            pca.components_,
            dims=["component", "feature"],
            coords={"component": component_coord, "feature": self._feature_coord_},
        )
        components = components_stacked.unstack("feature")
        components.attrs.update(
            {"long_name": "Principal components", "cmap": "coolwarm"}
        )
        self.components_: xr.DataArray = components

        mean_stacked = xr.DataArray(
            pca.mean_,
            dims=["feature"],
            coords={"feature": self._feature_coord_},
        )
        self.mean_: xr.DataArray = mean_stacked.unstack("feature")

        self.explained_variance_: xr.DataArray = xr.DataArray(
            pca.explained_variance_,
            dims=["component"],
            coords={"component": component_coord},
        )
        self.explained_variance_ratio_: xr.DataArray = xr.DataArray(
            pca.explained_variance_ratio_,
            dims=["component"],
            coords={"component": component_coord},
        )
        self.singular_values_: xr.DataArray = xr.DataArray(
            pca.singular_values_,
            dims=["component"],
            coords={"component": component_coord},
        )

        self.n_components_ = int(pca.n_components_)
        self.n_samples_ = int(X_proc.shape[0])
        self.n_features_in_ = int(X_proc.shape[1])
        self.noise_variance_ = float(pca.noise_variance_)

        if len(self.spatial_dims_) == 1 and self.spatial_dims_[0] in X.coords:
            feature_labels = np.asarray(X.coords[self.spatial_dims_[0]].values)
            if all(isinstance(value, (str, np.str_)) for value in feature_labels):
                self.feature_names_in_ = feature_labels.astype(str, copy=False)

        self._pca = pca

        return self

    def fit_transform(
        self, X: xr.DataArray, y: None = None, **fit_params: object
    ) -> xr.DataArray:
        """Fit PCA on `X` and return transformed component signals.

        Parameters
        ----------
        X : (time, ...) xarray.DataArray
            Input fUSI data.
        y : None, optional
            Ignored. Present for scikit-learn API compatibility.
        **fit_params : object
            Additional fit parameters. Unsupported for this estimator.

        Returns
        -------
        (time, component) xarray.DataArray
            PCA signals in component space.

        Raises
        ------
        TypeError
            If additional fit parameters are provided.
        """
        del y

        if fit_params:
            keys = ", ".join(sorted(fit_params))
            raise TypeError(f"Unexpected fit parameters for PCA.fit_transform: {keys}.")

        return self.fit(X).transform(X)

    def transform(self, X: xr.DataArray) -> xr.DataArray:
        """Project data into PCA component space.

        `X` is projected on the first principal components previously extracted from a
        training set.

        Parameters
        ----------
        X : (time, ...) xarray.DataArray
            Input fUSI data with the same spatial dimensions and sizes as the data used
            during fit.

        Returns
        -------
        (time, component) xarray.DataArray
            PCA signals in component space.
        """
        check_is_fitted(self)
        X_proc, _, _ = self._prepare_data(
            X,
            check_layout=True,
            operation_name="PCA.transform",
        )
        signals = self._pca.transform(X_proc)

        if "time" in X.coords:
            time_coord: npt.NDArray[np.generic] | xr.DataArray = X.coords["time"]
        else:
            time_coord = np.arange(X.sizes["time"], dtype=np.intp)

        transformed = xr.DataArray(
            signals,
            dims=["time", "component"],
            coords={
                "time": time_coord,
                "component": self.components_.coords["component"],
            },
        )
        transformed.attrs.update({"long_name": "PCA signals"})
        return transformed

    def inverse_transform(
        self, X: xr.DataArray | npt.NDArray[np.floating]
    ) -> xr.DataArray:
        """Reconstruct data from PCA component signals.

        Parameters
        ----------
        X : (time, component) xarray.DataArray or (time, component) numpy.ndarray
            Signals in PCA component space.

        Returns
        -------
        (time, ...) xarray.DataArray
            Reconstructed data in the fitted spatial geometry.

        Raises
        ------
        ValueError
            If `X` has invalid shape or component count.
        TypeError
            If `X` is neither `xarray.DataArray` nor `numpy.ndarray`.
        """
        check_is_fitted(self)

        if isinstance(X, xr.DataArray):
            if set(X.dims) != {"time", "component"}:
                raise ValueError(
                    "X must have exactly the dimensions {'time', 'component'} for "
                    "inverse_transform."
                )
            X_ordered = X.transpose("time", "component")
            signals = np.asarray(X_ordered.values, dtype=np.float64)
            time_coord = (
                X_ordered.coords["time"]
                if "time" in X_ordered.coords
                else np.arange(X_ordered.sizes["time"], dtype=np.intp)
            )
        elif isinstance(X, np.ndarray):
            if X.ndim != 2:
                raise ValueError(
                    f"X must be 2D with shape (time, component), got {X.shape}."
                )
            signals = np.asarray(X, dtype=np.float64)
            time_coord = np.arange(signals.shape[0], dtype=np.intp)
        else:
            raise TypeError(f"X must be DataArray or ndarray, got {type(X)}")

        if signals.shape[1] != self.n_components_:
            raise ValueError(
                f"X has {signals.shape[1]} components, but PCA was fitted with "
                f"{self.n_components_}."
            )

        reconstructed = self._pca.inverse_transform(signals)
        reconstructed_stacked = xr.DataArray(
            reconstructed,
            dims=["time", "feature"],
            coords={"time": time_coord, "feature": self._feature_coord_},
        )
        reconstructed_da = reconstructed_stacked.unstack("feature")
        reconstructed_da.attrs.update(self._fit_attrs_)
        reconstructed_da.name = self._fit_name_
        return reconstructed_da

    def _prepare_data(
        self,
        X: xr.DataArray,
        check_layout: bool,
        operation_name: str,
    ) -> tuple[npt.NDArray[np.float64], xr.DataArray, tuple[str, ...]]:
        """Validate and stack `(time, ...)` data into `(time, feature)` matrix."""
        validate_time_series(X, operation_name=operation_name)

        input_spatial_dims = tuple(str(dim) for dim in X.dims if dim != "time")
        if len(input_spatial_dims) == 0:
            raise ValueError(
                "X must have at least one spatial dimension besides 'time'."
            )

        if check_layout:
            spatial_dims = self.spatial_dims_
            if set(input_spatial_dims) != set(spatial_dims):
                raise ValueError(
                    "X spatial dimensions do not match fitted dimensions. "
                    f"Expected {spatial_dims}, got {input_spatial_dims}."
                )
            for dim in spatial_dims:
                if X.sizes[dim] != self._spatial_sizes_[dim]:
                    raise ValueError(
                        f"Spatial dimension '{dim}' has size {X.sizes[dim]} in X, "
                        f"but expected {self._spatial_sizes_[dim]} from fit."
                    )
        else:
            spatial_dims = input_spatial_dims

        X_ordered = X.transpose("time", *spatial_dims)
        X_stacked = X_ordered.stack(feature=list(spatial_dims))
        X_proc = np.asarray(X_stacked.values, dtype=np.float64)
        return X_proc, X_stacked, spatial_dims

    def __sklearn_is_fitted__(self) -> bool:
        """Check whether the estimator has been fitted."""
        return hasattr(self, "components_") and hasattr(self, "n_components_")

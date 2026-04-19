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
    """Principal component analysis for fUSI `xarray.DataArray` inputs.

    This estimator wraps `sklearn.decomposition.PCA`
    but keeps xarray metadata through `transform` and `inverse_transform`:

    - Input data are expected as `(time, ...)` where `...` are spatial dimensions.
    - `transform` returns a `(time, component)` `xarray.DataArray`.
    - `inverse_transform` reconstructs to `(time, ...)` using the fitted
      spatial coordinates.

    Parameters
    ----------
    n_components : int, float, "mle", or None, default: None
        Number of principal components to keep. Passed to sklearn PCA.
    whiten : bool, default: False
        Whether to whiten components. Passed to sklearn PCA.
    svd_solver : {"auto", "full", "covariance_eigh", "arpack", "randomized"}, \
            default: "auto"
        SVD solver strategy. Passed to sklearn PCA.
    tol : float, default: 0.0
        Tolerance for ARPACK solver. Passed to sklearn PCA.
    iterated_power : int or "auto", default: "auto"
        Number of power iterations for randomized solver. Passed to sklearn PCA.
    n_oversamples : int, default: 10
        Number of oversamples for randomized solver. Passed to sklearn PCA.
    power_iteration_normalizer : {"auto", "QR", "LU", "none"}, default: "auto"
        Power-iteration normalizer for randomized solver. Passed to sklearn PCA.
    random_state : int or None, default: None
        Seed for stochastic solvers. Passed to sklearn PCA.

    Attributes
    ----------
    components_ : (component, ...) xarray.DataArray
        Spatial principal components reshaped to the original spatial geometry.
    mean_ : (...) xarray.DataArray
        Per-feature mean map used by PCA.
    explained_variance_ : (component,) xarray.DataArray
        Explained variance for each component.
    explained_variance_ratio_ : (component,) xarray.DataArray
        Fraction of variance explained by each component.
    singular_values_ : (component,) xarray.DataArray
        Singular values for each selected component.

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
        """Fit PCA on `(time, ...)` data.

        Parameters
        ----------
        X : (time, ...) xarray.DataArray
            Input fUSI data.
        y : None, optional
            Ignored. Present for sklearn API compatibility.

        Returns
        -------
        PCA
            Fitted estimator.

        Raises
        ------
        ValueError
            If input has no `time` dimension, fewer than 2 timepoints, or no
            spatial dimensions.
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
        self.n_features_in_ = int(X_proc.shape[1])
        self.noise_variance_ = float(pca.noise_variance_)
        self._pca = pca

        return self

    def transform(self, X: xr.DataArray) -> xr.DataArray:
        """Project data into PCA component space.

        Parameters
        ----------
        X : (time, ...) xarray.DataArray
            Input data with the same spatial dimensions and sizes as the data
            used during fit.

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

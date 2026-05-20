"""Principal component decomposition for `(time, ...)` fUSI DataArrays."""

from typing import Literal

import numpy as np
import numpy.typing as npt
import xarray as xr
from sklearn.decomposition import PCA as _SklearnPCA

from confusius.decomposition._base import _BaseFUSIDecomposer


class PCA(_BaseFUSIDecomposer):
    """Principal component analysis (PCA) for fUSI data.

    Linear dimensionality reduction using singular value decomposition (SVD) of the data
    to project it to a lower dimensional space. The input data is centered but not
    scaled for each feature before applying the SVD.

    It uses the LAPACK implementation of the full SVD or a randomized truncated SVD by
    the method of Halko *et al.* (2009), depending on the shape of the input data and
    the number of components to extract.

    This estimator wraps [`sklearn.decomposition.PCA`][sklearn.decomposition.PCA] while
    keeping xarray metadata through [`transform`][confusius.decomposition.PCA.transform]
    and [`inverse_transform`][confusius.decomposition.PCA.inverse_transform]:

    - Input data are expected as `(time, ...)` where `...` are spatial dimensions.
    - `transform` returns a `(time, component)` DataArray.
    - `inverse_transform` reconstructs to `(time, ...)` using the fitted spatial
      coordinates.

    Parameters
    ----------
    n_components : int, float or "mle", optional
        Number of components to keep:

        - If `n_components` is not provided all components are kept: `n_components ==
          min(n_samples, n_features)`
        - If `n_components == "mle"` and `svd_solver == "full"`, Minka's MLE is used to
          guess the dimension. Use of `n_components == "mle"` will interpret `svd_solver
          == "auto"` as `svd_solver == "full"`.
        - If `0 < n_components < 1` and `svd_solver == "full"`, the number of
          components will be selected such that the amount of variance that needs to be
          explained is greater than the percentage specified by `n_components`.
        - If `svd_solver == "arpack"`, the number of components must be strictly less
          than the minimum of `n_features` and `n_samples`. Hence, the `None` case
          results in: `n_components == min(n_samples, n_features) - 1`.
    whiten : bool, default: False
        Whether to multiply the loading vectors by the square root of `n_samples` and
        then divide them by the singular values to ensure uncorrelated outputs with unit
        component-wise variances.

        Whitening will remove some information from the transformed signal (the relative
        variance scales of the components) but can sometimes improve the predictive
        accuracy of the downstream estimators by making their data respect some
        hard-wired assumptions.
    svd_solver : {"auto", "full", "covariance_eigh", "arpack", "randomized"}, \
            default: "auto"
        - `"auto"`: The solver is selected by a default policy based on data shape and
          `n_components`: if the input data has fewer than 1000 features and more than
          10 times as many samples, then the "covariance_eigh" solver is used.
          Otherwise, if the input data is larger than 500x500 and the number of
          components to extract is lower than 80% of the smallest dimension of the data,
          then the more efficient "randomized" method is selected. Otherwise the exact
          "full" SVD is computed and optionally truncated afterwards.
        - `"full"`: Run an exact full SVD calling the standard LAPACK solver via
          [`scipy.linalg.svd`][scipy.linalg.svd] and select the components by
          postprocessing.
        - `"covariance_eigh"`: Precompute the covariance matrix (on centered data), run
          a classical eigenvalue decomposition on the covariance matrix typically using
          LAPACK and select the components by postprocessing. This solver is very
          efficient for `n_samples >> n_features` and small `n_features`. It is,
          however, not tractable otherwise for large `n_features` (large memory
          footprint required to materialize the covariance matrix). Also note that
          compared to the "full" solver, this solver effectively doubles the condition
          number and is therefore less numerically stable (e.g. on input data with a
          large range of singular values).
        - `"arpack"`: Run SVD truncated to `n_components` calling ARPACK solver via
          [`scipy.sparse.linalg.svds`][scipy.sparse.linalg.svds]. It requires strictly
          `0 < n_components < min(X.shape)`.
        - `"randomized"`: Run randomized SVD by the method of Halko et al.

    tol : float, default: 0.0
        Tolerance for singular values computed by `svd_solver == "arpack"`. Must be of
        range `[0.0, infinity)`.
    iterated_power : int or "auto", default: "auto"
        Number of iterations for the power method computed by `svd_solver ==
        "randomized"`. Must be of range `[0, infinity)`.
    n_oversamples : int, default: 10
        This parameter is only relevant when `svd_solver="randomized"`. It corresponds
        to the additional number of random vectors to sample the range of `X` so as to
        ensure proper conditioning. See
        [`randomized_svd`][sklearn.utils.extmath.randomized_svd] for more details.
    power_iteration_normalizer : {"auto", "QR", "LU", "none"}, default: "auto"
        Power iteration normalizer for randomized SVD solver. Not used by ARPACK. See
        [`randomized_svd`][sklearn.utils.extmath.randomized_svd] for more details.
    random_state : int, optional
        Used when the `"arpack"` or `"randomized"` solvers are used. Pass an int for
        reproducible results across multiple function calls.
    mode : {"temporal", "spatial"}, default: "temporal"
        Whether to fit PCA along temporal or spatial orientation:

        - `"temporal"`: fit on `(time, voxels)`. The principal axes are
          spatial maps; the projected components are temporal time courses that capture
          dominant variance across voxels.
        - `"spatial"`: fit on `(voxels, time)`. The principal axes are time courses; the
          projected components are spatial maps that capture dominant variance across
          time.

    Attributes
    ----------
    maps_ : (n_components, ...) xarray.DataArray
        Principal directions in feature space (loadings), representing the axes of
        maximum variance. Equivalently, the right singular vectors of the centered input
        data, parallel to its eigenvectors. Sorted by decreasing `explained_variance_`
        and reshaped to the original spatial geometry.
    explained_variance_ : (n_components,) xarray.DataArray
        The amount of variance explained by each of the selected components. The
        variance estimation uses `n_samples - 1` degrees of freedom. Equal to
        `n_components` largest eigenvalues of the covariance matrix of `X`.
    explained_variance_ratio_ : (n_components,) xarray.DataArray
        Percentage of variance explained by each of the selected components. If
        `n_components` is not set then all components are stored and the sum of the
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
        is required to compute the estimated data covariance and score samples. Equal to
        the average of `min(n_features, n_samples) - n_components` smallest eigenvalues
        of the covariance matrix of `X`.
    n_features_in_ : int
        Number of features seen during fit.
    feature_names_in_ : (n_features_in_,) numpy.ndarray
        Feature names seen during fit. Defined only when flattened feature labels are
        all strings.

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> from confusius.decomposition import PCA
    >>>
    >>> rng = np.random.default_rng(0)
    >>> data = xr.DataArray(
    ...     rng.standard_normal((200, 5, 10, 20)),
    ...     dims=["time", "z", "y", "x"],
    ... )
    >>>
    >>> pca = PCA(n_components=5, random_state=0)
    >>> signals = pca.fit_transform(data)
    >>> signals.dims
    ('time', 'component')
    >>> reconstructed = pca.inverse_transform(signals)
    >>> reconstructed.dims
    ('time', 'z', 'y', 'x')

    References
    ----------
    [^1]:
        Halko, N., Martinsson, P. G., and Tropp, J. A. (2011). "Finding structure with
        randomness: Probabilistic algorithms for constructing approximate matrix
        decompositions". SIAM Review, 53(2), 217-288.
    """

    _signals_long_name = "PCA signals"

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
        mode: Literal["temporal", "spatial"] = "temporal",
    ) -> None:
        self.n_components = n_components
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.n_oversamples = n_oversamples
        self.power_iteration_normalizer = power_iteration_normalizer
        self.random_state = random_state
        self.mode = mode

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
        ValueError
            If `mode` is not `"temporal"` or `"spatial"`.
        """
        del y

        if self.mode not in {"temporal", "spatial"}:
            raise ValueError(
                f"mode must be 'temporal' or 'spatial', got '{self.mode}'."
            )

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

        self._store_fit_metadata(X, X_proc, X_stacked, spatial_dims)

        if self.mode == "temporal":
            self._fit_temporal(pca, X_proc)
        else:
            self._fit_spatial(pca, X_proc)

        self._store_feature_names(X)
        return self

    def _fit_temporal(
        self,
        pca: _SklearnPCA,
        X_proc: npt.NDArray[np.floating],
    ) -> None:
        """Fit PCA in temporal orientation `(time, voxel)`."""
        pca.fit(X_proc)

        component_coord = np.arange(pca.components_.shape[0], dtype=np.intp)
        self.maps_ = self._reshape_component_matrix(
            pca.components_,
            component_coord,
            long_name="Principal component maps",
        )
        self.mean_ = self._reshape_mean(pca.mean_)

        self.explained_variance_ = xr.DataArray(
            pca.explained_variance_,
            dims=["component"],
            coords={"component": component_coord},
        )
        self.explained_variance_ratio_ = xr.DataArray(
            pca.explained_variance_ratio_,
            dims=["component"],
            coords={"component": component_coord},
        )
        self.singular_values_ = xr.DataArray(
            pca.singular_values_,
            dims=["component"],
            coords={"component": component_coord},
        )

        self.n_components_ = int(pca.n_components_)
        self.n_samples_ = int(X_proc.shape[0])
        self.noise_variance_ = float(pca.noise_variance_)
        self._estimator = pca

    def _fit_spatial(
        self,
        pca: _SklearnPCA,
        X_proc: npt.NDArray[np.floating],
    ) -> None:
        """Fit PCA in spatial orientation `(voxel, time)`."""
        pca.fit(X_proc.T)

        spatial_maps_flat: npt.NDArray[np.floating] = pca.transform(X_proc.T).T
        voxel_mean: npt.NDArray[np.floating] = X_proc.mean(axis=0)

        component_coord = np.arange(spatial_maps_flat.shape[0], dtype=np.intp)
        self.maps_ = self._reshape_component_matrix(
            spatial_maps_flat,
            component_coord,
            long_name="Principal component maps",
        )
        self.mean_ = self._reshape_mean(voxel_mean)

        self.explained_variance_ = xr.DataArray(
            pca.explained_variance_,
            dims=["component"],
            coords={"component": component_coord},
        )
        self.explained_variance_ratio_ = xr.DataArray(
            pca.explained_variance_ratio_,
            dims=["component"],
            coords={"component": component_coord},
        )
        self.singular_values_ = xr.DataArray(
            pca.singular_values_,
            dims=["component"],
            coords={"component": component_coord},
        )

        self.n_components_ = int(spatial_maps_flat.shape[0])
        self.n_samples_ = int(X_proc.shape[0])
        self.noise_variance_ = float(pca.noise_variance_)
        self._spatial_components_flat_ = spatial_maps_flat
        self._spatial_feature_mean_ = voxel_mean
        self._estimator = pca

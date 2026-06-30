"""Non-negative matrix factorization for `(time, ...)` fUSI DataArrays."""

from typing import Literal

import numpy as np
import numpy.typing as npt
import xarray as xr
from sklearn.decomposition import NMF as _SklearnNMF

from confusius.decomposition._base import _BaseFUSIDecomposer


class NMF(_BaseFUSIDecomposer):
    r"""Non-negative matrix factorization (NMF) for fUSI data.

    Find two non-negative matrices, i.e. matrices with all non-negative elements,
    (`W`, `H`) whose product approximates the non-negative matrix `X`. This
    factorization can be used for example for dimensionality reduction, source
    separation, or topic extraction.

    The objective function is:

    $$
        \begin{aligned}
        L(W, H) &= 0.5 * ||X - WH||_{\text{loss}}^2 \\
                &+ \alpha_W \, l_{1,\text{ratio}} \, n_{\text{features}} \, ||\mathrm{vec}(W)||_1 \\
                &+ \alpha_H \, l_{1,\text{ratio}} \, n_{\text{samples}} \, ||\mathrm{vec}(H)||_1 \\
                &+ 0.5 \, \alpha_W \, (1 - l_{1,\text{ratio}}) \, n_{\text{features}} \, ||W||_{F}^2 \\
                &+ 0.5 \, \alpha_H \, (1 - l_{1,\text{ratio}}) \, n_{\text{samples}} \, ||H||_{F}^2,
        \end{aligned}
    $$

    where $||A||_{F}^2 = \sum_{i,j} A_{ij}^2$ is the Frobenius norm and $||vec(A)||_1 =
    \sum_{i,j} abs(A_{ij})$ is the elementwise L1 norm.

    The generic norm `||X - WH||_{\text{loss}}` may represent the Frobenius norm or
    another supported beta-divergence loss. The regularization terms are scaled by
    `n_features` for `W` and by `n_samples` for `H` to keep their impact balanced with
    respect to one another and to the data fit term.

    This estimator wraps [`sklearn.decomposition.NMF`][sklearn.decomposition.NMF]
    while keeping xarray metadata through
    [`transform`][confusius.decomposition.NMF.transform] and
    [`inverse_transform`][confusius.decomposition.NMF.inverse_transform]:

    - Input data are expected as `(time, ...)` where `...` are spatial dimensions.
    - `transform` returns a `(time, component)` DataArray.
    - `inverse_transform` reconstructs to `(time, ...)` using the fitted spatial
      coordinates.

    Parameters
    ----------
    n_components : int or {"auto"} or None, default: "auto"
        Number of components. If `None`, all features are kept. If
        `n_components="auto"`, the number of components is automatically inferred from
        `W` or `H` shapes.
    init : {"random", "nndsvd", "nndsvda", "nndsvdar"}, optional
        Method used to initialize the procedure.

        - `None`: `"nndsvda"` if `n_components <= min(n_samples, n_features)`,
          otherwise `"random"`.
        - `"random"`: non-negative random matrices, scaled with
          `sqrt(X.mean() / n_components)`.
        - `"nndsvd"`: non-negative double singular value decomposition (NNDSVD)
          initialization, better for sparseness.
        - `"nndsvda"`: NNDSVD with zeros filled with the average of `X`, better when
          sparsity is not desired.
        - `"nndsvdar"`: NNDSVD with zeros filled with small random values, generally
          faster but less accurate than `"nndsvda"` when sparsity is not desired.

    solver : {"cd", "mu"}, default: "cd"
        Numerical solver to use:

        - `"cd"`: coordinate descent solver.
        - `"mu"`: multiplicative update solver.

    beta_loss : float or {"frobenius", "kullback-leibler", "itakura-saito"}, \
            default: "frobenius"
        Beta divergence to be minimized, measuring the distance between `X` and the dot
        product `WH`. Note that values different from `"frobenius"` (or `2`) and
        `"kullback-leibler"` (or `1`) lead to significantly slower fits. Note that for
        `beta_loss <= 0` (or `"itakura-saito"`), the input matrix `X` cannot contain
        zeros. Used only in the `"mu"` solver.
    tol : float, default: 1e-4
        Tolerance of the stopping condition.
    max_iter : int, default: 200
        Maximum number of iterations before timing out.
    random_state : int or numpy.random.RandomState, optional
        Used for initialisation (when `init == "nndsvdar"` or `"random"`) and in
        coordinate descent. Pass an int for reproducible results across multiple
        function calls.
    alpha_W : float, default: 0.0
        Constant that multiplies the regularization terms of `W`. Set it to zero to have
        no regularization on `W`.
    alpha_H : float or "same", default: "same"
        Constant that multiplies the regularization terms of `H`. Set it to zero to have
        no regularization on `H`. If `"same"`, it takes the same value as `alpha_W`.
    l1_ratio : float, default: 0.0
        The regularization mixing parameter, with `0 <= l1_ratio <= 1`.

        - For `l1_ratio = 0`, the penalty is an elementwise L2 penalty (aka
          Frobenius Norm).
        - For `l1_ratio = 1`, it is an elementwise L1 penalty.
        - For `0 < l1_ratio < 1`, the penalty is a combination of L1 and L2.

    verbose : int, default: 0
        Whether to be verbose.
    shuffle : bool, default: False
        Whether to randomize the order of coordinates in the coordinate descent solver.
    mode : {"temporal", "spatial"}, default: "temporal"
        Whether to fit NMF along temporal or spatial orientation:

        - `"temporal"`: fit on `(time, voxels)`. The transformed data `W` are
          non-negative temporal time courses and the components matrix `H` reshapes to
          spatial maps.
        - `"spatial"`: fit on `(voxels, time)`. In the underlying sklearn fit, `W` lives
          on voxels and `H` lives on timepoints because the data matrix is transposed.
          This wrapper still exposes `maps_` as spatial maps and `transform` as `(time,
          component)` signals.

    mask : xarray.DataArray, optional
        Boolean spatial mask selecting voxels to include during fitting and projection.
        Must match the spatial dimensions and coordinates of the input data.

    Attributes
    ----------
    maps_ : (n_components, ...) xarray.DataArray
        Non-negative component maps reshaped to the original spatial geometry.

        - In `"temporal"` mode: the components matrix `H`, interpreted as spatial maps.
        - In `"spatial"` mode: spatial projection maps derived from the fitted NMF model
          on transposed data.
    n_components_ : int
        The number of components. It is the same as the `n_components` parameter if it
        was given. Otherwise, it is inferred from the fitted factorization.
    reconstruction_err_ : float
        Frobenius norm of the matrix difference, or beta-divergence, between the
        training data `X` and the reconstructed data `WH` from the fitted model.
    n_iter_ : int
        Actual number of iterations.
    n_features_in_ : int
        Number of features seen during fit.
    feature_names_in_ : (n_features_in_,) numpy.ndarray
        Names of features seen during fit. Defined only when flattened feature
        labels are all strings.

    Notes
    -----
    In sklearn's notation, the transformed data are named `W` and the components matrix
    is named `H`. In much of the NMF literature the convention is often the opposite
    because the data matrix is transposed.

    In `"spatial"` mode, the estimator is still fitted with `sklearn.decomposition.NMF`,
    but on the transposed data matrix `(voxels, time)`. The wrapper then computes
    `(time, component)` signals by projecting the original `(time, voxel)` matrix onto
    the fitted spatial maps, so that `transform` and `inverse_transform` keep the same
    public API in both modes.

    References
    ----------
    [^1]:
        Cichocki, A., and Anh-Huy, P. H. A. N. (2009). "Fast local algorithms
        for large scale nonnegative matrix and tensor factorizations". IEICE
        Transactions on Fundamentals of Electronics, Communications and Computer
        Sciences, 92(3), 708-721.

    [^2]:
        Fevotte, C., and Idier, J. (2011). "Algorithms for nonnegative matrix
        factorization with the beta-divergence". Neural Computation, 23(9).

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> from confusius.decomposition import NMF
    >>>
    >>> rng = np.random.default_rng(0)
    >>> k = 5
    >>> n_t, n_z, n_y, n_x = 200, 4, 6, 8
    >>> temporal = rng.random((n_t, k))
    >>> spatial = rng.random((k, n_z * n_y * n_x))
    >>> data = xr.DataArray(
    ...     (10.0 * (temporal @ spatial) + 1.0).reshape(n_t, n_z, n_y, n_x),
    ...     dims=["time", "z", "y", "x"],
    ... )
    >>>
    >>> nmf = NMF(n_components=k, init="nndsvda", random_state=0)
    >>> signals = nmf.fit_transform(data)
    >>> signals.dims
    ('time', 'component')
    >>> reconstructed = nmf.inverse_transform(signals)
    >>> reconstructed.dims
    ('time', 'z', 'y', 'x')
    """

    _signals_long_name = "NMF signals"

    def __init__(
        self,
        *,
        n_components: int | Literal["auto"] | None = "auto",
        init: Literal["random", "nndsvd", "nndsvda", "nndsvdar"] | None = None,
        solver: Literal["cd", "mu"] = "cd",
        beta_loss: float | Literal["frobenius", "kullback-leibler", "itakura-saito"] = (
            "frobenius"
        ),
        tol: float = 1e-4,
        max_iter: int = 200,
        random_state: int | np.random.RandomState | None = None,
        alpha_W: float = 0.0,
        alpha_H: float | Literal["same"] = "same",
        l1_ratio: float = 0.0,
        verbose: int = 0,
        shuffle: bool = False,
        mode: Literal["temporal", "spatial"] = "temporal",
        mask: xr.DataArray | None = None,
    ) -> None:
        self.n_components = n_components
        self.init = init
        self.solver = solver
        self.beta_loss = beta_loss
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.alpha_W = alpha_W
        self.alpha_H = alpha_H
        self.l1_ratio = l1_ratio
        self.verbose = verbose
        self.shuffle = shuffle
        self.mode = mode
        self.mask = mask

    def fit(self, X: xr.DataArray, y: None = None) -> "NMF":
        """Fit NMF on `(time, ...)` fUSI data.

        Parameters
        ----------
        X : (time, ...) xarray.DataArray
            Input fUSI data. Must be non-negative.
        y : None, optional
            Ignored. Present for scikit-learn API compatibility.

        Returns
        -------
        NMF
            Fitted estimator.

        Raises
        ------
        ValueError
            If input has no `time` dimension, fewer than 2 timepoints, no spatial
            dimensions, contains negative values, or `mode` is not `"temporal"` or
            `"spatial"`.
        """
        del y

        if self.mode not in {"temporal", "spatial"}:
            raise ValueError(
                f"mode must be 'temporal' or 'spatial', got '{self.mode}'."
            )

        X_proc, spatial_dims, feature_mask = self._prepare_data(
            X,
            check_layout=False,
            operation_name="NMF.fit",
        )

        if np.any(X_proc < 0.0):
            raise ValueError(
                "NMF requires non-negative input data, but negative values were "
                f"found along dimensions {spatial_dims}. NMF is typically applied "
                "to Power Doppler or other non-negative fUSI signals."
            )

        nmf = _SklearnNMF(
            n_components=self.n_components,
            init=self.init,
            solver=self.solver,
            beta_loss=self.beta_loss,
            tol=self.tol,
            max_iter=self.max_iter,
            random_state=self.random_state,
            alpha_W=self.alpha_W,
            alpha_H=self.alpha_H,
            l1_ratio=self.l1_ratio,
            verbose=self.verbose,
            shuffle=self.shuffle,
        )

        self._store_fit_metadata(X, X_proc, spatial_dims, feature_mask)

        if self.mode == "temporal":
            self._fit_temporal(nmf, X_proc)
        else:
            self._fit_spatial(nmf, X_proc)

        self._store_feature_names(X)
        return self

    def _fit_temporal(
        self,
        nmf: _SklearnNMF,
        X_proc: npt.NDArray[np.floating],
    ) -> None:
        """Fit NMF in temporal orientation `(time, voxel)`."""
        nmf.fit(X_proc)

        component_coord = np.arange(nmf.components_.shape[0], dtype=np.intp)
        self.maps_ = self._reshape_component_matrix(
            nmf.components_,
            component_coord,
            long_name="NMF spatial maps",
        )

        self.n_components_ = int(nmf.n_components_)
        self.n_iter_ = int(nmf.n_iter_)
        self.reconstruction_err_ = float(nmf.reconstruction_err_)
        self._estimator = nmf

    def _fit_spatial(
        self,
        nmf: _SklearnNMF,
        X_proc: npt.NDArray[np.floating],
    ) -> None:
        """Fit NMF in spatial orientation `(voxel, time)`."""
        nmf.fit(X_proc.T)

        spatial_maps_flat: npt.NDArray[np.floating] = nmf.transform(X_proc.T).T
        voxel_mean: npt.NDArray[np.floating] = X_proc.mean(axis=0)

        component_coord = np.arange(spatial_maps_flat.shape[0], dtype=np.intp)
        self.maps_ = self._reshape_component_matrix(
            spatial_maps_flat,
            component_coord,
            long_name="NMF spatial maps",
        )
        self.mean_ = self._reshape_mean(voxel_mean)

        self.n_components_ = int(spatial_maps_flat.shape[0])
        self.n_iter_ = int(nmf.n_iter_)
        self.reconstruction_err_ = float(nmf.reconstruction_err_)
        self._spatial_components_flat_ = spatial_maps_flat
        self._spatial_feature_mean_ = voxel_mean
        self._estimator = nmf

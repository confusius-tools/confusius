"""Non-negative matrix factorization for `(time, ...)` fUSI DataArrays."""

from typing import Literal

import numpy as np
import numpy.typing as npt
import xarray as xr
from sklearn.decomposition import NMF as _SklearnNMF

from confusius.decomposition._base import _BaseFUSIDecomposer


class NMF(_BaseFUSIDecomposer):
    """Non-negative matrix factorization (NMF) for fUSI data.

    Linear dimensionality reduction that factorizes a non-negative data matrix into two
    non-negative factors: a dictionary of spatial maps (`maps_`) and their associated
    non-negative temporal signals. The decomposition is computed by minimizing the
    Frobenius norm (or a beta-divergence, depending on the `beta_loss` parameter)
    between the input data and the reconstructed product of the two factors.

    This estimator wraps [`sklearn.decomposition.NMF`][sklearn.decomposition.NMF] while
    keeping xarray metadata through [`transform`][confusius.decomposition.NMF.transform]
    and [`inverse_transform`][confusius.decomposition.NMF.inverse_transform]:

    - Input data are expected as `(time, ...)` where `...` are spatial dimensions.
    - `transform` returns a `(time, component)` DataArray.
    - `inverse_transform` reconstructs to `(time, ...)` using the fitted spatial
      coordinates.

    Parameters
    ----------
    n_components : int, default: "auto"
        Number of components to keep. If not set, `n_components == min(n_samples,
        n_features)`. Note that NMF does not support a fractional `n_components` or
        `"mle"`: it must be an integer or `"auto"`.
    init : {"nndsvda", "nndsvdar", "random"}, optional
        Method used to initialize the procedure. If not set, lets sklearn pick
        `"nndsvda"` for non-negative sparse data and `"random"` otherwise.
    solver : {"cd", "mu"}, default: "cd"
        Numerical solver to use: `"cd"` is a Coordinate Descent solver (only one
        supporting beta-loss different from `"frobenius"`) while `"mu"` is a
        Multiplicative Update solver.
    beta_loss : float or {"frobenius", "kullback-leibler", "itakura-saito"}, \
            default: "frobenius"
        Beta divergence to be minimized, measuring the distance between `X` and the
        dot product `WH`. Note that values different from `"frobenius"` (or
        `"kullback-leibler"`, `"itakura-saito"` with beta values equal to 1 or 2
        respectively) are not strictly beta divergences and may lead to significantly
        slower convergence. See details in the [`sklearn` user guide][sklearn-nmf].
    tol : float, default: 1e-4
        Tolerance of the stopping condition.
    max_iter : int, default: 200
        Maximum number of iterations before timing out.
    random_state : int, optional
        Used for initialisation when `init == "random"` and for `"nndsvdar"`. Pass an
        int for reproducible results across multiple function calls.
    alpha_W : float, default: 0.0
        Constant that multiplies the regularization terms of `W`. Set it to zero (the
        default) to have no regularization on `W`.
    alpha_H : float or "same", default: "same"
        Constant that multiplies the regularization terms of `H`. If `"same"` (the
        default), the value depends on `alpha_W`: if `alpha_W == 0`, `alpha_H == 0`,
        else `alpha_H == alpha_W`.
    l1_ratio : float, default: 0.0
        The regularization mixing parameter, with `0 <= l1_ratio <= 1`. For `l1_ratio
        == 0` the penalty is an elementwise L2 penalty (aka Frobenius Norm). For
        `l1_ratio == 1` it is an elementwise L1 penalty. For `0 < l1_ratio < 1`, the
        penalty is a combination of L1 and L2.
    mode : {"temporal", "spatial"}, default: "temporal"
        Whether to fit NMF along temporal or spatial orientation:

        - `"temporal"`: fit on `(time, voxels)`. The dictionaries are spatial maps; the
          component signals are non-negative temporal time courses.
        - `"spatial"`: fit on `(voxels, time)`. The dictionaries are non-negative time
          courses; the component signals are spatial maps.
    mask : xarray.DataArray, optional
        Boolean spatial mask selecting voxels to include during fitting and projection.
        Must match the spatial dimensions and coordinates of the input data.
    verbose : int, default: 0
        Whether to be verbose.

    Attributes
    ----------
    maps_ : (n_components, ...) xarray.DataArray
        Non-negative dictionary elements, sorted by decreasing reconstruction
        contribution. In `"temporal"` mode these are spatial maps; in `"spatial"` mode
        these are time courses. Reshaped to the fitted spatial geometry.
    n_components_ : int
        The estimated number of components. When `n_components` is `"auto"`, this is
        `min(n_samples, n_features)`.
    n_iter_ : int
        Actual number of iterations run by the solver.
    reconstruction_err_ : float
        Frobenius norm of the matrix difference between the training data and the
        reconstructed data from the fit, equal to `||X - WH||_F`.
    n_features_in_ : int
        Number of features seen during fit.
    feature_names_in_ : (n_features_in_,) numpy.ndarray
        Feature names seen during fit. Defined only when flattened feature labels are
        all strings.

    Notes
    -----
    NMF requires strictly non-negative input data. fUSI Power Doppler signals are
    non-negative by construction. The wrapper raises a clear `ValueError` if the
    (masked) input data contains negative values.

    The `"spatial"` mode fits NMF on the transposed data and exposes a manual
    projection rather than relying on `sklearn.decomposition.NMF.transform`, because
    `sklearn`'s NMF `transform` is only defined for `(n_samples, n_features)` input.
    As a consequence, `transform` and `inverse_transform` do not call into
    `sklearn.decomposition.NMF` in `"spatial"` mode.

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
        n_components: int | Literal["auto"] = "auto",
        init: Literal["nndsvda", "nndsvdar", "random"] | None = None,
        solver: Literal["cd", "mu"] = "cd",
        beta_loss: float | Literal["frobenius", "kullback-leibler", "itakura-saito"] = (
            "frobenius"
        ),
        tol: float = 1e-4,
        max_iter: int = 200,
        random_state: int | None = None,
        alpha_W: float = 0.0,
        alpha_H: float | Literal["same"] = "same",
        l1_ratio: float = 0.0,
        mode: Literal["temporal", "spatial"] = "temporal",
        mask: xr.DataArray | None = None,
        verbose: int = 0,
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
        self.mode = mode
        self.mask = mask
        self.verbose = verbose

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

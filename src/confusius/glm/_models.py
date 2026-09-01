"""Core regression models for GLM analysis.

This module implements Ordinary Least Squares (OLS) and Autoregressive (AR) regression
models for fUSI GLM analysis. Adapted from nilearn.glm.regression.

Portions of this file are derived from Nilearn, which is licensed under the BSD-3-Clause
License. See `NOTICE` file for details.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.linalg as spl

if TYPE_CHECKING:
    import numpy.typing as npt


def _positive_reciprocal(x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Return element-wise reciprocal, but replace 1 / 0 with 0.

    Parameters
    ----------
    x : (n,) or (n, m) numpy.ndarray
        Input array.

    Returns
    -------
    (n,) or (n, m) numpy.ndarray
        Element-wise reciprocal with zeros preserved as zeros.
    """
    return np.where(x == 0, 0, 1.0 / x)


def _sum_of_squares(a: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
    """Return the sum of squares down each column, accumulated in float64.

    Faster than `numpy.sum(a**2, axis=0)`, which materializes the squared array, and
    exact enough to subtract from: fUSI data is often float32, and a float32
    accumulation costs ~1e-5 relative error once the result feeds a difference of
    two large sums of squares.

    Parameters
    ----------
    a : (n_timepoints, n_voxels) numpy.ndarray
        Array to reduce along its first axis.

    Returns
    -------
    (n_voxels,) numpy.ndarray
        Sum of the squared entries of each column.
    """
    return np.einsum("tv,tv->v", a, a, dtype=np.float64)


class OLSModel:
    """Ordinary Least Squares regression model.

    Implements standard OLS regression for GLM analysis. The model is specified with a
    design matrix and fit using the [`fit`][confusius.glm._models.OLSModel] method.

    This implementation is adapted from
    [`nilearn.glm.regression.OLSModel`][nilearn.glm.regression.OLSModel].

    Parameters
    ----------
    design : (n_timepoints, n_regressors) numpy.ndarray
        Design matrix.

    Attributes
    ----------
    design : (n_timepoints, n_regressors) numpy.ndarray
        The design matrix.
    whitened_design : (n_timepoints, n_regressors) numpy.ndarray
        Whitened design matrix (same as design for OLS).
    pinv_design : (n_regressors, n_timepoints) numpy.ndarray
        Moore-Penrose pseudoinverse of the whitened design.
    normalized_cov : (n_regressors, n_regressors) numpy.ndarray
        Parameter covariance up to the dispersion scale, `pinv_design @ pinv_design.T`.
        Per-voxel covariance is recovered as `dispersion[v] * normalized_cov`.
    df_total : int
        Total degrees of freedom (n_timepoints).
    df_model : int
        Model degrees of freedom (rank of design).
    df_residuals : int
        Residual degrees of freedom (`n_timepoints - rank`).

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.randn(100, 3)
    >>> y = np.random.randn(100, 10)
    >>> model = OLSModel(X)
    >>> results = model.fit(y)
    >>> results.theta.shape
    (3, 10)
    """

    def __init__(self, design: npt.NDArray[np.floating]) -> None:
        """Initialize OLS model with design matrix."""
        self.design = design

        # Whitening is identity for OLS
        self.whitened_design = self.whiten(self.design)

        # Moore-Penrose pseudoinverse: solves `beta = pinv_design @ Y`.
        self.pinv_design = spl.pinv(self.whitened_design)

        # Parameter covariance up to dispersion: `(X^T X)^-1` for OLS.
        self.normalized_cov = self.pinv_design @ self.pinv_design.T

        # Degrees of freedom
        self.df_total = self.whitened_design.shape[0]

        # Compute rank with tolerance
        eps = np.abs(self.design).sum() * np.finfo(np.float64).eps
        self.df_model = np.linalg.matrix_rank(self.design, eps)
        self.df_residuals = self.df_total - self.df_model

    def whiten(self, X: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Apply whitening transformation.

        For OLS, this is the identity transformation.

        Parameters
        ----------
        X : (n_timepoints, ...) numpy.ndarray
            Array to whiten.

        Returns
        -------
        (n_timepoints, ...) numpy.ndarray
            Whitened array (unchanged for OLS).
        """
        return X

    def fit(
        self, Y: npt.NDArray[np.floating], *, keep_residuals: bool = True
    ) -> RegressionResults:
        """Fit the model to data.

        Parameters
        ----------
        Y : (n_timepoints, n_voxels) numpy.ndarray
            Data with time in rows, voxels/features in columns.
        keep_residuals : bool, default: True
            Whether to keep the whitened residuals on the results. When `False` the
            residual sum of squares is read off the normal equations instead, which
            avoids allocating a second `(n_timepoints, n_voxels)` array, and
            `whitened_residuals` is left unset so `sse`/`mse`/`residuals` raise.

        Returns
        -------
        RegressionResults
            Fitted model results containing parameter estimates, residuals, and
            statistics.
        """
        whitened_Y = self.whiten(Y)

        # Estimate parameters: beta = (X^T X)^-1 X^T Y.
        beta = self.pinv_design @ whitened_Y

        whitened_residuals: npt.NDArray[np.floating] | None = None
        if keep_residuals:
            whitened_residuals = whitened_Y - self.whitened_design @ beta
            rss = _sum_of_squares(whitened_residuals)
        else:
            # `beta` solves the normal equations `X'X b = X'Y`, so the residual sum
            # of squares collapses to `||Y||² - b·(X'Y)` and never needs the residual
            # array. Both terms are large and close for the low-R² data a GLM
            # usually sees, hence the float64 accumulation inside
            # `_sum_of_squares`/`einsum`.
            rss = _sum_of_squares(whitened_Y) - np.einsum(
                "kv,kv->v", beta, self.whitened_design.T @ whitened_Y, dtype=np.float64
            )

        # Estimate dispersion (variance): sigma^2 = RSS / (n - p).
        if self.df_residuals > 0:
            dispersion = rss / self.df_residuals
        else:
            dispersion = np.full_like(rss, np.nan)

        return RegressionResults(
            theta=beta,
            Y=Y,
            model=self,
            whitened_Y=whitened_Y,
            whitened_residuals=whitened_residuals,
            dispersion=dispersion,
            cov=self.normalized_cov,
        )


class ARModel:
    """Autoregressive regression model with per-voxel AR(p) covariance structure.

    Fits an AR-whitened OLS model independently for each voxel. Each voxel's time series
    is whitened with its own AR coefficients before solving the normal equations.

    This implementation is adapted from [`nilearn.glm.regression.ARModel`][
    nilearn.glm.regression.ARModel].

    Parameters
    ----------
    design : (n_timepoints, n_regressors) numpy.ndarray
        Design matrix.
    rho : (order, n_voxels) numpy.ndarray
        Per-voxel AR coefficients. Row `i` contains the AR(i+1) coefficients for every
        voxel.

    Attributes
    ----------
    design : (n_timepoints, n_regressors) numpy.ndarray
        Design matrix.
    rho : (order, n_voxels) numpy.ndarray
        Per-voxel AR coefficients.
    order : int
        AR model order.
    df_model : int
        Model degrees of freedom (rank of design matrix).
    df_total : int
        Total degrees of freedom (n_timepoints).
    df_residuals : int
        Residual degrees of freedom.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.randn(100, 3)
    >>> y = np.random.randn(100, 10)
    >>> rho = np.full((1, 10), 0.3)  # AR(1) with coef 0.3 for each of 10 voxels
    >>> model = ARModel(X, rho=rho)
    >>> results = model.fit(y)
    """

    def __init__(
        self,
        design: npt.NDArray[np.floating],
        rho: npt.NDArray[np.floating],
    ) -> None:
        """Initialize AR model with design and per-voxel AR coefficients."""
        self.design = design
        rho = np.asarray(rho, dtype=np.float64)
        if rho.ndim != 2:
            raise ValueError(
                f"rho must be 2D with shape (order, n_voxels), got shape {rho.shape}"
            )
        self.rho = rho
        self.order = rho.shape[0]

        eps = np.abs(design).sum() * np.finfo(np.float64).eps
        self.df_model = np.linalg.matrix_rank(design, eps)
        self.df_total = design.shape[0]
        self.df_residuals = self.df_total - self.df_model

    def whiten(self, Y: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
        """Apply per-voxel AR(p) whitening to data.

        Unlike `OLSModel.whiten`, this operates on `(n_timepoints, n_voxels)` data only:
        the design matrix is handled implicitly via cross-product expansion in `fit`.

        Parameters
        ----------
        Y : (n_timepoints, n_voxels) numpy.ndarray
            Data to whiten.

        Returns
        -------
        (n_timepoints, n_voxels) numpy.ndarray
            Whitened data.
        """
        whitened = Y.copy().astype(np.float64)
        for i in range(self.order):
            whitened[(i + 1) :] -= self.rho[i] * Y[: -(i + 1)]
        return whitened

    def fit(
        self, Y: npt.NDArray[np.floating], *, keep_residuals: bool = True
    ) -> RegressionResults:
        """Fit the AR model per voxel.

        Each voxel's time series and the shared design matrix are whitened using that
        voxel's own AR coefficients. The normal equations are then solved in batch over
        all voxels.

        Parameters
        ----------
        Y : (n_timepoints, n_voxels) numpy.ndarray
            Data with time in rows, voxels in columns.
        keep_residuals : bool, default: True
            Whether to keep the whitened residuals on the results. When `False` the
            residual sum of squares is read off the normal equations instead, which
            avoids allocating a second `(n_timepoints, n_voxels)` array, and
            `whitened_residuals` is left unset so `sse`/`mse`/`residuals` raise.

        Returns
        -------
        RegressionResults
            Fitted results. The `cov` attribute has shape `(n_voxels, n_regressors,
            n_regressors)`: a per-voxel normalized covariance matrix.

        Raises
        ------
        ValueError
            If the number of voxels in `rho` does not match `Y`.
        """
        T, K = self.design.shape
        V = Y.shape[1]

        if self.rho.shape[1] != V:
            raise ValueError(
                f"rho has {self.rho.shape[1]} voxels but Y has {V} voxels."
            )

        whitened_Y = self.whiten(Y)

        # Avoid materializing per-voxel whitened designs of shape (V, T, K):
        # for large V and T that easily exhausts memory. Instead express each
        # `whitened_X[v] = X - sum_i rho[i,v] * L[i]`, where L[i] is the
        # i-step lag of X, and accumulate `XtX[v]` and `XtY[v]` from
        # design-only cross-products (A, B, C) and Y-coupled cross-products
        # (P, Q) — all small.

        X = self.design.astype(np.float64)

        # L[i, t, :] = X[t - (i + 1), :] for t >= i + 1, else 0. (order, T, K).
        L = np.zeros((self.order, T, K))
        for i in range(self.order):
            L[i, (i + 1) :, :] = X[: -(i + 1), :]

        # A = X.T @ X — Gram of the unwhitened design. (K, K).
        A = X.T @ X
        # B[i] = X.T @ L[i] — cross of X with the i-th lag. (order, K, K).
        B = np.einsum("tk,itl->ikl", X, L)
        # C[i, j] = L[i].T @ L[j] — Gram of the lags. (order, order, K, K).
        C = np.einsum("itk,jtl->ijkl", L, L)

        B_sym = B + B.transpose(0, 2, 1)  # B[i] + B[i].T for each lag.

        # `XtX[v] = A - sum_i rho[i,v]·B_sym[i] + sum_{ij} rho[i,v]·rho[j,v]·C[i,j]`.
        # The three-operand term needs `optimize=True` to contract pairwise; einsum's
        # default builds the whole `(order, order, V, K, K)` intermediate, which costs
        # ~6x more at AR(2) and grows with the order. The two-operand term is faster
        # without it.
        XtX = (
            A[np.newaxis]
            - np.einsum("iv,ikl->vkl", self.rho, B_sym)
            + np.einsum("iv,jv,ijkl->vkl", self.rho, self.rho, C, optimize=True)
        )  # (V, K, K)

        # P = X.T @ whitened_Y — clean cross of X with whitened Y. (K, V).
        P = X.T @ whitened_Y
        # Q[i] = L[i].T @ whitened_Y — lagged cross with whitened Y. (order, K, V).
        # `optimize=True` routes this through BLAS instead of einsum's own loop; on a
        # (600, 15) design and 147k voxels that is ~30x faster.
        Q = np.einsum("itk,tv->ikv", L, whitened_Y, optimize=True)
        # `XtY[v] = P[:, v] - sum_i rho[i,v] · Q[i, :, v]`.
        XtY = P.T - np.einsum("iv,ikv->vk", self.rho, Q)  # (V, K)

        # Per-voxel normalized covariance and beta. `pinv` is the general answer: a
        # rank-deficient design gives a min-norm solution instead of a `LinAlgError`
        # (matching `OLSModel`'s `spl.pinv` and nilearn's single-rho `ARModel`, which
        # inherits OLS's pinv). But `pinv` is a batched SVD over V matrices and
        # dominates the fit -- ~6x the cost of the LU-based `inv` on a (147k, 15, 15)
        # stack, for results that agree to ~1e-14. A full-rank design keeps every
        # `XtX[v]` invertible except for pathological rho, so take `inv` there and keep
        # `pinv` for the rank-deficient case, where `inv` would silently return garbage
        # rather than raise.
        if self.df_model == K:
            try:
                cov_beta = np.linalg.inv(XtX)  # (V, K, K)
            except np.linalg.LinAlgError:
                cov_beta = np.linalg.pinv(XtX)
        else:
            cov_beta = np.linalg.pinv(XtX)  # (V, K, K)
        # beta: (V, K) transposed to (K, V) to match convention.
        beta = np.einsum("vij,vj->vi", cov_beta, XtY).T

        wresid: npt.NDArray[np.floating] | None = None
        if keep_residuals:
            # Whitened residuals: wresid = wY - wX @ beta, computed without wX.
            # wX[v] @ beta[:,v] = X @ beta[:,v] - sum_i rho[i,v] * L[i] @ beta[:,v]
            wresid = whitened_Y - X @ beta
            for i in range(self.order):
                wresid += self.rho[i] * (L[i] @ beta)  # rho[i]: (V,), L[i]@beta: (T,V)
            rss = _sum_of_squares(wresid)
        else:
            # `beta[v]` solves `XtX[v] b = XtY[v]`, so per voxel the residual sum of
            # squares collapses to `||wY||² - b·XtY`, as in `OLSModel.fit`. This skips
            # the two `(n_timepoints, n_voxels)` products above, which otherwise
            # dominate the fit once the pseudoinverse is gone.
            rss = _sum_of_squares(whitened_Y) - np.einsum(
                "kv,vk->v", beta, XtY, dtype=np.float64
            )

        if self.df_residuals > 0:
            dispersion: npt.NDArray[np.float64] = rss / self.df_residuals
        else:
            dispersion = np.full(V, np.nan)

        return RegressionResults(
            theta=beta,
            Y=Y,
            model=self,
            whitened_Y=whitened_Y,
            whitened_residuals=wresid,
            dispersion=dispersion,
            cov=cov_beta,
        )


class RegressionResults:
    """Container for regression model results.

    Stores fitted parameters, residuals, and provides methods for computing contrasts
    and statistical tests.

    This implementation is adapted from [`nilearn.glm.regression.RegressionResults`][
    nilearn.glm.regression.RegressionResults] and
    [`nilearn.glm.model.LikelihoodModelResults`][
    nilearn.glm.model.LikelihoodModelResults].

    Parameters
    ----------
    theta : (n_regressors, n_voxels) numpy.ndarray
        Parameter estimates (beta coefficients).
    Y : (n_timepoints, n_voxels) numpy.ndarray
        Original data.
    model : OLSModel or ARModel
        Fitted model instance.
    whitened_Y : (n_timepoints, n_voxels) numpy.ndarray
        Whitened data.
    whitened_residuals : (n_timepoints, n_voxels) numpy.ndarray, optional
        Whitened residuals. If not provided, the diagnostic properties
        (`residuals`, `sse`, `mse`) raise, as they do after
        [`_strip_heavy_fields`][confusius.glm._models.RegressionResults._strip_heavy_fields].
    dispersion : (n_voxels,) numpy.ndarray
        Residual variance estimate per voxel.
    cov : (n_regressors, n_regressors) numpy.ndarray
        Normalized covariance matrix.

    Attributes
    ----------
    theta : (n_regressors, n_voxels) numpy.ndarray
        Parameter estimates.
    cov : (n_regressors, n_regressors) numpy.ndarray
        Covariance matrix.
    dispersion : (n_voxels,) numpy.ndarray
        Variance estimates.
    df_total : int
        Total degrees of freedom.
    df_model : int
        Model degrees of freedom.
    df_residuals : int
        Residual degrees of freedom.
    """

    def __init__(
        self,
        theta: npt.NDArray[np.floating],
        Y: npt.NDArray[np.floating],
        model: OLSModel | ARModel,
        whitened_Y: npt.NDArray[np.floating],
        whitened_residuals: npt.NDArray[np.floating] | None,
        dispersion: npt.NDArray[np.floating],
        cov: npt.NDArray[np.floating],
    ) -> None:
        """Initialize regression results container."""
        self.theta = theta
        self.dispersion = dispersion
        self.cov = cov

        # Heavy fields. Kept for diagnostics (residuals/predicted/sse/mse) but dropped
        # by `_strip_heavy_fields` when the parent estimator runs with
        # `minimize_memory=True`. Once stripped, the diagnostic properties raise;
        # contrast methods (`compute_t_contrast`, `compute_f_contrast`) keep working because they only
        # depend on `theta`, `cov`, `dispersion`, `df_residuals`.
        self.Y: npt.NDArray[np.floating] | None = Y
        self.model: OLSModel | ARModel | None = model
        self.whitened_Y: npt.NDArray[np.floating] | None = whitened_Y
        self.whitened_residuals: npt.NDArray[np.floating] | None = whitened_residuals

        self.df_total = Y.shape[0]
        self.df_model = model.df_model
        self.df_residuals = self.df_total - self.df_model

    def _strip_heavy_fields(self) -> None:
        """Drop arrays only used by diagnostic properties.

        Frees `Y`, `whitened_Y`, `whitened_residuals`, and the `model` reference. After
        this call, contrast computations still work but `residuals`, `predicted`, `sse`,
        `mse` raise. Used by
        [`FirstLevelModel`][confusius.glm.first_level.FirstLevelModel] when
        `minimize_memory=True`.
        """
        self.Y = None
        self.model = None
        self.whitened_Y = None
        self.whitened_residuals = None

    @property
    def residuals(self) -> npt.NDArray[np.floating]:
        """Raw residuals: `Y - X @ theta`.

        Returns
        -------
        (n_timepoints, n_voxels) numpy.ndarray
            Raw residuals.

        Raises
        ------
        RuntimeError
            If the heavy fields were stripped via `minimize_memory=True`.
        """
        if self.Y is None:
            raise RuntimeError(
                "Y was dropped by minimize_memory=True; refit with "
                "minimize_memory=False to access residuals."
            )
        return self.Y - self.predicted

    @property
    def predicted(self) -> npt.NDArray[np.floating]:
        """Fitted values: `X @ theta`.

        Returns
        -------
        (n_timepoints, n_voxels) numpy.ndarray
            Fitted values.

        Raises
        ------
        RuntimeError
            If the model reference was dropped via `minimize_memory=True`.
        """
        if self.model is None:
            raise RuntimeError(
                "model was dropped by minimize_memory=True; refit with "
                "minimize_memory=False to access predicted."
            )
        return self.model.design @ self.theta

    @property
    def sse(self) -> npt.NDArray[np.float64]:
        """Sum of squared errors (RSS).

        Returns
        -------
        (n_voxels,) numpy.ndarray
            Sum of squared whitened residuals per voxel.

        Raises
        ------
        RuntimeError
            If `whitened_residuals` was never kept or was dropped via
            `minimize_memory=True`.
        """
        if self.whitened_residuals is None:
            raise RuntimeError(
                "whitened_residuals was dropped by minimize_memory=True; "
                "refit with minimize_memory=False to access sse/mse."
            )
        return _sum_of_squares(self.whitened_residuals)

    @property
    def mse(self) -> npt.NDArray[np.floating]:
        """Mean squared error: `SSE / df_residuals`.

        Returns
        -------
        (n_voxels,) numpy.ndarray
            Mean squared error per voxel.
        """
        return self.sse / self.df_residuals

    def compute_t_contrast(
        self, contrast: npt.NDArray[np.floating]
    ) -> dict[str, npt.NDArray[np.floating]]:
        """Compute *t*-statistic for a contrast.

        Parameters
        ----------
        contrast : (n_regressors,) or (1, n_regressors) numpy.ndarray
            Contrast vector defining the linear combination of parameters.

        Returns
        -------
        dict
            Dictionary with keys:
            - `"effect"`: Contrast effect size (scalar or `(n_voxels,)`).
            - `"sd"`: Standard error of contrast (scalar or `(n_voxels,)`).
            - `"t"`: *t*-statistic (scalar or `(n_voxels,)`).
            - `"df_den"`: Denominator degrees of freedom.
        """
        if contrast.ndim == 1:
            contrast = contrast[np.newaxis, :]

        if contrast.shape[0] != 1:
            raise ValueError(
                f"t-contrast must have single row, got shape {contrast.shape}"
            )

        effect = (contrast @ self.theta).squeeze()

        # Compute variance: c^T @ cov_v @ c * dispersion_v.
        # cov is (V, K, K) for per-voxel AR, (K, K) for OLS.
        if self.cov.ndim == 3:
            c = contrast.ravel()
            var = np.einsum("k,vkl,l->v", c, self.cov, c)  # (V,)
        else:
            var = np.dot(np.dot(contrast, self.cov), contrast.T).squeeze()
        sd = np.sqrt(var * self.dispersion)

        t = effect * _positive_reciprocal(sd)

        return {
            "effect": effect,
            "sd": sd,
            "t": t,
            "df_den": self.df_residuals,
        }

    def compute_f_contrast(
        self, contrast: npt.NDArray[np.floating]
    ) -> dict[str, npt.NDArray[np.floating]]:
        """Compute *F*-statistic for a multi-dimensional contrast.

        Parameters
        ----------
        contrast : (q, n_regressors) numpy.ndarray
            Contrast matrix with `q` rows (`q > 1` for F-test).

        Returns
        -------
        dict
            Dictionary with keys:

            - `"effect"`: Raw contrast effect `c·θ`, shape `(q, n_voxels)`.
            - `"whitened_effect"`: Effect whitened by `cholesky(invcov)`, shape
              `(q, n_voxels)`. Designed so that
              `sum(whitened_effect**2, axis=0) / q / dispersion == F`, which is
              what [`Contrast`][confusius.glm._contrasts.Contrast] consumes
              downstream.
            - `"covariance"`: Per-voxel contrast covariance with dispersion,
              shape `(n_voxels, q, q)`.
            - `"dispersion"`: Per-voxel residual variance, shape `(n_voxels,)`.
            - `"F"`: *F*-statistic `(n_voxels,)`.
            - `"df_num"`: Numerator degrees of freedom (`q`).
            - `"df_den"`: Denominator degrees of freedom.
        """
        contrast = np.atleast_2d(np.asarray(contrast))
        q = contrast.shape[0]

        ctheta = contrast @ self.theta  # (q, n_voxels)

        # Build the per-voxel q×q contrast covariance at unit dispersion, then
        # whiten the effect with the Cholesky factor of its inverse. With
        # L @ L.T == invcov, `whitened = L.T @ ctheta` satisfies
        # ||whitened||² == ctheta' · invcov · ctheta, which is the quadratic
        # form that defines the F-statistic. This is the trick nilearn uses
        # (with sqrtm) so that downstream code only needs the standard
        # `sum(effect²) / dim / variance` formula to recover the correct F.
        if self.cov.ndim == 3:
            cov_q = np.einsum("qk,vkl,pl->vqp", contrast, self.cov, contrast)
            invcov = np.linalg.inv(cov_q)
            L = np.linalg.cholesky(invcov)  # (V, q, q)
            whitened = np.einsum("vpq,pv->qv", L, ctheta)  # (q, V)
            cov_full = cov_q * self.dispersion[:, None, None]
        else:
            cov_q = contrast @ self.cov @ contrast.T  # (q, q)
            invcov = spl.inv(cov_q)
            L = np.linalg.cholesky(invcov)  # (q, q)
            whitened = L.T @ ctheta  # (q, V)
            cov_full = cov_q[None, ...] * self.dispersion[:, None, None]

        f_stat = np.sum(whitened**2, axis=0) / (q * self.dispersion)

        return {
            "effect": ctheta,
            "whitened_effect": whitened,
            "covariance": cov_full,
            "dispersion": self.dispersion,
            "F": f_stat,
            "df_num": q,
            "df_den": self.df_residuals,
        }

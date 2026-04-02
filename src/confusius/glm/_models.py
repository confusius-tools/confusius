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
    """Return element-wise reciprocal, but replace 0 with 0.

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


class OLSModel:
    """Ordinary Least Squares regression model.

    Implements standard OLS regression for GLM analysis. The model is specified
    with a design matrix and fit using the [`fit`][confusius.glm._models.OLSModel]
    method.

    This implementation is adapted from
    [`nilearn.glm.regression.OLSModel`][nilearn.glm.regression.OLSModel].

    Parameters
    ----------
    design : (n_timepoints, n_regressors) numpy.ndarray
        Design matrix. Observations in rows, regressors in columns.

    Attributes
    ----------
    design : (n_timepoints, n_regressors) numpy.ndarray
        The design matrix.
    whitened_design : (n_timepoints, n_regressors) numpy.ndarray
        Whitened design matrix (same as design for OLS).
    calc_beta : (n_regressors, n_timepoints) numpy.ndarray
        Moore-Penrose pseudoinverse of whitened design.
    normalized_cov_beta : (n_regressors, n_regressors) numpy.ndarray
        Normalized covariance matrix: `calc_beta @ calc_beta.T`.
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

        # Moore-Penrose pseudoinverse
        self.calc_beta = spl.pinv(self.whitened_design)

        # Normalized covariance: (X^T X)^-1 for OLS
        self.normalized_cov_beta = self.calc_beta @ self.calc_beta.T

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

    def fit(self, Y: npt.NDArray[np.floating]) -> RegressionResults:
        """Fit the model to data.

        Parameters
        ----------
        Y : (n_timepoints, n_voxels) numpy.ndarray
            Data with time in rows, voxels/features in columns.

        Returns
        -------
        RegressionResults
            Fitted model results containing parameter estimates,
            residuals, and statistics.
        """
        whitened_Y = self.whiten(Y)

        # Estimate parameters: beta = (X^T X)^-1 X^T Y.
        beta = self.calc_beta @ whitened_Y

        # Compute whitened residuals.
        whitened_residuals = whitened_Y - self.whitened_design @ beta

        # Estimate dispersion (variance): sigma^2 = RSS / (n - p).
        rss = np.sum(whitened_residuals**2, axis=0)
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
            cov=self.normalized_cov_beta,
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

        Unlike `OLSModel.whiten`, this operates on `(n_timepoints, n_voxels)` data
        only — the design matrix is handled implicitly via cross-product expansion in
        `fit`.

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

    def fit(self, Y: npt.NDArray[np.floating]) -> RegressionResults:
        """Fit the AR model per voxel.

        Each voxel's time series and the shared design matrix are whitened using that
        voxel's own AR coefficients. The normal equations are then solved in batch over
        all voxels.

        Parameters
        ----------
        Y : (n_timepoints, n_voxels) numpy.ndarray
            Data with time in rows, voxels in columns.

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

        # Avoid materializing whitened_X of shape (V, T, K): for large V and T this can
        # exhaust memory. Instead precompute cross-products from the design lags.
        #
        # Define padded lags L[i] where L[i, t, :] = X[t-(i+1), :] for t >= i+1, else 0.
        # Then whitened_X[v] = X - sum_i(rho[i,v] * L[i]).
        #
        # Normal-equation cross-products follow by expanding:
        #   XtX[v] = A - rho·(B + Bᵀ) + rhoᵀ C rho
        #   XtY[v] = P[:,v] - sum_i(rho[i,v] * Q[i,:,v])
        #
        # where A=(K,K), B=(p,K,K), C=(p,p,K,K), P=(K,V), Q=(p,K,V) — all small.

        X = self.design.astype(np.float64)

        # Padded lags: (order, T, K)
        L = np.zeros((self.order, T, K))
        for i in range(self.order):
            L[i, (i + 1) :, :] = X[: -(i + 1), :]

        # Precompute design cross-products (independent of Y and rho).
        A = X.T @ X  # (K, K)
        B = np.einsum("tk,itl->ikl", X, L)  # (order, K, K): X.T @ L[i]
        C = np.einsum("itk,jtl->ijkl", L, L)  # (order, order, K, K): L[i].T @ L[j]

        B_sym = B + B.transpose(0, 2, 1)  # B[i] + B[i].T for each i

        # XtX[v] = A - sum_i rho[i,v]*B_sym[i] + sum_ij rho[i,v]*rho[j,v]*C[i,j]
        XtX = (
            A[np.newaxis]
            - np.einsum("iv,ikl->vkl", self.rho, B_sym)
            + np.einsum("iv,jv,ijkl->vkl", self.rho, self.rho, C)
        )  # (V, K, K)

        # XtY[v] = X.T @ wY[:,v] - sum_i rho[i,v] * L[i].T @ wY[:,v]
        P = X.T @ whitened_Y  # (K, V)
        Q = np.einsum("itk,tv->ikv", L, whitened_Y)  # (order, K, V)
        XtY = P.T - np.einsum("iv,ikv->vk", self.rho, Q)  # (V, K)

        # beta: (V, K) transposed to (K, V) to match convention.
        beta = np.linalg.solve(XtX, XtY[..., np.newaxis])[..., 0].T

        # Per-voxel normalized covariance: (V, K, K) = (XtX)^{-1}
        cov_beta = np.linalg.inv(XtX)

        # Whitened residuals: wresid = wY - wX @ beta, computed without wX.
        # wX[v] @ beta[:,v] = X @ beta[:,v] - sum_i rho[i,v] * L[i] @ beta[:,v]
        wresid = whitened_Y - X @ beta
        for i in range(self.order):
            wresid += self.rho[i] * (L[i] @ beta)  # rho[i]: (V,), L[i]@beta: (T,V)

        # Dispersion per voxel.
        rss = np.sum(wresid**2, axis=0)
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
    whitened_residuals : (n_timepoints, n_voxels) numpy.ndarray
        Whitened residuals.
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
        whitened_residuals: npt.NDArray[np.floating],
        dispersion: npt.NDArray[np.floating],
        cov: npt.NDArray[np.floating],
    ) -> None:
        """Initialize regression results container."""
        self.theta = theta
        self.Y = Y
        self.model = model
        self.whitened_Y = whitened_Y
        self.whitened_residuals = whitened_residuals
        self.dispersion = dispersion
        self.cov = cov

        self.df_total = Y.shape[0]
        self.df_model = model.df_model
        self.df_residuals = self.df_total - self.df_model

    @property
    def residuals(self) -> npt.NDArray[np.floating]:
        """Raw residuals: `Y - X @ theta`.

        Returns
        -------
        (n_timepoints, n_voxels) numpy.ndarray
            Raw residuals.
        """
        return self.Y - self.predicted

    @property
    def predicted(self) -> npt.NDArray[np.floating]:
        """Fitted values: `X @ theta`.

        Returns
        -------
        (n_timepoints, n_voxels) numpy.ndarray
            Fitted values.
        """
        return self.model.design @ self.theta

    @property
    def sse(self) -> npt.NDArray[np.float64]:
        """Sum of squared errors (RSS).

        Returns
        -------
        (n_voxels,) numpy.ndarray
            Sum of squared whitened residuals per voxel.
        """
        return np.sum(self.whitened_residuals**2, axis=0)

    @property
    def mse(self) -> npt.NDArray[np.floating]:
        """Mean squared error: `SSE / df_residuals`.

        Returns
        -------
        (n_voxels,) numpy.ndarray
            Mean squared error per voxel.
        """
        return self.sse / self.df_residuals

    def t_contrast(
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

    def f_contrast(
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
            - `"effect"`: Contrast effect vector `(q, n_voxels)`.
            - `"covariance"`: Covariance matrix.
            - `"F"`: *F*-statistic `(n_voxels,)`.
            - `"df_num"`: Numerator degrees of freedom (`q`).
            - `"df_den"`: Denominator degrees of freedom.
        """
        contrast = np.atleast_2d(np.asarray(contrast))
        q = contrast.shape[0]

        ctheta = contrast @ self.theta  # (q, n_voxels)

        # cov is (V, K, K) for per-voxel AR, (K, K) for OLS.
        if self.cov.ndim == 3:
            # Per-voxel contrast covariance: (V, q, q)
            cov = np.einsum("qk,vkl,pl->vqp", contrast, self.cov, contrast)
            invcov = np.linalg.inv(cov)
            ctheta_v = ctheta.T  # (V, q)
            f_stat = np.einsum("vij,vj,vi->v", invcov, ctheta_v, ctheta_v) / (
                q * self.dispersion
            )
        else:
            cov = contrast @ self.cov @ contrast.T  # (q, q)
            invcov = spl.inv(cov)
            f_stat = np.sum((invcov @ ctheta) * ctheta, axis=0) / (q * self.dispersion)

        return {
            "effect": ctheta,
            "covariance": cov * self.dispersion[:, np.newaxis, np.newaxis],
            "F": f_stat,
            "df_num": q,
            "df_den": self.df_residuals,
        }

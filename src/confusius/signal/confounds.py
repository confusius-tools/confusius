"""Confound regression functions for signal preprocessing.

Portions of this file are derived from Nilearn, which is licensed under the BSD-3-Clause
License. See `NOTICE` file for details.
"""

import warnings
from collections.abc import Callable
from typing import cast

import numpy as np
import pandas as pd
import scipy.linalg
import xarray as xr
from sklearn.utils.extmath import randomized_svd

from confusius._utils.mask import validate_spatial_or_feature_mask
from confusius._utils.stack import find_stack_level
from confusius.multipose.timing import consolidate_time_coordinate
from confusius.signal._utils import remove_zero_variance_voxels
from confusius.signal.detrending import detrend as detrend_signals
from confusius.signal.standardization import standardize
from confusius.validation import ensure_time_aligned, validate_time_series


def _prepare_confounds_for_regression(
    confounds: np.ndarray,
    standardize_confounds: bool,
) -> np.ndarray:
    """Prepare confounds for regression.

    Parameters
    ----------
    confounds : numpy.ndarray
        Confound regressors, shape `(time, n_confounds)`.
    standardize_confounds : bool
        Whether to z-score confounds. If `False`, confounds are divided by their
        maximum absolute value without centering.

    Returns
    -------
    numpy.ndarray
        Prepared confound regressors. Standardized constant columns are returned as
        zeros; unstandardized columns are divided by their maximum absolute value.
    """
    if standardize_confounds:
        confounds = confounds - confounds.mean(axis=0)
        confound_scale = confounds.std(axis=0, ddof=1)
        confound_scale[confound_scale < np.finfo(np.float64).eps] = 1
        return confounds / confound_scale

    confound_scale = np.max(np.abs(confounds), axis=0)
    confound_scale[confound_scale == 0] = 1
    return confounds / confound_scale


def _regress_confounds_numpy(
    signals: np.ndarray,
    confounds: np.ndarray,
    standardize_confounds: bool = True,
) -> np.ndarray:
    """Core confound regression using QR decomposition.

    Uses QR decomposition with column pivoting for numerical stability.
    Projects signals onto the orthogonal complement of confound space.

    Parameters
    ----------
    signals : numpy.ndarray
        Signals array of any shape, with time along first axis.
    confounds : numpy.ndarray
        Confound regressors, shape (time, n_confounds).
    standardize_confounds : bool, default=True
        Whether to z-score confounds before regression. If `False`, confounds are
        divided by their maximum absolute value for numerical stability without
        centering.

    Returns
    -------
    numpy.ndarray
        Residuals after confound regression, same shape as signals.

    Notes
    -----
    Based on nilearn.signal.clean implementation which follows
    Friston et al. (1994) for confound removal via projection onto
    the orthogonal of the signal space.
    """
    confounds = _prepare_confounds_for_regression(confounds, standardize_confounds)

    qr_result = scipy.linalg.qr(confounds, mode="economic", pivoting=True)
    Q, R, _ = cast(tuple[np.ndarray, np.ndarray, np.ndarray], qr_result)

    tol = np.finfo(np.float64).eps * 100.0
    rank = np.sum(np.abs(np.diag(R)) > tol)
    Q = Q[:, :rank]

    original_shape = signals.shape
    if signals.ndim > 2:
        signals_2d = signals.reshape(signals.shape[0], -1)
    else:
        signals_2d = signals

    # Parentheses enforce the low-cost order: compute (rank, n_voxels) first,
    # then map back to (time, n_voxels), avoiding a large (time, time) product.
    projection = Q @ (Q.T @ signals_2d)
    residuals_2d = signals_2d - projection

    if signals.ndim > 2:
        residuals = residuals_2d.reshape(original_shape)
    else:
        residuals = residuals_2d

    return residuals


def _regress_confounds_wrapper(data, axis, confounds, standardize_confounds):
    """Wrapper for confound regression that works with xr.apply_ufunc.

    Parameters
    ----------
    data : numpy.ndarray
        Input data array.
    axis : int
        Axis along which to apply regression (should be time axis).
    confounds : numpy.ndarray
        Confound regressors.
    standardize_confounds : bool
        Whether to standardize confounds.

    Returns
    -------
    numpy.ndarray
        Residuals after confound regression.
    """
    if axis != 0:
        data = np.moveaxis(data, axis, 0)

    result = _regress_confounds_numpy(data, confounds, standardize_confounds)

    if axis != 0:
        result = np.moveaxis(result, 0, axis)

    return result


def regress_confounds(
    signals: xr.DataArray,
    confounds: xr.DataArray | np.ndarray | pd.DataFrame,
    standardize_confounds: bool = True,
) -> xr.DataArray:
    """Remove confounds from signals via linear regression.

    This function performs confound regression by projecting the signals onto the
    orthogonal complement of the confound space. This removes variance in the signals
    that can be explained by the confounds.

    This function was adapted from `nilearn.signal.clean`.

    Parameters
    ----------
    signals : (time, ...) xarray.DataArray
        Signals to clean. Must have a `time` dimension. Can be any shape, e.g.,
        extracted signals `(time, space)` or VoxelData array `(time, k, j, i)`.

        !!! warning "Chunking along time is not supported"
            The `time` dimension must NOT be chunked. Chunk only spatial dimensions:
            `data.chunk({'time': -1})`.

    confounds : (time, n_confounds) xarray.DataArray, numpy.ndarray, or \
            pandas.DataFrame
        Confound regressors to remove. Can have shape `(time,)` for a single
        confound. A DataArray must have a `time` dimension; a
        DataFrame must have a `time` column, its other numeric columns being the
        confounds; a NumPy array must have time along its first axis. `time`
        coordinates must match those of `signals` within the default
        coordinate-comparison tolerance (`rtol=1e-5`, `atol=1e-8`); an input without
        `time` coordinates is assumed ordered like `signals` and takes its `time`
        coordinates, with a warning since alignment cannot be verified (see
        [`ensure_time_aligned`][confusius.validation.ensure_time_aligned]).
    standardize_confounds : bool, default: True
        Whether to z-score confounds before regression. If `False`, confounds are
        divided by their maximum absolute value for numerical stability without
        centering.

    Returns
    -------
    xarray.DataArray
        Residuals after confound regression, same shape and coordinates as input
        signals.

    Raises
    ------
    ValueError
        If `signals` does not have a `time` dimension, or if `confounds` have
        mismatched time dimension or invalid shape.
    TypeError
        If `confounds` is not a DataArray, NumPy array, or DataFrame.

    Warns
    -----
    UserWarning
        If `confounds` has no `time` coordinates, since alignment cannot be verified.
    UserWarning
        If `signals` has a `pose` dimension, since the same confounds are regressed
        from every pose.

    Notes
    -----
    - Uses QR decomposition with column pivoting for numerical stability.
    - Handles rank-deficient confound matrices (e.g., collinear confounds) by removing
      redundant columns.
    - Based on the projection method from Friston et al. (1994).

    References
    ----------
    [^1]:
        Friston, K. J., Holmes, A. P., Worsley, K. J., Poline, J. P., Frith, C. D., &
        Frackowiak, R. S. (1994). Statistical parametric maps in functional imaging: a
        general linear approach. Human brain mapping, 2(4), 189-210.

    Examples
    --------
    Remove motion parameters from extracted signals:

    >>> import xarray as xr
    >>> import numpy as np
    >>> from confusius.signal import regress_confounds
    >>> # Create signals (100 timepoints, 50 voxels)
    >>> signals = xr.DataArray(
    ...     np.random.randn(100, 50),
    ...     dims=["time", "space"],
    ...     coords={"time": np.arange(100) * 0.1}
    ... )
    >>> # Create motion confounds (6 motion parameters)
    >>> motion_params = xr.DataArray(
    ...     np.random.randn(100, 6),
    ...     dims=["time", "confound"],
    ...     coords={"time": signals.coords["time"]},
    ... )
    >>> # Remove motion effects
    >>> cleaned = regress_confounds(signals, motion_params)

    Works on 3D+t imaging data:

    >>> imaging_data = xr.DataArray(
    ...     np.random.randn(100, 10, 20, 30),
    ...     dims=["time", "z", "y", "x"]
    ... )
    >>> cleaned_imaging = regress_confounds(
    ...     imaging_data, motion_params
    ... )
    """
    time_axis, _ = validate_time_series(signals, "confound regression")

    confounds_array = ensure_time_aligned(
        signals, confounds, "confounds", ndim=2
    ).values
    if "pose" in signals.dims:
        warnings.warn(
            "signals have a 'pose' dimension: the same confounds are regressed from "
            "every pose. To use per-pose confounds, regress each pose separately.",
            stacklevel=find_stack_level(),
        )

    result = xr.apply_ufunc(
        _regress_confounds_wrapper,
        signals,
        kwargs={
            "axis": time_axis,
            "confounds": confounds_array,
            "standardize_confounds": standardize_confounds,
        },
        dask="parallelized",
        output_dtypes=[signals.dtype],
    )

    return result


def _left_singular_vectors_via_eigh(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the left singular vectors/values of `values` via a Gram-matrix eigh.

    Equivalent to `U, s, _ = scipy.linalg.svd(values, full_matrices=False)`, but
    computed by eigendecomposing whichever of `values @ values.T` (shape
    `(n_time, n_time)`) or `values.T @ values` (shape `(n_voxels, n_voxels)`) is
    smaller, instead of the full SVD of `values` itself — this is CompCor's usual
    shape (a handful to a few thousand timepoints against anywhere from hundreds
    to tens of thousands of selected noise voxels), where SVD's cost scales with
    both dimensions but the eigendecomposition only needs to factor the smaller
    of the two Gram matrices, matching nilearn's CompCor implementation for the
    `n_time < n_voxels` case and its mirror image otherwise.

    Parameters
    ----------
    values : (n_time, n_voxels) numpy.ndarray
        Standardized noise signals.

    Returns
    -------
    U : (n_time, k) numpy.ndarray
        Left singular vectors, `k = min(n_time, n_voxels)`, ordered by
        descending singular value.
    s : (k,) numpy.ndarray
        Singular values, descending.
    """
    n_time, n_voxels = values.shape

    if n_time <= n_voxels:
        eigvals, U = scipy.linalg.eigh(values @ values.T, check_finite=False)
        order = np.argsort(eigvals)[::-1]
        eigvals = np.clip(eigvals[order], 0.0, None)
        U = U[:, order]
    else:
        eigvals, V = scipy.linalg.eigh(values.T @ values, check_finite=False)
        order = np.argsort(eigvals)[::-1]
        eigvals = np.clip(eigvals[order], 0.0, None)
        V = V[:, order]
        s = np.sqrt(eigvals)
        # u_i = X @ v_i / s_i; guard s_i == 0 (rank-deficient / zero-variance
        # directions) instead of dividing by zero.
        safe_s = np.where(s > 0, s, 1.0)
        U = np.where(s > 0, (values @ V) / safe_s, 0.0)
        return U, s

    return U, np.sqrt(eigvals)


def _top_left_singular_vectors(
    values: np.ndarray, n_components: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the top `n_components` left singular vectors/values of `values`.

    Uses `sklearn.utils.extmath.randomized_svd` when the matrix is large and
    `n_components` is small relative to both dimensions — mirroring
    `sklearn.decomposition.PCA`'s own "randomized" solver heuristic. CompCor
    typically keeps only a handful of components (5-30) out of potentially
    thousands of timepoints and tens of thousands of selected voxels, where
    even `_left_singular_vectors_via_eigh`'s reduced Gram-matrix eigendecomposition
    computes the full spectrum on the smaller side for far more components than
    are actually kept; randomized SVD instead targets only the requested
    components directly, without ever forming a Gram matrix. Falls back to the
    exact eigendecomposition otherwise, since the random projection makes this
    an approximation (still validated against exact SVD via correlation in
    `tests/unit/test_signal/test_compcor.py`, matching to several nines for the
    top few components on realistic spectra).

    Parameters
    ----------
    values : (n_time, n_voxels) numpy.ndarray
        Standardized noise signals.
    n_components : int
        Number of components to keep.

    Returns
    -------
    U : (n_time, n_components) numpy.ndarray
        Top left singular vectors, descending by singular value.
    s : (n_components,) numpy.ndarray
        Top singular values, descending.
    """
    min_dim = min(values.shape)

    if values.size > 500 * 500 and n_components < 0.8 * min_dim:
        U, s, _Vt = randomized_svd(values, n_components=n_components, random_state=0)
        return U, s

    U, s = _left_singular_vectors_via_eigh(values)
    return U[:, :n_components], s[:n_components]


def _extract_compcor_components(
    noise_signals: xr.DataArray,
    n_components: int,
    do_detrend: bool,
) -> xr.DataArray:
    """Core CompCor extraction using PCA (SVD on standardized signals).

    Parameters
    ----------
    noise_signals : (time, space) xarray.DataArray
        Selected signals.
    n_components : int
        Number of components to extract.
    do_detrend : bool
        Whether to linearly detrend before PCA.

    Returns
    -------
    (time, component) xarray.DataArray
        Principal components (loadings) with:

        - `time` dimension with coordinates from input `noise_signals`
        - `component` dimension (0 to n_components-1)
        - `explained_variance_ratio` coordinate on `component` dimension

    Notes
    -----
    This function performs PCA by:

    1. Removing zero-variance voxels.
    2. Detrending (if requested).
    3. Standardizing (z-score) to give equal weight to each space.
    4. Computing the top `n_components` left singular vectors/values (see
       `_top_left_singular_vectors`).
    5. Computing explained variance from the total variance in the data and the
       kept singular values.
    """
    noise_signals = remove_zero_variance_voxels(noise_signals)

    if do_detrend:
        noise_signals = detrend_signals(noise_signals, order=1)

    noise_signals = standardize(noise_signals, method="zscore")

    if hasattr(noise_signals.data, "chunks"):
        import dask.array as da

        dask_values = noise_signals.data
        min_dim = min(dask_values.shape)
        if dask_values.size > 500 * 500 and n_components < 0.8 * min_dim:
            # dask's randomized-SVD counterpart to _top_left_singular_vectors,
            # for the same reason: CompCor keeps far fewer components than
            # either dimension, so a full (dask) SVD does needless work.
            # n_power_iter=4 matches the accuracy sklearn's randomized_svd
            # targets by default (its 'auto' n_iter is 4-7 for typical CompCor
            # component counts); seed=0 keeps results reproducible.
            U, s, _V = da.linalg.svd_compressed(
                dask_values, k=n_components, n_power_iter=4, seed=0
            )
            components = U
            total_variance = (dask_values**2).sum()
            explained_variance_ratio = (s**2) / total_variance
        else:
            svd = cast(
                Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]],
                da.linalg.svd,
            )
            U, s, _Vt = svd(dask_values)
            components = U[:, :n_components]
            total_variance = (s**2).sum()
            explained_variance_ratio = (s[:n_components] ** 2) / total_variance
    else:
        values = noise_signals.values
        components, s = _top_left_singular_vectors(values, n_components)
        # Total variance equals the sum of ALL singular values squared (the
        # Frobenius norm identity trace(X^T X) = sum(x_ij^2) = sum(s_i^2)), so
        # it can be read directly off the data without computing the full
        # spectrum — needed since `_top_left_singular_vectors` may only return
        # the top `n_components` singular values, not all of them.
        total_variance = float(np.sum(values**2))
        explained_variance_ratio = (s**2) / total_variance

    result = xr.DataArray(
        components,
        dims=["time", "component"],
        coords={
            "time": noise_signals.coords["time"],
            "component": np.arange(n_components),
            "explained_variance_ratio": (["component"], explained_variance_ratio),
        },
    )

    return result


def compute_compcor_confounds(
    signals: xr.DataArray,
    noise_mask: xr.DataArray | None = None,
    variance_threshold: float | None = None,
    n_components: int = 5,
    detrend: bool = False,
    skipna: bool = False,
) -> xr.DataArray:
    """Extract noise components using the CompCor method.

    CompCor (Component-based Noise Correction) extracts principal components from
    noise regions (aCompCor) or high-variance voxels (tCompCor) to use as confound
    regressors [^1].

    Parameters
    ----------
    signals : (time, ...) xarray.DataArray
        Signals from which to extract components. Must have a `time` dimension.
        For extracted signals, shape is typically `(time, space)`. For a full
        VoxelData array, shape is typically `(time, k, j, i)`.

        !!! warning "Chunking along time is not supported"
            The `time` dimension must NOT be chunked. Chunk only spatial dimensions:
            `data.chunk({'time': -1})`.

    noise_mask : xarray.DataArray, optional
        Binary mask indicating voxels to consider. Must have the same spatial
        dimensions and coordinates as `signals` (excluding time). Can be combined
        with `variance_threshold` for hybrid selection.
    variance_threshold : float, optional
        Variance percentile threshold (0-1) for selecting high-variance voxels.
        For example, 0.02 selects the top 2% highest-variance voxels from the
        voxels selected by `noise_mask` (if provided) or all voxels. Can be
        combined with `noise_mask` for hybrid selection.
    n_components : int, default: 5
        Number of principal components to extract.
    detrend : bool, default: False
        Whether to linearly detrend the selected voxels before SVD. Can improve
        component quality by removing slow drifts.
    skipna : bool, default: False
        Whether to skip NaN values when computing variance quantiles for tCompCor. If
        `False`, uses fast quantile calculation. If `True`, uses slower NaN-aware
        quantile calculation. Set to `True` only if your data contains NaN values.

    Returns
    -------
    (time, component) xarray.DataArray
        Extracted CompCor components. Each column (component) is a principal component
        that can be used as a confound regressor. The DataArray includes:

        - `time` dimension with coordinates matching the input signals.
        - `component` dimension (0 to `n_components - 1`).
        - `explained_variance_ratio` coordinate on `component` dimension, containing
          the proportion of total variance explained by each component.

    Raises
    ------
    ValueError
        If `signals` does not have a `time` dimension.
    ValueError
        If mask doesn't have the right dtype or its dimensions/coordinates don't match
        signal spatial dimensions.
    ValueError
        If both `noise_mask` and `variance_threshold` are `None` (must specify at
        least one).
    ValueError
        If `variance_threshold` is not in range `(0, 1)`.
    ValueError
        If `n_components` is not positive.
    ValueError
        If no voxels are selected (empty mask or threshold too high).

    Notes
    -----
    - **aCompCor**: Specify only `noise_mask` to extract components from
      anatomically-defined noise regions (e.g., white matter, CSF). Useful when
      anatomical segmentation is available.
    - **tCompCor**: Specify only `variance_threshold` to extract components from
      high-variance voxels. Useful when no anatomical masks are available.
    - **Hybrid**: Specify both `noise_mask` and `variance_threshold` to extract
      components from high-variance voxels within a specific anatomical region
      (e.g., high-variance white matter voxels).

    References
    ----------
    [^1]:
        Behzadi, Yashar, et al. “A Component Based Noise Correction Method (CompCor) for
        BOLD and Perfusion Based fMRI.” NeuroImage, vol. 37, no. 1, Aug. 2007, pp.
        90–101. DOI.org (Crossref), <https://doi.org/10.1016/j.neuroimage.2007.04.042>.

    Examples
    --------
    Extract aCompCor components from white matter:

    >>> import xarray as xr
    >>> import numpy as np
    >>> from confusius.signal import compute_compcor_confounds, regress_confounds
    >>> signals = xr.DataArray(
    ...     np.random.randn(100, 50),
    ...     dims=["time", "space"],
    ...     coords={"time": np.arange(100) * 0.1}
    ... )
    >>> wm_mask = xr.DataArray(
    ...     np.zeros(50, dtype=bool),
    ...     dims=["space"]
    ... )
    >>> wm_mask.values[:10] = True
    >>> a_compcor = compute_compcor_confounds(
    ...     signals,
    ...     noise_mask=wm_mask,
    ...     n_components=5,
    ...     detrend=True
    ... )
    >>> a_compcor.shape
    (100, 5)

    Extract tCompCor from high-variance voxels:

    >>> t_compcor = compute_compcor_confounds(
    ...     signals,
    ...     variance_threshold=0.2,
    ...     n_components=5,
    ...     detrend=True
    ... )

    Hybrid mode - high-variance WM voxels only:

    >>> hybrid_compcor = compute_compcor_confounds(
    ...     signals,
    ...     noise_mask=wm_mask,
    ...     variance_threshold=0.5,
    ...     n_components=5
    ... )

    Combine different CompCor variants for cleaning:

    >>> all_compcor = xr.concat([a_compcor, t_compcor, hybrid_compcor], dim="component")
    >>> cleaned = regress_confounds(signals, all_compcor)
    """
    validate_time_series(signals, "CompCor computation")

    if noise_mask is None and variance_threshold is None:
        raise ValueError(
            "Must specify at least one of 'noise_mask' or 'variance_threshold'."
        )

    if n_components <= 0:
        raise ValueError(f"'n_components' must be positive, got {n_components}.")

    if signals.ndim == 2 and "space" in signals.dims:
        signals_flat = signals
        spatial_dims = ["space"]
    else:
        time_dim = "time"
        spatial_dims = [d for d in signals.dims if d != time_dim]
        signals_flat = signals.stack(space=spatial_dims)

        time_coord = signals.coords.get("time")
        if time_coord is not None and time_coord.dims == ("time", "pose"):
            # Folding `pose` into `space` above breaks the per-pose (time, pose)
            # `time` coordinate: stacking broadcasts it to (time, space), one
            # timestamp per voxel instead of one per timepoint. CompCor pools
            # voxels across poses into a single PCA, so replace it with one
            # consolidated whole-array time value per timepoint, using the same
            # reference/duration accounting as consolidate_poses.
            signals_flat = signals_flat.assign_coords(
                time=consolidate_time_coordinate(time_coord)
            )

    n_voxels = signals_flat.sizes["space"]

    selected_voxels = np.ones(n_voxels, dtype=bool)

    if noise_mask is not None:
        noise_mask = validate_spatial_or_feature_mask(
            signals, noise_mask, "noise_mask", require_exact_dims=True
        )
        noise_mask_flat = noise_mask.values.flatten()

        if noise_mask_flat.shape[0] != n_voxels:
            raise ValueError(
                f"Noise mask size ({noise_mask_flat.shape[0]}) does not match "
                f"signals spatial size ({n_voxels})."
            )

        selected_voxels = np.logical_and(selected_voxels, noise_mask_flat)

    if variance_threshold is not None:
        if not (0 < variance_threshold < 1):
            raise ValueError(
                f"'variance_threshold' must be in range (0, 1), got {variance_threshold}."
            )

        masked_signals = signals_flat.isel(space=selected_voxels)
        variances = masked_signals.var(dim="time")

        threshold_value = float(
            variances.quantile(1 - variance_threshold, method="linear", skipna=skipna)
        )

        high_var_mask = np.zeros(n_voxels, dtype=bool)
        high_var_mask[selected_voxels] = variances.values >= threshold_value
        selected_voxels = high_var_mask

    n_selected = np.sum(selected_voxels)
    if n_selected == 0:
        raise ValueError(
            "No voxels selected for CompCor. Check your mask or variance_threshold."
        )

    if n_selected < n_components:
        raise ValueError(
            f"Number of selected voxels ({n_selected}) is less than "
            f"n_components ({n_components}). Reduce n_components or adjust selection."
        )

    signals_selected = signals_flat.isel(space=selected_voxels)

    result = _extract_compcor_components(signals_selected, n_components, detrend)

    return result

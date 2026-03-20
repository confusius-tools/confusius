"""Spatial smoothing functions for fUSI volumetric data."""

import warnings
from collections.abc import Sequence

import numpy as np
import scipy.ndimage
import xarray as xr

from confusius._utils import _compute_spacing

_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
"""FWM to Gaussian sigma conversion factor."""


def _gaussian_smooth(
    arr: np.ndarray, sigmas: list[float], ensure_finite: bool
) -> np.ndarray:
    """Apply a Gaussian filter to a NumPy array.

    Parameters
    ----------
    arr : numpy.ndarray
        Input array.
    sigmas : list of float
        Gaussian sigma in voxel units for each axis. Zero means no smoothing.
    ensure_finite : bool
        If True, replace non-finite values with zero before filtering.
    """
    if ensure_finite:
        arr = arr.copy()
        arr[~np.isfinite(arr)] = 0.0
    return scipy.ndimage.gaussian_filter(arr, sigma=sigmas)


def smooth_volume(
    data: xr.DataArray,
    fwhm: float | dict[str, float],
    dims: Sequence[str] | None = None,
    ensure_finite: bool = False,
) -> xr.DataArray:
    """Smooth a DataArray spatially using a Gaussian kernel.

    FWHM values are specified in physical units and converted to voxel-space sigma
    values using the coordinate spacing of each dimension. Smoothing is only applied to
    dimensions with uniform coordinate spacing.

    Parameters
    ----------
    data : xarray.DataArray
        Input data to smooth. Can have any number of dimensions, including a `time`
        dimension (which is never smoothed by default).

        !!! warning "Chunking along smoothed dimensions is not supported"
            Dimensions in `dims` must NOT be chunked. Chunk only along other dimensions,
            e.g. `data.chunk({"time": 10})`.

    fwhm : float or dict[str, float]
        Full width at half maximum of the Gaussian kernel in physical unit. A scalar
        applies the same FWHM to all smoothed dimensions. A dict maps dimension names to
        per-dimension FWHM values, e.g. `{"z": 0.5, "y": 0.2, "x": 0.2}`. All keys must
        be present in the set of smoothed dimensions.
    dims : sequence of str, optional
        Dimensions to smooth. Defaults to all dimensions except `"time"`. All specified
        dimensions must have uniform coordinate spacing; a `ValueError` is raised
        otherwise.
    ensure_finite : bool, default: False
        Whether to replace non-finite values (`NaN`, `Inf`) with zero before filtering.
        This prevents `NaN`s from propagating to neighbouring voxels through the
        Gaussian kernel.

    Returns
    -------
    xarray.DataArray
        Smoothed data with the same shape, dimensions, coordinates, and attributes as
        the input.

    Raises
    ------
    ValueError
        If any dimension in `dims` is not present in the DataArray, had non-uniform or
        undefined coordinate spacing, or is chunked (for Dask-backed arrays).
    ValueError
        If `fwhm` is a dict and any key is not in the set of smoothed dims.

    Notes
    -----
    The Gaussian sigma in voxel units is computed as:

    $$
        \\sigma_{\\text{vox}} = \\frac{\\text{FWHM}}{
            2 \\sqrt{2 \\ln 2} \\cdot \\Delta}
    $$

    where $\\Delta$ is the coordinate spacing for that dimension, in the same
    physical units as `fwhm`.

    Smoothing is applied via
    [`scipy.ndimage.gaussian_filter`][scipy.ndimage.gaussian_filter] with zero sigma for
    non-smoothed dimensions, making the operation separable and efficient.

    Examples
    --------
    Smooth isotropically with a 0.3 (coordinate-unit) FWHM kernel:

    >>> import xarray as xr
    >>> import numpy as np
    >>> import confusius  # noqa: F401
    >>> data = xr.DataArray(
    ...     np.random.randn(5, 10, 1, 20),
    ...     dims=["time", "z", "y", "x"],
    ...     coords={
    ...         "time": np.arange(5) * 0.2,
    ...         "z": np.arange(10) * 0.1,
    ...         "y": np.zeros(1),
    ...         "x": np.arange(20) * 0.1,
    ...     },
    ... )
    >>> smoothed = smooth_volume(data, fwhm=0.3)

    Smooth with anisotropic kernels:

    >>> smoothed = smooth_volume(data, fwhm={"z": 0.5, "y": 0.2, "x": 0.2})

    Smooth only selected dimensions:

    >>> smoothed = smooth_volume(data, fwhm=0.3, dims=["z", "x"])

    Suppress NaN propagation when some voxels are masked:

    >>> smoothed = smooth_volume(data, fwhm=0.3, ensure_finite=True)
    """
    all_dims = [str(d) for d in data.dims]

    if dims is None:
        smooth_dims = [d for d in all_dims if d != "time"]
    else:
        smooth_dims = list(dims)
        invalid = [d for d in smooth_dims if d not in all_dims]
        if invalid:
            raise ValueError(
                f"Dimensions {invalid} are not present in the DataArray. "
                f"Available dimensions: {all_dims}."
            )

    if isinstance(fwhm, dict):
        unknown = [k for k in fwhm if k not in smooth_dims]
        if unknown:
            raise ValueError(
                f"FWHM keys {unknown} are not in the set of smoothed dimensions "
                f"{smooth_dims}. Either add them to 'dims' or remove them from "
                f"'fwhm'."
            )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        spacing = _compute_spacing(data)

    smooth_spacing: dict[str, float] = {}
    for dim in smooth_dims:
        s = spacing.get(dim)
        if s is None:
            raise ValueError(
                f"Dimension '{dim}' has non-uniform or undefined coordinate spacing. "
                "Gaussian smoothing requires regularly sampled coordinates. "
                "Check that the coordinate spacing is uniform, or exclude this "
                "dimension from 'dims'."
            )
        smooth_spacing[dim] = s

    if hasattr(data.data, "chunks"):
        for dim in smooth_dims:
            axis = all_dims.index(dim)
            if len(data.data.chunks[axis]) > 1:
                raise ValueError(
                    f"Dimension '{dim}' is chunked, which would cause boundary "
                    "artifacts during Gaussian smoothing. Rechunk so that smoothed "
                    f"dimensions are not split: data.chunk({{'{dim}': -1}})."
                )

    if isinstance(fwhm, dict):
        fwhm_per_dim = {dim: fwhm.get(dim, 0.0) for dim in smooth_dims}
    else:
        fwhm_per_dim = {dim: float(fwhm) for dim in smooth_dims}

    # Non-smoothed dims get sigma=0 (identity, no filtering applied).
    sigmas = []
    for dim in all_dims:
        if dim in smooth_dims:
            fwhm_val = fwhm_per_dim[dim]
            sigma_vox = fwhm_val * _FWHM_TO_SIGMA / smooth_spacing[dim]
            sigmas.append(sigma_vox)
        else:
            sigmas.append(0.0)

    return xr.apply_ufunc(
        _gaussian_smooth,
        data,
        kwargs={"sigmas": sigmas, "ensure_finite": ensure_finite},
        dask="parallelized",
        output_dtypes=[data.dtype],
        keep_attrs=True,
    )

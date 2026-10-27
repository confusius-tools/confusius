"""Reconstruction-free, full-window SVD power Doppler for IQ VoxelData arrays.

Import `process_iq_to_power_doppler` from this script. This experimental alternative
uses the same descending, half-open component cutoffs as ConfUSIus's index filter.
Each clutter window is also its entire, uniformly weighted integration window.

Example::

    from tools.iq_to_power_doppler import process_iq_to_power_doppler

    pd = process_iq_to_power_doppler(
        iq, clutter_window_width=300, low_cutoff=20, method="subtract"
    ).compute()

Use `method="retained"` for direct retained-component energy instead. No file I/O
or IQ reconstruction is performed by this script.
"""

from __future__ import annotations

import warnings
from typing import Literal

import dask.array as da
import numpy as np
import numpy.typing as npt
import xarray as xr
from scipy import linalg

from confusius.iq import compute_processed_volume_timings, process_iq_blocks
from confusius.validation import ensure_voxeldata, validate_mask


def compute_power_doppler_volume(
    block: npt.NDArray,
    low_cutoff: int = 0,
    high_cutoff: int | None = None,
    clutter_mask: npt.NDArray | None = None,
    method: Literal["subtract", "retained"] = "subtract",
) -> npt.NDArray:
    """Compute one full-window power Doppler image without IQ reconstruction.

    Parameters
    ----------
    block : (time, k, j, i) numpy.ndarray
        Complex IQ block.
    low_cutoff : int, default: 0
        Inclusive first retained component, ordered by decreasing energy.
    high_cutoff : int, optional
        Exclusive last retained component. If not provided, retain through the
        last temporal component.
    clutter_mask : (k, j, i) numpy.ndarray, optional
        Boolean basis-estimation mask. If not provided, use all voxels. There
        must be at least as many selected voxels as frames.
    method : {"subtract", "retained"}, default: "subtract"
        `subtract` subtracts removed-component energy from original energy, with
        double-precision multiplication and accumulation. `retained` directly
        sums retained-component energies, avoiding energy cancellation.

    Returns
    -------
    (1, k, j, i) numpy.ndarray
        Double-precision power Doppler image with a singleton output time axis.
        Negative subtraction results are warned about, not clipped.

    Raises
    ------
    TypeError
        If IQ is not complex, or the mask is not boolean.
    ValueError
        If dimensions, mask, method, or cutoffs are invalid, or too few voxels
        are available to estimate a complete temporal basis.

    Notes
    -----
    Even in double precision, `subtract` can lose weak retained power through
    cancellation. A positive result does not guarantee numerical accuracy.
    Use `retained` when high dynamic range makes subtraction unsafe.
    """
    if block.ndim != 4 or block.shape[0] < 2:
        raise ValueError(
            "IQ block must have shape (time, k, j, i) with at least two frames."
        )
    if not np.issubdtype(block.dtype, np.complexfloating):
        raise TypeError("IQ block must be complex-valued.")
    if method not in ("subtract", "retained"):
        raise ValueError("method must be 'subtract' or 'retained'.")
    t = block.shape[0]
    high = t if high_cutoff is None else high_cutoff
    if (
        not isinstance(low_cutoff, int)
        or not isinstance(high, int)
        or not 0 <= low_cutoff < high <= t
    ):
        raise ValueError(
            "Cutoffs must be integers satisfying 0 <= low < high <= frames."
        )
    signals = block.reshape(t, -1)
    if clutter_mask is not None:
        if clutter_mask.shape != block.shape[1:]:
            raise ValueError("clutter_mask must match the IQ spatial shape.")
        if clutter_mask.dtype != np.bool_:
            raise TypeError("clutter_mask must be boolean.")
        masked = signals[:, clutter_mask.ravel()]
    else:
        masked = signals
    if masked.shape[1] < t:
        raise ValueError("Basis estimation requires at least as many voxels as frames.")

    if low_cutoff == 0 and high == t:
        energy = np.einsum("tn,tn->n", signals.real, signals.real, dtype=np.float64)
        energy += np.einsum("tn,tn->n", signals.imag, signals.imag, dtype=np.float64)
    else:
        masked = masked.astype(np.complex128, copy=False)
        _, vectors = linalg.eigh(masked @ masked.conj().T)
        vectors = vectors[:, ::-1]
        if method == "retained":
            coefficients = vectors[:, low_cutoff:high].conj().T @ signals
            energy = np.sum(np.abs(coefficients) ** 2, axis=0)
        else:
            removed = np.concatenate(
                (vectors[:, :low_cutoff], vectors[:, high:]), axis=1
            )
            coefficients = removed.conj().T @ signals
            energy = np.einsum("tn,tn->n", signals.real, signals.real, dtype=np.float64)
            energy += np.einsum(
                "tn,tn->n", signals.imag, signals.imag, dtype=np.float64
            )
            energy -= np.sum(np.abs(coefficients) ** 2, axis=0)
            if np.any(energy < 0):
                warnings.warn(
                    "Energy subtraction produced negative power; use method='retained' "
                    "to avoid cancellation. Values have not been clipped.",
                    RuntimeWarning,
                    stacklevel=2,
                )
    return (energy / t).reshape((1, *block.shape[1:]))


def process_iq_to_power_doppler(
    iq: xr.DataArray,
    clutter_window_width: int | None = None,
    clutter_window_stride: int | None = None,
    clutter_mask: xr.DataArray | None = None,
    low_cutoff: int = 0,
    high_cutoff: int | None = None,
    method: Literal["subtract", "retained"] = "subtract",
) -> xr.DataArray:
    """Process IQ VoxelData into one power Doppler volume per clutter window.

    Parameters
    ----------
    iq : xarray.DataArray
        Complex VoxelData array with dimensions `(time, k, j, i)`.
    clutter_window_width : int, optional
        Frames per decomposition and integration window. If not provided, use
        the input temporal chunk size, or all frames for an eager array.
    clutter_window_stride : int, optional
        Frames between window starts, at most the window width. If not provided,
        use non-overlapping windows. Incomplete trailing windows are dropped
        with the standard ConfUSIus warning.
    clutter_mask : xarray.DataArray, optional
        Boolean spatial VoxelData array aligned with `iq`, used only to estimate
        the basis. If not provided, use all voxels. Selected voxel count must be
        at least the clutter window width.
    low_cutoff : int, default: 0
        Inclusive first retained component, ordered by decreasing energy.
    high_cutoff : int, optional
        Exclusive last retained component. If not provided, retain through the
        last temporal component.
    method : {"subtract", "retained"}, default: "subtract"
        Power computation strategy; see `compute_power_doppler_volume`.

    Returns
    -------
    xarray.DataArray
        Dask-backed power Doppler VoxelData array with dimensions
        `(time, k, j, i)`, input geometry, and updated acquisition timings.
        Like the existing processing pipeline, constructing the graph evaluates
        initial blocks to determine output metadata.

    Raises
    ------
    TypeError
        If IQ is not complex-valued or the mask is not boolean.
    ValueError
        If VoxelData, mask alignment, window sizes, method, or cutoffs are invalid.

    Notes
    -----
    Only index-based SVD filtering is supported. Integration always spans the
    entire clutter window; shorter or weighted integration is not supported.
    """
    iq = ensure_voxeldata(
        iq,
        require_time=True,
        allow_pose=False,
        allow_extra_dims=False,
        require_dtype=np.complexfloating,
    )
    mask = None
    if clutter_mask is not None:
        clutter_mask = ensure_voxeldata(
            clutter_mask,
            allow_pose=False,
            allow_extra_dims=False,
        )
        validate_mask(clutter_mask, iq, "clutter_mask")
        mask = clutter_mask.values
    data = da.asarray(iq.data)
    width = data.chunksize[0] if clutter_window_width is None else clutter_window_width
    stride = width if clutter_window_stride is None else clutter_window_stride
    if not isinstance(width, int) or not 2 <= width <= iq.sizes["time"]:
        raise ValueError(
            "clutter_window_width must be an integer between 2 and the frame count."
        )
    if not isinstance(stride, int) or not 1 <= stride <= width:
        raise ValueError(
            "clutter_window_stride must be an integer between 1 and the window width."
        )
    result = process_iq_blocks(
        data,
        process_func=compute_power_doppler_volume,
        window_width=width,
        window_stride=stride,
        clutter_mask=mask,
        low_cutoff=low_cutoff,
        high_cutoff=high_cutoff,
        method=method,
    )
    times, durations = compute_processed_volume_timings(
        iq,
        clutter_window_width=width,
        clutter_window_stride=stride,
        inner_window_width=width,
        inner_window_stride=width,
    )
    duration = float(np.median(durations))
    if not np.allclose(durations, duration, rtol=1e-2, atol=0):
        warnings.warn(
            "Integration duration varies; storing the median duration.", stacklevel=2
        )
    time = xr.DataArray(
        times,
        dims="time",
        attrs={
            **iq.time.attrs,
            "volume_acquisition_duration": duration,
        },
    )
    # Slicing a template preserves the voxel-to-world index, unlike copying coordinates.
    starts = np.arange(0, iq.sizes["time"] - width + 1, stride)
    output = iq.isel(time=starts).copy(data=result).assign_coords(time=time)
    output.name = "power_doppler"
    output.attrs = {
        **iq.attrs,
        "units": "a.u.",
        "long_name": "Power Doppler intensity",
        "cmap": "gray",
        "power_doppler_method": method,
        "clutter_filters": f"Index-based SVD [{low_cutoff}, {high_cutoff or width}[",
        "power_doppler_integration_duration": duration,
        "clutter_filter_window_duration": duration,
    }
    return output

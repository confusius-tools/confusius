"""Utilities for processing beamformed IQ data."""

import warnings
from typing import TYPE_CHECKING, Any, Callable, Concatenate, Literal, cast

import numpy as np
import numpy.typing as npt
import xarray as xr

from confusius._utils import find_stack_level
from confusius.iq.clutter_filters import (
    clutter_filter_butterworth,
    clutter_filter_svd_from_cumulative_energy,
    clutter_filter_svd_from_energy,
    clutter_filter_svd_from_indices,
)
from confusius.validation import validate_iq, validate_mask

if TYPE_CHECKING:
    import dask.array as da


def compute_processed_volume_times(
    volume_times: npt.ArrayLike,
    n_input_volumes: int,
    clutter_window_width: int,
    clutter_window_stride: int,
    inner_window_width: int,
    inner_window_stride: int,
    timing_reference: Literal["start", "center", "end"] = "center",
) -> npt.NDArray[np.floating]:
    """Compute timestamps for processed IQ volumes.

    Given the timestamps of input IQ volumes and the windowing parameters used
    in reduction functions, computes the timestamps for each output volume.

    The reduction functions use nested sliding windows:

    1. **Outer windows** (clutter filtering): Defined by `clutter_window_width` and
       `clutter_window_stride`.
    2. **Inner windows** (Doppler/velocity): Within each outer window, defined by
       `inner_window_width` and `inner_window_stride`.

    Each output volume corresponds to one inner window. This function computes the
    timestamp for each output volume based on the provided `timing_reference`.

    Parameters
    ----------
    volume_times : array_like
        Timestamps of the input IQ volumes, in seconds.
    n_input_volumes : int
        Number of input IQ volumes. Used to compute window counts.
    clutter_window_width : int
        Width of the outer (clutter filtering) window, in volumes.
    clutter_window_stride : int
        Stride of the outer window, in volumes.
    inner_window_width : int
        Width of the inner (Doppler/velocity) window, in volumes.
    inner_window_stride : int
        Stride of the inner window, in volumes.
    timing_reference : {"start", "center", "end"}, default: "center"
        Which point in the inner window to use as the timestamp:

        - `"start"`: Use the timestamp of the first volume in the window.
        - `"center"`: Use the timestamp at the center of the window.
        - `"end"`: Use the timestamp of the last volume in the window.

    Returns
    -------
    numpy.ndarray
        Timestamps for each output volume, in the same units as `volume_times`.

    Examples
    --------
    >>> import numpy as np
    >>> from confusius.iq import compute_processed_volume_times
    >>> # 100 volumes at 10 Hz (0.1s spacing)
    >>> volume_times = np.arange(100) * 0.1
    >>> output_times = compute_processed_volume_times(
    ...     volume_times,
    ...     n_input_volumes=100,
    ...     clutter_window_width=50,
    ...     clutter_window_stride=50,
    ...     inner_window_width=50,
    ...     inner_window_stride=50,
    ...     timing_reference="center",
    ... )
    >>> output_times
    array([2.45, 7.45])
    """
    volume_times = np.asarray(volume_times)

    n_outer_windows = (
        n_input_volumes - clutter_window_width
    ) // clutter_window_stride + 1

    n_inner_windows = (
        clutter_window_width - inner_window_width
    ) // inner_window_stride + 1

    output_times = []

    for outer_idx in range(n_outer_windows):
        outer_start = outer_idx * clutter_window_stride

        for inner_idx in range(n_inner_windows):
            inner_start = outer_start + inner_idx * inner_window_stride

            if timing_reference == "start":
                ref_idx: int | float = inner_start
            elif timing_reference == "center":
                ref_idx = inner_start + (inner_window_width - 1) / 2
            elif timing_reference == "end":
                ref_idx = inner_start + inner_window_width - 1
            else:
                raise ValueError(
                    f"Unknown timing_reference: {timing_reference}. "
                    "Must be 'start', 'center', or 'end'."
                )

            # Interpolate timestamp for fractional indices (e.g., window center).
            if ref_idx == int(ref_idx):
                output_times.append(volume_times[int(ref_idx)])
            else:
                low_idx = int(ref_idx)
                high_idx = low_idx + 1
                frac = ref_idx - low_idx
                output_times.append(
                    volume_times[low_idx] * (1 - frac) + volume_times[high_idx] * frac
                )

    return np.array(output_times)


def _process_block_with_clutter_filter(
    block: npt.NDArray,
    process_block_func: Callable[Concatenate[npt.NDArray, ...], npt.NDArray],
    clutter_mask: npt.NDArray | None = None,
    low_cutoff: int | float | None = None,
    high_cutoff: int | float | None = None,
    window_width: int | None = None,
    window_stride: int | None = None,
    filter_method: Literal[
        "svd_indices", "svd_energy", "svd_cumulative_energy", "butterworth"
    ] = "svd_indices",
    fs: float | None = None,
    butterworth_order: int = 4,
    **kwargs: Any,
) -> npt.NDArray:
    """Process a block of IQ data into new volumes using clutter filtering.

    Four clutter filters are available (see `filter_method`). After clutter filtering,
    reduction is performed using a sliding window across volumes and a user-provided
        processing function (`process_block_func`).

    Parameters
    ----------
    block : (time, z, y, x) numpy.ndarray
        Array of complex IQ data.
    process_block_func : callable
        Function used to process a set of IQ volumes into a single volume.
        `process_block_func` must accept a `(time, z, y, x)` array of complex IQ
        data as first argument and return a `(z, y, x)` array corresponding to the
        processed volume. `process_block_func` may accept additional arguments
        provided via `kwargs`, but must ignore any extra keyword arguments. Note that
        `fs` will be passed to `process_block_func`.
    clutter_mask : (z, y, x) numpy.ndarray, optional
        Boolean mask to define clutter regions. Only used by SVD-based clutter filters
        to compute clutter vectors from masked voxels. If not provided, all voxels are
        used.
    low_cutoff : int or float, optional
        Low cutoff of the clutter filter. See `filter_method` for details. If not
        provided, the lower bound of the range is used.
    high_cutoff : int or float, optional
        High cutoff of the clutter filter. See `filter_method` for details. If not
        provided, the upper bound of the range is used.
    window_width : int, optional
        Width of the sliding window. If not provided, the window width will be set to
        the number of IQ volumes.
    window_stride : int, optional
        Stride of the sliding window. If not provided, `window_stride` will be set to
        `window_width`.
    filter_method : {"svd_indices", "svd_energy", "svd_cumulative_energy", "butterworth"}, \
            default: "svd_indices"
        `"svd_indices"`:
            By default, `filter_method` is `"svd_indices"` to use a static SVD clutter
            filter based on singular vector indices. Assuming singular vectors in
            decreasing singular value order, singular vectors with indices outside
            `[low_cutoff, high_cutoff[` are regressed out.

        `"svd_energy"`:
            The `"svd_energy"` clutter filter is an adaptive SVD clutter filter based
            on singular vector energy. Singular vectors with energy outside
            `[low_cutoff, high_cutoff]` are regressed out.

        `"svd_cumulative_energy"`:
            The `"svd_cumulative_energy"` clutter filter is an adaptive SVD clutter filter
            based on singular vector cumulative energy. Singular vectors with cumulative
            energy outside `[low_cutoff, high_cutoff]` are regressed out.

        `"butterworth"`:
            The `"butterworth"` clutter filter uses a Butterworth low-pass, high-pass,
            or band-pass filter.
    fs : float, optional
        When using the Butterworth clutter filter, the sampling frequency, in Hertz.
    butterworth_order : int, default: 4
        When using the Butterworth clutter filter, the order of the filter.
    kwargs
        Keyword arguments passed to `process_block_func`.

    Returns
    -------
    (windows, z, y, x) numpy.ndarray
        The computed volumes, with `windows` the number of sliding windows used.
    """
    if window_width is None:
        window_width = block.shape[0]
    if window_stride is None:
        window_stride = window_width

    if filter_method == "butterworth":
        if fs is None:
            raise ValueError(
                "When using the Butterworth clutter filter, the sampling frequency "
                "must be provided."
            )

        filtered_block = clutter_filter_butterworth(
            block,
            fs=fs,
            low_cutoff=low_cutoff,
            high_cutoff=high_cutoff,
            order=butterworth_order,
        )
    elif filter_method == "svd_indices":
        if not (
            (isinstance(low_cutoff, int) or low_cutoff is None)
            and (isinstance(high_cutoff, int) or high_cutoff is None)
        ):
            raise ValueError(
                "When using the static SVD clutter filter, low and high cutoffs must "
                "both be integers or None."
            )

        filtered_block = clutter_filter_svd_from_indices(
            block,
            mask=clutter_mask,
            low_cutoff=low_cutoff,
            high_cutoff=high_cutoff,
        )
    elif filter_method == "svd_energy":
        filtered_block = clutter_filter_svd_from_energy(
            block,
            mask=clutter_mask,
            low_cutoff=low_cutoff,
            high_cutoff=high_cutoff,
        )
    elif filter_method == "svd_cumulative_energy":
        filtered_block = clutter_filter_svd_from_cumulative_energy(
            block,
            mask=clutter_mask,
            low_cutoff=low_cutoff,
            high_cutoff=high_cutoff,
        )
    else:
        raise ValueError(f"Unknown clutter filter method: {filter_method}.")

    block_slicer = [
        np.s_[np.arange(i, i + window_width), ...]
        for i in range(0, block.shape[0] - window_width + 1, window_stride)
    ]
    volumes = np.stack(
        [
            process_block_func(filtered_block[sl], fs=fs, **kwargs)
            for sl in block_slicer
        ],
        axis=0,
    )

    return volumes


def compute_power_doppler_volume(
    block: npt.NDArray,
    filter_method: Literal[
        "svd_indices", "svd_energy", "svd_cumulative_energy", "butterworth"
    ] = "svd_indices",
    clutter_mask: npt.NDArray | None = None,
    low_cutoff: int | float | None = None,
    high_cutoff: int | float | None = None,
    fs: float | None = None,
    butterworth_order: int = 4,
    doppler_window_width: int | None = None,
    doppler_window_stride: int | None = None,
) -> npt.NDArray:
    """Compute power Doppler volumes from beamformed IQ data.

    This function computes power Doppler volumes by first applying clutter filtering to
    remove tissue clutter, then averaging the squared magnitude of the filtered IQ data
    within sliding temporal windows.

    Parameters
    ----------
    block : (time, z, y, x) numpy.ndarray
        Complex beamformed IQ data, where `time` is the temporal dimension and
        `(z, y, x)` are spatial dimensions.
    filter_method : {"svd_indices", "svd_energy", "svd_cumulative_energy", "butterworth"}, \
            default: "svd_indices"
        Clutter filtering method to apply before power Doppler computation.

        - `"svd_indices"`: Static SVD filter using singular vector indices.
        - `"svd_energy"`: Adaptive SVD filter using singular vector energies.
        - `"svd_cumulative_energy"`: Adaptive SVD filter using cumulative energies.
        - `"butterworth"`: Butterworth frequency-domain filter.

    clutter_mask : (z, y, x) numpy.ndarray, optional
        Boolean mask to define clutter regions. Only used by SVD-based clutter filters
        to compute clutter vectors from masked voxels. If not provided, all voxels are
        used.
    low_cutoff : int or float, optional
        Low cutoff of the clutter filter. See `filter_method` for details. If not
        provided, the lower bound of the range is used.
    high_cutoff : int or float, optional
        High cutoff of the clutter filter. See `filter_method` for details. If not
        provided, the upper bound of the range is used.
    fs : float, optional
        When using the Butterworth clutter filter, sampling frequency in Hertz.
    butterworth_order : int, default: 4
        Order of Butterworth filter. Effective order is doubled due to forward-backward
        filtering.
    doppler_window_width : int, optional
        Width of the sliding temporal window for power Doppler integration, in volumes.
        If not provided, uses all available volumes.
    doppler_window_stride : int, optional
        Stride of the sliding temporal window, in volumes. If not provided, equals
        `doppler_window_width`.

    Returns
    -------
    (windows, z, y, x) numpy.ndarray
        Power Doppler volumes, where `windows` is the number of temporal sliding
        windows and `(z, y, x)` are spatial dimensions.
    """

    def process_block_func(block: npt.NDArray, **_: Any) -> npt.NDArray:
        """Compute power Doppler signals from an array of IQ data.

        Parameters
        ----------
        block : (time, z, y, x) numpy.ndarray
            Complex IQ data.
        **_
            Additional unused keyword arguments (absorbed and ignored).

        Returns
        -------
        (z, y, x) numpy.ndarray
            Power Doppler volume.
        """
        return np.mean(np.abs(block) ** 2, axis=0)

    return _process_block_with_clutter_filter(
        block=block,
        process_block_func=process_block_func,
        clutter_mask=clutter_mask,
        low_cutoff=low_cutoff,
        high_cutoff=high_cutoff,
        window_width=doppler_window_width,
        window_stride=doppler_window_stride,
        filter_method=filter_method,
        fs=fs,
        butterworth_order=butterworth_order,
    )


def compute_axial_velocity_volume(
    block: npt.NDArray,
    fs: float,
    filter_method: Literal[
        "svd_indices", "svd_energy", "svd_cumulative_energy", "butterworth"
    ] = "svd_indices",
    clutter_mask: npt.NDArray | None = None,
    low_cutoff: int | float | None = None,
    high_cutoff: int | float | None = None,
    butterworth_order: int = 4,
    velocity_window_width: int | None = None,
    velocity_window_stride: int | None = None,
    lag: int = 1,
    absolute_velocity: bool = False,
    spatial_kernel: int = 1,
    ultrasound_frequency: float = 15.625e6,
    sound_velocity: float = 1540,
    estimation_method: Literal["average_angle", "angle_average"] = "average_angle",
) -> npt.NDArray:
    """Compute axial velocity volumes from beamformed IQ data.

    This function computes axial blood flow velocity volumes by first applying clutter
    filtering to remove tissue signals, then estimating velocity using the Kasai
    autocorrelation method within sliding temporal windows. Axial velocity imaging
    measures blood flow velocity along the ultrasound beam direction.

    Parameters
    ----------
    block : (time, z, y, x) numpy.ndarray
        Complex beamformed IQ data, where `time` is the temporal dimension and
        `(z, y, x)` are spatial dimensions.
    fs : float
        Volume sampling frequency in Hertz.
    filter_method : {"svd_indices", "svd_energy", "svd_cumulative_energy", "butterworth"}, \
            default: "svd_indices"
        Clutter filtering method to apply before velocity computation.

        - `"svd_indices"`: Static SVD filter using singular vector indices
        - `"svd_energy"`: Adaptive SVD filter using singular vector energies
        - `"svd_cumulative_energy"`: Adaptive SVD filter using cumulative energies
        - `"butterworth"`: Butterworth frequency-domain filter

    clutter_mask : (z, y, x) numpy.ndarray, optional
        Boolean mask to define clutter regions. Only used by SVD-based clutter filters
        to compute clutter vectors from masked voxels. If not provided, all voxels are
        used.
    low_cutoff : int or float, optional
        Low cutoff of the clutter filter. See `filter_method` for details. If not
        provided, the lower bound of the range is used.
    high_cutoff : int or float, optional
        High cutoff of the clutter filter. See `filter_method` for details. If not
        provided, the upper bound of the range is used.
    butterworth_order : int, default: 4
        Order of Butterworth filter. Effective order is doubled due to forward-backward
        filtering.
    velocity_window_width : int, optional
        Width of the sliding temporal window for velocity estimation, in volumes.
        If not provided, uses all available volumes.
    velocity_window_stride : int, optional
        Stride of the sliding temporal window, in volumes. If not provided, equals
        `velocity_window_width`.
    lag : int, default: 1
        Temporal lag in volumes for autocorrelation computation. Must be positive.
    absolute_velocity : bool, default: False
        If `True`, compute absolute velocity values. If `False`, preserve sign
        information.
    spatial_kernel : int, default: 1
        Size of the median filter kernel applied spatially to denoise. Must be
        positive and odd. If `1`, no spatial filtering is applied.
    ultrasound_frequency : float, default: 15.625e6
        Probe central frequency in Hertz.
    sound_velocity : float, default: 1540
        Speed of sound in the imaged medium, in meters per second.
    estimation_method : {"average_angle", "angle_average"}, default: "average_angle"
        Method for computing the velocity estimate.

        - `"average_angle"`: Compute the angle of the autocorrelation, then average
          (i.e., average of angles).
        - `"angle_average"`: Average the autocorrelation, then compute the angle
          (i.e., angle of average).

    Returns
    -------
    (windows, z, y, x) numpy.ndarray
        Axial velocity volumes, where `windows` is the number of temporal sliding
        windows and `(z, y, x)` are spatial dimensions. Velocity values are in meters
        per second.

    Notes
    -----
    The Kasai estimator computes velocity from the phase shift of the autocorrelation
    function between consecutive IQ volumes.
    """

    def process_block_func(
        block: npt.NDArray,
        spatial_kernel: int,
        lag: int,
        absolute_velocity: bool,
        fs: float,
        ultrasound_frequency: float,
        sound_velocity: float,
        estimation_method: Literal["average_angle", "angle_average"],
        **_,
    ) -> npt.NDArray:
        """Compute the Kasai estimator from an array of IQ data.

        Parameters
        ----------
        block : (time, z, y, x) numpy.ndarray
            Complex IQ data.
        spatial_kernel : int
            Size of the median filter kernel applied spatially to denoise. Must be
            positive and odd.
        lag : int
            Temporal lag in volumes for autocorrelation computation.
        absolute_velocity : bool
            If `True`, compute absolute velocity values. If `False`, preserve sign
            information.
        fs : float
            Volume sampling frequency in Hertz.
        ultrasound_frequency : float
            Probe central frequency in Hertz.
        sound_velocity : float
            Speed of sound in the imaged medium, in meters per second.
        estimation_method : {"average_angle", "angle_average"}
            Method for computing the velocity estimate.
        **_
            Additional unused keyword arguments (absorbed and ignored).

        Returns
        -------
        (z, y, x) numpy.ndarray
            Axial velocity volume in meters per second.
        """
        block_rolled_conjugate = np.roll(block, lag, axis=0).conj()
        block_rolled_conjugate[:lag, ...] = 0
        autocorrelation = cast("npt.NDArray", block * block_rolled_conjugate)[lag:]

        if estimation_method == "average_angle":
            autocorrelation_phase = np.angle(autocorrelation)
            if absolute_velocity:
                autocorrelation_phase = np.abs(autocorrelation_phase)
            average_autocorrelation_phase = autocorrelation_phase.mean(0)
        elif estimation_method == "angle_average":
            average_autocorrelation_phase = np.angle(autocorrelation.mean(0))
            if absolute_velocity:
                average_autocorrelation_phase = np.abs(average_autocorrelation_phase)
        else:
            raise ValueError(
                f"Unknown estimation method: {estimation_method}. "
                "Must be 'average_angle' or 'angle_average'."
            )

        if spatial_kernel > 1:
            import scipy.signal as sp_signal

            kernel_size = [min(spatial_kernel, s) for s in block.shape[1:]]
            # Median filter requires odd kernel sizes.
            kernel_size = [s + 1 if s % 2 == 0 else s for s in kernel_size]
            average_autocorrelation_phase = sp_signal.medfilt(
                average_autocorrelation_phase, kernel_size
            )

        return (
            average_autocorrelation_phase
            * fs
            * sound_velocity
            / (4 * np.pi * ultrasound_frequency)
        )

    return _process_block_with_clutter_filter(
        block=block,
        process_block_func=process_block_func,
        clutter_mask=clutter_mask,
        low_cutoff=low_cutoff,
        high_cutoff=high_cutoff,
        window_width=velocity_window_width,
        window_stride=velocity_window_stride,
        filter_method=filter_method,
        butterworth_order=butterworth_order,
        fs=fs,
        ultrasound_frequency=ultrasound_frequency,
        spatial_kernel=spatial_kernel,
        lag=lag,
        absolute_velocity=absolute_velocity,
        sound_velocity=sound_velocity,
        estimation_method=estimation_method,
    )


def process_iq_blocks(
    iq: "da.Array",
    process_func: Callable[Concatenate[npt.NDArray, ...], npt.NDArray],
    window_width: int | None = None,
    window_stride: int | None = None,
    drop_axis: int | tuple[int, ...] | None = None,
    new_axis: int | tuple[int, ...] | None = None,
    **kwargs: Any,
) -> "da.Array":
    """Process blocks of IQ data using sliding windows.

    This function applies a processing operation to IQ data using
    `dask.array.map_overlap` for efficient parallelized processing.

    !!! warning
        Depending on the window width and stride, some input volumes may be dropped if
        they do not fit into a complete window.

    Parameters
    ----------
    iq : (time, z, y, x) dask.array.Array
        Dask array of complex IQ data.
    process_func : callable
        Function to apply to each temporal window. It must accept a ``(window_volumes,
        z, y, x)` array as first argument and return a `(output_volumes, ...)`` array.
    window_width : int, optional
        Width of the sliding temporal window, in volumes. If not provided, uses the
        chunk size along the first dimension.
    window_stride : int, optional
        Stride of the sliding temporal window, in volumes. Must be less than or equal
        to `window_width`. If not provided, equals `window_width`.
    drop_axis : int or tuple[int, ...], optional
        Axes dropped by `process_func`.
    new_axis : int or tuple[int, ...], optional
        New axes added by `process_func`.
    **kwargs
        Additional keyword arguments passed to `process_func`.

    Returns
    -------
    dask.array.Array
        Processed array.
    """
    import dask.array as da

    if window_width is None:
        window_width = cast(int, iq.chunksize[0])  # type: ignore
    if window_stride is None:
        window_stride = window_width

    if window_stride > window_width:
        raise ValueError(
            "`window_stride` must be less than or equal to `window_width`."
        )

    n_volumes = iq.shape[0]
    overlap_width = window_width - window_stride

    n_windows = (n_volumes - window_width) // window_stride + 1
    total_frames_in_windows = window_width + (n_windows - 1) * window_stride
    remaining_frames = n_volumes - total_frames_in_windows

    if remaining_frames > 0:
        warnings.warn(
            f"{remaining_frames} input volumes will be dropped because they do not fit "
            "into a complete window. Adjust `window_width` and `window_stride` to avoid "
            "this.",
            stacklevel=find_stack_level(),
        )
        iq = iq[:total_frames_in_windows]

    chunks_volumes = (window_width,) + (window_stride,) * (n_windows - 1)
    iq = iq.rechunk((chunks_volumes,) + iq.shape[1:])

    dummy_result = process_func(iq.blocks[0].compute(), **kwargs)

    return da.map_overlap(
        process_func,
        iq,
        depth={0: (overlap_width, 0)},
        boundary="none",
        trim=False,
        chunks=dummy_result.shape,
        drop_axis=drop_axis,
        new_axis=new_axis,
        meta=np.array((), dtype=dummy_result.dtype),
        **kwargs,
    )


def process_iq_to_power_doppler(
    iq: xr.DataArray,
    clutter_window_width: int | None = None,
    clutter_window_stride: int | None = None,
    filter_method: Literal[
        "svd_indices", "svd_energy", "svd_cumulative_energy", "butterworth"
    ] = "svd_indices",
    clutter_mask: xr.DataArray | None = None,
    low_cutoff: int | float | None = None,
    high_cutoff: int | float | None = None,
    butterworth_order: int = 4,
    doppler_window_width: int | None = None,
    doppler_window_stride: int | None = None,
) -> xr.DataArray:
    """Process blocks of beamformed IQ into power Doppler volumes using sliding windows.

    This function computes power Doppler volumes from blocks of beamformed IQ data using
    nested sliding windows. A first sliding window is used for clutter filtering. Inside
    each clutter-filtered window, power Doppler volumes are computed using a second
    sliding window. See the notes section for an illustration of the nested sliding
    window approach.

    Parameters
    ----------
    iq : xarray.DataArray
        Xarray DataArray containing complex beamformed IQ data with dimensions
        `(time, z, y, x)`, where `time` is the temporal dimension and
        `(z, y, x)` are spatial dimensions. The DataArray should have a
        `compound_sampling_frequency` attribute (required when using the
        Butterworth filter).
    clutter_window_width : int, optional
        Width of the sliding temporal window for clutter filtering, in volumes. If not
        provided, uses the chunk size of the IQ data along the temporal dimension.
    clutter_window_stride : int, optional
        Stride of the sliding temporal window for clutter filtering, in volumes. If not
        provided, equals `clutter_window_width`.
    filter_method : {"svd_indices", "svd_energy", "svd_cumulative_energy", "butterworth"}, \
            default: "svd_indices"
        Clutter filtering method to apply before power Doppler computation.

        - `"svd_indices"`: Static SVD filter using singular vector indices.
        - `"svd_energy"`: Adaptive SVD filter using singular vector energies.
        - `"svd_cumulative_energy"`: Adaptive SVD filter using cumulative energies.
        - `"butterworth"`: Butterworth frequency-domain filter.

    clutter_mask : (z, y, x) xarray.DataArray, optional
        Boolean mask to define clutter regions. Only used by SVD-based clutter filters
        to compute clutter vectors from masked voxels. If not provided, all voxels are
        used. The mask spatial coordinates `(z, y, x)` must match the IQ data
        coordinates.
    low_cutoff : int or float, optional
        Low cutoff of the clutter filter. See `filter_method` for details. If not
        provided, the lower bound of the range is used.
    high_cutoff : int or float, optional
        High cutoff of the clutter filter. See `filter_method` for details. If not
        provided, the upper bound of the range is used.
    butterworth_order : int, default: 4
        Order of Butterworth filter. Effective order is doubled due to forward-backward
        filtering.
    doppler_window_width : int, optional
        Width of the sliding temporal window for power Doppler integration, in volumes.
        If not provided, equals `clutter_window_width`.
    doppler_window_stride : int, optional
        Stride of the sliding temporal window for power Doppler integration, in volumes.
        If not provided, equals `doppler_window_width`.

    Returns
    -------
    (clutter_windows * doppler_windows, z, y, x) xarray.DataArray
        Power Doppler volumes as an Xarray DataArray with updated time coordinates,
        where `clutter_windows` is the number of clutter filter sliding windows and
        `doppler_windows` is the number of power Doppler sliding windows per clutter
        window.

    Notes
    -----
    The nested sliding window approach can be visualized as follows::

        Input IQ volumes (temporal dimension):
        [0][1][2][3][4][5][6][7][8][9][10][11]

        ┌─────────────────────────────────────────────────────────────────────┐
        │ OUTER WINDOWS: Clutter Filtering (width=6, stride=3)                │
        └─────────────────────────────────────────────────────────────────────┘

         Window 1:  [0][1][2][3][4][5][6][7][8][9][10][11]
                    [════════════════]
                            │
                            ├─ Clutter filter applied
                            │
                            └─ INNER WINDOWS: Power Doppler (width=3, stride=2)
                               ┌───────────────────┐
                               │[0][1][2] ──> PD output 1
                               │      [2][3][4] ──> PD output 2
                               │            [4][5] ──> (incomplete, dropped)
                               └───────────────────┘

         Window 2:  [0][1][2][3][4][5][6][7][8][9][10][11]
                             [════════════════]
                                     │
                                     ├─ Clutter filter applied
                                     │
                                     └─ INNER WINDOWS: Power Doppler (width=3, stride=2)
                                        ┌───────────────────┐
                                        │[3][4][5] ──> PD output 3
                                        │      [5][6][7] ──> PD output 4
                                        │            [7][8] ──> (incomplete, dropped)
                                        └───────────────────┘

         Window 3:  [0][1][2][3][4][5][6][7][8][9][10][11]
                                      [══════════════════]
                                               │
                                                └─ (...)
    """
    import dask.array as da
    from dask.array import Array

    validate_iq(iq)

    clutter_mask_array = None
    if clutter_mask is not None:
        validate_mask(clutter_mask, iq, "clutter_mask")
        clutter_mask_array = clutter_mask.values

    dask_iq: Array = iq.data
    if not isinstance(dask_iq, Array):
        dask_iq = da.from_array(dask_iq)

    if clutter_window_width is None:
        clutter_window_width = cast(int, dask_iq.chunksize[0])
    if clutter_window_stride is None:
        clutter_window_stride = clutter_window_width
    if doppler_window_width is None:
        doppler_window_width = clutter_window_width
    if doppler_window_stride is None:
        doppler_window_stride = doppler_window_width

    # Validation ensures fs is present and of the correct type (required for Butterworth
    # filter).
    fs = iq.attrs.get("compound_sampling_frequency")

    result = process_iq_blocks(
        dask_iq,
        process_func=compute_power_doppler_volume,
        window_width=clutter_window_width,
        window_stride=clutter_window_stride,
        filter_method=filter_method,
        clutter_mask=clutter_mask_array,
        low_cutoff=low_cutoff,
        high_cutoff=high_cutoff,
        fs=fs,
        butterworth_order=butterworth_order,
        doppler_window_width=doppler_window_width,
        doppler_window_stride=doppler_window_stride,
    )

    output_times_values = compute_processed_volume_times(
        volume_times=iq.coords["time"].values,
        n_input_volumes=dask_iq.shape[0],
        clutter_window_width=clutter_window_width,
        clutter_window_stride=clutter_window_stride,
        inner_window_width=doppler_window_width,
        inner_window_stride=doppler_window_stride,
    )
    output_times = xr.DataArray(
        output_times_values,
        dims="time",
        attrs=iq.coords["time"].attrs,
    )

    output_attrs = {
        "units": "a.u.",
        "long_name": "Power Doppler intensity",
        "cmap": "gray",
        "clutter_filter_method": filter_method,
        "clutter_window_width": clutter_window_width,
        "clutter_window_stride": clutter_window_stride,
        "doppler_window_width": doppler_window_width,
        "doppler_window_stride": doppler_window_stride,
    }
    if low_cutoff is not None:
        output_attrs["clutter_low_cutoff"] = low_cutoff
    if high_cutoff is not None:
        output_attrs["clutter_high_cutoff"] = high_cutoff

    return xr.DataArray(
        result,
        name="power_doppler",
        dims=iq.dims,
        coords={
            "time": output_times,
            "z": iq.coords["z"],
            "y": iq.coords["y"],
            "x": iq.coords["x"],
        },
        attrs={**iq.attrs, **output_attrs},
    )


def process_iq_to_axial_velocity(
    iq: xr.DataArray,
    clutter_window_width: int | None = None,
    clutter_window_stride: int | None = None,
    filter_method: Literal[
        "svd_indices", "svd_energy", "svd_cumulative_energy", "butterworth"
    ] = "svd_indices",
    clutter_mask: xr.DataArray | None = None,
    low_cutoff: int | float | None = None,
    high_cutoff: int | float | None = None,
    butterworth_order: int = 4,
    velocity_window_width: int | None = None,
    velocity_window_stride: int | None = None,
    lag: int = 1,
    absolute_velocity: bool = False,
    spatial_kernel: int = 1,
    estimation_method: Literal["average_angle", "angle_average"] = "average_angle",
) -> xr.DataArray:
    """Process blocks of beamformed IQ into axial velocity volumes using sliding windows.

    This function computes axial velocity volumes from blocks of beamformed IQ data using
    nested sliding windows. A first sliding window is used for clutter filtering. Inside
    each clutter-filtered window, axial velocity volumes are computed using a second
    sliding window. See the notes section for an illustration of the nested sliding
    window approach.

    Parameters
    ----------
    iq : xarray.DataArray
        Xarray DataArray containing complex beamformed IQ data with dimensions
        `(time, z, y, x)`, where `time` is the temporal dimension and
        `(z, y, x)` are spatial dimensions. The DataArray must have the
        following attributes:

        - `compound_sampling_frequency`: Volume acquisition rate in Hz.
        - `transmit_frequency`: Ultrasound probe central frequency in Hz.
        - `sound_velocity`: Speed of sound in the imaged medium in m/s.
    clutter_window_width : int, optional
        Width of the sliding temporal window for clutter filtering, in volumes. If not
        provided, uses the chunk size of the IQ data along the temporal dimension.
    clutter_window_stride : int, optional
        Stride of the sliding temporal window for clutter filtering, in volumes. If not
        provided, equals `clutter_window_width`.
    filter_method : {"svd_indices", "svd_energy", "svd_cumulative_energy", "butterworth"}, \
            default: "svd_indices"
        Clutter filtering method to apply before velocity computation.

        - `"svd_indices"`: Static SVD filter using singular vector indices.
        - `"svd_energy"`: Adaptive SVD filter using singular vector energies.
        - `"svd_cumulative_energy"`: Adaptive SVD filter using cumulative energies.
        - `"butterworth"`: Butterworth frequency-domain filter.

    clutter_mask : (z, y, x) xarray.DataArray, optional
        Boolean mask to define clutter regions. Only used by SVD-based clutter filters
        to compute clutter vectors from masked voxels. If not provided, all voxels are
        used. The mask spatial coordinates `(z, y, x)` must match the IQ data
        coordinates.
    low_cutoff : int or float, optional
        Low cutoff of the clutter filter. See `filter_method` for details. If not
        provided, the lower bound of the range is used.
    high_cutoff : int or float, optional
        High cutoff of the clutter filter. See `filter_method` for details. If not
        provided, the upper bound of the range is used.
    butterworth_order : int, default: 4
        Order of Butterworth filter. Effective order is doubled due to forward-backward
        filtering.
    velocity_window_width : int, optional
        Width of the sliding temporal window for velocity estimation, in volumes.
        If not provided, equals `clutter_window_width`.
    velocity_window_stride : int, optional
        Stride of the sliding temporal window for velocity estimation, in volumes.
        If not provided, equals `velocity_window_width`.
    lag : int, default: 1
        Temporal lag in volumes for autocorrelation computation. Must be positive.
    absolute_velocity : bool, default: False
        If `True`, compute absolute velocity values. If `False`, preserve sign
        information.
    spatial_kernel : int, default: 1
        Size of the median filter kernel applied spatially to denoise. Must be
        positive and odd. If `1`, no spatial filtering is applied.
    estimation_method : {"average_angle", "angle_average"}, default: "average_angle"
        Method for computing the velocity estimate.

        - `"average_angle"`: Compute the angle of the autocorrelation, then average
          (i.e., average of angles).
        - `"angle_average"`: Average the autocorrelation, then compute the angle
          (i.e., angle of average).

    Returns
    -------
    (clutter_windows * velocity_windows, z, y, x) xarray.DataArray
        Axial velocity volumes as an Xarray DataArray with updated time coordinates,
        where `clutter_windows` is the number of clutter filter sliding windows and
        `velocity_windows` is the number of velocity sliding windows per clutter window.
        Velocity values are in meters per second.

    Notes
    -----
    The nested sliding window approach can be visualized as follows::

        Input IQ volumes (temporal dimension):
        [0][1][2][3][4][5][6][7][8][9][10][11]

        ┌─────────────────────────────────────────────────────────────────────┐
        │ OUTER WINDOWS: Clutter filtering (width=6, stride=3)                │
        └─────────────────────────────────────────────────────────────────────┘

         Window 1:  [0][1][2][3][4][5][6][7][8][9][10][11]
                    [════════════════]
                            │
                            ├─ Clutter filter applied
                            │
                            └─ INNER WINDOWS: Axial velocity (width=3, stride=2)
                               ┌───────────────────┐
                               │[0][1][2] ──> AV output 1
                               │      [2][3][4] ──> AV output 2
                               │            [4][5] ──> (incomplete, dropped)
                               └───────────────────┘

         Window 2:  [0][1][2][3][4][5][6][7][8][9][10][11]
                             [════════════════]
                                     │
                                     ├─ Clutter filter applied
                                     │
                                     └─ INNER WINDOWS: Axial velocity (width=3, stride=2)
                                        ┌───────────────────┐
                                        │[3][4][5] ──> AV output 3
                                        │      [5][6][7] ──> AV output 4
                                        │            [7][8] ──> (incomplete, dropped)
                                        └───────────────────┘

         Window 3:  [0][1][2][3][4][5][6][7][8][9][10][11]
                                      [══════════════════]
                                               │
                                               └─ (...)
    """
    import dask.array as da
    from dask.array import Array

    validate_iq(iq)

    clutter_mask_array = None
    if clutter_mask is not None:
        validate_mask(clutter_mask, iq, "clutter_mask")
        clutter_mask_array = clutter_mask.values

    dask_iq: Array = iq.data
    if not isinstance(dask_iq, Array):
        dask_iq = da.from_array(dask_iq)

    if clutter_window_width is None:
        clutter_window_width = cast(int, dask_iq.chunksize[0])
    if clutter_window_stride is None:
        clutter_window_stride = clutter_window_width
    if velocity_window_width is None:
        velocity_window_width = clutter_window_width
    if velocity_window_stride is None:
        velocity_window_stride = velocity_window_width

    # Validation ensures these attributes are present and of the correct type.
    fs = iq.attrs["compound_sampling_frequency"]
    ultrasound_frequency = iq.attrs["transmit_frequency"]
    sound_velocity = iq.attrs["sound_velocity"]

    result = process_iq_blocks(
        dask_iq,
        process_func=compute_axial_velocity_volume,
        window_width=clutter_window_width,
        window_stride=clutter_window_stride,
        fs=fs,
        filter_method=filter_method,
        clutter_mask=clutter_mask_array,
        low_cutoff=low_cutoff,
        high_cutoff=high_cutoff,
        butterworth_order=butterworth_order,
        velocity_window_width=velocity_window_width,
        velocity_window_stride=velocity_window_stride,
        lag=lag,
        absolute_velocity=absolute_velocity,
        spatial_kernel=spatial_kernel,
        ultrasound_frequency=ultrasound_frequency,
        sound_velocity=sound_velocity,
        estimation_method=estimation_method,
    )

    output_times_values = compute_processed_volume_times(
        volume_times=iq.coords["time"].values,
        n_input_volumes=dask_iq.shape[0],
        clutter_window_width=clutter_window_width,
        clutter_window_stride=clutter_window_stride,
        inner_window_width=velocity_window_width,
        inner_window_stride=velocity_window_stride,
    )
    output_times = xr.DataArray(
        output_times_values,
        dims="time",
        attrs=iq.coords["time"].attrs,
    )

    velocity_cmap = "viridis" if absolute_velocity else "coolwarm"
    output_attrs = {
        "units": "m/s",
        "long_name": "Axial velocity",
        "cmap": velocity_cmap,
        "clutter_filter_method": filter_method,
        "clutter_window_width": clutter_window_width,
        "clutter_window_stride": clutter_window_stride,
        "velocity_window_width": velocity_window_width,
        "velocity_window_stride": velocity_window_stride,
        "lag": lag,
        "absolute_velocity": absolute_velocity,
        "spatial_kernel": spatial_kernel,
        "ultrasound_frequency": ultrasound_frequency,
        "sound_velocity": sound_velocity,
        "estimation_method": estimation_method,
    }
    if low_cutoff is not None:
        output_attrs["clutter_low_cutoff"] = low_cutoff
    if high_cutoff is not None:
        output_attrs["clutter_high_cutoff"] = high_cutoff

    return xr.DataArray(
        result,
        name="axial_velocity",
        dims=iq.dims,
        coords={
            "time": output_times,
            "z": iq.coords["z"],
            "y": iq.coords["y"],
            "x": iq.coords["x"],
        },
        attrs={**iq.attrs, **output_attrs},
    )

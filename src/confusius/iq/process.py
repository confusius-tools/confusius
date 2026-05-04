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
from confusius.timing import (
    convert_time_reference,
    get_representative_time_step,
    get_time_coord_to_seconds_factor,
)
from confusius.validation import validate_iq, validate_mask

if TYPE_CHECKING:
    import dask.array as da


def _format_clutter_filter_spec(
    filter_method: Literal[
        "svd_indices", "svd_energy", "svd_cumulative_energy", "butterworth"
    ],
    low_cutoff: int | float | None,
    high_cutoff: int | float | None,
) -> str:
    """Format a human-readable BIDS clutter filter specification.

    Parameters
    ----------
    filter_method : {"svd_indices", "svd_energy", "svd_cumulative_energy", "butterworth"}
        Clutter filter family.
    low_cutoff : int or float or None
        Lower filter cutoff, when applicable.
    high_cutoff : int or float or None
        Upper filter cutoff, when applicable.

    Returns
    -------
    str
        Human-readable clutter filter specification suitable for the fUSI-BIDS
        `ClutterFilters` field.
    """
    labels = {
        "svd_indices": "Index-based SVD",
        "svd_energy": "Energy-based SVD",
        "svd_cumulative_energy": "Cumulative energy SVD",
        "butterworth": "Butterworth",
    }
    label = labels[filter_method]

    if low_cutoff is None and high_cutoff is None:
        return label

    low = "-inf" if low_cutoff is None else f"{low_cutoff:g}"
    high = "+inf" if high_cutoff is None else f"{high_cutoff:g}"
    closing = "[" if filter_method == "svd_indices" else "]"
    return f"{label} [{low}, {high}{closing}"


def _get_volume_acquisition_duration(iq: xr.DataArray) -> float:
    """Return the input IQ volume acquisition duration in coordinate units.

    Prefers the explicit `volume_acquisition_duration` stored on the time coordinate.
    Falls back to the scanner provenance attribute `compound_sampling_frequency`, then
    to the representative time step of the `time` coordinate.

    Parameters
    ----------
    iq : xarray.DataArray
        Input IQ data array.

    Returns
    -------
    float
        Volume acquisition duration in the same units as the `time` coordinate.
    """
    duration = iq.coords["time"].attrs.get("volume_acquisition_duration")
    if isinstance(duration, int | float) and duration > 0:
        return float(duration)

    compound_sampling_frequency = iq.attrs.get("compound_sampling_frequency")
    if (
        isinstance(compound_sampling_frequency, int | float)
        and compound_sampling_frequency > 0
    ):
        warnings.warn(
            "Using `compound_sampling_frequency` to infer `volume_acquisition_duration`. "
            "Prefer an explicit `volume_acquisition_duration` on the time coordinate.",
            stacklevel=find_stack_level(),
        )
        return 1.0 / (
            float(compound_sampling_frequency) * get_time_coord_to_seconds_factor(iq)
        )

    time_step, approximate = get_representative_time_step(iq)
    if time_step is not None:
        warning = (
            "Using the representative `time` coordinate spacing to infer "
            "`volume_acquisition_duration`. Prefer an explicit "
            "`volume_acquisition_duration` on the time coordinate."
        )
        if approximate:
            warning += " Since timings are irregular, the inferred duration is based on a median approximation."
        warnings.warn(warning, stacklevel=find_stack_level())
        return float(time_step)

    raise ValueError(
        "Cannot determine volume acquisition duration: neither "
        "`volume_acquisition_duration` in the time coordinate attributes nor "
        "`compound_sampling_frequency` in the DataArray attributes nor enough `time` "
        "coordinate values are available."
    )


def _get_filter_sampling_frequency(
    iq: xr.DataArray,
    *,
    filter_method: Literal[
        "svd_indices", "svd_energy", "svd_cumulative_energy", "butterworth"
    ],
) -> float | None:
    """Return the sampling frequency to use for clutter filtering.

    Butterworth filtering requires regular sampling. For now this is checked globally on
    the full input time coordinate; a more precise implementation would validate each
    clutter-filter window separately to allow for dead times or irregular sampling
    between windows.
    """
    time_step, approximate = get_representative_time_step(iq, unit="s")
    if time_step is None:
        return None

    if filter_method == "butterworth" and approximate:
        # TODO: Check temporal regularity on each clutter-filter window rather than on
        # the full input time coordinate.
        raise ValueError(
            "Butterworth clutter filtering requires a regularly sampled `time` "
            "coordinate."
        )

    return 1.0 / time_step


def _summarize_window_duration(
    durations: npt.NDArray,
    *,
    description: str,
    uniformity_tolerance: float = 1e-2,
) -> float:
    """Return a representative window duration for metadata.

    When output windows have slightly or substantially different durations, metadata
    fields can only store a single representative value. In that case the median window
    duration is used and a warning is emitted.

    Parameters
    ----------
    durations : array_like
        Durations of the output windows, in coordinate units.
    description : str
        Description of the duration being summarized, used in the warning message if
        durations are not uniform.
    uniformity_tolerance : float, default: 1e-2
        Maximum allowed per-interval relative deviation from the median duration. Each
        duration must satisfy `|duration - median| <= uniformity_tolerance * |median|`
        for the durations to be considered uniform.

    Returns
    -------
    float
        Representative window duration. If durations are uniform within the specified
        tolerance, returns the common duration. Otherwise, returns the median duration
        and emits a warning.
    """
    if durations.size == 1:
        return float(durations[0])

    median = float(np.median(durations))
    is_uniform = bool(
        np.allclose(durations, median, rtol=uniformity_tolerance, atol=0.0)
    )

    if not is_uniform:
        warnings.warn(
            f"{description} varies across output windows. Storing the median duration "
            "in `time.attrs['volume_acquisition_duration']`.",
            stacklevel=find_stack_level(),
        )

    return median


def _compute_clutter_filter_window_metadata(
    iq: xr.DataArray, *, clutter_window_width: int, clutter_window_stride: int
) -> tuple[float, float]:
    """Compute representative clutter-filter window duration and stride metadata.

    Parameters
    ----------
    iq : xarray.DataArray
        Input IQ data.
    clutter_window_width : int
        Width of the clutter-filter window, in volumes.
    clutter_window_stride : int
        Stride of the clutter-filter window, in volumes.

    Returns
    -------
    clutter_window_duration : float
        Representative clutter-filter window duration in the input time-coordinate
        units.
    clutter_window_stride_duration : float
        Representative clutter-filter window stride in the input time-coordinate
        units.
    """
    iq_time_reference = iq.coords["time"].attrs.get(
        "volume_acquisition_reference", "start"
    )
    if iq_time_reference not in {"start", "center", "end"}:
        raise ValueError(
            f"Unknown volume_acquisition_reference: {iq_time_reference!r}. Must be "
            "'start', 'center', or 'end'."
        )

    iq_volume_duration = _get_volume_acquisition_duration(iq)
    iq_volume_timings = np.asarray(iq.coords["time"].values)
    window_starts = convert_time_reference(
        iq_volume_timings,
        iq_volume_duration,
        from_reference=iq_time_reference,
        to_reference="start",
    )

    n_outer_windows = (
        len(iq_volume_timings) - clutter_window_width
    ) // clutter_window_stride + 1
    outer_starts = np.arange(n_outer_windows) * clutter_window_stride
    outer_ends = outer_starts + clutter_window_width - 1

    clutter_window_durations = (
        iq_volume_timings[outer_ends]
        - iq_volume_timings[outer_starts]
        + iq_volume_duration
    )
    clutter_window_duration = _summarize_window_duration(
        clutter_window_durations, description="Clutter filter window duration"
    )

    if n_outer_windows > 1:
        clutter_window_stride_duration = _summarize_window_duration(
            np.diff(window_starts[outer_starts]),
            description="Clutter filter window stride",
        )
    else:
        time_step, _ = get_representative_time_step(iq)
        if time_step is None:
            clutter_window_stride_duration = clutter_window_stride * iq_volume_duration
        else:
            clutter_window_stride_duration = clutter_window_stride * time_step

    return clutter_window_duration, float(clutter_window_stride_duration)


def _compute_inner_window_metadata(
    iq: xr.DataArray,
    *,
    output_timings: npt.NDArray[np.floating],
    output_durations: npt.NDArray[np.floating],
    output_window_stride: int,
    inner_windows_per_outer_window: int,
    duration_description: str,
    stride_description: str,
) -> tuple[float, float]:
    """Compute representative inner-window duration and stride metadata.

    Parameters
    ----------
    iq : xarray.DataArray
        Input IQ data.
    output_timings : array-like
        Output timings for each processed window.
    output_durations : array-like
        Durations for each processed window.
    output_window_stride : int
        Output-window stride in volumes.
    inner_windows_per_outer_window : int
        Number of inner windows produced for each outer window.
    duration_description : str
        Description used when duration values vary across windows.
    stride_description : str
        Description used when stride values vary across windows.

    Returns
    -------
    output_window_duration : float
        Representative output-window duration in time-coordinate units.
    output_window_stride_duration : float
        Representative output-window stride in time-coordinate units.
    """
    output_window_duration = _summarize_window_duration(
        output_durations,
        description=duration_description,
    )

    output_reference = iq.coords["time"].attrs.get(
        "volume_acquisition_reference", "start"
    )
    if output_reference not in {"start", "center", "end"}:
        raise ValueError(
            f"Unknown volume_acquisition_reference: {output_reference!r}. Must be "
            "'start', 'center', or 'end'."
        )

    if output_timings.size > 1:
        output_starts = convert_time_reference(
            output_timings,
            output_durations,
            from_reference=output_reference,
            to_reference="start",
        )
        if inner_windows_per_outer_window > 1:
            n_outer_windows, remainder = divmod(
                output_starts.size, inner_windows_per_outer_window
            )
            if remainder == 0 and n_outer_windows > 0:
                output_starts_per_outer = output_starts.reshape(
                    n_outer_windows, inner_windows_per_outer_window
                )
                stride_values = np.diff(output_starts_per_outer, axis=1).ravel()
            else:
                stride_values = np.diff(output_starts)
        else:
            stride_values = np.diff(output_starts)

        output_window_stride_duration = _summarize_window_duration(
            stride_values, description=stride_description
        )
        return output_window_duration, float(output_window_stride_duration)

    time_step, _ = get_representative_time_step(iq)
    if time_step is None:
        output_window_stride_duration = (
            output_window_stride * _get_volume_acquisition_duration(iq)
        )
    else:
        output_window_stride_duration = output_window_stride * time_step

    return output_window_duration, float(output_window_stride_duration)


def compute_processed_volume_timings(
    iq: xr.DataArray,
    clutter_window_width: int,
    clutter_window_stride: int,
    inner_window_width: int,
    inner_window_stride: int,
    processed_time_reference: Literal["start", "center", "end"] | None = None,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Compute timings from processing input IQ volumes with nested sliding windows.

    The IQ processing functions
    [`process_iq_to_power_doppler`][confusius.iq.process_iq_to_power_doppler] and
    [`process_iq_to_axial_velocity`][confusius.iq.process_iq_to_axial_velocity] use
    nested sliding windows:

    1. **Outer windows** (clutter filtering): Defined by `clutter_window_width` and
       `clutter_window_stride`.
    2. **Inner windows** (Doppler/velocity): Within each outer window, defined by
       `inner_window_width` and `inner_window_stride`.

    This function computes timings and durations for the output volumes corresponding to
    the inner windows, based on the input IQ timing metadata and the windowing
    parameters.

    Parameters
    ----------
    iq : xarray.DataArray
        Input IQ data. Timing information is read from `iq.coords["time"]` and its
        attributes.
    clutter_window_width : int
        Width of the outer (clutter filtering) window, in volumes.
    clutter_window_stride : int
        Stride of the outer window, in volumes.
    inner_window_width : int
        Width of the inner (Doppler/velocity) window, in volumes.
    inner_window_stride : int
        Stride of the inner window, in volumes.
    processed_time_reference : {"start", "center", "end"}, optional
        Which point of the output window bin to use as the output timings:

        - `"start"`: Start of the first volume's bin (onset).
        - `"center"`: Midpoint of the full window bin.
        - `"end"`: End of the last volume's bin.

        If not provided, the input `volume_acquisition_reference` is reused.

    Returns
    -------
    output_timings : numpy.ndarray
        Timings for each output volume, in the same units as `iq.coords["time"]`.
    output_durations : numpy.ndarray
        Duration of each output window, in the same units as `iq.coords["time"]`.

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> from confusius.iq import compute_processed_volume_timings
    >>> # 100 volumes at 10 Hz, volume_duration = 0.1 s
    >>> iq = xr.DataArray(
    ...     np.ones((100, 1, 1, 1), dtype=np.complex128),
    ...     dims=("time", "z", "y", "x"),
    ...     coords={
    ...         "time": xr.DataArray(
    ...             np.arange(100) * 0.1,
    ...             dims=("time",),
    ...             attrs={
    ...                 "units": "s",
    ...                 "volume_acquisition_duration": 0.1,
    ...                 "volume_acquisition_reference": "start",
    ...             },
    ...         ),
    ...         "z": [0],
    ...         "y": [0],
    ...         "x": [0],
    ...     },
    ... )
    >>> output_timings, output_durations = compute_processed_volume_timings(
    ...     iq,
    ...     clutter_window_width=50,
    ...     clutter_window_stride=50,
    ...     inner_window_width=25,
    ...     inner_window_stride=25,
    ...     processed_time_reference="center",
    ... )
    >>> output_timings
    array([1.25, 3.75, 6.25, 8.75])
    >>> output_durations
    array([2.5, 2.5, 2.5, 2.5])
    """
    iq_time_reference = iq.coords["time"].attrs.get(
        "volume_acquisition_reference", "start"
    )
    if processed_time_reference is None:
        processed_time_reference = iq_time_reference
    iq_volume_duration = _get_volume_acquisition_duration(iq)
    iq_volume_timings = np.asarray(iq.coords["time"].values)

    if iq_time_reference not in {"start", "center", "end"}:
        raise ValueError(
            f"Unknown iq_time_reference: {iq_time_reference!r}. Must be 'start', 'center', or 'end'."
        )
    if processed_time_reference not in {"start", "center", "end"}:
        raise ValueError(
            "Unknown processed_time_reference: "
            f"{processed_time_reference!r}. Must be 'start', 'center', or 'end'."
        )

    window_starts = convert_time_reference(
        iq_volume_timings,
        iq_volume_duration,
        from_reference=iq_time_reference,
        to_reference="start",
    )

    n_outer_windows = (
        len(iq_volume_timings) - clutter_window_width
    ) // clutter_window_stride + 1

    n_inner_windows = (
        clutter_window_width - inner_window_width
    ) // inner_window_stride + 1

    outer_starts = np.arange(n_outer_windows) * clutter_window_stride
    inner_offsets = np.arange(n_inner_windows) * inner_window_stride

    # Timings of the first and last IQ volumes in each inner window. Shape:
    # (n_outer_windows, n_inner_windows), then flatten.
    inner_window_first_volume_indices = (
        outer_starts[:, np.newaxis] + inner_offsets[np.newaxis, :]
    ).ravel()
    inner_window_last_volume_indices = (
        inner_window_first_volume_indices + inner_window_width - 1
    )

    inner_window_first_volume_timings = iq_volume_timings[
        inner_window_first_volume_indices
    ]
    inner_window_last_volume_timings = iq_volume_timings[
        inner_window_last_volume_indices
    ]

    inner_window_durations = (
        inner_window_last_volume_timings
        - inner_window_first_volume_timings
        + iq_volume_duration
    )

    # Recover the onset from whatever reference the input timings use, then apply the
    # requested output reference within the actual window span. The span is based on the
    # observed first/last timing plus one volume duration, so it remains correct when
    # there is dead time between acquisitions.
    output_timings = convert_time_reference(
        window_starts[inner_window_first_volume_indices],
        inner_window_durations,
        from_reference="start",
        to_reference=processed_time_reference,
    )
    return output_timings, inner_window_durations


def process_iq_block_with_clutter_filter(
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
        When using the Butterworth clutter filter, the sampling frequency, in hertz.
    butterworth_order : int, default: 4
        When using the Butterworth clutter filter, the order of the filter.
    **kwargs
        Additional keyword arguments passed to `process_block_func`.

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
        When using the Butterworth clutter filter, sampling frequency in hertz.
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

    return process_iq_block_with_clutter_filter(
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


def compute_bmode_volume(
    block: npt.NDArray,
    **_: Any,
) -> npt.NDArray:
    """Compute a B-mode volume from beamformed IQ data.

    This function computes a B-mode volume by averaging the magnitude of the IQ data
    across the temporal dimension. Unlike power Doppler, no clutter filtering is applied
    and the magnitude (not squared magnitude) is averaged.

    Parameters
    ----------
    block : (time, z, y, x) numpy.ndarray
        Complex beamformed IQ data, where `time` is the temporal dimension and
        `(z, y, x)` are spatial dimensions.
    **_
        Additional unused keyword arguments (absorbed and ignored).

    Returns
    -------
    (1, z, y, x) numpy.ndarray
        B-mode volume, computed as the mean magnitude of the IQ data across time.
    """
    return np.mean(np.abs(block), axis=0)[np.newaxis]


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
    transmit_frequency: float = 15.625e6,
    beamforming_sound_velocity: float = 1540,
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
        Volume sampling frequency in hertz.
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
    transmit_frequency : float, default: 15.625e6
        Ultrasound transmit frequency in hertz.
    beamforming_sound_velocity : float, default: 1540
        Speed of sound assumed during beamforming, in meters per second.
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
        transmit_frequency: float,
        beamforming_sound_velocity: float,
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
            Volume sampling frequency in hertz.
        transmit_frequency : float
            Ultrasound transmit frequency in hertz.
        beamforming_sound_velocity : float
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
            * beamforming_sound_velocity
            / (4 * np.pi * transmit_frequency)
        )

    return process_iq_block_with_clutter_filter(
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
        transmit_frequency=transmit_frequency,
        spatial_kernel=spatial_kernel,
        lag=lag,
        absolute_velocity=absolute_velocity,
        beamforming_sound_velocity=beamforming_sound_velocity,
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
        Function to apply to each temporal window. It must accept a `(window_volumes,
        z, y, x)` array as first argument and return a `(output_volumes, ...)` array.
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
    total_volumes_in_windows = window_width + (n_windows - 1) * window_stride
    remaining_volumes = n_volumes - total_volumes_in_windows

    if remaining_volumes > 0:
        warnings.warn(
            f"{remaining_volumes} input volumes will be dropped because they do not fit "
            "into a complete window. Adjust `window_width` and `window_stride` to avoid "
            "this.",
            stacklevel=find_stack_level(),
        )
        iq = iq[:total_volumes_in_windows]

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
        `(z, y, x)` are spatial dimensions. The DataArray may carry a
        `compound_sampling_frequency` attribute as scanner provenance, but processing
        uses the `time` coordinate as the source of truth for temporal spacing.
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

    validate_iq(iq, require_attrs=False)

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

    fs = _get_filter_sampling_frequency(iq, filter_method=filter_method)

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

    output_times_values, doppler_window_durations = compute_processed_volume_timings(
        iq,
        clutter_window_width=clutter_window_width,
        clutter_window_stride=clutter_window_stride,
        inner_window_width=doppler_window_width,
        inner_window_stride=doppler_window_stride,
    )
    doppler_windows_per_clutter_window = (
        clutter_window_width - doppler_window_width
    ) // doppler_window_stride + 1
    doppler_window_duration, doppler_window_stride_duration = (
        _compute_inner_window_metadata(
            iq,
            output_timings=output_times_values,
            output_durations=doppler_window_durations,
            output_window_stride=doppler_window_stride,
            inner_windows_per_outer_window=doppler_windows_per_clutter_window,
            duration_description="Power Doppler integration duration",
            stride_description="Power Doppler integration stride",
        )
    )
    clutter_window_duration, clutter_window_stride_duration = (
        _compute_clutter_filter_window_metadata(
            iq,
            clutter_window_width=clutter_window_width,
            clutter_window_stride=clutter_window_stride,
        )
    )
    output_times = xr.DataArray(
        output_times_values,
        dims="time",
        attrs={
            **iq.coords["time"].attrs,
            # By default, we assume that the acquisition time reference is the start of
            # the first volume in the window.
            "volume_acquisition_reference": iq.coords["time"].attrs.get(
                "volume_acquisition_reference", "start"
            ),
            "volume_acquisition_duration": doppler_window_duration,
        },
    )

    output_attrs: dict[str, str | int | float] = {
        "units": "a.u.",
        "long_name": "Power Doppler intensity",
        "cmap": "gray",
        "clutter_filters": _format_clutter_filter_spec(
            filter_method, low_cutoff, high_cutoff
        ),
        "clutter_filter_window_duration": clutter_window_duration,
        "clutter_filter_window_stride": clutter_window_stride_duration,
        "power_doppler_integration_duration": doppler_window_duration,
        "power_doppler_integration_stride": doppler_window_stride_duration,
    }
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


def process_iq_to_bmode(
    iq: xr.DataArray,
    bmode_window_width: int | None = None,
    bmode_window_stride: int | None = None,
) -> xr.DataArray:
    """Process blocks of beamformed IQ into B-mode volumes using sliding windows.

    This function computes B-mode volumes from beamformed IQ data using a single sliding
    temporal window. Unlike power Doppler, no clutter filtering is applied; the mean
    magnitude (not squared magnitude) of the IQ data within each window is computed.

    Parameters
    ----------
    iq : xarray.DataArray
        Xarray DataArray containing complex beamformed IQ data with dimensions
        `(time, z, y, x)`, where `time` is the temporal dimension and
        `(z, y, x)` are spatial dimensions.
    bmode_window_width : int, optional
        Width of the sliding temporal window for B-mode integration, in volumes. If not
        provided, uses the chunk size of the IQ data along the temporal dimension.
    bmode_window_stride : int, optional
        Stride of the sliding temporal window, in volumes. If not provided, equals
        `bmode_window_width`.

    Returns
    -------
    (windows, z, y, x) xarray.DataArray
        B-mode volumes as an Xarray DataArray with updated time coordinates, where
        `windows` is the number of sliding windows.
    """
    import dask.array as da
    from dask.array import Array

    validate_iq(iq, require_attrs=False)

    dask_iq: Array = iq.data
    if not isinstance(dask_iq, Array):
        dask_iq = da.from_array(dask_iq)

    if bmode_window_width is None:
        bmode_window_width = cast(int, dask_iq.chunksize[0])
    if bmode_window_stride is None:
        bmode_window_stride = bmode_window_width

    result = process_iq_blocks(
        dask_iq,
        process_func=compute_bmode_volume,
        window_width=bmode_window_width,
        window_stride=bmode_window_stride,
    )

    output_times_values, bmode_window_durations = compute_processed_volume_timings(
        iq,
        clutter_window_width=bmode_window_width,
        clutter_window_stride=bmode_window_stride,
        inner_window_width=bmode_window_width,
        inner_window_stride=bmode_window_width,
    )
    bmode_window_duration, bmode_window_stride_duration = (
        _compute_inner_window_metadata(
            iq,
            output_timings=output_times_values,
            output_durations=bmode_window_durations,
            output_window_stride=bmode_window_stride,
            inner_windows_per_outer_window=1,
            duration_description="B-mode integration duration",
            stride_description="B-mode integration stride",
        )
    )
    output_times = xr.DataArray(
        output_times_values,
        dims="time",
        attrs={
            **iq.coords["time"].attrs,
            # By default, we assume that the acquisition time reference is the start of
            # the first volume in the window.
            "volume_acquisition_reference": iq.coords["time"].attrs.get(
                "volume_acquisition_reference", "start"
            ),
            "volume_acquisition_duration": bmode_window_duration,
        },
    )

    output_attrs: dict[str, str | int | float] = {
        "units": "a.u.",
        "long_name": "B-mode intensity",
        "cmap": "gray",
        "bmode_integration_duration": bmode_window_duration,
        "bmode_integration_stride": bmode_window_stride_duration,
    }

    return xr.DataArray(
        result,
        name="bmode",
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

        - `transmit_frequency`: Ultrasound transmit frequency in hertz.
        - `beamforming_sound_velocity`: Speed of sound assumed during beamforming in
          meters per second.
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

    validate_iq(iq, require_attrs=True)

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

    fs = _get_filter_sampling_frequency(iq, filter_method=filter_method)

    # Validation ensures these attributes exist, so we can safely access them here.
    transmit_frequency = iq.attrs["transmit_frequency"]
    beamforming_sound_velocity = iq.attrs["beamforming_sound_velocity"]

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
        transmit_frequency=transmit_frequency,
        beamforming_sound_velocity=beamforming_sound_velocity,
        estimation_method=estimation_method,
    )

    output_times_values, velocity_window_durations = compute_processed_volume_timings(
        iq,
        clutter_window_width=clutter_window_width,
        clutter_window_stride=clutter_window_stride,
        inner_window_width=velocity_window_width,
        inner_window_stride=velocity_window_stride,
    )
    velocity_windows_per_clutter_window = (
        clutter_window_width - velocity_window_width
    ) // velocity_window_stride + 1
    velocity_window_duration, velocity_window_stride_duration = (
        _compute_inner_window_metadata(
            iq,
            output_timings=output_times_values,
            output_durations=velocity_window_durations,
            output_window_stride=velocity_window_stride,
            inner_windows_per_outer_window=velocity_windows_per_clutter_window,
            duration_description="Axial velocity integration duration",
            stride_description="Axial velocity integration stride",
        )
    )
    clutter_window_duration, clutter_window_stride_duration = (
        _compute_clutter_filter_window_metadata(
            iq,
            clutter_window_width=clutter_window_width,
            clutter_window_stride=clutter_window_stride,
        )
    )
    output_times = xr.DataArray(
        output_times_values,
        dims="time",
        attrs={
            **iq.coords["time"].attrs,
            # By default, we assume that the acquisition time reference is the start of
            # the first volume in the window.
            "volume_acquisition_reference": iq.coords["time"].attrs.get(
                "volume_acquisition_reference", "start"
            ),
            "volume_acquisition_duration": velocity_window_duration,
        },
    )

    velocity_cmap = "viridis" if absolute_velocity else "coolwarm"
    output_attrs = {
        "units": "m/s",
        "long_name": "Axial velocity",
        "cmap": velocity_cmap,
        "clutter_filters": _format_clutter_filter_spec(
            filter_method, low_cutoff, high_cutoff
        ),
        "clutter_filter_window_duration": clutter_window_duration,
        "clutter_filter_window_stride": clutter_window_stride_duration,
        "axial_velocity_lag": lag,
        "axial_velocity_absolute": absolute_velocity,
        "axial_velocity_spatial_kernel": spatial_kernel,
        "transmit_frequency": transmit_frequency,
        "beamforming_sound_velocity": beamforming_sound_velocity,
        "axial_velocity_estimation_method": estimation_method,
        "axial_velocity_integration_duration": velocity_window_duration,
        "axial_velocity_integration_stride": velocity_window_stride_duration,
    }
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
